#!/usr/bin/env python3
"""
snapstats.py — SnapRAID content file scrub/sync age statistics

Usage:  python3 snapstats.py [/path/to/snapraid.content]
"""

import sys
import os
import argparse
import shutil
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
import json
import pickle
import time

# ── Terminal color support ────────────────────────────────────────────────────

def _supports_color():
    import os
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

USE_COLOR = _supports_color()

class C:
    """ANSI color codes — reassigned at runtime if --no-color is passed."""
    RESET    = ""; DIM    = ""
    TODAY    = ""; WEEK   = ""; TWO_WK = ""
    OLD      = ""; NONE   = ""; EMPTY  = ""
    HEADER   = ""; LABEL  = ""; NUM    = ""
    EMPTY_CHR = "·"

def init_colors(enabled):
    C.RESET    = "\033[0m"      if enabled else ""
    C.DIM      = "\033[2m"      if enabled else ""
    C.TODAY    = "\033[92m"     if enabled else ""
    C.WEEK     = "\033[32m"     if enabled else ""
    C.TWO_WK   = "\033[33m"     if enabled else ""
    C.OLD      = "\033[31m"     if enabled else ""
    C.NONE     = "\033[90m"     if enabled else ""
    C.EMPTY    = "\033[38;5;236m" if enabled else ""
    C.HEADER   = "\033[1;36m"   if enabled else ""
    C.LABEL    = "\033[37m"     if enabled else ""
    C.NUM      = "\033[1;37m"   if enabled else ""
    C.EMPTY_CHR = "▒"          if enabled else "·"

init_colors(USE_COLOR)

def age_color(label):
    """Pick a bar color based on the age label string."""
    if label == "today":        return C.TODAY
    if label == "not scrubbed": return C.NONE
    # Nd  or  N+
    try:
        days = int(label.rstrip("d+"))
        if label.endswith("+"): return C.OLD
        if days <= 7:   return C.TODAY
        if days <= 14:  return C.WEEK
        if days <= 30:  return C.TWO_WK
        return C.OLD
    except (ValueError, IndexError):
        return C.RESET

SNAPRAID_CONF = "/etc/snapraid.conf"

WIDTH_MAX     = 80   # cap output at this width
WIDTH_BAR_MIN = 50   # below this: text-only mode (bar needs at least 5 chars)
WIDTH_TEXT_MIN= 43   # below this: too narrow, error out
BAR_FIXED     = 45   # fixed chars around the bar in a full line

def get_content_path(conf_path=SNAPRAID_CONF):
    """Return the first accessible 'content' path from snapraid.conf."""
    if not os.path.exists(conf_path):
        print(f"ERROR: {conf_path} not found", file=sys.stderr)
        sys.exit(1)
    with open(conf_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("content "):
                path = line.split(None, 1)[1].strip()
                if os.path.exists(path):
                    return path
    print(f"ERROR: No accessible content file found in {conf_path}", file=sys.stderr)
    sys.exit(1)


# ── Cache ─────────────────────────────────────────────────────────────────────

CACHE_PATH = "/var/cache/snapstats/snapstats.cache"

def _content_sig(path):
    """Fingerprint: mtime + size — cheap, no hashing of the 2.7GB file."""
    st = os.stat(path)
    return (path, st.st_mtime, st.st_size)

CACHE_TTL = 2 * 3600  # seconds before cache is considered stale

def load_cache(content_path, force_use=False):
    try:
        cache_age = time.time() - os.stat(CACHE_PATH).st_mtime
        if not force_use and cache_age > CACHE_TTL:
            return None
        with open(CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        if cached.get("sig") == _content_sig(content_path):
            return cached["data"]
    except Exception:
        pass
    return None

def save_cache(content_path, data):
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({"sig": _content_sig(content_path), "data": data}, f)
    except Exception as e:
        print(f"  WARNING: Could not write cache: {e}", file=sys.stderr)


# ── Low-level binary readers ──────────────────────────────────────────────────

def read_varint32(f):
    """
    SnapRAID variable-length integer (32-bit).
    Bytes with high bit CLEAR are continuation bytes (7 bits each).
    Byte with high bit SET is the terminal byte (7 bits).
    """
    v = s = 0
    while True:
        b = f.read(1)
        if not b:
            raise EOFError("Unexpected end of file reading varint32")
        b = b[0]
        if (b & 0x80) == 0:        # continuation
            v |= b << s
            s += 7
        else:                       # terminal
            v |= (b & 0x7f) << s
            return v

def read_varint64(f):
    """Same encoding as varint32 but for 64-bit values."""
    v = s = 0
    while True:
        b = f.read(1)
        if not b:
            raise EOFError("Unexpected end of file reading varint64")
        b = b[0]
        if (b & 0x80) == 0:
            v |= b << s
            s += 7
        else:
            v |= (b & 0x7f) << s
            return v

def read_bstring(f):
    """Length-prefixed string: varint32 length, then raw UTF-8 bytes."""
    length = read_varint32(f)
    return f.read(length).decode("utf-8", errors="replace")

# ── Parser ────────────────────────────────────────────────────────────────────

def parse_content(path, verbose=False):
    """
    Parse a SnapRAID content file.

    Returns:
        files       — list of dicts with keys: path, disk, size, runs
                      runs = list of (block_pos, count, state_char)
        info_ranges — sorted list of (start_pos, end_pos, unix_timestamp, justsynced, bad)
        block_size  — bytes per block
        block_max   — total number of block slots
    """
    size_bytes = os.path.getsize(path)
    if verbose: print(f"Parsing {path} ({size_bytes / 1024**3:.2f} GB)...", flush=True)

    files       = []
    info_ranges = []  # will be sorted by start_pos
    disk_mapping = [] # index → disk name
    block_size  = 256 * 1024   # default 256 KB, overridden by 'z' record
    hash_size   = 16           # default 16 bytes, overridden by 'y' record
    block_max   = 0

    with open(path, "rb") as f:

        # ── 12-byte header ────────────────────────────────────────────────────
        header = f.read(12)
        if len(header) < 12 or header[:7] != b"SNAPCNT":
            raise ValueError("Not a SnapRAID content file (bad magic)")
        if verbose: print(f"  Format  : {header[:8].decode()}")

        # ── Record loop ───────────────────────────────────────────────────────
        file_count = 0

        while True:
            cmd_byte = f.read(1)
            if not cmd_byte:
                break
            cmd = chr(cmd_byte[0])

            # ── Block size ────────────────────────────────────────────────────
            if cmd == "z":
                block_size = read_varint32(f)
                if verbose: print(f"  Block   : {block_size // 1024} KB")

            # ── Hash size ─────────────────────────────────────────────────────
            elif cmd == "y":
                hash_size = read_varint32(f)
                if verbose: print(f"  Hashsize: {hash_size}")

            # ── Block max ─────────────────────────────────────────────────────
            elif cmd == "x":
                block_max = read_varint32(f)
                if verbose: print(f"  Blockmax: {block_max:,}")

            # ── Checksum config ───────────────────────────────────────────────
            elif cmd in ("c", "C"):
                f.read(1)          # hash type sub-byte
                f.read(hash_size)  # hash seed

            # ── Disk mapping ──────────────────────────────────────────────────
            elif cmd in ("m", "M"):
                name = read_bstring(f)
                read_varint32(f)   # v_pos (base block for this disk)
                if cmd == "M":
                    read_varint32(f)   # total_blocks
                    read_varint32(f)   # free_blocks
                read_bstring(f)    # uuid
                disk_mapping.append(name)
                if verbose: print(f"  Disk [{len(disk_mapping)-1}] : {name}")

            # ── Parity info (just skip) ───────────────────────────────────────
            elif cmd == "P":
                read_varint32(f)   # level
                read_varint32(f)   # total_blocks
                read_varint32(f)   # free_blocks
                read_bstring(f)    # uuid

            elif cmd == "Q":
                read_varint32(f)   # level
                read_varint32(f)   # total_blocks
                read_varint32(f)   # free_blocks
                read_bstring(f)    # uuid
                n_files = read_varint32(f)
                for _ in range(n_files):
                    read_bstring(f)    # parity file path
                    read_varint64(f)   # size
                    read_varint32(f)   # free_blocks

            # ── File record ───────────────────────────────────────────────────
            elif cmd == "f":
                mapping    = read_varint32(f)
                size       = read_varint64(f)
                _mtime_sec = read_varint64(f)
                _mtime_ns  = read_varint32(f)
                _inode     = read_varint64(f)
                fpath      = read_bstring(f)

                disk = (disk_mapping[mapping]
                        if mapping < len(disk_mapping)
                        else f"?disk{mapping}")

                nblocks = (size + block_size - 1) // block_size if size > 0 else 0

                runs = []   # (block_pos, count, state)
                idx  = 0

                while idx < nblocks:
                    state_byte = f.read(1)
                    if not state_byte:
                        raise EOFError("EOF inside file block list")
                    state  = chr(state_byte[0])
                    v_pos  = read_varint32(f)
                    v_count = read_varint32(f)

                    runs.append((v_pos, v_count, state))

                    # Skip block hashes (not needed for stats)
                    if state != "n":
                        f.read(hash_size * v_count)

                    idx += v_count

                files.append({
                    "path" : fpath,
                    "disk" : disk,
                    "size" : size,
                    "runs" : runs,
                })

                file_count += 1
                if file_count % 50_000 == 0:
                    if verbose: print(f"  {file_count:,} files parsed...", flush=True)

            # ── Info/scrub record ─────────────────────────────────────────────
            elif cmd == "i":
                v_oldest = read_varint64(f)
                v_pos    = 0

                while v_pos < block_max:
                    v_count = read_varint32(f)
                    flag    = read_varint32(f)

                    has_info  = bool(flag & 1)
                    bad       = bool(flag & 2)
                    justsynced = bool(flag & 8)

                    if has_info:
                        t64 = read_varint64(f)
                        timestamp = v_oldest + t64
                        info_ranges.append(
                            (v_pos, v_pos + v_count, timestamp, justsynced, bad)
                        )

                    v_pos += v_count

            # ── Hole record ───────────────────────────────────────────────────
            elif cmd == "h":
                read_varint32(f)   # mapping (we don't use it)
                v_pos = 0
                while v_pos < block_max:
                    v_count = read_varint32(f)
                    sub     = f.read(1)[0]
                    if chr(sub) == "o":
                        f.read(hash_size * v_count)   # deleted block hashes
                    # 'O' = empty run, no data
                    v_pos += v_count

            # ── Symlink ───────────────────────────────────────────────────────
            elif cmd == "s":
                read_varint32(f)   # mapping
                read_bstring(f)    # sub
                read_bstring(f)    # linkto

            # ── Hardlink ──────────────────────────────────────────────────────
            elif cmd == "a":
                read_varint32(f)   # mapping
                read_bstring(f)    # sub
                read_bstring(f)    # linkto

            # ── Directory ─────────────────────────────────────────────────────
            elif cmd == "r":
                read_varint32(f)   # mapping
                read_bstring(f)    # sub

            # ── CRC end marker ────────────────────────────────────────────────
            elif cmd == "N":
                f.read(4)   # 4-byte little-endian CRC32, we don't verify it
                break       # this is always the last record

            else:
                print(f"  WARNING: Unknown record '{cmd}' "
                      f"(0x{cmd_byte[0]:02x}) — stopping parse")
                break

    if verbose: print(f"  Total files : {len(files):,}")
    if verbose: print(f"  Info ranges : {len(info_ranges):,}")
    return files, info_ranges, block_size, block_max, disk_mapping


# ── Block info lookup ─────────────────────────────────────────────────────────

# ── Per-file statistics ───────────────────────────────────────────────────────

SENTINEL_JUSTSYNCED = 1  # placeholder — real timestamps are >> 1e9

def build_info_array(info_ranges, block_max, verbose=False):
    """
    Build a flat numpy uint32 array of size block_max.
      0            = no info (never scrubbed)
      1            = justsynced (synced but not independently scrubbed)
      >1           = Unix timestamp of last scrub
    ~257 MB for block_max=67M
    """
    if verbose: print(f"  Allocating info array ({block_max * 4 / 1024**2:.0f} MB)...", flush=True)
    arr = np.zeros(block_max, dtype=np.uint32)
    for (start, end, timestamp, justsynced, _bad) in info_ranges:
        if justsynced:
            # Only mark as justsynced where not already scrubbed
            mask = arr[start:end] == 0
            arr[start:end][mask] = SENTINEL_JUSTSYNCED
        else:
            arr[start:end] = np.uint32(timestamp)
    if verbose: print(f"  Info array ready.", flush=True)
    return arr


def compute_file_stats(files, info_ranges, block_max, verbose=False):
    """
    For each file, determine:
      - fully_synced: True if all blocks are in state 'b' (BLK)
      - oldest_scrub : Unix timestamp of the oldest scrubbed block,
                       or None if any block is unscrubbed / justsynced
      - latest_scrub : Unix timestamp of the most recently scrubbed block

    Returns list of (file_dict, fully_synced, oldest_scrub_ts, latest_scrub_ts)
    """
    info_arr = build_info_array(info_ranges, block_max, verbose=verbose)

    results = []
    total = len(files)

    for i, file in enumerate(files):
        if i % 50_000 == 0 and i > 0 and verbose:
            print(f"  Computing stats: {i:,}/{total:,}...", flush=True)

        runs = file["runs"]
        all_synced = True
        oldest_scrub = None
        latest_scrub = None
        fully_scrubbed = True

        for (v_pos, v_count, state) in runs:
            if state != "b":
                all_synced = False

            # Slice the info array for this run
            chunk = info_arr[v_pos:v_pos + v_count]

            # Any blocks with no info or justsynced = not fully scrubbed
            if np.any(chunk <= SENTINEL_JUSTSYNCED):
                fully_scrubbed = False

            # Scrubbed blocks have value > 1
            scrubbed = chunk[chunk > SENTINEL_JUSTSYNCED]
            if scrubbed.size > 0:
                t_min = int(scrubbed.min())
                t_max = int(scrubbed.max())
                if oldest_scrub is None or t_min < oldest_scrub:
                    oldest_scrub = t_min
                if latest_scrub is None or t_max > latest_scrub:
                    latest_scrub = t_max

        if not fully_scrubbed:
            oldest_scrub = None

        results.append((file, all_synced, oldest_scrub, latest_scrub))

    return results


# ── Output ────────────────────────────────────────────────────────────────────

def format_size(n_bytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def print_bar(label, count, total_files, size_bytes, max_count, bar_width=0):
    pct = count / total_files * 100 if total_files else 0
    col = age_color(label)
    if bar_width > 0:
        filled = int(bar_width * count / max_count) if max_count else 0
        bar    = (col + "█" * filled + C.RESET +
                  C.EMPTY + C.EMPTY_CHR * (bar_width - filled) + C.RESET)
        print(f"  {C.LABEL}{label:>13s}{C.RESET} "
              f"│{bar}│ "
              f"{col}{pct:5.1f}%{C.RESET}  "
              f"{C.NUM}{count:>6,}{C.RESET}f  "
              f"{C.DIM}{format_size(size_bytes):>9s}{C.RESET}")
    else:
        print(f"  {C.LABEL}{label:>13s}{C.RESET}  "
              f"{col}{pct:5.1f}%{C.RESET}  "
              f"{C.NUM}{count:>6,}{C.RESET}f  "
              f"{C.DIM}{format_size(size_bytes):>9s}{C.RESET}")


def build_summary(file_stats, all_disks=None, max_age=None):
    """Return a dict of aggregated stats for exit-code checks and JSON output."""
    now_ts  = datetime.now(timezone.utc).timestamp()
    total   = len(file_stats)
    oldest_scrub_ts = None
    oldest_scrub_file = None
    never_real = 0
    never_size = 0

    per_disk = {}
    for (file, fully_synced, oldest_scrub, _) in file_stats:
        disk = file["disk"]
        if disk not in per_disk:
            per_disk[disk] = {"files": 0, "size": 0, "oldest_scrub_days": None}
        per_disk[disk]["files"] += 1
        per_disk[disk]["size"]  += file["size"]

        if file["size"] > 0 and oldest_scrub is None:
            never_real += 1
            never_size += file["size"]

        if oldest_scrub is not None:
            age = int((now_ts - oldest_scrub) / 86400)
            if per_disk[disk]["oldest_scrub_days"] is None or age > per_disk[disk]["oldest_scrub_days"]:
                per_disk[disk]["oldest_scrub_days"] = age
            if oldest_scrub_ts is None or oldest_scrub < oldest_scrub_ts:
                oldest_scrub_ts  = oldest_scrub
                oldest_scrub_file = file["path"]

    oldest_days = int((now_ts - oldest_scrub_ts) / 86400) if oldest_scrub_ts else None
    max_age_exceeded = (max_age is not None and oldest_days is not None and oldest_days > max_age)

    result = {
        "generated":          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_files":        total,
        "total_size":         sum(f["size"] for f, *_ in file_stats),
        "unscrubbed_files":   never_real,
        "unscrubbed_size":    never_size,
        "oldest_scrub_days":  oldest_days,
        "oldest_scrub_file":  oldest_scrub_file,
        "per_disk":           per_disk,
        "empty_disks":        [d for d in (all_disks or []) if d not in per_disk],
    }
    if max_age is not None:
        result["max_age_threshold"] = max_age
        result["max_age_exceeded"]  = max_age_exceeded
    return result


def print_report(file_stats, all_disks=None, show_disk=False, top_n=None, max_age=None):
    now       = datetime.now(timezone.utc)
    today_ts  = now.timestamp()

    # Bucket files by scrub age in days
    buckets_count = defaultdict(int)
    buckets_size  = defaultdict(int)

    # Per-disk buckets
    disk_buckets_count = defaultdict(lambda: defaultdict(int))
    disk_buckets_size  = defaultdict(lambda: defaultdict(int))
    disk_files_total   = defaultdict(int)
    disk_size_total    = defaultdict(int)

    never_count = never_size = 0
    empty_count = 0          # zero-byte files (no blocks to scrub, excluded from chart)

    for (file, fully_synced, oldest_scrub, _latest) in file_stats:
        sz   = file["size"]
        disk = file["disk"]
        disk_files_total[disk] += 1
        disk_size_total[disk]  += sz

        if oldest_scrub is None:
            if sz == 0:
                empty_count += 1   # zero-byte: no blocks, skip from chart
            else:
                never_count += 1
                never_size  += sz
                disk_buckets_count[disk]["never"] += 1
                disk_buckets_size[disk]["never"]  += sz
        else:
            age_days = int((today_ts - oldest_scrub) / 86400)
            cutoff   = max_age if max_age is not None else 61
            bucket   = min(age_days, cutoff)
            buckets_count[bucket] += 1
            buckets_size[bucket]  += sz
            disk_buckets_count[disk][bucket] += 1
            disk_buckets_size[disk][bucket]  += sz


    total_files = len(file_stats)
    total_size  = sum(f["size"] for f, *_ in file_stats)

    raw_w    = shutil.get_terminal_size((80, 24)).columns
    term_w   = min(raw_w, WIDTH_MAX)
    if term_w < WIDTH_TEXT_MIN:
        print(f"ERROR: Terminal too narrow ({raw_w} cols, need at least {WIDTH_TEXT_MIN})", file=sys.stderr)
        sys.exit(1)
    bar_w    = max(0, min(term_w - BAR_FIXED, WIDTH_MAX - BAR_FIXED)) if term_w >= WIDTH_BAR_MIN else 0

    # ── Overall distribution ──────────────────────────────────────────────────
    all_counts = list(buckets_count.values()) + [never_count, 0]
    max_count  = max(all_counts) if all_counts else 1

    ts_str   = now.strftime("%Y-%m-%d %H:%M UTC")
    hdr_info = f"  {total_files:,} files / {format_size(total_size)}  {ts_str}"
    hdr_title= "SNAPRAID SCRUB AGE"
    pad      = max(0, term_w - len(hdr_title) - len(hdr_info) - 4)
    print()
    print(C.HEADER + f"  {hdr_title}" + " " * pad + hdr_info + C.RESET)
    print(C.HEADER + "─" * term_w + C.RESET)

    cutoff = max_age if max_age is not None else 61
    cutoff_label = f"{cutoff}+"
    print_bar("today", buckets_count.get(0, 0), total_files, buckets_size.get(0, 0), max_count, bar_w)
    for d in range(1, cutoff):
        c = buckets_count.get(d, 0)
        s = buckets_size.get(d, 0)
        if c > 0:
            print_bar(f"{d}d", c, total_files, s, max_count, bar_w)
    print_bar(cutoff_label, buckets_count.get(cutoff, 0), total_files, buckets_size.get(cutoff, 0), max_count, bar_w)
    print_bar("not scrubbed", never_count, total_files, never_size, max_count, bar_w)

    # Sync status — single line footer
    synced_count = total_files
    synced_size  = total_size
    sync_ok  = f"last sync: {synced_count:,} files / {format_size(synced_size)}"
    empty_disks = [d for d in (all_disks or []) if d not in disk_files_total]
    empty_disk_note = f"  {C.DIM}[{",".join(empty_disks)} empty]{C.RESET}" if (empty_disks and show_disk) else ""
    empty_note = f"  {C.DIM}(+{empty_count} empty){C.RESET}" if empty_count else ""
    print(C.HEADER + "─" * term_w + C.RESET)
    print(f"  {C.DIM}{sync_ok}{C.RESET}{empty_note}{empty_disk_note}")

    # ── Per-disk breakdown ────────────────────────────────────────────────────
    raw_w  = shutil.get_terminal_size((80, 24)).columns
    term_w = min(raw_w, WIDTH_MAX)
    bar_w  = max(0, min(term_w - BAR_FIXED, WIDTH_MAX - BAR_FIXED)) if term_w >= WIDTH_BAR_MIN else 0
    if show_disk:
        print()
        print(C.HEADER + "─" * term_w + C.RESET)
        print(C.HEADER + "  PER-DISK BREAKDOWN" + C.RESET)
        print(C.HEADER + "─" * term_w + C.RESET)

    if show_disk:
        for disk in sorted(disk_files_total.keys()):
            d_total  = disk_files_total[disk]
            d_size   = disk_size_total[disk]
            d_counts = disk_buckets_count[disk]
            d_sizes  = disk_buckets_size[disk]
            d_never  = d_counts.get("never", 0)

            # Find oldest scrubbed bucket for this disk
            scrubbed_buckets = {k: v for k, v in d_counts.items() if k != "never"}
            if scrubbed_buckets:
                oldest_bucket = max(scrubbed_buckets.keys())
            else:
                oldest_bucket = None

            oldest_str = (f"{oldest_bucket}+ days" if oldest_bucket == 61
                          else f"{oldest_bucket} days" if oldest_bucket is not None
                          else "n/a")

            print()
            print(f"  Disk {disk}  —  {d_total:,} files  {format_size(d_size)}  "
                  f"(oldest scrub: {oldest_str})")


            d_max = max(d_counts.values()) if d_counts else 1
            print_bar("today", d_counts.get(0, 0), d_total, d_sizes.get(0, 0), d_max, bar_w)
            for day in range(1, cutoff):
                c = d_counts.get(day, 0)
                s = d_sizes.get(day, 0)
                if c > 0:
                    print_bar(f"{day}d", c, d_total, s, d_max, bar_w)
            print_bar(cutoff_label, d_counts.get(cutoff, 0), d_total, d_sizes.get(cutoff, 0), d_max, bar_w)
            print_bar("not scrubbed", d_never, d_total, d_sizes.get("never", 0), d_max, bar_w)

        for disk in sorted(d for d in (all_disks or []) if d not in disk_files_total):
            print()
            print(f"  {C.HEADER}{disk}{C.RESET}  {C.DIM}(empty — no files tracked){C.RESET}")

    # ── Oldest files ──────────────────────────────────────────────────────────
    if top_n is None:
        return
    term_w = min(shutil.get_terminal_size((80, 24)).columns, WIDTH_MAX)
    print()
    print(C.HEADER + "─" * term_w + C.RESET)
    print(f"{C.HEADER}  TOP {top_n} OLDEST FILES (by oldest un-scrubbed block){C.RESET}")
    print(C.HEADER + "─" * term_w + C.RESET)
    print()

    # Filter zero-byte files, sort by oldest_scrub ascending
    def sort_key(entry):
        _, _, oldest_scrub, _ = entry
        return oldest_scrub if oldest_scrub is not None else 0

    sorted_stats = sorted(
        (e for e in file_stats if e[0]["size"] > 0),
        key=sort_key
    )

    print(f"  {'Age':>12s}  {'Disk':>4s}  {'Size':>10s}  Path")
    print(f"  {'─'*12}  {'─'*4}  {'─'*10}  {'─'*40}")

    for (file, _synced, oldest_scrub, _latest) in sorted_stats[:top_n]:
        if oldest_scrub is None:
            age_str = "not scrubbed"
        else:
            age_days = int((today_ts - oldest_scrub) / 86400)
            age_str  = f"{age_days}d"
        print(f"  {age_str:>12s}  {file['disk']:>4s}  "
              f"{format_size(file['size']):>10s}  {file['path']}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SnapRAID content file statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  snapstats.py                        # scrub age report only
  snapstats.py --disk                 # add per-disk breakdown
  snapstats.py --top 30               # add 30 oldest files
  snapstats.py --disk --top 50        # all sections
  snapstats.py --all                  # all sections (default top 20)
  snapstats.py --content /path/to/snapraid.content
  snapstats.py --max-age 30           # exit code 2 if oldest > 30 days
  snapstats.py --no-cache             # force re-parse
  snapstats.py --json                 # machine-readable JSON summary

Cache:
  {CACHE_PATH}  (TTL: {CACHE_TTL // 3600}h, keyed on content file mtime+size)

Cache file: /var/cache/snapstats/snapstats.cache  (TTL: 2h, keyed on content file mtime+size)


  {C.TODAY}██{C.RESET} today / ≤7 days
  {C.WEEK}██{C.RESET} 8–14 days
  {C.TWO_WK}██{C.RESET} 15–30 days
  {C.OLD}██{C.RESET} 31+ days
  {C.NONE}██{C.RESET} never scrubbed

Exit codes:
  0  all good
  2  oldest scrub exceeds --max-age threshold
  3  configuration / runtime error
        """
    )
    parser.add_argument(
        "--content", metavar="PATH",
        help="Path to snapraid.content file (default: read from /etc/snapraid.conf)"
    )
    parser.add_argument(
        "--conf", metavar="PATH", default=SNAPRAID_CONF,
        help=f"Path to snapraid.conf (default: {SNAPRAID_CONF})"
    )
    parser.add_argument(
        "--disk", action="store_true",
        help="Show per-disk scrub age breakdown"
    )
    parser.add_argument(
        "--top", metavar="N", type=int, nargs="?", const=20,
        help="Show N oldest files (default 20 if flag given without a number)"
    )
    parser.add_argument(
        "--all", dest="show_all", action="store_true",
        help="Show all sections (equivalent to --disk --top 20)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show parse progress output"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress all output; use exit code only"
    )
    parser.add_argument(
        "--no-color", dest="no_color", action="store_true",
        help="Disable color output"
    )
    parser.add_argument(
        "--max-age", metavar="DAYS", type=int, default=None,
        help="Exit with code 2 if any file has scrub age older than DAYS"
    )
    parser.add_argument(
        "--no-cache", dest="no_cache", action="store_true",
        help="Ignore and overwrite existing cache"
    )
    parser.add_argument(
        "--use-cache", dest="use_cache", action="store_true",
        help="Use cached stats even if older than 2 hours"
    )
    parser.add_argument(
        "--json", dest="json_out", action="store_true",
        help="Output summary as JSON (implies --no-color, suppresses report)"
    )

    args = parser.parse_args()

    if args.max_age is not None and args.max_age < 1:
        parser.error("--max-age must be at least 1")
    if args.quiet and args.verbose:
        parser.error("--quiet and --verbose are mutually exclusive")
    if args.no_cache and args.use_cache:
        parser.error("--no-cache and --use-cache are mutually exclusive")
    if args.no_color:
        init_colors(False)

    if args.show_all:
        args.disk = True
        if args.top is None:
            args.top = 20

    if args.content:
        path = args.content
        if not os.path.exists(path):
            print(f"ERROR: Content file not found: {path}", file=sys.stderr)
            sys.exit(1)
    else:
        path = get_content_path(args.conf)
        if args.verbose:
            print(f"Using content file: {path}")

    if args.json_out:
        init_colors(False)

    # ── Run conditions ───────────────────────────────────────────────────────
    if not args.quiet and not args.json_out:
        flags = []
        if args.disk:       flags.append("--disk")
        if args.top:        flags.append(f"--top {args.top}")
        if args.max_age:    flags.append(f"--max-age {args.max_age}")
        if args.no_cache:   flags.append("--no-cache")
        if args.use_cache:  flags.append("--use-cache")
        if args.no_color:   flags.append("--no-color")
        if args.content:    flags.append(f"--content {args.content}")
        if args.conf != SNAPRAID_CONF: flags.append(f"--conf {args.conf}")
        if flags:
            print(f"{C.DIM}  options: {" ".join(flags)}{C.RESET}")

    # ── Cache ────────────────────────────────────────────────────────────────
    file_stats = None
    all_disks = None
    if not args.no_cache:
        cached = load_cache(path, force_use=args.use_cache)
        if cached is not None:
            file_stats, all_disks = cached
            if args.verbose:
                print("Using cached stats.")

    if file_stats is None:
        if args.verbose:
            print()
        files, info_ranges, block_size, block_max, all_disks = parse_content(path, verbose=args.verbose)
        if args.verbose:
            print()
            print("Computing per-file statistics...", flush=True)
        file_stats = compute_file_stats(files, info_ranges, block_max, verbose=args.verbose)
        save_cache(path, (file_stats, all_disks))  # stored as tuple

    # ── JSON output ───────────────────────────────────────────────────────────
    if args.json_out:
        print(json.dumps(build_summary(file_stats, all_disks, max_age=args.max_age), indent=2))
        sys.exit(0)

    # ── Report ────────────────────────────────────────────────────────────────
    if not args.quiet:
        print_report(file_stats, all_disks=all_disks, show_disk=args.disk, top_n=args.top, max_age=args.max_age)

    # ── Exit code ─────────────────────────────────────────────────────────────
    summary = build_summary(file_stats, all_disks, max_age=args.max_age)
    exit_code = 0
    if args.max_age is not None and summary["oldest_scrub_days"] is not None:
        if summary["oldest_scrub_days"] > args.max_age:
            if args.verbose:
                print(f"  WARNING: oldest scrub is {summary['oldest_scrub_days']}d "
                      f"(threshold: {args.max_age}d)")
            exit_code = max(exit_code, 2)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

