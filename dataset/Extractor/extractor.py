#!/usr/bin/env python3
# /// script
# dependencies = [
#   "pandas",
# ]
# ///
"""
================================================================================
Linux Kernel Git Time Series Extractor & Resampler (Parallel Multi-Core Edition)
================================================================================

HOW TO CLONE THE REPOSITORY:
----------------------------
The working tree of the Linux kernel is over 50 GB. To analyze history and churn,
you do NOT need the unpacked files on disk; you only need the Git object database.

Run this command in your terminal (~4.5 GB download, saves ~45 GB of disk):

    git clone --bare https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git /path/to/linux-bare.git

This creates a bare repository containing all 1.3M+ commits, tree objects,
and diff histories required for code churn calculation.
================================================================================
"""

import argparse
import csv
from datetime import datetime
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import pandas as pd

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# 1. Default path to your cloned Linux repository (bare clone or regular clone).
DEFAULT_REPO_PATH = Path("/home/cybervitor/.local/gits/linux")

# 2. Default number of parallel workers (defaults to CPU count minus 1)
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# 3. Work-chunk size (commits per batch for dynamic load balancing)
CHUNK_SIZE = 10000

# 4. Output directory configuration:
# - Raw commit event log is saved in this script's directory (Dataset/Extractor/)
EXTRACTOR_DIR = Path(__file__).resolve().parent
# - Resampled time series CSV files are saved in the parent directory (Dataset/)
DATASET_DIR = EXTRACTOR_DIR.parent

# Output filenames
RAW_CSV_PATH = EXTRACTOR_DIR / "kernel_raw_commits.csv"
DAILY_TS_PATH = DATASET_DIR / "kernel_daily_ts.csv"
WEEKLY_TS_PATH = DATASET_DIR / "kernel_weekly_ts.csv"
TEMP_CHUNKS_DIR = EXTRACTOR_DIR / "_kernel_ts_chunks"

# Regex to parse git log's --shortstat line:
# Example: " 3 files changed, 24 insertions(+), 5 deletions(-)"
SHORTSTAT_RE = re.compile(
    r"(\d+)\s+file[s]?\s+changed"
    r"(?:,\s+(\d+)\s+insertion[s]?\(\+\))?"
    r"(?:,\s+(\d+)\s+deletion[s]?\(-\))?"
)


def verify_repo(path: Path) -> None:
    """Verifies that the target path exists and is a valid Git repository."""
    if not path.exists():
        print(f"[ERROR] Path does not exist: {path}", file=sys.stderr, flush=True)
        print("Please configure DEFAULT_REPO_PATH at the top of this script or pass the path as an argument:", file=sys.stderr, flush=True)
        print(f"  python {Path(__file__).name} /path/to/linux-bare.git", file=sys.stderr, flush=True)
        sys.exit(1)

    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--is-bare-repository"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            f"[ERROR] '{path}' is not a valid Git repository. Run the clone command first.",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)


def get_all_commit_hashes(repo_path: Path) -> list[str]:
    """
    Rapidly retrieves all reachable commit hashes using git rev-list.
    Runs in ~1-2 seconds for 1.3M commits because it only walks commit objects.
    """
    print("[*] Enumerating commit graph via `git rev-list --all`...", flush=True)
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-list", "--all"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    if result.returncode != 0:
        print(f"[ERROR] Failed to list commits: {result.stderr}", file=sys.stderr, flush=True)
        sys.exit(result.returncode)

    hashes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return hashes


def _process_chunk_worker(args: tuple[int, list[str], Path, Path]) -> tuple[int, int]:
    """
    Worker function executed in parallel across CPU cores.
    Feeds a batch of commit hashes to `git log --no-walk=unsorted --stdin --shortstat`.
    Parses metadata and churn stats, writing the result directly to a chunk CSV.
    """
    chunk_idx, commit_hashes, repo_path, chunk_file = args

    git_cmd = [
        "git",
        "-C",
        str(repo_path),
        "log",
        "--no-walk=unsorted",
        "--stdin",
        "--date=iso-strict",
        "--pretty=format:COMMIT:%H|%aI|%cI|%ae|%P",
        "--shortstat",
    ]

    process = subprocess.Popen(
        git_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        errors="replace",
    )

    stdin_payload = "\n".join(commit_hashes) + "\n"

    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(stdin_payload)
    process.stdin.close()

    current_record = None
    processed_count = 0

    with open(chunk_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "commit_hash",
            "author_date",
            "committer_date",
            "author_email",
            "is_merge",
            "files_changed",
            "insertions",
            "deletions",
        ])

        for line in process.stdout:
            line_str = line.strip()
            if not line_str:
                continue

            if line_str.startswith("COMMIT:"):
                if current_record:
                    writer.writerow(current_record)
                    processed_count += 1

                payload = line_str[len("COMMIT:") :]
                parts = payload.split("|")
                commit_hash = parts[0]
                author_date = parts[1] if len(parts) > 1 else ""
                committer_date = parts[2] if len(parts) > 2 else ""
                author_email = parts[3] if len(parts) > 3 else ""
                parents = parts[4] if len(parts) > 4 else ""

                is_merge = 1 if " " in parents.strip() else 0

                current_record = [
                    commit_hash,
                    author_date,
                    committer_date,
                    author_email,
                    is_merge,
                    0,  # files_changed
                    0,  # insertions
                    0,  # deletions
                ]
            else:
                match = SHORTSTAT_RE.search(line_str)
                if match and current_record:
                    files = int(match.group(1)) if match.group(1) else 0
                    ins = int(match.group(2)) if match.group(2) else 0
                    dels = int(match.group(3)) if match.group(3) else 0
                    current_record[5] = files
                    current_record[6] = ins
                    current_record[7] = dels

        if current_record:
            writer.writerow(current_record)
            processed_count += 1

    process.wait()
    return chunk_idx, processed_count


def extract_git_commits_parallel(repo_path: Path, output_csv: Path, num_workers: int) -> None:
    """
    Orchestrates parallel extraction of commit history and code churn:
    1. Collects all commit hashes (~1-2 seconds)
    2. Shards hashes into batches
    3. Distributes batches across multiprocessing Pool
    4. Merges temporary chunk CSVs into the final raw CSV
    """
    commit_hashes = get_all_commit_hashes(repo_path)
    total_commits = len(commit_hashes)
    print(f"[✓] Discovered {total_commits:,} total commits.", flush=True)

    if TEMP_CHUNKS_DIR.exists():
        shutil.rmtree(TEMP_CHUNKS_DIR)
    TEMP_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    chunks = [
        commit_hashes[i : i + CHUNK_SIZE]
        for i in range(0, total_commits, CHUNK_SIZE)
    ]
    total_chunks = len(chunks)

    print(
        f"[*] Launching {num_workers} parallel workers for {total_chunks} chunks "
        f"({CHUNK_SIZE:,} commits/chunk)...",
        flush=True,
    )

    worker_args = [
        (idx, chunk, repo_path, TEMP_CHUNKS_DIR / f"chunk_{idx:05d}.csv")
        for idx, chunk in enumerate(chunks)
    ]

    completed_chunks = 0
    total_extracted = 0

    with mp.Pool(processes=num_workers) as pool:
        for idx, count in pool.imap_unordered(_process_chunk_worker, worker_args):
            completed_chunks += 1
            total_extracted += count
            pct = (completed_chunks / total_chunks) * 100
            print(
                f"    -> Progress: [{completed_chunks:>4}/{total_chunks}] ({pct:>5.1f}%) "
                f"| Commits parsed: {total_extracted:,}",
                flush=True,
            )

    print("[*] Merging chunk files into unified CSV...", flush=True)
    chunk_files = sorted(TEMP_CHUNKS_DIR.glob("chunk_*.csv"))

    with open(output_csv, mode="w", newline="", encoding="utf-8") as out_f:
        header_written = False
        for cfile in chunk_files:
            with open(cfile, mode="r", encoding="utf-8") as in_f:
                lines = in_f.readlines()
                if not lines:
                    continue
                if not header_written:
                    out_f.write(lines[0])
                    header_written = True
                out_f.writelines(lines[1:])

    shutil.rmtree(TEMP_CHUNKS_DIR, ignore_errors=True)
    print(f"[✓] Raw commits extracted and merged: {total_extracted:,} -> {output_csv}", flush=True)


def build_time_series(raw_csv: Path, daily_csv: Path, weekly_csv: Path) -> None:
    """
    Loads raw commit data, sorts chronologically by committer date (UTC),
    and resamples into uniform Daily and Weekly time series tables.
    """
    print("[*] Loading extracted data into Pandas for time series resampling...", flush=True)

    df = pd.read_csv(
        raw_csv,
        usecols=[
            "committer_date",
            "author_email",
            "is_merge",
            "files_changed",
            "insertions",
            "deletions",
        ],
    )

    df["committer_date"] = pd.to_datetime(df["committer_date"], utc=True)
    df = df.set_index("committer_date").sort_index()

    # Pre-calculate indicators for efficient single-pass aggregation
    df["non_merge_commits"] = (df["is_merge"] == 0).astype(int)
    df["merge_commits"] = (df["is_merge"] == 1).astype(int)

    def aggregate_window(rule: str) -> pd.DataFrame:
        agg_spec = {
            "is_merge": "size",
            "non_merge_commits": "sum",
            "merge_commits": "sum",
            "author_email": "nunique",
            "files_changed": "sum",
            "insertions": "sum",
            "deletions": "sum",
        }
        ts_df = df.resample(rule).agg(agg_spec).rename(
            columns={
                "is_merge": "total_commits",
                "author_email": "unique_authors",
            }
        ).fillna(0)

        ts_df["total_churn"] = ts_df["insertions"] + ts_df["deletions"]
        ts_df["net_code_growth"] = ts_df["insertions"] - ts_df["deletions"]
        return ts_df

    # 1. Daily Time Series
    print("    -> Building daily time series (s=7 seasonality)...", flush=True)
    daily_ts = aggregate_window("D")
    daily_ts.to_csv(daily_csv, date_format="%Y-%m-%d")
    print(f"[✓] Saved Daily TS ({len(daily_ts):,} rows) to: {daily_csv}", flush=True)

    # 2. Weekly Time Series
    print("    -> Building weekly time series (smoothed weekend variance)...", flush=True)
    weekly_ts = aggregate_window("W-MON")
    weekly_ts.to_csv(weekly_csv, date_format="%Y-%m-%d")
    print(f"[✓] Saved Weekly TS ({len(weekly_ts):,} rows) to: {weekly_csv}", flush=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and resample Linux kernel commit history into regular time series."
    )
    parser.add_argument(
        "repo_path",
        nargs="?",
        default=DEFAULT_REPO_PATH,
        type=Path,
        help=f"Path to cloned Linux repository (default: {DEFAULT_REPO_PATH})",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Force full re-extraction from Git even if raw CSV already exists",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("EXTRACT_WORKERS", DEFAULT_WORKERS)),
        help=f"Number of worker processes (default: {DEFAULT_WORKERS})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    repo_path = args.repo_path.resolve()
    num_workers = args.workers
    force_extract = args.force_extract

    start_time = datetime.now()
    print("=" * 65, flush=True)
    print("Linux Kernel Time Series Parallel Pipeline", flush=True)
    print(f"Target repository : {repo_path}", flush=True)
    print(f"Raw CSV target    : {RAW_CSV_PATH}", flush=True)
    print(f"Resampled TS dir  : {DATASET_DIR}", flush=True)
    print(f"Parallel workers  : {num_workers} processes", flush=True)
    print("=" * 65, flush=True)

    verify_repo(repo_path)

    # If raw CSV already exists and is populated, skip re-extracting from Git
    if not force_extract and RAW_CSV_PATH.exists() and RAW_CSV_PATH.stat().st_size > 1024:
        print(f"[*] Found existing raw commits file ({RAW_CSV_PATH.name}).", flush=True)
        print("    Skipping Git extraction. (Pass --force-extract to re-run).", flush=True)
    else:
        extract_git_commits_parallel(repo_path, RAW_CSV_PATH, num_workers)

    build_time_series(RAW_CSV_PATH, DAILY_TS_PATH, WEEKLY_TS_PATH)

    elapsed = datetime.now() - start_time
    print("=" * 65, flush=True)
    print(f"[✓] Complete pipeline finished in {elapsed.total_seconds():.1f}s", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
