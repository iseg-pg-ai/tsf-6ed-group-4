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

RELEASE CALENDAR:
-----------------
The release cycle (merge window vs. rc phase) is read separately from the
project's annotated tags. That walk touches ~90 refs, not 1.3M commits, so it
is cheap enough to be re-run on its own at any time:

    python extractor.py --tags-only

The resulting calendar is cached in dataset/kernel_releases.csv and left-joined
(backwards only) onto both time series, so no future release date ever reaches
an observation dated before it.
================================================================================
"""

import argparse
import csv
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

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
# - Raw commit event log is saved in this script's directory (dataset/Extractor/)
EXTRACTOR_DIR = Path(__file__).resolve().parent
# - Resampled time series CSV files are saved in the parent directory (dataset/)
dataset_DIR = EXTRACTOR_DIR.parent

# Output filenames
RAW_CSV_PATH = EXTRACTOR_DIR / "kernel_raw_commits.csv"
DAILY_TS_PATH = dataset_DIR / "kernel_daily_ts.csv"
WEEKLY_TS_PATH = dataset_DIR / "kernel_weekly_ts.csv"
RELEASES_CSV_PATH = dataset_DIR / "kernel_releases.csv"
TEMP_CHUNKS_DIR = EXTRACTOR_DIR / "_kernel_ts_chunks"

# Regex to parse git log's --shortstat line:
# Example: " 3 files changed, 24 insertions(+), 5 deletions(-)"
SHORTSTAT_RE = re.compile(
    r"(\d+)\s+file[s]?\s+changed"
    r"(?:,\s+(\d+)\s+insertion[s]?\(\+\))?"
    r"(?:,\s+(\d+)\s+deletion[s]?\(-\))?"
)

# Matches upstream release tags only: v2.6.12, v3.18, v4.0, v6.12-rc7.
# Deliberately rejects non-release refs such as `v2.6.11-tree` and stable-tree
# patch tags such as `v6.6.30`.
TAG_NAME_RE = re.compile(r"^v(?P<major>\d+)\.(?P<minor>\d+)(?:\.(?P<patch>\d+))?(?:-rc(?P<rc>\d+))?$")

# Number of weeks the merge window stays open after a final release.
MERGE_WINDOW_WEEKS = 2

# Columns appended to both time series by the release-cycle enrichment.
RELEASE_COLUMNS = [
    "last_release_tag",
    "last_release_date",
    "weeks_since_release",
    "is_merge_window",
    "rc_week",
]


def verify_repo(path: Path) -> None:
    """Verifies that the target path exists and is a valid Git repository."""
    if not path.exists():
        print(f"[ERROR] Path does not exist: {path}", file=sys.stderr, flush=True)
        print(
            "Please configure DEFAULT_REPO_PATH at the top of this script or pass the path as an argument:",
            file=sys.stderr,
            flush=True,
        )
        print(f"  python {Path(__file__).name} /path/to/linux-bare.git", file=sys.stderr, flush=True)
        sys.exit(1)

    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--is-bare-repository"],
        capture_output=True,
        text=True,
        check=False,
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
        capture_output=True,
        text=True,
        check=False,
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

    chunks = [commit_hashes[i : i + CHUNK_SIZE] for i in range(0, total_commits, CHUNK_SIZE)]
    total_chunks = len(chunks)

    print(
        f"[*] Launching {num_workers} parallel workers for {total_chunks} chunks ({CHUNK_SIZE:,} commits/chunk)...",
        flush=True,
    )

    worker_args = [
        (idx, chunk, repo_path, TEMP_CHUNKS_DIR / f"chunk_{idx:05d}.csv") for idx, chunk in enumerate(chunks)
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


def extract_release_tags(repo_path: Path, output_csv: Path) -> pd.DataFrame:
    """Reads the upstream release tags and writes a normalised release calendar.

    Only `refs/tags` is walked, so this costs a few seconds regardless of how
    large the history is. Annotated tags expose the signing date through
    `%(taggerdate)`, which is the authoritative release moment; the peeled commit
    date and the plain commit date act as fallbacks for lightweight tags.
    """
    print("[*] Reading release tags via `git for-each-ref refs/tags`...", flush=True)

    fmt = "%(refname:short)|%(taggerdate:iso-strict)|%(*committerdate:iso-strict)|%(committerdate:iso-strict)"
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_path),
            "for-each-ref",
            "--sort=taggerdate",
            f"--format={fmt}",
            "refs/tags",
        ],
        capture_output=True,
        text=True,
        check=False,
        errors="replace",
    )
    if result.returncode != 0:
        print(f"[ERROR] Failed to list release tags: {result.stderr}", file=sys.stderr, flush=True)
        sys.exit(result.returncode)

    records: list[dict[str, object]] = []

    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue

        tag, tagger_date, peeled_date, commit_date = (part.strip() for part in parts[:4])

        match = TAG_NAME_RE.match(tag)
        if not match:
            continue

        raw_date = tagger_date or peeled_date or commit_date
        if not raw_date:
            continue

        timestamp = pd.to_datetime(raw_date, utc=True, errors="coerce")
        if pd.isna(timestamp):
            continue

        rc_group = match.group("rc")
        patch_group = match.group("patch")

        records.append({
            "tag": tag,
            "release_date": timestamp,
            "is_rc": 0 if rc_group is None else 1,
            "rc_number": 0 if rc_group is None else int(rc_group),
            "version_major": int(match.group("major")),
            "version_minor": int(match.group("minor")),
            "version_patch": int(patch_group) if patch_group else 0,
        })

    if not records:
        print(
            "[WARN] No valid release tags found. The release-cycle columns will be skipped.",
            file=sys.stderr,
            flush=True,
        )
        return pd.DataFrame()

    df_tags = pd.DataFrame(records).sort_values("release_date").reset_index(drop=True)
    df_tags.to_csv(output_csv, index=False, date_format="%Y-%m-%d")

    total = len(df_tags)
    finals = int((df_tags["is_rc"] == 0).sum())
    print(
        f"[OK] Release calendar: {total} tags ({finals} final releases, {total - finals} rc) -> {output_csv}",
        flush=True,
    )
    return df_tags


def load_release_tags(path: Path) -> pd.DataFrame:
    """Loads the cached release calendar, or an empty frame if it is absent."""
    if not path.exists():
        print(
            f"[WARN] Release calendar not found at {path}. The release-cycle columns will be skipped.",
            file=sys.stderr,
            flush=True,
        )
        return pd.DataFrame()

    df_tags = pd.read_csv(path, parse_dates=["release_date"])
    if df_tags.empty:
        return df_tags

    df_tags["release_date"] = pd.to_datetime(df_tags["release_date"], utc=True)
    return df_tags.sort_values("release_date").reset_index(drop=True)


def enrich_with_release_cycle(ts_df: pd.DataFrame, releases: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Attaches the release-cycle state observable at each point in time.

    The join is `direction="backward"`, so an observation only ever sees the most
    recent *past* final release. Using the next release date instead would leak
    future information into the regressors and invalidate any out-of-sample
    comparison.

    A `W-MON` bin is labelled with its **end** date, so the cycle counter is
    evaluated at the bin's start. Otherwise the week containing a release would
    be labelled as if the whole week came after it, shifting the phase by one.
    """
    finals = releases.loc[releases["is_rc"] == 0, ["tag", "release_date"]].copy()
    if finals.empty:
        return ts_df

    period_days = 7 if rule.startswith("W") else 1

    frame = ts_df.reset_index()
    frame["_reference_date"] = frame["committer_date"] - pd.Timedelta(days=period_days - 1)

    calendar = finals.rename(columns={"tag": "last_release_tag", "release_date": "last_release_date"}).sort_values(
        "last_release_date"
    )

    merged = pd.merge_asof(
        frame.sort_values("_reference_date").reset_index(drop=True),
        calendar,
        left_on="_reference_date",
        right_on="last_release_date",
        direction="backward",
        allow_exact_matches=True,
    )

    elapsed_days = (merged["_reference_date"] - merged["last_release_date"]).dt.total_seconds() / 86400.0

    weeks = (elapsed_days // 7.0).astype("Int64")
    merged["weeks_since_release"] = weeks
    merged["is_merge_window"] = (weeks < MERGE_WINDOW_WEEKS).astype("Int64")

    # Observations before the first tag keep NA in every release-cycle column.
    # The comparison below is built from `notna()` first so that `mask` never
    # receives a nullable condition: pandas would treat NA as True and silently
    # turn "unknown" into "merge window".
    phase = weeks - (MERGE_WINDOW_WEEKS - 1)
    negative = phase.notna() & (phase < 0)
    merged["rc_week"] = phase.mask(negative, 0).astype("Int64")

    merged = merged.drop(columns=["_reference_date"]).set_index("committer_date").sort_index()

    ordered = [column for column in ts_df.columns] + RELEASE_COLUMNS
    return merged[ordered]


def report_release_cycle(ts_df: pd.DataFrame, label: str) -> None:
    """Prints a short sanity check of the release-cycle attachment."""
    if "weeks_since_release" not in ts_df.columns:
        return

    matched = int(ts_df["weeks_since_release"].notna().sum())
    total = len(ts_df)
    merge_mean = ts_df.loc[ts_df["is_merge_window"] == 1, "total_commits"].mean()
    rc_mean = ts_df.loc[ts_df["is_merge_window"] == 0, "total_commits"].mean()

    print(f"    -> Release cycle [{label}]: {matched}/{total} observations matched to a past release")
    if matched < total:
        print(
            f"       [!] {total - matched} observations precede the first available tag "
            "and stay unmatched (NaN) by design."
        )

    print(
        f"       Mean total_commits: merge window = {merge_mean:.1f} "
        f"| rc phase = {rc_mean:.1f} | ratio = {merge_mean / rc_mean:.2f}x"
    )


def build_time_series(
    raw_csv: Path,
    daily_csv: Path,
    weekly_csv: Path,
    releases: pd.DataFrame,
) -> None:
    """
    Loads raw commit data, sorts chronologically by committer date (UTC),
    resamples into uniform Daily and Weekly time series tables, and attaches the
    release-cycle state (merge window vs. rc phase) to every observation.
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
        resampled = df.resample(rule).agg(agg_spec)
        assert isinstance(resampled, pd.DataFrame)
        ts_df = resampled.rename(
            columns={
                "is_merge": "total_commits",
                "author_email": "unique_authors",
            }
        ).fillna(0)

        ts_df["total_churn"] = ts_df["insertions"] + ts_df["deletions"]
        ts_df["net_code_growth"] = ts_df["insertions"] - ts_df["deletions"]

        if not releases.empty:
            ts_df = enrich_with_release_cycle(ts_df, releases, rule)

        return ts_df

    # 1. Daily Time Series
    print("    -> Building daily time series (s=7 seasonality)...", flush=True)
    daily_ts = aggregate_window("D")
    report_release_cycle(daily_ts, "daily")
    daily_ts.to_csv(daily_csv, date_format="%Y-%m-%d")
    print(f"[✓] Saved Daily TS ({len(daily_ts):,} rows) to: {daily_csv}", flush=True)

    # 2. Weekly Time Series
    print("    -> Building weekly time series (smoothed weekend variance)...", flush=True)
    weekly_ts = aggregate_window("W-MON")
    report_release_cycle(weekly_ts, "weekly")
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
        "--force-tags",
        action="store_true",
        help="Force re-reading of release tags even if the release calendar CSV already exists",
    )
    parser.add_argument(
        "--tags-only",
        action="store_true",
        help="Only refresh the release calendar (dataset/kernel_releases.csv) and exit",
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
    force_tags = args.force_tags

    start_time = datetime.now()
    print("=" * 65, flush=True)
    print("Linux Kernel Time Series Parallel Pipeline", flush=True)
    print(f"Target repository : {repo_path}", flush=True)
    print(f"Raw CSV target    : {RAW_CSV_PATH}", flush=True)
    print(f"Release calendar  : {RELEASES_CSV_PATH}", flush=True)
    print(f"Resampled TS dir  : {dataset_DIR}", flush=True)
    print(f"Parallel workers  : {num_workers} processes", flush=True)
    print("=" * 65, flush=True)

    # The release calendar alone can be refreshed without touching the history.
    if args.tags_only:
        verify_repo(repo_path)
        extract_release_tags(repo_path, RELEASES_CSV_PATH)
        print("=" * 65, flush=True)
        print("[OK] Release calendar refreshed. Time series were left untouched.", flush=True)
        print("=" * 65, flush=True)
        return

    raw_cached = not force_extract and RAW_CSV_PATH.exists() and RAW_CSV_PATH.stat().st_size > 1024
    tags_cached = not force_tags and RELEASES_CSV_PATH.exists() and RELEASES_CSV_PATH.stat().st_size > 64

    # Git access is only required for whatever is actually missing, so a pure
    # resample can still run on a machine without the 4.5 GB clone.
    if not (raw_cached and tags_cached):
        verify_repo(repo_path)

    if raw_cached:
        print(f"[*] Found existing raw commits file ({RAW_CSV_PATH.name}).", flush=True)
        print("    Skipping Git extraction. (Pass --force-extract to re-run).", flush=True)
    else:
        extract_git_commits_parallel(repo_path, RAW_CSV_PATH, num_workers)

    if tags_cached:
        print(f"[*] Found existing release calendar ({RELEASES_CSV_PATH.name}).", flush=True)
        print("    Skipping tag walk. (Pass --force-tags to re-run).", flush=True)
        releases = load_release_tags(RELEASES_CSV_PATH)
    else:
        releases = extract_release_tags(repo_path, RELEASES_CSV_PATH)

    build_time_series(RAW_CSV_PATH, DAILY_TS_PATH, WEEKLY_TS_PATH, releases)

    elapsed = datetime.now() - start_time
    print("=" * 65, flush=True)
    print(f"[✓] Complete pipeline finished in {elapsed.total_seconds():.1f}s", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
