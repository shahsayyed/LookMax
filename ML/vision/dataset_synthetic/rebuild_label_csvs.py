#!/usr/bin/env python3
"""
rebuild_label_csvs.py -- regenerates COMPLETE labels_<Category>.csv files for
all 28,000 images from the deterministic task list (full_run.py's
build_full_task_list() + prompt_builder.row_for_csv()), instead of relying on
the per-host label CSVs that were supposed to sync back from Vast.ai.

Why this exists: as of 2026-09-09 the local raw_generated/ CSVs only cover
~5,141 of the 28,000 generated images (~18%) -- most remote hosts' CSVs never
made it back before their instances were destroyed (same root cause noted in
verify_dataset_vertex.py's docstring for generation_log.jsonl). The 28,000
images themselves ARE all present on disk; only their CSV label rows are
missing. Since prompt/label generation is fully deterministic (seeded rng,
verified byte-for-byte against every one of the 5,141 rows that DID survive
sync -- 0 mismatches across all 4 categories), the complete, correct label
set can just be rebuilt from scratch rather than chased down per-host.

Every existing CSV (partial or not) is backed up with a timestamped .bak
before being overwritten. Only rows whose image file actually exists on disk
are written, so a genuinely-missing image (if any) doesn't get a label row.

Usage:
  python3 rebuild_label_csvs.py                  # dry-run: report only
  python3 rebuild_label_csvs.py --write           # back up + overwrite CSVs
"""
import argparse
import csv
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
RAW_DIR = REPO_ROOT / "ML" / "data" / "vision_synthetic" / "raw_generated"
IMAGES_DIR = RAW_DIR / "images"

sys.path.insert(0, str(SCRIPT_DIR))
import taxonomy as tx  # noqa: E402
import full_run as fr  # noqa: E402
import prompt_builder as pb  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help="Actually back up + overwrite the CSVs (default: dry-run report only)")
    args = parser.parse_args()

    tasks = fr.build_full_task_list()
    by_category = {c: [] for c in tx.ALL_CATEGORIES}
    for t in tasks:
        by_category[t["category"]].append(t)

    for category in tx.ALL_CATEGORIES:
        items = by_category[category]
        rows = []
        missing_images = 0
        for t in items:
            fn = t["filename"]
            if not (IMAGES_DIR / fn).exists():
                missing_images += 1
                continue
            rows.append(pb.row_for_csv(category, t["tier"], fn, t["task"]))

        csv_path = RAW_DIR / f"labels_{category}.csv"
        existing_count = 0
        if csv_path.exists():
            existing_count = sum(1 for _ in csv.DictReader(csv_path.open()))

        print(f"{category}: {len(rows)} rows to write ({missing_images} task(s) with no image on disk, "
              f"skipped) -- existing file currently has {existing_count} row(s)")

        if args.write:
            if csv_path.exists():
                bak = csv_path.with_suffix(f".csv.bak.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
                shutil.copy2(csv_path, bak)
                print(f"  backed up existing {csv_path.name} -> {bak.name}")
            # Also back up + remove now-superseded shard files so a future
            # merge_shards.py run doesn't clobber this with stale partial data.
            for shard_path in sorted(RAW_DIR.glob(f"labels_{category}_shard*.csv")):
                bak = shard_path.with_suffix(f".csv.bak.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
                shutil.move(str(shard_path), str(bak))
                print(f"  moved stale shard {shard_path.name} -> {bak.name}")
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=tx.schema_columns(category))
                writer.writeheader()
                writer.writerows(rows)
            print(f"  -> wrote {len(rows)} rows to {csv_path.name}")

    if not args.write:
        print("\n(dry-run -- pass --write to actually back up and overwrite the CSVs)")


if __name__ == "__main__":
    main()
