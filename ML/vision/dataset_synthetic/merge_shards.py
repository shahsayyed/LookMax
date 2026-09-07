"""
merge_shards.py -- combines per-shard, per-category label CSVs
(labels_<Category>_shard<k>.csv, written by full_run.py --shard k
--num-shards n) into one labels_<Category>.csv per category.

WARNS on duplicate filenames instead of silently double-counting them.
A duplicate filename across shard files means two workers were (almost
certainly by mistake) given the SAME --shard value, so they generated
overlapping task indices -- concatenating blindly would double-count
those rows in the merged dataset without any visible sign anything went
wrong. This script keeps the FIRST occurrence of each filename, drops
later duplicates, and prints exactly which filenames/shard files
collided so the mistake is visible and fixable.

Usage:
    python3 merge_shards.py [--data-dir /data] [--categories Men_Grooming ...]
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taxonomy as tx
import full_run


def merge_category(output_dir, category):
    shard_files = sorted(output_dir.glob(f"labels_{category}_shard*.csv"))
    if not shard_files:
        # Non-sharded run already writes labels_<Category>.csv directly --
        # nothing to merge, not an error.
        single = output_dir / f"labels_{category}.csv"
        if single.exists():
            print(f"{category}: no shard files found; {single.name} already exists (non-sharded run). Skipping.")
        else:
            print(f"{category}: no shard files and no existing labels file found. Skipping.")
        return

    fieldnames = tx.schema_columns(category)
    seen = {}
    duplicates = []
    order = []

    for shard_file in shard_files:
        with open(shard_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fn = row.get("filename")
                if fn in seen:
                    duplicates.append((fn, seen[fn], shard_file.name))
                    continue
                seen[fn] = shard_file.name
                order.append(row)

    merged_path = output_dir / f"labels_{category}.csv"
    with open(merged_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in order:
            writer.writerow(row)

    print(f"{category}: merged {len(shard_files)} shard file(s) -> {merged_path.name} "
          f"({len(order)} unique rows).")
    if duplicates:
        print(f"  WARNING: {len(duplicates)} duplicate filename(s) found across shard files -- "
              f"kept the first occurrence, dropped the rest. This almost always means two workers "
              f"were given the SAME --shard value by mistake. Duplicates:")
        for fn, first_shard, dup_shard in duplicates[:20]:
            print(f"    {fn}: kept from {first_shard}, dropped duplicate in {dup_shard}")
        if len(duplicates) > 20:
            print(f"    ... and {len(duplicates) - 20} more.")


def merge_logs(output_dir):
    shard_logs = sorted(output_dir.glob("generation_log_shard*.jsonl"))
    if not shard_logs:
        return
    import json
    seen_indices = set()
    merged_lines = []
    main_log = output_dir / "generation_log.jsonl"
    if main_log.exists():
        with open(main_log) as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        idx = d.get("index")
                        if idx is not None:
                            seen_indices.add(idx)
                        merged_lines.append(line.strip())
                    except Exception:
                        merged_lines.append(line.strip())

    duplicates = 0
    for s_log in shard_logs:
        with open(s_log) as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        idx = d.get("index")
                        if idx is not None and idx in seen_indices:
                            duplicates += 1
                            continue
                        if idx is not None:
                            seen_indices.add(idx)
                        merged_lines.append(line.strip())
                    except Exception:
                        merged_lines.append(line.strip())

    with open(main_log, "w") as f:
        for line in merged_lines:
            f.write(line + "\n")
    print(f"generation_log: merged {len(shard_logs)} shard log(s) -> {main_log.name} ({len(merged_lines)} entries).")


def main():
    parser = argparse.ArgumentParser(description="Merge per-shard label CSVs into one CSV per category.")
    parser.add_argument("--data-dir", default=str(full_run.DEFAULT_DATA_DIR))
    parser.add_argument("--categories", nargs="*", default=tx.ALL_CATEGORIES)
    args = parser.parse_args()

    output_dir = full_run.output_paths(args.data_dir)["output_dir"]
    if not output_dir.exists():
        sys.exit(f"Output dir {output_dir} does not exist -- nothing to merge.")

    for category in args.categories:
        merge_category(output_dir, category)
    merge_logs(output_dir)


if __name__ == "__main__":
    main()
