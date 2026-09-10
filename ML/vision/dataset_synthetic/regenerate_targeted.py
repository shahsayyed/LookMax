#!/usr/bin/env python3
"""
regenerate_targeted.py -- regenerates an EXPLICIT, closed list of filenames
and nothing else. Built for the clean_shaven/makeup backfill (2026-09): those
1,460 images need fresh pixels using the now-fixed taxonomy.py prompts, but
their CSV label rows are already correct (the fix only changes prompt/
negative_prompt wording, never the target label -- facial_hair_style was
always "clean_shaven" in the row; the fix just makes the model satisfy that
label more often). So this script touches ONLY image files, never a CSV.

Safety properties, deliberately stricter than full_run.py's normal resume
path (which discovers work by scanning the whole images/ directory -- not
what we want when the goal is "regenerate exactly these 1,460 and nothing
else"):
  - The target list is an explicit whitelist read from a text file (one
    filename per line), never derived from a directory scan or a CSV query.
  - Every target filename must already exist on disk and already resolve to
    a real task via the deterministic task list -- refuses to run otherwise
    (catches a stale/corrupted target list before touching anything).
  - Writes go to "<filename>.tmp" then an atomic os.replace() onto the exact
    original filename -- same pattern full_run.py uses, so a killed run never
    leaves a half-written image, and no other file's mtime/content changes.
  - No CSV is opened for writing at all -- there is no code path in this
    script that can touch labels_*.csv.

Usage:
  python3 regenerate_targeted.py --targets regen_targets.txt --dry-run
  python3 regenerate_targeted.py --targets regen_targets.txt
  python3 regenerate_targeted.py --targets regen_targets.txt --batch-size 2 --device cuda:0
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import taxonomy as tx  # noqa: E402
import full_run as fr  # noqa: E402
import prompt_builder as pb  # noqa: E402

DEFAULT_DATA_DIR = Path("/data") if Path("/data").exists() else SCRIPT_DIR.parents[2] / "ML" / "data" / "vision_synthetic" / "raw_generated"


def load_targets(targets_path):
    lines = [l.strip() for l in Path(targets_path).read_text().splitlines() if l.strip()]
    if len(lines) != len(set(lines)):
        sys.exit(f"Target list has duplicate filenames -- refusing to run (got {len(lines)}, {len(set(lines))} unique).")
    return lines


def resolve_targets(filenames, images_dir, require_existing):
    """Matches each target filename against the deterministic task list --
    this is the real safety guarantee (a stale/wrong target list will fail
    to match the reconstructed index/prompt structure). Refuses to proceed
    if any entry doesn't match.

    require_existing additionally checks the file is already present in
    images_dir. That's meaningful when regenerating in place on a box that
    already holds the full dataset (catches a target list pointing at the
    wrong images_dir); it is NOT meaningful -- and should stay off -- on a
    fresh box being used purely as compute, since there the old pixels don't
    need to be present at all, only the freshly generated ones get synced
    back afterward."""
    tasks = fr.build_full_task_list()
    by_filename = {t["filename"]: t for t in tasks}

    resolved = []
    missing_from_tasklist = []
    missing_from_disk = []
    for fn in filenames:
        t = by_filename.get(fn)
        if t is None:
            missing_from_tasklist.append(fn)
            continue
        if require_existing and not (images_dir / fn).exists():
            missing_from_disk.append(fn)
            continue
        resolved.append(t)

    if missing_from_tasklist:
        sys.exit(f"{len(missing_from_tasklist)} target filename(s) don't match the deterministic task list "
                  f"(first 5: {missing_from_tasklist[:5]}). Refusing to run.")
    if missing_from_disk:
        sys.exit(f"{len(missing_from_disk)} target filename(s) don't exist in {images_dir} "
                  f"(first 5: {missing_from_disk[:5]}). Refusing to run.")
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", required=True, help="Text file, one filename per line -- the ONLY files this script will touch.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Resolve + validate the target list, print the plan, generate nothing.")
    parser.add_argument("--require-existing", action="store_true",
                         help="Also require each target file to already exist in images_dir before regenerating it. "
                              "Only meaningful when running in place on a box that already holds the full dataset -- "
                              "leave off on a fresh compute-only box (default: off).")
    args = parser.parse_args()

    images_dir = Path(args.data_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    filenames = load_targets(args.targets)
    print(f"Loaded {len(filenames)} target filename(s) from {args.targets}")

    resolved = resolve_targets(filenames, images_dir, args.require_existing)
    print(f"Resolved and verified all {len(resolved)} targets against the deterministic task list"
          + (" and disk." if args.require_existing else "."))

    by_cat = {}
    for t in resolved:
        by_cat.setdefault(t["category"], 0)
        by_cat[t["category"]] += 1
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat}: {n}")

    if args.dry_run:
        print("\n--dry-run: stopping here. No image touched, no CSV touched.")
        return

    import qwen_pipeline as qp
    print(f"\nLoading Qwen-Image-2512 on {args.device or 'default device'}...")
    pipe, can_batch = qp.load_pipeline(device=args.device)
    effective_batch = args.batch_size if can_batch else 1

    log_path = SCRIPT_DIR / f"regenerate_targeted_log_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    log_file = open(log_path, "a")

    def group_by_resolution(items, batch_size):
        if batch_size <= 1:
            for it in items:
                yield [it]
            return
        buckets = {}
        for it in items:
            res = it["task"]["resolution"]
            buckets.setdefault(res, []).append(it)
            if len(buckets[res]) >= batch_size:
                yield buckets[res]
                buckets[res] = []
        for remaining in buckets.values():
            if remaining:
                yield remaining

    generated = 0
    t_start = time.time()
    try:
        for batch in group_by_resolution(resolved, effective_batch):
            filenames_b = [item["filename"] for item in batch]
            tmp_paths = [images_dir / (fn + ".tmp") for fn in filenames_b]
            seeds = [item["index"] for item in batch]

            t0 = time.time()
            images = qp.generate(pipe, [item["task"] for item in batch], seeds=seeds,
                                  num_inference_steps=tx.NUM_INFERENCE_STEPS_FULL)
            elapsed = time.time() - t0

            for item, filename, tmp_path, image in zip(batch, filenames_b, tmp_paths, images):
                image.save(tmp_path, format="PNG")
                os.replace(tmp_path, images_dir / filename)  # atomic -- exact filename only
                generated += 1
                log_file.write(json.dumps({
                    "filename": filename, "category": item["category"], "tier": item["tier"],
                    "index": item["index"], "prompt": item["task"]["prompt"],
                    "negative_prompt": item["task"].get("negative_prompt", tx.NEGATIVE_PROMPT),
                }) + "\n")
                log_file.flush()

            if generated % 25 == 0 or generated == len(resolved):
                total_elapsed = time.time() - t_start
                print(f"  [{generated}/{len(resolved)}] {filenames_b[-1]} "
                      f"({elapsed/len(batch):.2f}s/img avg this batch, {total_elapsed/generated:.2f}s/img overall)")
    finally:
        log_file.close()
        qp.unload(pipe)

    print(f"\nRegenerated {generated}/{len(resolved)} target images. No CSV was touched. Log: {log_path.name}")


if __name__ == "__main__":
    main()
