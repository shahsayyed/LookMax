"""
full_run.py -- the 28,000-image production run (Qwen-Image-2512).

Only ever import heavy GPU deps (torch, diffusers -- via qwen_pipeline.py)
LAZILY inside functions that actually generate. Task-list construction,
--dry-run, and --benchmark's pre-flight printing must all work with no
CUDA stack present, so validation_sweep.py can import
build_full_task_list() from this module for --coverage-only without
paying for or requiring a GPU import.

TASK LIST: seeded (random.Random(TASK_SEED)), NOT random.Random() with no
seed -- see build_full_task_list()'s docstring. This is load-bearing for
resume: index N must mean the exact same prompt/labels on every run, or
"skip if the file already exists" silently corrupts the dataset (an image
gets skipped because SOME file with that name exists, but a re-run without
the seed would have produced different content for that name).

ATOMIC WRITES: each image is saved to a ".tmp" name first, and only
renamed to its final name after its CSV row is written AND flushed (same
pattern as ML/archive/dataset_generator_v7/generate_qwen_dataset.py). A
process killed mid-image (OOM, Ctrl+C, spot-instance preemption) can never
leave a half-written file that LOOKS done and gets silently skipped
forever on the next run.

DEFAULT BATCH SIZE IS 1 -- measured, not a cautious guess: on an RTX PRO
6000 Blackwell (96GB), batch=1 ran ~0.64s/step/image while batch=4 ran
~0.73-0.75s/step/image for this exact model at 1024x1024 -- i.e. batching
gave NO throughput benefit (slightly worse), because a 20B-param
transformer at this resolution already saturates the GPU's compute at
batch=1. See qwen_pipeline.py's module docstring. If you're benchmarking
on different hardware, re-check with --benchmark before assuming a higher
gen_batch_size helps.
"""
import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taxonomy as tx
import prompt_builder as pb

# --------------------------------------------------------------------------
# Seeding -- see module docstring. TASK_SEED drives every sampled value in
# every task's content; SHUFFLE_SEED is a SEPARATE deterministic seed used
# only to reorder the (already-built) task list, so a partial run
# interleaves categories/tiers (see build_full_task_list()) without
# changing what task index N actually contains.
# --------------------------------------------------------------------------
TASK_SEED = 42
SHUFFLE_SEED = 43

# Default is a LOCAL directory next to this script, deliberately NOT
# hardcoded to "/data" the way the archived pipeline was -- that hardcoding
# meant `full_run.py --dry-run` (and this project's own acceptance check)
# couldn't run on a laptop with no /data mount. On an actual Vast.ai box,
# set LOOKMAX_DATA_DIR=/data (see install.sh / PLAN.md) or pass
# --data-dir /data explicitly -- the underlying lesson (don't silently
# write 28,000 images to a tiny /workspace loop device) is preserved by
# check_disk_space() actually checking whatever directory is in play,
# not by hardcoding one path that only exists on one specific host.
DEFAULT_DATA_DIR = Path(os.environ.get("LOOKMAX_DATA_DIR", str(Path(__file__).resolve().parent / "output")))
MIN_FREE_GB = 150  # ~58GB model cache + 28,000 PNGs (grooming 1024x1024 + outfit 768x1024) with headroom
DEFAULT_GEN_BATCH_SIZE = 1
MAX_CONSECUTIVE_FAILURES = 50


def output_paths(data_dir):
    output_dir = Path(data_dir) / "qwen_dataset_output"
    return {
        "output_dir": output_dir,
        "images_dir": output_dir / "images",
        "log_path": output_dir / "generation_log.jsonl",
    }


def labels_csv_path(output_dir, category, shard=None):
    if shard is None:
        return output_dir / f"labels_{category}.csv"
    return output_dir / f"labels_{category}_shard{shard}.csv"


def schema_path(output_dir, category):
    return output_dir / f"label_schema_{category}.json"


# --------------------------------------------------------------------------
# Task list
# --------------------------------------------------------------------------
def build_full_task_list(seed=TASK_SEED, shuffle_seed=SHUFFLE_SEED):
    """Builds every one of the 28,000 tasks in a fixed, deterministic
    order (category -> tier -> count, consuming a single seeded rng
    sequentially so index content is reproducible), then applies a
    SEPARATE deterministic shuffle so category/tier are interleaved --
    a partial run (first N of the shuffled list) still yields a roughly
    balanced dataset across all 4 categories and all 4 tiers, rather than
    finishing Men_Grooming entirely before starting anything else.

    Returns a list of dicts: {"index", "category", "tier", "filename",
    "task" (prompt_builder's build_task() result)}. `index` is the task's
    identity for resume purposes -- it is NOT the position in this list
    after shuffling is applied elsewhere; each task carries its own fixed
    index baked into its filename, so shuffling the list's order changes
    only the ORDER of generation, never which content index N means.
    """
    rng = random.Random(seed)
    tasks = []
    index = 0
    for category in tx.ALL_CATEGORIES:
        total = tx.CATEGORY_COUNTS[category]
        tier_counts = tx.tier_counts_for_total(total)
        for tier, count in tier_counts.items():
            for _ in range(count):
                task = pb.build_task(category, tier, rng)
                filename = f"{index:05d}_{category}_{tier}.png"
                tasks.append({"index": index, "category": category, "tier": tier,
                               "filename": filename, "task": task})
                index += 1

    order_rng = random.Random(shuffle_seed)
    order_rng.shuffle(tasks)
    return tasks


def write_schema_files(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for category in tx.ALL_CATEGORIES:
        with open(schema_path(output_dir, category), "w") as f:
            json.dump(tx.get_label_schema(category), f, indent=2)


# --------------------------------------------------------------------------
# Disk safety (ported from ML/archive/dataset_generator_v7/generate_qwen_dataset.py
# and install_qwen.sh -- Vast.ai's /workspace-is-tiny quirk. Checks the
# ACTUAL data_dir argument, never trusts cwd or shell env.)
# --------------------------------------------------------------------------
def check_disk_space(data_dir, min_free_gb=MIN_FREE_GB):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(data_dir).free / (1024 ** 3)
    if free_gb < min_free_gb:
        sys.exit(
            f"!! Only {free_gb:.1f}GB free on {data_dir} -- need at least {min_free_gb}GB for the full "
            f"run (model cache + ~28,000 images). Run 'df -h' and confirm {data_dir} is your LARGE disk "
            f"(on Vast.ai, /workspace is a tiny loop device -- use /data). Pass --data-dir to point "
            f"elsewhere if needed."
        )
    print(f"Disk check OK: {free_gb:.1f}GB free on {data_dir} (need >= {min_free_gb}GB).")


# --------------------------------------------------------------------------
# Resume: scan existing images, skip anything already on disk
# --------------------------------------------------------------------------
_FILENAME_RE = re.compile(r"^(\d{5})_")


def already_done_indices(images_dir):
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return set()
    done = set()
    for f in images_dir.iterdir():
        if f.is_file() and not f.name.endswith(".tmp"):
            m = _FILENAME_RE.match(f.name)
            if m:
                done.add(int(m.group(1)))
    return done


# --------------------------------------------------------------------------
# Batching -- group consecutive same-resolution tasks so gen_batch_size > 1
# stays correct even though the shuffled task order interleaves categories
# (grooming is square, outfit is portrait -- one pipe() call can't mix
# resolutions). With the default gen_batch_size=1 this always yields
# groups of size 1.
# --------------------------------------------------------------------------
def group_by_resolution(pending, gen_batch_size):
    group = []
    group_res = None
    for item in pending:
        res = item["task"]["resolution"]
        if group and (res != group_res or len(group) >= gen_batch_size):
            yield group
            group = []
        group.append(item)
        group_res = res
    if group:
        yield group


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------
def run_benchmark(data_dir, num_shards=None):
    import qwen_pipeline as qp

    tasks = build_full_task_list()
    sample = tasks[:5]
    if len(sample) < 5:
        sys.exit("Not enough tasks to benchmark (need at least 5).")

    print("Loading pipeline for benchmark...")
    pipe, can_batch = qp.load_pipeline()

    timings = []
    for i, item in enumerate(sample):
        t0 = time.time()
        qp.generate(pipe, [item["task"]], seeds=[item["index"]], num_inference_steps=tx.NUM_INFERENCE_STEPS_FULL)
        elapsed = time.time() - t0
        label = "warm-up (discarded)" if i == 0 else "timed"
        print(f"  image {i+1}/5: {elapsed:.1f}s [{label}]")
        if i > 0:
            timings.append(elapsed)

    qp.unload(pipe)

    avg = sum(timings) / len(timings)
    total_images = sum(tx.CATEGORY_COUNTS.values())
    total_seconds = avg * total_images
    total_hours = total_seconds / 3600
    print(f"\nAverage: {avg:.1f}s/image (over {len(timings)} timed images, first discarded as warm-up)")
    print(f"Projected total for all {total_images} images: {total_hours:.1f} GPU-hours ({total_seconds/86400:.1f} days)")

    if num_shards:
        print(f"\nShard table ({num_shards} shards):")
        print(f"  {'shard':>6}  {'images':>8}  {'est. hours':>11}")
        base = total_images // num_shards
        remainder = total_images % num_shards
        for k in range(num_shards):
            n = base + (1 if k < remainder else 0)
            print(f"  {k:>6}  {n:>8}  {(n*avg)/3600:>11.1f}")


# --------------------------------------------------------------------------
# Main generation loop
# --------------------------------------------------------------------------
def run_generation(target_count, gen_batch_size, shard, num_shards, data_dir, dry_run):
    paths = output_paths(data_dir)
    output_dir, images_dir = paths["output_dir"], paths["images_dir"]

    write_schema_files(output_dir)
    tasks = build_full_task_list()
    total_planned = len(tasks)
    print(f"Total planned images: {total_planned}")
    for category, count in tx.CATEGORY_COUNTS.items():
        print(f"  {category:16s}: {count}")

    if shard is not None:
        if num_shards is None:
            sys.exit("--shard requires --num-shards")
        tasks = [t for t in tasks if t["index"] % num_shards == shard]
        print(f"Shard {shard}/{num_shards}: {len(tasks)} tasks assigned to this worker.")

    if dry_run:
        print("\n[DRY RUN] Not touching the GPU, disk, or CSVs beyond the schema files above.")
        print(f"[DRY RUN] {len(tasks)} tasks would be processed by this invocation "
              f"(shard={shard}, num_shards={num_shards}).")
        return

    check_disk_space(data_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    done = already_done_indices(images_dir)
    pending = [t for t in tasks if t["index"] not in done]
    already_done_count = len(tasks) - len(pending)
    print(f"Already done (this shard's scope): {already_done_count}  |  Remaining: {len(pending)}")
    if not pending:
        print("Nothing left to generate for this invocation's scope.")
        return

    to_process = pending[:target_count] if target_count is not None else pending
    print(f"This run will generate up to {len(to_process)} new images.")

    import qwen_pipeline as qp
    print("Loading Qwen-Image-2512...")
    pipe, can_batch = qp.load_pipeline()
    effective_batch = gen_batch_size or DEFAULT_GEN_BATCH_SIZE
    if not can_batch and effective_batch > 1:
        print(f"Note: requested gen_batch_size={effective_batch} but this GPU needs CPU offload -- forcing 1.")
        effective_batch = 1
    print(f"gen_batch_size for this run: {effective_batch}")

    # Per-category CSV writers, opened in append mode -- header written
    # only if the file doesn't exist yet (so this is safe across resumes).
    csv_files, csv_writers = {}, {}
    for category in tx.ALL_CATEGORIES:
        path = labels_csv_path(output_dir, category, shard)
        write_header = not path.exists()
        f = open(path, "a", newline="")
        writer = csv.DictWriter(f, fieldnames=tx.schema_columns(category))
        if write_header:
            writer.writeheader()
            f.flush()
        csv_files[category], csv_writers[category] = f, writer

    log_file = open(paths["log_path"], "a")

    generated_this_run = 0
    consecutive_failures = 0
    try:
        for batch in group_by_resolution(to_process, effective_batch):
            filenames = [item["filename"] for item in batch]
            tmp_paths = [images_dir / (fn + ".tmp") for fn in filenames]
            seeds = [item["index"] for item in batch]

            try:
                images = qp.generate(pipe, [item["task"] for item in batch], seeds=seeds,
                                      num_inference_steps=tx.NUM_INFERENCE_STEPS_FULL)
            except Exception as e:
                print(f"Batch failed (indices {[b['index'] for b in batch]}): {e}")
                for p in tmp_paths:
                    if p.exists():
                        p.unlink()
                consecutive_failures += len(batch)
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    sys.exit(f"Aborting: {consecutive_failures} consecutive failures "
                              f"(>= {MAX_CONSECUTIVE_FAILURES}). Check GPU/model state before resuming.")
                continue

            for item, filename, tmp_path, image in zip(batch, filenames, tmp_paths, images):
                image.save(tmp_path)

                category, tier, index = item["category"], item["tier"], item["index"]
                row = pb.row_for_csv(category, tier, filename, item["task"])
                csv_writers[category].writerow(row)
                csv_files[category].flush()

                log_file.write(json.dumps({
                    "index": index, "filename": filename, "category": category, "tier": tier,
                    "score": row["score"], "prompt": item["task"]["prompt"],
                    "negative_prompt": tx.NEGATIVE_PROMPT,
                    "seed": index, "num_inference_steps": tx.NUM_INFERENCE_STEPS_FULL,
                    "true_cfg_scale": tx.TRUE_CFG_SCALE, "gen_batch_size": effective_batch,
                    "shard": shard,
                }) + "\n")
                log_file.flush()

                os.replace(tmp_path, images_dir / filename)
                generated_this_run += 1
                consecutive_failures = 0
                if generated_this_run % 25 == 0 or generated_this_run == len(to_process):
                    print(f"  [{generated_this_run}/{len(to_process)}] {filename}")
    finally:
        for f in csv_files.values():
            f.close()
        log_file.close()
        qp.unload(pipe)

    still_remaining = len(pending) - generated_this_run
    print(f"\nThis run generated {generated_this_run} new images.")
    if still_remaining > 0:
        print(f"{still_remaining} still remaining in this invocation's scope. Run again to continue.")
    else:
        print("All images in this invocation's scope are generated.")


def main():
    parser = argparse.ArgumentParser(description="LookMax full 28,000-image Qwen-Image-2512 dataset run.")
    parser.add_argument("target_count", nargs="?", type=int, default=None,
                         help="Max NEW images this invocation generates before exiting. Omit to run until done.")
    parser.add_argument("gen_batch_size", nargs="?", type=int, default=None,
                         help=f"Images per GPU forward pass. Default {DEFAULT_GEN_BATCH_SIZE} -- see module docstring.")
    parser.add_argument("--shard", type=int, default=None, help="This worker's shard index (0-based).")
    parser.add_argument("--num-shards", type=int, default=None, help="Total number of shards.")
    parser.add_argument("--benchmark", action="store_true", help="Time 5 images, project total GPU-hours, exit.")
    parser.add_argument("--dry-run", action="store_true", help="Plan the run and write schema files, touch nothing else.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help=f"Large-disk root (default {DEFAULT_DATA_DIR}).")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(args.data_dir, args.num_shards)
        return

    run_generation(args.target_count, args.gen_batch_size, args.shard, args.num_shards, args.data_dir, args.dry_run)


if __name__ == "__main__":
    main()
