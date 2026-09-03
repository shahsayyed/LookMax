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
import subprocess
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
def run_benchmark(data_dir, num_shards=None, batch_size=1, compare_modes=False, device=None):
    import qwen_pipeline as qp

    tasks = build_full_task_list()
    if compare_modes:
        print(f"\n{'='*88}\nBENCHMARK: COMPARING MODES ON THIS HARDWARE\n{'='*88}")
        print("Evaluating throughput across Sequential, Batched, and Concurrent Multi-Worker modes.")
        sample = [tasks[0], tasks[1], tasks[7000], tasks[7001]]  # 2 grooming square, 2 outfit portrait

        # Warmup
        print("Running warm-up forward pass...")
        pipe, can_batch = qp.load_pipeline(device=device)
        qp.generate(pipe, [sample[0]["task"]], seeds=[sample[0]["index"]], num_inference_steps=tx.NUM_INFERENCE_STEPS_FULL)
        qp.unload(pipe)

        results = []

        # Mode 1: Sequential (1 worker, batch=1)
        print("\n[Mode 1] Testing Sequential Generation (workers=1, batch=1)...")
        pipe, can_batch = qp.load_pipeline(device=device)
        t0 = time.time()
        for item in sample:
            qp.generate(pipe, [item["task"]], seeds=[item["index"]], num_inference_steps=tx.NUM_INFERENCE_STEPS_FULL)
        t_seq = time.time() - t0
        qp.unload(pipe)
        results.append(("Sequential (Workers=1, Batch=1)", 1, 1, t_seq))
        print(f"  -> {len(sample)} images in {t_seq:.1f}s ({t_seq / len(sample):.2f}s/image)")

        # Mode 2: Batched (1 worker, batch=2)
        if can_batch:
            print("\n[Mode 2] Testing Batched Generation (workers=1, batch=2)...")
            pipe, can_batch = qp.load_pipeline(device=device)
            t0 = time.time()
            for batch in group_by_resolution(sample, 2):
                qp.generate(pipe, [item["task"] for item in batch], seeds=[item["index"] for item in batch],
                            num_inference_steps=tx.NUM_INFERENCE_STEPS_FULL)
            t_batch = time.time() - t0
            qp.unload(pipe)
            results.append(("Batched (Workers=1, Batch=2)", 1, 2, t_batch))
            print(f"  -> {len(sample)} images in {t_batch:.1f}s ({t_batch / len(sample):.2f}s/image)")
        else:
            print("\n[Mode 2] Batched generation skipped (VRAM offload mode forces batch=1).")

        # Mode 3: Concurrent Independent Workers (workers=2, batch=1)
        print("\n[Mode 3] Testing Concurrent Independent Workers (workers=2, batch=1)...")
        script_path = Path(__file__).resolve()
        sub_dir = Path(data_dir) / "benchmark_multi_worker_tmp"
        sub_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        cmd_p0 = [
            sys.executable, str(script_path), "2", "1", "--shard", "0", "--num-shards", "2",
            "--data-dir", str(sub_dir)
        ]
        cmd_p1 = [
            sys.executable, str(script_path), "2", "1", "--shard", "1", "--num-shards", "2",
            "--data-dir", str(sub_dir)
        ]
        if device:
            cmd_p0.extend(["--device", device])
            cmd_p1.extend(["--device", device])
        p0 = subprocess.Popen(cmd_p0)
        p1 = subprocess.Popen(cmd_p1)
        p0.wait()
        p1.wait()
        t_concurrent = time.time() - t0
        shutil.rmtree(sub_dir, ignore_errors=True)
        results.append(("Concurrent (Workers=2, Batch=1)", 2, 1, t_concurrent))
        print(f"  -> {len(sample)} images in {t_concurrent:.1f}s ({t_concurrent / len(sample):.2f}s/image)")

        total_production_images = sum(tx.CATEGORY_COUNTS.values())
        print(f"\n{'='*88}\nBENCHMARK RESULTS & PROJECTIONS (Full 28,000-Image Run)\n{'='*88}")
        print(f"{'Mode':<34} | {'Workers':>7} | {'Batch':>5} | {'Sec/Image':>9} | {'Img/Min':>7} | {'Full Run Est.':>14}")
        print("-" * 88)
        base_sec = results[0][3] / len(sample)
        for name, workers, b_size, total_t in results:
            sec_img = total_t / len(sample)
            img_min = 60.0 / sec_img
            full_hrs = (sec_img * total_production_images) / 3600.0
            speedup = base_sec / sec_img
            speedup_str = f" ({speedup:.2f}x)" if speedup != 1.0 else ""
            print(f"{name:<34} | {workers:>7} | {b_size:>5} | {sec_img:>8.2f}s | {img_min:>7.1f} | {full_hrs:>10.1f}h{speedup_str}")
        print("-" * 88)
        fastest = min(results, key=lambda x: x[3])
        print(f"Recommendation: Fastest configuration on this hardware is '{fastest[0]}' ({fastest[3] / len(sample):.2f}s/image).\n")
        return

    # Standard benchmark
    sample = tasks[:5]
    if len(sample) < 5:
        sys.exit("Not enough tasks to benchmark (need at least 5).")

    print(f"Loading pipeline for benchmark on {device or 'default device'} (batch_size={batch_size})...")
    pipe, can_batch = qp.load_pipeline(device=device)
    effective_batch = batch_size if can_batch else 1

    batches = list(group_by_resolution(sample, effective_batch))
    timings = []
    for i, batch in enumerate(batches):
        t0 = time.time()
        qp.generate(pipe, [item["task"] for item in batch], seeds=[item["index"] for item in batch],
                    num_inference_steps=tx.NUM_INFERENCE_STEPS_FULL)
        elapsed = time.time() - t0
        label = "warm-up (discarded)" if i == 0 else "timed"
        print(f"  batch {i+1}/{len(batches)} ({len(batch)} image(s)): {elapsed:.1f}s [{label}]")
        if i > 0:
            timings.append(elapsed / len(batch))

    qp.unload(pipe)

    avg = sum(timings) / len(timings) if timings else 0
    total_images = sum(tx.CATEGORY_COUNTS.values())
    total_seconds = avg * total_images
    total_hours = total_seconds / 3600
    print(f"\nAverage: {avg:.1f}s/image (over {len(timings)} timed batches, first discarded as warm-up)")
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
# Multi-worker runner (orchestrates concurrent workers locally)
# --------------------------------------------------------------------------
def run_multi_worker(num_workers, gen_batch_size, target_count, data_dir, device=None):
    paths = output_paths(data_dir)
    output_dir = paths["output_dir"]
    check_disk_space(data_dir)
    write_schema_files(output_dir)

    print(f"\n{'='*88}\nCONCURRENT MULTI-WORKER RUN ({num_workers} workers, batch_size={gen_batch_size})\n{'='*88}\n")
    print(f"Spawning {num_workers} independent worker processes simultaneously on the GPU...")

    script_path = Path(__file__).resolve()
    processes = []
    t_start = time.time()

    per_worker_target = None
    if target_count is not None:
        per_worker_target = (target_count + num_workers - 1) // num_workers

    for k in range(num_workers):
        cmd = [
            sys.executable, str(script_path),
            "--shard", str(k),
            "--num-shards", str(num_workers),
            "--batch-size", str(gen_batch_size),
            "--data-dir", str(data_dir),
        ]
        if per_worker_target is not None:
            cmd.extend(["--target-count", str(per_worker_target)])
        if device:
            cmd.extend(["--device", device])
        print(f"Launching Worker {k}: {' '.join(cmd)}")
        p = subprocess.Popen(cmd)
        processes.append((k, p))

    failed = False
    for k, p in processes:
        ret = p.wait()
        if ret != 0:
            print(f"Worker {k} failed with exit code {ret}!")
            failed = True

    if failed:
        sys.exit("One or more worker processes failed. Check logs above before resuming.")

    total_elapsed = time.time() - t_start
    print(f"\nAll {num_workers} workers completed in {total_elapsed:.1f}s ({total_elapsed/3600:.2f} hours).")
    print("Auto-merging per-shard label CSVs into per-category CSVs...")

    import merge_shards
    for category in tx.ALL_CATEGORIES:
        merge_shards.merge_category(output_dir, category)

    print(f"\nMulti-worker run complete and merged successfully.")


# --------------------------------------------------------------------------
# Main generation loop
# --------------------------------------------------------------------------
def run_generation(target_count, gen_batch_size, shard, num_shards, data_dir, dry_run, device=None):
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
    print(f"Loading Qwen-Image-2512 on {device or 'default device'}...")
    pipe, can_batch = qp.load_pipeline(device=device)
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
    worker_tag = f"[Shard {shard}/{num_shards}] " if shard is not None else ""

    try:
        for batch in group_by_resolution(to_process, effective_batch):
            filenames = [item["filename"] for item in batch]
            tmp_paths = [images_dir / (fn + ".tmp") for fn in filenames]
            seeds = [item["index"] for item in batch]

            t0 = time.time()
            try:
                images = qp.generate(pipe, [item["task"] for item in batch], seeds=seeds,
                                      num_inference_steps=tx.NUM_INFERENCE_STEPS_FULL)
            except Exception as e:
                print(f"{worker_tag}Batch failed (indices {[b['index'] for b in batch]}): {e}")
                for p in tmp_paths:
                    if p.exists():
                        p.unlink()
                consecutive_failures += len(batch)
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    sys.exit(f"{worker_tag}Aborting: {consecutive_failures} consecutive failures "
                              f"(>= {MAX_CONSECUTIVE_FAILURES}). Check GPU/model state before resuming.")
                continue

            elapsed = time.time() - t0

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
                    print(f"  {worker_tag}[{generated_this_run}/{len(to_process)}] {filename} "
                          f"({len(batch)} image(s) in {elapsed:.1f}s, {elapsed/len(batch):.2f}s/img)")
    finally:
        for f in csv_files.values():
            f.close()
        log_file.close()
        qp.unload(pipe)

    still_remaining = len(pending) - generated_this_run
    print(f"\n{worker_tag}This run generated {generated_this_run} new images.")
    if still_remaining > 0:
        print(f"{still_remaining} still remaining in this invocation's scope. Run again to continue.")
    else:
        print("All images in this invocation's scope are generated.")


def main():
    parser = argparse.ArgumentParser(description="LookMax full 28,000-image Qwen-Image-2512 dataset run.")
    parser.add_argument("pos_target_count", nargs="?", type=int, default=None,
                         help="Max NEW images this invocation generates before exiting (positional).")
    parser.add_argument("pos_gen_batch_size", nargs="?", type=int, default=None,
                         help=f"Images per GPU forward pass (positional, default {DEFAULT_GEN_BATCH_SIZE}).")
    parser.add_argument("--target-count", type=int, default=None, help="Max NEW images to generate.")
    parser.add_argument("--batch-size", "--gen-batch-size", type=int, default=None,
                        help=f"Images per GPU forward pass (default {DEFAULT_GEN_BATCH_SIZE}).")
    parser.add_argument("--num-workers", "--workers", type=int, default=1,
                        help="Number of concurrent worker processes to run on the GPU (default 1).")
    parser.add_argument("--shard", type=int, default=None, help="This worker's shard index (0-based).")
    parser.add_argument("--num-shards", type=int, default=None, help="Total number of shards.")
    parser.add_argument("--device", default=None, help="Target PyTorch device, e.g. 'cuda:0'.")
    parser.add_argument("--benchmark", action="store_true", help="Time images, project total GPU-hours, exit.")
    parser.add_argument("--compare-modes", action="store_true",
                        help="Benchmark and compare single vs batched vs concurrent multi-worker modes.")
    parser.add_argument("--dry-run", action="store_true", help="Plan the run and write schema files, touch nothing else.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help=f"Large-disk root (default {DEFAULT_DATA_DIR}).")
    args = parser.parse_args()

    target_count = args.target_count if args.target_count is not None else args.pos_target_count
    gen_batch_size = args.batch_size if args.batch_size is not None else (
        args.pos_gen_batch_size if args.pos_gen_batch_size is not None else DEFAULT_GEN_BATCH_SIZE
    )

    if args.compare_modes or args.benchmark:
        run_benchmark(args.data_dir, args.num_shards, batch_size=gen_batch_size,
                      compare_modes=args.compare_modes, device=args.device)
        return

    if args.num_workers > 1 and args.shard is None and not args.dry_run:
        run_multi_worker(args.num_workers, gen_batch_size, target_count, args.data_dir, device=args.device)
        return

    run_generation(target_count, gen_batch_size, args.shard, args.num_shards, args.data_dir, args.dry_run, device=args.device)


if __name__ == "__main__":
    main()
