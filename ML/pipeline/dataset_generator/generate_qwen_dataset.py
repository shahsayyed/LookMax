import os
import csv
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path("/data")
MIN_FREE_GB = 100  # full run's own output (24k PNGs at 1024x1024) needs room alongside the ~58GB model cache
DATA_DIR.mkdir(parents=True, exist_ok=True)
_free_gb = shutil.disk_usage(DATA_DIR).free / (1024 ** 3)
if _free_gb < MIN_FREE_GB:
    sys.exit(f"!! Only {_free_gb:.1f}GB free on {DATA_DIR} -- need at least {MIN_FREE_GB}GB for the full run "
              f"(model cache + ~24,000 PNGs). Run 'df -h' and check /data is your large disk.")

os.environ["HF_HOME"] = str(DATA_DIR / "huggingface_cache")
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

import torch
from tqdm import tqdm
import qwen_taxonomy_v7 as tax
from qwen_pipeline_utils import load_qwen_pipeline, chunked

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ==========================================
# FULL DATASET GENERATION -- Qwen-Image-2512, taxonomy v7
# Run test_qwen_prompts_v7.py and test_qwen_variations_v7.py FIRST and
# review their output. This script is the 24,000-image commitment -- don't
# run it before confirming the taxonomy actually fixed what it's meant to.
#
# USAGE:
#   python3 generate_qwen_dataset.py                  -> run until all 24,000 done, auto batch size
#   python3 generate_qwen_dataset.py 500               -> generate up to 500 NEW images this invocation, then exit
#   python3 generate_qwen_dataset.py 500 4              -> same, but force 4 images per GPU forward pass
# Two DIFFERENT "batch" concepts, don't confuse them:
#   arg 1, TARGET_COUNT -- how many new images this invocation will generate
#                          before exiting (for splitting the whole run across
#                          multiple sittings/sessions).
#   arg 2, GEN_BATCH_SIZE -- how many images are generated in PARALLEL in one
#                          GPU forward pass (real throughput). Auto-picked
#                          based on detected VRAM if you don't set it -- see
#                          qwen_pipeline_utils.py. Forced to 1 automatically
#                          if this GPU needs enable_model_cpu_offload().
#
# To auto-loop TARGET_COUNT batches unattended in tmux until done:
#   while python3 generate_qwen_dataset.py 500; do :; done
#
# ATOMIC WRITES: each image is saved to a ".tmp" name first and only renamed
# to its final filename after its CSV row AND log entry are both written and
# flushed. A run interrupted mid-image (OOM, power loss, Ctrl+C) can never
# leave a half-written file that looks "done" and gets silently skipped
# forever on the next run.
#
# TWO OUTPUT FILES, different purposes:
#   labels.csv            -- training-ready: filename, category, tier, score,
#                             + binary attribute columns.
#   generation_log.jsonl   -- full provenance, one JSON object per image: the
#                             EXACT prompt text sent to the model, negative
#                             prompt, seed, inference steps, cfg scale,
#                             gen_batch_size used, and a timestamp.
#
# NUM_INFERENCE_STEPS is 40 here (vs 28 in the two smaller test scripts)
# since this is the final training data, not a quick check.
# ==========================================
OUTPUT_DIR = DATA_DIR / "qwen_dataset_output"
IMAGES_DIR = OUTPUT_DIR / "images"
CSV_PATH = OUTPUT_DIR / "labels.csv"
LOG_PATH = OUTPUT_DIR / "generation_log.jsonl"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PER_CATEGORY = 6000  # x4 categories = 24,000, matching generate_flux_dataset.py's original total
NUM_INFERENCE_STEPS = 40
TRUE_CFG_SCALE = 4.0
NEGATIVE_PROMPT = "blurry, low quality, deformed, extra limbs, watermark, text artifacts"

# Default parallel batch size WHEN the GPU can hold the full pipeline
# resident (see qwen_pipeline_utils.py). MEASURED on an RTX PRO 6000
# Blackwell (96GB): batch=1 ran ~0.64s/step/image; batch=4 ran ~0.73-0.75s/
# step/image -- i.e. batching gave NO throughput win (slightly worse, if
# anything), because a 20B-param transformer at 1024x1024 already saturates
# this GPU's compute at batch=1 -- there's no idle capacity left for
# batching to exploit, unlike lighter models. So the default here is 1
# (no batching) based on that measurement, not a cautious guess. If you're
# on different hardware, it's worth re-testing with
# test_qwen_variations_v7.py <N> first (cheap, ~64 images) before assuming
# this holds -- a GPU with a bigger gap between its compute throughput and
# this model's per-image compute need might actually benefit from batching.
DEFAULT_GEN_BATCH_SIZE = 1

# Score-band balance preserved from the original taxonomy (v2/v6): roughly
# equal thirds across Flaw(1-3) / Average(4-6) / Polished(9-10). Flaw is now
# internally split into two severities but the COMBINED flaw band still
# gets 1/3 of the total, same as before.
TIER_COUNTS = {
    "flaw_severe": TARGET_PER_CATEGORY // 6,   # 1000
    "flaw_mild": TARGET_PER_CATEGORY // 6,     # 1000
    "average": TARGET_PER_CATEGORY // 3,       # 2000
    "polished": TARGET_PER_CATEGORY // 3,      # 2000
}
assert sum(TIER_COUNTS.values()) == TARGET_PER_CATEGORY

# ==========================================
# FULL IDENTITY MATRIX (age x ethnicity x build, same axes as
# generate_flux_dataset.py) -- a face shape is randomly attached per task.
# ==========================================
AGES = ["19-year-old", "22-year-old", "28-year-old", "35-year-old", "45-year-old", "55-year-old"]
ETHNICITIES = ["Caucasian", "Black", "East Asian", "South Asian", "Hispanic", "Middle Eastern"]
MEN_BUILDS = ["slim", "athletic", "average build", "heavy set", "muscular", "very skinny"]
WOMEN_BUILDS = ["slim", "curvy", "athletic", "average build", "plus size"]

MEN_FACE_SHAPES = [
    "round face, double chin", "strong square jaw, deep set eyes", "sharp angular face, prominent nose",
    "wide flat face", "oval face, high cheekbones", "narrow face, pointed chin",
]
WOMEN_FACE_SHAPES = [
    "round soft face shape", "long narrow face, prominent cheekbones", "high cheekbones, visible age lines",
    "heart-shaped face", "oval face, defined jawline", "square jaw, full cheeks",
]


def make_identity(gender, rng):
    age = rng.choice(AGES)
    ethnicity = rng.choice(ETHNICITIES)
    build = rng.choice(MEN_BUILDS if gender == "man" else WOMEN_BUILDS)
    face = rng.choice(MEN_FACE_SHAPES if gender == "man" else WOMEN_FACE_SHAPES)
    return {"age": age, "ethnicity": ethnicity, "build": build, "face": face}


def build_tasks(rng):
    tasks = []
    for cat, gender, builder in [
        ("Men_Grooming", "man", tax.build_grooming_prompt),
        ("Women_Grooming", "woman", tax.build_grooming_prompt),
        ("Men_Outfit", "man", tax.build_outfit_prompt),
        ("Women_Outfit", "woman", tax.build_outfit_prompt),
    ]:
        for tier, count in TIER_COUNTS.items():
            for _ in range(count):
                identity = make_identity(gender, rng)
                prompt, score, labels = builder(gender, identity, tier, rng)
                tasks.append({"category": cat, "tier": tier, "prompt": prompt, "score": score, "labels": labels})
    rng.shuffle(tasks)
    return tasks


def filename_for(idx, task):
    return f"{idx:05d}_{task['category']}_{task['tier']}.png"


if __name__ == "__main__":
    target_count = None
    gen_batch_override = None
    if len(sys.argv) > 1:
        try:
            target_count = int(sys.argv[1])
        except ValueError:
            sys.exit(f"Usage: python3 {sys.argv[0]} [target_count] [gen_batch_size]")
    if len(sys.argv) > 2:
        try:
            gen_batch_override = int(sys.argv[2])
        except ValueError:
            sys.exit(f"Usage: python3 {sys.argv[0]} [target_count] [gen_batch_size]")

    # Seeded, NOT random.Random() -- must produce the exact same 24,000 tasks
    # in the exact same order every run, or "skip if file already exists"
    # resume silently breaks (see file header comment / prior debugging).
    rng = random.Random(42)
    TASKS = build_tasks(rng)

    pending_indices = [idx for idx, task in enumerate(TASKS) if not (IMAGES_DIR / filename_for(idx, task)).exists()]
    already_done = len(TASKS) - len(pending_indices)
    print(f"Total tasks: {len(TASKS)}  |  Already done: {already_done}  |  Remaining: {len(pending_indices)}")
    if not pending_indices:
        print("Nothing left to generate -- all images already exist.")
        sys.exit(0)

    to_process = pending_indices[:target_count] if target_count is not None else pending_indices
    print(f"This run will generate {len(to_process)} new images." if target_count is not None else
          "No target count given -- this run will keep going until all remaining images are done.")

    print("Loading Qwen/Qwen-Image-2512... This takes a moment.")
    pipe, can_batch = load_qwen_pipeline()

    gen_batch_size = gen_batch_override or (DEFAULT_GEN_BATCH_SIZE if can_batch else 1)
    if not can_batch and gen_batch_size > 1:
        print(f"Note: requested gen_batch_size={gen_batch_size} but this GPU needs CPU offload -- forcing 1.")
        gen_batch_size = 1
    print(f"Parallel batch size for this run: {gen_batch_size} image(s) per GPU forward pass.")

    all_label_keys = sorted({k for t in TASKS for k in t["labels"].keys()})
    csv_headers = ["filename", "category", "tier", "score"] + all_label_keys
    write_csv_header = not CSV_PATH.exists()

    csv_file = open(CSV_PATH, mode="a", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
    if write_csv_header:
        csv_writer.writeheader()
        csv_file.flush()

    log_file = open(LOG_PATH, mode="a")

    generated_this_run = 0
    progress = tqdm(total=len(to_process))

    for batch_indices in chunked(to_process, gen_batch_size):
        batch_tasks = [TASKS[i] for i in batch_indices]
        batch_filenames = [filename_for(i, TASKS[i]) for i in batch_indices]
        batch_tmp_paths = [IMAGES_DIR / (fn + ".tmp") for fn in batch_filenames]
        generators = [torch.Generator("cpu").manual_seed(i) for i in batch_indices]

        try:
            result = pipe(
                prompt=[t["prompt"] for t in batch_tasks],
                negative_prompt=NEGATIVE_PROMPT,
                height=1024,
                width=1024,
                num_inference_steps=NUM_INFERENCE_STEPS,
                true_cfg_scale=TRUE_CFG_SCALE,
                generator=generators,
            )
            images = result.images  # same order as the prompt list passed in

            for i, task, filename, tmp_path, image in zip(batch_indices, batch_tasks, batch_filenames, batch_tmp_paths, images):
                image.save(tmp_path)

                row = {"filename": filename, "category": task["category"], "tier": task["tier"], "score": task["score"]}
                for key in all_label_keys:
                    row[key] = task["labels"].get(key, 0)
                csv_writer.writerow(row)
                csv_file.flush()

                log_file.write(json.dumps({
                    "index": i,
                    "filename": filename,
                    "category": task["category"],
                    "tier": task["tier"],
                    "score": task["score"],
                    "prompt": task["prompt"],
                    "negative_prompt": NEGATIVE_PROMPT,
                    "seed": i,
                    "num_inference_steps": NUM_INFERENCE_STEPS,
                    "true_cfg_scale": TRUE_CFG_SCALE,
                    "gen_batch_size": gen_batch_size,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }) + "\n")
                log_file.flush()

                os.replace(tmp_path, IMAGES_DIR / filename)
                generated_this_run += 1
                progress.update(1)

        except Exception as e:
            print(f"Failed on batch (indices {batch_indices}): {e}")
            for tmp_path in batch_tmp_paths:
                if tmp_path.exists():
                    tmp_path.unlink()
            # Left un-generated -- resume will pick these up on the next run.

    progress.close()
    csv_file.close()
    log_file.close()

    # See test_qwen_prompts_v7.py's matching comment -- not required for
    # correctness (the driver reclaims everything when this process exits
    # regardless), just makes it visible in nvidia-smi immediately, which
    # matters more here since you'll likely check between batched runs.
    del pipe
    torch.cuda.empty_cache()

    still_remaining = len(pending_indices) - generated_this_run
    print(f"This run generated {generated_this_run} new images.")
    if still_remaining > 0:
        print(f"{still_remaining} still remaining. Run again to continue.")
    else:
        print("All images generated.")
