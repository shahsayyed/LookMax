import os
import random
import shutil
import sys
from pathlib import Path

DATA_DIR = Path("/data")
MIN_FREE_GB = 60
DATA_DIR.mkdir(parents=True, exist_ok=True)
_free_gb = shutil.disk_usage(DATA_DIR).free / (1024 ** 3)
if _free_gb < MIN_FREE_GB:
    sys.exit(f"!! Only {_free_gb:.1f}GB free on {DATA_DIR} -- need at least {MIN_FREE_GB}GB. Run 'df -h' and check /data is your large disk.")

os.environ["HF_HOME"] = str(DATA_DIR / "huggingface_cache")
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

import torch
import qwen_taxonomy_v7 as tax
from qwen_pipeline_utils import load_qwen_pipeline, chunked

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ==========================================
# TAXONOMY v7 -- FULL VARIATION TEST (64 images)
# Direct v7 counterpart to test_qwen_variations.py's 48-image v6 test.
# 4 identities x 4 tiers (flaw_severe, flaw_mild, average, polished) x
# {Grooming, Outfit} x {men, women} = 4*4*2*2 = 64.
#
# What changed vs v6 (see qwen_taxonomy_v7.py's docstring for the full
# rationale): each identity now gets its OWN 2-level flaw severity instead
# of one maximally-bad description, and "polished" draws from a pool of
# distinct outfit archetypes (only some involving a jacket) instead of one
# fixed jacket-based description. Use this to check BOTH things at once:
# (a) does flaw_severe visibly read as worse than flaw_mild for the same
# person, and (b) across the 8 polished-outfit images in this batch, do you
# see more than one archetype (not jacket every time)?
#
# PARALLEL BATCHING: also useful as a cheap dry run of generate_qwen_
# dataset.py's batching before committing to it on the full 24,000-image
# run -- same load_qwen_pipeline() GPU auto-detection and the same chunked
# multi-prompt pipe() call, just on 64 images instead of 24,000, so you can
# confirm a given gen_batch_size works (and see roughly how much faster it
# is) in a few minutes instead of finding out partway through a huge run.
#
#   python3 test_qwen_variations_v7.py            -> batch size auto-picked
#   python3 test_qwen_variations_v7.py 4            -> force 4 images/pass
# ==========================================
OUTPUT_DIR = Path("test_variations_qwen_v7")
OUTPUT_DIR.mkdir(exist_ok=True)

TASKS = []
for gender, identities in [("man", tax.MEN_IDENTITIES), ("woman", tax.WOMEN_IDENTITIES)]:
    for ident_idx, identity in enumerate(identities):
        ident_str = f"{identity['age']}_{identity['ethnicity']}_{gender}".replace(" ", "").replace("-year-old", "")
        # One rng per identity, seeded deterministically -- reproducible
        # across reruns, but each identity draws its own sequence so
        # different identities land on different polished archetypes /
        # clash pairs rather than all converging on the same choice.
        rng = random.Random(hash((gender, ident_str)) % 1_000_000)
        for cat, builder in [("Grooming", tax.build_grooming_prompt), ("Outfit", tax.build_outfit_prompt)]:
            for tier in tax.GROOMING_TIERS:
                prompt, score, labels = builder(gender, identity, tier, rng)
                TASKS.append({
                    "filename": f"qwenv7_{len(TASKS)+1:03d}_{gender}_{cat}_{tier}_{ident_str}.png",
                    "prompt": prompt,
                    "score": score,
                    "labels": labels,
                })

DEFAULT_GEN_BATCH_SIZE = 1  # measured to give no throughput benefit for this model on an RTX PRO 6000 -- see generate_qwen_dataset.py's comment on this constant

if __name__ == "__main__":
    gen_batch_override = None
    if len(sys.argv) > 1:
        try:
            gen_batch_override = int(sys.argv[1])
        except ValueError:
            sys.exit(f"Usage: python3 {sys.argv[0]} [gen_batch_size]")

    print(f"Generating {len(TASKS)} test images with Qwen-Image-2512 (taxonomy v7)...")
    print("Loading Qwen/Qwen-Image-2512...")
    pipe, can_batch = load_qwen_pipeline()

    gen_batch_size = gen_batch_override or (DEFAULT_GEN_BATCH_SIZE if can_batch else 1)
    if not can_batch and gen_batch_size > 1:
        print(f"Note: requested gen_batch_size={gen_batch_size} but this GPU needs CPU offload -- forcing 1.")
        gen_batch_size = 1
    print(f"Parallel batch size for this run: {gen_batch_size} image(s) per GPU forward pass.")

    done = 0
    for batch in chunked(TASKS, gen_batch_size):
        names = ", ".join(t["filename"] for t in batch)
        print(f"[{done+1}-{done+len(batch)}/{len(TASKS)}] Generating: {names}...")

        result = pipe(
            prompt=[t["prompt"] for t in batch],
            negative_prompt="blurry, low quality, deformed, extra limbs, watermark, text artifacts",
            height=1024,
            width=1024,
            num_inference_steps=28,
            true_cfg_scale=4.0,
            generator=[torch.Generator("cpu").manual_seed(done + i) for i in range(len(batch))],
        )
        for task, image in zip(batch, result.images):
            image.save(OUTPUT_DIR / task["filename"])
        done += len(batch)

    # See the matching comment in test_qwen_prompts_v7.py -- not required,
    # just makes the freed memory visible in nvidia-smi immediately.
    del pipe
    torch.cuda.empty_cache()

    print(f"Done. Check '{OUTPUT_DIR}'.")
    print("Things to check: (1) does flaw_severe vs flaw_mild look like two different severities for the same")
    print("identity, not a repeat of the same look; (2) across the 8 polished-outfit images, is there more than")
    print("one archetype (not a jacket every time); (3) do the flaw images look like a bad day, not a costume.")
