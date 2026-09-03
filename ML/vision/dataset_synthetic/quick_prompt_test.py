"""
quick_prompt_test.py -- fastest possible "does Qwen even work on this box"
check: 6 hand-written, hardcoded prompts, generated one at a time. NOT built
from taxonomy.py / prompt_builder.py's sampling -- deliberately standalone,
so a bug in the taxonomy/prompt-builder layer can never mask (or be masked
by) a base-model/environment problem, and so these exact prompts stay
stable for manually comparing Qwen against another model (Nano Banana,
ChatGPT, etc.) the way the archived
ML/archive/dataset_generator_v7/test_qwen_prompts_v7.py was used for the
comparison that picked Qwen over FLUX/Nano Banana in the first place.

Run this FIRST on any new GPU box -- right after install.sh and the
no-GPU checks (smoke_test.py --dry-run, validation_sweep.py
--coverage-only) -- before spending any more GPU time on the
taxonomy-driven checks below it (smoke_test.py --per-tier,
variation_test.py). If model loading or generation is broken, this is the
cheapest, fastest place to find out (6 images, a few minutes), not partway
through a 64-image variation test or the 28,000-image full run.

What the 6 prompts check (same intent as the archived v7 script, rewritten
in the current v8 clause style so this test exercises the real production
prompt shape, not a stale one):
  01/02 - grooming flaw_severe vs flaw_mild, SAME identity: proves the
          severity gradient reads as two different levels of "bad", not
          one maximally-distressed look repeated (the v6->v7 fix).
  03    - grooming average: the deliberately boring, forgettable middle --
          neither neglected nor groomed, the tier most likely to drift
          toward one end by accident.
  04    - outfit flaw: wrinkled/stained/ill-fitting, meant to read as a
          realistic bad day, not a costume.
  05/06 - outfit polished WITHOUT a jacket vs WITH a jacket, different
          identities: proves "polished" isn't visually anchored to
          outerwear (the jacket-monoculture bug caught by eye in the
          48-prompt review -- see ML/README.md).

Usage:
    python3 quick_prompt_test.py
    python3 quick_prompt_test.py --output-dir /data/quick_prompt_test_output
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taxonomy as tx
import prompt_builder as pb

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "quick_prompt_test_output"
MIN_FREE_GB = 70  # just the ~58GB model cache + headroom -- this script only generates 6 images


TEST_TASKS = {
    "01_men_grooming_flaw_severe.png": {
        "resolution": tx.GROOMING_RESOLUTION,
        "prompt": (
            "A head-and-shoulders portrait photograph of a 34-year-old Caucasian man, heavy set, "
            "round face, double chin, facing the camera directly, looking at the camera.\n"
            "Hair: visibly greasy and unwashed, strands clumped together with a strong oily sheen, "
            "flattened against the scalp.\n"
            "Facial hair: patchy uneven stubble growing in random clumps on the jaw, with bare "
            "visible skin gaps between patches.\n"
            "Skin: dry and visibly flaking, small rough patches on the forehead.\n"
            "Eyebrows: thick and unshaped, growing together above the nose.\n"
            "Setting: plain neutral-colored wall, bright even studio lighting, centered in frame, "
            "straight-on eye-level angle.\n"
            "Photorealistic candid photograph, natural skin texture, sharp focus, 85mm lens, "
            "head and shoulders in frame."
        ),
    },
    "02_men_grooming_flaw_mild.png": {
        "resolution": tx.GROOMING_RESOLUTION,
        "prompt": (
            "A head-and-shoulders portrait photograph of a 34-year-old Caucasian man, heavy set, "
            "round face, double chin, facing the camera directly, looking at the camera.\n"
            "Hair: flat and unbrushed with no styling, a little overgrown past a normal haircut "
            "length.\n"
            "Facial hair: about a week of unshaven growth, uneven in length with no defined edge.\n"
            "Skin: noticeable dark circles under the eyes, a dull tired-looking complexion.\n"
            "Eyebrows: a few stray long hairs in otherwise ordinary eyebrows.\n"
            "Setting: plain neutral-colored wall, bright even studio lighting, centered in frame, "
            "straight-on eye-level angle.\n"
            "Photorealistic candid photograph, natural skin texture, sharp focus, 85mm lens, "
            "head and shoulders in frame."
        ),
    },
    "03_women_grooming_average.png": {
        "resolution": tx.GROOMING_RESOLUTION,
        "prompt": (
            "A head-and-shoulders portrait photograph of a 41-year-old Middle Eastern woman, plus "
            "size, round face, facing the camera directly, looking at the camera.\n"
            "Hair: pulled back into a plain low ponytail with no product.\n"
            "Makeup: none, a bare face.\n"
            "Skin: ordinary, with minor visible texture that isn't obviously cared for but isn't "
            "neglected either.\n"
            "Eyebrows: natural and unshaped, otherwise unremarkable.\n"
            "Setting: plain neutral-colored wall, bright even studio lighting, centered in frame, "
            "straight-on eye-level angle.\n"
            "Photorealistic candid photograph, natural skin texture, sharp focus, 85mm lens, "
            "head and shoulders in frame."
        ),
    },
    "04_men_outfit_flaw.png": {
        "resolution": tx.OUTFIT_RESOLUTION,
        "prompt": (
            f"A full-body photograph of a 23-year-old South Asian man, lean build, "
            f"{pb.BODY_LOCK.format(build='lean')}, with short black hair, standing facing the "
            f"camera with the whole body from head to shoes visible.\n"
            "Upper body: wearing a mustard-yellow graphic t-shirt, hanging noticeably loose off the "
            "shoulders and bunching at the waist, several sizes too large, deeply wrinkled with "
            "visible crease lines across the chest and stomach, plus a coin-sized dried food stain "
            "near the hem.\n"
            "Lower body: wearing maroon joggers, ill-fitting and rumpled.\n"
            "Footwear: white sneakers, scuffed and dirty, laces untied.\n"
            "Overall styling: the whole outfit looking careless and thrown together.\n"
            "Setting: plain neutral-colored wall, bright even studio lighting, centered in frame, "
            "straight-on eye-level angle.\n"
            "Photorealistic candid photograph, natural skin texture, sharp focus, 85mm lens, "
            "full body in frame."
        ),
    },
    "05_men_outfit_polished_no_jacket.png": {
        "resolution": tx.OUTFIT_RESOLUTION,
        "prompt": (
            f"A full-body photograph of a 29-year-old Hispanic man, athletic build, "
            f"{pb.BODY_LOCK.format(build='athletic')}, with short dark hair, standing facing the "
            f"camera with the whole body from head to shoes visible.\n"
            "Upper body: wearing a crisp fitted black and olive crew-neck t-shirt, no jacket or "
            "outer layer, crisp and freshly pressed with no visible wrinkles, the fit tailored to "
            "the body with clean, flattering proportions.\n"
            "Lower body: wearing dark slim-tapered jeans, crisp and freshly pressed.\n"
            "Footwear: clean white minimalist sneakers, freshly cleaned.\n"
            "Overall styling: the whole outfit looking sharp and intentionally put together.\n"
            "Setting: plain neutral-colored wall, bright even studio lighting, centered in frame, "
            "straight-on eye-level angle.\n"
            "Photorealistic candid photograph, natural skin texture, sharp focus, 85mm lens, "
            "full body in frame."
        ),
    },
    "06_women_outfit_polished_with_jacket.png": {
        "resolution": tx.OUTFIT_RESOLUTION,
        "prompt": (
            f"A full-body photograph of a 27-year-old Black woman, curvy build, "
            f"{pb.BODY_LOCK.format(build='curvy')}, with shoulder-length hair, standing facing the "
            f"camera with the whole body from head to shoes visible.\n"
            "Upper body: wearing a fitted white and beige t-shirt, crisp and freshly pressed, the "
            "fit tailored to the body with clean, flattering proportions.\n"
            "Outer layer: wearing a beige cropped tailored blazer, crisp and well-kept.\n"
            "Lower body: wearing white high-waisted straight-leg trousers, crisp and freshly "
            "pressed.\n"
            "Footwear: clean minimalist white sneakers, freshly cleaned.\n"
            "Overall styling: the whole outfit looking sharp and intentionally put together.\n"
            "Setting: plain neutral-colored wall, bright even studio lighting, centered in frame, "
            "straight-on eye-level angle.\n"
            "Photorealistic candid photograph, natural skin texture, sharp focus, 85mm lens, "
            "full body in frame."
        ),
    },
}


def check_disk_space(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(output_dir).free / (1024 ** 3)
    if free_gb < MIN_FREE_GB:
        sys.exit(
            f"!! Only {free_gb:.1f}GB free on {output_dir} -- need at least {MIN_FREE_GB}GB for the "
            f"~58GB model download. Run 'df -h' and confirm this is your LARGE disk (on Vast.ai, "
            f"/workspace is a tiny loop device -- use /data). Pass --output-dir to point elsewhere "
            f"if needed."
        )
    print(f"Disk check OK: {free_gb:.1f}GB free on {output_dir} (need >= {MIN_FREE_GB}GB).")


def main():
    parser = argparse.ArgumentParser(
        description="Fastest sanity check: 6 hardcoded prompts, real GPU generation, no taxonomy involved."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                         help=f"Where images are written (default {DEFAULT_OUTPUT_DIR}).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    check_disk_space(output_dir)

    if not os.environ.get("HF_HOME"):
        print("NOTE: HF_HOME is not set in this shell -- the model cache may land on the wrong disk. "
              "See install.sh / PLAN.md step 1 (export HF_HOME=\"$LOOKMAX_DATA_DIR/huggingface_cache\").")

    import qwen_pipeline as qp
    print("Loading Qwen-Image-2512 (this can take a while on first run -- ~58GB download)...")
    pipe, _can_batch = qp.load_pipeline()

    for i, (filename, spec) in enumerate(TEST_TASKS.items()):
        print(f"[{i+1}/{len(TEST_TASKS)}] Generating {filename}...")
        task = {"prompt": spec["prompt"], "resolution": spec["resolution"]}
        images = qp.generate(pipe, [task], seeds=[42], num_inference_steps=tx.NUM_INFERENCE_STEPS_TEST)
        images[0].save(output_dir / filename)

    qp.unload(pipe)

    print(f"\nDone. Check '{output_dir}'.")
    print("Compare 01 vs 02 for the severity gradient (two different levels of bad, not a repeat),")
    print("and 05 vs 06 for polished-outfit diversity (not a jacket every time).")
    print("If these 6 look right, move on to smoke_test.py --per-tier and variation_test.py.")


if __name__ == "__main__":
    main()
