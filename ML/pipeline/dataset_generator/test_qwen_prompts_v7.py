import os
import shutil
import sys
from pathlib import Path

# See qwen_taxonomy_v7.py's module docstring and test_qwen_variations.py's
# comment history for why this is hard-pinned rather than left to the shell.
DATA_DIR = Path("/data")
MIN_FREE_GB = 60
DATA_DIR.mkdir(parents=True, exist_ok=True)
_free_gb = shutil.disk_usage(DATA_DIR).free / (1024 ** 3)
if _free_gb < MIN_FREE_GB:
    sys.exit(f"!! Only {_free_gb:.1f}GB free on {DATA_DIR} -- need at least {MIN_FREE_GB}GB. Run 'df -h' and check /data is your large disk.")

os.environ["HF_HOME"] = str(DATA_DIR / "huggingface_cache")
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

import torch
from qwen_pipeline_utils import load_qwen_pipeline

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ==========================================
# TAXONOMY v7 -- QUICK SANITY CHECK (6 hand-written prompts)
#
# Standalone, hardcoded -- NOT built from qwen_taxonomy_v7.py's combinatorial
# system, so you can eyeball the new writing style fast before running the
# full 64-image variation test. These 6 were picked specifically to prove
# out the two problems found in the v6 review:
#   1. Flaw severity gradient (prompts 1 vs 2: same person, "severe" vs
#      "mild" neglect -- should look like TWO DIFFERENT levels of bad, not
#      the same maximally-distressed look repeated) -- and both should look
#      like a realistic bad day, not a costume.
#   2. Polished outfit diversity (prompts 5 vs 6: polished with NO jacket
#      vs polished WITH a jacket -- proves both archetypes render well, so
#      "polished" isn't visually anchored to "wearing outerwear").
# Concrete color pairs are used instead of "mismatched" to stop the model
# from inventing a patchwork effect the way it did in the v6 review.
# ==========================================
OUTPUT_DIR = Path("test_qwen_prompts_v7_out")
OUTPUT_DIR.mkdir(exist_ok=True)

TEST_PROMPTS = {
    "01_men_grooming_flaw_severe.png":
        "A front-facing head-and-shoulders portrait, centered face, straight-on eye-level angle, looking at the camera, "
        "bright even studio lighting of a 34-year-old Caucasian man, heavy set, round face, double chin. They have "
        "visibly greasy, unwashed hair with strands clumped together and a strong oily sheen catching the light, "
        "flattened against the scalp, patchy uneven stubble growing in random clumps on the jaw with bare visible skin "
        "gaps between patches, dry visibly flaking skin with small rough patches on the forehead, and thick unshaped "
        "eyebrows that grow together above the nose. Photorealistic, ultra detailed, tack-sharp crisp focus throughout, 85mm lens.",

    "02_men_grooming_flaw_mild.png":
        "A front-facing head-and-shoulders portrait, centered face, straight-on eye-level angle, looking at the camera, "
        "bright even studio lighting of a 34-year-old Caucasian man, heavy set, round face, double chin. They have "
        "flat, unbrushed hair with no styling, a little overgrown past a normal haircut length, about a week of "
        "unshaven growth that's uneven in length with no defined edge, noticeable dark circles under the eyes and a "
        "dull tired-looking complexion, and a few stray long hairs in otherwise ordinary eyebrows. Photorealistic, "
        "ultra detailed, tack-sharp crisp focus throughout, 85mm lens.",

    "03_women_grooming_average.png":
        "A front-facing head-and-shoulders portrait, centered face, straight-on eye-level angle, looking at the camera, "
        "bright even studio lighting of a 41-year-old Middle Eastern woman, plus size, round face. They have hair "
        "pulled back into a plain low ponytail with no product, no makeup with a bare face, ordinary skin with minor "
        "visible texture that isn't obviously cared for but isn't neglected either, and natural unshaped eyebrows. "
        "Photorealistic, ultra detailed, tack-sharp crisp focus throughout, 85mm lens.",

    "04_men_outfit_flaw.png":
        "A front-facing full-body portrait, centered, standing straight, straight-on eye-level angle, looking at the "
        "camera, head to toe visible, bright even studio lighting of a 23-year-old South Asian man, lean build. He is "
        "wearing a mustard-yellow t-shirt with maroon joggers, scuffed dirty sneakers with the laces untied. The fit "
        "is the t-shirt hanging noticeably loose off the shoulders and bunching at the waist, several sizes too "
        "large. The fabric is deeply wrinkled with visible crease lines across the chest and stomach, plus a "
        "coin-sized dried food stain near the hem. Body stays their natural lean figure, not slimmer or heavier than "
        "described. Photorealistic, ultra detailed, tack-sharp crisp focus throughout, 85mm lens.",

    "05_men_outfit_polished_no_jacket.png":
        "A front-facing full-body portrait, centered, standing straight, straight-on eye-level angle, looking at the "
        "camera, head to toe visible, bright even studio lighting of a 29-year-old Hispanic man, athletic build. He "
        "is wearing a crisp fitted crew-neck t-shirt in a black and olive color combination -- no jacket, just the "
        "shirt -- and dark slim-tapered jeans, clean white minimalist sneakers. The fit is tailored to the body, "
        "following the shoulder line and ending precisely at the hip. The fabric is crisp, freshly pressed with no "
        "visible wrinkles. Body stays their natural athletic figure, not slimmer or heavier than described. "
        "Photorealistic, ultra detailed, tack-sharp crisp focus throughout, 85mm lens.",

    "06_women_outfit_polished_with_jacket.png":
        "A front-facing full-body portrait, centered, standing straight, straight-on eye-level angle, looking at the "
        "camera, head to toe visible, bright even studio lighting of a 27-year-old Black woman, curvy build. She is "
        "wearing a fitted t-shirt under a cropped tailored blazer, in a white and beige color combination, and "
        "high-waisted straight-leg trousers, clean minimalist white sneakers. The fit is tailored to the body with "
        "clean flattering proportions. The fabric is crisp, freshly pressed with no visible wrinkles. Body stays "
        "their natural curvy figure, not slimmer or heavier than described. Photorealistic, ultra detailed, "
        "tack-sharp crisp focus throughout, 85mm lens.",
}

if __name__ == "__main__":
    print("Loading Qwen/Qwen-Image-2512...")
    pipe, _can_batch = load_qwen_pipeline()  # this script always runs one prompt at a time regardless

    for filename, prompt in TEST_PROMPTS.items():
        print(f"Generating: {filename}...")
        image = pipe(
            prompt=prompt,
            negative_prompt="blurry, low quality, deformed, extra limbs, watermark, text artifacts",
            height=1024,
            width=1024,
            num_inference_steps=28,
            true_cfg_scale=4.0,
            generator=torch.Generator("cpu").manual_seed(42)
        ).images[0]
        image.save(OUTPUT_DIR / filename)

    # Not required for GPU memory to be freed -- that happens automatically
    # when this process exits, regardless. This just makes it happen the
    # instant we're done rather than whenever Python's GC gets around to it.
    del pipe
    torch.cuda.empty_cache()

    print(f"Done. Check '{OUTPUT_DIR}'. Compare 01 vs 02 for the severity gradient, and 05 vs 06 for polished-outfit diversity.")
