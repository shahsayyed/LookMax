import os
# Force HuggingFace to use high-performance transfer internally
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

import torch
from diffusers import FluxPipeline
from pathlib import Path

# ==========================================
# MODEL COMPARISON TEST SCRIPT
# Runs the 6 cross-model comparison prompts on FLUX.1 [dev] so the
# output can be lined up side-by-side against manual generations from
# Nano Banana Pro (Google AI Studio) and GPT Image 2 (ChatGPT).
# ==========================================
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Please set HF_TOKEN environment variable with your HuggingFace token.")

OUTPUT_DIR = Path("comparison_images")
OUTPUT_DIR.mkdir(exist_ok=True)

# Prefix every file with "flux_" so it's easy to sort next to the
# "google_" / "gpt_" images you generate manually for the same prompt.
COMPARISON_PROMPTS = {
    "flux_01_men_grooming_flaw.png": "A front-facing head-and-shoulders portrait, perfectly centered face, straight-on eye-level camera angle, looking directly at the camera, bright even studio lighting, of a 34-year-old Caucasian man, heavy set, round face, double chin, severely overgrown greasy unwashed hair, patchy unmaintained neckbeard stubble, dry flaky skin, unkempt bushy unibrow. Photorealistic, ultra detailed, 85mm lens.",

    "flux_02_women_grooming_flaw.png": "A front-facing head-and-shoulders portrait, perfectly centered face, straight-on eye-level camera angle, looking directly at the camera, bright even studio lighting, of a 26-year-old East Asian woman, athletic, heart-shaped face, greasy unwashed flat hair, smeared running mascara, dry chapped lips, visibly unkempt overgrown eyebrows. Photorealistic, ultra detailed, 85mm lens.",

    "flux_03_men_outfit_flaw.png": "A front-facing full-body portrait, perfectly centered, standing straight, straight-on eye-level camera angle, looking directly at the camera, head to toe visible, bright even studio lighting, of a 22-year-old South Asian man, lean build, oval face, wearing a heavily wrinkled stained oversized t-shirt, baggy ill-fitting sweatpants sagging at the waist, mismatched scuffed sneakers with untied laces. Photorealistic, ultra detailed, 85mm lens.",

    "flux_04_women_outfit_polished.png": "A front-facing full-body portrait, perfectly centered, standing straight, straight-on eye-level camera angle, looking directly at the camera, head to toe visible, bright even studio lighting, of a 29-year-old Black woman, curvy, oval face, wearing a perfectly fitted crisp white t-shirt tucked into tailored high-waisted jeans, sleek fitted bomber jacket, colors in tasteful harmony, crisp clean white sneakers. Photorealistic, ultra detailed, 85mm lens.",

    "flux_05_men_grooming_polished.png": "A front-facing head-and-shoulders portrait, perfectly centered face, straight-on eye-level camera angle, looking directly at the camera, bright even studio lighting, of a 31-year-old Hispanic man, athletic, square jawline, meticulously styled hair with high volume and visible styling pomade, extremely sharp razor-edge beard lineup, deeply hydrated glowing skin, groomed full eyebrows. Photorealistic, ultra detailed, 85mm lens.",

    "flux_06_women_grooming_average.png": "A front-facing head-and-shoulders portrait, perfectly centered face, straight-on eye-level camera angle, looking directly at the camera, bright even studio lighting, of a 40-year-old Middle Eastern woman, plus-size, round face, basic unstyled hair air-dried with no product, no makeup, natural bare skin, neatly trimmed natural eyebrows. Photorealistic, ultra detailed, 85mm lens.",
}

if __name__ == "__main__":
    print("Loading FLUX.1 [dev]... This takes a moment.")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()

    print(f"Generating {len(COMPARISON_PROMPTS)} comparison images...")

    for filename, prompt in COMPARISON_PROMPTS.items():
        print(f"Generating: {filename}...")
        image = pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=28,
            max_sequence_length=512
        ).images[0]

        filepath = OUTPUT_DIR / filename
        image.save(filepath)
        print(f"Saved to {filepath}")

    print(f"✅ Done! Check the '{OUTPUT_DIR}' folder. Generate the same 6 prompts manually")
    print("   on Nano Banana Pro (Google AI Studio) and GPT Image 2 (ChatGPT), name them")
    print("   google_01...06 / gpt_01...06 to match, then compare side-by-side.")
