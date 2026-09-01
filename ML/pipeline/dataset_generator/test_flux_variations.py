import os
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
import torch
from diffusers import FluxPipeline
from pathlib import Path

# ==========================================
# AGGRESSIVE VARIATION TEST SCRIPT (48 Images)
# Taxonomy v2: Effort-based flaws, diverse outfits
# ==========================================
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Please set HF_TOKEN environment variable with your HuggingFace token.")

OUTPUT_DIR = Path("test_variations_comprehensive")
OUTPUT_DIR.mkdir(exist_ok=True)

# Diverse genetics: age, ethnicity, face shape, body type all specified
MEN_IDENTITIES = [
    "28-year-old Caucasian man, heavy set, round face, double chin",
    "45-year-old Black man, athletic, strong square jaw, deep set eyes",
    "19-year-old South Asian man, very skinny, sharp angular face, prominent nose",
    "35-year-old East Asian man, average build, wide flat face"
]

WOMEN_IDENTITIES = [
    "22-year-old Hispanic woman, plus size, round soft face shape",
    "30-year-old Caucasian woman, slim, long narrow face, prominent cheekbones",
    "50-year-old Black woman, average build, high cheekbones, visible age lines",
    "26-year-old East Asian woman, athletic, heart-shaped face"
]

# v2: Effort-based flaws only (no biological traits like acne)
MEN_GROOMING_VARS = [
    {"level": "Flaw", "desc": "severely overgrown greasy unwashed hair, patchy unmaintained neckbeard stubble, dry flaky skin, unkempt bushy unibrow, sloppy grooming"},
    {"level": "Average", "desc": "basic short haircut with no styling product, natural flat unstyled hair, standard trimmed facial hair, bare natural skin"},
    {"level": "Polished", "desc": "meticulously styled hair with high volume and visible styling pomade, extremely sharp razor-edge beard lineup, deeply hydrated glowing skin, perfectly manicured groomed eyebrows"}
]

MEN_OUTFIT_VARS = [
    {"level": "Flaw", "desc": "wearing a severely wrinkled, heavily stained and overly baggy cheap grey t-shirt, terrible unflattering fit, sloppy untucked styling, mismatched clashing colors"},
    {"level": "Average", "desc": "wearing a basic plain t-shirt and standard jeans, acceptable average fit, neutral colors, clean everyday casual"},
    {"level": "Polished", "desc": "wearing a perfectly fitted smart-casual layered streetwear outfit, crisp clean fitted t-shirt under a sleek modern jacket, pristine condition fabrics, flawlessly tailored proportions, highly stylish"}
]

WOMEN_GROOMING_VARS = [
    {"level": "Flaw", "desc": "severely tangled unbrushed ratty hair, visibly greasy unwashed roots, clumpy uneven mascara, smudged lip color, harsh unblended foundation lines, dry flaky lips"},
    {"level": "Average", "desc": "basic simple ponytail, flat unstyled hair, minimal everyday natural makeup, bare natural skin"},
    {"level": "Polished", "desc": "flawless blowout hair with high volume and shine, perfectly blended professional makeup, sharp clean eyeliner, deeply hydrated glowing skin, perfectly shaped eyebrows"}
]

WOMEN_OUTFIT_VARS = [
    {"level": "Flaw", "desc": "wearing a heavily wrinkled, cheap unfitted casual dress, terrible unflattering fit that distorts proportions, sloppy and messy silhouette, mismatched clashing colors"},
    {"level": "Average", "desc": "wearing a standard casual top and jeans, average fit, normal everyday look, neutral matching colors, clean clothes"},
    {"level": "Polished", "desc": "wearing a perfectly fitted chic casual outfit, stylish cropped jacket and crisp high-waisted trousers, pristine clean fabrics, flawlessly tailored proportions, highly stylish everyday wear"}
]

TASKS = []

for ident in MEN_IDENTITIES:
    for var in MEN_GROOMING_VARS:
        TASKS.append({"cat": "Men_Grooming", "ident": ident, "level": var["level"], "desc": var["desc"]})

for ident in MEN_IDENTITIES:
    for var in MEN_OUTFIT_VARS:
        TASKS.append({"cat": "Men_Outfit", "ident": ident, "level": var["level"], "desc": var["desc"]})

for ident in WOMEN_IDENTITIES:
    for var in WOMEN_GROOMING_VARS:
        TASKS.append({"cat": "Women_Grooming", "ident": ident, "level": var["level"], "desc": var["desc"]})

for ident in WOMEN_IDENTITIES:
    for var in WOMEN_OUTFIT_VARS:
        TASKS.append({"cat": "Women_Outfit", "ident": ident, "level": var["level"], "desc": var["desc"]})


if __name__ == "__main__":
    print(f"Generating {len(TASKS)} test images with new taxonomy v2...")
    print("Loading FLUX.1 [dev]... This takes a moment.")

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()

    for idx, task in enumerate(TASKS):
        cat = task["cat"]
        ident = task["ident"]
        level = task["level"]
        desc = task["desc"]

        if "Grooming" in cat:
            alignment = "front-facing head-and-shoulders portrait, perfectly centered face, straight-on eye-level camera angle, looking directly at the camera, bright even studio lighting"
        else:
            alignment = "front-facing full-body portrait, perfectly centered, standing straight, straight-on eye-level camera angle, looking directly at the camera, head to toe visible, bright even studio lighting"

        prompt = f"A {alignment} of a {ident}, {desc}. Photorealistic, ultra detailed, 85mm lens."

        short_ident = ident.split(',')[0].replace(' ', '').replace('-year-old', '')
        filename = f"{idx+1:03d}_{cat}_{level}_{short_ident}.png"

        print(f"[{idx+1}/{len(TASKS)}] Generating: {filename}...")

        generator = torch.Generator("cpu").manual_seed(torch.randint(0, 1000000, (1,)).item())

        image = pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=28,
            max_sequence_length=512,
            generator=generator
        ).images[0]

        image.save(OUTPUT_DIR / filename)

    print(f"✅ Taxonomy v2 testing complete! Check the '{OUTPUT_DIR}' folder.")
