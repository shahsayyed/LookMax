import os
# Force HuggingFace to use high-performance transfer internally
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

import csv
import torch
from pathlib import Path
from tqdm import tqdm
from diffusers import FluxPipeline
import random

# ==========================================
# CONFIGURATION
# ==========================================
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Please set HF_TOKEN environment variable with your HuggingFace token.")

OUTPUT_DIR = Path("dataset_output")
IMAGES_DIR = OUTPUT_DIR / "images"
CSV_PATH = OUTPUT_DIR / "labels.csv"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
TARGET_TOTAL_IMAGES = 24000 

# ==========================================
# 1. IDENTITIES (The "Canvas")
# ==========================================
AGES = ["22-year-old", "28-year-old", "35-year-old", "45-year-old", "55-year-old"]
ETHNICITIES = ["Caucasian", "Black", "East Asian", "South Asian", "Hispanic", "Middle Eastern"]
MEN_BUILDS = ["slim", "athletic", "average build", "heavy set", "muscular"]
WOMEN_BUILDS = ["slim", "curvy", "athletic", "average build", "plus size"]

MEN_IDENTITIES = [f"{age} {ethnicity} man, {build}" for age in AGES for ethnicity in ETHNICITIES for build in MEN_BUILDS]
WOMEN_IDENTITIES = [f"{age} {ethnicity} woman, {build}" for age in AGES for ethnicity in ETHNICITIES for build in WOMEN_BUILDS]

# ==========================================
# 2. EXHAUSTIVE PROMPT MATRIX (TAXONOMY)
# ==========================================

# --- MEN GROOMING VARIATIONS ---
MEN_GROOMING_VARS = [
    # Level 1-3: Extreme Flaws
    {"desc": "extreme bedhead messy uncombed hair, patchy sparse facial hair, visible severe acne blemishes on face, looking tired", "score": 2, 
     "labels": {"hair_messy": 1, "hair_flat": 0, "hair_overgrown": 0, "hair_styled": 0, "beard_patchy": 1, "beard_neckbeard": 0, "beard_groomed": 0, "skin_acne": 1, "skin_dark_circles": 1, "skin_clear": 0, "eyebrows_unkempt": 1}},
    {"desc": "flat greasy unwashed hair, unshaven neckbeard, unibrow, oily skin with large pores", "score": 2, 
     "labels": {"hair_messy": 0, "hair_flat": 1, "hair_overgrown": 0, "hair_styled": 0, "beard_patchy": 0, "beard_neckbeard": 1, "beard_groomed": 0, "skin_acne": 0, "skin_dark_circles": 0, "skin_clear": 0, "eyebrows_unkempt": 1}},
    {"desc": "overgrown hair covering ears and neck, scruffy unmaintained stubble, tired eyes with dark circles", "score": 3, 
     "labels": {"hair_messy": 0, "hair_flat": 0, "hair_overgrown": 1, "hair_styled": 0, "beard_patchy": 1, "beard_neckbeard": 0, "beard_groomed": 0, "skin_acne": 0, "skin_dark_circles": 1, "skin_clear": 0, "eyebrows_unkempt": 1}},
    
    # Level 4-6: Average
    {"desc": "basic short haircut, clean shaven, normal skin with minor imperfections, unshaped eyebrows", "score": 5, 
     "labels": {"hair_messy": 0, "hair_flat": 1, "hair_overgrown": 0, "hair_styled": 0, "beard_patchy": 0, "beard_neckbeard": 0, "beard_groomed": 0, "skin_acne": 0, "skin_dark_circles": 0, "skin_clear": 1, "eyebrows_unkempt": 0}},
    {"desc": "average medium length hair, average beard, normal resting face", "score": 5, 
     "labels": {"hair_messy": 0, "hair_flat": 0, "hair_overgrown": 0, "hair_styled": 0, "beard_patchy": 0, "beard_neckbeard": 0, "beard_groomed": 0, "skin_acne": 0, "skin_dark_circles": 0, "skin_clear": 1, "eyebrows_unkempt": 0}},
    {"desc": "dry frizzy hair, clean shaven, dry skin", "score": 4, 
     "labels": {"hair_messy": 1, "hair_flat": 0, "hair_overgrown": 0, "hair_styled": 0, "beard_patchy": 0, "beard_neckbeard": 0, "beard_groomed": 0, "skin_acne": 0, "skin_dark_circles": 0, "skin_clear": 0, "eyebrows_unkempt": 0}},

    # Level 7-10: Highly Polished
    {"desc": "perfectly styled textured voluminous hair with pomade, sharp lined-up short beard, flawless clear glowing skin, groomed eyebrows", "score": 9, 
     "labels": {"hair_messy": 0, "hair_flat": 0, "hair_overgrown": 0, "hair_styled": 1, "beard_patchy": 0, "beard_neckbeard": 0, "beard_groomed": 1, "skin_acne": 0, "skin_dark_circles": 0, "skin_clear": 1, "eyebrows_unkempt": 0}},
    {"desc": "sharp skin fade haircut, clean shaven smooth skin, clear glowing complexion, highly photogenic", "score": 10, 
     "labels": {"hair_messy": 0, "hair_flat": 0, "hair_overgrown": 0, "hair_styled": 1, "beard_patchy": 0, "beard_neckbeard": 0, "beard_groomed": 0, "skin_acne": 0, "skin_dark_circles": 0, "skin_clear": 1, "eyebrows_unkempt": 0}},
    {"desc": "elegant parted styled hair, well-maintained full beard with sharp lines, perfectly moisturized skin", "score": 10, 
     "labels": {"hair_messy": 0, "hair_flat": 0, "hair_overgrown": 0, "hair_styled": 1, "beard_patchy": 0, "beard_neckbeard": 0, "beard_groomed": 1, "skin_acne": 0, "skin_dark_circles": 0, "skin_clear": 1, "eyebrows_unkempt": 0}}
]

# --- WOMEN GROOMING VARIATIONS ---
WOMEN_GROOMING_VARS = [
    # Level 1-3: Extreme Flaws
    {"desc": "frizzy highly messy unbrushed hair, uneven poorly blended foundation, smudged eyeliner, visible acne breakouts, dry lips", "score": 2, 
     "labels": {"hair_frizzy_messy": 1, "hair_flat": 0, "hair_styled_voluminous": 0, "makeup_uneven": 1, "makeup_heavy_clashing": 0, "makeup_flawless": 0, "skin_acne": 1, "skin_tired_eyes": 1, "skin_clear": 0, "eyebrows_messy": 1}},
    {"desc": "flat oily hair, extremely heavy clashing unnatural makeup, bright clashing eyeshadow, tired eyes with dark circles", "score": 2, 
     "labels": {"hair_frizzy_messy": 0, "hair_flat": 1, "hair_styled_voluminous": 0, "makeup_uneven": 0, "makeup_heavy_clashing": 1, "makeup_flawless": 0, "skin_acne": 0, "skin_tired_eyes": 1, "skin_clear": 0, "eyebrows_messy": 1}},
    {"desc": "messy bun, no makeup, pale tired skin, unplucked messy eyebrows", "score": 3, 
     "labels": {"hair_frizzy_messy": 1, "hair_flat": 0, "hair_styled_voluminous": 0, "makeup_uneven": 0, "makeup_heavy_clashing": 0, "makeup_flawless": 0, "skin_acne": 0, "skin_tired_eyes": 1, "skin_clear": 0, "eyebrows_messy": 1}},

    # Level 4-6: Average
    {"desc": "simple ponytail, minimal basic makeup, normal skin with minor blemishes", "score": 5, 
     "labels": {"hair_frizzy_messy": 0, "hair_flat": 0, "hair_styled_voluminous": 0, "makeup_uneven": 0, "makeup_heavy_clashing": 0, "makeup_flawless": 0, "skin_acne": 0, "skin_tired_eyes": 0, "skin_clear": 1, "eyebrows_messy": 0}},
    # Level 1-3: Extreme Flaws (effort-based, no biological traits)
    {"desc": "severely tangled unbrushed ratty hair, visibly greasy unwashed roots, clumpy uneven mascara, smudged lip color, harsh unblended foundation lines, dry flaky lips", "score": 2, 
     "labels": {"hair_frizzy_messy": 1, "hair_flat": 0, "hair_styled_voluminous": 0, "makeup_uneven": 1, "makeup_heavy_clashing": 0, "makeup_flawless": 0, "skin_clear": 0, "eyebrows_messy": 1}},
    {"desc": "flat greasy oily unwashed hair, heavy clashing mismatched makeup, bright clashing eyeshadow poorly applied, dry chapped lips", "score": 2, 
     "labels": {"hair_frizzy_messy": 0, "hair_flat": 1, "hair_styled_voluminous": 0, "makeup_uneven": 0, "makeup_heavy_clashing": 1, "makeup_flawless": 0, "skin_clear": 0, "eyebrows_messy": 1}},
    {"desc": "messy unbrushed bun, completely bare face with no makeup effort, dry flaky skin, unplucked sparse eyebrows", "score": 3, 
     "labels": {"hair_frizzy_messy": 1, "hair_flat": 0, "hair_styled_voluminous": 0, "makeup_uneven": 0, "makeup_heavy_clashing": 0, "makeup_flawless": 0, "skin_clear": 0, "eyebrows_messy": 1}},

    # Level 4-6: Average
    {"desc": "basic simple ponytail, flat unstyled hair, minimal everyday natural makeup, bare natural skin", "score": 5, 
     "labels": {"hair_frizzy_messy": 0, "hair_flat": 0, "hair_styled_voluminous": 0, "makeup_uneven": 0, "makeup_heavy_clashing": 0, "makeup_flawless": 0, "skin_clear": 1, "eyebrows_messy": 0}},
    {"desc": "straight hair lacking volume, basic mascara and lip gloss only, clear natural skin", "score": 6, 
     "labels": {"hair_frizzy_messy": 0, "hair_flat": 1, "hair_styled_voluminous": 0, "makeup_uneven": 0, "makeup_heavy_clashing": 0, "makeup_flawless": 0, "skin_clear": 1, "eyebrows_messy": 0}},

    # Level 7-10: Highly Polished
    {"desc": "flawless blowout hair with high volume and shine, perfectly blended professional makeup, sharp clean eyeliner, deeply hydrated glowing skin, perfectly shaped eyebrows", "score": 10, 
     "labels": {"hair_frizzy_messy": 0, "hair_flat": 0, "hair_styled_voluminous": 1, "makeup_uneven": 0, "makeup_heavy_clashing": 0, "makeup_flawless": 1, "skin_clear": 1, "eyebrows_messy": 0}},
    {"desc": "elegant sleek pulled-back styled hair, striking flawless blended makeup, perfectly contoured face, radiant moisturized glowing skin", "score": 10, 
     "labels": {"hair_frizzy_messy": 0, "hair_flat": 0, "hair_styled_voluminous": 1, "makeup_uneven": 0, "makeup_heavy_clashing": 0, "makeup_flawless": 1, "skin_clear": 1, "eyebrows_messy": 0}}
]

# --- MEN OUTFIT VARIATIONS ---
MEN_OUTFIT_VARS = [
    # Flaws
    {"desc": "wearing a severely wrinkled, heavily stained and overly baggy cheap grey t-shirt, terrible unflattering fit, sloppy untucked styling, mismatched clashing colors", "score": 2, 
     "labels": {"clothes_wrinkled": 1, "clothes_crisp": 0, "fit_too_baggy": 1, "fit_too_tight": 0, "fit_tailored": 0, "colors_clashing": 1, "colors_harmonious": 0, "styling_sloppy": 1, "styling_sharp": 0}},
    {"desc": "wearing a shirt uncomfortably tight pulling at the buttons, inappropriate layering, heavily wrinkled trousers, sloppy overall appearance", "score": 3, 
     "labels": {"clothes_wrinkled": 1, "clothes_crisp": 0, "fit_too_baggy": 0, "fit_too_tight": 1, "fit_tailored": 0, "colors_clashing": 0, "colors_harmonious": 0, "styling_sloppy": 1, "styling_sharp": 0}},
    {"desc": "wearing ill-fitting overly long pants pooling at ankles, faded mismatched shirt, wrinkled untucked", "score": 3, 
     "labels": {"clothes_wrinkled": 1, "clothes_crisp": 0, "fit_too_baggy": 1, "fit_too_tight": 0, "fit_tailored": 0, "colors_clashing": 1, "colors_harmonious": 0, "styling_sloppy": 1, "styling_sharp": 0}},

    # Average
    {"desc": "wearing a basic plain t-shirt and standard jeans, acceptable average fit, neutral colors, clean everyday casual", "score": 5, 
     "labels": {"clothes_wrinkled": 0, "clothes_crisp": 0, "fit_too_baggy": 0, "fit_too_tight": 0, "fit_tailored": 0, "colors_clashing": 0, "colors_harmonious": 1, "styling_sloppy": 0, "styling_sharp": 0}},

    # Polished (diverse: not just suits)
    {"desc": "wearing a perfectly fitted smart-casual layered streetwear outfit, crisp clean fitted t-shirt under a sleek modern jacket, pristine condition fabrics, flawlessly tailored proportions, highly stylish", "score": 10, 
     "labels": {"clothes_wrinkled": 0, "clothes_crisp": 1, "fit_too_baggy": 0, "fit_too_tight": 0, "fit_tailored": 1, "colors_clashing": 0, "colors_harmonious": 1, "styling_sloppy": 0, "styling_sharp": 1}},
    {"desc": "wearing well-fitted chino trousers and a crisp fitted polo shirt, perfectly color coordinated, pristine clean fabrics, sharp casual styling", "score": 9, 
     "labels": {"clothes_wrinkled": 0, "clothes_crisp": 1, "fit_too_baggy": 0, "fit_too_tight": 0, "fit_tailored": 1, "colors_clashing": 0, "colors_harmonious": 1, "styling_sloppy": 0, "styling_sharp": 1}}
]

# --- WOMEN OUTFIT VARIATIONS ---
WOMEN_OUTFIT_VARS = [
    # Flaws
    {"desc": "wearing a heavily wrinkled, cheap unfitted casual dress, terrible unflattering fit that distorts proportions, sloppy and messy silhouette, mismatched clashing colors", "score": 2, 
     "labels": {"clothes_wrinkled": 1, "clothes_crisp": 0, "fit_baggy_unflattering": 1, "fit_awkwardly_tight": 0, "fit_tailored": 0, "colors_clashing": 1, "colors_harmonious": 0, "proportions_bad": 1, "proportions_good": 0}},
    {"desc": "wearing clothes that are uncomfortably tight and riding up, unbalanced awkward proportions, clashing mismatched patterns", "score": 3, 
     "labels": {"clothes_wrinkled": 0, "clothes_crisp": 0, "fit_baggy_unflattering": 0, "fit_awkwardly_tight": 1, "fit_tailored": 0, "colors_clashing": 1, "colors_harmonious": 0, "proportions_bad": 1, "proportions_good": 0}},
    {"desc": "wearing an oversized bulky top that hides the figure unflatteringly, wrinkled fabric, sloppy silhouette", "score": 3, 
     "labels": {"clothes_wrinkled": 1, "clothes_crisp": 0, "fit_baggy_unflattering": 1, "fit_awkwardly_tight": 0, "fit_tailored": 0, "colors_clashing": 0, "colors_harmonious": 0, "proportions_bad": 1, "proportions_good": 0}},

    # Average
    {"desc": "wearing a standard casual top and jeans, average fit, normal everyday look, neutral matching colors, clean clothes", "score": 5, 
     "labels": {"clothes_wrinkled": 0, "clothes_crisp": 0, "fit_baggy_unflattering": 0, "fit_awkwardly_tight": 0, "fit_tailored": 0, "colors_clashing": 0, "colors_harmonious": 1, "proportions_bad": 0, "proportions_good": 0}},

    # Polished (diverse: not just formal dresses)
    {"desc": "wearing a perfectly fitted chic casual outfit, stylish cropped jacket and crisp high-waisted trousers, pristine clean fabrics, flawlessly tailored proportions, highly stylish everyday wear", "score": 10, 
     "labels": {"clothes_wrinkled": 0, "clothes_crisp": 1, "fit_baggy_unflattering": 0, "fit_awkwardly_tight": 0, "fit_tailored": 1, "colors_clashing": 0, "colors_harmonious": 1, "proportions_bad": 0, "proportions_good": 1}},
    {"desc": "wearing a perfectly fitted midi skirt and crisp tucked-in blouse, beautifully harmonious color palette, pristine fabrics, flattering well-balanced proportions, stylish casual-chic look", "score": 9, 
     "labels": {"clothes_wrinkled": 0, "clothes_crisp": 1, "fit_baggy_unflattering": 0, "fit_awkwardly_tight": 0, "fit_tailored": 1, "colors_clashing": 0, "colors_harmonious": 1, "proportions_bad": 0, "proportions_good": 1}}
]


# ==========================================
# 3. TASK GENERATION
# ==========================================
TASKS = []
for _ in range((TARGET_TOTAL_IMAGES // 4) // len(MEN_GROOMING_VARS) + 1):
    for var in MEN_GROOMING_VARS:
        TASKS.append({"category": "Men_Grooming", "identity": random.choice(MEN_IDENTITIES), "variation": var})

for _ in range((TARGET_TOTAL_IMAGES // 4) // len(WOMEN_GROOMING_VARS) + 1):
    for var in WOMEN_GROOMING_VARS:
        TASKS.append({"category": "Women_Grooming", "identity": random.choice(WOMEN_IDENTITIES), "variation": var})

for _ in range((TARGET_TOTAL_IMAGES // 4) // len(MEN_OUTFIT_VARS) + 1):
    for var in MEN_OUTFIT_VARS:
        TASKS.append({"category": "Men_Outfit", "identity": random.choice(MEN_IDENTITIES), "variation": var})

for _ in range((TARGET_TOTAL_IMAGES // 4) // len(WOMEN_OUTFIT_VARS) + 1):
    for var in WOMEN_OUTFIT_VARS:
        TASKS.append({"category": "Women_Outfit", "identity": random.choice(WOMEN_IDENTITIES), "variation": var})

random.shuffle(TASKS)
TASKS = TASKS[:TARGET_TOTAL_IMAGES]

# ==========================================
# 4. FLUX.1 SETUP & GENERATION LOOP
# ==========================================
if __name__ == "__main__":
    print("Loading FLUX.1 [dev]... This takes a moment.")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()

    print(f"Starting generation of {len(TASKS)} images...")

    all_label_keys = set()
    for t in TASKS:
        all_label_keys.update(t["variation"]["labels"].keys())
    all_label_keys = sorted(list(all_label_keys))

    csv_headers = ["filename", "category", "score"] + all_label_keys
    write_header = not CSV_PATH.exists()

    with open(CSV_PATH, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        if write_header:
            writer.writeheader()

        for idx, task in enumerate(tqdm(TASKS)):
            cat = task["category"]
            ident = task["identity"]
            var = task["variation"]
            
            if "Grooming" in cat:
                alignment_guide = "front-facing head-and-shoulders portrait, perfectly centered face, straight-on eye-level camera angle, looking directly at the camera, bright even studio lighting"
            else:
                alignment_guide = "front-facing full-body portrait, perfectly centered, standing straight, straight-on eye-level camera angle, looking directly at the camera, head to toe visible, bright even studio lighting"
                
            prompt = f"A {alignment_guide} of a {ident}, {var['desc']}. Photorealistic, ultra detailed, 85mm lens."
            
            try:
                image = pipe(
                    prompt,
                    height=1024,
                    width=1024,
                    guidance_scale=3.5,
                    num_inference_steps=28,
                    max_sequence_length=512
                ).images[0]

                filename = f"{idx:05d}_{cat}.png"
                filepath = IMAGES_DIR / filename
                image.save(filepath)

                row = {"filename": filename, "category": cat, "score": var["score"]}
                for key in all_label_keys:
                    row[key] = var["labels"].get(key, 0)
                
                writer.writerow(row)
                f.flush()
            except Exception as e:
                print(f"Failed on image {idx}: {e}")

    print("✅ Dataset generation complete!")
