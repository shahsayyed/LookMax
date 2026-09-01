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

# Modestly trimmed (redundant phrasing removed) vs the original alignment
# text, to leave more of CLIP's 77-token budget for the opener + identity --
# see the CLIP-budget note in the generation loop below.
ALIGN_GROOMING = "front-facing head-and-shoulders portrait, centered face, straight-on eye-level angle, looking at the camera, bright even studio lighting"
ALIGN_OUTFIT = "front-facing full-body portrait, centered, standing straight, straight-on eye-level angle, looking at the camera, head to toe visible, bright even studio lighting"

# ==========================================
# 1. IDENTITIES (The "Canvas")
# ==========================================
AGES = ["22-year-old", "28-year-old", "35-year-old", "45-year-old", "55-year-old"]
ETHNICITIES = ["Caucasian", "Black", "East Asian", "South Asian", "Hispanic", "Middle Eastern"]
MEN_BUILDS = ["slim", "athletic", "average build", "heavy set", "muscular"]
WOMEN_BUILDS = ["slim", "curvy", "athletic", "average build", "plus size"]

# Kept as structured dicts (not flat strings) so "build" can be
# re-injected into outfit descriptions at every effort tier -- see
# the body-shape preservation note in the generation loop below.
MEN_IDENTITIES = [{"age": age, "ethnicity": ethnicity, "gender": "man", "build": build}
                   for age in AGES for ethnicity in ETHNICITIES for build in MEN_BUILDS]
WOMEN_IDENTITIES = [{"age": age, "ethnicity": ethnicity, "gender": "woman", "build": build}
                     for age in AGES for ethnicity in ETHNICITIES for build in WOMEN_BUILDS]

# ==========================================
# 2. EXHAUSTIVE PROMPT MATRIX (TAXONOMY)
# ==========================================

# --- MEN GROOMING VARIATIONS (effort-based only, no biological traits) ---
# Each entry has a short "opener" that leads the final prompt, guaranteed to
# fit inside CLIP's 77-token window (FLUX's CLIP encoder truncates silently
# past that -- see the CLIP-budget note in the generation loop below), plus
# the full elaborate "desc" that continues for T5 (max_sequence_length=512).
MEN_GROOMING_VARS = [
    # Level 1-3: Extreme Flaws
    {"opener": "Greasy unwashed matted hair, patchy neckbeard, dry flaking skin, unibrow.",
     "desc": "visibly oily greasy hair matted flat with individual strands clumped together and a strong visible grease sheen catching the light -- the hair looks unmistakably unwashed and oily, thin patchy neckbeard growing in uneven blotchy patches with bare skin gaps clearly visible, dry flaking dead skin visibly peeling on the forehead, thick unshaped unibrow connecting across the nose. The greasy hair and patchy neglected beard are the most obvious features of his face, giving an unmistakably unkempt, low-effort appearance at a glance", "score": 2,
     "labels": {"hair_messy": 0, "hair_flat": 1, "hair_overgrown": 0, "hair_styled": 0, "beard_patchy": 1, "beard_neckbeard": 0, "beard_groomed": 0, "skin_flaky": 1, "skin_clear": 0, "eyebrows_unkempt": 1}},
    {"opener": "Severely overgrown shaggy hair, untamed neckbeard, tired puffy skin.",
     "desc": "severely overgrown shaggy hair hanging past the ears and covering the neck in an unkempt uncontrolled mass -- clearly never trimmed or maintained, thick untamed neckbeard stubble creeping down the throat with no defined edge, tired dull skin with visible under-eye puffiness. The overgrown hair and untamed neckbeard are the most obvious features of his face, giving an unmistakably unkempt, low-effort appearance at a glance", "score": 2,
     "labels": {"hair_messy": 1, "hair_flat": 0, "hair_overgrown": 1, "hair_styled": 0, "beard_patchy": 0, "beard_neckbeard": 1, "beard_groomed": 0, "skin_flaky": 0, "skin_clear": 0, "eyebrows_unkempt": 1}},
    {"opener": "Flat greasy unwashed hair, patchy stubble, flaking skin, unibrow.",
     "desc": "flat lifeless unwashed hair pressed down with visible grease at the roots -- the hair clearly hasn't been washed or styled, sparse patchy stubble growing in disconnected clumps across the jaw, dry cracked flaking skin visible on the cheeks and forehead, a bushy overgrown unibrow connecting across the nose. The greasy flat hair and patchy stubble are the most obvious features of his face, giving an unmistakably unkempt, low-effort appearance at a glance", "score": 3,
     "labels": {"hair_messy": 0, "hair_flat": 1, "hair_overgrown": 0, "hair_styled": 0, "beard_patchy": 1, "beard_neckbeard": 0, "beard_groomed": 0, "skin_flaky": 1, "skin_clear": 0, "eyebrows_unkempt": 1}},

    # Level 4-6: Average
    {"opener": "Plain unstyled short hair, ordinary trimmed facial hair, zero effort.",
     "desc": "plain unstyled short hair with no product lying flat with no defined shape, ordinary trimmed facial hair with no sharp lineup, plain skin with no visible skincare routine, natural unshaped eyebrows, zero grooming effort applied but no visible neglect", "score": 5,
     "labels": {"hair_messy": 0, "hair_flat": 1, "hair_overgrown": 0, "hair_styled": 0, "beard_patchy": 0, "beard_neckbeard": 0, "beard_groomed": 0, "skin_flaky": 0, "skin_clear": 1, "eyebrows_unkempt": 0}},
    {"opener": "Average medium-length hair, ordinary trimmed beard, zero grooming effort.",
     "desc": "average medium length hair with no styling, ordinary trimmed beard, normal resting face with zero grooming effort applied", "score": 5,
     "labels": {"hair_messy": 0, "hair_flat": 0, "hair_overgrown": 0, "hair_styled": 0, "beard_patchy": 0, "beard_neckbeard": 0, "beard_groomed": 0, "skin_flaky": 0, "skin_clear": 1, "eyebrows_unkempt": 0}},
    {"opener": "Dry unstyled frizzy hair, clean shaven, zero skincare effort.",
     "desc": "dry unstyled frizzy hair with no product, clean shaven with no visible product use, dry skin with no visible skincare routine", "score": 4,
     "labels": {"hair_messy": 1, "hair_flat": 0, "hair_overgrown": 0, "hair_styled": 0, "beard_patchy": 0, "beard_neckbeard": 0, "beard_groomed": 0, "skin_flaky": 0, "skin_clear": 0, "eyebrows_unkempt": 0}},

    # Level 7-10: Highly Polished
    {"opener": "Meticulously styled voluminous hair, sharp groomed beard, glowing skin.",
     "desc": "meticulously styled textured voluminous hair with a visible pomade sheen and sharp defined part, razor-sharp lined-up short beard with crisp clean edges, flawless clear glowing hydrated skin, perfectly groomed shaped eyebrows, an overall high-effort polished appearance", "score": 9,
     "labels": {"hair_messy": 0, "hair_flat": 0, "hair_overgrown": 0, "hair_styled": 1, "beard_patchy": 0, "beard_neckbeard": 0, "beard_groomed": 1, "skin_flaky": 0, "skin_clear": 1, "eyebrows_unkempt": 0}},
    {"opener": "Sharp crisp fade haircut, clean shaven, radiant glowing skin.",
     "desc": "sharp crisp skin fade haircut with clean defined edges, perfectly clean shaven smooth skin, clear radiant glowing complexion, an overall polished camera-ready appearance", "score": 10,
     "labels": {"hair_messy": 0, "hair_flat": 0, "hair_overgrown": 0, "hair_styled": 1, "beard_patchy": 0, "beard_neckbeard": 0, "beard_groomed": 0, "skin_flaky": 0, "skin_clear": 1, "eyebrows_unkempt": 0}},
    {"opener": "Elegantly styled hair, sharp full beard, glowing moisturized skin.",
     "desc": "elegantly parted styled hair with visible shine, well-maintained full beard with razor-sharp defined lines, perfectly moisturized glowing skin, neatly groomed eyebrows, an overall high-effort polished appearance", "score": 10,
     "labels": {"hair_messy": 0, "hair_flat": 0, "hair_overgrown": 0, "hair_styled": 1, "beard_patchy": 0, "beard_neckbeard": 0, "beard_groomed": 1, "skin_flaky": 0, "skin_clear": 1, "eyebrows_unkempt": 0}}
]

# --- WOMEN GROOMING VARIATIONS (effort-based only, no biological traits) ---
WOMEN_GROOMING_VARS = [
    # Level 1-3: Extreme Flaws
    {"opener": "Greasy unwashed tangled hair, smudged makeup, flaking dry lips.",
     "desc": "visibly unwashed hair with greasy flat oily roots showing a strong visible sheen at the scalp -- the hair looks unmistakably unwashed and oily, ends tangled and knotted with no brushing, frizz sticking out unevenly in all directions, thick clumpy mascara flaking onto the under-eye skin, lipstick smudged unevenly past the lip line, a harsh visible demarcation line where foundation ends at the jaw not matching the neck skin tone. The greasy hair and smudged makeup are the most obvious features of her face, giving an unmistakably unkempt, low-effort appearance at a glance", "score": 2,
     "labels": {"hair_frizzy_messy": 1, "hair_flat": 0, "hair_styled_voluminous": 0, "makeup_uneven": 1, "makeup_heavy_clashing": 0, "makeup_flawless": 0, "skin_clear": 0, "eyebrows_messy": 1}},
    {"opener": "Flat greasy oily hair, heavy clashing makeup, puffy under-eyes.",
     "desc": "flat oily hair pressed down with a strong visible grease sheen at the roots -- the hair clearly hasn't been washed, extremely heavy clashing unnatural makeup applied in thick uneven layers, bright clashing eyeshadow smeared past the crease, visible under-eye puffiness with no concealer applied. The greasy flat hair and heavy clashing makeup are the most obvious features of her face, giving an unmistakably unkempt, low-effort appearance at a glance", "score": 2,
     "labels": {"hair_frizzy_messy": 0, "hair_flat": 1, "hair_styled_voluminous": 0, "makeup_uneven": 0, "makeup_heavy_clashing": 1, "makeup_flawless": 0, "skin_clear": 0, "eyebrows_messy": 1}},
    {"opener": "Messy unbrushed bun, bare neglected face, sparse unkempt eyebrows.",
     "desc": "messy unbrushed bun with visible loose flyaway strands sticking out in every direction -- clearly hasn't been brushed, completely bare face with no makeup effort, dry flaky patches visible on the skin, unplucked sparse eyebrows growing in uneven scattered directions. The messy unbrushed hair and unkempt eyebrows are the most obvious features of her face, giving an unmistakably neglected, low-effort appearance at a glance", "score": 3,
     "labels": {"hair_frizzy_messy": 1, "hair_flat": 0, "hair_styled_voluminous": 0, "makeup_uneven": 0, "makeup_heavy_clashing": 0, "makeup_flawless": 0, "skin_clear": 0, "eyebrows_messy": 1}},

    # Level 4-6: Average
    {"opener": "Plain flat ponytail, minimal natural makeup, zero effort.",
     "desc": "basic simple ponytail with flat unstyled hair and no product, minimal everyday natural makeup applied with zero effort, bare natural skin with no visible routine", "score": 5,
     "labels": {"hair_frizzy_messy": 0, "hair_flat": 0, "hair_styled_voluminous": 0, "makeup_uneven": 0, "makeup_heavy_clashing": 0, "makeup_flawless": 0, "skin_clear": 1, "eyebrows_messy": 0}},
    {"opener": "Plain straight hair, basic mascara and lip gloss, zero effort.",
     "desc": "plain straight hair with no volume or styling product, basic mascara and lip gloss only applied with zero effort, ordinary skin with no visible skincare routine", "score": 6,
     "labels": {"hair_frizzy_messy": 0, "hair_flat": 1, "hair_styled_voluminous": 0, "makeup_uneven": 0, "makeup_heavy_clashing": 0, "makeup_flawless": 0, "skin_clear": 1, "eyebrows_messy": 0}},

    # Level 7-10: Highly Polished
    {"opener": "Flawless voluminous hair, perfectly blended makeup, glowing skin.",
     "desc": "flawless blowout hair with high volume and a glossy shine, perfectly blended professional makeup with sharp clean eyeliner, deeply hydrated glowing skin, perfectly shaped groomed eyebrows, an overall high-effort polished appearance", "score": 10,
     "labels": {"hair_frizzy_messy": 0, "hair_flat": 0, "hair_styled_voluminous": 1, "makeup_uneven": 0, "makeup_heavy_clashing": 0, "makeup_flawless": 1, "skin_clear": 1, "eyebrows_messy": 0}},
    {"opener": "Sleek styled hair, flawless contoured makeup, radiant glowing skin.",
     "desc": "elegant sleek pulled-back styled hair with visible shine, striking flawless blended makeup with sharp precise contouring, radiant moisturized glowing skin, perfectly groomed eyebrows, an overall high-effort polished appearance", "score": 10,
     "labels": {"hair_frizzy_messy": 0, "hair_flat": 0, "hair_styled_voluminous": 1, "makeup_uneven": 0, "makeup_heavy_clashing": 0, "makeup_flawless": 1, "skin_clear": 1, "eyebrows_messy": 0}}
]

# --- MEN OUTFIT VARIATIONS ---
MEN_OUTFIT_VARS = [
    # Flaws
    {"opener": "Heavily wrinkled, stained, mismatched, sloppy low-effort outfit.",
     "desc": "wearing a heavily wrinkled cheap grey t-shirt covered in deep creases and fold lines across the chest and stomach -- the fabric is visibly crumpled and unironed, visible dried yellow sweat stains under the armpits, mismatched clashing colors between the shirt and pants. The wrinkled fabric and visible stains are the most obvious features of the outfit, giving an unmistakably sloppy, low-effort look at a glance", "score": 2,
     "labels": {"clothes_wrinkled": 1, "clothes_crisp": 0, "fit_too_baggy": 1, "fit_too_tight": 0, "fit_tailored": 0, "colors_clashing": 1, "colors_harmonious": 0, "styling_sloppy": 1, "styling_sharp": 0}},
    {"opener": "Straining tight shirt, mismatched layers, wrinkled trousers, sloppy outfit.",
     "desc": "wearing a shirt visibly straining and uncomfortably tight with the buttons pulling apart to expose gaps of skin underneath -- the poor fit is immediately obvious, an inappropriate mismatched layering combination, deeply wrinkled trousers with creases across the thighs. The straining tight shirt and wrinkled trousers are the most obvious features of the outfit, giving an unmistakably sloppy, low-effort appearance at a glance", "score": 3,
     "labels": {"clothes_wrinkled": 1, "clothes_crisp": 0, "fit_too_baggy": 0, "fit_too_tight": 1, "fit_tailored": 0, "colors_clashing": 0, "colors_harmonious": 0, "styling_sloppy": 1, "styling_sharp": 0}},
    {"opener": "Ill-fitting bunched pants, faded clashing shirt, wrinkled sloppy outfit.",
     "desc": "wearing ill-fitting overly long pants pooling and bunching up in folds at the ankles -- the poor fit is immediately obvious, a faded washed-out shirt in a color that clashes with the pants, both visibly wrinkled and untucked. The bunched-up pants and clashing faded shirt are the most obvious features of the outfit, giving an unmistakably sloppy, low-effort appearance at a glance", "score": 3,
     "labels": {"clothes_wrinkled": 1, "clothes_crisp": 0, "fit_too_baggy": 1, "fit_too_tight": 0, "fit_tailored": 0, "colors_clashing": 1, "colors_harmonious": 0, "styling_sloppy": 1, "styling_sharp": 0}},

    # Average
    {"opener": "Plain basic t-shirt and jeans, zero styling effort.",
     "desc": "wearing a basic plain t-shirt and standard jeans with a normal fit, neutral matching colors, clean clothes with zero styling effort applied", "score": 5,
     "labels": {"clothes_wrinkled": 0, "clothes_crisp": 0, "fit_too_baggy": 0, "fit_too_tight": 0, "fit_tailored": 0, "colors_clashing": 0, "colors_harmonious": 1, "styling_sloppy": 0, "styling_sharp": 0}},

    # Polished (diverse: not just suits)
    {"opener": "Perfectly fitted, crisp, tailored streetwear outfit.",
     "desc": "wearing a perfectly fitted smart-casual layered streetwear outfit, a crisp clean fitted t-shirt under a sleek modern jacket with pristine sharp-edged fabric, expertly tailored to their own figure with a high-effort highly stylish finish", "score": 10,
     "labels": {"clothes_wrinkled": 0, "clothes_crisp": 1, "fit_too_baggy": 0, "fit_too_tight": 0, "fit_tailored": 1, "colors_clashing": 0, "colors_harmonious": 1, "styling_sloppy": 0, "styling_sharp": 1}},
    {"opener": "Crisp tailored chinos and polo, sharp casual outfit.",
     "desc": "wearing well-fitted chino trousers with crisp visible creases and a fitted polo shirt tucked in cleanly, perfectly color coordinated, pristine unwrinkled fabrics, an overall high-effort sharp casual appearance", "score": 9,
     "labels": {"clothes_wrinkled": 0, "clothes_crisp": 1, "fit_too_baggy": 0, "fit_too_tight": 0, "fit_tailored": 1, "colors_clashing": 0, "colors_harmonious": 1, "styling_sloppy": 0, "styling_sharp": 1}}
]

# --- WOMEN OUTFIT VARIATIONS ---
WOMEN_OUTFIT_VARS = [
    # Flaws
    {"opener": "Heavily wrinkled, mismatched, sloppy low-effort outfit.",
     "desc": "wearing a heavily wrinkled cheap dress covered in visible deep creases and fold lines across the fabric -- the fabric is visibly crumpled and unironed, visibly stretched or bunched fabric at the seams, mismatched clashing colors that visibly clash. The wrinkled fabric and clashing colors are the most obvious features of the outfit, giving an unmistakably sloppy, low-effort look at a glance", "score": 2,
     "labels": {"clothes_wrinkled": 1, "clothes_crisp": 0, "fit_baggy_unflattering": 1, "fit_awkwardly_tight": 0, "fit_tailored": 0, "colors_clashing": 1, "colors_harmonious": 0, "proportions_bad": 1, "proportions_good": 0}},
    {"opener": "Straining tight fit, clashing mismatched patterns, sloppy outfit.",
     "desc": "wearing clothes that are visibly straining and uncomfortably tight with fabric riding up and bunching at the waist -- the poor fit is immediately obvious, clashing mismatched patterns that visibly do not coordinate. The straining tight fit and clashing patterns are the most obvious features of the outfit, giving an unmistakably sloppy, low-effort appearance at a glance", "score": 3,
     "labels": {"clothes_wrinkled": 0, "clothes_crisp": 0, "fit_baggy_unflattering": 0, "fit_awkwardly_tight": 1, "fit_tailored": 0, "colors_clashing": 1, "colors_harmonious": 0, "proportions_bad": 1, "proportions_good": 0}},
    {"opener": "Oversized shapeless top, wrinkled fabric, sloppy low-effort outfit.",
     "desc": "wearing an oversized bulky shapeless top with no defined silhouette -- the shapeless cut is immediately obvious, visibly wrinkled fabric covered in deep creases. The shapeless bulky top and wrinkled fabric are the most obvious features of the outfit, giving an unmistakably sloppy, low-effort appearance at a glance", "score": 3,
     "labels": {"clothes_wrinkled": 1, "clothes_crisp": 0, "fit_baggy_unflattering": 1, "fit_awkwardly_tight": 0, "fit_tailored": 0, "colors_clashing": 0, "colors_harmonious": 0, "proportions_bad": 1, "proportions_good": 0}},

    # Average
    {"opener": "Plain casual top and jeans, zero styling effort.",
     "desc": "wearing a standard casual top and jeans with a normal fit, neutral matching colors, clean clothes with zero styling effort applied", "score": 5,
     "labels": {"clothes_wrinkled": 0, "clothes_crisp": 0, "fit_baggy_unflattering": 0, "fit_awkwardly_tight": 0, "fit_tailored": 0, "colors_clashing": 0, "colors_harmonious": 1, "proportions_bad": 0, "proportions_good": 0}},

    # Polished (diverse: not just formal dresses)
    {"opener": "Perfectly fitted, crisp, tailored chic outfit.",
     "desc": "wearing a perfectly fitted chic casual outfit, a stylish cropped jacket over crisp high-waisted trousers with pristine sharp-edged fabric, expertly tailored to their own figure with a high-effort stylish everyday finish", "score": 10,
     "labels": {"clothes_wrinkled": 0, "clothes_crisp": 1, "fit_baggy_unflattering": 0, "fit_awkwardly_tight": 0, "fit_tailored": 1, "colors_clashing": 0, "colors_harmonious": 1, "proportions_bad": 0, "proportions_good": 1}},
    {"opener": "Crisp tailored skirt and blouse, chic stylish outfit.",
     "desc": "wearing a perfectly fitted midi skirt and a crisp tucked-in blouse with sharp-edged unwrinkled fabric, a beautifully harmonious color palette, expertly tailored to their own figure with a high-effort stylish casual-chic finish", "score": 9,
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
            ident_str = f"{ident['age']} {ident['ethnicity']} {ident['gender']}, {ident['build']}"

            opener = var["opener"]
            desc = var["desc"]

            if "Grooming" in cat:
                alignment_guide = ALIGN_GROOMING
            else:
                alignment_guide = ALIGN_OUTFIT

            # CLIP-BUDGET FIX: FLUX uses two text encoders -- CLIP (hard
            # 77-token limit, contributes a global conditioning vector) and
            # T5-XXL (max_sequence_length=512, drives most fine-grained
            # detail). Earlier taxonomy versions kept adding reinforcement
            # text and pushed prompts past 200 tokens, so CLIP was silently
            # truncating BEFORE the flaw/effort description even started
            # (verified with the real CLIPTokenizer -- for a typical Outfit
            # prompt, CLIP's view cut off mid-identity, never reaching
            # "wearing a..." at all). Every prompt now leads with a short
            # "opener" (the variation's core flaw/effort keywords, plus body
            # preservation for outfits) guaranteed to fit inside CLIP's
            # window; the full elaborate "desc" continues after for T5,
            # which has room to spare at 512 tokens.
            if "Outfit" in cat:
                # Body-shape preservation folded into the SAME short opener
                # sentence (not a separate one) to keep the CLIP-priority
                # prefix as compact as possible. "Flattering"/"stylish"
                # language elsewhere in desc otherwise pulls FLUX toward its
                # slim-model prior even for heavy-set/plus-size identities --
                # the app must never train the model to treat body size as
                # something styling fixes.
                full_opener = f"{opener} Body stays their natural {ident['build']} figure, not slimmer or heavier."
                prompt = (
                    f"{full_opener} A sharp {alignment_guide} of a {ident_str}. "
                    f"They are {desc}. Photorealistic, ultra detailed, tack-sharp crisp focus throughout, 85mm lens."
                )
            else:
                prompt = f"{opener} A sharp {alignment_guide} of a {ident_str}, {desc}. Photorealistic, ultra detailed, tack-sharp crisp focus throughout, 85mm lens."

            try:
                image = pipe(
                    prompt,
                    height=1024,
                    width=1024,
                    # NOTE: test_flux_variations.py v4 tests guidance_scale=5.0 for
                    # stronger flaw-prompt adherence. Validate that test batch first;
                    # bump this to match if the higher CFG holds up visually.
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
