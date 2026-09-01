import os
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
import torch
from diffusers import QwenImagePipeline
from pathlib import Path

# ==========================================
# QWEN-IMAGE-2512 VARIATION TEST SCRIPT (48 Images)
# Direct counterpart to test_flux_variations.py -- same taxonomy v6
# prompts, same identities, same seeds, so the two output folders can
# be compared image-for-image on prompt adherence between FLUX.1 [dev]
# and Qwen-Image-2.0.
#
# Qwen-Image-2512 is Apache 2.0 (no gated-repo auth needed, unlike
# FLUX.1 [dev]) and uses a different text-conditioning setup, so the
# CLIP-77-token truncation issue that motivated the v6 "opener" prefix
# for Flux does not apply here the same way -- but we keep the same
# prompt text unchanged for a fair comparison rather than re-tuning it
# per-model.
# ==========================================

OUTPUT_DIR = Path("test_variations_qwen_v6")
OUTPUT_DIR.mkdir(exist_ok=True)

TRUE_CFG_SCALE = 4.0
NUM_INFERENCE_STEPS = 40
NEGATIVE_PROMPT = "blurry, low quality, deformed, extra limbs, watermark, text artifacts"

ALIGN_GROOMING = "front-facing head-and-shoulders portrait, centered face, straight-on eye-level angle, looking at the camera, bright even studio lighting"
ALIGN_OUTFIT = "front-facing full-body portrait, centered, standing straight, straight-on eye-level angle, looking at the camera, head to toe visible, bright even studio lighting"

MEN_IDENTITIES = [
    {"age": "28-year-old", "ethnicity": "Caucasian", "build": "heavy set", "face": "round face, double chin"},
    {"age": "45-year-old", "ethnicity": "Black", "build": "athletic", "face": "strong square jaw, deep set eyes"},
    {"age": "19-year-old", "ethnicity": "South Asian", "build": "very skinny", "face": "sharp angular face, prominent nose"},
    {"age": "35-year-old", "ethnicity": "East Asian", "build": "average build", "face": "wide flat face"},
]

WOMEN_IDENTITIES = [
    {"age": "22-year-old", "ethnicity": "Hispanic", "build": "plus size", "face": "round soft face shape"},
    {"age": "30-year-old", "ethnicity": "Caucasian", "build": "slim", "face": "long narrow face, prominent cheekbones"},
    {"age": "50-year-old", "ethnicity": "Black", "build": "average build", "face": "high cheekbones, visible age lines"},
    {"age": "26-year-old", "ethnicity": "East Asian", "build": "athletic", "face": "heart-shaped face"},
]

MEN_GROOMING_VARS = [
    {"level": "Flaw", "opener": "Greasy unwashed matted hair, patchy neckbeard, dry flaking skin, unibrow.",
     "desc": "visibly oily greasy hair matted flat with individual strands clumped together and a strong visible grease sheen catching the light -- the hair looks unmistakably unwashed and oily, thin patchy neckbeard growing in uneven blotchy patches with bare skin gaps clearly visible, dry flaking dead skin visibly peeling on the forehead, thick unshaped unibrow connecting across the nose. The greasy hair and patchy neglected beard are the most obvious features of his face, giving an unmistakably unkempt, low-effort appearance at a glance"},
    {"level": "Average", "opener": "Plain unstyled flat hair, no grooming product, ordinary skin, zero effort.",
     "desc": "plain unstyled short hair with no product lying flat with no defined shape, ordinary trimmed facial hair with no sharp lineup, plain skin with no visible skincare routine, natural unshaped eyebrows, zero grooming effort applied but no visible neglect"},
    {"level": "Polished", "opener": "Meticulously styled voluminous hair, sharp groomed beard, glowing skin.",
     "desc": "meticulously styled textured voluminous hair with a visible pomade sheen and sharp defined part, razor-sharp lined-up short beard with crisp clean edges, flawless clear glowing hydrated skin, perfectly groomed shaped eyebrows"}
]

MEN_OUTFIT_VARS = [
    {"level": "Flaw", "opener": "Heavily wrinkled, stained, mismatched, sloppy low-effort outfit.",
     "desc": "wearing a heavily wrinkled cheap grey t-shirt covered in deep creases and fold lines across the chest and stomach -- the fabric is visibly crumpled and unironed, visible dried yellow sweat stains under the armpits, mismatched clashing colors between the shirt and pants. The wrinkled fabric and visible stains are the most obvious features of the outfit, giving an unmistakably sloppy, low-effort look at a glance"},
    {"level": "Average", "opener": "Plain basic t-shirt and jeans, zero styling effort.",
     "desc": "wearing a basic plain t-shirt and standard jeans with a normal fit, neutral matching colors, clean clothes with zero styling effort applied"},
    {"level": "Polished", "opener": "Perfectly fitted, crisp, tailored, stylish outfit.",
     "desc": "wearing a perfectly fitted smart-casual layered streetwear outfit, a crisp clean fitted t-shirt under a sleek modern jacket with pristine sharp-edged fabric, expertly tailored to their own figure with a high-effort highly stylish finish"}
]

WOMEN_GROOMING_VARS = [
    {"level": "Flaw", "opener": "Greasy unwashed tangled hair, smudged makeup, flaking dry lips.",
     "desc": "visibly unwashed hair with greasy flat oily roots showing a strong visible sheen at the scalp -- the hair looks unmistakably unwashed and oily, ends tangled and knotted with no brushing, frizz sticking out unevenly in all directions, thick clumpy mascara flaking onto the under-eye skin, lipstick smudged unevenly past the lip line, a harsh visible demarcation line where foundation ends at the jaw not matching the neck skin tone. The greasy hair and smudged makeup are the most obvious features of her face, giving an unmistakably unkempt, low-effort appearance at a glance"},
    {"level": "Average", "opener": "Plain flat ponytail, minimal natural makeup, zero effort.",
     "desc": "basic simple ponytail with flat unstyled hair and no product, minimal everyday natural makeup applied with zero effort, bare natural skin with no visible routine"},
    {"level": "Polished", "opener": "Flawless voluminous hair, perfectly blended makeup, glowing skin.",
     "desc": "flawless blowout hair with high volume and a glossy shine, perfectly blended professional makeup with sharp clean eyeliner, deeply hydrated glowing skin, perfectly shaped groomed eyebrows"}
]

WOMEN_OUTFIT_VARS = [
    {"level": "Flaw", "opener": "Heavily wrinkled, ill-fitting, mismatched, sloppy low-effort outfit.",
     "desc": "wearing a heavily wrinkled cheap dress covered in visible deep creases and fold lines across the fabric -- the fabric is visibly crumpled and unironed, visibly stretched or bunched fabric at the seams, mismatched clashing colors that visibly clash. The wrinkled fabric and clashing colors are the most obvious features of the outfit, giving an unmistakably sloppy, low-effort look at a glance"},
    {"level": "Average", "opener": "Plain casual top and jeans, zero styling effort.",
     "desc": "wearing a standard casual top and jeans with a normal fit, neutral matching colors, clean clothes with zero styling effort applied"},
    {"level": "Polished", "opener": "Perfectly fitted, crisp, tailored, stylish outfit.",
     "desc": "wearing a perfectly fitted chic casual outfit, a stylish cropped jacket over crisp high-waisted trousers with pristine sharp-edged fabric, expertly tailored to their own figure with a high-effort stylish finish"}
]

TASKS = []

for cat, gender_word, identities, var_sets in [
    ("Men_Grooming", "man", MEN_IDENTITIES, MEN_GROOMING_VARS),
    ("Men_Outfit", "man", MEN_IDENTITIES, MEN_OUTFIT_VARS),
    ("Women_Grooming", "woman", WOMEN_IDENTITIES, WOMEN_GROOMING_VARS),
    ("Women_Outfit", "woman", WOMEN_IDENTITIES, WOMEN_OUTFIT_VARS),
]:
    is_outfit = "Outfit" in cat
    for ident in identities:
        ident_str = f"{ident['age']} {ident['ethnicity']} {gender_word}, {ident['build']}, {ident['face']}"
        seed = hash((cat, ident_str)) % 1_000_000
        for var in var_sets:
            TASKS.append({"cat": cat, "ident_str": ident_str, "build": ident["build"], "is_outfit": is_outfit,
                          "level": var["level"], "opener": var["opener"], "desc": var["desc"], "seed": seed})


if __name__ == "__main__":
    print(f"Generating {len(TASKS)} test images with Qwen-Image-2512 (same taxonomy v6 prompts as Flux)...")
    print("Loading Qwen/Qwen-Image-2512... This takes a moment (and a first-run download of ~40GB).")

    pipe = QwenImagePipeline.from_pretrained(
        "Qwen/Qwen-Image-2512",
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()

    for idx, task in enumerate(TASKS):
        cat = task["cat"]
        ident_str = task["ident_str"]
        level = task["level"]
        opener = task["opener"]
        desc = task["desc"]
        seed = task["seed"]

        alignment = ALIGN_GROOMING if "Grooming" in cat else ALIGN_OUTFIT

        if task["is_outfit"]:
            full_opener = f"{opener} Body stays their natural {task['build']} figure, not slimmer or heavier."
            prompt = (
                f"{full_opener} A sharp {alignment} of a {ident_str}. "
                f"They are {desc}. Photorealistic, ultra detailed, tack-sharp crisp focus throughout, 85mm lens."
            )
        else:
            prompt = f"{opener} A sharp {alignment} of a {ident_str}, {desc}. Photorealistic, ultra detailed, tack-sharp crisp focus throughout, 85mm lens."

        short_ident = f"{ident_str.split(',')[0]}".replace(' ', '').replace('-year-old', '')
        filename = f"qwen_{idx+1:03d}_{cat}_{level}_{short_ident}_seed{seed}.png"

        print(f"[{idx+1}/{len(TASKS)}] Generating: {filename}...")

        generator = torch.Generator("cpu").manual_seed(seed)

        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=1024,
            width=1024,
            num_inference_steps=NUM_INFERENCE_STEPS,
            true_cfg_scale=TRUE_CFG_SCALE,
            generator=generator
        ).images[0]

        image.save(OUTPUT_DIR / filename)

    print(f"✅ Qwen-Image-2512 testing complete! Check the '{OUTPUT_DIR}' folder.")
    print("Compare each identity's Flaw/Average/Polished triplet (same seed) against the")
    print("matching files in test_variations_comprehensive_v6/ from test_flux_variations.py:")
    print("  1. Is the flaw obviously visible for EVERY identity?")
    print("  2. Is body shape/size still consistent across all three tiers?")
    print("  3. Is Average still as sharp/detailed as Flaw and Polished?")
