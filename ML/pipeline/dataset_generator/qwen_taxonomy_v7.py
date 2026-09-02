"""
Shared prompt taxonomy v7 for Qwen-Image-2512 generation.

WHY THIS EXISTS (v6 -> v7):
v6 (test_flux_variations.py / test_qwen_variations.py) used ONE hand-written
paragraph per tier (Flaw / Average / Polished) per category. Manual review of
the full 48-image Qwen-vs-Nano-Banana comparison surfaced two real problems
with that approach, independent of which image model is used:

  1. FLAW SEVERITY WAS FLAT. The single "Flaw" description was written to be
     maximally, unambiguously bad (to defeat FLUX's earlier tendency to
     collapse flaws into invisibility). Once tested on models that actually
     follow instructions (Qwen, Nano Banana), this produced costume-level
     "unhoused person" results -- e.g. clashing colors rendered as a literal
     patchwork of mismatched fabric -- when a real "needs improvement" user
     photo is usually just mildly rumpled, not theatrically distressed.
     v7 fixes this with an explicit severity gradient per attribute (severe
     vs mild), and with CONCRETE phrasing (named colors, named garments,
     named locations/sizes for stains) instead of vague adjectives like
     "mismatched" or "stained" that give the model room to invent something
     more dramatic than intended.

  2. POLISHED WAS A JACKET MONOCULTURE. The single "Polished" outfit
     description explicitly said "a sleek modern jacket" / "a stylish
     cropped jacket". Every model tested rendered a jacket, every time --
     not because the models are biased, but because that's literally the
     only instruction they were given. This directly contradicts this
     project's own design principle ("Polished does NOT mean suits/blazers
     -- fit, crispness, and color harmony matter, not formality"). v7 fixes
     this with a POOL of distinct polished-top archetypes, most of which do
     NOT involve outerwear, so a classifier trained on this data can't learn
     "jacket present" as a shortcut for "high score".

STRUCTURE: each visual attribute (hair, facial hair, skin, eyebrows for
grooming; top garment, fit, fabric condition, color coordination, bottom,
footwear for outfit) is its OWN independent axis with its own options per
tier. A prompt is built by picking one option per axis for the requested
tier and concatenating them -- so images vary on individual attributes
independently ("single kind of attribute" variation) rather than swapping
one big paragraph block wholesale. This also produces natural combinatorial
diversity at scale: e.g. Men_Outfit_Polished alone has 4 top-garment
options x 3 bottom options x 2 footwear options = 24 distinct combinations
before identity is even varied.

LABEL SCHEMA: kept identical to generate_flux_dataset.py's existing CSV
columns (hair_messy, beard_patchy, clothes_wrinkled, colors_harmonious,
etc.) for compatibility with 04_train_coreml_models.py -- each axis option
below carries a `labels` dict that gets merged into the row.
"""
import random

# ==========================================
# ALIGNMENT / FRAMING (unchanged from v6 -- this part was never the problem)
# ==========================================
ALIGN_GROOMING = "front-facing head-and-shoulders portrait, centered face, straight-on eye-level angle, looking at the camera, bright even studio lighting"
ALIGN_OUTFIT = "front-facing full-body portrait, centered, standing straight, straight-on eye-level angle, looking at the camera, head to toe visible, bright even studio lighting"

STYLE_SUFFIX = "Photorealistic, ultra detailed, tack-sharp crisp focus throughout, 85mm lens."

# ==========================================
# IDENTITIES (same matrix as v6)
# ==========================================
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

# ==========================================
# GROOMING AXES -- MEN
# Each axis: tier -> list of {"desc": ..., "labels": {...}}. "flaw_severe" and
# "flaw_mild" both map to the Flaw score band (1-3) but at different
# intensities; pick flaw_severe for score ~1-2, flaw_mild for score ~3.
# ==========================================
MEN_HAIR = {
    "flaw_severe": [
        {"desc": "visibly greasy, unwashed hair with strands clumped together and a strong oily sheen catching the light, flattened against the scalp",
         "labels": {"hair_messy": 0, "hair_flat": 1, "hair_overgrown": 0, "hair_styled": 0}},
    ],
    "flaw_mild": [
        {"desc": "flat, unbrushed hair with no styling, a little overgrown past a normal haircut length, clearly not styled today",
         "labels": {"hair_messy": 1, "hair_flat": 0, "hair_overgrown": 1, "hair_styled": 0}},
    ],
    "average": [
        {"desc": "plain short hair with no product, lying naturally, neither messy nor styled",
         "labels": {"hair_messy": 0, "hair_flat": 1, "hair_overgrown": 0, "hair_styled": 0}},
    ],
    "polished": [
        {"desc": "a short textured crop styled with matte paste, natural volume on top, clean tapered sides",
         "labels": {"hair_messy": 0, "hair_flat": 0, "hair_overgrown": 0, "hair_styled": 1}},
        {"desc": "a classic side part with a light pomade sheen and a sharp, clean hairline",
         "labels": {"hair_messy": 0, "hair_flat": 0, "hair_overgrown": 0, "hair_styled": 1}},
        {"desc": "a crisp skin fade with tight, freshly cut edges and short neat hair on top",
         "labels": {"hair_messy": 0, "hair_flat": 0, "hair_overgrown": 0, "hair_styled": 1}},
    ],
}

MEN_FACIAL_HAIR = {
    "flaw_severe": [
        {"desc": "patchy, uneven stubble growing in random clumps on the jaw and neck with bare visible skin gaps between patches",
         "labels": {"beard_patchy": 1, "beard_neckbeard": 0, "beard_groomed": 0}},
    ],
    "flaw_mild": [
        {"desc": "about a week of unshaven growth, uneven in length, with no defined edge or lineup",
         "labels": {"beard_patchy": 0, "beard_neckbeard": 1, "beard_groomed": 0}},
    ],
    "average": [
        {"desc": "a basic short trimmed beard with no defined lineup, ordinary upkeep",
         "labels": {"beard_patchy": 0, "beard_neckbeard": 0, "beard_groomed": 0}},
    ],
    "polished": [
        {"desc": "precisely lined-up short stubble with crisp razor-sharp edges along the cheeks and neck",
         "labels": {"beard_patchy": 0, "beard_neckbeard": 0, "beard_groomed": 1}},
        {"desc": "a full beard, evenly trimmed to a uniform length with sharp defined cheek and neck lines",
         "labels": {"beard_patchy": 0, "beard_neckbeard": 0, "beard_groomed": 1}},
        {"desc": "completely clean-shaven with smooth, even skin",
         "labels": {"beard_patchy": 0, "beard_neckbeard": 0, "beard_groomed": 0}},
    ],
}

MEN_SKIN = {
    "flaw_severe": [
        {"desc": "dry, visibly flaking skin with small rough patches on the forehead and around the nose",
         "labels": {"skin_acne": 0, "skin_dark_circles": 0, "skin_clear": 0}},
    ],
    "flaw_mild": [
        {"desc": "noticeable dark circles under the eyes and a dull, tired-looking complexion",
         "labels": {"skin_acne": 0, "skin_dark_circles": 1, "skin_clear": 0}},
    ],
    "average": [
        {"desc": "ordinary skin texture with a few small visible pores, not obviously cared for but not neglected either",
         "labels": {"skin_acne": 0, "skin_dark_circles": 0, "skin_clear": 1}},
    ],
    "polished": [
        {"desc": "smooth, evenly hydrated skin with a healthy natural glow",
         "labels": {"skin_acne": 0, "skin_dark_circles": 0, "skin_clear": 1}},
        {"desc": "clear, matte skin with minimal visible texture or shine",
         "labels": {"skin_acne": 0, "skin_dark_circles": 0, "skin_clear": 1}},
    ],
}

MEN_EYEBROWS = {
    "flaw_severe": [{"desc": "thick, unshaped eyebrows that grow together above the nose", "labels": {"eyebrows_unkempt": 1}}],
    "flaw_mild": [{"desc": "a few stray long hairs in otherwise ordinary eyebrows", "labels": {"eyebrows_unkempt": 1}}],
    "average": [{"desc": "natural, unshaped eyebrows, neither groomed nor unkempt", "labels": {"eyebrows_unkempt": 0}}],
    "polished": [{"desc": "neatly shaped eyebrows with clean, defined edges", "labels": {"eyebrows_unkempt": 0}}],
}

# ==========================================
# GROOMING AXES -- WOMEN
# ==========================================
WOMEN_HAIR = {
    "flaw_severe": [
        {"desc": "visibly greasy, unwashed hair with an oily sheen at the roots, strands clumped and flattened against the scalp",
         "labels": {"hair_frizzy_messy": 0, "hair_flat": 1, "hair_styled_voluminous": 0}},
    ],
    "flaw_mild": [
        {"desc": "tangled, unbrushed hair with visible frizz sticking out unevenly, clearly not brushed today",
         "labels": {"hair_frizzy_messy": 1, "hair_flat": 0, "hair_styled_voluminous": 0}},
    ],
    "average": [
        {"desc": "hair pulled back into a plain, low ponytail with no product",
         "labels": {"hair_frizzy_messy": 0, "hair_flat": 1, "hair_styled_voluminous": 0}},
    ],
    "polished": [
        {"desc": "a smooth blowout with visible shine and soft volume through the mid-lengths",
         "labels": {"hair_frizzy_messy": 0, "hair_flat": 0, "hair_styled_voluminous": 1}},
        {"desc": "sleek hair pulled back into a polished low bun with no flyaways",
         "labels": {"hair_frizzy_messy": 0, "hair_flat": 0, "hair_styled_voluminous": 1}},
        {"desc": "loose natural waves with a clean, defined middle part and visible shine",
         "labels": {"hair_frizzy_messy": 0, "hair_flat": 0, "hair_styled_voluminous": 1}},
    ],
}

WOMEN_MAKEUP = {
    "flaw_severe": [
        {"desc": "mascara that has smudged into two small dark streaks below each eye, as if the person has been sweating or crying",
         "labels": {"makeup_uneven": 1, "makeup_heavy_clashing": 0, "makeup_flawless": 0}},
        {"desc": "bright orange and blue eyeshadow applied unevenly in two clashing colors, with foundation visibly two shades lighter than the neck",
         "labels": {"makeup_uneven": 0, "makeup_heavy_clashing": 1, "makeup_flawless": 0}},
    ],
    "flaw_mild": [
        {"desc": "slightly smudged eyeliner and unevenly applied foundation with a visible line at the jaw",
         "labels": {"makeup_uneven": 1, "makeup_heavy_clashing": 0, "makeup_flawless": 0}},
    ],
    "average": [
        {"desc": "no makeup, bare face",
         "labels": {"makeup_uneven": 0, "makeup_heavy_clashing": 0, "makeup_flawless": 0}},
    ],
    "polished": [
        {"desc": "perfectly blended natural makeup with soft definition and a clean, even base",
         "labels": {"makeup_uneven": 0, "makeup_heavy_clashing": 0, "makeup_flawless": 1}},
        {"desc": "sharp, precisely applied eyeliner with a flawlessly blended base",
         "labels": {"makeup_uneven": 0, "makeup_heavy_clashing": 0, "makeup_flawless": 1}},
    ],
}

WOMEN_SKIN = {
    "flaw_severe": [{"desc": "dry, visibly flaking skin around the nose and on the cheeks", "labels": {"skin_clear": 0}}],
    "flaw_mild": [{"desc": "a dull, uneven skin tone with visible tiredness under the eyes", "labels": {"skin_clear": 0}}],
    "average": [{"desc": "ordinary bare skin with minor visible texture, not obviously cared for but not neglected", "labels": {"skin_clear": 1}}],
    "polished": [
        {"desc": "deeply hydrated, glowing skin with an even tone", "labels": {"skin_clear": 1}},
        {"desc": "clear, radiant skin with a soft natural highlight", "labels": {"skin_clear": 1}},
    ],
}

WOMEN_EYEBROWS = {
    "flaw_severe": [{"desc": "unplucked, sparse eyebrows with an uneven shape", "labels": {"eyebrows_messy": 1}}],
    "flaw_mild": [{"desc": "slightly overgrown eyebrows with a few stray hairs", "labels": {"eyebrows_messy": 1}}],
    "average": [{"desc": "natural, unshaped eyebrows", "labels": {"eyebrows_messy": 0}}],
    "polished": [{"desc": "perfectly shaped, groomed eyebrows with clean defined edges", "labels": {"eyebrows_messy": 0}}],
}

# ==========================================
# OUTFIT AXES -- MEN
# Concrete color pairs used instead of "mismatched clashing colors" to stop
# the model from inventing a patchwork effect.
# ==========================================
MEN_CLASH_PAIRS = [
    "a faded orange t-shirt with olive-green sweatpants",
    "a mustard-yellow t-shirt with maroon joggers",
    "a lime-green t-shirt with brown cargo shorts",
]
MEN_HARMONY_PAIRS = [
    "white and navy", "black and olive", "grey and burgundy", "navy and tan",
]

MEN_TOP_GARMENT = {
    # bundles_bottom=True: these already name both the top AND the bottom
    # garment (that's what makes them a "clash" -- two specific garments
    # picked to visibly not go together), so build_outfit_prompt() skips
    # appending a separately-picked MEN_BOTTOM description for these.
    "flaw_severe": [{"desc": f"wearing {p}", "labels": {"colors_clashing": 1, "colors_harmonious": 0}, "bundles_bottom": True} for p in MEN_CLASH_PAIRS],
    "flaw_mild": [{"desc": "wearing a plain grey t-shirt and denim jeans in unremarkable, non-clashing colors", "labels": {"colors_clashing": 0, "colors_harmonious": 0}}],
    "average": [{"desc": "wearing a plain t-shirt and jeans in two neutral colors that don't clash but don't especially complement each other either", "labels": {"colors_clashing": 0, "colors_harmonious": 0}}],
    # {colors} is filled in per-call in build_outfit_prompt() using rng.choice(MEN_HARMONY_PAIRS) --
    # NOT random.choice() here at module-load time, which would freeze one combination forever.
    # NOTE: these describe the TOP only -- the bottom garment is a separate,
    # independently-picked axis (MEN_BOTTOM below). Do not bundle a bottom
    # garment into these strings, or it will collide/contradict whatever
    # MEN_BOTTOM picks (e.g. "trousers" here + "jeans" from MEN_BOTTOM).
    "polished": [
        {"desc": "wearing a crisp fitted crew-neck t-shirt in a {colors} color combination -- no jacket, just the shirt",
         "labels": {"colors_clashing": 0, "colors_harmonious": 1}},
        {"desc": "wearing a fitted long-sleeve button-down shirt, tucked in, in a {colors} color combination",
         "labels": {"colors_clashing": 0, "colors_harmonious": 1}},
        {"desc": "wearing a fine-knit crew-neck sweater over a collared shirt, in a {colors} color combination",
         "labels": {"colors_clashing": 0, "colors_harmonious": 1}},
        {"desc": "wearing a crisp fitted t-shirt under a sleek unstructured bomber jacket, in a {colors} color combination",
         "labels": {"colors_clashing": 0, "colors_harmonious": 1}},
    ],
}

MEN_FIT = {
    "flaw_severe": [
        {"desc": "the t-shirt hanging noticeably loose off the shoulders and bunching at the waist, several sizes too large", "labels": {"fit_too_baggy": 1, "fit_too_tight": 0, "fit_tailored": 0}},
        {"desc": "the t-shirt uncomfortably tight, straining and pulling across the chest and stomach, clearly a size too small", "labels": {"fit_too_baggy": 0, "fit_too_tight": 1, "fit_tailored": 0}},
    ],
    "flaw_mild": [{"desc": "the fit slightly off -- a bit loose through the body, clearly not tailored", "labels": {"fit_too_baggy": 1, "fit_too_tight": 0, "fit_tailored": 0}}],
    "average": [{"desc": "a standard, unremarkable fit, neither baggy nor tight", "labels": {"fit_too_baggy": 0, "fit_too_tight": 0, "fit_tailored": 0}}],
    "polished": [{"desc": "tailored to the body, following the shoulder line and ending precisely at the hip", "labels": {"fit_too_baggy": 0, "fit_too_tight": 0, "fit_tailored": 1}}],
}

MEN_FABRIC_CONDITION = {
    "flaw_severe": [{"desc": "deeply wrinkled with visible crease lines across the chest and stomach, plus a coin-sized dried food stain near the hem", "labels": {"clothes_wrinkled": 1, "clothes_crisp": 0}}],
    "flaw_mild": [{"desc": "noticeably creased, as if pulled straight out of a laundry hamper, no stains", "labels": {"clothes_wrinkled": 1, "clothes_crisp": 0}}],
    "average": [{"desc": "clean and reasonably smooth fabric, unremarkable", "labels": {"clothes_wrinkled": 0, "clothes_crisp": 0}}],
    "polished": [{"desc": "crisp, freshly pressed fabric with no visible wrinkles", "labels": {"clothes_wrinkled": 0, "clothes_crisp": 1}}],
}

MEN_BOTTOM = {
    "flaw_severe": [{"desc": "baggy grey sweatpants sagging at the waist", "labels": {}}],
    "flaw_mild": [{"desc": "plain denim jeans, slightly too loose", "labels": {}}],
    "average": [{"desc": "standard-fit denim jeans", "labels": {}}],
    "polished": [
        {"desc": "tailored charcoal trousers with a clean crease", "labels": {}},
        {"desc": "well-fitted chinos", "labels": {}},
        {"desc": "dark slim-tapered jeans", "labels": {}},
    ],
}

MEN_FOOTWEAR = {
    "flaw_severe": [{"desc": "scuffed, dirty sneakers with the laces untied", "labels": {"styling_sloppy": 1, "styling_sharp": 0}}],
    "flaw_mild": [{"desc": "worn, slightly dirty sneakers", "labels": {"styling_sloppy": 1, "styling_sharp": 0}}],
    "average": [{"desc": "plain, clean sneakers", "labels": {"styling_sloppy": 0, "styling_sharp": 0}}],
    "polished": [
        {"desc": "clean white minimalist sneakers", "labels": {"styling_sloppy": 0, "styling_sharp": 1}},
        {"desc": "polished brown leather boots", "labels": {"styling_sloppy": 0, "styling_sharp": 1}},
        {"desc": "clean suede loafers", "labels": {"styling_sloppy": 0, "styling_sharp": 1}},
    ],
}

# ==========================================
# OUTFIT AXES -- WOMEN
# ==========================================
WOMEN_CLASH_PAIRS = [
    "a faded pink oversized t-shirt with mustard-yellow sweatpants",
    "a lime-green top with a brown patterned skirt",
    "an orange blouse with maroon leggings",
]
WOMEN_HARMONY_PAIRS = [
    "cream and black", "white and beige", "navy and blush pink", "olive and white",
]

WOMEN_TOP_GARMENT = {
    # See bundles_bottom comment on MEN_TOP_GARMENT above.
    "flaw_severe": [{"desc": f"wearing {p}", "labels": {"colors_clashing": 1, "colors_harmonious": 0}, "bundles_bottom": True} for p in WOMEN_CLASH_PAIRS],
    "flaw_mild": [{"desc": "wearing a plain oversized t-shirt and leggings in unremarkable, non-clashing colors", "labels": {"colors_clashing": 0, "colors_harmonious": 0}}],
    "average": [{"desc": "wearing a plain top and jeans in two neutral colors that don't clash but don't especially complement each other either", "labels": {"colors_clashing": 0, "colors_harmonious": 0}}],
    # {colors} filled in per-call, see MEN_TOP_GARMENT comment above.
    # NOTE: top only -- see the matching comment on MEN_TOP_GARMENT above.
    "polished": [
        {"desc": "wearing a crisp fitted blouse, tucked in, in a {colors} color combination -- no jacket, just the blouse",
         "labels": {"colors_clashing": 0, "colors_harmonious": 1}},
        {"desc": "wearing a fitted knit sweater in a {colors} color combination",
         "labels": {"colors_clashing": 0, "colors_harmonious": 1}},
        {"desc": "wearing a crisp fitted t-shirt, tucked in, in a {colors} color combination",
         "labels": {"colors_clashing": 0, "colors_harmonious": 1}},
        {"desc": "wearing a fitted t-shirt under a cropped tailored blazer, in a {colors} color combination",
         "labels": {"colors_clashing": 0, "colors_harmonious": 1}},
    ],
}

WOMEN_FIT = {
    "flaw_severe": [
        {"desc": "the top hanging shapelessly, several sizes too large and bunching at the waist", "labels": {"fit_baggy_unflattering": 1, "fit_awkwardly_tight": 0, "fit_tailored": 0}},
        {"desc": "the top uncomfortably tight and riding up, clearly a size too small", "labels": {"fit_baggy_unflattering": 0, "fit_awkwardly_tight": 1, "fit_tailored": 0}},
    ],
    "flaw_mild": [{"desc": "the fit slightly off -- a bit loose through the body, clearly not tailored", "labels": {"fit_baggy_unflattering": 1, "fit_awkwardly_tight": 0, "fit_tailored": 0}}],
    "average": [{"desc": "a standard, unremarkable fit, neither baggy nor tight", "labels": {"fit_baggy_unflattering": 0, "fit_awkwardly_tight": 0, "fit_tailored": 0}}],
    "polished": [{"desc": "tailored to the body with clean, flattering proportions", "labels": {"fit_baggy_unflattering": 0, "fit_awkwardly_tight": 0, "fit_tailored": 1}}],
}

WOMEN_FABRIC_CONDITION = {
    "flaw_severe": [{"desc": "deeply wrinkled fabric with visible crease lines, plus a coin-sized stain near the hem", "labels": {"clothes_wrinkled": 1, "clothes_crisp": 0}}],
    "flaw_mild": [{"desc": "noticeably creased, as if pulled straight out of a laundry hamper, no stains", "labels": {"clothes_wrinkled": 1, "clothes_crisp": 0}}],
    "average": [{"desc": "clean and reasonably smooth fabric, unremarkable", "labels": {"clothes_wrinkled": 0, "clothes_crisp": 0}}],
    "polished": [{"desc": "crisp, freshly pressed fabric with no visible wrinkles", "labels": {"clothes_wrinkled": 0, "clothes_crisp": 1}}],
}

WOMEN_BOTTOM = {
    "flaw_severe": [{"desc": "baggy grey sweatpants", "labels": {}}],
    "flaw_mild": [{"desc": "plain leggings, slightly too loose", "labels": {}}],
    "average": [{"desc": "standard-fit denim jeans", "labels": {}}],
    "polished": [
        {"desc": "a tailored midi skirt", "labels": {}},
        {"desc": "high-waisted straight-leg trousers", "labels": {}},
        {"desc": "dark slim-fit jeans", "labels": {}},
    ],
}

WOMEN_FOOTWEAR = {
    "flaw_severe": [{"desc": "worn, scuffed sneakers", "labels": {}}],
    "flaw_mild": [{"desc": "plain, slightly worn flats", "labels": {}}],
    "average": [{"desc": "plain, clean sneakers", "labels": {}}],
    "polished": [
        {"desc": "clean minimalist white sneakers", "labels": {}},
        {"desc": "polished pointed-toe flats", "labels": {}},
        {"desc": "clean ankle boots", "labels": {}},
    ],
}

# ==========================================
# TIER -> SCORE MAPPING
# ==========================================
TIER_SCORE = {"flaw_severe": 1, "flaw_mild": 3, "average": 5, "polished": 9}
GROOMING_TIERS = ["flaw_severe", "flaw_mild", "average", "polished"]
OUTFIT_TIERS = ["flaw_severe", "flaw_mild", "average", "polished"]

# ==========================================
# PROMPT BUILDERS
# ==========================================
def _pick(axis, tier, rng):
    return rng.choice(axis[tier])


def build_grooming_prompt(gender, identity, tier, rng):
    """gender: 'man' or 'woman'. identity: one row from MEN_/WOMEN_IDENTITIES. tier: one of GROOMING_TIERS."""
    ident_str = f"{identity['age']} {identity['ethnicity']} {gender}, {identity['build']}, {identity['face']}"
    hair_axis, skin_axis, brow_axis = (MEN_HAIR, MEN_SKIN, MEN_EYEBROWS) if gender == "man" else (WOMEN_HAIR, WOMEN_SKIN, WOMEN_EYEBROWS)
    hair = _pick(hair_axis, tier, rng)
    skin = _pick(skin_axis, tier, rng)
    brow = _pick(brow_axis, tier, rng)
    labels = {**hair["labels"], **skin["labels"], **brow["labels"]}

    if gender == "man":
        facial = _pick(MEN_FACIAL_HAIR, tier, rng)
        labels = {**labels, **facial["labels"]}
        detail = f"{hair['desc']}, {facial['desc']}, {skin['desc']}, {brow['desc']}"
    else:
        makeup = _pick(WOMEN_MAKEUP, tier, rng)
        labels = {**labels, **makeup["labels"]}
        detail = f"{hair['desc']}, {makeup['desc']}, {skin['desc']}, {brow['desc']}"

    prompt = f"A {ALIGN_GROOMING} of a {ident_str}. They have {detail}. {STYLE_SUFFIX}"
    return prompt, TIER_SCORE[tier], labels


def build_outfit_prompt(gender, identity, tier, rng):
    ident_str = f"{identity['age']} {identity['ethnicity']} {gender}, {identity['build']}, {identity['face']}"
    top_axis, fit_axis, fabric_axis, bottom_axis, shoe_axis = (
        (MEN_TOP_GARMENT, MEN_FIT, MEN_FABRIC_CONDITION, MEN_BOTTOM, MEN_FOOTWEAR) if gender == "man"
        else (WOMEN_TOP_GARMENT, WOMEN_FIT, WOMEN_FABRIC_CONDITION, WOMEN_BOTTOM, WOMEN_FOOTWEAR)
    )
    top = _pick(top_axis, tier, rng)
    fit = _pick(fit_axis, tier, rng)
    fabric = _pick(fabric_axis, tier, rng)
    bottom = _pick(bottom_axis, tier, rng)
    shoe = _pick(shoe_axis, tier, rng)
    labels = {**top["labels"], **fit["labels"], **fabric["labels"], **bottom["labels"], **shoe["labels"]}

    top_desc = top["desc"]
    if "{colors}" in top_desc:
        harmony_pairs = MEN_HARMONY_PAIRS if gender == "man" else WOMEN_HARMONY_PAIRS
        top_desc = top_desc.format(colors=rng.choice(harmony_pairs))

    if top.get("bundles_bottom"):
        # top_desc already names both garments (e.g. the CLASH_PAIRS) --
        # don't also append the independently-picked bottom, it would
        # contradict (two different pairs of pants in one prompt).
        detail = f"{top_desc}, {shoe['desc']}. The fit is {fit['desc']}. The fabric is {fabric['desc']}."
    else:
        detail = f"{top_desc} and {bottom['desc']}, {shoe['desc']}. The fit is {fit['desc']}. The fabric is {fabric['desc']}."
    body_lock = f"Body stays their natural {identity['build']} figure, not slimmer or heavier than described."
    prompt = f"A {ALIGN_OUTFIT} of a {ident_str}. {detail} {body_lock} {STYLE_SUFFIX}"
    return prompt, TIER_SCORE[tier], labels
