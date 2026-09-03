"""
Shared prompt taxonomy for Qwen-Image-2512 synthetic dataset generation.
SINGLE SOURCE OF TRUTH for every variation matrix used by prompt_builder.py,
smoke_test.py, validation_sweep.py, and full_run.py. Nothing in this pipeline
should hand-duplicate a list of options that lives here.

WHY THIS FILE LOOKS THE WAY IT DOES (carried forward from
ML/archive/dataset_generator_v7/qwen_taxonomy_v7.py -- read that file's
docstring for the full history):
  - Concrete, sensory phrasing for flaws ("grease sheen", "clumped
    strands"), never vague intensity adjectives ("severely", "unkempt") --
    vague adjectives get softened back toward the model's aesthetic prior.
  - Flaw severity is a GRADIENT (flaw_severe vs flaw_mild), not one
    maximally-bad description, so "needs improvement" doesn't read as
    costume-level distress.
  - "Polished" is never a single archetype -- garment IDENTITY and effort
    CONDITION are wholly independent axes (see the bucket model below), so a
    classifier can't learn "jacket present" or "beige" as a shortcut for
    score.
  - Effort is entirely about things a user can change in minutes/hours
    (styling, cleanliness, fit, product use). It never encodes body size,
    face shape, or skin conditions like acne -- seeing ML/README.md's
    "Architecture Philosophy" section. NOTE: the archived v7 MEN_SKIN axis
    still carried a `skin_acne` label key that was hardcoded to 0 in every
    single entry (i.e. a dead, always-zero column smuggled in a "no acne"
    taxonomy) -- this file drops that key entirely rather than carrying the
    dead column forward.

THE THREE-BUCKET MODEL (see the build spec for the full rationale):
  Bucket A -- in the prompt, labelled, trained.      e.g. upper_type, score
  Bucket B -- in the prompt, NOT labelled, NOT trained. e.g. background,
              lighting, requested colour (colour is logged as *meta* for
              provenance/pixel-snapping, but never a trained head -- colour
              comes free from pose-derived regions on-device at runtime).
  Bucket B is load-bearing: it is sampled INDEPENDENTLY of effort tier, so
  the model can't learn "polished = studio-lit" or "flaw = dim room".
  Bucket C -- NOT in the prompt, measured from pixels after generation.
              e.g. qa_pass, measured colour, colour harmony. See
              extract_measured_labels.py.

ORDINAL FLAGS: "effort flags become 0/1/2 (absent/present/severe)". This is
implemented generically -- every flaw-style column is a pure function of
the sampled tier (severity_for_tier below), which makes it structurally
impossible to reproduce the historical bug where a flag had levels 0 and 2
but zero level-1 examples (that requires some *hand-curated* per-entry
label dict to forget to set the mild value -- there are no more per-entry
label dicts for these columns, so there's nothing to forget).

GARMENT IDENTITY vs EFFORT (core restructure -- see build spec section 2):
  - "Effort variant" (per tier) supplies CONDITION phrases (wrinkled vs
    crisp, baggy vs tailored, scuffed vs polished) and the ordinal/binary
    condition labels. Sampled from GROOMING_TIERS / OUTFIT_TIERS.
  - "Garment composition" (upper/mid/lower/footwear slots, pattern,
    colour) supplies WHAT is worn. Sampled independently via
    sample_outfit() below, using coherent (not uniform-independent) slot
    sampling so outfits are internally plausible.
  Every slot entry now carries a real label (fixing the archived bug where
  MEN_BOTTOM/WOMEN_BOTTOM/MEN_FOOTWEAR/WOMEN_FOOTWEAR had empty `labels: {}`
  dicts -- garment type was rendered into the prompt but never recorded).
"""
import random

# ==========================================================================
# TIERS / SCORE
# ==========================================================================
GROOMING_TIERS = ["flaw_severe", "flaw_mild", "average", "polished"]
OUTFIT_TIERS = ["flaw_severe", "flaw_mild", "average", "polished"]

# Continuous score jitter within each tier's band instead of one fixed value
# per tier. A regression head trained on exactly 4 discrete targets (as the
# archived taxonomy did: 1, 3, 5, 9) doesn't behave like a real regression
# problem. Bands match ML/README.md's original score table (Flaw 1-3,
# Average 4-6, Polished 7-10), leaving 3-4 and 6-7 as intentional gaps
# (severe/mild split absorbs some of the flaw band; there is no "average"
# blur into "polished").
SCORE_BANDS = {
    "flaw_severe": (1.0, 2.0),
    "flaw_mild": (2.1, 3.0),
    "average": (4.0, 6.0),
    "polished": (7.0, 10.0),
}


def sample_score(tier, rng):
    lo, hi = SCORE_BANDS[tier]
    return round(rng.uniform(lo, hi), 1)


def severity_for_tier(tier):
    """0=absent, 1=present(mild), 2=severe. Pure function of tier -- see
    module docstring on why this structurally prevents the missing-middle
    ordinal bug."""
    return {"flaw_severe": 2, "flaw_mild": 1, "average": 0, "polished": 0}[tier]


def positive_for_tier(tier):
    """Binary polish-achieved indicator: 1 only for the polished tier."""
    return 1 if tier == "polished" else 0


# ==========================================================================
# IDENTITY MATRIX (age x ethnicity x build x face shape). NEVER a label
# column -- purely rendering-time diversity, exactly like the archived
# pipeline. A face/body descriptor here is what stops the model from
# learning genetics as an "improvement" signal, not something we then turn
# around and score.
# ==========================================================================
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
    return {
        "age": rng.choice(AGES),
        "ethnicity": rng.choice(ETHNICITIES),
        "build": rng.choice(MEN_BUILDS if gender == "man" else WOMEN_BUILDS),
        "face": rng.choice(MEN_FACE_SHAPES if gender == "man" else WOMEN_FACE_SHAPES),
    }


def identity_str(gender, identity):
    return f"{identity['age']} {identity['ethnicity']} {gender}, {identity['build']}, {identity['face']}"


# ==========================================================================
# BUCKET B -- ENVIRONMENT (background / lighting / framing). Replaces the
# archived code's single hardcoded "bright even studio lighting" string
# that was baked into EVERY grooming and outfit image regardless of tier --
# confirmed in ML/archive/dataset_generator_v7/qwen_taxonomy_v7.py's
# ALIGN_GROOMING / ALIGN_OUTFIT. Sampled independently of tier/score so the
# model cannot learn "polished = studio-lit, flaw = dim room".
# ==========================================================================
BACKGROUNDS = [
    ("plain_wall", "standing against a plain neutral-colored wall"),
    ("bedroom", "standing in a bedroom with a bed and dresser visible behind them"),
    ("bathroom_mirror", "standing in front of a bathroom mirror, a faint reflection visible"),
    ("hallway", "standing in a plain apartment hallway"),
    ("living_room", "standing in a living room with a sofa visible behind them"),
    ("office", "standing in a casual office space with desks visible behind them"),
    ("outdoors", "standing outdoors on a city sidewalk with buildings in the background"),
    ("studio", "standing against a seamless studio backdrop"),
]

LIGHTING = [
    ("window_light", "soft natural window light from the side"),
    ("studio_lighting", "bright even studio lighting"),
    ("warm_overhead", "warm overhead indoor lighting"),
    ("cool_overhead", "cool-toned overhead fluorescent lighting"),
    ("dim", "dim, slightly underexposed indoor lighting"),
    ("slightly_harsh", "slightly harsh direct on-camera flash lighting"),
    ("overcast", "soft overcast daylight"),
    ("evening_lamp", "warm evening lamp lighting"),
]

FRAMING = [
    ("centered", "centered in frame, straight-on eye-level angle"),
    ("slightly_off_center", "slightly off-center in frame, straight-on eye-level angle"),
    ("slightly_low_angle", "centered in frame, slightly low camera angle"),
    ("slightly_high_angle", "centered in frame, slightly high camera angle"),
]


def sample_environment(rng):
    bg_key, bg_phrase = rng.choice(BACKGROUNDS)
    li_key, li_phrase = rng.choice(LIGHTING)
    fr_key, fr_phrase = rng.choice(FRAMING)
    return {
        "background_phrase": bg_phrase, "requested_background": bg_key,
        "lighting_phrase": li_phrase, "requested_lighting": li_key,
        "framing_phrase": fr_phrase, "requested_framing": fr_key,
    }


# Hair colour/length variety for OUTFIT shots only -- Bucket B, unlabelled,
# so grooming quality can't leak into the outfit score (a great outfit
# photographed with unstyled hair must still score high on Outfit).
OUTFIT_HAIR_DESCRIPTORS = [
    "short black hair", "long brown hair tied back", "shoulder-length blonde hair",
    "short red hair", "long dark hair worn loose", "cropped grey hair",
    "medium-length brown hair", "long black hair in a ponytail", "short auburn hair",
    "curly dark hair", "straight platinum blonde hair", "short salt-and-pepper hair",
]


def sample_outfit_hair(rng):
    return rng.choice(OUTFIT_HAIR_DESCRIPTORS)


# ==========================================================================
# BUCKET B -- COLOUR (restricted). Palette of named families with anchor
# RGB values used later by extract_measured_labels.py to snap a measured
# pixel colour (in CIELab) back to a family name. Sampled independently of
# tier/score -- polished must not drift toward "beige = effort". Recorded
# as meta (requested_*_color), never trained: colour comes free on-device
# from pose-derived regions at runtime, so training a classifier for it
# wastes model capacity.
# ==========================================================================
COLOR_ANCHORS_RGB = {
    "black": (20, 20, 20),
    "white": (240, 240, 235),
    "grey": (130, 130, 132),
    "navy": (28, 38, 68),
    "blue": (52, 92, 162),
    "red": (172, 32, 32),
    "green": (52, 110, 62),
    "beige": (212, 192, 156),
    "brown": (92, 62, 40),
    "burgundy": (110, 26, 46),
}
ALL_COLOR_NAMES = list(COLOR_ANCHORS_RGB.keys())

# Per-garment (by label_key) colour restrictions. Prevents implausible
# pairings like a "burgundy denim jacket". Garments not listed here may be
# any colour in the palette.
GARMENT_COLOR_RESTRICTIONS = {
    "denim_jacket": ["blue", "navy", "black", "grey", "white"],
    "denim_jeans": ["blue", "navy", "black", "grey", "white"],
    "leather_dress_shoes": ["black", "brown"],
    "oxford_shoes": ["black", "brown"],
    "heeled_pumps": ["black", "brown", "burgundy"],
    "pointed_flats": ["black", "brown", "burgundy"],
    "ankle_boots": ["black", "brown"],
}


def sample_color(label_key, rng):
    allowed = GARMENT_COLOR_RESTRICTIONS.get(label_key, ALL_COLOR_NAMES)
    return rng.choice(allowed)


# ==========================================================================
# BUCKET A -- GARMENT SLOTS. Per gender, four independent slots. Each pool
# entry is (label_key, prompt_phrase, formality) with formality 0-3
# (0=very casual .. 3=formal). "mid" (outer layer) includes a `none` entry.
# Women's `upper` includes `dress`; when picked, `lower` is forced to None
# (see sample_outfit()) so we never generate "a dress and jeans".
# ==========================================================================
MEN_UPPER = [
    ("tank_top", "a tank top", 0),
    ("graphic_tee", "a graphic t-shirt", 0),
    ("plain_crewneck_tee", "a plain crew-neck t-shirt", 0),
    ("henley", "a henley shirt", 1),
    ("polo_shirt", "a polo shirt", 1),
    ("flannel_shirt", "a flannel shirt", 1),
    ("thermal_henley", "a waffle-knit thermal henley", 1),
    ("crewneck_sweater", "a crewneck sweater", 2),
    ("turtleneck_sweater", "a fine-knit turtleneck sweater", 2),
    ("oxford_button_down", "an oxford button-down shirt", 2),
    ("chambray_shirt", "a chambray button-up shirt", 2),
    ("dress_shirt", "a dress shirt", 3),
]
MEN_MID = [
    ("none", "", 0),
    ("hoodie", "a pullover hoodie", 0),
    ("denim_jacket", "a denim jacket", 1),
    ("bomber_jacket", "a bomber jacket", 1),
    ("cardigan", "a cardigan", 2),
    ("blazer", "a tailored blazer", 3),
]
MEN_LOWER = [
    ("sweatpants", "sweatpants", 0),
    ("athletic_shorts", "athletic shorts", 0),
    ("cargo_shorts", "cargo shorts", 0),
    ("denim_jeans", "denim jeans", 1),
    ("joggers", "tapered joggers", 1),
    ("cargo_pants", "cargo pants", 1),
    ("chino_pants", "chino pants", 2),
    ("corduroy_pants", "corduroy pants", 2),
    ("tailored_trousers", "tailored trousers", 3),
    ("dress_pants", "dress pants", 3),
]
MEN_FOOTWEAR = [
    ("slides", "slide sandals", 0),
    ("flip_flops", "flip flops", 0),
    ("canvas_sneakers", "canvas sneakers", 1),
    ("running_shoes", "running shoes", 1),
    ("leather_sneakers", "minimalist leather sneakers", 2),
    ("suede_desert_boots", "suede desert boots", 2),
    ("leather_dress_shoes", "leather dress shoes", 3),
    ("oxford_shoes", "oxford shoes", 3),
]

WOMEN_UPPER = [
    ("tank_top", "a tank top", 0),
    ("graphic_tee", "a graphic t-shirt", 0),
    ("casual_camisole", "a casual camisole", 0),
    ("henley", "a henley top", 1),
    ("casual_blouse", "a casual blouse", 1),
    ("wrap_top", "a wrap top", 1),
    ("silk_blouse", "a silk blouse", 2),
    ("fitted_sweater", "a fitted sweater", 2),
    ("turtleneck_sweater", "a fine-knit turtleneck sweater", 2),
    ("tailored_blouse", "a tailored blouse", 3),
    ("dress_shirt", "a fitted dress shirt", 3),
    ("dress", "a dress", 3),
]
WOMEN_MID = [
    ("none", "", 0),
    ("hoodie", "a pullover hoodie", 0),
    ("denim_jacket", "a denim jacket", 1),
    ("cropped_jacket", "a cropped jacket", 1),
    ("cardigan", "a cardigan", 2),
    ("blazer", "a tailored blazer", 3),
]
WOMEN_LOWER = [
    ("sweatpants", "sweatpants", 0),
    ("leggings", "leggings", 0),
    ("denim_shorts", "denim shorts", 0),
    ("denim_jeans", "denim jeans", 1),
    ("joggers", "tapered joggers", 1),
    ("casual_skirt", "a casual skirt", 1),
    ("wide_leg_trousers", "wide-leg trousers", 2),
    ("midi_skirt", "a midi skirt", 2),
    ("tailored_trousers", "tailored trousers", 3),
    ("pencil_skirt", "a pencil skirt", 3),
]
WOMEN_FOOTWEAR = [
    ("slides", "slide sandals", 0),
    ("flip_flops", "flip flops", 0),
    ("sneakers", "casual sneakers", 1),
    ("ballet_flats", "ballet flats", 1),
    ("ankle_boots", "ankle boots", 2),
    ("block_heels", "block-heel sandals", 2),
    ("pointed_flats", "pointed-toe flats", 3),
    ("heeled_pumps", "heeled pumps", 3),
]

SLOTS_BY_GENDER = {
    "man": {"upper": MEN_UPPER, "mid": MEN_MID, "lower": MEN_LOWER, "footwear": MEN_FOOTWEAR},
    "woman": {"upper": WOMEN_UPPER, "mid": WOMEN_MID, "lower": WOMEN_LOWER, "footwear": WOMEN_FOOTWEAR},
}

PATTERNS = ["solid", "striped", "checked", "printed"]
UPPER_PATTERN_WEIGHTS = [0.55, 0.16, 0.15, 0.14]
LOWER_PATTERN_WEIGHTS = [0.75, 0.08, 0.10, 0.07]


def _weighted_choice(rng, options, weights):
    return rng.choices(options, weights=weights, k=1)[0]


def _formality_bucket_pick(pool, target, rng):
    """Coherent sampling for one slot: 75% of the time, restrict to entries
    within +/-1 of `target` formality (weighting an EXACT match 3:1 over a
    neighbour); 25% of the time, pick from the whole pool unrestricted.
    Uniform independent sampling makes `formal` (needs all four slots high
    at once) vanishingly rare and produces incoherent outfits (dress shoes
    with track pants) -- see build spec section 3."""
    if rng.random() < 0.75:
        lo, hi = target - 1, target + 1
        band = [e for e in pool if lo <= e[2] <= hi]
        weights = [3 if e[2] == target else 1 for e in band]
        return rng.choices(band, weights=weights, k=1)[0]
    return rng.choice(pool)


def sample_outfit(gender, rng):
    """Independently samples the garment-composition axis: which item in
    which slot, its pattern, its colour. Returns a dict with per-slot
    entries plus the derived `formality` tier. Does NOT touch effort/tier
    at all -- garment identity and effort condition are fully decoupled
    (build spec section 2)."""
    slots = SLOTS_BY_GENDER[gender]
    target = rng.randint(0, 3)

    upper = _formality_bucket_pick(slots["upper"], target, rng)
    mid = _formality_bucket_pick(slots["mid"], target, rng)
    footwear = _formality_bucket_pick(slots["footwear"], target, rng)

    is_dress = upper[0] == "dress"
    if is_dress:
        lower = None  # force lower=none -- never "a dress and jeans"
    else:
        lower = _formality_bucket_pick(slots["lower"], target, rng)

    upper_pattern = _weighted_choice(rng, PATTERNS, UPPER_PATTERN_WEIGHTS)
    lower_pattern = _weighted_choice(rng, PATTERNS, LOWER_PATTERN_WEIGHTS) if lower else "none"

    upper_color = sample_color(upper[0], rng)
    mid_color = sample_color(mid[0], rng) if mid[0] != "none" else None
    lower_color = sample_color(lower[0], rng) if lower else None
    footwear_color = sample_color(footwear[0], rng)

    # Formality is DERIVED from the mean formality of the slots actually
    # WORN (mid=none and a dress's forced lower=none are excluded), using
    # THRESHOLDS rather than round() -- banker's rounding on round() would
    # misclassify e.g. a mean of 1.5 as something other than smart_casual.
    worn_formalities = [upper[2]]
    if mid[0] != "none":
        worn_formalities.append(mid[2])
    if lower:
        worn_formalities.append(lower[2])
    worn_formalities.append(footwear[2])
    mean_formality = sum(worn_formalities) / len(worn_formalities)
    formality = formality_tier(mean_formality)

    return {
        "upper": upper, "mid": mid, "lower": lower, "footwear": footwear,
        "upper_pattern": upper_pattern, "lower_pattern": lower_pattern,
        "upper_color": upper_color, "mid_color": mid_color,
        "lower_color": lower_color, "footwear_color": footwear_color,
        "formality": formality, "mean_formality": mean_formality,
    }


def formality_tier(mean_formality):
    """Thresholds, NOT round() -- see sample_outfit()'s docstring."""
    if mean_formality < 1.0:
        return "casual"
    elif mean_formality < 2.0:
        return "smart_casual"
    elif mean_formality < 2.75:
        return "business"
    return "formal"


FORMALITY_CLASSES = ["casual", "smart_casual", "business", "formal"]

# ==========================================================================
# BUCKET A -- OUTFIT EFFORT CONDITION (decoupled from garment identity).
# Supplies upper_mod / lower_mod / mid short phrase / footwear condition /
# overall styling phrases, plus the ordinal severity + binary
# polish-achieved labels. severity_for_tier()/positive_for_tier() above do
# the actual label math; these dicts are pure text.
# ==========================================================================
FABRIC_CONDITION_PHRASE = {
    "flaw_severe": "deeply wrinkled with visible crease lines, plus a coin-sized dried stain near the hem",
    "flaw_mild": "noticeably creased, as if pulled straight out of a laundry hamper",
    "average": "clean and reasonably smooth, unremarkable",
    "polished": "crisp and freshly pressed, with no visible wrinkles",
}
FIT_CONDITION_PHRASE = {
    ("flaw_severe", "baggy"): "hanging noticeably loose and bunching, several sizes too large",
    ("flaw_severe", "tight"): "uncomfortably tight and straining across the body, clearly a size too small",
    ("flaw_mild", "baggy"): "a bit loose through the body, clearly not tailored",
    ("flaw_mild", "tight"): "a little snug through the body, clearly not tailored",
    ("average", "baggy"): "a standard, unremarkable fit, neither baggy nor tight",
    ("average", "tight"): "a standard, unremarkable fit, neither baggy nor tight",
    ("polished", "baggy"): "tailored to the body with clean, flattering proportions",
    ("polished", "tight"): "tailored to the body with clean, flattering proportions",
}
FOOTWEAR_CONDITION_PHRASE = {
    "flaw_severe": "visibly scuffed and dirty, with the laces untied",
    "flaw_mild": "a bit worn and scuffed",
    "average": "plain and clean, unremarkable",
    "polished": "clean and freshly polished",
}
STYLING_PHRASE = {
    "flaw_severe": "the whole outfit looking sloppy and thrown-together, with no thought given to how the pieces work together",
    "flaw_mild": "the outfit looking like not much thought went into how it was put together",
    "average": "an ordinary, unremarkable overall look, neither sloppy nor especially put-together",
    "polished": "the whole outfit looking sharp and intentionally put together",
}


def sample_fit_direction(rng):
    return "baggy" if rng.random() < 0.5 else "tight"


def build_outfit_condition(tier, rng):
    """Returns (mods dict of text fragments, labels dict) for one outfit
    image's effort condition -- independent of which garments were picked
    by sample_outfit()."""
    direction = sample_fit_direction(rng)
    fit_phrase = FIT_CONDITION_PHRASE[(tier, direction)]
    fabric_phrase = FABRIC_CONDITION_PHRASE[tier]
    footwear_phrase = FOOTWEAR_CONDITION_PHRASE[tier]
    styling_phrase = STYLING_PHRASE[tier]

    severity = severity_for_tier(tier)
    positive = positive_for_tier(tier)
    labels = {
        "fit_baggy": severity if direction == "baggy" else 0,
        "fit_tight": severity if direction == "tight" else 0,
        "fit_tailored": positive,
        "fabric_wrinkled": severity,
        "fabric_crisp": positive,
        "footwear_worn": severity,
        "footwear_polished": positive,
        "styling_sloppy": severity,
        "styling_sharp": positive,
    }
    mods = {
        "upper_mod": f"{fabric_phrase}, the fit {fit_phrase}",
        "lower_mod": f"{fabric_phrase}, the fit {fit_phrase}",
        "mid_short_phrase": {"flaw_severe": "looking worn and neglected", "flaw_mild": "a bit rumpled",
                              "average": "in ordinary condition", "polished": "crisp and well-kept"}[tier],
        "footwear_mod": footwear_phrase,
        "styling": styling_phrase,
    }
    return mods, labels


# ==========================================================================
# BUCKET A -- GROOMING COMPOSITION AXES (facial_hair_style / hair_length /
# makeup_style) -- sampled INDEPENDENTLY of effort tier, so e.g. a full
# beard or heavy makeup can appear at any effort level. Zeroed-out
# incompatible labels: clean_shaven forces facial_hair_untidy/groomed=0;
# makeup_style=none forces makeup_uneven/flawless=0 (there is nothing to be
# untidy or flawless ABOUT).
# ==========================================================================
FACIAL_HAIR_STYLES = ["clean_shaven", "stubble", "short_beard", "full_beard", "moustache"]
FACIAL_HAIR_STYLE_WORDS = {
    "stubble": "light stubble", "short_beard": "a short beard",
    "full_beard": "a full beard", "moustache": "a moustache",
}
FACIAL_HAIR_CONDITION_PHRASE = {
    "flaw_severe": "growing in patchy, uneven clumps with bare skin gaps between patches, unkempt",
    "flaw_mild": "growing in with an uneven length and no defined edge or lineup",
    "average": "an ordinary, unremarkable trim with no defined lineup",
    "polished": "precisely trimmed with crisp, clean edges",
}

HAIR_LENGTHS = ["buzz_cut", "short", "medium", "long"]
HAIR_LENGTH_WORDS = {
    "buzz_cut": "very short buzzed hair", "short": "short hair",
    "medium": "medium-length hair", "long": "long hair",
}
MEN_HAIR_CONDITION_PHRASE = {
    "flaw_severe": "visibly greasy and unwashed, strands clumped together with an oily sheen, flattened against the scalp",
    "flaw_mild": "flat and unbrushed with no styling, clearly not styled today",
    "average": "lying naturally with no product, neither messy nor styled",
    "polished": "neatly styled with a touch of product, clean and intentional",
}
WOMEN_HAIR_CONDITION_PHRASE = {
    "flaw_severe": "visibly greasy and unwashed, an oily sheen at the roots, strands clumped and flattened",
    "flaw_mild": "tangled and unbrushed, with visible frizz sticking out unevenly",
    "average": "worn plainly with no product",
    "polished": "smooth and styled with visible shine",
}

MAKEUP_STYLES = ["none", "minimal", "everyday", "full"]
MAKEUP_STYLE_WORDS = {
    "minimal": "minimal makeup", "everyday": "everyday makeup", "full": "a full face of makeup",
}
MAKEUP_CONDITION_PHRASE = {
    "flaw_severe": "applied unevenly with two clashing colors and visibly smudged edges",
    "flaw_mild": "slightly smudged and unevenly blended",
    "average": "applied simply with no particular effort",
    "polished": "flawlessly blended with precise, clean application",
}

SKIN_CONDITION_PHRASE = {
    "flaw_severe": "dry, visibly flaking skin with rough patches on the forehead and around the nose",
    "flaw_mild": "a dull, uneven skin tone with visible tiredness under the eyes",
    "average": "ordinary skin texture with a few small visible pores",
    "polished": "smooth, evenly hydrated skin with a healthy natural glow",
}
EYEBROWS_CONDITION_PHRASE = {
    "flaw_severe": "thick and completely unshaped, growing together above the nose",
    "flaw_mild": "a few stray long hairs among otherwise ordinary eyebrows",
    "average": "natural and unshaped, neither groomed nor unkempt",
    "polished": "neatly shaped with clean, defined edges",
}


def build_hair(gender, tier, rng):
    length = rng.choice(HAIR_LENGTHS)
    condition = (MEN_HAIR_CONDITION_PHRASE if gender == "man" else WOMEN_HAIR_CONDITION_PHRASE)[tier]
    phrase = f"{HAIR_LENGTH_WORDS[length]}, {condition}"
    labels = {
        "hair_untidy": severity_for_tier(tier),
        "hair_styled": positive_for_tier(tier),
        "hair_length": length,
    }
    return phrase, labels


def build_facial_hair(tier, rng):
    style = rng.choice(FACIAL_HAIR_STYLES)
    if style == "clean_shaven":
        phrase = "completely clean-shaven with smooth, even skin"
        labels = {"facial_hair_style": style, "facial_hair_untidy": 0, "facial_hair_groomed": 0}
    else:
        condition = FACIAL_HAIR_CONDITION_PHRASE[tier]
        phrase = f"{FACIAL_HAIR_STYLE_WORDS[style]}, {condition}"
        labels = {
            "facial_hair_style": style,
            "facial_hair_untidy": severity_for_tier(tier),
            "facial_hair_groomed": positive_for_tier(tier),
        }
    return phrase, labels


def build_makeup(tier, rng):
    style = rng.choice(MAKEUP_STYLES)
    if style == "none":
        phrase = "no makeup, a bare face"
        labels = {"makeup_style": style, "makeup_uneven": 0, "makeup_flawless": 0}
    else:
        condition = MAKEUP_CONDITION_PHRASE[tier]
        phrase = f"{MAKEUP_STYLE_WORDS[style]}, {condition}"
        labels = {
            "makeup_style": style,
            "makeup_uneven": severity_for_tier(tier),
            "makeup_flawless": positive_for_tier(tier),
        }
    return phrase, labels


def build_skin(tier, rng):
    phrase = SKIN_CONDITION_PHRASE[tier]
    labels = {"skin_neglected": severity_for_tier(tier), "skin_healthy": positive_for_tier(tier)}
    return phrase, labels


def build_eyebrows(tier, rng):
    phrase = EYEBROWS_CONDITION_PHRASE[tier]
    labels = {"eyebrows_unkempt": severity_for_tier(tier), "eyebrows_groomed": positive_for_tier(tier)}
    return phrase, labels


# ==========================================================================
# NEGATIVE PROMPT / INFERENCE DEFAULTS (build spec section 8/13). Flaw
# words are deliberately absent from the negative prompt -- negating
# "wrinkled" globally would destroy the entire flaw tier. No LoRA / no
# Lightning distillation anywhere in this pipeline (verified absent from
# the archived code too) -- few-step distilled models lose prompt
# adherence worst on negative attributes first.
# ==========================================================================
NEGATIVE_PROMPT = (
    "deformed hands, extra fingers, fused fingers, extra limbs, missing limbs, "
    "multiple people, cropped head, cropped feet, out of frame, "
    "text, watermark, logo, signature, cartoon, 3d render, plastic skin, airbrushed skin, "
    "blurry, low quality, distorted proportions"
)
TRUE_CFG_SCALE = 4.0
NUM_INFERENCE_STEPS_FULL = 40
NUM_INFERENCE_STEPS_TEST = 28

GROOMING_RESOLUTION = (1024, 1024)          # square: face detail matters most
OUTFIT_RESOLUTION = (768, 1024)             # portrait: a square crop cuts feet off

# ==========================================================================
# IMAGE BUDGET (build spec section 12)
# ==========================================================================
CATEGORY_COUNTS = {
    "Men_Grooming": 6000,
    "Women_Grooming": 6000,
    "Men_Outfit": 8000,
    "Women_Outfit": 8000,
}
CATEGORY_GENDER = {"Men_Grooming": "man", "Women_Grooming": "woman", "Men_Outfit": "man", "Women_Outfit": "woman"}
CATEGORY_KIND = {"Men_Grooming": "grooming", "Women_Grooming": "grooming", "Men_Outfit": "outfit", "Women_Outfit": "outfit"}
GROOMING_CATEGORIES = ["Men_Grooming", "Women_Grooming"]
OUTFIT_CATEGORIES = ["Men_Outfit", "Women_Outfit"]
ALL_CATEGORIES = ["Men_Grooming", "Women_Grooming", "Men_Outfit", "Women_Outfit"]


def tier_counts_for_total(total):
    """Same score-band balance as the archived taxonomy: roughly equal
    thirds across Flaw(severe+mild)/Average/Polished, with Flaw internally
    split evenly into severe/mild. Any integer-division remainder is added
    to `average` (harmless, keeps thin-class risk on the flaw tiers off the
    table rather than shaving from them)."""
    base = total // 6
    counts = {
        "flaw_severe": base,
        "flaw_mild": base,
        "average": base * 2,
        "polished": base * 2,
    }
    remainder = total - sum(counts.values())
    counts["average"] += remainder
    assert sum(counts.values()) == total
    return counts


# ==========================================================================
# LABEL SCHEMA -- generated FROM the taxonomy above, never hand-duplicated
# (build spec section 9). Types: regression (score, weight 1.0), ordinal
# (3 levels 0/1/2, weight 0.3), categorical (weight 0.3; formality 0.5),
# meta (not trained -- provenance / pixel-snapping only).
# ==========================================================================
def _slot_classes(pool):
    return [e[0] for e in pool]


def get_label_schema(category):
    gender = CATEGORY_GENDER[category]
    kind = CATEGORY_KIND[category]
    schema = [{"name": "score", "type": "regression", "loss_weight": 1.0}]

    if kind == "grooming":
        schema += [
            {"name": "hair_untidy", "type": "ordinal", "levels": 3, "loss_weight": 0.3},
            {"name": "hair_styled", "type": "categorical", "classes": ["0", "1"], "loss_weight": 0.3},
            {"name": "hair_length", "type": "categorical", "classes": HAIR_LENGTHS, "loss_weight": 0.3},
            {"name": "skin_neglected", "type": "ordinal", "levels": 3, "loss_weight": 0.3},
            {"name": "skin_healthy", "type": "categorical", "classes": ["0", "1"], "loss_weight": 0.3},
            {"name": "eyebrows_unkempt", "type": "ordinal", "levels": 3, "loss_weight": 0.3},
            {"name": "eyebrows_groomed", "type": "categorical", "classes": ["0", "1"], "loss_weight": 0.3},
        ]
        if gender == "man":
            schema += [
                {"name": "facial_hair_style", "type": "categorical", "classes": FACIAL_HAIR_STYLES, "loss_weight": 0.3},
                {"name": "facial_hair_untidy", "type": "ordinal", "levels": 3, "loss_weight": 0.3},
                {"name": "facial_hair_groomed", "type": "categorical", "classes": ["0", "1"], "loss_weight": 0.3},
            ]
        else:
            schema += [
                {"name": "makeup_style", "type": "categorical", "classes": MAKEUP_STYLES, "loss_weight": 0.3},
                {"name": "makeup_uneven", "type": "ordinal", "levels": 3, "loss_weight": 0.3},
                {"name": "makeup_flawless", "type": "categorical", "classes": ["0", "1"], "loss_weight": 0.3},
            ]
        schema += [
            {"name": "requested_background", "type": "meta"},
            {"name": "requested_lighting", "type": "meta"},
            {"name": "requested_framing", "type": "meta"},
        ]
    else:
        slots = SLOTS_BY_GENDER[gender]
        schema += [
            {"name": "upper_type", "type": "categorical", "classes": _slot_classes(slots["upper"]), "loss_weight": 0.3},
            {"name": "upper_pattern", "type": "categorical", "classes": PATTERNS, "loss_weight": 0.3},
            {"name": "mid_type", "type": "categorical", "classes": _slot_classes(slots["mid"]), "loss_weight": 0.3},
            {"name": "lower_type", "type": "categorical", "classes": _slot_classes(slots["lower"]) + ["none"], "loss_weight": 0.3},
            {"name": "lower_pattern", "type": "categorical", "classes": PATTERNS + ["none"], "loss_weight": 0.3},
            {"name": "footwear_type", "type": "categorical", "classes": _slot_classes(slots["footwear"]), "loss_weight": 0.3},
            {"name": "formality", "type": "categorical", "classes": FORMALITY_CLASSES, "loss_weight": 0.5},
            {"name": "fit_baggy", "type": "ordinal", "levels": 3, "loss_weight": 0.3},
            {"name": "fit_tight", "type": "ordinal", "levels": 3, "loss_weight": 0.3},
            {"name": "fit_tailored", "type": "categorical", "classes": ["0", "1"], "loss_weight": 0.3},
            {"name": "fabric_wrinkled", "type": "ordinal", "levels": 3, "loss_weight": 0.3},
            {"name": "fabric_crisp", "type": "categorical", "classes": ["0", "1"], "loss_weight": 0.3},
            {"name": "footwear_worn", "type": "ordinal", "levels": 3, "loss_weight": 0.3},
            {"name": "footwear_polished", "type": "categorical", "classes": ["0", "1"], "loss_weight": 0.3},
            {"name": "styling_sloppy", "type": "ordinal", "levels": 3, "loss_weight": 0.3},
            {"name": "styling_sharp", "type": "categorical", "classes": ["0", "1"], "loss_weight": 0.3},
            {"name": "requested_upper_color", "type": "meta"},
            {"name": "requested_mid_color", "type": "meta"},
            {"name": "requested_lower_color", "type": "meta"},
            {"name": "requested_footwear_color", "type": "meta"},
            {"name": "requested_background", "type": "meta"},
            {"name": "requested_lighting", "type": "meta"},
            {"name": "requested_framing", "type": "meta"},
            {"name": "requested_hair_desc", "type": "meta"},
        ]
    return schema


def schema_columns(category):
    """Column order for this category's CSV: filename/category/tier first,
    then every schema field in schema order."""
    return ["filename", "category", "tier"] + [f["name"] for f in get_label_schema(category)]
