"""
prompt_builder.py -- composes ONE (prompt, resolution, label-row) triple per
image from taxonomy.py's independently-sampled axes. smoke_test.py,
validation_sweep.py, and full_run.py all call build_task() so every script
shares byte-identical prompt construction and label-row shape -- nothing
downstream hand-duplicates a template.

PROMPT STYLE: clause-structured, one labelled body-region/attribute per
line, terminated with a period. Qwen-Image-2512 uses Qwen2.5-VL-7B as its
text encoder (not a 77-token CLIP encoder), so long, clearly delimited
clauses are the right shape here -- unlike the archived FLUX pipeline
(see ML/archive/dataset_generator_v7/PLAN.md's CLIP-truncation postmortem),
there's no hard token budget forcing everything into one dense paragraph.
Each clause still gets a SHORT tier phrase rather than the full effort-mod
text repeated verbatim across clauses -- repetition wastes encoder
attention on restating the same idea instead of adding new information.

BODY-SHAPE LOCK (outfit prompts only): ML/README.md documents a real,
previously-caught bug ("taxonomy v4" note) where "polished" styling
language visibly slimmed heavy-set/plus-size/curvy identities relative to
their own flaw-tier render, because styling text reshapes body proportions
through cross-attention even with pose otherwise fixed. The archived fix
was a body-preservation clause placed early (not trailing) in the prompt,
restating the identity's build. That clause has no equivalent axis in
taxonomy.py (build is baked into identity_str only once), so it is added
here, directly after the identity clause, before any effort/styling
language -- this is a deviation from the literal template in the build
spec, added specifically because this failure mode is real and documented,
not hypothetical.
"""
import taxonomy as tx


# --------------------------------------------------------------------------
# Small text helpers
# --------------------------------------------------------------------------
def _strip_article(phrase):
    """'a graphic t-shirt' -> 'graphic t-shirt' (so a pattern/colour can be
    inserted between the article and the noun)."""
    for art in ("an ", "a "):
        if phrase.startswith(art):
            return phrase[len(art):]
    return phrase


def _garment_clause(pattern, color, phrase):
    """pattern='striped', color='navy', phrase='a graphic t-shirt' ->
    'a striped navy graphic t-shirt'. Article presence is taken from the
    ORIGINAL phrase, not assumed -- taxonomy.py's lower-body pool mixes
    singular countable nouns ('a midi skirt') with plural/mass nouns with
    no article ('cargo shorts', 'denim jeans', 'tailored trousers'), and
    forcing 'a' onto the latter produces ungrammatical prompts like 'a
    cargo shorts' that a real LLM text encoder (Qwen2.5-VL, not a bag-of-
    tokens CLIP encoder) is more likely to stumble on than ignore."""
    has_article = phrase.startswith("a ") or phrase.startswith("an ")
    noun = _strip_article(phrase)
    if has_article:
        return f"a {pattern} {color} {noun}"
    return f"{pattern} {color} {noun}"


def _plain_garment_clause(color, phrase):
    """No pattern axis (mid layer, footwear): 'a navy pullover hoodie'."""
    noun = _strip_article(phrase)
    return f"a {color} {noun}"


BODY_LOCK = (
    "body proportions unchanged from their natural {build} figure, "
    "not slimmer or heavier than that"
)


# --------------------------------------------------------------------------
# GROOMING
# --------------------------------------------------------------------------
def build_grooming_task(category, tier, rng):
    gender = tx.CATEGORY_GENDER[category]
    identity = tx.make_identity(gender, rng)
    ident_str = tx.identity_str(gender, identity)
    env = tx.sample_environment(rng)

    hair_phrase, hair_labels = tx.build_hair(gender, tier, rng)
    skin_phrase, skin_labels = tx.build_skin(tier, rng)
    brow_phrase, brow_labels = tx.build_eyebrows(tier, rng)

    row = {"score": tx.sample_score(tier, rng)}
    row.update(hair_labels)
    row.update(skin_labels)
    row.update(brow_labels)

    if gender == "man":
        facial_phrase, facial_labels = tx.build_facial_hair(tier, rng)
        row.update(facial_labels)
        third_clause = f"Facial hair: {facial_phrase}."
    else:
        makeup_phrase, makeup_labels = tx.build_makeup(tier, rng)
        row.update(makeup_labels)
        third_clause = f"Makeup: {makeup_phrase}."

    row.update({
        "requested_background": env["requested_background"],
        "requested_lighting": env["requested_lighting"],
        "requested_framing": env["requested_framing"],
    })

    prompt = (
        f"A head-and-shoulders portrait photograph of a {ident_str}, "
        f"facing the camera directly, looking at the camera.\n"
        f"Hair: {hair_phrase}.\n"
        f"{third_clause}\n"
        f"Skin: {skin_phrase}.\n"
        f"Eyebrows: {brow_phrase}.\n"
        f"Setting: {env['background_phrase']}, {env['lighting_phrase']}, {env['framing_phrase']}.\n"
        f"Photorealistic candid photograph, natural skin texture, sharp focus, "
        f"85mm lens, head and shoulders in frame."
    )
    return {"prompt": prompt, "resolution": tx.GROOMING_RESOLUTION, "row": row}


# --------------------------------------------------------------------------
# OUTFIT
# --------------------------------------------------------------------------
def build_outfit_task(category, tier, rng):
    gender = tx.CATEGORY_GENDER[category]
    identity = tx.make_identity(gender, rng)
    ident_str = tx.identity_str(gender, identity)
    env = tx.sample_environment(rng)
    hair_desc = tx.sample_outfit_hair(rng)

    outfit = tx.sample_outfit(gender, rng)
    mods, cond_labels = tx.build_outfit_condition(tier, rng)

    row = {"score": tx.sample_score(tier, rng)}
    row.update(cond_labels)
    row.update({
        "upper_type": outfit["upper"][0],
        "upper_pattern": outfit["upper_pattern"],
        "mid_type": outfit["mid"][0],
        "lower_type": outfit["lower"][0] if outfit["lower"] else "none",
        "lower_pattern": outfit["lower_pattern"],
        "footwear_type": outfit["footwear"][0],
        "formality": outfit["formality"],
        "requested_upper_color": outfit["upper_color"],
        "requested_mid_color": outfit["mid_color"] if outfit["mid_color"] else "none",
        "requested_lower_color": outfit["lower_color"] if outfit["lower_color"] else "none",
        "requested_footwear_color": outfit["footwear_color"],
        "requested_background": env["requested_background"],
        "requested_lighting": env["requested_lighting"],
        "requested_framing": env["requested_framing"],
        "requested_hair_desc": hair_desc,
    })

    body_lock = BODY_LOCK.format(build=identity["build"])
    upper_clause = _garment_clause(outfit["upper_pattern"], outfit["upper_color"], outfit["upper"][1])

    lines = [
        f"A full-body photograph of a {ident_str}, {body_lock}, with {hair_desc}, "
        f"standing facing the camera with the whole body from head to shoes visible.",
        f"Upper body: wearing {upper_clause}, {mods['upper_mod']}.",
    ]

    if outfit["mid"][0] != "none":
        mid_clause = _plain_garment_clause(outfit["mid_color"], outfit["mid"][1])
        lines.append(f"Outer layer: wearing {mid_clause}, {mods['mid_short_phrase']}.")

    if outfit["lower"] is not None:
        lower_clause = _garment_clause(outfit["lower_pattern"], outfit["lower_color"], outfit["lower"][1])
        lines.append(f"Lower body: wearing {lower_clause}, {mods['lower_mod']}.")

    footwear_noun = _strip_article(outfit["footwear"][1])
    lines.append(f"Footwear: {outfit['footwear_color']} {footwear_noun}, {mods['footwear_mod']}.")
    lines.append(f"Overall styling: {mods['styling']}.")
    lines.append(f"Setting: {env['background_phrase']}, {env['lighting_phrase']}, {env['framing_phrase']}.")
    lines.append(
        "Photorealistic candid photograph, natural skin texture, sharp focus, "
        "85mm lens, full body in frame."
    )
    prompt = "\n".join(lines)
    return {"prompt": prompt, "resolution": tx.OUTFIT_RESOLUTION, "row": row}


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------
def build_task(category, tier, rng):
    """Returns {"prompt": str, "resolution": (w, h), "row": {field_name: value, ...}}
    where row's keys are exactly the non-meta+meta field names from
    taxonomy.get_label_schema(category) (excluding filename/category/tier,
    which the caller fills in once a filename is assigned)."""
    kind = tx.CATEGORY_KIND[category]
    if kind == "grooming":
        return build_grooming_task(category, tier, rng)
    return build_outfit_task(category, tier, rng)


def row_for_csv(category, tier, filename, task):
    """Assemble the full CSV row dict (filename/category/tier + every
    schema column in schema order) for one generated image."""
    row = {"filename": filename, "category": category, "tier": tier}
    for field in tx.get_label_schema(category):
        name = field["name"]
        row[name] = task["row"].get(name, "")
    return row
