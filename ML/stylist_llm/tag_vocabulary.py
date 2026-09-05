"""
tag_vocabulary.py -- the ONE place that translates the real vision model's
label schema (ML/vision/dataset_synthetic/taxonomy.py) into the tag-string
prompt format the stylist LLM consumes at both training and inference time.

WHY THIS FILE EXISTS: the original brief for this model invented its own
tag format ("collar: unbuttoned_spread", "priority_defect: blazer_creasing")
that does not correspond to any field the real vision model actually
produces. Checked against taxonomy.get_label_schema() directly: outfit
categories have upper_type/upper_pattern/mid_type/lower_type/lower_pattern/
footwear_type/formality (categorical), fit_baggy/fit_tight/fabric_wrinkled/
footwear_worn/styling_sloppy (ordinal 0-2 severity), no free-text collar
state and no generic "defect" field. If the LLM were trained on the brief's
invented vocabulary, it would never see a matching real prompt once the
actual on-device vision model is generating the tags. This module is the
fix: every tag string below is built FROM taxonomy.py's real field names
and real class lists, never hand-duplicated.

ISOLATION: this is a deliberate, ONE-DIRECTIONAL exception to the
stylist_llm/ <-> dataset_synthetic/ isolation described in
ML/stylist_llm/PLAN.md. This file imports FROM taxonomy.py
(read-only -- it is the upstream interface contract, not a training
dependency). Nothing in dataset_synthetic/ imports anything from
stylist_llm/, and no data/checkpoints/config are shared beyond this one
translation layer.

OCCASION is NOT a vision-model output -- taxonomy.py has no such axis at
all. In the real app it comes from the user's own selection in the UI (see
the original brief's Swift snippet: `userSession.selectedOccasion`), so
OCCASIONS is defined here as this pipeline's own input, not imported.

GROOMING TAGS: the original brief only showed an outfit example. The real
vision pipeline scores grooming (hair/skin/eyebrows/facial-hair-or-makeup)
as an entirely separate category with its own photo (see ML/README.md's
"oval for grooming, rectangle for outfit" framing), so this module builds
a symmetric grooming tag format too -- the on-device stylist should be able
to advise on a grooming photo, not just an outfit one.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vision" / "dataset_synthetic"))
import taxonomy as vision_tx  # noqa: E402  (see module docstring -- deliberate, read-only)


# --------------------------------------------------------------------------
# Apple Vision native signals (posture, lighting, color harmony) --
# derived from iOS Vision framework (VNDetectHumanBodyPoseRequest,
# CoreImage / saliency color analysis).
# --------------------------------------------------------------------------
POSTURE_STATES = [
    "upright_aligned", "slight_slouch", "lateral_lean", "shoulders_uneven"
]
LIGHTING_CONDITIONS = [
    "well_lit", "dim_overhead", "harsh_shadows", "soft_window_light"
]
COLOR_HARMONIES = [
    "classic_contrast", "monochromatic_neutral", "clashing_tones",
    "earthy_analogous", "complementary_pop"
]


def posture_states():
    return list(POSTURE_STATES)


def lighting_conditions():
    return list(LIGHTING_CONDITIONS)


def color_harmonies():
    return list(COLOR_HARMONIES)


# --------------------------------------------------------------------------
# Occasion -- not part of the vision model's output, see module docstring.
# --------------------------------------------------------------------------
OCCASIONS = [
    "Casual Dinner",
    "First Date",
    "Tech Job Interview",
    "Summer Wedding",
    "Everyday Errands",
    "Formal Business Meeting",
    "Night Out",
    "Gym / Athleisure",
    "Creative Field Interview",
    "Family Gathering",
]


def occasions():
    return list(OCCASIONS)


# --------------------------------------------------------------------------
# Garment / grooming vocabulary -- pulled straight from taxonomy.py, never
# hand-duplicated. prune_vocabulary.py uses full_vocabulary_terms() as a
# MUST-KEEP floor so the pruned tokenizer can always spell every real class
# name, not just whatever one Gemini run happened to produce.
# --------------------------------------------------------------------------
def garment_vocabulary(gender):
    """Every real garment-slot type name for this gender."""
    slots = vision_tx.SLOTS_BY_GENDER[gender]
    names = set()
    for slot_entries in slots.values():
        for entry in slot_entries:
            names.add(entry[0])
    names.discard("none")
    return sorted(names)


def full_vocabulary_terms():
    """Every literal word/class-name the pruned tokenizer must be able to
    spell losslessly: garment types (both genders), patterns, formality
    classes, hair lengths, facial hair styles, makeup styles, occasions,
    Apple Vision signals, plus the tag-format's own field names."""
    terms = set()
    for gender in ("man", "woman"):
        terms.update(garment_vocabulary(gender))
    terms.update(vision_tx.PATTERNS)
    terms.update(vision_tx.FORMALITY_CLASSES)
    terms.update(vision_tx.HAIR_LENGTHS)
    terms.update(vision_tx.FACIAL_HAIR_STYLES)
    terms.update(vision_tx.MAKEUP_STYLES)
    terms.update(OCCASIONS)
    terms.update(POSTURE_STATES)
    terms.update(LIGHTING_CONDITIONS)
    terms.update(COLOR_HARMONIES)
    terms.update([
        "top", "outer", "bottoms", "shoes", "hair", "skin", "eyebrows",
        "facial_hair", "makeup", "priority_defect", "overall_score",
        "occasion", "gender", "vibe", "tags", "formality", "pattern",
        "none", "male", "female", "casual", "polished",
        # Apple Vision & fine-grained defect fields
        "posture", "lighting", "color_harmony", "fit", "fabric",
        "footwear_condition", "baggy_level", "tight_level", "tailored",
        "wrinkled_level", "crisp", "worn_level",
        # ChatML role names -- rendered as plain word tokens by the chat
        # template's own text (e.g. "<|im_start|>system\n"), NOT covered
        # by tokenizer.all_special_ids (only the <|im_start|>/<|im_end|>
        # markers themselves are registered special tokens). Caught by
        # finetune.py --dry-run raising RemappedTokenizer's hard-fail on
        # this exact gap during this pipeline's own verification.
        "system", "user", "assistant",
    ])
    return terms


# --------------------------------------------------------------------------
# Priority defect -- picks the single worst real trained field, never an
# invented one. Fixed tie-break order (earlier = higher priority on a tie).
# --------------------------------------------------------------------------
_OUTFIT_DEFECT_FIELDS = ["fabric_wrinkled", "fit_baggy", "fit_tight", "footwear_worn", "styling_sloppy"]
_GROOMING_COMMON_DEFECT_FIELDS = ["hair_untidy", "skin_neglected", "eyebrows_unkempt"]


def _defect_fields_for(category):
    if vision_tx.CATEGORY_KIND[category] == "outfit":
        return list(_OUTFIT_DEFECT_FIELDS)
    fields = list(_GROOMING_COMMON_DEFECT_FIELDS)
    if vision_tx.CATEGORY_GENDER[category] == "man":
        fields.append("facial_hair_untidy")
    else:
        fields.append("makeup_uneven")
    return fields


def priority_defect(category, row):
    """row: a dict of {field_name: value} matching taxonomy.get_label_schema(category)
    (either a real vision-model inference output, or a synthetic taxonomy-driven
    training row). Returns the field name with the highest ordinal severity
    (0-2), or None if nothing is flagged (severity 0 everywhere)."""
    best_field, best_severity = None, 0
    for field in _defect_fields_for(category):
        val = row.get(field)
        if val is None:
            continue
        severity = int(val)
        if severity > best_severity:
            best_field, best_severity = field, severity
    return best_field


# --------------------------------------------------------------------------
# Prompt assembly -- the exact tags block, built from real field names.
# --------------------------------------------------------------------------
def _outfit_tag_lines(category, row):
    lines = [f"- top: {row.get('upper_type', 'unknown')} (pattern: {row.get('upper_pattern', 'solid')})"]
    mid_type = row.get("mid_type", "none")
    if mid_type and mid_type != "none":
        lines.append(f"- outer: {mid_type}")
    lower_type = row.get("lower_type", "none")
    if lower_type and lower_type != "none":
        lines.append(f"- bottoms: {lower_type} (pattern: {row.get('lower_pattern', 'solid')})")
    lines.append(f"- shoes: {row.get('footwear_type', 'unknown')}")
    lines.append(f"- formality: {row.get('formality', 'casual')}")

    # Detailed fit observations if present
    fit_parts = []
    if int(row.get("fit_baggy", 0)) > 0:
        fit_parts.append(f"baggy_level: {row.get('fit_baggy')}")
    if int(row.get("fit_tight", 0)) > 0:
        fit_parts.append(f"tight_level: {row.get('fit_tight')}")
    if int(row.get("fit_tailored", 0)) > 0:
        fit_parts.append("tailored")
    if fit_parts:
        lines.append(f"- fit: {', '.join(fit_parts)}")

    # Detailed fabric observations if present
    fabric_parts = []
    if int(row.get("fabric_wrinkled", 0)) > 0:
        fabric_parts.append(f"wrinkled_level: {row.get('fabric_wrinkled')}")
    if int(row.get("fabric_crisp", 0)) > 0:
        fabric_parts.append("crisp")
    if fabric_parts:
        lines.append(f"- fabric: {', '.join(fabric_parts)}")

    # Footwear condition if present
    footwear_parts = []
    if int(row.get("footwear_worn", 0)) > 0:
        footwear_parts.append(f"worn_level: {row.get('footwear_worn')}")
    if int(row.get("footwear_polished", 0)) > 0:
        footwear_parts.append("polished")
    if footwear_parts:
        lines.append(f"- footwear_condition: {', '.join(footwear_parts)}")

    # Apple Vision native signals
    if "color_harmony" in row:
        lines.append(f"- color_harmony: {row['color_harmony']}")
    if "posture" in row:
        lines.append(f"- posture: {row['posture']}")
    if "lighting" in row:
        lines.append(f"- lighting: {row['lighting']}")

    return lines


def _grooming_tag_lines(category, row):
    gender = vision_tx.CATEGORY_GENDER[category]
    lines = [
        f"- hair: {row.get('hair_length', 'short')} "
        f"(styled: {row.get('hair_styled', 0)}, untidy_level: {row.get('hair_untidy', 0)})",
        f"- skin: (neglected_level: {row.get('skin_neglected', 0)}, healthy: {row.get('skin_healthy', 0)})",
        f"- eyebrows: (unkempt_level: {row.get('eyebrows_unkempt', 0)}, groomed: {row.get('eyebrows_groomed', 0)})",
    ]
    if gender == "man":
        lines.append(
            f"- facial_hair: {row.get('facial_hair_style', 'clean_shaven')} "
            f"(untidy_level: {row.get('facial_hair_untidy', 0)})"
        )
    else:
        lines.append(
            f"- makeup: {row.get('makeup_style', 'none')} (uneven_level: {row.get('makeup_uneven', 0)})"
        )

    # Apple Vision native signals
    if "posture" in row:
        lines.append(f"- posture: {row['posture']}")
    if "lighting" in row:
        lines.append(f"- lighting: {row['lighting']}")

    return lines


def format_tag_prompt(category, occasion, row):
    """Builds the full <|im_start|>user ... tags block (everything after
    the system turn) from a real-schema-shaped label row. `category` is one
    of taxonomy.ALL_CATEGORIES (Men_Grooming/Women_Grooming/Men_Outfit/
    Women_Outfit) -- it determines gender/vibe and which tag lines apply."""
    gender = vision_tx.CATEGORY_GENDER[category]
    kind = vision_tx.CATEGORY_KIND[category]
    gender_label = "Male" if gender == "man" else "Female"

    if kind == "outfit":
        tag_lines = _outfit_tag_lines(category, row)
    else:
        tag_lines = _grooming_tag_lines(category, row)

    defect = priority_defect(category, row)
    score = row.get("score", 5.0)

    lines = [
        f"Occasion: {occasion}",
        f"Gender/Vibe: {gender_label}",
        "Tags:",
        *tag_lines,
        f"- priority_defect: {defect if defect else 'none'}",
        f"- overall_score: {float(score):.1f}/10",
    ]
    return "\n".join(lines)


def format_full_prompt(category, occasion, row, system_prompt):
    """Full ChatML-style prompt including the system turn and the empty
    assistant turn the model is expected to complete -- exactly the shape
    the CoreML runtime side will build at inference time (see
    export_coreml.py / the iOS PromptBuilder this mirrors)."""
    tag_block = format_tag_prompt(category, occasion, row)
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{tag_block}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
