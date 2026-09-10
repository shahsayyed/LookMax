#!/usr/bin/env python3
"""
relabel_batch.py -- for the two failure categories confirmed NOT fixable via
prompt engineering (pattern_miss, facial_hair_texture -- see vast_test A/B
results, 2026-09), correct the CSV label to match what's actually shown
instead of discarding/regenerating the image. Reuses verify_batch.py's Batch
Prediction plumbing (GCS-staged JSONL in/out) but with a classification
system prompt instead of a valid/invalid verdict prompt.

Two-step, both explicit:
  1. `--classify` submits a Batch Prediction job per flagged image asking the
     VLM what the attribute ACTUALLY shows (not whether it matches the
     prompt), and writes a correction mapping to qa_processed/. Costs a
     small amount of Vertex API spend and does not touch any CSV.
  2. `--apply <mapping.json>` merges those corrections into the production
     label CSVs under ML/data/vision_synthetic/raw_generated/. Dry-run by
     default (prints a diff); pass --write to actually modify the CSVs (a
     timestamped .bak copy of every touched CSV is written first).

Usage:
  python3 relabel_batch.py --classify pattern_miss --audit-report <path>
  python3 relabel_batch.py --classify facial_hair_texture --audit-report <path>
  python3 relabel_batch.py --apply qa_processed/relabel_pattern_miss_*.json           # dry-run
  python3 relabel_batch.py --apply qa_processed/relabel_pattern_miss_*.json --write   # actually edit CSVs
"""
import argparse
import base64
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
RAW_DIR = REPO_ROOT / "ML" / "data" / "vision_synthetic" / "raw_generated"
IMAGES_DIR = RAW_DIR / "images"
QA_OUT_DIR = REPO_ROOT / "ML" / "data" / "vision_synthetic" / "qa_processed"
CREDENTIALS_FILE = REPO_ROOT / "lookmax-generation-513e3f9ab69e.json"
CHUNKS_CACHE_DIR = SCRIPT_DIR / ".relabel_chunks_cache"

GCS_BUCKET = "lookmax-generation-batch-staging"
MODEL = "gemini-3.8-flash"
LOCATION = "global"

sys.path.insert(0, str(SCRIPT_DIR))
import taxonomy as tx  # noqa: E402
import full_run as fr  # noqa: E402
from verify_dataset_vertex import parse_verdict_json  # noqa: E402
from verify_batch import submit_chunk as _submit_chunk_generic, collect_chunk_results, get_client_and_project  # noqa: E402


# --------------------------------------------------------------------------
# Per-category classification prompts. Each asks the VLM to report what is
# ACTUALLY shown (never "does it match"), scoped to exactly one column so a
# response maps unambiguously onto a CSV edit.
#
# facial_hair_texture / eyebrows / skin_condition all use the SAME shape:
# taxonomy.py's own tier phrases (flaw_severe/flaw_mild/average/polished --
# see FACIAL_HAIR_CONDITION_PHRASE / EYEBROWS_CONDITION_PHRASE /
# SKIN_CONDITION_PHRASE) ARE the classification vocabulary, and the label
# values are re-derived from the classified tier via tx.severity_for_tier() /
# tx.positive_for_tier() -- the exact same function that generated the
# original (wrong) label, just fed the tier the pixels actually show instead
# of the tier that was requested. This only corrects the condition/severity
# columns, never the identity columns (facial_hair_style, etc).
# --------------------------------------------------------------------------
def _tier_classification_prompt(attribute_label, phrase_dict, extra_context=""):
    bullets = "\n".join(f'  - "{tier}": {phrase}' for tier, phrase in phrase_dict.items())
    return f"""\
You are labelling {attribute_label} for a computer-vision training dataset.
You will be shown one image and the original generation prompt (which may be
WRONG about what's actually shown -- ignore what the prompt claims, judge
only the pixels).{extra_context}

Classify the {attribute_label} actually visible into exactly ONE of these
four tiers (pick whichever description is the closest match):
{bullets}

Return ONLY this JSON object, no other text:
{{"actual_tier": "flaw_severe"|"flaw_mild"|"average"|"polished"}}
"""


PATTERN_MISS_SYSTEM_PROMPT = """\
You are labelling fabric pattern for a computer-vision training dataset. You
will be shown one image of a person wearing an outfit and the original
generation prompt (which may be WRONG about what the fabric actually shows --
ignore what the prompt claims, judge only the pixels).

For the UPPER-BODY garment and, if present and visibly a separate lower-body
garment, the LOWER-BODY garment, classify the fabric pattern actually visible
as exactly one of: "solid", "striped", "checked", "printed".
  - "solid": one uniform color, no visible pattern/print/graphic.
  - "striped": parallel stripes.
  - "checked": a grid/plaid/check pattern.
  - "printed": any other visible print, graphic, logo, or non-geometric design.

If a slot is not visible or not applicable (e.g. no separate lower garment,
occluded), use "unclear" for that slot.

Return ONLY this JSON object, no other text:
{"upper_pattern": "solid"|"striped"|"checked"|"printed"|"unclear",
 "lower_pattern": "solid"|"striped"|"checked"|"printed"|"unclear"|"none"}
"""

FACIAL_HAIR_TEXTURE_SYSTEM_PROMPT = _tier_classification_prompt(
    "facial hair condition/texture", tx.FACIAL_HAIR_CONDITION_PHRASE,
    " The person already has some facial hair (style is not in question, only its condition).")
EYEBROWS_SYSTEM_PROMPT = _tier_classification_prompt("eyebrow condition", tx.EYEBROWS_CONDITION_PHRASE)
SKIN_CONDITION_SYSTEM_PROMPT = _tier_classification_prompt("facial skin condition", tx.SKIN_CONDITION_PHRASE)
MAKEUP_SYSTEM_PROMPT = _tier_classification_prompt(
    "makeup application quality", tx.MAKEUP_CONDITION_PHRASE,
    " The person is wearing makeup (whether they're wearing makeup at all is not in question, only how well it's applied).")

# --------------------------------------------------------------------------
# SALVAGE prompts -- for clean_shaven, the image is about to be REGENERATED
# (that class is too depleted to fix by relabeling alone, see relabel_batch.py
# usage notes). Before it's overwritten, classify what it actually shows so a
# real, correctly-labeled image isn't thrown away just because it wasn't the
# specific thing asked for. Three outcomes per image: the audit's original
# "invalid" call was itself wrong (keep the original row, no edit) / the
# style genuinely differs (copy it out under a new filename with a corrected
# row, as BONUS data -- the original index still gets regenerated to backfill
# the clean_shaven class) / VLM can't tell (discard the copy).
# --------------------------------------------------------------------------
CLEAN_SHAVEN_SALVAGE_SYSTEM_PROMPT = f"""\
You are labelling facial hair for a computer-vision training dataset. You
will be shown one image of a man and the original generation prompt, which
claims the face is completely clean-shaven -- that claim may be WRONG; ignore
it and judge only the pixels.

First, classify the facial hair STYLE actually visible as exactly one of:
  - "clean_shaven": no facial hair at all (the prompt was actually correct).
  - "stubble": very short, even growth.
  - "short_beard": a short, filled-in beard.
  - "full_beard": a full, longer beard.
  - "moustache": moustache only, no beard.

If the style is NOT "clean_shaven", also classify its CONDITION as exactly
one of these four tiers:
  - "flaw_severe": {tx.FACIAL_HAIR_CONDITION_PHRASE['flaw_severe']}
  - "flaw_mild": {tx.FACIAL_HAIR_CONDITION_PHRASE['flaw_mild']}
  - "average": {tx.FACIAL_HAIR_CONDITION_PHRASE['average']}
  - "polished": {tx.FACIAL_HAIR_CONDITION_PHRASE['polished']}
If style IS "clean_shaven", set actual_tier to "n/a".

Return ONLY this JSON object, no other text:
{{"actual_style": "clean_shaven"|"stubble"|"short_beard"|"full_beard"|"moustache",
 "actual_tier": "flaw_severe"|"flaw_mild"|"average"|"polished"|"n/a"}}
"""

# --------------------------------------------------------------------------
# Second wave -- found by locally clustering the "unclassified" residual from
# the first audit pass (2026-09-09) once identity/meta noise (body build,
# face shape, hoodie pullover/zip, requested_hair_desc, lace state -- none of
# them trained columns) was stripped out. Every one of these is a small
# fraction of its population pool (well inside safe-to-relabel range), so all
# are in-place relabels, no regeneration needed.
# --------------------------------------------------------------------------
PATTERN_SOLID_BLEED_SYSTEM_PROMPT = PATTERN_MISS_SYSTEM_PROMPT  # same ask, mirror-image direction

HAIR_LENGTH_SYSTEM_PROMPT = f"""\
You are labelling hair length for a computer-vision training dataset. You
will be shown one head-and-shoulders portrait and the original generation
prompt (which may be WRONG about the hair length -- ignore what the prompt
claims, judge only the pixels).

Classify the hair length actually visible as exactly one of:
  - "buzz_cut": {tx.HAIR_LENGTH_WORDS['buzz_cut']}
  - "short": {tx.HAIR_LENGTH_WORDS['short']}
  - "medium": {tx.HAIR_LENGTH_WORDS['medium']}
  - "long": {tx.HAIR_LENGTH_WORDS['long']}

Return ONLY this JSON object, no other text:
{{"actual_hair_length": "buzz_cut"|"short"|"medium"|"long"}}
"""

FACIAL_HAIR_STYLE_SYSTEM_PROMPT = f"""\
You are labelling facial hair for a computer-vision training dataset. You
will be shown one image of a man and the original generation prompt, which
names a specific facial hair STYLE -- that claim may be WRONG; ignore it and
judge only the pixels.

First, classify the facial hair STYLE actually visible as exactly one of:
  - "clean_shaven": no facial hair at all.
  - "stubble": very short, even growth.
  - "short_beard": a short, filled-in beard.
  - "full_beard": a full, longer beard.
  - "moustache": moustache only, no beard.

If the style is NOT "clean_shaven", also classify its CONDITION as exactly
one of these four tiers:
  - "flaw_severe": {tx.FACIAL_HAIR_CONDITION_PHRASE['flaw_severe']}
  - "flaw_mild": {tx.FACIAL_HAIR_CONDITION_PHRASE['flaw_mild']}
  - "average": {tx.FACIAL_HAIR_CONDITION_PHRASE['average']}
  - "polished": {tx.FACIAL_HAIR_CONDITION_PHRASE['polished']}
If style IS "clean_shaven", set actual_tier to "n/a".

Return ONLY this JSON object, no other text:
{{"actual_style": "clean_shaven"|"stubble"|"short_beard"|"full_beard"|"moustache",
 "actual_tier": "flaw_severe"|"flaw_mild"|"average"|"polished"|"n/a"}}
"""

FIT_SYSTEM_PROMPT = f"""\
You are labelling garment fit for a computer-vision training dataset. You
will be shown one full-body image and the original generation prompt (which
may be WRONG about the fit -- ignore what the prompt claims, judge only the
pixels).

First, classify the overall fit DIRECTION actually visible as exactly one of:
  - "baggy": loose/oversized through the body.
  - "tight": snug/straining through the body.
  - "neutral": neither -- a standard or tailored fit.

If direction is NOT "neutral", also classify its SEVERITY as exactly one of
these four tiers (using {{direction}} as whichever direction you picked):
  - "flaw_severe": {tx.FIT_CONDITION_PHRASE[('flaw_severe','baggy')]} (if baggy) / {tx.FIT_CONDITION_PHRASE[('flaw_severe','tight')]} (if tight)
  - "flaw_mild": {tx.FIT_CONDITION_PHRASE[('flaw_mild','baggy')]} (if baggy) / {tx.FIT_CONDITION_PHRASE[('flaw_mild','tight')]} (if tight)
If direction IS "neutral", classify whether it looks specifically TAILORED
("polished": {tx.FIT_CONDITION_PHRASE[('polished','baggy')]}) or just ordinary
("average": {tx.FIT_CONDITION_PHRASE[('average','baggy')]}).

Return ONLY this JSON object, no other text:
{{"actual_direction": "baggy"|"tight"|"neutral",
 "actual_tier": "flaw_severe"|"flaw_mild"|"average"|"polished"}}
"""

FABRIC_WRINKLED_SYSTEM_PROMPT = _tier_classification_prompt(
    "garment fabric condition", tx.FABRIC_CONDITION_PHRASE,
    " Note: if the prompt asked for a 'coin-sized stain' and the image shows a literal coin/coin-shaped object lying on "
    "the fabric rather than an actual stain, that does NOT count as a stain being present -- judge only real fabric condition.")

MEN_FOOTWEAR_NAMES = "\n".join(f'  - "{k}": {phrase}' for k, phrase, _f in tx.MEN_FOOTWEAR)
WOMEN_FOOTWEAR_NAMES = "\n".join(f'  - "{k}": {phrase}' for k, phrase, _f in tx.WOMEN_FOOTWEAR)

FOOTWEAR_TYPE_MEN_SYSTEM_PROMPT = f"""\
You are labelling footwear type for a computer-vision training dataset. You
will be shown one full-body image of a man and the original generation
prompt (which may be WRONG about the footwear -- ignore what the prompt
claims, judge only the pixels).

Classify the footwear actually visible as exactly one of:
{MEN_FOOTWEAR_NAMES}

Return ONLY this JSON object, no other text:
{{"actual_footwear_type": "slides"|"flip_flops"|"canvas_sneakers"|"running_shoes"|"leather_sneakers"|"suede_desert_boots"|"leather_dress_shoes"|"oxford_shoes"}}
"""

FOOTWEAR_TYPE_WOMEN_SYSTEM_PROMPT = f"""\
You are labelling footwear type for a computer-vision training dataset. You
will be shown one full-body image of a woman and the original generation
prompt (which may be WRONG about the footwear -- ignore what the prompt
claims, judge only the pixels).

Classify the footwear actually visible as exactly one of:
{WOMEN_FOOTWEAR_NAMES}

Return ONLY this JSON object, no other text:
{{"actual_footwear_type": "slides"|"flip_flops"|"sneakers"|"ballet_flats"|"ankle_boots"|"block_heels"|"pointed_flats"|"heeled_pumps"}}
"""

CATEGORY_PROMPTS = {
    "pattern_miss": PATTERN_MISS_SYSTEM_PROMPT,
    "facial_hair_texture": FACIAL_HAIR_TEXTURE_SYSTEM_PROMPT,
    "eyebrows": EYEBROWS_SYSTEM_PROMPT,
    "skin_condition": SKIN_CONDITION_SYSTEM_PROMPT,
    "makeup": MAKEUP_SYSTEM_PROMPT,
    "clean_shaven_salvage": CLEAN_SHAVEN_SALVAGE_SYSTEM_PROMPT,
    # Same style+tier classification as clean_shaven_salvage, reused for the
    # residual failures left AFTER regeneration+reverify (2026-09-10):
    # post-regen, the clean_shaven class is no longer critically depleted
    # (75.1% correct at that point), so these get relabeled in place instead
    # of another salvage+regen round -- see build_corrections' handling of
    # this category (shares the facial_hair_style_miss branch).
    "clean_shaven_residual": CLEAN_SHAVEN_SALVAGE_SYSTEM_PROMPT,
    "pattern_solid_bleed": PATTERN_SOLID_BLEED_SYSTEM_PROMPT,
    "hair_length_grooming": HAIR_LENGTH_SYSTEM_PROMPT,
    "facial_hair_style_miss": FACIAL_HAIR_STYLE_SYSTEM_PROMPT,
    "fit_miss": FIT_SYSTEM_PROMPT,
    "fabric_wrinkled_miss": FABRIC_WRINKLED_SYSTEM_PROMPT,
    "footwear_type_men": FOOTWEAR_TYPE_MEN_SYSTEM_PROMPT,
    "footwear_type_women": FOOTWEAR_TYPE_WOMEN_SYSTEM_PROMPT,
}

# Tier-classification categories -> which two CSV columns get
# severity_for_tier()/positive_for_tier() applied to the classified tier.
TIER_CATEGORY_COLUMNS = {
    "facial_hair_texture": ("facial_hair_untidy", "facial_hair_groomed"),
    "eyebrows": ("eyebrows_unkempt", "eyebrows_groomed"),
    "skin_condition": ("skin_neglected", "skin_healthy"),
    "makeup": ("makeup_uneven", "makeup_flawless"),
    "fabric_wrinkled_miss": ("fabric_wrinkled", "fabric_crisp"),
}

# Simple categorical categories -> which single CSV column the classified
# value is written into directly (no tier math involved).
CATEGORICAL_CATEGORY_COLUMNS = {
    "hair_length_grooming": ("actual_hair_length", "hair_length"),
    "footwear_type_men": ("actual_footwear_type", "footwear_type"),
    "footwear_type_women": ("actual_footwear_type", "footwear_type"),
}

# Categories that COPY the image under a new filename + append a new CSV row
# (salvage) rather than editing the existing row in place -- used for classes
# too depleted by their failure rate to fix via in-place relabeling alone
# (see module docstring). The original index is ALSO regenerated separately
# to backfill the depleted class -- salvaging and regenerating the same
# index are not in conflict because salvage never edits the original file/
# row, only adds a copy under a new filename.
SALVAGE_CATEGORIES = {"clean_shaven_salvage", "makeup"}

# Which failure-detection keywords + row-precondition select images into
# each bucket from a full-audit report (same technique used throughout the
# vast_test A/B sweep to separate a target attribute's real failures from
# co-occurring, already-separately-handled ones).
def _is_flaw_severe_grooming(item):
    return item["category"] in ("Men_Grooming", "Women_Grooming") and item["tier"] == "flaw_severe"


BUCKET_FILTERS = {
    "pattern_miss": {
        "item_check": lambda item: (item["task"]["row"].get("upper_pattern") == "printed"
                                     or item["task"]["row"].get("lower_pattern") == "printed"),
        "keywords": ["pattern", "print", "solid", "plain", "graphic"],
    },
    "facial_hair_texture": {
        "item_check": lambda item: item["task"]["row"].get("facial_hair_style") not in (None, "", "clean_shaven"),
        "keywords": ["patchy", "texture", "dense rather than patchy", "uneven facial hair", "facial hair"],
    },
    # eyebrows_unkempt/skin_neglected are tier-derived (severity_for_tier), so the
    # "severe" signal we're correcting only ever applies within the flaw_severe
    # grooming pool -- restrict to it to avoid picking up unrelated mentions.
    "eyebrows": {
        "item_check": _is_flaw_severe_grooming,
        "keywords": ["eyebrow", "unibrow"],
    },
    "skin_condition": {
        "item_check": _is_flaw_severe_grooming,
        "keywords": ["dry", "flak"],
    },
    "makeup": {
        "item_check": lambda item: (item["category"] == "Women_Grooming" and item["tier"] == "flaw_severe"
                                     and item["task"]["row"].get("makeup_style") not in (None, "", "none")),
        "keywords": ["makeup"],
    },
    "clean_shaven_salvage": {
        "item_check": lambda item: item["task"]["row"].get("facial_hair_style") == "clean_shaven",
        "keywords": ["stubble", "beard", "mustache", "shadow", "unshaven", "facial hair"],
    },
    "clean_shaven_residual": {
        "item_check": lambda item: item["task"]["row"].get("facial_hair_style") == "clean_shaven",
        "keywords": ["stubble", "beard", "mustache", "moustache", "shadow", "unshaven", "facial hair"],
    },
    # --- second wave, found by clustering the "unclassified" residual ---
    "pattern_solid_bleed": {
        "item_check": lambda item: (item["task"]["row"].get("upper_pattern") == "solid"
                                     or item["task"]["row"].get("lower_pattern") == "solid"),
        "keywords": ["pattern", "patterned", "plaid", "checked", "striped", "printed", "graphic"],
    },
    "hair_length_grooming": {
        "item_check": lambda item: item["category"] in ("Men_Grooming", "Women_Grooming"),
        "keywords": ["hair"],
    },
    "facial_hair_style_miss": {
        "item_check": lambda item: item["task"]["row"].get("facial_hair_style") not in (None, "", "clean_shaven"),
        "keywords": ["beard", "moustache", "stubble", "clean-shaven", "clean shaven"],
    },
    "fit_miss": {
        "item_check": lambda item: item["category"] in ("Men_Outfit", "Women_Outfit"),
        "keywords": ["loose", "baggy", "tight", "snug", "straining"],
    },
    "fabric_wrinkled_miss": {
        "item_check": lambda item: item["category"] in ("Men_Outfit", "Women_Outfit"),
        "keywords": ["stain", "wrinkl", "crease", "coin"],
    },
    "footwear_type_men": {
        "item_check": lambda item: item["category"] == "Men_Outfit",
        "keywords": ["footwear", "sandals", "sneaker", "boots", "flip flop", "desert boot", "dress shoe", "oxford"],
    },
    "footwear_type_women": {
        "item_check": lambda item: item["category"] == "Women_Outfit",
        "keywords": ["footwear", "sandals", "sneaker", "boots", "flats", "heels", "pumps", "flip flop"],
    },
}


def classify_flagged(category, audit_report_path):
    d = json.loads(Path(audit_report_path).read_text())
    results = d["results"]

    tasks = fr.build_full_task_list()
    by_index = {t["index"]: t for t in tasks}
    idx_re = re.compile(r"^(\d{5})_")

    filt = BUCKET_FILTERS[category]
    items = []  # (filename, image_path, original_prompt, index)
    for r in results:
        if r.get("verdict") != "invalid":
            continue
        m = idx_re.match(r["filename"])
        if not m:
            continue
        idx = int(m.group(1))
        item = by_index[idx]
        if not filt["item_check"](item):
            continue
        issues = " ".join(r.get("issues", [])).lower()
        if not any(k in issues for k in filt["keywords"]):
            continue
        img_path = IMAGES_DIR / r["filename"]
        if not img_path.exists():
            continue
        items.append((r["filename"], img_path, by_index[idx]["task"]["prompt"], idx))

    print(f"[{category}] {len(items)} flagged images selected for relabel classification")
    return items


def build_request_line(key, image_path, prompt_text, system_prompt):
    img_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "key": key,
        "request": {
            "contents": [{
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": img_b64}},
                    {"text": f"ORIGINAL GENERATION PROMPT (may be wrong -- judge only the pixels):\n{prompt_text}\n\nClassify:"},
                ],
            }],
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "generation_config": {"temperature": 0.1, "response_mime_type": "application/json"},
        },
    }


def submit_chunk(chunk_items, client, chunk_label, system_prompt):
    from google.genai import types

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    gcs_prefix = f"gs://{GCS_BUCKET}/batch_relabel/{chunk_label}_{ts}"
    input_path = SCRIPT_DIR / f".relabel_input_{chunk_label}_{ts}.jsonl"
    with open(input_path, "w") as f:
        for fn, img_path, prompt_text, _idx in chunk_items:
            f.write(json.dumps(build_request_line(fn, img_path, prompt_text, system_prompt)) + "\n")

    gcs_input = f"{gcs_prefix}/input.jsonl"
    subprocess.run(["gcloud", "storage", "cp", str(input_path), gcs_input], check=True, capture_output=True)
    input_path.unlink(missing_ok=True)

    job = client.batches.create(
        model=MODEL, src=gcs_input,
        config=types.CreateBatchJobConfig(dest=f"{gcs_prefix}/output/"),
    )
    print(f"    [{chunk_label}] job {job.name.split('/')[-1]} submitted ({len(chunk_items)} images)", flush=True)
    return job


def run_classify(category, audit_report_path, chunk_size, max_concurrent, poll_interval):
    items = classify_flagged(category, audit_report_path)
    if not items:
        sys.exit(f"No flagged images found for category={category} in {audit_report_path}")

    system_prompt = CATEGORY_PROMPTS[category]
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    print(f"Classifying {len(items)} images via Batch Prediction ({MODEL}), "
          f"split into {len(chunks)} job(s) of up to {chunk_size} each")

    client, project_id = get_client_and_project()
    CHUNKS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    label = f"relabel_{category}"
    all_results = []
    pending_idxs = []
    for i in range(len(chunks)):
        cache_path = CHUNKS_CACHE_DIR / f"{label}_chunk{i:04d}.json"
        if cache_path.exists():
            chunk_results = json.loads(cache_path.read_text())
            print(f"[chunk {i+1}/{len(chunks)}] already done (cached), {len(chunk_results)} results -- skipping")
            all_results.extend(chunk_results)
        else:
            pending_idxs.append(i)

    in_flight = {}
    next_idx_pos = 0

    def top_up():
        nonlocal next_idx_pos
        while len(in_flight) < max_concurrent and next_idx_pos < len(pending_idxs):
            i = pending_idxs[next_idx_pos]
            next_idx_pos += 1
            job = submit_chunk(chunks[i], client, f"{label}_c{i:04d}", system_prompt)
            in_flight[i] = job

    def poll_with_retry(job_name, max_retries=5):
        for attempt in range(max_retries):
            try:
                return client.batches.get(name=job_name)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = (attempt + 1) * 15
                print(f"    [poll retry {attempt+1}/{max_retries} after {type(e).__name__}, waiting {wait}s]", flush=True)
                time.sleep(wait)

    top_up()
    done_count = len(chunks) - len(pending_idxs)
    while in_flight:
        time.sleep(poll_interval)
        for i in list(in_flight.keys()):
            job = poll_with_retry(in_flight[i].name)
            if not str(job.state).endswith(("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED", "PARTIALLY_SUCCEEDED")):
                in_flight[i] = job
                continue
            del in_flight[i]
            chunk_results = collect_chunk_results(job, project_id, f"{label}_c{i:04d}")
            cache_path = CHUNKS_CACHE_DIR / f"{label}_chunk{i:04d}.json"
            cache_path.write_text(json.dumps(chunk_results, indent=2))
            all_results.extend(chunk_results)
            done_count += 1
            print(f"[chunk {i+1}/{len(chunks)}] done, {len(chunk_results)} results "
                  f"({done_count}/{len(chunks)} chunks complete, {len(in_flight)} in flight)", flush=True)
            top_up()

    QA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = QA_OUT_DIR / f"relabel_{category}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"category": category, "model": MODEL, "generated_at": ts, "results": all_results}, f, indent=2)
    print(f"\nFull classification written to: {out_path.relative_to(REPO_ROOT)}")

    for f in CHUNKS_CACHE_DIR.glob(f"{label}_chunk*.json"):
        f.unlink()
    return out_path


# --------------------------------------------------------------------------
# Apply corrections to the production CSVs
# --------------------------------------------------------------------------
VALID_PATTERNS = {"solid", "striped", "checked", "printed"}
VALID_TIERS = {"flaw_severe", "flaw_mild", "average", "polished"}


def parse_classification(text):
    return parse_verdict_json(text)


VALID_FACIAL_HAIR_STYLES = {"clean_shaven", "stubble", "short_beard", "full_beard", "moustache"}
VALID_HAIR_LENGTHS = {"buzz_cut", "short", "medium", "long"}
VALID_FOOTWEAR = {k for k, _p, _f in tx.MEN_FOOTWEAR} | {k for k, _p, _f in tx.WOMEN_FOOTWEAR}


def build_corrections(category, classification_path):
    d = json.loads(Path(classification_path).read_text())
    corrections = {}  # filename -> {column: new_value}
    skipped = 0
    for r in d["results"]:
        fn = r.get("filename")
        if category in ("pattern_miss", "pattern_solid_bleed"):
            up = r.get("upper_pattern")
            lo = r.get("lower_pattern")
            edit = {}
            if up in VALID_PATTERNS:
                edit["upper_pattern"] = up
            if lo in VALID_PATTERNS:
                edit["lower_pattern"] = lo
            if edit:
                corrections[fn] = edit
            else:
                skipped += 1
        elif category in TIER_CATEGORY_COLUMNS:
            tier = r.get("actual_tier")
            if tier not in VALID_TIERS:
                skipped += 1
                continue
            severity_col, positive_col = TIER_CATEGORY_COLUMNS[category]
            corrections[fn] = {
                severity_col: tx.severity_for_tier(tier),
                positive_col: tx.positive_for_tier(tier),
            }
        elif category in CATEGORICAL_CATEGORY_COLUMNS:
            resp_key, csv_col = CATEGORICAL_CATEGORY_COLUMNS[category]
            val = r.get(resp_key)
            valid_set = VALID_HAIR_LENGTHS if category == "hair_length_grooming" else VALID_FOOTWEAR
            if val not in valid_set:
                skipped += 1
                continue
            corrections[fn] = {csv_col: val}
        elif category in ("facial_hair_style_miss", "clean_shaven_residual"):
            style = r.get("actual_style")
            tier = r.get("actual_tier")
            if style not in VALID_FACIAL_HAIR_STYLES:
                skipped += 1
                continue
            if style == "clean_shaven":
                corrections[fn] = {"facial_hair_style": style, "facial_hair_untidy": 0, "facial_hair_groomed": 0}
            elif tier in VALID_TIERS:
                corrections[fn] = {
                    "facial_hair_style": style,
                    "facial_hair_untidy": tx.severity_for_tier(tier),
                    "facial_hair_groomed": tx.positive_for_tier(tier),
                }
            else:
                skipped += 1
        elif category == "fit_miss":
            direction = r.get("actual_direction")
            tier = r.get("actual_tier")
            if direction not in ("baggy", "tight", "neutral") or tier not in VALID_TIERS:
                skipped += 1
                continue
            severity = tx.severity_for_tier(tier)
            corrections[fn] = {
                "fit_baggy": severity if direction == "baggy" else 0,
                "fit_tight": severity if direction == "tight" else 0,
                "fit_tailored": tx.positive_for_tier(tier),
            }
    print(f"[{category}] {len(corrections)} images with a usable correction, {skipped} skipped "
          f"(unclear/unparseable VLM response)")
    return corrections


def apply_corrections(corrections, write):
    """Groups filenames by their CATEGORY (from the filename convention) to
    find the right CSV, patches matching rows in place. Dry-run prints a
    diff; --write backs up each touched CSV (.bak, timestamped) then
    overwrites it."""
    by_csv = {}  # csv_path -> list of (row_index_in_file, filename, edit)
    fn_re = re.compile(r"^\d{5}_(.+?)_(flaw_severe|flaw_mild|average|polished)\.png$")

    for fn, edit in corrections.items():
        m = fn_re.match(fn)
        if not m:
            print(f"  ! could not parse category from filename {fn}, skipping")
            continue
        category = m.group(1)
        csv_path = RAW_DIR / f"labels_{category}.csv"
        if not csv_path.exists():
            shard_paths = sorted(RAW_DIR.glob(f"labels_{category}_shard*.csv"))
            if not shard_paths:
                print(f"  ! no CSV found for category {category} (fn={fn}), skipping")
                continue
            csv_path = None  # resolved per-row below among shards
            by_csv.setdefault("__shards__:" + category, []).append((fn, edit))
            continue
        by_csv.setdefault(csv_path, []).append((fn, edit))

    total_changed = 0
    for key, edits in by_csv.items():
        if isinstance(key, str) and key.startswith("__shards__:"):
            category = key.split(":", 1)[1]
            targets = sorted(RAW_DIR.glob(f"labels_{category}_shard*.csv"))
        else:
            targets = [key]

        edit_by_fn = {fn: e for fn, e in edits}
        for csv_path in targets:
            rows = list(csv.DictReader(csv_path.open()))
            fieldnames = list(rows[0].keys()) if rows else []
            changed_here = 0
            for row in rows:
                fn = row.get("filename")
                if fn in edit_by_fn:
                    for col, new_val in edit_by_fn[fn].items():
                        old_val = row.get(col)
                        if str(old_val) != str(new_val):
                            print(f"  {csv_path.name}: {fn}  {col}: {old_val!r} -> {new_val!r}")
                            row[col] = new_val
                            changed_here += 1
                    del edit_by_fn[fn]
            if changed_here and write:
                bak = csv_path.with_suffix(f".csv.bak.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
                shutil.copy2(csv_path, bak)
                with csv_path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"  -> wrote {changed_here} correction(s) to {csv_path.name} (backup: {bak.name})")
            total_changed += changed_here
        if key not in by_csv.get("__resolved__", []):
            unresolved = edit_by_fn
            for fn in unresolved:
                print(f"  ! filename {fn} not found in any target CSV row")

    print(f"\n{'Would change' if not write else 'Changed'} {total_changed} label value(s) total.")
    if not write:
        print("(dry-run -- pass --write to actually modify the CSVs)")


# --------------------------------------------------------------------------
# Salvage: for images about to be REGENERATED (their class is too depleted
# to fix via in-place relabeling), classify what they actually show first.
# Three outcomes -- see SALVAGE_CATEGORIES docstring above:
#   no_op     -- audit's "invalid" call was itself wrong; leave the row as is
#   salvage   -- copy the file under a new filename + append a new CSV row
#                with the corrected label, as bonus data
#   discard   -- VLM couldn't classify confidently; do nothing, image is
#                just overwritten by the regen run like any other failure
# --------------------------------------------------------------------------
def build_salvage_plan(category, classification_path):
    if category != "clean_shaven_salvage":
        raise ValueError(f"{category} is not a salvage category")
    d = json.loads(Path(classification_path).read_text())
    no_op, salvage, discard = [], {}, []
    for r in d["results"]:
        fn = r.get("filename")
        style = r.get("actual_style")
        tier = r.get("actual_tier")
        if style == "clean_shaven":
            no_op.append(fn)
        elif style in ("stubble", "short_beard", "full_beard", "moustache") and tier in VALID_TIERS:
            salvage[fn] = {
                "facial_hair_style": style,
                "facial_hair_untidy": tx.severity_for_tier(tier),
                "facial_hair_groomed": tx.positive_for_tier(tier),
            }
        else:
            discard.append(fn)
    print(f"[{category}] {len(no_op)} false-positive (no edit needed), "
          f"{len(salvage)} salvageable as bonus data, {len(discard)} discarded (unclear)")
    return no_op, salvage, discard


def _load_category_csvs(category):
    """Returns [(csv_path, rows, fieldnames), ...] -- one entry per file if
    the category is sharded, else a single entry."""
    csv_path = RAW_DIR / f"labels_{category}.csv"
    paths = [csv_path] if csv_path.exists() else sorted(RAW_DIR.glob(f"labels_{category}_shard*.csv"))
    out = []
    for p in paths:
        rows = list(csv.DictReader(p.open()))
        fieldnames = list(rows[0].keys()) if rows else []
        out.append((p, rows, fieldnames))
    return out


def apply_salvage(salvage, write):
    """salvage: filename -> {column: new_value}. Finds each filename's
    original row (to copy every OTHER column unchanged), writes a new row
    under a `salvage_` prefixed filename, and copies the image file to
    match. Dry-run prints what would happen; --write actually copies files
    and appends CSV rows (each touched CSV is backed up first)."""
    fn_re = re.compile(r"^\d{5}_(.+?)_(flaw_severe|flaw_mild|average|polished)\.png$")
    by_category = {}
    for fn in salvage:
        m = fn_re.match(fn)
        if not m:
            print(f"  ! could not parse category from filename {fn}, skipping")
            continue
        by_category.setdefault(m.group(1), []).append(fn)

    total_salvaged = 0
    for category, filenames in by_category.items():
        csvs = _load_category_csvs(category)
        if not csvs:
            print(f"  ! no CSV found for category {category}, skipping {len(filenames)} image(s)")
            continue
        remaining = set(filenames)
        for csv_path, rows, fieldnames in csvs:
            new_rows = []
            for row in rows:
                fn = row.get("filename")
                if fn not in remaining:
                    continue
                edit = salvage[fn]
                new_row = dict(row)
                new_row.update(edit)
                new_filename = f"salvage_{fn}"
                new_row["filename"] = new_filename
                new_rows.append(new_row)
                remaining.discard(fn)
                src_img = IMAGES_DIR / fn
                dst_img = IMAGES_DIR / new_filename
                print(f"  {csv_path.name}: salvage {fn} -> {new_filename}  {edit}")
                if write:
                    if not dst_img.exists():
                        shutil.copy2(src_img, dst_img)
            if new_rows and write:
                bak = csv_path.with_suffix(f".csv.bak.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
                shutil.copy2(csv_path, bak)
                with csv_path.open("a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerows(new_rows)
                print(f"  -> appended {len(new_rows)} row(s) to {csv_path.name} (backup: {bak.name})")
            total_salvaged += len(new_rows) if write else (len(filenames) - len(remaining))
        for fn in remaining:
            print(f"  ! filename {fn} not found in any {category} CSV row")

    print(f"\n{'Would salvage' if not write else 'Salvaged'} {total_salvaged} image(s) as bonus training rows.")
    if not write:
        print("(dry-run -- pass --write to actually copy files and append CSV rows)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classify", choices=list(CATEGORY_PROMPTS), help="Run VLM classification for this category")
    parser.add_argument("--audit-report", type=str, help="Path to the full-audit vlm_batch_report JSON (required with --classify)")
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--apply", type=str, help="Path to a relabel_<category>_*.json classification result (in-place edit)")
    parser.add_argument("--salvage", type=str, help="Path to a relabel_<salvage-category>_*.json classification result (copy + append new row)")
    parser.add_argument("--write", action="store_true", help="Actually modify CSVs/copy files (default: dry-run diff only)")
    args = parser.parse_args()

    if args.classify:
        if not args.audit_report:
            sys.exit("--audit-report is required with --classify")
        run_classify(args.classify, args.audit_report, args.chunk_size, args.max_concurrent, args.poll_interval)
    elif args.apply:
        m = re.search(r"relabel_(\w+?)_\d{8}T\d{6}Z\.json$", Path(args.apply).name)
        if not m:
            sys.exit("Could not infer category from filename; expected relabel_<category>_<timestamp>.json")
        category = m.group(1)
        if category in SALVAGE_CATEGORIES:
            sys.exit(f"{category} is a salvage category -- use --salvage, not --apply")
        corrections = build_corrections(category, args.apply)
        apply_corrections(corrections, write=args.write)
    elif args.salvage:
        m = re.search(r"relabel_(\w+?)_\d{8}T\d{6}Z\.json$", Path(args.salvage).name)
        if not m:
            sys.exit("Could not infer category from filename; expected relabel_<category>_<timestamp>.json")
        category = m.group(1)
        if category not in SALVAGE_CATEGORIES:
            sys.exit(f"{category} is not a salvage category -- use --apply, not --salvage")
        if category == "clean_shaven_salvage":
            no_op, salvage, discard = build_salvage_plan(category, args.salvage)
        else:
            # Plain tier-classification categories (e.g. makeup) being salvaged
            # instead of edited in place: build_corrections() already produces
            # the right {filename: {column: value}} shape for apply_salvage().
            salvage = build_corrections(category, args.salvage)
        apply_salvage(salvage, write=args.write)
    else:
        parser.error("pass --classify <category> --audit-report <path>, --apply <path>, or --salvage <path>")


if __name__ == "__main__":
    main()
