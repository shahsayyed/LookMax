# Post-Generation QA, Audit & Relabeling

`PLAN.md` covers generating the 28,000-image dataset. This document covers
what happened **after** that run finished (2026-09-09 → 2026-09-11): finding
and fixing a critical label-loss bug, auditing every image against its
label with a VLM, and correcting/regenerating what didn't match — the
process that turned "28,000 images exist" into "29,436 images are actually
safe to train on." Read this before touching `raw_generated/` or
`qa_processed/`'s CSVs, or before re-running any part of the audit — the
scripts here are built to be safely re-run, but the reasoning behind *why*
each step exists isn't obvious from the code alone.

---

## 0. Prerequisite: this whole process depends on deterministic reconstruction

Every tool in this document leans on one fact, verified repeatedly: calling
`full_run.py`'s `build_full_task_list()` with the default seed reproduces
the **exact same prompt, negative prompt, and label row** for a given
filename, every time, on any machine — because task generation is a single
seeded RNG consumed in a fixed order. This is the load-bearing fact
behind `rebuild_label_csvs.py`, `verify_dataset_vertex.py`'s prompt
reconstruction, and `relabel_batch.py`'s `classify_flagged()`. If this ever
stops being true (e.g. someone adds a non-deterministic sampling call to
`taxonomy.py`), every tool in this document silently produces wrong output
with no error — there is no runtime check for it, only the byte-for-byte
verification described in step 1 below.

---

## 1. The label-loss discovery

**Symptom**: `raw_generated/labels_<Category>.csv` files only contained
~18% of the rows they should have (5,141 of 28,000). All 28,000 **images**
were present on disk — only their CSV label rows were missing.

**Cause**: several remote Vast.ai hosts were destroyed (spot preemption,
manual cleanup) before their `labels_*.csv` files synced back to this
machine. Same root cause as the note in `verify_dataset_vertex.py`'s
docstring about `generation_log.jsonl` never making it back either.

**Fix — `rebuild_label_csvs.py`**: since label generation is fully
deterministic (§0), the complete label set doesn't need to be recovered
from anywhere — it can be **regenerated from scratch** and will exactly
match whatever did survive. Verified byte-for-byte against all 5,141
surviving rows across all 4 categories (0 mismatches) before trusting it.
Backs up every existing CSV/shard file with a timestamp before overwriting.

```bash
python3 rebuild_label_csvs.py            # dry-run report
python3 rebuild_label_csvs.py --write    # backs up + writes the full 28,000 rows/category
```

If you ever see label CSVs that look incomplete relative to the images on
disk again, this is the fix — not re-syncing from a remote host (which may
no longer exist).

---

## 2. The semantic audit — does the pixel content match the label?

A complete, correct label CSV only helps if the **image actually shows**
what the label claims. `verify_dataset_vertex.py` / `verify_batch.py` audit
this using Gemini (`gemini-3.8-flash` via Vertex AI), reconstructing each
image's exact original prompt via `build_full_task_list()` and asking the
model to judge:

1. `body_type_match` — does build/weight match the prompt (identity axis —
   **not** a trained label, see `taxonomy.py`'s IDENTITY MATRIX comment;
   failing this alone does not corrupt a training label)
2. `face_match` — age/ethnicity/face shape (same: identity, not trained)
3. `attribute_match` — do the **trained** grooming/outfit attributes named
   in the prompt actually show (garment type, color, pattern, the named
   flaw at flaw tiers)? This is the one that matters for label integrity.
4. `severe_anatomy_defect` — extra/missing limbs, multiple people, cropped
   head/feet, watermark, badly wrong hand — image is unusable regardless of
   label correctness.

`verdict = invalid` if any of (1), (3), or (4) fail. **A `body_type_match`-
or `face_match`-only failure is not a real problem** — those axes are never
trained labels, so the image is fine to train on even if the audit flagged
it; this distinction mattered a lot when triaging results (§4).

Two implementations, same rubric:
- `verify_dataset_vertex.py` — synchronous, for small samples (`--sample-size N`)
- `verify_batch.py` — Vertex AI **Batch Prediction** (`--manifest` or
  `--filter`/`--sample-size`), for hundreds+ at once. Chunked
  (`--chunk-size`, default 500) with per-chunk resume caching, submitted
  concurrently (`--max-concurrent`) so per-job queue latency (several
  minutes, mostly fixed cost) is paid once in parallel rather than
  serially.

**Full 28,000-image audit** (already run, report at
`ML/data/vision_synthetic/qa_processed/vlm_batch_report_full_audit_28k_20260909T154320Z.json`):
**22,316 valid, 5,684 invalid** (~$42 in batch mode, ~30-60 min at
`--chunk-size 500 --max-concurrent 10`).

---

## 3. Finding real fixes: the A/B testing methodology

For the categories the audit clustered as failing, the question was
whether a **prompt wording change** actually fixes the underlying
generation problem, or whether it's not fixable via prompting at all (in
which case relabeling is the answer, not regeneration — see §5). This
needed real GPU experimentation, done on an **isolated `vast_test`
instance**, never touching production data.

**`prompt_experiment.py`** — the harness: `find_matching_tasks()` selects a
sample by row-key filter (e.g. `facial_hair_style=clean_shaven`) and/or a
free-text regex over the reconstructed prompt; `--variants-json` tests
several named wording strategies in one pipeline load (avoids reloading the
58GB model per variant); `--skip-baseline` builds a "controlled" experiment
where an already-validated fix for a *different* attribute is baked into
every condition, to isolate one axis from cross-contamination with another
(flaw_severe-tier images stack multiple simultaneous flaw axes — facial
hair, eyebrows, skin, makeup — randomly, so an eyebrows experiment's raw
numbers can be drowned out by an unrelated facial-hair problem in the same
sample unless controlled for).

**What was actually learned about the mechanism** (this generalizes beyond
any one attribute, worth knowing before trying to fix something new):
- **Negative-prompt reinforcement** works for "suppress an unwanted
  element" attributes (e.g. clean-shaven: reinforcing *against* facial hair
  in the negative prompt, not describing bareness in the positive prompt).
- **Positive-prompt reinforcement**, often needing *repeated/emphatic*
  phrasing (not just stated once), works for "elicit a specific positive
  detail" attributes (e.g. makeup clashing colors, unibrow).
- Reinforcement of **either kind backfires** for some texture/rendering
  attributes (pattern miss, patchy-vs-dense facial hair) where the
  unreinforced baseline is already the best available option — more
  emphasis just pushes the model further from what was asked.
- Token fragmentation (BPE) is **not** a reliable predictor of rendering
  difficulty — empirically disconfirmed. ALL-CAPS emphasis phrasing *does*
  fragment badly and correlated with the worst-performing wording variant
  tried.
- `tokenizer_probe.py` / `TOKENIZER_PROBE.md` are useful for a cheap
  pre-flight sanity check (catching obviously-bad tokenization before
  spending GPU time) but are **not a substitute** for the empirical A/B
  loop above — a clean tokenization doesn't predict a good render.

### Results, by category
| Category | Outcome | Fix |
|---|---|---|
| `clean_shaven` (facial hair) | **Fixed** | Negative-prompt reinforcement |
| `pattern_solid_bleed` (solid requested, pattern shows) | **Fixed** | Negative-prompt reinforcement |
| `makeup` (clashing colors, flaw_severe) | **Fixed** | Repeated positive-prompt reinforcement |
| `eyebrows` (unibrow, flaw_severe) | **Fixed** | Positive-prompt reinforcement |
| `skin_condition` (dry/flaking, flaw_severe) | **Fixed** | Negative-prompt reinforcement |
| `pattern_miss` (printed requested, solid shows) | Not fixable via prompting | Relabel instead (§5) |
| `facial_hair_texture` (patchy vs dense, flaw_severe) | Not fixable via prompting | Relabel instead (§5) |
| `double_chin`, `pullover_hoodie` | Non-issue | Not trained labels / no such distinction in schema |
| `garment_color`, `hair_condition`, `fit_tight` | Non-issue | Original clustering signal was cross-attribute contamination, not a real failure |

The 5 validated fixes are **already applied** in production `taxonomy.py` /
`prompt_builder.py` (the `TARGETED PROMPT-ENGINEERING FIXES` section and the
per-task `negative_prompt` wiring in `prompt_builder.py`'s
`build_grooming_task()`/`build_outfit_task()`) — any *future* generation
run already benefits from these without further action.

---

## 4. Classifying all 5,684 invalid images

Clustering `attribute_match=false` issue text is not just "count the
category failures" — a large fraction of raw failures turned out to be
either identity-axis noise (§2) or non-issues once cross-attribute
contamination was filtered out. The methodology, applied iteratively:

1. Split off `body_type_match`/`face_match`-only failures — never a real
   label problem (§2). **1,330 images.**
2. For the rest, filter each image's issue text to keyword sets scoped to
   **one attribute at a time**, so a co-occurring unrelated problem (e.g.
   an eyebrows issue mentioned alongside an unrelated pattern issue) isn't
   misattributed.
3. Iteratively refine keyword lists against real sample clauses — several
   rounds were needed (e.g. "body build" vs "body type" phrasing, generic
   "patterned" vs specific pattern-name mentions, Grooming-only hair being
   real vs Outfit hair being `meta`/non-issue) until the residual
   "unclassified" bucket converged to near-zero.

**Full disposition of all 5,684:**
| Bucket | Count | Treatment |
|---|---|---|
| Recoverable to valid (identity/meta noise only) | 1,911 | No action — label was never wrong |
| Relabel in-place (11 categories, §5) | ~2,979 (net of overlap) | `relabel_batch.py --apply` |
| Salvage + regenerate (clean_shaven, makeup) | 1,460 | `relabel_batch.py --salvage` + `regenerate_targeted.py` (§6) |
| Severe anatomy defects | 17 | Excluded from training (`qa_pass=0`, §7) |
| Genuinely unresolved residual | ~18 | Accepted as negligible (0.06% of dataset) |

---

## 5. Relabeling — `relabel_batch.py`

For any category where the image genuinely shows something different from
its label, but **that different thing is still a valid, existing label
value** (e.g. shows "solid" when "printed" was requested — "solid" is
already a real class), the fix is to correct the label, not regenerate the
pixels. Whether relabeling is safe depends on **population math, not just
whether it "seems minor"**: relabeling only makes sense when the affected
fraction is small relative to that attribute's total population — if
95% of a class's population is wrong (as `clean_shaven` was), relabeling
would wipe out the class entirely and you *must* regenerate instead (§6).

**How it works**: for each category, a VLM classification pass (Batch
Prediction, same infra as §2 but a classification prompt instead of a
verdict prompt) is run over the flagged images, asking "what does this
attribute *actually* show" (never "does it match"). The response is mapped
back to CSV columns three ways:
- **Categorical** (e.g. pattern, footwear type, hair length): write the
  classified value directly.
- **Tier-derived** (e.g. eyebrows/skin/facial-hair *condition* severity):
  classify which of the 4 tier-phrase buckets (flaw_severe/flaw_mild/
  average/polished) the pixels actually match, then re-derive the label via
  `taxonomy.severity_for_tier()`/`positive_for_tier()` — the *same*
  functions that produced the original (wrong) label, just fed the tier the
  pixels show instead of the tier that was requested.
- **Combined style+tier** (facial hair style specifically): both an
  identity value (clean_shaven/stubble/short_beard/...) and, if not
  clean-shaven, a condition tier.

13 categories were run this way in total (11 in-place, 2 via the salvage
path — see below): `pattern_miss`, `facial_hair_texture`, `eyebrows`,
`skin_condition`, `pattern_solid_bleed`, `hair_length_grooming`,
`facial_hair_style_miss`, `fit_miss`, `fabric_wrinkled_miss`,
`footwear_type_men`, `footwear_type_women` (in-place), plus
`clean_shaven_salvage` and `makeup` (salvage — §6). A later pass,
`clean_shaven_residual`, relabeled the 297 images still wrong *after*
regeneration (§6) — safe by then since the class was no longer critically
depleted (75.1% correct post-regen).

```bash
python3 relabel_batch.py --classify <category> --audit-report <path>   # Vertex classification, ~$1.50/1000 images
python3 relabel_batch.py --apply <result.json>                          # dry-run diff
python3 relabel_batch.py --apply <result.json> --write                  # actually edit the CSV (backs up first)
```

3,047+ label values were corrected this way across the whole dataset, zero
images regenerated for these categories.

---

## 6. Salvage + regenerate — `clean_shaven` and `makeup`

These two categories were too depleted by their failure rate for
relabeling alone (`clean_shaven`: 95.3% of its class was wrong;
`makeup`-clash: 44.4%) — relabeling would have redistributed the class into
other already-well-represented values instead of restoring what was
missing. These **do** get regenerated with the now-fixed prompts (§3). But
before overwriting, every flagged image is first **salvaged**: classified
(same style+tier or tier-only classification as §5) and, if it resolves to
a real, confident label, **copied to a new filename** (`salvage_<original>`)
with the corrected row **appended** as a bonus training example — nothing
usable is thrown away just because it wasn't the specific thing asked for.
Three outcomes per image: audit was itself wrong (kept as-is, no copy
needed) / real different content (salvaged as bonus data) / VLM can't tell
(discarded, only case where pixels aren't reused).

```bash
python3 relabel_batch.py --classify clean_shaven_salvage --audit-report <path>
python3 relabel_batch.py --salvage <result.json> --write   # copies files + appends new CSV rows
```

Result: **1,460 bonus salvaged images** added (1,137 clean_shaven +
323 makeup), on top of the original 28,000.

### Regenerating the 1,460 originals — `regenerate_targeted.py`

Deliberately **not** `full_run.py`'s normal resume path (which discovers
work by scanning the whole `images/` directory for what's missing — wrong
model for "regenerate exactly these 1,460 and nothing else"). Instead:
- Takes an **explicit filename whitelist** (`regen_targets.txt`), never a
  directory scan.
- Every filename must match the deterministic task list (§0) before
  anything runs — refuses to proceed on a stale/wrong target list.
- **Never requires the target files to pre-exist** (`--require-existing`
  defaults off) — safe to point at a completely fresh, empty compute box;
  no need to upload the (soon to be replaced) old images there first.
- Writes via the same `<filename>.tmp` → atomic `os.replace()` pattern as
  `full_run.py`, and **never opens a CSV** — the label was already correct
  (the fix only changes the generation prompt, never the target label), so
  only pixels need to change.

Companion **`pull_regenerated.py`** downloads completed images back as
they're generated (`--watch N` polls every N seconds), restricted to the
same whitelist, using `rsync -rtz --checksum` — deliberately **not**
`-a` (archive mode tries to preserve file ownership, which can never
succeed pulling from a root-owned Linux box to a macOS user account, and
silently made rsync re-"transfer" — and mis-report — already-correct files
on every poll; content+mtime comparison only, checksum-based since
size+mtime quick-check separately under-detected real changes in this
pipeline). Progress is tracked by files rsync actually reports transferring
this session (`--out-format=%n`), not by local file existence — the
target files already exist locally under their old, wrong pixels before
any pull happens, so an existence check falsely reads "already done"
immediately (a real bug hit and fixed during this run).

### Re-verification result
Ran the same audit rubric (§2) against all 1,460 regenerated images:
- **makeup: 100% on-target valid** at production scale.
- **clean_shaven: 75.1% on-target valid overall**, but with a sharp tier
  split — 79-85% at flaw_mild/average/polished, only **32.8% at
  flaw_severe** (that tier stacks the clean_shaven fix against the
  eyebrows and skin_condition reinforcements added at the same tier,
  competing for the model's attention). The 297 residual failures were
  relabeled per §5 rather than attempting a 3rd regeneration round.

---

## 7. Final assembly — the training-ready manifest

The training pipeline (`ML/vision/training/multihead_common.py`'s
`discover_synthetic_source()`) prefers
`qa_processed/labels_<Category>_measured.csv` (with a `qa_pass` gate) over
`raw_generated/labels_<Category>.csv` (unfiltered). Two things had to
happen to produce that file correctly:

1. **`extract_measured_labels.py`** run over each corrected
   `raw_generated/labels_<Category>.csv` — this is *pixel*-level QA (blur,
   brightness, face-count, cropped-feet proxy, measured-vs-requested color
   binding), entirely separate from and blind to the semantic audit above.
   It does **not** know about severe anatomy defects (those don't
   necessarily fail blur/brightness/face-count).
2. So: the 17 severe-anatomy-defect filenames (§4) are explicitly forced
   to `qa_pass=0` afterward — nothing else in the pipeline would have
   caught them.

```bash
python3 extract_measured_labels.py raw_generated/labels_<Category>.csv raw_generated/images \
    --output qa_processed/labels_<Category>_measured.csv
# then force qa_pass=0 for the 17 known anatomy-defect filenames (see the module docstring)
```

**Final numbers** (as of 2026-09-10):

| Category | Total rows | `qa_pass=1` |
|---|---|---|
| Men_Grooming | 7,137 | 7,136 (99.99%) |
| Women_Grooming | 6,323 | 6,318 (99.92%) |
| Men_Outfit | 8,000 | 7,991 (99.89%) |
| Women_Outfit | 8,000 | 7,991 (99.89%) |
| **Total** | **29,460** | **29,436 (99.92%)** |

(7,137/6,323 > 6,000 because of the 1,460 salvaged bonus rows from §6, all
in the Grooming categories.)

One known gap: face-count QA (`extract_measured_labels.py`'s
`face_detected`/`multiple_faces` check) was skipped for this entire run —
this machine's OpenCV build is missing `cv2.CascadeClassifier`. Non-fatal
by design (documented in that script's own module docstring) — every
other check still ran. Worth re-running with a proper OpenCV build if
face-count QA specifically matters later.

---

## Tool reference (this document's scripts, in the order they'd run)

| Script | Purpose |
|---|---|
| `rebuild_label_csvs.py` | Deterministically regenerates complete label CSVs from the task list (§1) |
| `verify_dataset_vertex.py` | Synchronous VLM audit, small samples (§2) |
| `verify_batch.py` | Vertex Batch Prediction VLM audit, hundreds+ at once (§2) |
| `prompt_experiment.py` | A/B prompt-wording testing harness, isolated GPU box (§3) |
| `tokenizer_probe.py` | Cheap pre-flight tokenization sanity check — not a substitute for `prompt_experiment.py` |
| `relabel_batch.py` | Classify-then-correct CSV labels in place, or salvage+copy before regeneration (§5, §6) |
| `regenerate_targeted.py` | Regenerates an explicit filename whitelist, touches nothing else (§6) |
| `pull_regenerated.py` | Downloads regenerated images back as they complete, checksum-verified (§6) |
| `extract_measured_labels.py` | Pixel-level QA gate + measured color binding — produces the final `qa_pass` column (§7) |
