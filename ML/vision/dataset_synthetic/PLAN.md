# Dataset Generator: Qwen-Image-2512 Synthetic Image Pipeline

This is the **v8** synthetic dataset generator for LookMax, producing the
**28,000-image** training set (Men_Grooming 6,000 / Women_Grooming 6,000 /
Men_Outfit 8,000 / Women_Outfit 8,000) using **Qwen-Image-2512**
(`diffusers` `QwenImagePipeline`) instead of FLUX.1 [dev]. Qwen won
decisively against Flux and Google Nano Banana Pro on prompt adherence for
"flaw"-tier images in prior head-to-head testing (see
`ML/archive/dataset_generator_v7/`) — it holds up on negative/flaw
descriptors that Flux and Lightning-distilled models soften away.

The previous pipeline (taxonomy v2-v7, FLUX.1 [dev] and an early Qwen
attempt) is archived in full at `ML/archive/dataset_generator_v7/` — read
it before changing anything here. It contains real, hard-won operational
fixes this pipeline depends on (see "Lessons carried forward" below), not
just historical clutter.

---

## Scripts

| Script | Purpose |
|---|---|
| `taxonomy.py` | Single source of truth for every sampling axis — tiers/score bands, identity matrix, environment (background/lighting/framing), colour palette, garment slots + coherent outfit sampling, grooming condition axes, negative prompt, image budget, and the label schema generator. Nothing else hand-duplicates a list from here. |
| `prompt_builder.py` | Composes one (prompt, resolution, label row) per image from `taxonomy.py`'s axes. Clause-structured prompts (see "Prompt style" below). |
| `qwen_pipeline.py` | GPU-aware model loading (VRAM auto-detect: full-resident vs CPU-offload) + the one `generate()` every GPU-touching script shares. |
| `smoke_test.py` | `--dry-run` (prompts + labels, no GPU) and `--per-tier` (16 real images, one per category x tier, with a manifest). Run both before anything else. |
| `quick_prompt_test.py` | The FIRST real-GPU check on a new box: 6 hardcoded prompts (not built from the taxonomy), one at a time. Fastest possible "does the model even load and generate here" confirmation, and a stable reference set for manually comparing against another model. |
| `variation_test.py` | `--dry-run` (no GPU) and the real run (64 images by default: 4 categories x 4 tiers x 4 samples/cell, real taxonomy sampling). Deeper diversity/quality gate than `smoke_test.py --per-tier` — checks variation WITHIN a tier, not just the gradient across tiers. Run right before `full_run.py --benchmark`. |
| `validation_sweep.py` | `--coverage-only` (simulate the full 28,000-image plan, no GPU, flag any class under 250 examples) and `--check-binding` (measure colour adherence on already-generated images). The gate before `full_run.py`. |
| `full_run.py` | The production run. Sharding, resume, atomic writes, `--benchmark`. |
| `merge_shards.py` | Combines per-shard label CSVs into one CSV per category; warns (doesn't silently double-count) on duplicate filenames. |
| `extract_measured_labels.py` | Bucket C: post-generation pixel-measured colour/QA columns, added without deleting rows. |
| `install.sh` | Remote GPU box setup — disk-safety check, torch-presence check, dependency install. |

---

## Lessons carried forward from `ML/archive/dataset_generator_v7/`

These are real incidents from running the previous pipeline, not
precautions taken "just in case" — read them before assuming a shortcut
is safe.

### 1. Vast.ai disk quirk: `/workspace` is tiny, the large disk is elsewhere
On Vast.ai images, `/workspace` is commonly mapped to a small (~10GB) loop
device, while the disk you actually paid for is mounted at `/` or `/data`
depending on the template. `/workspace` is also the default directory you
land in over SSH — an easy trap. `install.sh` and `full_run.py` both
check the ACTUAL directory in play (`$LOOKMAX_DATA_DIR` / `--data-dir`),
never the cwd, and abort with a clear message before downloading or
generating anything if there isn't enough room.

**`full_run.py`'s own default output directory is a LOCAL folder next to
the script**, not a hardcoded `/data` — this is a deliberate deviation
from the archived script, made so `full_run.py --dry-run` (and this
project's acceptance check) works out of the box on a laptop with no
`/data` mount. **On an actual remote GPU box, you must set this
explicitly**, every session:
```bash
export LOOKMAX_DATA_DIR=/data
```
or pass `--data-dir /data` on every `full_run.py` / `merge_shards.py`
invocation. `install.sh` prints this reminder at the end of setup.

### 2. tmux + shell env: don't trust either across a reattach
Three separate incidents on these boxes proved you cannot trust:
- **The shell environment** — some auto-tmux setups spawn fresh login
  shells that don't reliably inherit a `.bashrc` export, and at least one
  container image sets its own default (`HF_HOME=/workspace/.hf_home`,
  pointing straight at the tiny disk).
- **The script's own location** — if the repo is ever cloned into
  `/workspace` instead of `/data` (easy, since `/workspace` is the SSH
  landing directory), "cache next to wherever this script lives" would
  silently reproduce the same bug.

So: **set `HF_HOME` and `LOOKMAX_DATA_DIR` explicitly, in the SAME shell
you run the Python scripts from, every time you attach or reattach** —
don't assume a previous session's exports survived:
```bash
tmux new -s qwengen        # or: tmux attach -t qwengen
export HF_HOME="/data/huggingface_cache"
export HF_XET_HIGH_PERFORMANCE=1
export LOOKMAX_DATA_DIR=/data
```
The full generation run takes on the order of days (see the benchmark
step below for this run's actual GPU-hour projection) — always inside
tmux, never a bare foreground shell that dies on disconnect.

### 3. GPU VRAM auto-detection, not a hardcoded assumption
`qwen_pipeline.py` checks the ACTUAL GPU on the current machine at
runtime (`torch.cuda.get_device_properties`), not whichever card was used
when the script was last edited:
- **≥ 80GB VRAM** (e.g. RTX PRO 6000 Blackwell 96GB): full pipeline
  resident (`.to("cuda")`, no offload) — faster, and technically able to
  batch.
- **< 80GB VRAM** (e.g. RTX 6000 Ada 48GB — confirmed OOM on `.to("cuda")`
  in earlier testing): `enable_model_cpu_offload()`, which forces batch
  size 1 (only the actively-running component is ever resident).

### 4. Measured: batching gave NO throughput benefit on this model
On an RTX PRO 6000 Blackwell (96GB, full-resident mode): batch=1 ran
~0.64s/step/image; batch=4 ran ~0.73-0.75s/step/image. **Batching was
slightly WORSE, not better** — a 20B-parameter transformer at 1024x1024
already saturates that GPU's compute at batch=1, so there's no idle
capacity for batching to exploit (unlike lighter models). This is why
`full_run.py`'s default `gen_batch_size` is **1**, based on a real
measurement, not caution for its own sake. If you're on different
hardware, re-check with `python3 full_run.py --benchmark` before assuming
a higher batch size helps — it might, on a card with a bigger gap between
compute throughput and this model's per-image compute need.

### 5. Dataset size does not affect on-device inference speed
To head off a predictable confusion: generating 28,000 images (vs. the
archived pipeline's 24,000) has **zero** effect on how fast the trained
CoreML model runs on a user's iPhone. On-device inference speed is a
function of the exported model's backbone architecture and quantization
(`ML/pipeline/04_train_coreml_models.py`'s `BACKBONE`/`compute_precision`
settings), entirely unrelated to how many training images went into it.
More training data can change accuracy, never runtime latency.

### 6. Why prompts don't need FLUX's CLIP-77-token workaround
The archived pipeline's biggest single bug (see its `PLAN.md`'s "Known
Issues" table) was CLIP's hard 77-token limit silently truncating flaw
descriptions before they even started, because FLUX.1 uses CLIP for part
of its conditioning. **Qwen-Image-2512 uses Qwen2.5-VL-7B as its text
encoder** — a real instruction-following LLM, not a 77-token bag-of-words
encoder — so `prompt_builder.py`'s clause-structured, multi-line prompts
(one labelled body-region/attribute per line) are the right shape here
and don't need FLUX's "short opener that fits in 77 tokens" workaround.
Still keep each clause's tier phrasing SHORT rather than repeating the
same idea across clauses — that's about not wasting the encoder's
attention on restating one idea, not a hard token budget.

---

## Prompt style

Clause-structured, one labelled body-region/attribute per line:
```
A full-body photograph of a 35-year-old Caucasian man, athletic, oval face,
high cheekbones, body proportions unchanged from their natural athletic
figure, not slimmer or heavier than that, with short black hair, standing
facing the camera with the whole body from head to shoes visible.
Upper body: wearing a solid navy chambray button-up shirt, crisp and
freshly pressed, with no visible wrinkles, the fit tailored to the body
with clean, flattering proportions.
Outer layer: wearing a grey tailored blazer, crisp and well-kept.
Lower body: wearing solid black tailored trousers, crisp and freshly
pressed, with no visible wrinkles, the fit tailored to the body with
clean, flattering proportions.
Footwear: black oxford shoes, clean and freshly polished.
Overall styling: the whole outfit looking sharp and intentionally put
together.
Setting: standing against a plain neutral-colored wall, bright even studio
lighting, centered in frame, straight-on eye-level angle.
Photorealistic candid photograph, natural skin texture, sharp focus, 85mm
lens, full body in frame.
```

The **body-proportions clause** right after the identity is a deliberate
addition beyond a bare template, ported forward from a real, documented
bug: `ML/README.md`'s "taxonomy v4" note describes "polished" styling
language visibly slimming heavy-set/plus-size/curvy identities relative
to their own flaw-tier render, because styling text reshapes body
proportions through cross-attention even with an otherwise-fixed identity
description. Restating the build early (not trailing) is the fix that
worked before; there's no dedicated taxonomy axis for it, so
`prompt_builder.py` adds it directly.

`taxonomy.NEGATIVE_PROMPT` deliberately contains **no flaw words**
("wrinkled", "greasy", "patchy", etc.) — negating those globally would
destroy the entire flaw tier. It targets deformed hands/extra limbs,
multiple people, cropped head/feet, text/watermark, cartoon/3D render,
and plastic/airbrushed skin instead.

**No LoRA, no Qwen-Image-Lightning distillation anywhere in this
pipeline** (verified absent — grep for "lightning\|lora" across `ML/`
turns up nothing but this very sentence and an unrelated "Flora" match).
Few-step distilled models lose prompt adherence worst on negative/flaw
attributes first, since distillation pulls generation toward the model's
aesthetic mode — exactly the failure this dataset can't afford at the low
end of the score range.

---

## Running the full pipeline, in order

### 1. Setup (remote GPU box)
```bash
mkdir -p /data && cd /data
git clone <repo-url> /data/LookMax
cd /data/LookMax/ML/pipeline/dataset_generator
bash install.sh
```

### 2. Cheap sanity checks first — no GPU
```bash
python3 -m py_compile *.py
python3 smoke_test.py --dry-run
```
Read every one of the 16 printed prompts. This is the cheapest point to
catch an implausible garment/colour pairing or an off-tone effort phrase
— fixing it here costs nothing; catching it after a GPU run costs real
money and time.

### 3. Coverage sweep — no GPU, before ANY GPU time is spent
```bash
python3 validation_sweep.py --coverage-only
```
Simulates the exact same deterministic 28,000-task list `full_run.py`
will use and flags any class under 250 examples. Must show
"no class under 250 examples" before proceeding.

### 4. Six hardcoded prompts — the FIRST real GPU touch on a new box
```bash
tmux new -s qwengen
export HF_HOME="/data/huggingface_cache"
export HF_XET_HIGH_PERFORMANCE=1
export LOOKMAX_DATA_DIR=/data
python3 quick_prompt_test.py --output-dir /data/quick_prompt_test_output
```
These 6 prompts are hardcoded, not built from `taxonomy.py` — deliberately
standalone, so a taxonomy/prompt-builder bug can never mask (or be masked
by) a base-model or environment problem. This is the cheapest, fastest
place to discover "the model doesn't load" or "generation errors out on
this box" — a few minutes, not partway through a 64-image or 28,000-image
run. Check `01` vs `02` for the flaw severity gradient, and `05` vs `06`
for polished-outfit diversity (not a jacket every time).

### 5. Sixteen real images — one per category x tier
```bash
python3 smoke_test.py --per-tier
```
Review `smoke_test_output/images/` side by side. flaw_severe / flaw_mild /
average / polished must read as **three visually distinct groups**, not a
blur of near-identical images — Qwen-2512's improved realism (relative to
FLUX) can silently soften grease/wrinkles/bad fit toward its aesthetic
prior. That failure is silent (the label still says `flaw_severe`) and
would poison exactly the end of the scale this product depends on most.

### 6. Colour-binding check on those 16 images
```bash
python3 validation_sweep.py --check-binding smoke_test_output/manifest.csv smoke_test_output/images
```
Gate: combined colour match ≥ 70%. Garment-type (≥80% target) and pattern
(≥65% target) adherence are **not** automatically measurable from pixels
alone — use the manual review checklist this command also prints.

### 7. Sixty-four real images — the deepest diversity/quality gate
```bash
python3 variation_test.py --dry-run    # read the prompts first, no GPU
python3 variation_test.py               # 64 real images (4 categories x 4 tiers x 4 samples/cell)
```
This is `smoke_test.py --per-tier` generalized: instead of ONE image per
category x tier, it generates several, so you can see diversity WITHIN a
tier (different identities, garments, colours, environments), not just the
gradient across tiers. Review `variation_test_output/images/` and
`manifest.csv` against the checklist the script prints at the end
(severity gradient across several samples, no jacket monoculture in
polished, backgrounds/lighting varying independently of tier). This is the
last visual gate before spending real GPU time on the full run — if
something is going to be systematically wrong across the full 28,000
images, this is where it's cheapest to catch it.

### 8. Benchmark on THIS hardware
```bash
python3 full_run.py --benchmark
```
Times 5 images (discards the first as warm-up), projects total GPU-hours
for all 28,000 images, and — if you pass `--num-shards N` — prints a
per-shard time table. Don't skip this: the "no benefit from batching"
finding above was measured on ONE specific GPU, not guaranteed on yours.

### 9. The full run
Single machine, unattended until done:
```bash
tmux new -s qwengen
export HF_HOME="/data/huggingface_cache"
export LOOKMAX_DATA_DIR=/data
python3 full_run.py
```
Split across N machines/processes (each takes disjoint task indices,
`index % num_shards == shard`):
```bash
python3 full_run.py --shard 0 --num-shards 4   # on worker 0
python3 full_run.py --shard 1 --num-shards 4   # on worker 1
# ...
```
Safe to stop (Ctrl+C) and restart anytime — resume is based on which
`.png` files already exist under `images/`; the task list itself is
deterministic (seeded), so index N always means the same prompt/labels on
every run. Each image is written to a `.tmp` name and only renamed after
its CSV row is flushed, so a killed process can never leave a
half-written file that looks done and gets silently skipped forever.

Pass a `target_count` to generate a bounded number of new images per
invocation (useful for looping in short sessions):
```bash
while python3 full_run.py 500; do :; done
```

### 10. Merge shards (if you sharded)
```bash
python3 merge_shards.py --data-dir /data
```
Warns loudly (rather than silently double-counting) if two workers were
accidentally given the same `--shard` value.

### 11. Measure pixel-level labels (Bucket C)
```bash
python3 extract_measured_labels.py \
    /data/qwen_dataset_output/labels_Men_Outfit.csv \
    /data/qwen_dataset_output/images \
    --output /data/qwen_dataset_output/labels_Men_Outfit_measured.csv
```
Repeat per category. Adds measured colour, colour-match, colour-harmony,
and QA-gate columns without deleting any row — an over-strict QA gate
could silently skew the trained score distribution more than the bad
images it flags. If the QA pass rate is below ~85%, the script prints a
warning; inspect `qa_reasons` before discarding anything downstream.

### 12. Bring the finished dataset back to this machine

Everything above runs on the remote GPU box under `$LOOKMAX_DATA_DIR`
(e.g. `/data/qwen_dataset_output/`). Once a run (or a shard) is complete
and measured, copy it back to this repo's local convention — a sibling of
the real-data-pipeline's numbered stages, kept clearly separate by name so
the two are never confused:

```bash
# from your local machine
rsync -avz -e "ssh -p <port>" \
    root@<instance-host>:/data/qwen_dataset_output/ \
    ML/data/4_Synthetic_Qwen/raw_generated/

# after running extract_measured_labels.py (step 11), the *_measured.csv
# files and any QA-flagged-but-kept images go here instead:
rsync -avz -e "ssh -p <port>" \
    root@<instance-host>:/data/qwen_dataset_output/*_measured.csv \
    ML/data/4_Synthetic_Qwen/qa_processed/
```

`ML/data/1_Raw_Scrapes/`, `2_VLM_Processing/`, and `3_CoreML_Training_Data/`
are real photos from the separate scraping pipeline (see
`ML/pipeline/real_data_pipeline/`) — `4_Synthetic_Qwen/` is deliberately a
new, distinctly-numbered sibling rather than reusing 1-3, so synthetic and
real data can never be silently merged or mistaken for each other.
Combining the two into one training-ready set (if that's ever wanted) is
a deliberate future step, not something either pipeline does implicitly.

---

## Output layout

```
$LOOKMAX_DATA_DIR/qwen_dataset_output/
├── images/                              <- all categories, one shared dir
│   ├── 00000_Men_Grooming_polished.png
│   ├── 00001_Women_Outfit_flaw_mild.png
│   └── ...
├── label_schema_Men_Grooming.json       <- generated FROM taxonomy.py
├── label_schema_Women_Grooming.json
├── label_schema_Men_Outfit.json
├── label_schema_Women_Outfit.json
├── labels_Men_Grooming.csv              <- one CSV per category (different
├── labels_Women_Grooming.csv               head sets -- see below)
├── labels_Men_Outfit.csv
├── labels_Women_Outfit.csv
└── generation_log.jsonl                 <- full provenance: exact prompt,
                                             seed, steps, cfg scale, timestamp
```
(Sharded runs additionally produce `labels_<Category>_shard<k>.csv` per
worker until `merge_shards.py` combines them.)

**Why one CSV per category, not one combined CSV**: each category has a
different head set (grooming has hair/skin/eyebrows/facial-hair-or-makeup;
outfit has garment-slot/pattern/fit/formality columns). A single wide CSV
would be mostly empty cells and would obscure which columns a given
category's model head actually trains on.

**Label schema contract**: `label_schema_<Category>.json` — a list of
`{"name", "type", ..., "loss_weight"}` objects. Types: `regression`
(score, weight 1.0), `ordinal` (3 levels 0/1/2, weight 0.3), `categorical`
(weight 0.3; `formality` 0.5), `meta` (not trained — provenance/
pixel-snapping only, e.g. `requested_upper_color`). Generated by
`taxonomy.get_label_schema()`, never hand-duplicated.

---

## Known issues & fixes

| Issue | Cause | Fix |
|---|---|---|
| `CUDA out of memory` on `.to("cuda")` | Card has < ~80GB VRAM, can't hold the ~58GB pipeline resident | `qwen_pipeline.py` auto-detects this and uses `enable_model_cpu_offload()` instead |
| `full_run.py --dry-run` tries to create `/data` and fails on a laptop | Default output dir isn't `/data` (deliberately, see "Lessons" #1) — but if you set `LOOKMAX_DATA_DIR=/data` on a machine without that mount, it'll still try | Only set `LOOKMAX_DATA_DIR=/data` on a box that actually has `/data`; leave it unset for local testing |
| `No space left on device` mid-download | `HF_HOME` defaulted to a small disk | `export HF_HOME="$LOOKMAX_DATA_DIR/huggingface_cache"` explicitly, every session (see "Lessons" #2) |
| A resumed run seems to skip images that were never actually generated | Task list wasn't built with the same seed, so "index N" meant something different between runs | Never call `build_full_task_list()` with a non-default seed; it exists precisely so index N is stable |
| Two shard CSVs contain the same filename | Two workers were given the same `--shard` value | `merge_shards.py` warns and keeps only the first occurrence — check the printed list and re-run the affected shard with the correct index |
| QA pass rate looks surprisingly low in `extract_measured_labels.py` | Could be a real generation problem, OR an over-strict gate (see its module docstring) | Read `qa_reasons` per failing row before concluding the images are bad — rows are never deleted, so this is always inspectable after the fact |
| `face_detected`/`multiple_faces` QA never fires, cascade warning printed | Some OpenCV builds/environments ship without `cv2.CascadeClassifier` wired up (seen on at least one dev machine while building this pipeline) | Non-fatal by design — `extract_measured_labels.py` skips just that one check and still runs colour/blur/brightness/cropped-feet checks; reinstall `opencv-python-headless` cleanly (no conflicting `opencv-python` alongside it) if you need face QA specifically |
