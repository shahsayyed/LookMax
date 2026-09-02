# Dataset Generator: FLUX.1 [dev] Synthetic Image Pipeline

This folder contains all scripts for generating the **24,000-image synthetic training dataset** for LookMax using FLUX.1 [dev].

---

## Scripts

| Script | Purpose |
|---|---|
| `generate_flux_dataset.py` | Master generation script — generates all 24,000 images and writes `labels.csv` |
| `test_flux_prompts.py` | Quick 4-image sanity check (one per model category) |
| `test_flux_variations.py` | Full 48-image demographic + level validation test |
| `requirements.txt` | Python dependencies for the remote GPU server |

---

## Remote Server Setup (Vast.ai)

We run generation on a **rented GPU server** via [Vast.ai](https://vast.ai) because FLUX.1 [dev] requires ~24 GB VRAM and takes ~19 seconds per image at 1024×1024 resolution.

### Hardware Used
- **GPU:** RTX 5090 (32 GB VRAM)
- **VRAM Note:** The full FLUX.1 [dev] pipeline requires ~33 GB when the T5 text encoder is included. We use `pipe.enable_model_cpu_offload()` to keep the image generator on GPU (24 GB) and offload the text encoder to system RAM. This is the maximum safe configuration for a 32 GB card.
- **Disk:** 150 GB allocated (IMPORTANT: see disk quirk below)
- **OS/Image:** `vastai/pytorch:cuda-12.8.1-auto` (Jupyter interface)

### Critical Disk Quirk on Vast.ai
The host mounts the large disk to `/` (root), but maps `/workspace` to a tiny 10 GB loop device. **Do NOT use `/workspace` for model weights or outputs.** Use `/data` instead:

```bash
# Create the working directory on the large drive
mkdir /data
cp -r /workspace/dataset_generator /data/
cd /data/dataset_generator
```

### Required Environment Variables
```bash
export HF_TOKEN="hf_your_token_here"         # HuggingFace access token (FLUX.1 [dev] is gated)
export HF_HOME="/data/huggingface_cache"      # Redirect 33GB model download to large drive
export HF_XET_HIGH_PERFORMANCE=1              # Enable Xet high-speed multi-threaded download
```

> **Note on HF_TRANSFER:** The older `HF_HUB_ENABLE_HF_TRANSFER=1` env var is deprecated. Use `HF_XET_HIGH_PERFORMANCE=1` which uses the newer Xet downloader achieving 40–80 MB/s vs the default ~2 MB/s single-stream.

### Install Dependencies
```bash
pip install -r requirements.txt
```
> `requirements.txt` intentionally does NOT pin a PyTorch version. The Vast.ai image ships with an optimized CUDA build of PyTorch — overwriting it breaks the GPU drivers. We install only the missing HuggingFace libraries on top.

---

## Running the Generation

### Step 1: Test First (Required)
Always validate prompts before committing to the full 24,000-image run:

```bash
# Quick 4-image test (one per model category)
python3 test_flux_prompts.py

# Full 48-image demographic + level validation (taxonomy v3)
python3 test_flux_variations.py
```

> **v6 taxonomy check:** `test_flux_variations.py` now writes to `test_variations_comprehensive_v6/`. While v5 was generating, its own console logs exposed the real root cause of the persistent flaw-adherence weakness: `"The following part of your input was truncated because CLIP can only handle sequences up to 77 tokens"` — on every single image. v3 through v5 each added more reinforcement text and pushed prompts to 180-220+ CLIP tokens; T5 (512-token budget) saw everything, but CLIP (hard 77-token cap) was silently losing most or all of the flaw/effort description. Confirmed with the real `CLIPTokenizer`: a v5 Outfit prompt's CLIP view cut off *before the outfit description even started*.
>
> v6 fixes this by leading every prompt with a short "opener" (core flaw/effort keywords, plus body preservation for outfits) verified to fit completely inside CLIP's 77-token window, even for the longest identity combinations. This should be the most impactful single fix of the whole taxonomy-iteration process — v4/v5's targeted fixes (body-shape reinforcement, evaluative closers) were reasonable ideas but were fighting a token-budget problem underneath them the whole time.
>
> For each identity's triplet in the v6 batch, check:
> - **Is the flaw obviously visible now for every identity**, including the ones that were weakest before (Black man, South Asian man, Hispanic woman)? This is the main thing v6 targets.
> - **Is body shape/size still consistent across all three tiers?**
> - **Is Average still as sharp as Flaw and Polished?**
>
> If grooming flaw visibility is *still* weak for the same identities after v6 — with the CLIP truncation actually fixed this time — that's real evidence prompt engineering has hit its ceiling for this model/setup, and the next step should be a post-generation VLM QA pass (see the recommendation below) rather than further wording changes. `generate_flux_dataset.py`'s variation text and prompt construction have already been updated to match v6; its `guidance_scale` is deliberately left at 3.5 pending this final visual check.

### Step 2: Run inside tmux (Critical)
The full generation takes ~130 hours. Run inside `tmux` so the process survives browser disconnections:

```bash
tmux new -s fluxgen

# Inside tmux:
export HF_TOKEN="hf_your_token_here"
export HF_HOME="/data/huggingface_cache"
export HF_XET_HIGH_PERFORMANCE=1
python3 generate_flux_dataset.py

# Detach safely (leaves process running):
# Press Ctrl+B, then D

# Reattach later:
tmux attach -t fluxgen
```

### Step 3: Monitor Progress
The script prints `[idx/24000] Generating: filename...` for every image. Check the `dataset_output/` folder size periodically.

---

## Qwen-Image-2512 Comparison Test

`test_qwen_variations.py` is the Qwen counterpart to `test_flux_variations.py` — same 48-prompt taxonomy v6 identities/prompts/seeds, so the two output folders (`test_variations_comprehensive_v6/` vs `test_variations_qwen_v6/`) compare image-for-image on prompt adherence. This exists because Flux collapsed toward "attractive" on flaw/average-tier prompts in manual testing (see chat history), while Qwen-Image-2.0 held up on the same prompts.

Setup on a fresh Vast.ai box:

```bash
mkdir -p /data && cd /data
git clone <your-repo-url> /data/LookMax
cd /data/LookMax/ML/pipeline/dataset_generator

bash install_qwen.sh   # checks /data has room, checks torch is present, installs diffusers-from-source + transformers>=4.51.3 -- does not touch torch or /workspace

tmux new -s qwengen
python3 test_qwen_variations.py
# Ctrl+B, D to detach; tmux attach -t qwengen to reattach
```

No `export HF_HOME=...` step needed — `test_qwen_variations.py` pins its own cache to `/data/huggingface_cache` unconditionally at the top of the file (see the comment there for why: three separate incidents proved relying on the shell's env or the script's own checkout location for this both fail across sessions on these boxes). `install_qwen.sh` checks free space against `/data` the same way, independent of your current directory.

Qwen-Image-2512 is Apache 2.0 and not a gated repo, so `HF_TOKEN` is optional (Flux.1 [dev] requires it; Qwen doesn't). First run downloads ~58GB of weights (the 20B transformer plus the large Qwen2.5-VL text encoder) to `/data/huggingface_cache`.

**If you already have a completed download under a different path** (e.g. `/data/dataset_generator/huggingface_cache` from before this fixed-path change), move it once rather than re-downloading:
```bash
mv /data/dataset_generator/huggingface_cache /data/huggingface_cache
```

---

## Taxonomy v7 + Full Dataset Generation (Qwen-Image-2512)

v6 (above) was a prompt-adherence comparison test. v7 is the taxonomy actually used to generate the real dataset, fixing two problems the v6 comparison surfaced on review: flaw-tier severity was flat (one maximally-bad description instead of a gradient), and "Polished" outfits were a jacket monoculture (only one archetype, always involving outerwear). See `qwen_taxonomy_v7.py`'s module docstring for the full rationale.

| Script | Purpose |
|---|---|
| `qwen_taxonomy_v7.py` | Shared taxonomy module — independent attribute axes (hair/facial hair/skin/eyebrows; top/fit/fabric/color/bottom/footwear), each with its own severity gradient. Imported by all three scripts below. |
| `qwen_pipeline_utils.py` | GPU-aware model loader — auto-detects VRAM and picks full-GPU-resident (`.to("cuda")`, needed for parallel batching) vs `enable_model_cpu_offload()` (smaller cards, forces batch size 1). |
| `test_qwen_prompts_v7.py` | 6 hardcoded prompts demonstrating the two fixes directly. Run first. |
| `test_qwen_variations_v7.py` | 64-image systematic test (4 identities × 4 tiers × grooming/outfit × men/women). Run second. |
| `generate_qwen_dataset.py` | The full 24,000-image run. Run last, after reviewing the two test scripts' output. |

### Running the full generation

```bash
python3 generate_qwen_dataset.py [target_count] [gen_batch_size]
```

- No args: runs until all 24,000 images exist.
- `target_count` (e.g. `500`): generate up to that many NEW images this invocation, then exit. Safe to stop and restart anytime — resume is based on which output files already exist, and the task list is seeded deterministically (`random.Random(42)`) so index N always means the same task on every run.
- `gen_batch_size` (e.g. `4`): how many images to generate in ONE parallel GPU forward pass — real throughput, not just multiple processes. Only takes effect if the detected GPU has enough VRAM to hold the full ~58GB pipeline resident (currently ≥80GB — see `qwen_pipeline_utils.py`); otherwise it's forced to 1 automatically. Start conservative (2-4) and watch `nvidia-smi` before raising it — there's no pre-measured safe ceiling for this model's per-image activation memory yet.

To run unattended in tmux until fully done:
```bash
tmux new -s qwengen
while python3 generate_qwen_dataset.py 500 4; do :; done
```

Two output files land in `/data/qwen_dataset_output/`:
- `labels.csv` — training-ready (filename, category, tier, score, binary attribute columns).
- `generation_log.jsonl` — full provenance, one JSON line per image: exact prompt text, negative prompt, seed, steps, cfg scale, batch size used, timestamp. Use this to trace exactly what produced any specific image.

Each image is written to a `.tmp` name and only renamed to its final filename after its CSV row and log line are both flushed — a run killed mid-image can't leave a half-written file that looks complete and gets silently skipped forever.

---

## Prompt Taxonomy (v2 — Effort-Based)

### Design Principles
1. **Effort only, no biology:** Flaws never include biological traits (acne, face shape, body type). We only penalise choices the user can control: greasy/unstyled hair, sloppy clothing, unblended makeup.
2. **Polished ≠ Formal:** A polished score does NOT require a suit. A perfectly fitted streetwear outfit with pristine fabrics scores 10/10. The model rates **fit, crispness, and color harmony**.
3. **Average = no effort:** Average is defined as "clean but zero styling" — no product, flat hair, standard jeans. This creates a clear visual gap between Average and Polished.

### Identity Matrix
Identities combine **age × ethnicity × body type × face shape** to cover diverse real-world users:

```
Ages:        22, 28, 35, 45, 55
Ethnicities: Caucasian, Black, East Asian, South Asian, Hispanic, Middle Eastern
Body types:  slim, athletic, average build, heavy set, muscular (men)
             slim, curvy, athletic, average build, plus size (women)
Face shapes: explicitly included to prevent model from learning genetic bias
```

### Score Scale
| Score | Label | Description |
|---|---|---|
| 1–3 | Flaw | Zero grooming effort, sloppy clothing, obvious neglect |
| 4–6 | Average | Clean but completely unstyled, no products, basic clothes |
| 7–10 | Polished | High-effort styling, product use, excellent fit and color coordination |

---

## Output Structure

```
dataset_output/
├── images/
│   ├── 00001_Men_Grooming.png
│   ├── 00002_Women_Outfit.png
│   └── ...
└── labels.csv      ← Auto-written during generation; safe to resume after interruption
```

`labels.csv` columns: `filename, category, score, [binary attribute flags]`

---

## Timeline & Cost

| Metric | Value |
|---|---|
| Time per image (RTX 5090) | ~19 seconds |
| Total images | 24,000 |
| **Total time** | **~130 hours** |
| Server cost (Vast.ai RTX 5090) | ~\$0.41/hr |
| **Estimated total cost** | **~\$53** |

---

## Known Issues & Fixes

| Issue | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | Tried `pipe.to("cuda")` — model exceeds 32GB VRAM | Use `pipe.enable_model_cpu_offload()` |
| `No space left on device` | HuggingFace cache defaulted to `/root/.cache` on 10GB loop drive | Set `HF_HOME="/data/huggingface_cache"` |
| Slow download (~2 MB/s) | Single-stream TCP throttling by HuggingFace CDN | Set `HF_XET_HIGH_PERFORMANCE=1` |
| `HF_HUB_ENABLE_HF_TRANSFER deprecated` | Old env var replaced by Xet system | Use `HF_XET_HIGH_PERFORMANCE=1` instead |
| `File reconstruction error: IO Error` | Corrupted partial download from force-killed process | `rm -rf /data/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev` and re-run |
| Schnell ignoring flaw prompts | "Prompt collapse" in 4-step distilled models | Must use FLUX.1 [dev] (28 steps) |
| Flaw-tier images look attractive/Polished-ish | [dev] softens abstract intensity adjectives ("severely", "unkempt") back toward its aesthetic prior even at 28 steps | Taxonomy v3: concrete sensory flaw detail (grease sheen, flaking, stains) + `guidance_scale=5.0`; validate with `test_flux_variations.py` before running the master script |
| Master script's grooming labels included `skin_acne`/`skin_dark_circles` | Pre-v2 variation entries were never deleted when the "effort-only, no biology" taxonomy v2 was introduced — silently violated the design principle for ~half of grooming-flaw prompts | Fixed: `MEN_GROOMING_VARS`/`WOMEN_GROOMING_VARS` in `generate_flux_dataset.py` now contain only effort-based v3 entries |
| "Polished" outfit renders visibly slimmer than "Flaw"/"Average" for the same identity+seed | Outfit-tier language ("flawlessly tailored proportions that flatter the body", "highly stylish") pulls FLUX toward its slim-fashion-model prior, partially overriding the identity's stated build — even with a fixed seed the text still reshapes body proportions through cross-attention. The app must never train the model to treat body size/weight as something "improvement" changes. | Taxonomy v4: identities are structured (age/ethnicity/build/face) so the build word is re-injected directly into the outfit description at every tier ("body shape and size unchanged -- still their natural {build} figure, not slimmer or heavier"), not just stated once in the identity clause |
| "Average"-tier images visibly blurrier/softer-focus than Flaw/Polished for the same identity+seed | v3's Average descriptions ended with self-referential closers like "an unremarkable average appearance" — describing the *photo* as unremarkable, not just the outfit/grooming effort; FLUX appears to read that as a cue for soft/amateur-snapshot rendering | Taxonomy v4: dropped those closers, and moved a sharp-focus/high-resolution anchor to the FRONT of every prompt (all tiers) instead of only trailing at the very end |
| v4 full-batch review: grooming Flaw vs Polished still nearly indistinguishable for some identities (Black man, South Asian man, Hispanic woman), clear for others (Caucasian, East Asian) | Same prompt-collapse pattern, only partially addressed by v4's sensory detail — purely descriptive adjectives still get smoothed away for some identities. Risk: if flaw signal is systematically weaker for some ethnicities, the trained model could end up less sensitive to poor grooming in those groups — a fairness bug, not just a quality one. | Taxonomy v5: every Flaw description now restates its core defect a second time in different words and ends with a blunt evaluative closer ("...are the most obvious features... giving an unmistakably low-effort appearance at a glance") |
| v4 full-batch review: one outfit triplet (45yo Black man, "athletic" build) rendered visibly heavier in Flaw than in Polished despite the same seed and the v4 body clause | v4's body-preservation clause was appended at the very END of the outfit description — not forceful enough to counter "fold lines across the chest and stomach" cueing a rounder torso for a moderate (non-extreme) build under "sloppy" framing | Taxonomy v5: body-preservation clause moved to its own sentence right after the identity, BEFORE the effort description (front-loaded, same principle as the blur fix); also dropped "the fabric hanging in a shapeless overly baggy way that swallows the body" from the Men's Flaw #1 outfit variation in the master script, which was never in the test script and likely contributed to the drift |
| **ROOT CAUSE, found while v5 was mid-run: CLIP's hard 77-token limit** — the generation logs show `"CLIP can only handle sequences up to 77 tokens"` on every single image | FLUX.1 uses TWO text encoders: CLIP (hard 77-token cap, contributes a global conditioning vector) and T5-XXL (`max_sequence_length=512`, drives most fine-grained detail via cross-attention). Every taxonomy round from v3 onward kept ADDING reinforcement text, growing prompts to 180-220+ tokens. T5 saw all of it, but CLIP silently truncated at 77 — verified with the real `CLIPTokenizer`: for a typical v5 Outfit prompt (45yo Black man), CLIP's view cut off at *"...natural athletic figure throughout"* and **never reached the actual outfit description at all** (`"wearing a heavily wrinkled..."` was entirely invisible to CLIP). This is a better explanation for the identity-dependent flaw-adherence weakness than an "aesthetic prior" theory — each prompt-engineering round made the token-budget problem *worse*, not better, since longer reinforcement text pushes the real content further past the cutoff. | Taxonomy v6: every prompt now leads with a short (10-25 word) "opener" — the variation's core flaw/effort keywords, plus body preservation for outfits — verified with the real CLIP tokenizer to land completely inside the 77-token window even for the longest identity combinations (Middle Eastern ethnicity + "average build"). The full elaborate description (with v5's reinforcement) still follows for T5. The alignment guide is also modestly trimmed (redundant phrasing removed) to leave more budget for the opener + identity. |

---

## Cleanup Commands

```bash
# Delete old model weights to free space
rm -rf /data/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev
rm -rf /data/huggingface_cache/hub/models--black-forest-labs--FLUX.1-schnell

# Wipe test image folders for a clean re-run
rm -rf /data/dataset_generator/test_images/*
rm -rf /data/dataset_generator/test_variations_comprehensive/*

# Wipe full dataset output for a fresh generation run
rm -rf /data/dataset_generator/dataset_output/*

# Remove old workspace files from the 10GB drive
rm -rf /workspace/dataset_generator
rm -rf ~/.cache/huggingface
```
