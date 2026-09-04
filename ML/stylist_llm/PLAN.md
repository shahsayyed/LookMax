# On-Device Stylist LLM Pipeline

A dedicated, ultra-compact text generator, fully isolated from
`ML/vision/dataset_synthetic/` (the vision image pipeline) and
`ML/vision/dataset_real/` (the real-photo scraper). It ingests the
vision model's detected tags + an occasion the user picked in-app, and
produces a single-shot, <50-word "5-minute fix" -- no cloud call, no
multi-turn chat.

Base model: `HuggingFaceTB/SmolLM2-135M-Instruct`, vocabulary-pruned and
fully fine-tuned (not LoRA -- see below), exported as a CoreML `.mlpackage`
targeting **iOS 18+** (confirmed product decision -- this bumps the whole
app's minimum deployment target, coordinate with the vision model's current
iOS17 target before shipping). **The export is currently STATELESS (no
KV-cache), NOT the stateful design originally planned** -- see "Current
Status" immediately below for why, and "Corrections from the original
brief" further down for the full history of deviations.

This pipeline started from a detailed architecture brief. Several parts of
that brief did not hold up against this specific codebase or against real
testing while building this out -- every deviation below is a *verified*
correction, not a stylistic preference.

---

## Current Status (2026-09-04)

**Read this section first if picking this pipeline up in a new session --
it's the authoritative "where things actually are," not the original plan.**
The pipeline has been run end-to-end once: real dataset, real checkpoint,
real CoreML export, real quality verification. Sections further down
("Scripts", "Corrections from the original brief", "Running the full
pipeline") describe the pipeline's design and mostly still apply, but a few
specific claims in them (stateful export, INT4 default, Gemini-only
generation) are superseded by what's below.

### What's done and verified

- **Dataset**: 4,994 examples generated (of a 5,000 target; 6 skipped after
  exhausting retries), 4,985 passed `qa_review.py` (99.8% pass rate).
  Generated via **Ollama running `qwen2.5:14b-instruct` locally**, not
  Gemini. A real side-by-side test (Gemini 2.5-flash / Gemini 3.6-flash /
  Gemma 4 26B-A4B / `qwen2.5:3b` / `llama3.2` / `qwen2.5:14b-instruct`, all
  against the same real taxonomy-sampled prompts through the actual
  `qa_review.py` gate) found local `qwen2.5:14b-instruct` matched cloud
  quality with zero cost, no rate limit, ~2-13s/call. `generate_synthetic_dataset.py`
  now has a `--backend {ollama,gemini}` flag (`config.py`'s
  `GENERATOR_BACKEND`, default `"ollama"`). Separately confirmed
  `gemini-2.5-flash` (the pipeline's original hardcoded default) is now dead
  ("no longer available to new users") -- `config.py`'s `GEMINI_MODEL`
  defaults to `gemini-3.6-flash` for anyone using the Gemini backend instead.
  Retained vocabulary after pruning came out to 2,819 tokens, below the
  3,500 `MIN_VOCAB_TOKENS` sanity floor (the dataset's language is narrower
  than ideal, likely from single-model/single-temperature generation) --
  flagged to the user, who chose to proceed rather than regenerate with more
  diversity. Not revisited since.
- **Vocabulary pruning**: run for real against the QA-reviewed dataset:
  134,515,008 -> 107,827,200 params (19.8% reduction, tied embeddings).
- **Fine-tuning**: completed, 18m22s on this Mac's Apple Silicon via MPS
  (close to the brief's ~20min estimate, now independently confirmed rather
  than assumed). eval_loss 1.127 -> 1.023 over 3 epochs, final train_loss
  0.955. Checkpoint at `checkpoints/finetune_run/final` (415MB, FP32).
  Smoke-tested with real advice -- coherent, on-brief, correctly reads
  `priority_defect` and `overall_score`.
- **CoreML export**: succeeded, but via a materially different architecture
  than originally planned -- see "Major pivot" below.
- **Quality verification**: built a reusable 20-prompt comparison harness
  (full 4-category x 4-tier grid from `taxonomy.py`, checkpoint vs. exported
  `.mlpackage`, identical greedy decoding on both sides) to catch regressions
  the automated QA gate alone wouldn't. Found INT4 quantization caused real
  problems (4/20 responses trailed off before naturally stopping, 3/20 went
  incoherent, including one severe token-repetition loop --
  `"...define your buzz buzz buzz..."` repeated 53 times). The identical test
  against an FP16 build (same architecture, no weight-quantization step) came
  back 20/20 clean, 17/20 byte-identical to the FP32 checkpoint -- isolating
  the cause to INT4 specifically, not the stateless export architecture.
  **FP16 (207MB) is the current shipped default** (`config.py`'s
  `QUANT_DTYPE = "none"`). INT8 (104MB, real measured size) is untested for
  quality. INT4 (52MB) is confirmed bad as currently shipped (no
  repetition-guard mitigation tried yet -- see open issues).

### Major pivot: stateful KV-cache was abandoned

The original spec (and several passages still below, describing the
ORIGINAL intent -- kept for the deviation record, not as current fact)
called for a stateful, KV-cached CoreML export via
`transformers.cache_utils.StaticCache` + `ct.StateType`, targeting iOS 18+
ANE for the <80ms/token latency target. **This was abandoned** after
exhausting two independent PyTorch export paths, both hitting confirmed,
narrow upstream bugs rather than anything fixable in this codebase:

- **`torch.jit.trace`**: its tracer is value-based (it records what happened
  during ONE concrete execution), and repeatedly either dropped or conflated
  per-layer cache-position buffers that happened to share a value during
  that one trace, no matter how the shared state was restructured (tried:
  30 independent per-layer counters, then a single counter shared via
  closure -- both failed differently, at different points in the conversion
  pipeline).
- **`torch.export`**: traces correctly (buffer identity is tracked
  structurally, not by observed value -- this DID fix the jit.trace-specific
  conflation problem), but CoreML's automatic state derivation for
  torch.export models depends on `graph_signature.buffers_to_mutate`, which
  is populated only by `ExportedProgram.run_decompositions()` -- and that
  call itself crashes for this model with a PyTorch-internal
  `AssertionError` ("expected compiled_fn to be GraphModule, got
  `<class 'function'>`") deep in AOT-autograd's joint-graph export path.
  Confirmed via a minimal, isolated reproduction to happen identically on
  **both** torch 2.13.0 (this environment) and torch 2.7.0 (coremltools'
  own last-tested version, ruling out a version-skew explanation) --
  triggered specifically by this model's mutable `StaticCache` buffers, a
  genuine upstream PyTorch bug, not a coremltools bug or something wrong in
  this codebase.

**Current architecture instead**: the exported model recomputes the whole
growing sequence (prompt + every token generated so far) on every forward
call -- no persistent state, no `StaticCache`, no `ct.StateType`. This is
O(n²) total compute across a full generation instead of O(n), but
empirically still fast: ~5-8s per full 40-60 word response, measured via
coremltools on this Mac's CPU/ANE (`ct.ComputeUnit.ALL`, no forced CPU-only
workaround needed) -- **not** a real iOS device measurement. The iOS side
must feed the prompt **one token at a time, looping over the WHOLE growing
sequence on every call** (not a single batched prefill, and not a
stateful/cached call) -- see `smoke_test.py`'s `run_mlpackage_mode` for the
reference implementation of exactly this loop.

Getting the stateless `torch.export` path working still required ~10
targeted monkey-patches to coremltools 9.0's torch frontend (all in
`export_coreml.py`, each with a docstring naming the exact confirmed bug and
fix, not asserted from memory): a NumPy 2.x incompatibility in its
`int`/`bool` cast handler, a mixed static/dynamic-shape bug in its
`view`/`reshape` converter, several missing `to()` ATen overloads, a
registry containment-check gap for in-place ops under the torch.export
frontend, a missing `alias` op registration, a missing `diff` op
implementation (needed once the sequence dimension became genuinely dynamic
via `dynamic_shapes`), a missing `new_ones` op, and an overly strict
`bitwise_and` dtype check. Every one of these was hit as a real, blocking
error during this build and confirmed fixed by the next stage succeeding --
none are speculative.

### Known open issues (flagged during review, not yet fixed)

1. **`tag_vocabulary.py`'s outfit tag block loses real information.**
   `_outfit_tag_lines()` only reports garment type/pattern + formality, then
   a single `priority_defect` field NAME (not even its severity value)
   picked from `fabric_wrinkled`/`fit_baggy`/`fit_tight`/`footwear_worn`/
   `styling_sloppy` by first-match-wins on ties. Confirmed with a real
   example where FOUR fields were simultaneously flagged at severity 1 (fit
   too tight, fabric wrinkled, footwear worn, styling sloppy) -- only
   `fabric_wrinkled` reached the model; the other three real issues never
   did. `_grooming_tag_lines()` does NOT have this problem -- it reports
   every attribute individually (hair, skin, eyebrows, facial hair/makeup,
   each with its own severity). The fix (not yet applied) is to make
   `_outfit_tag_lines()` mirror that pattern. Token budget is not a
   blocker: measured real prompts are 146-171 tokens against a 256
   `MAX_INPUT_TOKENS` budget, room to spare.
2. **The vision model has no color output at all, not just an unused one.**
   `requested_upper_color` etc. are typed `"meta"` in `taxonomy.py`'s label
   schema, and `ML/vision/training/multihead_common.py` explicitly builds
   trainable heads for every field EXCEPT meta ones -- so color was never a
   real, trained vision-model output, only synthetic-generation QA
   provenance (verifying Qwen actually rendered the requested color). The
   stylist LLM can never reference garment colors unless the vision model
   is retrained with a real color-detection head -- a materially bigger
   change than issue #1 above, not yet scoped.
3. **INT8 quantization is untested for quality.** Only FP16 (clean, 20/20)
   and INT4 (confirmed bad, 13/20 clean) have been run through the 20-prompt
   comparison. If the 207MB FP16 size ever needs to come down, INT8 (104MB,
   real measured size) is the next thing to try -- re-run the comparison
   before trusting it, don't assume quality carries over.
4. **No real iOS device testing has happened at all.** Every
   latency/correctness number in this file was measured via coremltools on
   this Mac (`ct.ComputeUnit.ALL`), never a real iPhone. Per-token latency
   against the 80ms target, and whether this export genuinely schedules on
   a real device's ANE the same way it does here, are both unverified.
5. **No repetition-loop guard exists in the generation loop.** The INT4
   repetition-loop failure mode (greedy decoding getting stuck, e.g. "buzz"
   x53) would likely be fully preventable with a simple no-repeat n-gram
   constraint in the token-generation loop, independent of quantization
   level -- cheap, no re-export needed, not yet implemented. Worth doing
   regardless of which quantization level eventually ships, and specifically
   worth trying before writing off INT4/INT8 as unusable.
6. **iOS 17 vs 18 deployment target was never revisited** after the
   stateless pivot. `config.py`'s `IOS_MIN_DEPLOYMENT = "iOS18"` was
   originally required by the (now-abandoned) stateful export's
   `ct.StateType`. The current stateless export may not need iOS 18 at all,
   but this hasn't been tested or decided -- see `config.py`'s comment on
   `IOS_MIN_DEPLOYMENT`.

---

## Isolation

| | Vision pipeline | Stylist LLM pipeline |
|---|---|---|
| Code | `ML/vision/dataset_synthetic/` | `ML/stylist_llm/` |
| Config | `ML/vision/config.py` (shared with real_data_pipeline + trainer) | `ML/stylist_llm/config.py` (own, self-contained) |
| Data | `ML/data/vision_*` | `ML/data/stylist_llm/` |
| Checkpoints | `ML/vision/dataset_synthetic/output/` | `ML/stylist_llm/checkpoints/` (gitignored) |
| Export target | `ML/models/*.mlpackage` | `ML/models/StylistEngine.mlpackage` (same folder -- both ship in the same iOS app) |

**One deliberate, one-directional exception**: `tag_vocabulary.py` imports
`ML/vision/dataset_synthetic/taxonomy.py` (read-only) to build the input
tag format from the vision model's REAL label schema, instead of a
hand-invented vocabulary that could drift from what the real vision model
emits. Nothing in `dataset_generator/` imports anything from
`stylist_llm/`, and no data/checkpoints/training config is shared beyond
this one translation layer. See `tag_vocabulary.py`'s module docstring.

---

## Scripts

| Script | Purpose |
|---|---|
| `config.py` | Self-contained paths + hyperparameters. Not shared with the vision pipeline's config.py. `GENERATOR_BACKEND` (default `"ollama"`) and `QUANT_DTYPE` (default `"none"`, i.e. FP16) reflect the decisions in "Current Status" above, not the original brief's defaults. |
| `tag_vocabulary.py` | Translates the REAL vision label schema (`taxonomy.get_label_schema()`) into the tag-string prompt format. The interface contract between the two pipelines. See open issue #1 above for a known gap in its outfit branch. |
| `generate_synthetic_dataset.py` | Samples training contexts by reusing `dataset_generator/prompt_builder.build_task()`. `--backend ollama` (default, local, free) or `--backend gemini` (cloud, needs `GEMINI_API_KEY`, costs money). `--dry-run` needs neither. |
| `qa_review.py` | QA gate on generated advice -- word count, meta-chatter, and (most important) the same effort-vs-genetics check the vision pipeline already needed once. `--self-check` verifies the rules against fixtures, no data needed. |
| `prune_vocabulary.py` | Shrinks the base model's embedding/lm_head to only the tokens this pipeline needs. `--dry-run` reports the floor, no GPU/data needed. |
| `remap_tokenizer.py` | Wraps the original tokenizer + the pruned id map so encode()/decode() work against the pruned model. Hard-fails on an uncovered token rather than silently degrading. |
| `finetune.py` | Full fine-tune (not LoRA) with SFT loss masking. `--dry-run` builds + inspects the tokenized dataset, no GPU. |
| `export_coreml.py` | **Stateless** (no KV-cache -- see "Major pivot" above) CoreML export targeting iOS 18. `--quant {int4,int8,none}` (default from `config.py`'s `QUANT_DTYPE`, currently `none`/FP16). Its own module docstring has the full, current, authoritative account of the export architecture and every coremltools patch it applies. |
| `smoke_test.py` | Generate real advice from a checkpoint (`--checkpoint`, fast) or the exported `.mlpackage` itself (`--mlpackage`, runs the actual exported artifact on this Mac via coremltools -- feeds the whole growing sequence each call, matching the stateless export's real calling contract, not a stateful/cached one). |
| `install.sh` | pip install + next-steps printout. |

---

## Corrections from the original brief (verified, not opinions)

1. **The input tag format was invented, not derived from this repo's
   vision model.** The brief showed `- top: white_oxford_shirt (collar:
   unbuttoned_spread)` and `priority_defect: blazer_creasing` -- neither
   exists in `taxonomy.get_label_schema()`. Fixed by `tag_vocabulary.py`
   building the tag block from the real field names
   (`upper_type`/`fabric_wrinkled`/`formality`/etc.) and picking
   `priority_defect` from the real ordinal severity fields. (See "Known
   open issues" #1 above -- this fix is itself now known to be incomplete
   for the outfit branch specifically.)

2. **Vocabulary pruning based only on the generated dataset is unsafe.**
   Any token not seen in one generation run becomes literally impossible to
   produce afterward. Fixed by `prune_vocabulary.py` keeping the union of
   dataset tokens AND `tag_vocabulary.full_vocabulary_terms()` (every real
   garment/pattern/formality/occasion word) as a floor.

3. **SmolLM2-135M-Instruct ties its embedding and lm_head weights**
   (confirmed via `AutoConfig(...).tie_word_embeddings == True`), so
   pruning the vocabulary saves ONE ~26M-param matrix, not two. Measured
   directly (real run against the QA-reviewed dataset): **134.5M ->
   107.83M params**, not the brief's 85-90M estimate. Real measured
   `.mlpackage` sizes at each quantization level (checkpoint is FP32
   PyTorch, 415MB, for reference): **FP16 207MB, INT8 104MB, INT4 52MB** --
   all above the brief's <45MB target regardless of quantization level, a
   gap that was already known before quality testing even began. See
   "Current Status" above for why FP16 (the largest of the three) is the
   current shipping choice anyway.

4. **The brief's `ct.RangeDim(1, 128)` input bound is too small for the
   real prompt.** Measured directly: a real system prompt + tag block from
   `tag_vocabulary.py` tokenizes to **146-171 tokens BEFORE the assistant's
   advice even starts**. Using 128 as a training truncation cap (an early
   version of `finetune.py` did exactly this, copying the brief's constant
   verbatim) silently truncated every training example down to **zero
   supervised tokens** -- caught via `finetune.py --dry-run` against a real
   fixture, not a hypothetical. Fixed: `config.MAX_INPUT_TOKENS = 256`
   (the prompt-side bound) and `config.MAX_TOTAL_TOKENS = 320` (training
   truncation cap; also the dynamic-shape upper bound the stateless CoreML
   export declares via `ct.RangeDim`, now that there's no StaticCache
   capacity to size).

5. **The brief's CoreML quantization code calls the wrong API at the
   wrong stage** (`coremltools.optimize.torch.quantization.linear_quantize_weights`,
   a pre-conversion torch-side function, applied to an already-converted
   `.mlpackage`). Fixed: `export_coreml.py` uses the correct post-conversion
   API, confirmed present in the installed coremltools 9.0:
   `coremltools.optimize.coreml.linear_quantize_weights` (only called at
   all when `--quant` isn't `"none"` -- see "Current Status").

6. **The brief showed no KV-cache handling at all.** The original fix
   attempted here (wrapping the model in `transformers.cache_utils.StaticCache`
   and exporting as a stateful CoreML model) was itself abandoned after
   hitting confirmed upstream PyTorch/coremltools bugs -- see "Major pivot"
   above for the full account. The shipped export recomputes the whole
   sequence each call instead, and is O(n²) rather than O(n) as a result;
   this is currently accepted as fast enough (empirically, not yet on real
   hardware) rather than fixed.

7. **LoRA was dropped in favor of a full fine-tune** -- at ~108M params
   with a generous compute budget either way and no need to preserve broad
   chat ability for this narrow task, LoRA's usual benefits (VRAM savings,
   capability preservation) don't apply here, and skipping it also skips
   the merge-before-export step entirely.

8. **Grooming tags were added** -- the brief only showed an outfit example,
   but the real vision pipeline scores grooming (hair/skin/eyebrows/
   facial-hair-or-makeup) as a separate category with its own photo (see
   `ML/README.md`'s "oval for grooming, rectangle for outfit"). `tag_vocabulary.py`
   builds a symmetric tag format for both -- though, per open issue #1
   above, grooming's version is actually more complete/granular than
   outfit's.

9. **Occasion has no vision-model equivalent** -- `taxonomy.py` has no
   occasion axis at all; in the real app it comes from the user's own
   selection in the UI (see the brief's `userSession.selectedOccasion`).
   `tag_vocabulary.OCCASIONS` is defined in this pipeline, not imported.

---

## Running the full pipeline, in order

### 1. Setup
```bash
cd ML/stylist_llm
bash install.sh
```

### 2. Cheap sanity checks -- no GPU, no API key, no Ollama call
```bash
python3 generate_synthetic_dataset.py --dry-run --count 10
python3 qa_review.py --self-check
python3 prune_vocabulary.py --dry-run
```
Read the printed prompts. This is the cheapest point to catch a bad tag
format or an unrealistic occasion pairing.

### 3. Generate the real synthetic dataset
```bash
# Default: local, free, no rate limit -- needs Ollama running with the model pulled
ollama pull qwen2.5:14b-instruct
python3 generate_synthetic_dataset.py --count 5000

# Or, cloud instead:
export GEMINI_API_KEY="your-key-here"
python3 generate_synthetic_dataset.py --count 5000 --backend gemini
```
Resumable (Ctrl+C and re-run picks up where it left off -- see the script's
module docstring). Writes to `ML/data/stylist_llm/raw_generated/stylist_advice.jsonl`.

### 4. QA review
```bash
python3 qa_review.py ../data/stylist_llm/raw_generated/stylist_advice.jsonl
```
Every row is kept with a `qa.pass`/`qa.reasons` field -- nothing is deleted.
Inspect the failure reasons if the fail rate is above ~10%.

### 5. Prune the vocabulary
```bash
python3 prune_vocabulary.py ../data/stylist_llm/qa_reviewed/stylist_advice.jsonl
```
Saves the pruned base model + token id map to `checkpoints/pruned_base/`
(gitignored). Check the printed retained-token count against
`MIN_VOCAB_TOKENS` -- it's a sanity floor, not a hard gate; the real run
came in below it and was shipped anyway (see "Current Status").

### 6. Fine-tune
```bash
python3 finetune.py --dry-run ../data/stylist_llm/qa_reviewed/stylist_advice.jsonl   # inspect first
python3 finetune.py ../data/stylist_llm/qa_reviewed/stylist_advice.jsonl
```
~18-20 min on Apple Silicon via MPS (independently measured: 18m22s on the
real 4,985-example dataset).

### 7. Smoke-test the checkpoint before spending time on export
```bash
python3 smoke_test.py --checkpoint checkpoints/finetune_run/final
```
Read the 5 generated examples. Confirm advice is specific, in the 30-50
word window, and never implies losing weight / changing body shape /
clearing skin (the same check `qa_review.py` runs on training data --
worth eyeballing on real model OUTPUT too).

### 8. Export to CoreML
```bash
python3 export_coreml.py checkpoints/finetune_run/final          # ships config.py's default (currently FP16)
python3 export_coreml.py checkpoints/finetune_run/final --quant int8   # to try a smaller, unverified-quality option
```
See export_coreml.py's module docstring for the full, current account of
the export architecture -- it is stateless, not stateful, and required
several coremltools patches to work at all (all documented inline there).

### 9. Verify quality, not just that conversion succeeded
```bash
python3 smoke_test.py --mlpackage ../../models/StylistEngine.mlpackage
```
This alone is NOT enough to trust a quantization change -- it only runs 5
examples. Before trusting any new `--quant` level, re-run the full 20-prompt
checkpoint-vs-export comparison (the harness that caught the INT4
regression; not currently checked into this repo as a script -- it was
built ad hoc during the session that found the INT4 issue and should be
recreated/reused rather than skipped).

### 10. On-device verification (not scriptable from here)
Drag `StylistEngine.mlpackage` into Xcode, confirm per-token latency
against the 80ms target on a real iOS 18 device, and confirm the Swift
`PromptBuilder` (see the original brief's Swift snippet) matches
`tag_vocabulary.format_full_prompt()` byte-for-byte. Also confirm the Swift
side feeds the prompt ONE TOKEN AT A TIME over the whole growing sequence
(no batched prefill, no state object) -- see `smoke_test.py`'s
`run_mlpackage_mode` for the reference loop shape. **None of this has been
done yet** -- see "Known open issues" #4 above.

---

## Output layout

```
ML/data/stylist_llm/
├── raw_generated/       <- generate_synthetic_dataset.py's JSONL output (4,994 examples)
├── qa_reviewed/         <- qa_review.py's output (every row kept, qa.pass/qa.reasons added; 4,985 passed)
└── pruned_vocab/
    ├── pruned_base/          <- prune_vocabulary.py's pruned model (107.83M params)
    └── token_id_map.json

ML/stylist_llm/checkpoints/    <- gitignored
└── finetune_run/final/        <- the fine-tuned checkpoint (415MB, FP32)

ML/models/StylistEngine.mlpackage   <- current shipped export: stateless, FP16, 207MB
```
