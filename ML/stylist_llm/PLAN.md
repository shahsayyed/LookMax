# On-Device Stylist LLM Pipeline

A dedicated, ultra-compact text generator, fully isolated from
`ML/vision/dataset_synthetic/` (the vision image pipeline) and
`ML/vision/dataset_real/` (the real-photo scraper). It ingests the
vision model's detected tags + an occasion the user picked in-app, and
produces a single-shot, <50-word "5-minute fix" -- no cloud call, no
multi-turn chat.

Base model: `HuggingFaceTB/SmolLM2-135M-Instruct`, vocabulary-pruned and
fully fine-tuned (not LoRA -- see below), exported as a stateful, INT4
CoreML `.mlpackage` targeting **iOS 18+** (confirmed product decision --
this bumps the whole app's minimum deployment target, coordinate with the
vision model's current iOS17 target before shipping).

This pipeline started from a detailed architecture brief. Several parts of
that brief did not hold up against this specific codebase or against real
testing while building this out -- every deviation below is a *verified*
correction, not a stylistic preference. See "Corrections from the original
brief" for the full list with evidence.

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
| `config.py` | Self-contained paths + hyperparameters. Not shared with the vision pipeline's config.py. |
| `tag_vocabulary.py` | Translates the REAL vision label schema (`taxonomy.get_label_schema()`) into the tag-string prompt format. The interface contract between the two pipelines. |
| `generate_synthetic_dataset.py` | Samples training contexts by reusing `dataset_generator/prompt_builder.build_task()`, calls Gemini for the advice text. `--dry-run` needs no API key. |
| `qa_review.py` | QA gate on generated advice -- word count, meta-chatter, and (most important) the same effort-vs-genetics check the vision pipeline already needed once. `--self-check` verifies the rules against fixtures, no data needed. |
| `prune_vocabulary.py` | Shrinks the base model's embedding/lm_head to only the tokens this pipeline needs. `--dry-run` reports the floor, no GPU/data needed. |
| `remap_tokenizer.py` | Wraps the original tokenizer + the pruned id map so encode()/decode() work against the pruned model. Hard-fails on an uncovered token rather than silently degrading. |
| `finetune.py` | Full fine-tune (not LoRA) with SFT loss masking. `--dry-run` builds + inspects the tokenized dataset, no GPU. |
| `export_coreml.py` | Stateful (KV-cache), INT4 CoreML export targeting iOS 18. The highest-risk stage -- see "Known risks". |
| `smoke_test.py` | Generate real advice from a checkpoint (`--checkpoint`, fast) or the exported `.mlpackage` itself (`--mlpackage`, runs the real stateful cache on this Mac via coremltools). |
| `install.sh` | pip install + next-steps printout. |

---

## Corrections from the original brief (verified, not opinions)

1. **The input tag format was invented, not derived from this repo's
   vision model.** The brief showed `- top: white_oxford_shirt (collar:
   unbuttoned_spread)` and `priority_defect: blazer_creasing` -- neither
   exists in `taxonomy.get_label_schema()`. Fixed by `tag_vocabulary.py`
   building the tag block from the real field names
   (`upper_type`/`fabric_wrinkled`/`formality`/etc.) and picking
   `priority_defect` from the real ordinal severity fields.

2. **Vocabulary pruning based only on the generated dataset is unsafe.**
   Any token not seen in one Gemini run becomes literally impossible to
   produce afterward. Fixed by `prune_vocabulary.py` keeping the union of
   dataset tokens AND `tag_vocabulary.full_vocabulary_terms()` (every real
   garment/pattern/formality/occasion word) as a floor.

3. **SmolLM2-135M-Instruct ties its embedding and lm_head weights**
   (confirmed via `AutoConfig(...).tie_word_embeddings == True`), so
   pruning the vocabulary saves ONE ~26M-param matrix, not two. Measured
   directly (`prune_vocabulary.py` run against a real fixture): **134.5M
   -> ~106-109M params**, not the brief's 85-90M estimate. At INT4 that's
   roughly **50-55MB of weight storage**, not the brief's <45MB target --
   before CoreML packaging overhead. This is a real product-decision gap
   (accept ~55MB, try a smaller base model, or ship as a post-install
   download instead of embedding in the initial `.ipa`), not something
   silently absorbed here.

4. **The brief's `ct.RangeDim(1, 128)` input bound is too small for the
   real prompt.** Measured directly: a real system prompt + tag block from
   `tag_vocabulary.py` tokenizes to **146-171 tokens BEFORE the assistant's
   advice even starts**. Using 128 as a training truncation cap (an early
   version of `finetune.py` did exactly this, copying the brief's constant
   verbatim) silently truncated every training example down to **zero
   supervised tokens** -- caught via `finetune.py --dry-run` against a real
   fixture, not a hypothetical. Fixed: `config.MAX_INPUT_TOKENS = 256`
   (the prompt-side bound) and `config.MAX_TOTAL_TOKENS = 320` (training
   truncation cap AND the exported model's StaticCache capacity, since the
   cache must hold input + every generated token together).

5. **The brief's CoreML quantization code calls the wrong API at the
   wrong stage** (`coremltools.optimize.torch.quantization.linear_quantize_weights`,
   a pre-conversion torch-side function, applied to an already-converted
   `.mlpackage`). Fixed: `export_coreml.py` uses the correct post-conversion
   API, confirmed present in the installed coremltools 9.0:
   `coremltools.optimize.coreml.linear_quantize_weights`.

6. **The brief showed no KV-cache handling at all**, which would mean
   recomputing the full growing prefix on every new token -- the <80ms
   target is very unlikely to hold that way for a 30-50 word response.
   Fixed: `export_coreml.py` wraps the model with
   `transformers.cache_utils.StaticCache` and exposes its per-layer
   key/value buffers to `ct.convert(..., states=[...])` as persistent
   state. The cache mechanics (prefill, then continued decode from a
   saved `cache_position`) were verified directly against the real,
   downloaded SmolLM2-135M-Instruct weights -- see `export_coreml.py`'s
   module docstring for exactly what was and wasn't run end-to-end.

7. **LoRA was dropped in favor of a full fine-tune** -- at ~109M params
   with a generous compute budget either way and no need to preserve broad
   chat ability for this narrow task, LoRA's usual benefits (VRAM savings,
   capability preservation) don't apply here, and skipping it also skips
   the merge-before-export step entirely.

8. **Grooming tags were added** -- the brief only showed an outfit example,
   but the real vision pipeline scores grooming (hair/skin/eyebrows/
   facial-hair-or-makeup) as a separate category with its own photo (see
   `ML/README.md`'s "oval for grooming, rectangle for outfit"). `tag_vocabulary.py`
   builds a symmetric tag format for both.

9. **Occasion has no vision-model equivalent** -- `taxonomy.py` has no
   occasion axis at all; in the real app it comes from the user's own
   selection in the UI (see the brief's `userSession.selectedOccasion`).
   `tag_vocabulary.OCCASIONS` is defined in this pipeline, not imported.

---

## Known risks (ranked, highest first)

| Risk | Why it matters | Mitigation in place |
|---|---|---|
| Stateful CoreML export (`export_coreml.py`) hasn't been trace-and-converted end-to-end | No fine-tuned checkpoint exists yet to trace; stateful HF->CoreML export is genuinely one of the harder corners of on-device LLM deployment even for experienced practitioners | The underlying StaticCache mechanics ARE verified against real weights (see script docstring). Budget real iteration time for the `torch.jit.trace` + `ct.convert` step specifically. |
| 100% synthetic training data, no independent ground truth | Unlike the vision pipeline (synthetic + 14,079 real photos), there's nothing to anchor against | `qa_review.py`'s gate, including the effort-vs-genetics check -- but this is automated, not a substitute for a human sampling real output before shipping |
| Actual on-device latency vs. the 80ms target | Never measured on real ANE hardware within this pipeline | `smoke_test.py --mlpackage` verifies output correctness on a Mac; latency must be measured on a real iOS 18 device separately |
| ~50-55MB actual model size vs. <45MB target | Affects `.ipa` bundle size decisions | Flagged in `export_coreml.py`'s output at export time; needs a product decision, not a code fix |
| Vocabulary floor coverage | A term missing from `tag_vocabulary.full_vocabulary_terms()` hard-fails at encode time (by design -- see `remap_tokenizer.py`) rather than silently degrading | Already caught and fixed once during this pipeline's own build (ChatML role names `system`/`user`/`assistant` were initially missing from the floor) -- re-run `prune_vocabulary.py --dry-run` after any `tag_vocabulary.py` change |

---

## Running the full pipeline, in order

### 1. Setup
```bash
cd ML/stylist_llm
bash install.sh
```

### 2. Cheap sanity checks -- no GPU, no API key
```bash
python3 generate_synthetic_dataset.py --dry-run --count 10
python3 qa_review.py --self-check
python3 prune_vocabulary.py --dry-run
```
Read the printed prompts. This is the cheapest point to catch a bad tag
format or an unrealistic occasion pairing.

### 3. Generate the real synthetic dataset -- needs GEMINI_API_KEY, costs real money
```bash
export GEMINI_API_KEY="your-key-here"
python3 generate_synthetic_dataset.py --count 5000
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
`MIN_VOCAB_TOKENS`.

### 6. Fine-tune
```bash
python3 finetune.py --dry-run ../data/stylist_llm/qa_reviewed/stylist_advice.jsonl   # inspect first
python3 finetune.py ../data/stylist_llm/qa_reviewed/stylist_advice.jsonl
```
~12 min on a T4/RTX GPU, ~20 min on Apple Silicon via MPS (brief's own
estimate -- not independently re-measured here since no full dataset has
been generated yet).

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
python3 export_coreml.py checkpoints/finetune_run/final
```
See "Known risks" above -- budget real iteration time here.

### 9. Smoke-test the exported package
```bash
python3 smoke_test.py --mlpackage ../../models/StylistEngine.mlpackage
```
Runs the actual exported artifact's stateful cache on this Mac.

### 10. On-device verification (not scriptable from here)
Drag `StylistEngine.mlpackage` into Xcode, confirm per-token latency
against the 80ms target on a real iOS 18 device, and confirm the Swift
`PromptBuilder` (see the original brief's Swift snippet) matches
`tag_vocabulary.format_full_prompt()` byte-for-byte.

---

## Output layout

```
ML/data/stylist_llm/
├── raw_generated/       <- generate_synthetic_dataset.py's JSONL output
├── qa_reviewed/         <- qa_review.py's output (every row kept, qa.pass/qa.reasons added)
└── pruned_vocab/         <- (unused; prune_vocabulary.py writes to checkpoints/ by default, see config.py)

ML/stylist_llm/checkpoints/    <- gitignored: pruned_base/, finetune_run/
ML/models/StylistEngine.mlpackage       <- final export, same folder as the vision model's .mlpackage files
```
