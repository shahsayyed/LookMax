# Tokenizer probing for prompt engineering (Qwen-Image-2512)

## What this is
`tokenizer_probe.py` (in this directory) loads *only* the Qwen-Image-2512
text encoder's tokenizer (Qwen2.5-VL's `Qwen2Tokenizer`) from
`Qwen/Qwen-Image-2512` on Hugging Face — not the 20B transformer, not the
VAE, no GPU. It downloads a few MB (`vocab.json`, `merges.txt`,
`tokenizer_config.json`) from the public repo (no HF token needed) and lets
you inspect how a prompt/word gets tokenized, without generating any
images. Safe to run locally, in parallel with a real generation job on a
remote box.

## Usage
```bash
cd ML/vision/dataset_synthetic

# Does this word exist as one clean token, or does it always fragment?
python3 tokenizer_probe.py --vocab-check unibrow --vocab-check stubble

# Tokenize every prompt_append / extra_negative string in a variants file
# (same JSON format prompt_experiment.py --variants-json consumes)
python3 tokenizer_probe.py --variants-json eyebrows_variants.json

# Compare two candidate phrasings side by side
python3 tokenizer_probe.py --compare "clean shaven" "freshly shaved, no stubble"
```

## What it tells you — and what it doesn't
This is a **tokenization** check, not an **understanding** check.

- Qwen-Image's text encoder doesn't predict tokens — it turns the token
  sequence into contextualized embeddings that condition the diffusion
  transformer. Whether the model actually *renders* an attribute (e.g.
  "unibrow", "five o'clock shadow") depends on whether that concept was
  well represented in the model's image-caption training data — the
  tokenizer has no visibility into that.
- BPE fragmentation into 2+ subword pieces is normal for most English
  words (the model handles it fine via attention). "This word splits into
  3 tokens" is **not** a useful predictor of poor rendering on its own —
  don't over-index on fragmentation count as a proxy for prompt quality.

## Where it's actually useful
A cheap **pre-filter**, before spending GPU time:
- Catch typos/encoding oddities that tokenize into garbage before testing them.
- Compare token budget of candidate phrasings to avoid tripping the
  encoder's context length / truncation.
- Sanity-check that a negative-prompt clause tokenizes as expected.

## What still decides whether wording actually works
Empirical A/B image generation — `prompt_experiment.py` in this
directory, which generates the same seed under baseline vs. variant
prompt/negative-prompt and lets you compare outputs directly. That
remains the primary signal in the prompt-engineering loop; the tokenizer
probe is a fast triage step in front of it, not a replacement for it.
