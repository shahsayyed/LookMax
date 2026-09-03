"""
config.py -- self-contained configuration for the on-device stylist LLM
pipeline. Deliberately NOT part of ML/vision/config.py (the vision
pipeline's config) -- the two pipelines are meant to be fully
isolated (separate data folders, separate training code, separate
checkpoints), and sharing one config module would be a silent coupling
point neither pipeline actually needs. The one deliberate, one-directional
exception is tag_vocabulary.py importing taxonomy.py's real label schema
(see that file's docstring) -- that's an interface CONTRACT, not a shared
config/training dependency.

Edit this file to tune the pipeline without touching execution scripts.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass

# ─── Paths ────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent   # LookMax/
ML_ROOT     = REPO_ROOT / "ML"
STYLIST_DIR = Path(__file__).resolve().parent                        # this pipeline's own code dir
MODELS_DIR  = ML_ROOT / "models"                                     # shared final .mlpackage destination
                                                                       # (both models ship in the same iOS
                                                                       # app bundle -- this is the only path
                                                                       # this pipeline shares with the vision
                                                                       # one, and only as an export TARGET)

# Data lives at ML/data/stylist_llm/
STYLIST_DATA_DIR = ML_ROOT / "data" / "stylist_llm"
RAW_GENERATED_DIR = STYLIST_DATA_DIR / "raw_generated"    # Gemini output, JSONL, unfiltered
QA_REVIEWED_DIR    = STYLIST_DATA_DIR / "qa_reviewed"      # after qa_review.py -- has qa_pass column-equivalent
VOCAB_DIR          = STYLIST_DATA_DIR / "pruned_vocab"     # token id mapping + pruned tokenizer artifacts

# Working training checkpoints (LoRA/full-FT weights, merged model before CoreML
# export) stay local to this pipeline's own folder, gitignored -- never written
# under ML/data/ or ML/models/ until export_coreml.py produces the final artifact.
CHECKPOINTS_DIR = STYLIST_DIR / "checkpoints"

# ─── Base model ───────────────────────────────────────────────────────────
BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

# ─── Synthetic data generation (Gemini) ────────────────────────────────────
# A SEPARATE API config from ML/vision/config.py's GEMINI_MODEL -- that one
# is the VLM used for classifying scraped photos (a vision task); this is a
# plain text-generation call for writing styling-advice training examples.
# Deliberately not sharing the constant so the two can be tuned/rotated
# independently without cross-pipeline surprise.
GEMINI_MODEL = os.environ.get("LOOKMAX_STYLIST_GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TEMPERATURE = 0.7
SYNTHETIC_TARGET_COUNT = 5000
GEMINI_REQUESTS_PER_MINUTE = 14.0  # matches the pacing already proven safe in real_data_pipeline/03_classify_and_sort.py

# ─── Vocabulary pruning ─────────────────────────────────────────────────────
# Floor, not a target -- prune_vocabulary.py keeps every token that's either
# (a) seen in the generated dataset, or (b) in tag_vocabulary.py's
# full_vocabulary_terms() MUST-KEEP set (every real garment/pattern/formality/
# occasion word, tokenized), whichever is LARGER. See prune_vocabulary.py's
# module docstring for why a dataset-only cut is unsafe.
MIN_VOCAB_TOKENS = 3500

# ─── Fine-tuning ─────────────────────────────────────────────────────────
# Full fine-tune, not LoRA -- see finetune.py's module docstring for why:
# at ~90M params post-pruning, with a generous compute budget either way and
# no need to preserve the base model's general chat ability (this is a
# narrow, single-purpose generator), LoRA's main benefits (VRAM savings,
# preserving broad capability) don't buy anything here, and skipping it
# also skips the LoRA-merge step before CoreML export entirely. This is a
# deliberate deviation from the original brief, not an oversight.
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
TRAIN_SPLIT = 0.90  # 90/10 train/val -- dataset is small (thousands, not tens of thousands), keep most of it for training

# ─── Generation contract (must match the iOS runtime side exactly) ────────
MAX_NEW_TOKENS = 60
MIN_RESPONSE_WORDS = 30
MAX_RESPONSE_WORDS = 50
STOP_TOKEN = "<|im_end|>"
SYSTEM_PROMPT = (
    "You are a direct personal stylist. Using the provided outfit tags and "
    "occasion, provide exactly one high-leverage 5-minute fix under 50 words. "
    "Focus strictly on fit, proportion, and grooming."
)

# ─── CoreML export ──────────────────────────────────────────────────────────
# iOS 18+ is a deliberate, explicit product decision (confirmed) -- it's what
# makes the stateful/KV-cache export path (ct.StateType) and INT4 linear
# quantization actually available; iOS 17 would mean a slower, larger export
# with no KV cache (see export_coreml.py's module docstring for the risk this
# was chosen to avoid). This bumps the WHOLE APP's minimum deployment target,
# not just this model -- coordinate with the vision model's iOS17 target
# before shipping (see ML/README.md).
IOS_MIN_DEPLOYMENT = "iOS18"
QUANT_DTYPE = "int4"
QUANT_MODE = "linear_symmetric"
TARGET_LATENCY_MS = 80

# CORRECTED FROM THE BRIEF's ct.RangeDim(1, 128): measured directly against
# real tag_vocabulary.py output (system prompt + a real tags block, no
# assistant turn) -- prefix length came out to 146-171 tokens BEFORE the
# assistant's advice even starts, already over 128. Using 128 as a training
# sequence cap (as finetune.py originally did, copying this constant
# verbatim) silently truncated every example down to ZERO supervised
# (assistant) tokens -- caught via finetune.py --dry-run against a real
# fixture, not a hypothetical concern. See PLAN.md's "Known risks" table.
MAX_INPUT_TOKENS = 256   # the system+tags prompt bound (inference input side, ct.RangeDim upper bound)
MAX_TOTAL_TOKENS = 320   # MAX_INPUT_TOKENS + MAX_NEW_TOKENS + headroom -- training truncation cap AND
                          # export_coreml.py's StaticCache max_cache_len (needs room for input + generated tokens)
