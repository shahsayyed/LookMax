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

# ─── Synthetic data generation ─────────────────────────────────────────────
# Two interchangeable backends for writing the ADVICE text -- generate_synthetic_dataset.py
# dispatches on GENERATOR_BACKEND. Chosen after a real side-by-side test during this
# pipeline's build (3 real taxonomy-sampled prompts, run through the actual qa_review.py
# gate): "ollama" (qwen2.5:14b-instruct, local) matched gemini-3.6-flash's QA-pass rate
# and word-count compliance with zero per-call cost and no rate limit, so it's the
# default. Switch to "gemini" for a cloud comparison or if local quality regresses on
# the full 5,000-example run.
GENERATOR_BACKEND = os.environ.get("LOOKMAX_STYLIST_GENERATOR_BACKEND", "gemini")

# Local backend (alternative) -- Ollama, no API key, no rate limit.
OLLAMA_MODEL = os.environ.get("LOOKMAX_STYLIST_OLLAMA_MODEL", "qwen2.5:14b-instruct")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Cloud backend (default) -- Gemma 4 31B IT / Gemini for checklist generation
GEMINI_MODEL = os.environ.get("LOOKMAX_STYLIST_GEMINI_MODEL", "gemma-4-31b-it")
GEMINI_GENERATION_MAX_OUTPUT_TOKENS = 2048  # Room for internal thinking tokens without truncating visible text
GEMINI_TEMPERATURE = 0.7
SYNTHETIC_TARGET_COUNT = 6000
GEMINI_REQUESTS_PER_MINUTE = 15.0  # Controlled rate limit (15 requests per minute to prevent server queuing)

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
MAX_NEW_TOKENS = 120  # Generous headroom for 4-5 bullet checklist lines (~60-90 tokens)
MIN_RESPONSE_WORDS = 25
MAX_RESPONSE_WORDS = 85
STOP_TOKEN = "<|im_end|>"
SYSTEM_PROMPT = (
    "You are LookMax's direct personal stylist. Given the outfit/grooming tags, occasion, and score, "
    "produce an actionable CHECKLIST of specific improvements.\n\n"
    "RULES:\n"
    "1. Output format:\n"
    "   - If overall_score >= 9.0: Output 2 to 3 polish or refinement tips.\n"
    "   - If overall_score < 9.0: Output 4 to 5 high-priority corrections.\n"
    "2. Every line MUST start with '- ' and be under 14 words.\n"
    "3. Total response: 30 to 80 words. No preamble, no headers, no closing remarks.\n"
    "4. Focus strictly on observable items: fit, wrinkles, shoe condition, grooming lineup, color harmony, posture.\n"
    "5. STRICT FORBIDDEN:\n"
    "   - Do NOT mention unobservable items (cologne, fragrance).\n"
    "   - Do NOT tell clean-shaven users to shave.\n"
    "   - Do NOT suggest pocket squares or ties unless a blazer/suit is present.\n"
    "   - Never mention body weight, skin conditions, or genetics."
)

# ─── CoreML export ──────────────────────────────────────────────────────────
IOS_MIN_DEPLOYMENT = "iOS18"
QUANT_DTYPE = "none"
QUANT_MODE = "linear_symmetric"
TARGET_LATENCY_MS = 80

# Token budget adjusted to prevent truncation:
# Input prefix (system + rich tags) is ~260-310 tokens -> MAX_INPUT_TOKENS = 384
# Output checklist (4-5 lines) is ~60-90 tokens -> MAX_NEW_TOKENS = 120
# Total sequence cap = 512 (well within SmolLM2's 8K context, exported to CoreML RangeDim(1, 512))
MAX_INPUT_TOKENS = 384   # the system+tags prompt bound (inference input side)
MAX_TOTAL_TOKENS = 512   # sequence cap for training truncation AND CoreML export RangeDim
