"""
LookMax ML Pipeline — Centralized Configuration
------------------------------------------------
All paths, scraping targets, VLM settings, training hyperparameters, and
aesthetic-score thresholds are defined here. Edit this file to tune the pipeline
without touching any of the execution scripts.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load .env from repository root if it exists
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass

# ─── Repository Root ─────────────────────────────────────────────────────────
# Resolve dynamically from this file's location so the pipeline is portable
REPO_ROOT  = Path(__file__).resolve().parent.parent.parent   # LookMax/
ML_ROOT    = REPO_ROOT / "ML"
PIPELINE   = ML_ROOT / "pipeline"
DATA_ROOT  = ML_ROOT / "data"
MODELS_DIR = ML_ROOT / "models"

# ─── Stage Directories ───────────────────────────────────────────────────────
RAW_SCRAPES_DIR       = DATA_ROOT / "1_Raw_Scrapes"
VLM_PROCESSING_DIR    = DATA_ROOT / "2_VLM_Processing"
REJECTED_DIR          = VLM_PROCESSING_DIR / "filtered_rejected"
METADATA_LOGS_DIR     = VLM_PROCESSING_DIR / "metadata_logs"
TRAINING_DATA_DIR     = DATA_ROOT / "3_CoreML_Training_Data"
ANNOTATIONS_FILE      = METADATA_LOGS_DIR / "dataset_annotations.jsonl"

# ─── Reddit Scraping Sources ─────────────────────────────────────────────────
REDDIT_SOURCES = [
    {"subreddit": "OUTFITS",               "folder": "reddit_outfits"},
    {"subreddit": "malefashionadvice",      "folder": "reddit_malefashionadvice"},
    {"subreddit": "femalefashionadvice",    "folder": "reddit_femalefashionadvice"},
    {"subreddit": "streetwear",             "folder": "reddit_streetwear"},
    {"subreddit": "Posture",               "folder": "reddit_posture"},
    {"subreddit": "mensfashion",            "folder": "reddit_mensfashion"},
    {"subreddit": "femalefashion",          "folder": "reddit_femalefashion"},
]

# Listings to crawl for each subreddit (hot/top gives the most upvoted quality posts)
REDDIT_LISTINGS = ["hot", "top"]
REDDIT_TOP_PERIODS = ["all", "year"]    # For "top" listing only

# ─── Reddit Playwright Scraper Settings ─────────────────────────────────────
REDDIT_PROFILE_DIR = PIPELINE / "reddit_profile"
REDDIT_QUERIES_FILE = PIPELINE / "reddit_queries.json"
REDDIT_SCRAPE_OUTPUT_JSON = METADATA_LOGS_DIR / "reddit_images.json"
REDDIT_DELAY_MIN_SEC = 5.0          # Human-like randomized minimum delay per request (seconds)
REDDIT_DELAY_MAX_SEC = 10.0         # Human-like randomized maximum delay per request (seconds)
REDDIT_BATCH_SIZE = 8               # Requests before triggering a human cooling-off pause
REDDIT_BATCH_COOLDOWN_SEC = 30.0    # Duration of human browsing cooldown pause (seconds)
REDDIT_CATEGORY_COOLDOWN_SEC = 15.0 # Duration of pause between switching categories (seconds)

# ─── Demographic Folder Buckets ──────────────────────────────────────────────
DEMOGRAPHICS = [
    "Men_Under_35",
    "Men_35_to_50",
    "Men_Over_50",
    "Women_Under_35",
    "Women_35_to_50",
    "Women_Over_50",
]

AESTHETIC_TIERS = [
    "1_Needs_Improvement",
    "2_Average",
    "3_Polished",
]

# ─── Image Quality Heuristics (Phase 3 — Step A) ─────────────────────────────
MIN_IMAGE_DIMENSION_PX  = 480        # Shortest edge in pixels
MAX_ASPECT_RATIO        = 2.5        # Width/height or height/width cap
BLUR_LAPLACIAN_THRESHOLD = 60.0      # Variance below this → image is blurry

# ─── VLM Classification Settings (Phase 3 — Step B) ─────────────────────────

# Primary engine: "mlx_vlm" | "ollama" | "gemini"
VLM_ENGINE = os.environ.get("LOOKMAX_VLM_ENGINE", "ollama")

# MLX-VLM (M3 Max native — recommended)
MLX_MODEL_ID = "mlx-community/Qwen2-VL-7B-Instruct-4bit"

# Ollama (local HTTP)
OLLAMA_HOST  = "http://localhost:11434"
OLLAMA_MODEL = "llava:7b"  # Compatible with Ollama 0.32.x; alternatives: "llava:13b"

# Gemini Cloud API
GEMINI_MODEL = "gemma-4-31b-it"

# VLM batch concurrency
VLM_PARALLEL_WORKERS = 8            # Concurrent VLM worker threads
VLM_REQUEST_TIMEOUT_SEC = 30        # Per-image VLM timeout

# ─── Training Hyperparameters (Phase 4) ──────────────────────────────────────
BACKBONE = "mobilenet_v3_large"     # "mobilenet_v3_large" | "fastvit_t8"
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 3e-4
TRAIN_SPLIT = 0.80                  # 80% train, 20% validation
NUM_CLASSES = len(AESTHETIC_TIERS)  # 3 classes

# Early stopping — stop training if val_loss doesn't improve for N epochs
EARLY_STOPPING_PATIENCE = 5

# ─── Synthetic Qwen Dataset (Phase 1b output → Phase 4 input) ───────────────
# See ML/pipeline/dataset_generator/ (Phase 1b, untouched by the trainer) and
# ML/pipeline/dataset_generator/taxonomy.py for the label schema this feeds.
SYNTHETIC_DIR     = DATA_ROOT / "4_Synthetic_Qwen"
SYNTHETIC_RAW_DIR = SYNTHETIC_DIR / "raw_generated"     # rsync'd from the GPU box, unfiltered
SYNTHETIC_QA_DIR  = SYNTHETIC_DIR / "qa_processed"      # after extract_measured_labels.py — has qa_pass

# ─── Category Consolidation (Phase 4/5) ──────────────────────────────────────
# The synthetic pipeline's 4 categories (no age split) replace the old
# 2-stream x 6-demographic (12 model) layout. Real-data's 3 age-bucket
# folders per gender+stream get pooled into one dataset per category.
CATEGORIES = ["Men_Grooming", "Women_Grooming", "Men_Outfit", "Women_Outfit"]

# category -> real-data stream folder name (TRAINING_DATA_DIR/<stream>/<demographic>/<tier>)
CATEGORY_TO_STREAM = {
    "Men_Grooming":   "Face_Grooming",
    "Women_Grooming": "Face_Grooming",
    "Men_Outfit":     "Outfit",
    "Women_Outfit":   "Outfit",
}

# ─── Phase 5 Fine-Tune Defaults ──────────────────────────────────────────────
FINETUNE_LEARNING_RATE = LEARNING_RATE / 10   # lower LR than Phase A pretrain
FINETUNE_NUM_EPOCHS    = 10                   # fewer epochs than Phase A pretrain
REPLAY_RATIO_DEFAULT   = 0.3                  # fraction of each batch drawn from synthetic replay

# ─── Scraping Rate Limiting ───────────────────────────────────────────────────
RATE_LIMIT_MIN_SEC = 1.0
RATE_LIMIT_MAX_SEC = 2.0
DOWNLOAD_WORKERS   = 8              # Threads for parallel image downloads
MAX_POSTS_PER_LISTING = 500         # Reddit posts to fetch per subreddit/listing

# ─── User-Agent ──────────────────────────────────────────────────────────────
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 " \
             "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 LookMaxPipeline/1.0"

# ─── Supported Image Extensions ──────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
