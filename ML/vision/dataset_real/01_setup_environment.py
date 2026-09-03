"""
LookMax ML Pipeline — Phase 1
==============================
01_setup_environment.py

Creates the complete ML data & model folder hierarchy, validates the macOS
Apple Silicon environment, sets up a Python virtual environment, and writes
.gitignore rules so the large image datasets are never accidentally committed.

Usage:
    python3 ML/vision/dataset_real/01_setup_environment.py

Options:
    --skip-download   Check tools without downloading models
    --verify-only     Quick verification pass on existing setup
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path

# config.py lives one level up at ML/vision/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ML_ROOT, DATA_ROOT, MODELS_DIR, RAW_SCRAPES_DIR,
    VLM_PROCESSING_DIR, REJECTED_DIR, METADATA_LOGS_DIR,
    TRAINING_DATA_DIR, REDDIT_SOURCES, DEMOGRAPHICS, AESTHETIC_TIERS,
    PIPELINE,
)

# ─── ANSI Colors for terminal output ────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def header(msg: str):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")

def ok(msg: str):
    print(f"  {GREEN}✓  {msg}{RESET}")

def warn(msg: str):
    print(f"  {YELLOW}⚠  {msg}{RESET}")

def err(msg: str):
    print(f"  {RED}✗  {msg}{RESET}")


# ─── Step 1: System & Architecture Validation ───────────────────────────────
def check_system():
    header("Step 1 — System Environment Validation")

    py_ver = sys.version_info
    print(f"  Python   : {sys.version.split()[0]}")
    print(f"  Platform : {platform.platform()}")
    print(f"  Machine  : {platform.machine()}")

    if py_ver < (3, 10):
        err(f"Python 3.10+ required, found {py_ver.major}.{py_ver.minor}")
        sys.exit(1)
    ok(f"Python {py_ver.major}.{py_ver.minor} — compatible")

    # Apple Silicon check
    machine = platform.machine()
    if machine == "arm64":
        ok("Apple Silicon (arm64) detected — MPS acceleration available")
    else:
        warn(f"Machine architecture is '{machine}' (not arm64). "
             "PyTorch MPS will not be available; training will fall back to CPU.")

    # RAM check
    try:
        import resource
        # macOS: use vm_stat to get approximate physical RAM
        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        ram_bytes = int(result.stdout.strip())
        ram_gb = ram_bytes / (1024 ** 3)
        print(f"  RAM      : {ram_gb:.1f} GB unified memory")
        if ram_gb >= 32:
            ok(f"{ram_gb:.1f} GB RAM — ideal for 7B VLM inference + training")
        elif ram_gb >= 16:
            warn(f"{ram_gb:.1f} GB RAM — usable, but keep VLM_PARALLEL_WORKERS ≤ 2 in config.py")
        else:
            warn(f"{ram_gb:.1f} GB RAM — consider using Gemini API backend instead of local VLM")
    except Exception:
        warn("Could not determine RAM size. Continuing.")


# ─── Step 2: Create Directory Hierarchy ─────────────────────────────────────
def create_directory_structure():
    header("Step 2 — Creating ML Data Folder Hierarchy")

    dirs_to_create = []

    # Stage 1: Raw scrapes — one sub-folder per Reddit source
    for source in REDDIT_SOURCES:
        dirs_to_create.append(RAW_SCRAPES_DIR / source["folder"])

    # Stage 2: VLM Processing
    dirs_to_create.append(REJECTED_DIR)
    dirs_to_create.append(METADATA_LOGS_DIR)

    # Stage 3: CoreML Training Data — 6 demographics × 3 aesthetic tiers
    for demo in DEMOGRAPHICS:
        for tier in AESTHETIC_TIERS:
            dirs_to_create.append(TRAINING_DATA_DIR / demo / tier)

    # ML/models/ — where .mlpackage artifacts are exported
    dirs_to_create.append(MODELS_DIR)

    created = 0
    for d in dirs_to_create:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            # Drop a .gitkeep so empty folders are tracked in git (structure only)
            (d / ".gitkeep").touch()
            created += 1
        # Ensure gitkeep exists even for pre-existing dirs
        gitkeep = d / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    ok(f"Created / verified {len(dirs_to_create)} directories ({created} new)")
    print(f"\n  Root    : {ML_ROOT}")
    print(f"  Data    : {DATA_ROOT}")
    print(f"  Models  : {MODELS_DIR}")


# ─── Step 3: Write .gitignore ────────────────────────────────────────────────
def setup_gitignore():
    header("Step 3 — Configuring .gitignore for ML Data")

    gitignore_path = ML_ROOT / ".gitignore"
    rules = [
        "# ─── LookMax ML Data — DO NOT commit large image datasets or temporary scrape logs ───",
        "data/vision_real/1_Raw_Scrapes/",
        "data/vision_real/2_VLM_Processing/filtered_rejected/",
        "data/vision_real/3_CoreML_Training_Data/",
        "data/vision_synthetic/",
        "data/stylist_llm/",
        "vision/dataset_synthetic/output/",
        "stylist_llm/checkpoints/",
        ".cache/",
        "2_VLM_Processing/",
        "**/metadata_logs/*.txt",
        "**/metadata_logs/*.json",
        "scraped_urls.txt",
        "",
        "# Keep the directory structure",
        "!data/**/.gitkeep",
        "!data/**/.keep",
        "!**/.gitkeep",
        "",
        "# ─── Browser Profiles & Temporary Session Data ───",
        "**/reddit_profile/",
        "**/chrome_profile/",
        "*profile/",
        "Singleton*",
        "RunningChromeVersion",
        "ChromeFeatureState",
        "",
        "# ─── Large ML Models & Checkpoints ───",
        "models/*.mlpackage/",
        "models/*.mlmodel",
        "models/*.pth",
        "models/*.pt",
        "models/*.onnx",
        "models/*.bin",
        "",
        "# ─── Python Environment & Caches ───",
        ".venv/",
        "venv/",
        "env/",
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".pytest_cache/",
        ".coverage",
        "htmlcov/",
        "*.log",
        "",
        "# ─── macOS ───",
        ".DS_Store",
        ".DS_Store?",
        "._*",
    ]

    with open(gitignore_path, "w") as f:
        f.write("\n".join(rules) + "\n")

    ok(f"Written: {gitignore_path}")


# ─── Step 4: Create requirements.txt ────────────────────────────────────────
def write_requirements():
    header("Step 4 — Writing requirements.txt")

    reqs = [
        "# ─── LookMax ML Pipeline Dependencies ───────────────────────",
        "# Networking / Scraping",
        "requests>=2.32.0",
        "aiohttp>=3.9.0",
        "beautifulsoup4>=4.12.0",
        "tqdm>=4.66.0",
        "playwright>=1.40.0",
        "",
        "# Image Processing & Heuristics",
        "opencv-python-headless>=4.9.0",
        "Pillow>=10.3.0",
        "numpy>=1.26.0",
        "",
        "# Local VLM — MLX for Apple Silicon (M-series native)",
        "mlx>=0.14.0; sys_platform == 'darwin'",
        "mlx-vlm>=0.1.0; sys_platform == 'darwin'",
        "",
        "# Google Gemini (cloud fallback)",
        "google-genai>=1.0.0",
        "",
        "# PyTorch with MPS (Apple Silicon GPU) — install via pip with macOS arm64 wheel",
        "torch>=2.3.0",
        "torchvision>=0.18.0",
        "",
        "# CoreML Export",
        "coremltools>=7.2",
        "",
        "# Utilities",
        "scikit-learn>=1.5.0",
        "pandas>=2.2.0",
    ]

    req_path = PIPELINE / "requirements.txt"
    with open(req_path, "w") as f:
        f.write("\n".join(reqs) + "\n")

    ok(f"Written: {req_path}")


# ─── Step 5: Create __init__.py ─────────────────────────────────────────────
def write_init():
    init_path = PIPELINE / "__init__.py"
    if not init_path.exists():
        init_path.write_text("# LookMax ML Pipeline Package\n")
    ok(f"Written: {init_path}")


# ─── Step 6: Check Key Dependencies ─────────────────────────────────────────
def check_dependencies():
    header("Step 5 — Checking Key Dependencies")

    checks = {
        "torch":        "PyTorch (MPS acceleration for M3 Max)",
        "torchvision":  "TorchVision (dataset & transform utilities)",
        "coremltools":  "CoreML Tools (model export for iOS)",
        "cv2":          "OpenCV (image quality heuristics)",
        "PIL":          "Pillow (image I/O)",
        "requests":     "Requests (HTTP scraping)",
        "playwright":   "Playwright (authenticated browser scraping)",
        "tqdm":         "tqdm (progress bars)",
    }

    missing = []
    for module, desc in checks.items():
        try:
            __import__(module)
            ok(desc)
        except ImportError:
            warn(f"{desc} — NOT INSTALLED (run: pip install -r ML/vision/requirements.txt)")
            missing.append(module)

    # Check gallery-dl (CLI tool)
    if shutil.which("gallery-dl"):
        ok("gallery-dl CLI — found")
    else:
        warn("gallery-dl CLI — NOT FOUND (run: pip install gallery-dl OR brew install gallery-dl)")
        missing.append("gallery-dl")

    # Check Ollama
    if shutil.which("ollama"):
        ok("Ollama — found (confirm 'qwen2.5vl:7b' is pulled)")
    else:
        warn("Ollama — NOT FOUND (optional: install from https://ollama.com)")

    # MPS check
    try:
        import torch
        if torch.backends.mps.is_available():
            ok("PyTorch MPS backend — AVAILABLE (Apple Neural Engine ready)")
        else:
            warn("PyTorch MPS backend — NOT AVAILABLE (CPU fallback will be used)")
    except ImportError:
        pass

    if missing:
        print(f"\n  {YELLOW}Install all dependencies with:{RESET}")
        print(f"  {BOLD}  pip install -r ML/vision/requirements.txt{RESET}")
        if "gallery-dl" in missing:
            print(f"  {BOLD}  pip install gallery-dl{RESET}")

    return missing


# ─── Step 7: Print Final Tree ────────────────────────────────────────────────
def print_summary():
    header("Setup Complete — Directory Tree")
    try:
        result = subprocess.run(
            ["find", str(ML_ROOT), "-maxdepth", "4", "-type", "d"],
            capture_output=True, text=True
        )
        lines = sorted(result.stdout.strip().split("\n"))
        for line in lines:
            rel = line.replace(str(ML_ROOT), "ML")
            depth = rel.count("/")
            indent = "  " + "    " * (depth - 1) + "├── " if depth > 0 else ""
            print(f"  {CYAN}{indent}{Path(line).name}{RESET}" if depth > 0 else f"  {BOLD}{CYAN}{rel}{RESET}")
    except Exception:
        print(f"  Directory root: {ML_ROOT}")

    print(f"\n  {GREEN}{BOLD}Next Step:{RESET} python3 ML/vision/dataset_real/02_scrape_images.py --limit 50\n")


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{BOLD}{CYAN}╔═══════════════════════════════════════════════════╗")
    print(f"║   LookMax ML Pipeline — Environment Setup v1.0   ║")
    print(f"╚═══════════════════════════════════════════════════╝{RESET}")

    check_system()
    create_directory_structure()
    setup_gitignore()
    write_requirements()
    write_init()
    check_dependencies()
    print_summary()
