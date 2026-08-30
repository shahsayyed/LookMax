# LookMax ML

Machine learning pipeline and demographic CoreML model training for LookMax.

## Structure

```
ML/
├── pipeline/           # Python scripts (Phases 1–4)
│   ├── config.py       # Centralized configuration
│   ├── 01_setup_environment.py
│   ├── 02_scrape_images.py
│   ├── reddit_scraper.py   # Authenticated Playwright Reddit JSON scraper
│   ├── 03_classify_and_sort.py
│   └── 04_train_coreml_models.py
├── models/             # Exported .mlpackage artifacts for iOS
└── data/               # Large image datasets (gitignored)
    ├── 1_Raw_Scrapes/
    ├── 2_VLM_Processing/
    └── 3_CoreML_Training_Data/
```

## Quick Start

```bash
# 1. Set up folders & verify environment
python3 ML/pipeline/01_setup_environment.py

# 2. Install dependencies & Playwright browser
pip install -r ML/pipeline/requirements.txt
playwright install chromium

# 3. Scrape images
# Option A: Playwright Reddit JSON scraper (authenticated browser session)
python3 ML/pipeline/reddit_scraper.py --login      # One-time manual login
python3 ML/pipeline/reddit_scraper.py --download --limit 200

# Option B: Multi-source free APIs (Unsplash, Pexels, Pixabay)
python3 ML/pipeline/02_scrape_images.py --limit 200

# 4. Classify and auto-sort (choose your VLM engine)
export GEMINI_API_KEY=your_key_here   # for Gemini backend
python3 ML/pipeline/03_classify_and_sort.py --engine ollama

# 5. Train CoreML models
python3 ML/pipeline/04_train_coreml_models.py

# Drag ML/models/LookMax_*.mlpackage into Xcode!
```

## VLM Engine Options

| Engine    | Command Flag        | Notes                                      |
|-----------|---------------------|--------------------------------------------|
| `ollama`  | `--engine ollama`   | Local inference. Pull `ollama pull qwen2.5vl:7b` |
| `mlx_vlm` | `--engine mlx_vlm`  | M3 Max native. `pip install mlx-vlm` required  |
| `gemini`  | `--engine gemini`   | Cloud. Set `GEMINI_API_KEY` env var         |
