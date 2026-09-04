# Graph Report - LookMax  (2026-09-04)

## Corpus Check
- 13 files · ~116,246 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2105 nodes · 3492 edges · 168 communities (112 shown, 26 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 171 edges (avg confidence: 0.81)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Community 30
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 36
- Community 37
- Community 38
- Community 39
- Community 40
- Community 41
- Community 42
- Community 43
- Community 44
- Community 45
- Community 46
- Community 47
- Community 48
- Community 49
- Community 50
- Community 51
- Community 52
- Community 53
- Community 54
- Community 55
- Community 56
- Community 57
- Community 58
- Community 59
- Community 60
- Community 61
- Community 62
- Community 63
- Community 64
- Community 65
- Community 66
- Community 67
- Community 68
- Community 69
- Community 70
- Community 71
- Community 72
- Community 73
- Community 74
- Community 75
- Community 76
- Community 77
- Community 78
- Community 79
- Community 80
- Community 81
- Community 82
- Community 83
- Community 84
- Community 85
- Community 86
- Community 87
- Community 88
- Community 89
- Community 90
- Community 91
- Community 92
- Community 93
- Community 94
- Community 95
- Community 96
- Community 97
- Community 98
- Community 99
- Community 100
- Community 101
- Community 102
- Community 103
- Community 104
- Community 105
- Community 106
- Community 107
- Community 108
- Community 109
- Community 110
- Community 111
- Community 112
- Community 113
- Community 114
- Community 115
- Community 116
- Community 117
- Community 118
- Community 119
- Community 120
- Community 121
- Community 122
- Community 123
- Community 126
- Community 127
- Community 128
- Community 129
- Community 132
- Community 133
- Community 134
- Community 145
- Community 146
- Community 151
- Community 153
- Community 156
- Community 157
- Community 159

## God Nodes (most connected - your core abstractions)
1. `CameraController` - 38 edges
2. `LookSession` - 32 edges
3. `LookItem` - 29 edges
4. `CameraController` - 28 edges
5. `OccasionCategory` - 23 edges
6. `BeforeAfterComparisonView` - 21 edges
7. `CodingKeys` - 20 edges
8. `CodingKeys` - 20 edges
9. `Theme` - 19 edges
10. `CodingKeys` - 19 edges

## Surprising Connections (you probably didn't know these)
- `Effort-vs-Genetics Product Philosophy` --rationale_for--> `SYSTEM_PROMPT text`  [INFERRED]
  AGENTS.md → ML/stylist_llm/config.py
- `Effort-vs-Genetics Product Philosophy` --rationale_for--> `Body-Proportion Clause Fix`  [AMBIGUOUS]
  AGENTS.md → ML/vision/dataset_synthetic/PLAN.md
- `ML/README.md ML Pipeline Overview` --references--> `IOS_MIN_DEPLOYMENT setting`  [EXTRACTED]
  ML/README.md → ML/stylist_llm/config.py
- `Stylist LLM Pipeline PLAN.md` --references--> `qa_review.py QA gate`  [EXTRACTED]
  ML/stylist_llm/PLAN.md → ML/stylist_llm/qa_review.py
- `Outfit Tag Priority-Defect Information Loss` --references--> `priority_defect`  [EXTRACTED]
  ML/stylist_llm/PLAN.md → ML/stylist_llm/tag_vocabulary.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **coremltools 9.0 torch-frontend bug workarounds** — ml_stylist_llm_export_coreml_patch_coremltools_numpy2_cast_bug, ml_stylist_llm_export_coreml_patch_coremltools_mixed_shape_view_bug, ml_stylist_llm_export_coreml_patch_coremltools_to_op_overloads, ml_stylist_llm_export_coreml_patch_coremltools_alias_op, ml_stylist_llm_export_coreml_patch_coremltools_diff_op, ml_stylist_llm_export_coreml_patch_coremltools_new_ones_op, ml_stylist_llm_export_coreml_patch_coremltools_bitwise_and_mixed_dtype, ml_stylist_llm_export_coreml_patch_coremltools_inplace_op_dispatch_for_exir [EXTRACTED 1.00]
- **Deterministic task-context sampling reusing the real vision taxonomy** — ml_stylist_llm_generate_synthetic_dataset_build_task_contexts, ml_stylist_llm_smoke_test_build_review_contexts, ml_vision_dataset_synthetic_full_run_build_full_task_list, ml_vision_dataset_synthetic_variation_test_build_variation_tasks, dataset_synthetic_prompt_builder_build_task [EXTRACTED 1.00]
- **Duplicated Vast.ai actual-data-dir disk-space guard** — ml_vision_dataset_synthetic_full_run_check_disk_space, ml_vision_dataset_synthetic_quick_prompt_test_check_disk_space, ml_vision_dataset_synthetic_variation_test_check_disk_space [INFERRED 0.95]

## Communities (168 total, 26 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (43): AVCaptureConnection, AVCaptureDevice, AVCaptureDeviceInput, AVCaptureOutput, AVCapturePhoto, AVCapturePhotoCaptureDelegate, AVCapturePhotoOutput, AVCaptureVideoDataOutputSampleBufferDelegate (+35 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (64): append_annotation_record(), classify_with_gemini(), classify_with_mlx_vlm(), classify_with_ollama(), _get_gemini_client(), heuristic_filter(), image_to_base64(), load_processed_files() (+56 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (35): For --cluster-only, For git commit hook, For /graphify add, For /graphify explain, For /graphify path, For /graphify query, For native CLAUDE.md integration, For --update (incremental re-extraction) (+27 more)

### Community 3 - "Community 3"
Cohesion: 0.10
Nodes (22): CGImage, GeminiLookEvaluation, LookAnalysisEngine, OccasionProfile, CGRect, Color, Double, OccasionCategory (+14 more)

### Community 4 - "Community 4"
Cohesion: 0.06
Nodes (35): For --cluster-only, For git commit hook, For /graphify add, For /graphify explain, For /graphify path, For /graphify query, For native CLAUDE.md integration, For --update (incremental re-extraction) (+27 more)

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (31): dataset_token_ids(), floor_token_ids(), _load_tokenizer(), main(), prune_vocabulary.py -- shrinks the base model's embedding/lm_head matrices to…, Token ids used across every QA-passed example's full ChatML text (system + user…, run_dry_run(), run_prune() (+23 more)

### Community 6 - "Community 6"
Cohesion: 0.06
Nodes (33): architectures, attention_bias, attention_dropout, bos_token_id, dtype, eos_token_id, head_dim, hidden_act (+25 more)

### Community 7 - "Community 7"
Cohesion: 0.06
Nodes (33): architectures, attention_bias, attention_dropout, bos_token_id, dtype, eos_token_id, head_dim, hidden_act (+25 more)

### Community 8 - "Community 8"
Cohesion: 0.06
Nodes (33): architectures, attention_bias, attention_dropout, bos_token_id, dtype, eos_token_id, head_dim, hidden_act (+25 more)

### Community 9 - "Community 9"
Cohesion: 0.06
Nodes (33): architectures, attention_bias, attention_dropout, bos_token_id, dtype, eos_token_id, head_dim, hidden_act (+25 more)

### Community 10 - "Community 10"
Cohesion: 0.06
Nodes (33): architectures, attention_bias, attention_dropout, bos_token_id, dtype, eos_token_id, head_dim, hidden_act (+25 more)

### Community 11 - "Community 11"
Cohesion: 0.11
Nodes (31): color_harmony(), _crop_fraction(), _lazy_imports(), _load_face_cascade(), main(), measured_dominant_color(), _palette_lab(), process() (+23 more)

### Community 12 - "Community 12"
Cohesion: 0.11
Nodes (15): Content, Color, GlassCardModifier, NeonGlowModifier, Theme, View, Color, String (+7 more)

### Community 13 - "Community 13"
Cohesion: 0.11
Nodes (29): _download_single_image(), LookMax ML Pipeline — Reddit Playwright JSON Image Scraper =====================, Save category image URL map to a JSON file., Print a clean summary of scraped image counts per category., Ensure playwright package is available before executing browser flows., Discover existing Google Chrome profiles on the host system., Print all discovered Google Chrome profiles with their folder and display names., Launch a persistent Chromium context storing session cookies in user_data_dir. (+21 more)

### Community 14 - "Community 14"
Cohesion: 0.10
Nodes (22): chunked(), load_qwen_pipeline(), Shared GPU-aware pipeline loader for the Qwen-Image-2512 scripts.  Qwen-Image-25, Returns (pipe, can_batch). can_batch is True only when the full pipeline     is, Split seq into consecutive chunks of at most `size` items each -- the last chunk, build_grooming_prompt(), build_outfit_prompt(), _pick() (+14 more)

### Community 15 - "Community 15"
Cohesion: 0.13
Nodes (26): finetune_category(), main(), --checkpoint may be a single .pt file (only valid for one category) or a…, resolve_checkpoint(), compute_losses(), discover_real_samples(), discover_synthetic_source(), evaluate() (+18 more)

### Community 16 - "Community 16"
Cohesion: 0.16
Nodes (17): CameraGridView, BeforeAfterComparisonView, View, BeforeAfterComparisonView, .microMetricsSection, .performanceInsightsCard, .scoreBadge, .scoreDiff (+9 more)

### Community 17 - "Community 17"
Cohesion: 0.07
Nodes (28): should_epoch_stop, should_evaluate, should_log, should_save, should_training_stop, best_global_step, best_metric, best_model_checkpoint (+20 more)

### Community 18 - "Community 18"
Cohesion: 0.07
Nodes (28): should_epoch_stop, should_evaluate, should_log, should_save, should_training_stop, best_global_step, best_metric, best_model_checkpoint (+20 more)

### Community 19 - "Community 19"
Cohesion: 0.07
Nodes (28): should_epoch_stop, should_evaluate, should_log, should_save, should_training_stop, best_global_step, best_metric, best_model_checkpoint (+20 more)

### Community 20 - "Community 20"
Cohesion: 0.12
Nodes (23): LookSession, .averageScore, .bestLook, .firstLook, .formattedDate, .latestLook, .tagsFormatted, Date (+15 more)

### Community 21 - "Community 21"
Cohesion: 0.11
Nodes (28): BASE_MODEL_ID (SmolLM2-135M-Instruct), IOS_MIN_DEPLOYMENT setting, MAX_TOTAL_TOKENS bound, QUANT_DTYPE setting, _build_stateless_wrapper, main, alias op registration patch, bitwise_and mixed-dtype patch (+20 more)

### Community 22 - "Community 22"
Cohesion: 0.13
Nodes (21): BrowserContext, fetch_json(), RateLimitTracker, Scrape image URLs across multiple categories with rate-limiting controls,     in, Navigate to a Reddit .json URL using an authenticated Playwright page and parse, Tracks page requests and enforces human browsing pauses., Call after each request. Executes randomized delay and batch cooldowns., Call after finishing a category before moving to the next. (+13 more)

### Community 23 - "Community 23"
Cohesion: 0.20
Nodes (26): bytes, download_image(), load_seen(), log_seen(), main(), rate_limit(), LookMax ML Pipeline — Phase 2 (v3) ===================================== 02_scra, scrape_pexels() (+18 more)

### Community 24 - "Community 24"
Cohesion: 0.16
Nodes (26): classify_description(), load_celebahq_annotations(), main(), LookMax ML Pipeline — CelebA-HQ ($1024\\times1024$) Facial Ingestion Engine ====, Evaluates a CelebA-HQ natural language annotation and routes into     LookMax de, Downloads and loads the 30,000 CelebA-HQ natural language annotations., Streams $1024x1024$ CelebA-HQ images shard-by-shard, mapping the true     numeri, Validates, optionally resizes, and writes high-res image to disk. (+18 more)

### Community 25 - "Community 25"
Cohesion: 0.16
Nodes (26): classify_and_score_photo(), download_image_task(), find_dataset_dir(), load_dataset_metadata(), main(), LookMax ML Pipeline — Unsplash Research Dataset Loader =========================, Locate the Unsplash dataset directory containing TSV files., Load photos.tsv000 and keywords.tsv000 from the dataset folder using high-speed (+18 more)

### Community 26 - "Community 26"
Cohesion: 0.12
Nodes (18): bool, float, int, str, Module, build_backbone(), discover_real_samples(), encode_target() (+10 more)

### Community 27 - "Community 27"
Cohesion: 0.14
Nodes (12): Coordinator, ImagePicker, Coordinator, Coordinator, ImagePicker, Any, Context, UIImage (+4 more)

### Community 28 - "Community 28"
Cohesion: 0.08
Nodes (25): 10. Merge shards (if you sharded), 11. Measure pixel-level labels (Bucket C), 12. Bring the finished dataset back to this machine, 1. Setup (remote GPU box), 1. Vast.ai disk quirk: `/workspace` is tiny, the large disk is elsewhere, 2. Cheap sanity checks first — no GPU, 2. tmux + shell env: don't trust either across a reattach, 3. Coverage sweep — no GPU, before ANY GPU time is spent (+17 more)

### Community 29 - "Community 29"
Cohesion: 0.11
Nodes (8): build_reddit_json_url(), clean_and_unescape_url(), extract_image_urls(), Construct a valid Reddit .json endpoint URL supporting feeds and search queries., Unescapes HTML entities (&amp; -> &) and strips whitespace., Unit Tests for Reddit Playwright JSON Scraper (reddit_scraper.py) ==============, TestRedditScraper, clean_and_unescape_url()

### Community 30 - "Community 30"
Cohesion: 0.11
Nodes (21): EmptySessionsView, ProfileHeaderBanner, SessionCardView, StartSessionCTA, Theme, .body, EmptySessionsView, .body (+13 more)

### Community 31 - "Community 31"
Cohesion: 0.14
Nodes (18): build_eyebrows(), build_facial_hair(), build_hair(), build_makeup(), build_outfit_condition(), build_skin(), get_label_schema(), positive_for_tier() (+10 more)

### Community 32 - "Community 32"
Cohesion: 0.14
Nodes (16): _build_dataset(), _load_examples(), main(), Returns a list of {"input_ids": [...], "labels": [...]} -- labels match…, run_dry_run(), run_finetune(), remap_tokenizer.py -- wraps the ORIGINAL (unpruned) tokenizer plus…, Uses the base tokenizer's chat template (ChatML) to build the prompt STRING,… (+8 more)

### Community 33 - "Community 33"
Cohesion: 0.12
Nodes (16): LookCarouselView, LookCarouselView, .body, LookItem, UUID, Void, SessionDetailView, .body (+8 more)

### Community 34 - "Community 34"
Cohesion: 0.09
Nodes (22): Cleanup Commands, Critical Disk Quirk on Vast.ai, Dataset Generator: FLUX.1 [dev] Synthetic Image Pipeline, Design Principles, Hardware Used, Identity Matrix, Install Dependencies, Known Issues & Fixes (+14 more)

### Community 35 - "Community 35"
Cohesion: 0.19
Nodes (21): main(), map_lookmax_demographic(), LookMax ML Pipeline — FairFace In-The-Wild Dataset Ingestion Engine ============, Maps FairFace age & gender integer codes to LookMax demographic categories., Streams candid FairFace records shard-by-shard, balancing demographics     until, Validates, optionally resizes to standard training dimension, and writes image t, Main execution function for FairFace ingestion., run_pipeline() (+13 more)

### Community 36 - "Community 36"
Cohesion: 0.13
Nodes (11): Check if the URL path ends with a recognized image extension., Check if post is a video, gifv, or non-image embed., Check if post is removed, deleted, or empty., Extract direct image URLs from a Reddit listing JSON response.      Handles:, build_reddit_json_url(), extract_image_urls(), is_deleted_or_removed(), is_direct_image_url() (+3 more)

### Community 37 - "Community 37"
Cohesion: 0.16
Nodes (18): bool, str, bool, str, finetune_category(), main(), --checkpoint may be a single .pt file (only valid for one category)     or a dir, resolve_checkpoint() (+10 more)

### Community 38 - "Community 38"
Cohesion: 0.10
Nodes (20): 10. On-device verification (not scriptable from here), 1. Setup, 2. Cheap sanity checks -- no GPU, no API key, no Ollama call, 3. Generate the real synthetic dataset, 4. QA review, 5. Prune the vocabulary, 6. Fine-tune, 7. Smoke-test the checkpoint before spending time on export (+12 more)

### Community 39 - "Community 39"
Cohesion: 0.25
Nodes (19): _clean_stale_profile_locks(), create_persistent_context(), _ensure_playwright(), get_system_chrome_profiles(), is_deleted_or_removed(), is_direct_image_url(), is_session_valid(), is_video_or_media_embed() (+11 more)

### Community 40 - "Community 40"
Cohesion: 0.17
Nodes (14): Binary polish-achieved indicator: 1 only for the polished tier., build_eyebrows(), build_facial_hair(), build_hair(), build_makeup(), build_outfit_condition(), build_skin(), positive_for_tier() (+6 more)

### Community 41 - "Community 41"
Cohesion: 0.11
Nodes (18): CodingKeys, badPoints, detectedFaceShape, detectedOutfitColor, fitNote, fitScore, goodPoints, groomingScore (+10 more)

### Community 42 - "Community 42"
Cohesion: 0.11
Nodes (18): CodingKeys, badPoints, detectedFaceShape, detectedOutfitColor, fitNote, fitScore, goodPoints, groomingScore (+10 more)

### Community 43 - "Community 43"
Cohesion: 0.16
Nodes (10): Data, Date, FaceBiometricSignature, String, UserProfile, ProfileOnboardingView, .body, .canSave (+2 more)

### Community 44 - "Community 44"
Cohesion: 0.17
Nodes (17): GEMINI_MODEL setting, GENERATOR_BACKEND setting, OLLAMA_MODEL setting, build_task_contexts, _call_gemini, _call_ollama, _gemini_client, main (+9 more)

### Community 45 - "Community 45"
Cohesion: 0.18
Nodes (13): AnyClass, AVCaptureVideoPreviewLayer, CameraPreviewUIView, CameraPreviewView, CameraPreviewUIView, CameraPreviewUIView, .layerClass, .previewLayer (+5 more)

### Community 46 - "Community 46"
Cohesion: 0.22
Nodes (6): AVFoundation, CoreImage, CameraGridView, .body, SwiftUI, Vision

### Community 47 - "Community 47"
Cohesion: 0.18
Nodes (8): Binding, ContentView, .bottomTabBar, Int, LookSession, String, UserProfile, ContentView

### Community 48 - "Community 48"
Cohesion: 0.12
Nodes (15): CodingKey, CodingKeys, createdAt, id, looks, occasion, tags, title (+7 more)

### Community 49 - "Community 49"
Cohesion: 0.19
Nodes (7): Combine, CGImagePropertyOrientation, Foundation, ImageIO, CGImagePropertyOrientation, UIImage, UIKit

### Community 50 - "Community 50"
Cohesion: 0.28
Nodes (14): Identifiable, LookItem, .formattedDate, .formattedTime, .image, Date, Decoder, Double (+6 more)

### Community 51 - "Community 51"
Cohesion: 0.12
Nodes (16): CodingKeys, category, effortTime, fitNote, formalityScore, goodPoints, headlineBadge, improvementPoints (+8 more)

### Community 52 - "Community 52"
Cohesion: 0.23
Nodes (9): Bool, CGPoint, CGRect, Double, FaceBiometricSignature, String, UserProfile, ProfileOnboardingView (+1 more)

### Community 53 - "Community 53"
Cohesion: 0.14
Nodes (8): build_backbone(), _FeatureExtractor, _merge_batches(), MultiHeadModel, Loads a pretrained backbone and strips its final classification layer, keeping…, One shared backbone + one small linear head per trainable schema field:…, ReplayMixedLoader, Yields batches built from `real_n` real samples + `synth_n`     synthetic replay

### Community 54 - "Community 54"
Cohesion: 0.12
Nodes (16): CodingKeys, category, effortTime, fitNote, formalityScore, goodPoints, headlineBadge, improvementPoints (+8 more)

### Community 55 - "Community 55"
Cohesion: 0.13
Nodes (14): add_prefix_space, backend, bos_token, clean_up_tokenization_spaces, eos_token, errors, extra_special_tokens, is_local (+6 more)

### Community 56 - "Community 56"
Cohesion: 0.13
Nodes (14): add_prefix_space, backend, bos_token, clean_up_tokenization_spaces, eos_token, errors, extra_special_tokens, is_local (+6 more)

### Community 57 - "Community 57"
Cohesion: 0.13
Nodes (14): add_prefix_space, backend, bos_token, clean_up_tokenization_spaces, eos_token, errors, extra_special_tokens, is_local (+6 more)

### Community 58 - "Community 58"
Cohesion: 0.17
Nodes (13): DEFAULT_GEN_BATCH_SIZE, Batch=1 Default Throughput Finding, Dataset Generator PLAN.md, Qwen-Image-2512 Selection Over FLUX/Nano Banana, Vast.ai /workspace Disk Quirk, GPU VRAM Auto-Detection Lesson, BODY_LOCK template, check_disk_space (+5 more)

### Community 59 - "Community 59"
Cohesion: 0.24
Nodes (13): build_variation_tasks, check_disk_space, group_by_resolution, main, variation_test.py -- broader diversity/quality check than smoke_test.py's…, Launch num_workers concurrent worker processes on the GPU, each handling a…, Benchmark and compare generation modes: 1. Sequential (1 worker, batch=1) 2.…, Deterministic: samples_per_cell tasks per (category, tier), in a fixed order,… (+5 more)

### Community 60 - "Community 60"
Cohesion: 0.13
Nodes (14): add_prefix_space, backend, bos_token, clean_up_tokenization_spaces, eos_token, errors, extra_special_tokens, is_local (+6 more)

### Community 61 - "Community 61"
Cohesion: 0.24
Nodes (8): GeminiTweak, Any, Double, OccasionCategory, String, GeminiLookEvaluation, GeminiTweak, GeminiVisionService

### Community 62 - "Community 62"
Cohesion: 0.29
Nodes (14): already_done_indices, build_full_task_list, check_disk_space, group_by_resolution, labels_csv_path(), main, output_paths(), full_run.py -- the 28,000-image production run (Qwen-Image-2512). Only ever… (+6 more)

### Community 63 - "Community 63"
Cohesion: 0.20
Nodes (10): LookDetailCard, SuggestionRow, LookDetailCard, .body, SuggestionRow, .body, Bool, LookItem (+2 more)

### Community 64 - "Community 64"
Cohesion: 0.19
Nodes (9): ScoreComparisonBar, .interactiveCurtain, ScoreComparisonBar, .body, CGFloat, Double, LookItem, UUID (+1 more)

### Community 65 - "Community 65"
Cohesion: 0.34
Nodes (13): check_dependencies(), check_system(), create_directory_structure(), err(), header(), ok(), print_summary(), LookMax ML Pipeline — Phase 1 ============================== 01_setup_environmen (+5 more)

### Community 66 - "Community 66"
Cohesion: 0.15
Nodes (7): HapticManager, CreateSessionSheet, .body, LookSession, OccasionCategory, Void, CreateSessionSheet

### Community 67 - "Community 67"
Cohesion: 0.25
Nodes (5): LookItem, LookSession, String, UIImage, SessionStorageManager

### Community 68 - "Community 68"
Cohesion: 0.21
Nodes (13): Outfit Tag Priority-Defect Information Loss, _defect_fields_for(), format_full_prompt(), format_tag_prompt, full_vocabulary_terms(), garment_vocabulary(), _grooming_tag_lines(), OCCASIONS (+5 more)

### Community 69 - "Community 69"
Cohesion: 0.24
Nodes (12): build_grooming_task(), build_outfit_task(), build_task, _garment_clause(), _plain_garment_clause(), prompt_builder.py -- composes ONE (prompt, resolution, label-row) triple per ima, Assemble the full CSV row dict (filename/category/tier + every     schema column, a graphic t-shirt' -> 'graphic t-shirt' (so a pattern/colour can be     inserted (+4 more)

### Community 70 - "Community 70"
Cohesion: 0.24
Nodes (12): Returns {"prompt": str, "resolution": (w, h), "row": {field_name: value, ...}}, No pattern axis (mid layer, footwear): 'a navy pullover hoodie'., build_grooming_task(), build_outfit_task(), build_task(), _garment_clause(), _plain_garment_clause(), prompt_builder.py -- composes ONE (prompt, resolution, label-row) triple per… (+4 more)

### Community 71 - "Community 71"
Cohesion: 0.18
Nodes (13): _formality_bucket_pick(), formality_tier(), Coherent sampling for one slot: 75% of the time, restrict to entries     within, Independently samples the garment-composition axis: which item in     which slot, Thresholds, NOT round() -- see sample_outfit()'s docstring., sample_color(), sample_outfit(), _weighted_choice() (+5 more)

### Community 72 - "Community 72"
Cohesion: 0.35
Nodes (12): check_dependencies(), check_system(), create_directory_structure(), err(), header(), ok(), print_summary(), LookMax ML Pipeline — Phase 1 ==============================… (+4 more)

### Community 73 - "Community 73"
Cohesion: 0.29
Nodes (6): CGSize, GeminiVisionService, .apiKey, Data, Int, UIImage

### Community 74 - "Community 74"
Cohesion: 0.17
Nodes (7): Dataset, encode_target(), Converts one raw label value (a CSV string, or a python float/str already) into…, Wraps a flat list of (image_path, tier) samples pooled across all age-…, RealWorldScoreDataset, SyntheticCsvDataset, One row per generated image. Every trainable schema field gets a     real target

### Community 75 - "Community 75"
Cohesion: 0.30
Nodes (10): iter_smoke_tasks(), main(), smoke_test.py -- cheap sanity checks before spending any GPU time.    --dry-run, Deterministic: one task per (category, tier), in a fixed order., run_dry_run(), run_per_tier(), iter_smoke_tasks(), main() (+2 more)

### Community 76 - "Community 76"
Cohesion: 0.29
Nodes (4): .body, SessionStorageManager, .documentsDir, URL

### Community 77 - "Community 77"
Cohesion: 0.27
Nodes (9): AGENTS.md Agent Rules, Dual-Model On-Device Architecture, Effort-vs-Genetics Product Philosophy, ML/README.md ML Pipeline Overview, _PacedRateLimiter, Minimal local rate limiter -- deliberately not importing real_data_pipeline's…, Stylist LLM Pipeline PLAN.md, Pipeline Isolation Principle (+1 more)

### Community 78 - "Community 78"
Cohesion: 0.35
Nodes (7): StyleSuggestion, .iconColor, Bool, Color, String, UUID, StyleSuggestion

### Community 79 - "Community 79"
Cohesion: 0.31
Nodes (10): MAX_NEW_TOKENS setting, MIN_VOCAB_TOKENS floor, SYSTEM_PROMPT text, prune_vocabulary.py vocabulary pruning, build_review_contexts, main, _print_result, smoke_test.py -- generate a handful of real prompts and eyeball the output,… (+2 more)

### Community 80 - "Community 80"
Cohesion: 0.20
Nodes (9): CaseIterable, Color, OccasionCategory, businessMeeting, casualEveryday, custom, dateNight, formalEvent (+1 more)

### Community 81 - "Community 81"
Cohesion: 0.20
Nodes (9): 1. Vision Model Pipeline (`ML/vision/`), 2. Stylist LLM Pipeline (`ML/stylist_llm/`), Architecture Philosophy, Directory Structure, iOS Integration, LookMax ML Pipeline, Model Architecture, The Two Training Pipelines (+1 more)

### Community 82 - "Community 82"
Cohesion: 0.20
Nodes (6): evaluate(), Every schema entry except `meta` — meta fields (e.g.     requested_upper_color), Wraps a flat list of (image_path, tier) samples pooled across all     age-demogr, Runs a full pass, returning total masked loss, score MAE/RMSE (over     samples, RealWorldScoreDataset, trainable_fields()

### Community 83 - "Community 83"
Cohesion: 0.22
Nodes (8): chunked, generate, qwen_pipeline.py -- single shared model-loading + generation entry point for…, Best-effort GPU memory release -- not required for correctness (the driving…, Split seq into consecutive chunks of at most `size` items each., tasks: list of dicts each with 'prompt' and 'resolution' (w, h) -- all tasks in…, unload, ALL_CATEGORIES

### Community 84 - "Community 84"
Cohesion: 0.28
Nodes (3): remap_tokenizer.py -- wraps the ORIGINAL (unpruned) tokenizer plus prune_vocabul, Uses the base tokenizer's chat template (ChatML) to build the         prompt STR, RemappedTokenizer

### Community 85 - "Community 85"
Cohesion: 0.36
Nodes (4): Equatable, FaceBiometricSignature, Double, FaceBiometricSignature

### Community 86 - "Community 86"
Cohesion: 0.25
Nodes (8): Error, GeminiServiceError, .errorDescription, imageTooLarge, missingAPIKey, networkError, parsingError, rateLimited

### Community 87 - "Community 87"
Cohesion: 0.25
Nodes (7): Core Philosophy, Dual On-Device CoreML Pipeline, Getting Started, iOS App, LookMax, ML Pipelines, System Architecture

### Community 88 - "Community 88"
Cohesion: 0.33
Nodes (5): App, LookMaxApp, .body, LookMaxApp, Scene

### Community 89 - "Community 89"
Cohesion: 0.29
Nodes (7): LocalizedError, GeminiServiceError, imageTooLarge, missingAPIKey, networkError, parsingError, rateLimited

### Community 90 - "Community 90"
Cohesion: 0.48
Nodes (6): check_record(), main(), qa_review.py -- QA gate on generated advice text before fine-tuning on it.…, Returns (passed: bool, reasons: list[str]) -- reasons is non-empty only on…, run_review(), run_self_check()

### Community 91 - "Community 91"
Cohesion: 0.48
Nodes (6): check_record(), main(), qa_review.py -- QA gate on generated advice text before fine-tuning on it.  Unli, Returns (passed: bool, reasons: list[str]) -- reasons is non-empty     only on f, run_review(), run_self_check()

### Community 92 - "Community 92"
Cohesion: 0.33
Nodes (5): argv, bspVersion, languages, name, version

### Community 93 - "Community 93"
Cohesion: 0.33
Nodes (5): bos_token_id, eos_token_id, _from_model_config, pad_token_id, transformers_version

### Community 94 - "Community 94"
Cohesion: 0.33
Nodes (5): bos_token_id, eos_token_id, _from_model_config, pad_token_id, transformers_version

### Community 95 - "Community 95"
Cohesion: 0.33
Nodes (5): bos_token_id, eos_token_id, _from_model_config, pad_token_id, transformers_version

### Community 96 - "Community 96"
Cohesion: 0.33
Nodes (5): bos_token_id, eos_token_id, _from_model_config, pad_token_id, transformers_version

### Community 97 - "Community 97"
Cohesion: 0.33
Nodes (5): bos_token_id, eos_token_id, _from_model_config, pad_token_id, transformers_version

### Community 98 - "Community 98"
Cohesion: 0.40
Nodes (4): colors, info, author, version

### Community 99 - "Community 99"
Cohesion: 0.40
Nodes (4): images, info, author, version

### Community 100 - "Community 100"
Cohesion: 0.60
Nodes (4): main(), validation_sweep.py -- the gate before the full run.    --coverage-only   Simula, run_check_binding(), run_coverage_only()

### Community 102 - "Community 102"
Cohesion: 0.60
Nodes (4): main(), validation_sweep.py -- the gate before the full run. --coverage-only Simulate…, run_check_binding(), run_coverage_only()

### Community 103 - "Community 103"
Cohesion: 0.50
Nodes (3): 1. iOS Application Development Rules, 2. Core Machine Learning & Modeling Rules, Agent Rules for LookMax

### Community 104 - "Community 104"
Cohesion: 0.50
Nodes (3): info, author, version

### Community 105 - "Community 105"
Cohesion: 0.50
Nodes (3): LookMax Backend, Planned Architecture, Status

### Community 106 - "Community 106"
Cohesion: 0.50
Nodes (3): ARBiometricOverlayView, CameraController, OccasionCategory

### Community 107 - "Community 107"
Cohesion: 0.83
Nodes (3): Codable, GeminiLookEvaluation, GeminiTweak

### Community 109 - "Community 109"
Cohesion: 0.67
Nodes (3): main(), merge_category, merge_shards.py -- combines per-shard, per-category label CSVs (labels_<Category

### Community 110 - "Community 110"
Cohesion: 0.67
Nodes (3): main(), merge_category(), merge_shards.py -- combines per-shard, per-category label CSVs…

### Community 111 - "Community 111"
Cohesion: 0.50
Nodes (4): get_label_schema(), Column order for this category's CSV: filename/category/tier first, then every…, schema_columns(), _slot_classes()

### Community 112 - "Community 112"
Cohesion: 0.50
Nodes (3): 1. Mandatory Build Verification, 2. Automatic Build & Deployment, Workspace Rules: Build Verification & Deployment

### Community 113 - "Community 113"
Cohesion: 0.67
Nodes (3): Graphify Knowledge Graph Pipeline, Claude Graphify Skill Invocation, Graphify Codebase Knowledge Graph Rules

## Ambiguous Edges - Review These
- `generate` → `ALL_CATEGORIES`  [AMBIGUOUS]
  ML/vision/dataset_synthetic/qwen_pipeline.py · relation: shares_data_with
- `Effort-vs-Genetics Product Philosophy` → `Body-Proportion Clause Fix`  [AMBIGUOUS]
  AGENTS.md · relation: rationale_for

## Knowledge Gaps
- **625 isolated node(s):** `PreToolUse`, `sweetpad.build.xcodeWorkspacePath`, `install_qwen.sh script`, `architectures`, `attention_bias` (+620 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 873 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **26 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `generate` and `ALL_CATEGORIES`?**
  _Edge tagged AMBIGUOUS (relation: shares_data_with) - confidence is low._
- **What is the exact relationship between `Effort-vs-Genetics Product Philosophy` and `Body-Proportion Clause Fix`?**
  _Edge tagged AMBIGUOUS (relation: rationale_for) - confidence is low._
- **Why does `ThreadPoolExecutor` connect `Community 23` to `Community 1`, `Community 35`, `Community 22`, `Community 24`, `Community 25`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._
- **Why does `generate` connect `Community 83` to `Community 1`, `Community 58`, `Community 59`, `Community 62`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._
- **Why does `classify_with_mlx_vlm()` connect `Community 1` to `Community 32`, `Community 83`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `CameraController` (e.g. with `.body` and `CustomCameraView`) actually correct?**
  _`CameraController` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `LookSession` (e.g. with `.analyzeAndAddLook()` and `.selectedLook`) actually correct?**
  _`LookSession` has 2 INFERRED edges - model-reasoned connections that need verification._