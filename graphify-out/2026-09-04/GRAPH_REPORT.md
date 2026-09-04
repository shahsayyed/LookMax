# Graph Report - LookMax  (2026-09-03)

## Corpus Check
- 106 files · ~109,986 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 908 nodes · 1547 edges · 61 communities (40 shown, 10 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 114 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- SwiftUI UI Components and Theme
- Real-World Fine-Tuning Pipeline
- iOS Camera and AVFoundation
- VLM Image Classification and Heuristics
- Cloud Gemini Vision Integration
- iOS Frameworks and System Imports
- LookMax Styling Analysis Engine
- Core Architecture and Product Principles
- Evaluation Data Models and Codable Schemas
- Stylist LLM Vocabulary Pruning
- Image Web Scraping Engine
- Stylist LLM Supervised Fine-Tuning
- Session Storage and Data Persistence
- Reddit Image URL Parser
- Synthetic Dataset Batch Runner
- Haptic Feedback and UI Theme
- Qwen Sampling Taxonomy Schema
- User Profile and Onboarding Flow
- Qwen Pipeline Loader and Utilities
- Reddit Playwright Scraper Engine
- Browser Context and Image Downloader
- Unsplash Dataset Ingestion Engine
- Heuristic Image Metrics and Color Harmony
- Image Picker and UIKit Coordinator
- CelebA-HQ Dataset Ingestion Engine
- Session Detail and Look Ingestion
- Synthetic Dataset Generation with Gemini
- Synthetic Prompt Builder and Taxonomy Clauses
- Environment Setup and Dependency Verification
- Grooming and Outfit Condition Builders
- Scraper Rate Limiter and Request Tracker
- Synthetic Variation Testing Suite
- LookMax Main Content and Tab Navigation
- Stylist Advice QA Review Gate
- Synthetic Pipeline Smoke Testing
- CoreML Model Export and State Wrapper
- Session Creation UI Flow
- Face Biometric Signature Matching
- Qwen Hardware Sanity Check
- Build Verification Rules
- Graphify Knowledge Graph Integration
- Media Embed Filter Heuristics
- FLUX Dataset Generator Archive
- Qwen Environment Installation Script
- Stylist LLM Configuration
- Project Setup Script
- Centralized Vision Pipeline Configuration
- Reddit Scraper Unit Tests
- Synthetic Ingestion Shell Setup
- Safe Python and GPU Verification

## God Nodes (most connected - your core abstractions)
1. `CameraController` - 38 edges
2. `LookSession` - 32 edges
3. `LookItem` - 29 edges
4. `OccasionCategory` - 23 edges
5. `BeforeAfterComparisonView` - 21 edges
6. `CodingKeys` - 20 edges
7. `Theme` - 19 edges
8. `CodingKeys` - 19 edges
9. `CustomCameraView` - 18 edges
10. `TestRedditScraper` - 17 edges

## Surprising Connections (you probably didn't know these)
- `Mandatory Build Verification Rule` --semantically_similar_to--> `iOS Xcodebuild Verification Rule`  [INFERRED] [semantically similar]
  .agents/rules/build_and_deploy.md → AGENTS.md
- `Effort-Based Styling Evaluation Rubric` --semantically_similar_to--> `Effort vs Genetics Product Philosophy`  [INFERRED] [semantically similar]
  README.md → AGENTS.md
- `Dual-Model On-Device Architecture` --semantically_similar_to--> `Dual On-Device CoreML Pipeline Specification`  [INFERRED] [semantically similar]
  AGENTS.md → README.md
- `.body` --calls--> `ContentView`  [INFERRED]
  iOS/LookMax/LookMaxApp.swift → iOS/LookMax/ContentView.swift
- `.averageScore` --references--> `LookItem`  [INFERRED]
  iOS/LookMax/Models/LookSession.swift → iOS/LookMax/Models/LookItem.swift

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Effort-vs-Genetics Alignment Architecture** — agents_effort_vs_genetics_philosophy, readme_effort_based_scoring_rubric, ml_vision_dataset_synthetic_plan_clause_structured_prompts, ml_stylist_llm_plan_smollm2_135m_pipeline [EXTRACTED 1.00]
- **Dual-Model On-Device Execution Pipeline** — readme_dual_on_device_pipeline, ml_readme_vision_model_pipeline, ml_readme_stylist_llm_pipeline, ml_readme_tag_vocabulary_contract [EXTRACTED 1.00]
- **Synthetic Dataset Evolution from FLUX to Qwen** — ml_archive_dataset_generator_v7_plan_flux_dataset_pipeline, ml_archive_dataset_generator_v7_plan_clip_truncation_vulnerability, ml_vision_dataset_synthetic_plan_qwen_image_pipeline [EXTRACTED 1.00]

## Communities (61 total, 10 thin omitted)

### Community 0 - "SwiftUI UI Components and Theme"
Cohesion: 0.05
Nodes (52): Content, Identifiable, Color, String, GlassCardModifier, NeonGlowModifier, CGFloat, Double (+44 more)

### Community 1 - "Real-World Fine-Tuning Pipeline"
Cohesion: 0.06
Nodes (42): Dataset, finetune_category(), main(), LookMax ML Pipeline — Phase 5 (Phase B: Real-World Fine-Tune)…, --checkpoint may be a single .pt file (only valid for one category) or a…, resolve_checkpoint(), build_backbone(), compute_losses() (+34 more)

### Community 2 - "iOS Camera and AVFoundation"
Cohesion: 0.06
Nodes (37): AVCaptureConnection, AVCaptureDevice, AVCaptureDeviceInput, AVCaptureOutput, AVCapturePhoto, AVCapturePhotoCaptureDelegate, AVCapturePhotoOutput, AVCaptureVideoDataOutputSampleBufferDelegate (+29 more)

### Community 3 - "VLM Image Classification and Heuristics"
Cohesion: 0.06
Nodes (46): append_annotation_record(), classify_with_gemini(), classify_with_mlx_vlm(), classify_with_ollama(), _get_gemini_client(), heuristic_filter(), image_to_base64(), load_processed_files() (+38 more)

### Community 4 - "Cloud Gemini Vision Integration"
Cohesion: 0.07
Nodes (37): CGSize, Codable, Error, CodingKeys, category, effortTime, fitNote, formalityScore (+29 more)

### Community 5 - "iOS Frameworks and System Imports"
Cohesion: 0.06
Nodes (23): AnyClass, App, AVCaptureVideoPreviewLayer, AVFoundation, Combine, CoreImage, Foundation, ImageIO (+15 more)

### Community 6 - "LookMax Styling Analysis Engine"
Cohesion: 0.08
Nodes (27): CaseIterable, CGImage, LookAnalysisEngine, OccasionProfile, CGRect, Double, String, LookAnalysisResult (+19 more)

### Community 7 - "Core Architecture and Product Principles"
Cohesion: 0.08
Nodes (30): Dual-Model On-Device Architecture, Effort vs Genetics Product Philosophy, Vision and Stylist Pipeline Isolation Contract, LookMax Android Application, LookMax Backend API Services, CLIP 77-Token Truncation Vulnerability, FLUX.1 [dev] Synthetic Dataset Generator (v7 Archive), Qwen 48-Prompt Demographic and Level Benchmark Suite (+22 more)

### Community 8 - "Evaluation Data Models and Codable Schemas"
Cohesion: 0.08
Nodes (26): CodingKey, CodingKeys, badPoints, detectedFaceShape, detectedOutfitColor, fitNote, fitScore, goodPoints (+18 more)

### Community 9 - "Stylist LLM Vocabulary Pruning"
Cohesion: 0.12
Nodes (23): dataset_token_ids(), floor_token_ids(), _load_tokenizer(), main(), prune_vocabulary.py -- shrinks the base model's embedding/lm_head matrices to…, Token ids needed to losslessly spell every real class name/term., Token ids used across every QA-passed example's full ChatML text (system + user…, run_dry_run() (+15 more)

### Community 10 - "Image Web Scraping Engine"
Cohesion: 0.17
Nodes (24): download_image(), load_seen(), log_seen(), main(), Path, rate_limit(), LookMax ML Pipeline — Phase 2 (v3) =====================================…, scrape_pexels() (+16 more)

### Community 11 - "Stylist LLM Supervised Fine-Tuning"
Cohesion: 0.15
Nodes (16): _build_dataset(), _load_examples(), main(), finetune.py -- supervised fine-tuning on the QA-passed synthetic dataset. FULL…, Returns a list of {"input_ids": [...], "labels": [...]} -- labels match…, run_dry_run(), run_finetune(), remap_tokenizer.py -- wraps the ORIGINAL (unpruned) tokenizer plus… (+8 more)

### Community 12 - "Session Storage and Data Persistence"
Cohesion: 0.15
Nodes (14): Binding, .body, SessionStorageManager, .documentsDir, LookSession, .averageScore, .bestLook, .firstLook (+6 more)

### Community 13 - "Reddit Image URL Parser"
Cohesion: 0.13
Nodes (11): build_reddit_json_url(), clean_and_unescape_url(), extract_image_urls(), is_deleted_or_removed(), is_direct_image_url(), Construct a valid Reddit .json endpoint URL supporting feeds and search…, Unescapes HTML entities (&amp; -> &) and strips whitespace., Check if the URL path ends with a recognized image extension. (+3 more)

### Community 14 - "Synthetic Dataset Batch Runner"
Cohesion: 0.14
Nodes (20): already_done_indices(), build_full_task_list(), check_disk_space(), group_by_resolution(), labels_csv_path(), main(), output_paths(), full_run.py -- the 28,000-image production run (Qwen-Image-2512). Only ever… (+12 more)

### Community 15 - "Haptic Feedback and UI Theme"
Cohesion: 0.12
Nodes (18): HapticManager, Theme, .hudPrompt, .body, .interactiveCurtain, EmptySessionsView, .body, ProfileHeaderBanner (+10 more)

### Community 16 - "Qwen Sampling Taxonomy Schema"
Cohesion: 0.12
Nodes (15): _formality_bucket_pick(), formality_tier(), get_label_schema(), Shared prompt taxonomy for Qwen-Image-2512 synthetic dataset generation. SINGLE…, Coherent sampling for one slot: 75% of the time, restrict to entries within…, Independently samples the garment-composition axis: which item in which slot,…, Thresholds, NOT round() -- see sample_outfit()'s docstring., Same score-band balance as the archived taxonomy: roughly equal thirds across… (+7 more)

### Community 17 - "User Profile and Onboarding Flow"
Cohesion: 0.15
Nodes (14): Data, Date, String, UserProfile, ProfileOnboardingView, .body, .canSave, Bool (+6 more)

### Community 18 - "Qwen Pipeline Loader and Utilities"
Cohesion: 0.15
Nodes (14): build_tasks(), make_identity(), chunked(), load_qwen_pipeline(), Shared GPU-aware pipeline loader for the Qwen-Image-2512 scripts. Qwen-…, Returns (pipe, can_batch). can_batch is True only when the full pipeline is…, Split seq into consecutive chunks of at most `size` items each -- the last…, build_grooming_prompt() (+6 more)

### Community 19 - "Reddit Playwright Scraper Engine"
Cohesion: 0.17
Nodes (19): create_persistent_context(), _ensure_playwright(), get_system_chrome_profiles(), is_session_valid(), login_and_save_session(), main(), print_available_chrome_profiles(), print_scrape_summary() (+11 more)

### Community 20 - "Browser Context and Image Downloader"
Cohesion: 0.15
Nodes (18): BrowserContext, _clean_stale_profile_locks(), _download_single_image(), fetch_json(), normalize_categories(), Any, Path, Scrape image URLs across multiple categories with rate-limiting controls,… (+10 more)

### Community 21 - "Unsplash Dataset Ingestion Engine"
Cohesion: 0.19
Nodes (16): classify_and_score_photo(), download_image_task(), find_dataset_dir(), load_dataset_metadata(), main(), Any, DataFrame, Path (+8 more)

### Community 22 - "Heuristic Image Metrics and Color Harmony"
Cohesion: 0.21
Nodes (16): color_harmony(), _crop_fraction(), _lazy_imports(), _load_face_cascade(), main(), measured_dominant_color(), _palette_lab(), process() (+8 more)

### Community 23 - "Image Picker and UIKit Coordinator"
Cohesion: 0.18
Nodes (10): Coordinator, ImagePicker, Any, Context, UIImage, NSObject, UIImagePickerController, UIImagePickerControllerDelegate (+2 more)

### Community 24 - "CelebA-HQ Dataset Ingestion Engine"
Cohesion: 0.22
Nodes (15): classify_description(), load_celebahq_annotations(), main(), Any, DataFrame, Path, LookMax ML Pipeline — CelebA-HQ ($1024\\times1024$) Facial Ingestion Engine…, Evaluates a CelebA-HQ natural language annotation and routes into LookMax… (+7 more)

### Community 25 - "Session Detail and Look Ingestion"
Cohesion: 0.15
Nodes (11): CGImagePropertyOrientation, UIImage, String, UIImage, SessionDetailView, .body, .selectedLook, String (+3 more)

### Community 26 - "Synthetic Dataset Generation with Gemini"
Cohesion: 0.21
Nodes (12): build_task_contexts(), _call_gemini(), _gemini_client(), main(), _PacedRateLimiter, generate_synthetic_dataset.py -- generates the 3,500-5,000 instruction pairs…, Minimal local rate limiter -- deliberately not importing real_data_pipeline's…, Deterministic: `count` (category, tier, occasion, row) contexts, in a fixed… (+4 more)

### Community 27 - "Synthetic Prompt Builder and Taxonomy Clauses"
Cohesion: 0.22
Nodes (13): build_grooming_task(), build_outfit_task(), build_task(), _garment_clause(), _plain_garment_clause(), prompt_builder.py -- composes ONE (prompt, resolution, label-row) triple per…, Returns {"prompt": str, "resolution": (w, h), "row": {field_name: value, ...}}…, Assemble the full CSV row dict (filename/category/tier + every schema column in… (+5 more)

### Community 28 - "Environment Setup and Dependency Verification"
Cohesion: 0.35
Nodes (12): check_dependencies(), check_system(), create_directory_structure(), err(), header(), ok(), print_summary(), LookMax ML Pipeline — Phase 1 ==============================… (+4 more)

### Community 29 - "Grooming and Outfit Condition Builders"
Cohesion: 0.24
Nodes (12): build_eyebrows(), build_facial_hair(), build_hair(), build_makeup(), build_outfit_condition(), build_skin(), positive_for_tier(), Returns (mods dict of text fragments, labels dict) for one outfit image's… (+4 more)

### Community 30 - "Scraper Rate Limiter and Request Tracker"
Cohesion: 0.25
Nodes (4): RateLimitTracker, Tracks page requests and enforces human browsing pauses., Call after each request. Executes randomized delay and batch cooldowns., Call after finishing a category before moving to the next.

### Community 31 - "Synthetic Variation Testing Suite"
Cohesion: 0.39
Nodes (7): build_variation_tasks(), check_disk_space(), main(), variation_test.py -- broader diversity/quality check than smoke_test.py's…, Deterministic: samples_per_cell tasks per (category, tier), in a fixed order,…, run_dry_run(), run_generation()

### Community 32 - "LookMax Main Content and Tab Navigation"
Cohesion: 0.38
Nodes (4): ContentView, .bottomTabBar, Int, String

### Community 33 - "Stylist Advice QA Review Gate"
Cohesion: 0.48
Nodes (6): check_record(), main(), qa_review.py -- QA gate on generated advice text before fine-tuning on it.…, Returns (passed: bool, reasons: list[str]) -- reasons is non-empty only on…, run_review(), run_self_check()

### Community 34 - "Synthetic Pipeline Smoke Testing"
Cohesion: 0.48
Nodes (6): iter_smoke_tasks(), main(), smoke_test.py -- cheap sanity checks before spending any GPU time. --dry-run…, Deterministic: one task per (category, tier), in a fixed order., run_dry_run(), run_per_tier()

### Community 35 - "CoreML Model Export and State Wrapper"
Cohesion: 0.53
Nodes (5): _build_stateful_wrapper(), main(), export_coreml.py -- converts the fine-tuned, pruned model to a CoreML…, run_dry_run(), run_export()

### Community 37 - "Face Biometric Signature Matching"
Cohesion: 0.67
Nodes (3): Equatable, FaceBiometricSignature, Double

### Community 38 - "Qwen Hardware Sanity Check"
Cohesion: 0.67
Nodes (3): check_disk_space(), main(), quick_prompt_test.py -- fastest possible "does Qwen even work on this box"…

### Community 39 - "Build Verification Rules"
Cohesion: 0.67
Nodes (3): iOS Xcodebuild Verification Rule, Automatic Build and Deployment Rule, Mandatory Build Verification Rule

### Community 40 - "Graphify Knowledge Graph Integration"
Cohesion: 0.67
Nodes (3): Graphify Knowledge Graph Pipeline, Claude Graphify Skill Invocation, Graphify Codebase Knowledge Graph Rules

## Knowledge Gaps
- **86 isolated node(s):** `install_qwen.sh script`, `install.sh script`, `install.sh script`, `ImageIO`, `.isUltraWideAvailable` (+81 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 355 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `classify_with_mlx_vlm()` connect `VLM Image Classification and Heuristics` to `Stylist LLM Supervised Fine-Tuning`?**
  _High betweenness centrality (0.103) - this node is a cross-community bridge._
- **Why does `main()` connect `VLM Image Classification and Heuristics` to `Image Web Scraping Engine`?**
  _High betweenness centrality (0.091) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `CameraController` (e.g. with `.body` and `CustomCameraView`) actually correct?**
  _`CameraController` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `LookSession` (e.g. with `.analyzeAndAddLook()` and `.selectedLook`) actually correct?**
  _`LookSession` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `LookItem` (e.g. with `.averageScore` and `.body`) actually correct?**
  _`LookItem` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `install_qwen.sh script`, `install.sh script`, `install.sh script` to the rest of the system?**
  _86 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `SwiftUI UI Components and Theme` be split into smaller, more focused modules?**
  _Cohesion score 0.051251956181533644 - nodes in this community are weakly interconnected._