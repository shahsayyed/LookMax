# Agent Rules for LookMax

## 1. iOS Application Development Rules

1. **Verify iOS App Builds When App Code Changes**:
   Whenever changes are made to the iOS application codebase (under `iOS/`), run a build test using `xcodebuild` to ensure there are no build errors. Do not run `xcodebuild` for Python ML scripts or pipeline-only changes.
   ```bash
   xcodebuild build -project iOS/LookMax.xcodeproj -scheme LookMax -destination 'generic/platform=iOS Simulator'
   ```

2. **Automatic Re-deployment**:
   Automatically trigger the iOS build when iOS code is modified so that the newest binaries are prepared for testing on the simulator or target device.

---

## 2. Core Machine Learning & Modeling Rules

3. **Effort-vs-Genetics Product Philosophy (Non-Negotiable)**:
   > **We rate effort and execution, never genetics or unchangeable features.**
   LookMax scores styling and grooming execution on a 1–10 scale.
   - Allowed signals: hairstyle, grooming lineup, outfit fit, fabric wrinkles/crispness, color harmony, accessory alignment, occasion appropriateness.
   - Strictly forbidden signals: body weight/size, face shape/symmetry, skin conditions (e.g. acne), age.
   Any synthetic dataset prompt, classification rubric, or generated advice violating this principle is a critical bug.

4. **Dual-Model On-Device Architecture**:
   - **Vision Model Pipeline (`ML/vision/`)**: 4 consolidated categories (`Men_Grooming`, `Women_Grooming`, `Men_Outfit`, `Women_Outfit`). Multi-head architecture (SmoothL1 regression for `score` + cross-entropy for attribute heads). Phase A pretraining on synthetic Qwen data (`pretrain_synthetic.py`), Phase B fine-tuning on real data with synthetic replay (`finetune_real_world.py`). Output: `LookMax_<Category>.mlpackage` (iOS 17+).
   - **Stylist LLM Pipeline (`ML/stylist_llm/`)**: Pruned `SmolLM2-135M-Instruct`, exported **stateless** (no KV-cache — a stateful design was attempted and abandoned after confirmed upstream PyTorch/coremltools bugs) at **FP16** (`StylistEngine.mlpackage`, 207MB, iOS 18+ ANE — INT4 was tried and caused verified quality regressions). Turns vision tags + user occasion into single-shot <50-word 5-minute fixes. Pipeline has been run end-to-end once — see `ML/stylist_llm/PLAN.md`'s "Current Status" section for the full account and open issues.
   - **Pipeline Isolation**: The two pipelines are strictly isolated in code and training checkpoints. The ONLY shared link is `tag_vocabulary.py` reading `taxonomy.py` to maintain tag format consistency.

5. **Safe Python & GPU Verification**:
   - Always run verification checks (`--dry-run`, `--self-check`, or `--coverage-only`) before initiating GPU-heavy jobs or paid API calls (Gemini).
   - Never commit large datasets (`ML/data/`), model checkpoints (`.pt`, `.bin`), or browser profile caches (`.cache/`).
