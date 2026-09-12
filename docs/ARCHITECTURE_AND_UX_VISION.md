# LookMax — Current Architecture & UX Vision

*Prepared as a briefing document for design partners (e.g. Claude Design). Purpose: explain what LookMax does today, how it's built, and what "great" would look like if we rebuilt the experience around best-in-class UX.*

---

## 1. What LookMax Is

LookMax is a personal styling and grooming coach. A user photographs themselves, picks an occasion (business meeting, date night, formal event, casual, custom), and the app scores their outfit + grooming + posture execution on a 1–10 scale, then hands back a short list of **high-leverage, ~5-minute-or-less fixes** to raise that score.

**Core philosophy — effort and execution, not genetics.** The product deliberately only scores things a person can act on in the next five minutes: fit, grooming neatness, posture, color coordination, occasion-appropriateness. It explicitly never scores or comments on facial symmetry, body weight/size, skin conditions, or age. This is a trust and safety boundary as much as a product boundary — it's what keeps the tool feeling like a coach rather than a judge.

The pitch to the user is a loop: **photograph → get scored → get 2–3 fast fixes → try again → see the score go up → keep a log of that improvement over time.**

---

## 2. Current System Architecture

```
LookMax/
├── iOS/            Native SwiftUI app (the only shipped surface today)
├── ML/             Model pipelines: vision scoring model + on-device stylist LLM
├── Backend/        Placeholder — no cloud sync/API yet
├── Android/        Placeholder — not started
```

Today LookMax is **iOS-only**, and the intelligence is a **hybrid** of on-device heuristics, an in-progress on-device CoreML model, and a cloud VLM call — not the fully on-device pipeline the README describes as the target state. That gap between "current" and "target" is itself an important thing to design around (see §5).

### 2.1 The analysis pipeline, end to end

1. **Capture** — `CameraController` + `CustomCameraView` drive AVFoundation capture, with `ARBiometricOverlayView` overlaying live face/pose guidance while framing the shot.
2. **On-device Vision pass** (`LookAnalysisEngine.analyze`, always runs, free, instant, no network):
   - Runs Apple's Vision framework: face detection, face capture quality, human body pose, image classification.
   - Applies an **occasion profile** — each occasion (formal event, business meeting, date night, casual, custom) has its own baseline score and "formality strictness weight," so the same outfit scores differently depending on what it's being judged against.
   - Heuristically derives: face shape, lighting/clarity score, posture alignment (neck-to-root vector), hair/beard presence, eyewear presence, outfit formality (formal vs. casual keyword classification), dominant garment color.
   - Produces a `LookAnalysisResult`: overall score, a "potential score" (score + sum of top fix impacts), sub-scores (posture/fit/grooming/lighting), good points, bad points, and a ranked list of `StyleSuggestion`s (each with an icon, category, recommendation text, and estimated effort time like "30 sec" or "2 mins").
3. **Cloud VLM pass** (`GeminiVisionService`, optional, needs network + API key) — sends the compressed photo to Gemini for a richer structured evaluation (formality/sharpness/occasion-match sub-scores, good/bad points, tweaks with effort time).
4. **Merge** (`LookAnalysisEngine.merge`) — Gemini's result is treated as the primary source of truth when available; on-device Vision output fills gaps and contributes any suggestion categories Gemini didn't cover. This gives graceful degradation: the app still works fully offline, just with a coarser, more heuristic score.
5. **Present** — the merged result renders as a dashboard: overall score, sub-score bars, good/bad points, and a ranked "5-minute fixes" list with effort estimates, via `DashboardComponents`, `ScoreComparisonBar`, `LookDetailCard`.

### 2.2 The two purpose-built ML models (in the `ML/` pipeline, target state)

1. **Vision Model** (`LookMax_<Category>.mlpackage`, iOS 17+) — a multi-head model, one variant each for `Men_Grooming`, `Women_Grooming`, `Men_Outfit`, `Women_Outfit`. Predicts a continuous 1–10 effort score plus discrete attribute tags (e.g. `fabric_wrinkled: 2`, `formality: smart_casual`). Trained in two phases: pretrained on a large procedurally-generated synthetic image dataset (built with Qwen-Image-2512), then fine-tuned on real scraped/labeled photos with synthetic replay to avoid catastrophic forgetting. This is meant to **replace** today's Vision-framework heuristics with a model actually trained for this exact scoring task.
2. **Stylist LLM** (`StylistEngine.mlpackage`, iOS 18+, Apple Neural Engine) — a vocabulary-pruned, fine-tuned SmolLM2-135M-Instruct, exported stateless at FP16 (207MB; INT4 was tried and reverted after it caused truncation/repetition failures). It takes the vision model's attribute tags plus the user's selected occasion and generates a single-shot, under-50-word actionable fix — fully offline, no network call. This is meant to **replace** the Gemini cloud call with a private, on-device, zero-latency-to-server alternative. On-device latency on a real iPhone has not yet been measured.

Both models are trained/exported outside the app in the `ML/` pipeline and dropped into the iOS bundle as CoreML packages; the app itself doesn't do any training.

### 2.3 User profile & personalization

`UserProfile` is a lightweight local record: name, a list of reference photos (`photoDataList`), a list of `FaceBiometricSignature`s (a face embedding/landmark signature, presumably used to recognize "this is the same person" across sessions so history can be trusted as one continuous person's timeline), and a creation date. It's persisted via `UserDefaults` (JSON-encoded), created during a one-time `ProfileOnboardingView` flow. There is currently **no cloud account, no login, no cross-device sync** — everything lives in local app storage on one phone.

### 2.4 Logging progress over time

- A **`LookSession`** is a named, occasion-tagged container (e.g. "Client Dinner — Date Night") holding an ordered list of `LookItem`s (one per photo/attempt within that session), auto-generated descriptive tags, and computed properties like `averageScore`, `bestLook`, `latestLook`.
- Each **`LookItem`** stores its photo path, score, and (implicitly) the fixes suggested for it.
- `SessionStorageManager` (an `ObservableObject`, singleton) persists the list of sessions as JSON in `UserDefaults` and writes each photo as a compressed JPEG to the app's Documents directory, keyed by UUID filename. Deleting a session or a look cleans up its image files.
- `BeforeAfterComparisonView` and `ScoreComparisonBar` let the user visually compare two looks side by side (e.g. first attempt vs. latest attempt in the same session) to see the improvement.

So today's "log of progress" is: a flat, on-device, per-device history of sessions → looks → scores + fixes, with a manual before/after comparison view. There's no trend chart, no long-term score-over-time graph, no reminders, no streaks, no cross-session insight ("your posture score has improved 1.4 points over the last 5 business-meeting sessions").

---

## 3. What the App Solves for the User Today

- **Answers "how do I look, really?"** without a judgmental human — a private, always-available second opinion before a big meeting, date, or event.
- **Converts vague self-doubt into 2–3 concrete, timed actions** ("30 sec", "2 mins") instead of an overwhelming or vague critique.
- **Adapts standards to context** — the same shirt-and-jeans outfit is scored generously for "casual everyday" and penalized for "formal event," which matches how humans actually judge appearance.
- **Builds a private, judgment-free improvement trail** the user can look back on, without ever surfacing anything about their unchangeable physical traits.
- **Works offline** for the core loop (device-only Vision heuristics), with the cloud VLM as an enhancement layer, not a hard dependency.

---

## 4. Known Gaps / Honest Limitations (as of now)

These are worth naming explicitly to whoever is designing the "great" version, since they're the real product-risk surface:

- **README says "100% on-device, privacy-first"; the shipped app calls a cloud Gemini API for the primary score.** The purpose-built on-device CoreML models exist in the training pipeline but are not yet the thing running in the merge step as primary — the Vision-framework heuristic is the offline fallback, not the intended flagship model.
- **No accounts, no sync, no backup.** A lost or reset phone loses the entire style history.
- **No trends/insight layer.** Sessions are logged but never aggregated into "how am I improving" over weeks/months.
- **Onboarding stores raw reference photos + biometric signatures locally with no stated retention/deletion policy** surfaced to the user in-product — this is exactly the kind of thing that needs an unmistakably clear, reassuring privacy UX given the personal nature of the data (photos of your face/body used to judge your appearance).
- **Stylist LLM on-device latency is unmeasured** — the offline "instant fix" promise isn't yet verified on real hardware.
- **Android and Backend are placeholders** — today this is a single-device, single-platform product.

---

## 5. The UX Vision: What "Great" Should Feel Like

The ask for the redesign: take this same underlying capability (score my look, tell me what to fix, remember my progress) and wrap it in an experience so good people *want* to open it every day, not just before a big event — the kind of product polish and flow where the interaction itself feels rewarding, independent of the score you get.

Direction to design around:

1. **The loop should feel alive, not transactional.** Today: photo → number → list → done. The target feeling: a short, satisfying ritual — like checking a fitness ring or a habit streak — where the score reveal, the fix suggestions, and the "try again" moment all have their own small delight, not just a results screen.
2. **Progress should be the emotional payoff, not the raw score.** A 6.8 the first time you use the app should feel like a *baseline*, not a grade. The redesign should foreground the trendline — "your average business-meeting score is up 1.2 points this month" — over any single photo's number. This is where the current flat session log needs to become an actual longitudinal story of the user's improvement.
3. **Fixes need to feel achievable in the moment, not like homework.** The existing effort-time labels ("30 sec", "2 mins") are the right instinct — the redesign should lean harder into that: a fix should feel like a single tappable, checkable action inside the results screen, with an obvious way to immediately redo the photo and see the delta, in one continuous flow rather than separate screens.
4. **Trust and privacy need to be a visible, designed part of the experience, not a disclaimer.** Given the app explicitly promises to never comment on unchangeable traits, and given it stores face photos and biometric signatures, the UX should make the user *feel* that boundary — e.g. visibly show what is and isn't being evaluated, make on-device vs. cloud processing legible and controllable, and make data deletion trivially discoverable.
5. **Personalization should compound.** The user profile today is just a name and some photos. The vision: the app should feel like it *knows* the user's context — their go-to occasions, their recurring problem areas (posture keeps coming up? outfit color keeps getting flagged?), and should proactively surface the one thing most worth fixing next, rather than a generic checklist every time.
6. **Occasion selection should feel like styling for a real moment in someone's life**, not choosing from a dropdown of five enum cases — e.g. calendar-aware suggestions, richer custom occasions, and visual language that shifts with the occasion's mood (formal event vs. casual everyday should not just change a baseline number, it should change how the whole screen feels).
7. **The eventual multi-model architecture (on-device vision model + on-device stylist LLM) is a UX opportunity, not just an engineering one.** Once inference is fully local and fast, the product can promise and *show* instant, private, no-loading-spinner feedback — that speed and privacy story is a real differentiator worth designing the interface around (e.g. live-camera feedback before the shutter even clicks, rather than capture → wait → results).

**The brief for the design partner, in one line:** *Keep the substance — private, judgment-free, effort-focused coaching with fast fixes and a visible improvement trail — but rebuild the shell around a UX so smooth and rewarding that using it becomes a habit, not a occasional utility reached for only before big events.*
