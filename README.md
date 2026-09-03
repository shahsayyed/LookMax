# LookMax

> **Personal AI Stylist & Grooming Coach — 100% On-Device, Privacy-First.**

LookMax is an AI-powered personal styling and grooming platform that evaluates outfit execution and grooming effort on a 1–10 scale, returning actionable, high-leverage "5-minute fixes".

---

## Core Philosophy

> **We rate effort and execution, not genetics.**

LookMax specifically measures elements within the user's immediate control:
* **Style & Fit**: Garment silhouette, proportion, tailoring, fabric crispness vs. wrinkles, color coordination, occasion match.
* **Grooming**: Hairstyle neatness, flyaways, facial hair lineup, neatness.
* **Posture & Presence**: Real-time spine alignment, shoulder posture, jawline angle.

LookMax **never** penalizes or provides advice regarding unchangeable genetic features, body weight/size, facial symmetry, skin conditions (e.g., acne), or age.

---

## System Architecture

```
LookMax/
├── iOS/                              ← Native iOS Application (SwiftUI + AVFoundation + Vision + CoreML)
│   ├── LookMax/                      ← App source code (Views, Models, Managers, Services)
│   └── LookMax.xcodeproj             ← Xcode project configuration
├── ML/                               ← Machine Learning Pipelines
│   ├── vision/                       ← Vision Model Pipeline (Effort scoring & attribute tagging)
│   │   ├── dataset_synthetic/        ← Qwen-Image-2512 procedural synthetic dataset generator
│   │   ├── dataset_real/             ← Web scrapers (Unsplash, Reddit) & VLM auto-classifier
│   │   └── training/                 ← Multi-head PyTorch training (Phase A pretrain + Phase B fine-tune)
│   ├── stylist_llm/                  ← On-Device Stylist LLM (SmolLM2-135M stateful CoreML INT4)
│   ├── data/                         ← Separated datasets (vision_real/, vision_synthetic/, stylist_llm/)
│   └── models/                       ← Final exported CoreML .mlpackage artifacts for Xcode
├── Backend/                          ← Placeholder for future cloud sync & API services
├── Android/                          ← Placeholder for future Android application
├── AGENTS.md                         ← Guidelines and operational rules for AI coding assistants
└── README.md                         ← Project documentation root
```

---

## Dual On-Device CoreML Pipeline

LookMax deploys two specialized, isolated AI models directly to Apple devices:

1. **Vision Model (`LookMax_<Category>.mlpackage`, iOS 17+)**:
   * Evaluates photos across 4 categories: `Men_Grooming`, `Women_Grooming`, `Men_Outfit`, `Women_Outfit`.
   * Multi-head architecture predicts a continuous 1–10 effort score plus discrete attribute tags (e.g. `fabric_wrinkled: 2`, `formality: smart_casual`).
   * Pretrained on procedurally generated synthetic images, then fine-tuned on real-world photos with synthetic replay.

2. **Stylist LLM (`StylistEngine.mlpackage`, iOS 18+ ANE)**:
   * Vocabulary-pruned, fine-tuned `SmolLM2-135M-Instruct` running INT4 quantization on Apple Neural Engine.
   * Consumes vision attribute tags and the user's selected occasion (e.g., Business Meeting, Date Night) to generate a single-shot, <50-word actionable fix in milliseconds without internet access.

---

## Getting Started

### iOS App
Open `iOS/LookMax.xcodeproj` in Xcode 16+.
Select your simulator or connected device and press **Cmd + R** to run.

To verify builds via command line:
```bash
xcodebuild build -project iOS/LookMax.xcodeproj -scheme LookMax -destination 'generic/platform=iOS Simulator'
```

### ML Pipelines
See [ML/README.md](ML/README.md) for complete instructions on dataset generation, training, and CoreML export.
