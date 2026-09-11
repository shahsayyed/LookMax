#!/usr/bin/env python3
"""
inspect_phaseA.py -- deeper diagnostic pass on trained Phase A checkpoints,
beyond the aggregate per-head accuracy already in *_phaseA_metrics.json.

Note on methodology: pretrain_synthetic.py's train/val split uses
torch.utils.data.random_split() with NO fixed generator/seed, so the exact
held-out validation set from training cannot be reconstructed after the
fact. This script draws a FRESH random sample instead (seeded, so it's at
least reproducible run-to-run) -- meaning results here include some images
the model saw during training, so absolute accuracy numbers will look
slightly optimistic versus the real held-out val metrics already recorded.
The diagnostic value is elsewhere: confusion patterns (which classes get
mixed up with which), whether any class has collapsed to near-zero
predictions, and whether the score head's output actually separates the
four quality tiers it's supposed to track -- these patterns are real
regardless of train/val leakage.

Usage:
  python3 inspect_phaseA.py --category Men_Grooming --sample-size 400
  python3 inspect_phaseA.py --category all --sample-size 300
"""
import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SYNTHETIC_QA_DIR, SYNTHETIC_RAW_DIR, MODELS_DIR, CATEGORIES, get_image_size

import torch
from torch.utils.data import DataLoader

from multihead_common import (
    GREEN, YELLOW, RED, CYAN, BOLD, RESET, header, get_device, get_transforms,
    MultiHeadModel, SyntheticCsvDataset, discover_synthetic_source, trainable_fields,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dataset_synthetic"))
import taxonomy as tx


def inspect_category(category, sample_size, device):
    header(f"Inspecting Phase A: {category}")
    ckpt_path = MODELS_DIR / f"LookMax_{category}_phaseA.pt"
    if not ckpt_path.exists():
        print(f"  {RED}No checkpoint found at {ckpt_path}{RESET}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    schema = ckpt["schema"]
    trainable = trainable_fields(schema)

    src = discover_synthetic_source(category, SYNTHETIC_QA_DIR, SYNTHETIC_RAW_DIR)
    rows = src["rows"]
    rng = random.Random(42)
    sample_rows = rng.sample(rows, min(sample_size, len(rows)))
    print(f"  Sampling {len(sample_rows)} of {len(rows)} qa_pass rows "
          f"(fresh random sample -- NOT the original held-out val split, see module docstring)")

    img_size = get_image_size(category)
    ds = SyntheticCsvDataset(sample_rows, src["images_dir"], schema, get_transforms(img_size, is_train=False))
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    model = MultiHeadModel(ckpt["backbone"], schema, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    # Collect predictions + targets per head, plus tier (for score calibration)
    preds = defaultdict(list)
    targets_all = defaultdict(list)
    score_preds, score_tiers = [], []
    row_idx = 0
    with torch.no_grad():
        for images, targets, masks in loader:
            images = images.to(device)
            outputs = model(images)
            bs = images.size(0)
            for f in trainable:
                name = f["name"]
                out = outputs[name].cpu()
                if f["type"] == "regression":
                    score_preds.extend(out.squeeze(-1).tolist())
                    for i in range(bs):
                        score_tiers.append(sample_rows[row_idx + i].get("tier", ""))
                else:
                    pred_idx = out.argmax(dim=-1).tolist()
                    preds[name].extend(pred_idx)
                    targets_all[name].extend(targets[name].tolist())
            row_idx += bs

    # ── Score calibration by tier ────────────────────────────────────────
    if score_preds:
        by_tier = defaultdict(list)
        for pred, tier in zip(score_preds, score_tiers):
            by_tier[tier].append(pred)
        print(f"\n  {CYAN}Score head -- mean predicted score by true tier "
              f"(should increase monotonically, roughly tracking SCORE_BANDS):{RESET}")
        for tier in ["flaw_severe", "flaw_mild", "average", "polished"]:
            vals = by_tier.get(tier, [])
            if vals:
                lo, hi = tx.SCORE_BANDS[tier]
                mean_pred = sum(vals) / len(vals)
                print(f"    {tier:14s} (true band {lo}-{hi}): predicted mean={mean_pred:5.2f}  n={len(vals)}")

    # ── Per-head confusion summary ───────────────────────────────────────
    print(f"\n  {CYAN}Per-head diagnostics (categorical/ordinal heads):{RESET}")
    for f in trainable:
        name = f["name"]
        if f["type"] == "regression":
            continue
        p = preds[name]
        t = targets_all[name]
        if not p:
            continue
        n_classes = f["levels"] if f["type"] == "ordinal" else len(f["classes"])
        class_names = [str(i) for i in range(n_classes)] if f["type"] == "ordinal" else f["classes"]

        correct = sum(1 for pi, ti in zip(p, t) if pi == ti)
        acc = correct / len(p)

        # Per-class recall + whether a class collapsed (predicted 0 times but present in targets)
        pred_counts = Counter(p)
        target_counts = Counter(t)
        per_class_correct = Counter(pi for pi, ti in zip(p, t) if pi == ti)
        collapsed = []
        weak_classes = []
        for cls_idx, cls_name in enumerate(class_names):
            n_true = target_counts.get(cls_idx, 0)
            if n_true == 0:
                continue
            recall = per_class_correct.get(cls_idx, 0) / n_true
            if pred_counts.get(cls_idx, 0) == 0:
                collapsed.append(cls_name)
            elif recall < 0.5:
                weak_classes.append((cls_name, recall, n_true))

        flag = ""
        if collapsed:
            flag = f"  {RED}⚠ NEVER PREDICTED: {collapsed}{RESET}"
        elif weak_classes:
            worst = sorted(weak_classes, key=lambda x: x[1])[0]
            flag = f"  {YELLOW}⚠ weak class: {worst[0]} (recall={worst[1]:.0%}, n={worst[2]}){RESET}"

        print(f"    {name:24s} acc={acc:5.1%}  n={len(p):4d}{flag}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", default="all", choices=CATEGORIES + ["all"])
    parser.add_argument("--sample-size", type=int, default=400)
    args = parser.parse_args()

    device = get_device()
    targets = CATEGORIES if args.category == "all" else [args.category]
    for category in targets:
        inspect_category(category, args.sample_size, device)


if __name__ == "__main__":
    main()
