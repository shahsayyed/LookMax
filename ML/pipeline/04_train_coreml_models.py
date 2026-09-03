"""
LookMax ML Pipeline — Phase 4 (Phase A: Synthetic Pretrain)
=============================================================
04_train_coreml_models.py

Trains ONE multi-head model per category (Men_Grooming, Women_Grooming,
Men_Outfit, Women_Outfit — see config.py's CATEGORIES) on the synthetic
Qwen-Image-2512 dataset produced by ML/pipeline/dataset_generator/. Heads
are built dynamically FROM each category's label_schema_<Category>.json
(taxonomy.get_label_schema()) — one SmoothL1 regression head for `score`
(loss weight 1.0, the dominant term) plus one small classification head
per categorical/ordinal schema entry (loss weight 0.3, or 0.5 for
`formality`), skipping `meta` entries entirely. See multihead_common.py
for the shared model/dataset/loss code.

This is Phase A only — it does NOT export to CoreML. It saves a plain
PyTorch checkpoint (.pt) per category; 05_finetune_real_world.py (Phase B)
loads that checkpoint, fine-tunes on real scraped-and-VLM-classified
photos mixed with synthetic replay, and does the CoreML export.

Data source: ML/data/4_Synthetic_Qwen/qa_processed/ (preferred — has a
qa_pass column from extract_measured_labels.py; trains only on
qa_pass==1), falling back to raw_generated/ with a printed warning if
qa_processed/ is empty or missing. Both are legitimately empty right now
(the 28,000-image generation run hasn't happened yet) — --dry-run and a
real run must both report that cleanly, not crash.

Usage:
    python3 ML/pipeline/04_train_coreml_models.py --dry-run
    python3 ML/pipeline/04_train_coreml_models.py --category Men_Grooming
    python3 ML/pipeline/04_train_coreml_models.py --epochs 5 --backbone mobilenet_v3_large
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    SYNTHETIC_QA_DIR, SYNTHETIC_RAW_DIR, MODELS_DIR, CATEGORIES,
    BACKBONE, IMAGE_SIZE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    TRAIN_SPLIT, EARLY_STOPPING_PATIENCE,
)

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, random_split
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from multihead_common import (
        GREEN, YELLOW, RED, CYAN, BOLD, RESET, header, get_device, get_transforms,
        MultiHeadModel, SyntheticCsvDataset, compute_losses, evaluate,
        discover_synthetic_source, trainable_fields,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Per-Category Training Orchestrator
# ──────────────────────────────────────────────────────────────────────────────
def train_category(category: str, device, args, dry_run: bool) -> dict:
    header(f"Phase A (synthetic pretrain): {category}")

    src = discover_synthetic_source(category, SYNTHETIC_QA_DIR, SYNTHETIC_RAW_DIR)
    if src["warning"]:
        print(f"  {YELLOW}⚠ {src['warning']}{RESET}")

    print(f"  Source     : {src['source'] or 'none'}")
    print(f"  Total rows : {src['total_rows']}")
    print(f"  Usable rows: {len(src['rows'])} (after qa_pass filtering, if applicable)")

    if not src["available"]:
        status = "dry_run_no_data" if dry_run else "no_data"
        if not dry_run:
            print(f"  {RED}✗ Skipping {category} — no synthetic training data available.{RESET}")
        else:
            print(f"  {YELLOW}[DRY-RUN] Nothing to train on yet for {category}.{RESET}")
        return {"category": category, "status": status, "total_rows": src["total_rows"], "usable_rows": 0}

    schema = src["schema"]
    trainable = trainable_fields(schema)
    print(f"  Heads      : score (regression, weight 1.0) + "
          f"{sum(1 for f in trainable if f['name'] != 'score')} attribute head(s)")

    if dry_run:
        print(f"  {YELLOW}[DRY-RUN] Would train {args.epochs} epoch(s) on {len(src['rows'])} images "
              f"(backbone={args.backbone}, image_size={args.image_size}, batch_size={args.batch_size}).{RESET}")
        return {"category": category, "status": "dry_run", "total_rows": src["total_rows"],
                "usable_rows": len(src["rows"])}

    # ── Build datasets ───────────────────────────────────────────────────
    rows = src["rows"]
    full_ds = SyntheticCsvDataset(rows, src["images_dir"], schema, get_transforms(args.image_size, is_train=True))
    n_train = max(1, int(len(full_ds) * TRAIN_SPLIT))
    n_val = len(full_ds) - n_train
    if n_val == 0:
        n_train -= 1
        n_val = 1
    train_ds, val_ds_idx = random_split(full_ds, [n_train, n_val])
    # Re-wrap val split with no-augmentation transform, same indices.
    val_full_ds = SyntheticCsvDataset(rows, src["images_dir"], schema, get_transforms(args.image_size, is_train=False))
    val_ds = torch.utils.data.Subset(val_full_ds, val_ds_idx.indices)

    print(f"  Train      : {n_train} | Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # ── Build model ──────────────────────────────────────────────────────
    model = MultiHeadModel(args.backbone, schema, pretrained=True).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    print(f"\n  {'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>10}  {'Score MAE':>10}  {'Score RMSE':>11}")
    print(f"  {'─'*6}  {'─'*11}  {'─'*10}  {'─'*10}  {'─'*11}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss, running_n = 0.0, 0
        for images, targets, masks in train_loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            masks = {k: v.to(device) for k, v in masks.items()}

            optimizer.zero_grad()
            outputs = model(images)
            loss, _ = compute_losses(outputs, targets, masks, schema)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_n += images.size(0)

        train_loss = running_loss / running_n
        val_metrics = evaluate(model, val_loader, schema, device)
        scheduler.step()

        elapsed = time.time() - t0
        marker = " ←best" if val_metrics["loss"] < best_val_loss else ""
        mae = val_metrics.get("score_mae")
        rmse = val_metrics.get("score_rmse")
        print(f"  {epoch:>6}  {train_loss:>11.4f}  {val_metrics['loss']:>10.4f}  "
              f"{(mae if mae is not None else float('nan')):>10.4f}  "
              f"{(rmse if rmse is not None else float('nan')):>11.4f}{marker}  ({elapsed:.1f}s)")

        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "score_mae": round(mae, 4) if mae is not None else None,
            "score_rmse": round(rmse, 4) if rmse is not None else None,
        })

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  {YELLOW}Early stopping at epoch {epoch} "
                      f"(no val improvement for {EARLY_STOPPING_PATIENCE} epochs){RESET}")
                break

    if best_state:
        model.load_state_dict(best_state)

    # ── Final validation summary (per-head accuracy) ────────────────────
    final_metrics = evaluate(model, val_loader, schema, device)
    print(f"\n  Final validation — loss={final_metrics['loss']:.4f}", end="")
    if final_metrics.get("score_mae") is not None:
        print(f", score_mae={final_metrics['score_mae']:.3f}, score_rmse={final_metrics['score_rmse']:.3f}")
    else:
        print()
    print(f"  Per-head accuracy (validation):")
    for name, acc in final_metrics["head_accuracy"].items():
        acc_str = f"{acc:.1%}" if acc is not None else "n/a"
        print(f"    {name:28s}: {acc_str}")

    # ── Save checkpoint (Phase B loads this) ────────────────────────────
    model = model.cpu()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS_DIR / f"LookMax_{category}_phaseA.pt"
    torch.save({
        "phase": "A",
        "category": category,
        "backbone": args.backbone,
        "image_size": args.image_size,
        "schema": schema,
        "model_state_dict": model.state_dict(),
    }, ckpt_path)
    print(f"\n  {GREEN}✅ Checkpoint saved: {ckpt_path.name}{RESET}")

    metrics = {
        "category": category, "status": "trained",
        "total_rows": src["total_rows"], "usable_rows": len(rows),
        "train_size": n_train, "val_size": n_val,
        "best_val_loss": round(best_val_loss, 4),
        "epochs_trained": len(history),
        "history": history,
        "final_score_mae": final_metrics.get("score_mae"),
        "final_score_rmse": final_metrics.get("score_rmse"),
        "final_head_accuracy": final_metrics["head_accuracy"],
        "checkpoint_path": str(ckpt_path),
        "backbone": args.backbone,
        "image_size": args.image_size,
        "source": src["source"],
    }
    metrics_path = MODELS_DIR / f"LookMax_{category}_phaseA_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  {GREEN}✅ Metrics saved  : {metrics_path.name}{RESET}")

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="LookMax Phase 4 (Phase A) — Synthetic Pretrain (multi-head: score + attributes)"
    )
    parser.add_argument("--category", type=str, default="all",
                         choices=CATEGORIES + ["all"],
                         help="Which category to train (Men_Grooming, Women_Grooming, "
                              "Men_Outfit, Women_Outfit, or all)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, dest="learning_rate")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--backbone", type=str, default=BACKBONE,
                         choices=["mobilenet_v3_large", "mobilenet_v3_small", "efficientnet_b0"])
    parser.add_argument("--dry-run", action="store_true",
                         help="Validate synthetic data presence without training")
    args = parser.parse_args()

    if not HAS_TORCH:
        print("ERROR: PyTorch not installed. Run: pip install -r ML/pipeline/requirements.txt")
        sys.exit(1)

    print(f"\n{'═'*58}")
    print(f"  LookMax ML Pipeline — Phase 4a: Synthetic Pretrain")
    print(f"{'═'*58}")
    print(f"  Category   : {args.category}")
    print(f"  Backbone   : {args.backbone}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Image size : {args.image_size}×{args.image_size}")
    print(f"  Output     : {MODELS_DIR}")
    print(f"  Dry-run    : {args.dry_run}")

    device = get_device()

    targets = CATEGORIES if args.category == "all" else [args.category]

    all_metrics = []
    for category in targets:
        result = train_category(category, device, args, dry_run=args.dry_run)
        all_metrics.append(result)

    # ── Final Summary ─────────────────────────────────────────────────────
    print(f"\n{'═'*58}")
    print(f"  Phase A Training Summary")
    print(f"{'═'*58}")
    any_trained = False
    for m in all_metrics:
        status = m.get("status", "unknown")
        mae = m.get("final_score_mae") or m.get("best_val_loss")
        extra = f"val_loss={m['best_val_loss']:.4f}" if m.get("best_val_loss") is not None else ""
        print(f"  {m['category']:16s} → {status:20s} {extra}")
        if status == "trained":
            any_trained = True

    print(f"\n  Checkpoints (.pt) → {MODELS_DIR}")
    print(f"  Next: 05_finetune_real_world.py --checkpoint <path> --category <category>")
    print(f"{'═'*58}\n")

    if not args.dry_run and not any_trained:
        print(f"{RED}ERROR: no category had usable synthetic data — nothing was trained. "
              f"Run ML/pipeline/dataset_generator/full_run.py (see PLAN.md) first.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
