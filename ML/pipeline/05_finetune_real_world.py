"""
LookMax ML Pipeline — Phase 5 (Phase B: Real-World Fine-Tune)
=================================================================
05_finetune_real_world.py

Fine-tunes a Phase A checkpoint (see 04_train_coreml_models.py) on real
scraped-and-VLM-classified photos (ML/data/3_CoreML_Training_Data/), mixed
every batch with a "replay" fraction of synthetic Qwen data to prevent
catastrophic forgetting of the attribute heads — real photos only carry a
tier label (1_Needs_Improvement / 2_Average / 3_Polished), so they can
supervise the `score` head but nothing else. See multihead_common.py for
the shared model/dataset/loss code this depends on.

REAL DATA -> SCORE ANCHOR: tier is mapped to a score sampled uniformly
within a band that matches ML/pipeline/dataset_generator/taxonomy.py's
SCORE_BANDS *exactly* (see RealWorldScoreDataset in multihead_common.py) —
Phase A's synthetic score scale and Phase B's real-tier score scale are
the same continuous 1-10 scale, not two that merely happen to overlap.

PARTIAL-LABEL MASKING: every real sample's attribute-head targets are
masked to 0 contribution in the loss (multihead_common.compute_losses) —
implemented as a per-sample mask multiplied into each head's loss BEFORE
summing, not by skipping heads outright, so one mixed real+synthetic
batch trains every head in a single forward/backward pass.

CATEGORY CONSOLIDATION: real data's 3 age-bucket folders per gender+stream
(Men_Under_35, Men_35_to_50, Men_Over_50, etc.) are pooled into one
dataset per category, matching the synthetic pipeline's category
granularity (no age split) — see config.py's CATEGORY_TO_STREAM and
multihead_common.discover_real_samples().

Output: a final checkpoint per category, then a CoreML .mlpackage export
(multi-output: `score` + one output per attribute head) plus a metrics
JSON, matching the LookMax_<Category>.mlpackage naming convention.

Usage:
    python3 ML/pipeline/05_finetune_real_world.py --dry-run --checkpoint /path/to/phaseA.pt
    python3 ML/pipeline/05_finetune_real_world.py --category Men_Grooming \\
        --checkpoint ML/models/LookMax_Men_Grooming_phaseA.pt
    python3 ML/pipeline/05_finetune_real_world.py --category all \\
        --checkpoint ML/models   # directory containing LookMax_<Category>_phaseA.pt per category
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    SYNTHETIC_QA_DIR, SYNTHETIC_RAW_DIR, TRAINING_DATA_DIR, MODELS_DIR,
    CATEGORIES, CATEGORY_TO_STREAM, DEMOGRAPHICS, AESTHETIC_TIERS,
    BACKBONE, IMAGE_SIZE, BATCH_SIZE, TRAIN_SPLIT,
    FINETUNE_LEARNING_RATE, FINETUNE_NUM_EPOCHS, REPLAY_RATIO_DEFAULT,
    EARLY_STOPPING_PATIENCE,
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
        MultiHeadModel, SyntheticCsvDataset, RealWorldScoreDataset, ReplayMixedLoader,
        compute_losses, evaluate, discover_synthetic_source, discover_real_samples,
        trainable_fields, export_to_coreml,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint resolution
# ──────────────────────────────────────────────────────────────────────────────
def resolve_checkpoint(checkpoint_arg: str, category: str, multi_category: bool) -> tuple:
    """--checkpoint may be a single .pt file (only valid for one category)
    or a directory containing Phase A's LookMax_<Category>_phaseA.pt
    naming convention (required when --category all). Returns
    (resolved_path_or_None, note) — note explains a None/mismatch."""
    if checkpoint_arg is None:
        return None, "no --checkpoint given"
    p = Path(checkpoint_arg)
    if p.is_dir():
        return p / f"LookMax_{category}_phaseA.pt", None
    if multi_category:
        # A single explicit file can't serve every category.
        return None, (f"--checkpoint {checkpoint_arg!r} is a single file, but --category all "
                       f"needs a directory containing LookMax_<Category>_phaseA.pt per category")
    return p, None


# ──────────────────────────────────────────────────────────────────────────────
# Per-Category Fine-Tune Orchestrator
# ──────────────────────────────────────────────────────────────────────────────
def finetune_category(category: str, device, args, dry_run: bool) -> dict:
    header(f"Phase B (real-world fine-tune): {category}")

    multi_category = (args.category == "all")
    ckpt_path, ckpt_note = resolve_checkpoint(args.checkpoint, category, multi_category)

    ckpt_ok = ckpt_path is not None and ckpt_path.exists()
    if ckpt_path is not None:
        print(f"  Checkpoint : {ckpt_path}{'' if ckpt_ok else f'  {YELLOW}[NOT FOUND]{RESET}'}")
    else:
        print(f"  Checkpoint : {YELLOW}[UNRESOLVED]{RESET} ({ckpt_note})")

    real = discover_real_samples(category, TRAINING_DATA_DIR, CATEGORY_TO_STREAM, DEMOGRAPHICS, AESTHETIC_TIERS)
    print(f"  Real data  : {real['total']} images pooled from {real['matched_demographics']} "
          f"({real['stream']}/*/<{'/'.join(AESTHETIC_TIERS)}>)")
    for tier, cnt in real["per_tier_counts"].items():
        print(f"    {tier:24s}: {cnt} images")

    synth = discover_synthetic_source(category, SYNTHETIC_QA_DIR, SYNTHETIC_RAW_DIR)
    if synth["warning"]:
        print(f"  {YELLOW}⚠ {synth['warning']}{RESET}")
    print(f"  Synthetic replay source: {synth['source'] or 'none'} "
          f"({len(synth['rows'])} usable rows)")
    print(f"  Replay ratio: {args.replay_ratio:.0%} synthetic / {1 - args.replay_ratio:.0%} real per batch")

    if dry_run:
        plan_status = "dry_run"
        if not ckpt_ok:
            print(f"  {YELLOW}[DRY-RUN] Checkpoint not found — a real run would fail cleanly here.{RESET}")
        if real["total"] == 0:
            print(f"  {YELLOW}[DRY-RUN] No real training data for {category} yet.{RESET}")
        if args.replay_ratio > 0 and not synth["available"]:
            print(f"  {YELLOW}[DRY-RUN] No synthetic replay data yet — a real run would need "
                  f"--replay-ratio 0 or synthetic data to proceed.{RESET}")
        if ckpt_ok and real["total"] > 0 and (synth["available"] or args.replay_ratio <= 0):
            print(f"  {YELLOW}[DRY-RUN] Would fine-tune {args.epochs} epoch(s), "
                  f"batch_size={args.batch_size}, lr={args.learning_rate}.{RESET}")
        return {"category": category, "status": plan_status, "real_images": real["total"],
                "synthetic_rows": len(synth["rows"]), "checkpoint_found": ckpt_ok}

    # ── Non-dry-run validation — clear errors, no stack traces ──────────
    if not ckpt_ok:
        msg = (f"No Phase A checkpoint found for {category} "
               f"(looked at: {ckpt_path if ckpt_path else args.checkpoint}). "
               f"Run 04_train_coreml_models.py first, or pass --checkpoint <path to .pt / dir>.")
        print(f"  {RED}✗ {msg}{RESET}")
        return {"category": category, "status": "no_checkpoint", "error": msg}

    if real["total"] == 0:
        msg = (f"No real training data found for {category} under {TRAINING_DATA_DIR}. "
               f"Run the real_data_pipeline/ scrape+classify steps first.")
        print(f"  {RED}✗ {msg}{RESET}")
        return {"category": category, "status": "no_real_data", "error": msg}

    if args.replay_ratio > 0 and not synth["available"]:
        msg = (f"No synthetic replay data found for {category} in qa_processed/ or raw_generated/, "
               f"and --replay-ratio={args.replay_ratio} > 0. Either run "
               f"dataset_generator/full_run.py first, or pass --replay-ratio 0 to fine-tune "
               f"real-only (NOT recommended — this will drift/forget the attribute heads).")
        print(f"  {RED}✗ {msg}{RESET}")
        return {"category": category, "status": "no_replay_data", "error": msg}

    # ── Load checkpoint ───────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    schema = ckpt["schema"]
    backbone_name = args.backbone or ckpt.get("backbone", BACKBONE)
    image_size = args.image_size or ckpt.get("image_size", IMAGE_SIZE)
    print(f"  Loaded Phase A checkpoint: backbone={backbone_name}, image_size={image_size}, "
          f"schema fields={len(trainable_fields(schema))}")

    model = MultiHeadModel(backbone_name, schema, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    # ── Real dataset split ────────────────────────────────────────────────
    real_train_full = RealWorldScoreDataset(real["samples"], schema, get_transforms(image_size, is_train=True))
    n_real_train = max(1, int(len(real_train_full) * TRAIN_SPLIT))
    n_real_val = len(real_train_full) - n_real_train
    if n_real_val == 0:
        n_real_train -= 1
        n_real_val = 1
    real_train_ds, real_val_idx = random_split(real_train_full, [n_real_train, n_real_val])
    real_val_full = RealWorldScoreDataset(real["samples"], schema, get_transforms(image_size, is_train=False))
    real_val_ds = torch.utils.data.Subset(real_val_full, real_val_idx.indices)
    print(f"  Real train : {n_real_train} | Real val: {n_real_val}")

    # ── Synthetic replay split (train pool only feeds the mixer; a
    #    separate held-out slice is used to confirm the attribute heads
    #    haven't drifted). ─────────────────────────────────────────────────
    synth_rows = synth["rows"]
    synth_train_ds, synth_val_ds = None, None
    if synth_rows:
        synth_train_full = SyntheticCsvDataset(synth_rows, synth["images_dir"], schema,
                                                get_transforms(image_size, is_train=True))
        n_synth_train = max(1, int(len(synth_train_full) * TRAIN_SPLIT))
        n_synth_val = len(synth_train_full) - n_synth_train
        if n_synth_val == 0 and len(synth_train_full) > 1:
            n_synth_train -= 1
            n_synth_val = 1
        if n_synth_val > 0:
            synth_train_ds, synth_val_idx = random_split(synth_train_full, [n_synth_train, n_synth_val])
            synth_val_full = SyntheticCsvDataset(synth_rows, synth["images_dir"], schema,
                                                  get_transforms(image_size, is_train=False))
            synth_val_ds = torch.utils.data.Subset(synth_val_full, synth_val_idx.indices)
        else:
            synth_train_ds = synth_train_full
        print(f"  Synth train: {len(synth_train_ds)} | Synth val: {len(synth_val_ds) if synth_val_ds else 0}")

    train_loader = ReplayMixedLoader(
        real_train_ds, synth_train_ds if synth_train_ds is not None else [],
        batch_size=args.batch_size, replay_ratio=args.replay_ratio, shuffle=True,
    )
    real_val_loader = DataLoader(real_val_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=2, pin_memory=True)
    synth_val_loader = (DataLoader(synth_val_ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=2, pin_memory=True) if synth_val_ds else None)

    # ── Fine-tune loop ────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    print(f"\n  {'Epoch':>6}  {'Train Loss':>11}  {'Real Val Loss':>13}  {'Score MAE':>10}  {'Score RMSE':>11}")
    print(f"  {'─'*6}  {'─'*11}  {'─'*13}  {'─'*10}  {'─'*11}")

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
        real_val_metrics = evaluate(model, real_val_loader, schema, device)
        scheduler.step()

        elapsed = time.time() - t0
        marker = " ←best" if real_val_metrics["loss"] < best_val_loss else ""
        mae = real_val_metrics.get("score_mae")
        rmse = real_val_metrics.get("score_rmse")
        print(f"  {epoch:>6}  {train_loss:>11.4f}  {real_val_metrics['loss']:>13.4f}  "
              f"{(mae if mae is not None else float('nan')):>10.4f}  "
              f"{(rmse if rmse is not None else float('nan')):>11.4f}{marker}  ({elapsed:.1f}s)")

        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "real_val_loss": round(real_val_metrics["loss"], 4),
            "real_val_score_mae": round(mae, 4) if mae is not None else None,
            "real_val_score_rmse": round(rmse, 4) if rmse is not None else None,
        })

        if real_val_metrics["loss"] < best_val_loss:
            best_val_loss = real_val_metrics["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  {YELLOW}Early stopping at epoch {epoch} "
                      f"(no real-val improvement for {EARLY_STOPPING_PATIENCE} epochs){RESET}")
                break

    if best_state:
        model.load_state_dict(best_state)

    # ── Anti-forgetting check: attribute-head accuracy on synthetic val ──
    replay_check = None
    if synth_val_loader is not None:
        replay_check = evaluate(model, synth_val_loader, schema, device)
        print(f"\n  Synthetic replay val (anti-forgetting check) — loss={replay_check['loss']:.4f}")
        print(f"  Per-head accuracy (synthetic val, should stay high — this is what replay protects):")
        for name, acc in replay_check["head_accuracy"].items():
            acc_str = f"{acc:.1%}" if acc is not None else "n/a"
            print(f"    {name:28s}: {acc_str}")

    final_real_metrics = evaluate(model, real_val_loader, schema, device)
    print(f"\n  Final real-world validation — loss={final_real_metrics['loss']:.4f}", end="")
    if final_real_metrics.get("score_mae") is not None:
        print(f", score_mae={final_real_metrics['score_mae']:.3f}, score_rmse={final_real_metrics['score_rmse']:.3f}")
    else:
        print()

    # ── Save final checkpoint ────────────────────────────────────────────
    model = model.cpu()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    final_ckpt_path = MODELS_DIR / f"LookMax_{category}_final.pt"
    torch.save({
        "phase": "B", "category": category, "backbone": backbone_name,
        "image_size": image_size, "schema": schema, "model_state_dict": model.state_dict(),
    }, final_ckpt_path)
    print(f"  {GREEN}✅ Checkpoint saved: {final_ckpt_path.name}{RESET}")

    # ── CoreML export ─────────────────────────────────────────────────────
    print(f"\n  {CYAN}Exporting to CoreML (.mlpackage)...{RESET}")
    try:
        pkg_path = export_to_coreml(
            model=model, schema=schema, image_size=image_size, category=category,
            backbone_name=backbone_name, output_path=MODELS_DIR / f"LookMax_{category}.mlpackage",
            extra_metadata={
                "phase": "B_real_finetune",
                "replay_ratio": args.replay_ratio,
                "real_images": real["total"],
                "synthetic_replay_rows": len(synth_rows),
            },
        )
        pkg_size_mb = sum(f.stat().st_size for f in pkg_path.rglob("*") if f.is_file()) / 1_048_576
        print(f"  {GREEN}✅ Exported: {pkg_path.name} ({pkg_size_mb:.1f} MB){RESET}")
        export_status = "exported"
        export_error = None
    except Exception as e:
        print(f"  {RED}✗ CoreML export failed: {e}{RESET}")
        pkg_path, pkg_size_mb = None, None
        export_status = "export_failed"
        export_error = str(e)

    metrics = {
        "category": category, "status": export_status,
        "real_images": real["total"], "real_train": n_real_train, "real_val": n_real_val,
        "synthetic_replay_rows": len(synth_rows),
        "replay_ratio": args.replay_ratio,
        "best_real_val_loss": round(best_val_loss, 4),
        "epochs_trained": len(history),
        "history": history,
        "final_score_mae": final_real_metrics.get("score_mae"),
        "final_score_rmse": final_real_metrics.get("score_rmse"),
        "replay_head_accuracy": replay_check["head_accuracy"] if replay_check else None,
        "checkpoint_path": str(final_ckpt_path),
        "model_path": str(pkg_path) if pkg_path else None,
        "model_size_mb": round(pkg_size_mb, 2) if pkg_size_mb else None,
        "export_error": export_error,
        "backbone": backbone_name, "image_size": image_size,
    }
    metrics_path = MODELS_DIR / f"LookMax_{category}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  {GREEN}✅ Metrics saved  : {metrics_path.name}{RESET}")

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="LookMax Phase 5 (Phase B) — Real-World Fine-Tune with Synthetic Replay"
    )
    parser.add_argument("--category", type=str, default="all",
                         choices=CATEGORIES + ["all"],
                         help="Which category to fine-tune (Men_Grooming, Women_Grooming, "
                              "Men_Outfit, Women_Outfit, or all)")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Path to a Phase A .pt checkpoint (single file), or a directory "
                              "containing LookMax_<Category>_phaseA.pt per category (required "
                              "for --category all). Required unless --dry-run.")
    parser.add_argument("--epochs", type=int, default=FINETUNE_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=FINETUNE_LEARNING_RATE, dest="learning_rate")
    parser.add_argument("--image-size", type=int, default=None,
                         help="Override the image size recorded in the checkpoint "
                             f"(defaults to whatever Phase A used, else config.IMAGE_SIZE={IMAGE_SIZE}).")
    parser.add_argument("--backbone", type=str, default=None,
                         choices=["mobilenet_v3_large", "mobilenet_v3_small", "efficientnet_b0"],
                         help="Override the backbone recorded in the checkpoint (defaults to "
                              "whatever Phase A used).")
    parser.add_argument("--replay-ratio", type=float, default=REPLAY_RATIO_DEFAULT,
                         help="Fraction of each training batch drawn from synthetic replay data "
                              "(default 0.3 = 30%% synthetic / 70%% real).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Validate checkpoint/data presence without training")
    args = parser.parse_args()

    if not HAS_TORCH:
        print("ERROR: PyTorch not installed. Run: pip install -r ML/pipeline/requirements.txt")
        sys.exit(1)

    if not args.dry_run and not args.checkpoint:
        print("ERROR: --checkpoint is required unless --dry-run is set.")
        sys.exit(1)

    if not (0.0 <= args.replay_ratio < 1.0):
        print(f"ERROR: --replay-ratio must be in [0.0, 1.0) — got {args.replay_ratio}.")
        sys.exit(1)

    print(f"\n{'═'*58}")
    print(f"  LookMax ML Pipeline — Phase 5: Real-World Fine-Tune")
    print(f"{'═'*58}")
    print(f"  Category     : {args.category}")
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  LR           : {args.learning_rate}")
    print(f"  Image size   : {args.image_size if args.image_size else '(from checkpoint)'}")
    print(f"  Replay ratio : {args.replay_ratio:.0%} synthetic / {1 - args.replay_ratio:.0%} real")
    print(f"  Output       : {MODELS_DIR}")
    print(f"  Dry-run      : {args.dry_run}")

    device = get_device()

    targets = CATEGORIES if args.category == "all" else [args.category]

    all_metrics = []
    for category in targets:
        result = finetune_category(category, device, args, dry_run=args.dry_run)
        all_metrics.append(result)

    # ── Final Summary ─────────────────────────────────────────────────────
    print(f"\n{'═'*58}")
    print(f"  Phase B Fine-Tune Summary")
    print(f"{'═'*58}")
    any_exported = False
    for m in all_metrics:
        status = m.get("status", "unknown")
        extra = f"val_loss={m['best_real_val_loss']:.4f}" if m.get("best_real_val_loss") is not None else ""
        print(f"  {m['category']:16s} → {status:20s} {extra}")
        if status == "exported":
            any_exported = True

    print(f"\n  .mlpackage files → {MODELS_DIR}")
    print(f"  Drag & drop into your Xcode project to begin iOS integration.")
    print(f"{'═'*58}\n")

    if not args.dry_run and not any_exported:
        print(f"{RED}ERROR: no category was successfully fine-tuned/exported. See errors above.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
