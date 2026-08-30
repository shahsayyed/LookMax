"""
LookMax ML Pipeline — Phase 4
==============================
04_train_coreml_models.py

Trains a demographic-specific image classifier for each of the 6 demographic
buckets in 3_CoreML_Training_Data/ and exports each trained model as a
ready-to-use Apple CoreML .mlpackage artifact.

Architecture:
  • Backbone  : MobileNetV3-Large (default) or FastViT-T8
  • Classes   : 1_Needs_Improvement | 2_Average | 3_Polished (3 classes)
  • Backend   : PyTorch with Apple MPS (GPU) on M3 Max
  • Export    : coremltools → .mlpackage (drag-and-drop into Xcode)

Output (per demographic):
  ML/models/LookMax_{Demographic}.mlpackage
  ML/models/LookMax_{Demographic}_metrics.json

Usage:
    python3 ML/pipeline/04_train_coreml_models.py
    python3 ML/pipeline/04_train_coreml_models.py --epochs 5 --backbone mobilenet_v3_large
    python3 ML/pipeline/04_train_coreml_models.py --demographic Men_Under_35
    python3 ML/pipeline/04_train_coreml_models.py --epochs 2 --dry-run
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    TRAINING_DATA_DIR, MODELS_DIR, DEMOGRAPHICS, AESTHETIC_TIERS,
    BACKBONE, IMAGE_SIZE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    TRAIN_SPLIT, NUM_CLASSES, EARLY_STOPPING_PATIENCE,
)

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, random_split
    import torchvision.transforms as T
    from torchvision.datasets import ImageFolder
    import torchvision.models as tvm
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import coremltools as ct
    from coremltools.models.neural_network import NeuralNetworkBuilder
    HAS_CT = True
except ImportError:
    HAS_CT = False

try:
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ─── Color Output ────────────────────────────────────────────────────────────
GREEN = "\033[92m"; YELLOW = "\033[93m"; CYAN = "\033[96m"
BOLD  = "\033[1m";  RESET  = "\033[0m"

def header(msg: str):
    print(f"\n{BOLD}{CYAN}{'─'*58}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*58}{RESET}")


# ──────────────────────────────────────────────────────────────────────────────
# Device Selection
# ──────────────────────────────────────────────────────────────────────────────
def get_device() -> "torch.device":
    if torch.backends.mps.is_available():
        print(f"  {GREEN}✓ Apple MPS backend active (M3 Max GPU){RESET}")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"  {YELLOW}→ CUDA GPU detected{RESET}")
        return torch.device("cuda")
    else:
        print(f"  {YELLOW}⚠ CPU-only — training will be slower{RESET}")
        return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Data Transforms
# ──────────────────────────────────────────────────────────────────────────────
def get_transforms(image_size: int, is_train: bool) -> T.Compose:
    if is_train:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.08),
            T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.92, 1.08)),
            T.RandomGrayscale(p=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize(int(image_size * 1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# ──────────────────────────────────────────────────────────────────────────────
# Model Factory
# ──────────────────────────────────────────────────────────────────────────────
def build_model(backbone: str, num_classes: int) -> "nn.Module":
    """
    Load a pre-trained backbone and replace the classifier head
    with a custom 3-class output layer for aesthetic tier prediction.
    """
    if backbone == "mobilenet_v3_large":
        model = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        print(f"  Backbone : MobileNetV3-Large (~5.4M params, ~4.2MB)")

    elif backbone == "mobilenet_v3_small":
        model = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        print(f"  Backbone : MobileNetV3-Small (~2.5M params, ~2MB)")

    elif backbone == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        print(f"  Backbone : EfficientNet-B0 (~5.3M params, ~4.1MB)")

    else:
        # Default fallback: MobileNetV3-Large
        model = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        print(f"  Backbone : MobileNetV3-Large (default)")

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device) -> tuple[float, float, list, list]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


# ──────────────────────────────────────────────────────────────────────────────
# CoreML Export
# ──────────────────────────────────────────────────────────────────────────────
def export_to_coreml(
    model: "nn.Module",
    demographic: str,
    image_size: int,
    class_labels: list[str],
    output_dir: Path,
    stream: str = "Outfit",
) -> Path:
    """
    Converts a trained PyTorch model to .mlpackage using coremltools.
    Returns the path to the saved .mlpackage.
    """
    model.eval()
    model.cpu()

    # Trace with a dummy input
    example_input = torch.zeros(1, 3, image_size, image_size)
    traced = torch.jit.trace(model, example_input)

    # Convert to CoreML
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, image_size, image_size),
                scale=1.0 / 255.0,
                bias=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[
            ct.TensorType(name="classLabelProbs")
        ],
        convert_to="mlprogram",   # .mlpackage format (Xcode 14+)
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,  # Halves model size, negligible accuracy loss
    )

    # ── Add metadata ──────────────────────────────────────────────────────
    mlmodel.author  = "LookMax ML Pipeline"
    mlmodel.license = "Proprietary — NexurTech"
    mlmodel.short_description = (
        f"LookMax {stream} aesthetic tier classifier for {demographic.replace('_', ' ')}. "
        f"Input: 224×224 RGB image. Output: Probability for "
        f"[1_Needs_Improvement, 2_Average, 3_Polished]."
    )
    mlmodel.version = "1.0"

    # ── User-defined metadata ─────────────────────────────────────────────
    mlmodel.user_defined_metadata["stream"] = stream
    mlmodel.user_defined_metadata["demographic"] = demographic
    mlmodel.user_defined_metadata["classes"] = ", ".join(class_labels)
    mlmodel.user_defined_metadata["input_size"] = str(image_size)
    mlmodel.user_defined_metadata["backbone"] = BACKBONE

    # ── Save ─────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    pkg_path = output_dir / f"LookMax_{stream}_{demographic}.mlpackage"
    mlmodel.save(str(pkg_path))

    return pkg_path


# ──────────────────────────────────────────────────────────────────────────────
# Per-Demographic Training Orchestrator
# ──────────────────────────────────────────────────────────────────────────────
def train_demographic(
    demographic: str,
    device: "torch.device",
    args,
    dry_run: bool,
    stream: str = "Outfit",
) -> dict:
    """
    Full train → validate → export cycle for one demographic in a given stream.
    Returns a metrics dict.
    """
    demo_dir = TRAINING_DATA_DIR / stream / demographic
    if not demo_dir.exists():
        demo_dir = TRAINING_DATA_DIR / demographic

    header(f"Training: [{stream}] {demographic}")

    # ── Check dataset size ───────────────────────────────────────────────
    class_folders = [demo_dir / t for t in AESTHETIC_TIERS]
    class_counts = {}
    for cf in class_folders:
        imgs = [f for f in cf.glob("*") if f.is_file() and not f.name.startswith(".")]
        class_counts[cf.name] = len(imgs)

    total_images = sum(class_counts.values())
    print(f"  Stream   : {stream}")
    print(f"  Dataset  :")
    for cls, cnt in class_counts.items():
        print(f"    {cls:30s}: {cnt} images")
    print(f"    {'TOTAL':30s}: {total_images} images")

    if total_images < 30:
        print(f"  {YELLOW}⚠  Skipping [{stream}] {demographic} — insufficient data "
              f"(need ≥30 images, have {total_images}).{RESET}")
        return {"stream": stream, "demographic": demographic, "status": "skipped_insufficient_data",
                "total_images": total_images}

    if dry_run:
        print(f"  {YELLOW}[DRY-RUN] Would train {args.epochs} epoch(s) on {total_images} images.{RESET}")
        return {"stream": stream, "demographic": demographic, "status": "dry_run", "total_images": total_images}

    # ── Build DataLoaders ────────────────────────────────────────────────
    full_dataset = ImageFolder(
        root=str(demo_dir),
        transform=get_transforms(args.image_size, is_train=True),
    )
    n_train = int(len(full_dataset) * TRAIN_SPLIT)
    n_val   = len(full_dataset) - n_train
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    # Override val transform (no augmentation for evaluation)
    val_ds.dataset = ImageFolder(
        root=str(demo_dir),
        transform=get_transforms(args.image_size, is_train=False),
    )

    class_labels = full_dataset.classes
    print(f"  Classes  : {class_labels}")
    print(f"  Train    : {n_train} | Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=2, pin_memory=True)

    # ── Build Model ──────────────────────────────────────────────────────
    model = build_model(args.backbone, NUM_CLASSES)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training Loop ────────────────────────────────────────────────────
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None
    history = []

    print(f"\n  {'Epoch':>6}  {'Train Loss':>11}  {'Train Acc':>10}  {'Val Loss':>10}  {'Val Acc':>8}")
    print(f"  {'─'*6}  {'─'*11}  {'─'*10}  {'─'*10}  {'─'*8}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        marker = " ←best" if val_acc > best_val_acc else ""
        print(f"  {epoch:>6}  {train_loss:>11.4f}  {train_acc:>9.1%}  "
              f"{val_loss:>10.4f}  {val_acc:>7.1%}{marker}  ({elapsed:.1f}s)")

        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4), "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  {YELLOW}Early stopping at epoch {epoch} (no val improvement for {EARLY_STOPPING_PATIENCE} epochs){RESET}")
                break

    # ── Restore best weights ──────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)
    model = model.to("cpu")

    # ── Classification Report ─────────────────────────────────────────────
    if HAS_SKLEARN and val_preds:
        print(f"\n  Classification Report (Validation):")
        report = classification_report(val_labels, val_preds,
                                        target_names=class_labels, zero_division=0)
        for line in report.strip().split("\n"):
            print(f"    {line}")

    # ── CoreML Export ─────────────────────────────────────────────────────
    print(f"\n  {CYAN}Exporting to CoreML (.mlpackage)...{RESET}")
    try:
        pkg_path = export_to_coreml(
            model=model,
            demographic=demographic,
            image_size=args.image_size,
            class_labels=class_labels,
            output_dir=MODELS_DIR,
            stream=stream,
        )
        pkg_size_mb = sum(f.stat().st_size for f in pkg_path.rglob("*") if f.is_file()) / 1_048_576
        print(f"  {GREEN}✅ Exported: {pkg_path.name} ({pkg_size_mb:.1f} MB){RESET}")

        # Save metrics JSON alongside model
        metrics = {
            "stream": stream,
            "demographic": demographic,
            "status": "trained",
            "total_images": total_images,
            "class_counts": class_counts,
            "best_val_accuracy": round(best_val_acc, 4),
            "epochs_trained": len(history),
            "history": history,
            "model_path": str(pkg_path),
            "model_size_mb": round(pkg_size_mb, 2),
            "backbone": args.backbone,
            "image_size": args.image_size,
            "classes": class_labels,
        }
        metrics_path = MODELS_DIR / f"LookMax_{stream}_{demographic}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    except Exception as e:
        print(f"  ✗ CoreML export failed: {e}")
        return {"stream": stream, "demographic": demographic, "status": "export_failed", "error": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="LookMax Phase 4 — CoreML Model Training & Export (Dual-Stream)"
    )
    parser.add_argument("--stream",      type=str,   default="all",
                        choices=["Outfit", "Face_Grooming", "all"],
                        help="Which stream to train (Outfit, Face_Grooming, or all)")
    parser.add_argument("--epochs",     type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE, dest="learning_rate")
    parser.add_argument("--image-size", type=int,   default=IMAGE_SIZE)
    parser.add_argument("--backbone",   type=str,   default=BACKBONE,
                        choices=["mobilenet_v3_large", "mobilenet_v3_small", "efficientnet_b0"])
    parser.add_argument("--demographic", type=str, default=None,
                        help="Train a single demographic (e.g. Men_Under_35)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate dataset presence without training")
    args = parser.parse_args()

    if not HAS_TORCH:
        print("ERROR: PyTorch not installed. Run: pip install -r ML/pipeline/requirements.txt")
        sys.exit(1)
    if not HAS_CT:
        print("ERROR: coremltools not installed. Run: pip install coremltools")
        sys.exit(1)

    print(f"\n{'═'*58}")
    print(f"  LookMax ML Pipeline — Phase 4: CoreML Training (Dual-Stream)")
    print(f"{'═'*58}")
    print(f"  Stream     : {args.stream}")
    print(f"  Backbone   : {args.backbone}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Image size : {args.image_size}×{args.image_size}")
    print(f"  Output     : {MODELS_DIR}")
    print(f"  Dry-run    : {args.dry_run}")

    device = get_device()

    active_streams = ["Outfit", "Face_Grooming"] if args.stream == "all" else [args.stream]
    targets = [args.demographic] if args.demographic else DEMOGRAPHICS

    all_metrics = []
    for stream in active_streams:
        for demographic in targets:
            demo_dir = TRAINING_DATA_DIR / stream / demographic
            if not demo_dir.exists():
                demo_dir = TRAINING_DATA_DIR / demographic
            if not demo_dir.exists():
                print(f"\n  {YELLOW}⚠ Skipping [{stream}] {demographic} — folder not found.{RESET}")
                continue
            result = train_demographic(demographic, device, args, dry_run=args.dry_run, stream=stream)
            all_metrics.append(result)

    # ── Final Summary ─────────────────────────────────────────────────────
    print(f"\n{'═'*58}")
    print(f"  Training Summary")
    print(f"{'═'*58}")
    for m in all_metrics:
        status = m.get("status", "unknown")
        val_acc = m.get("best_val_accuracy", None)
        acc_str = f"val_acc={val_acc:.1%}" if val_acc else ""
        stream_name = m.get("stream", "Outfit")
        print(f"  [{stream_name:13s}] {m['demographic']:20s} → {status:25s} {acc_str}")

    print(f"\n  .mlpackage files → {MODELS_DIR}")
    print(f"  Drag & drop into your Xcode project to begin iOS integration.")
    print(f"{'═'*58}\n")


if __name__ == "__main__":
    main()
