"""
LookMax ML Pipeline — shared multi-head model / dataset / loss code.
=====================================================================
Used by both:
  • 04_train_coreml_models.py  (Phase A — synthetic pretrain, full multi-head supervision)
  • 05_finetune_real_world.py  (Phase B — real-world fine-tune with synthetic replay)

Both phases train the exact same architecture: one shared CNN backbone plus
one small head per entry in label_schema_<Category>.json (skipping `meta`
entries — those are provenance-only, e.g. requested_upper_color, and are
never trained; see taxonomy.py's module docstring on the three-bucket
model). Keeping the model class, target encoding, and the masked
multi-head loss in ONE module means:
  - Phase B's checkpoint load is guaranteed architecture-identical to
    what Phase A produced (same schema -> same heads, in the same order).
  - The score-vs-attribute loss weighting (schema `loss_weight`: 1.0 for
    score, 0.3/0.5 for everything else) is read from the schema exactly
    once, not re-hardcoded in two places where it could drift apart.

This module does NOT read or write anything under
ML/pipeline/dataset_generator/ or ML/pipeline/real_data_pipeline/ except
importing taxonomy.py (read-only) for SCORE_BANDS.
"""
import csv
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# taxonomy.py lives in the sibling dataset_generator/ package — imported
# (read-only) for SCORE_BANDS so Phase B's real-data score anchors can be
# DERIVED from it rather than hand-duplicated (see RealWorldScoreDataset).
sys.path.insert(0, str(Path(__file__).resolve().parent / "dataset_generator"))
import taxonomy as tx  # noqa: E402

try:
    import coremltools as ct
    HAS_CT = True
except ImportError:
    HAS_CT = False


# ─── Color Output (shared terminal style) ────────────────────────────────────
GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"; CYAN = "\033[96m"
BOLD = "\033[1m"; RESET = "\033[0m"


def header(msg: str):
    print(f"\n{BOLD}{CYAN}{'─'*58}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*58}{RESET}")


def get_device() -> "torch.device":
    if torch.backends.mps.is_available():
        print(f"  {GREEN}✓ Apple MPS backend active{RESET}")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"  {YELLOW}→ CUDA GPU detected{RESET}")
        return torch.device("cuda")
    else:
        print(f"  {YELLOW}⚠ CPU-only — training will be slower{RESET}")
        return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Data Transforms (unchanged from the previous single-head trainer)
# ──────────────────────────────────────────────────────────────────────────────
def get_transforms(image_size: int, is_train: bool) -> "T.Compose":
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
# Schema helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_schema(schema_path: Path) -> list:
    with open(schema_path) as f:
        return json.load(f)


def trainable_fields(schema: list) -> list:
    """Every schema entry except `meta` — meta fields (e.g.
    requested_upper_color) are provenance-only and never get a head."""
    return [f for f in schema if f["type"] != "meta"]


def encode_target(field: dict, raw_value) -> float | int:
    """Converts one raw label value (a CSV string, or a python float/str
    already) into the scalar a torch.tensor() target expects: float for
    regression, integer class-index otherwise."""
    ftype = field["type"]
    if ftype == "regression":
        return float(raw_value)
    if ftype == "ordinal":
        return int(float(raw_value))
    if ftype == "categorical":
        classes = field["classes"]
        s = str(raw_value)
        if s not in classes:
            # Defensive fallback for a blank/malformed CSV cell — training
            # correctness for this row is governed by qa_pass filtering
            # upstream, not by this function; this just avoids a crash.
            return 0
        return classes.index(s)
    raise ValueError(f"encode_target() called on a non-trainable field: {field['name']!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Backbone factory — returns a feature-extractor module (pooled, flattened,
# pre-final-layer features) so N heads can share one trunk.
# ──────────────────────────────────────────────────────────────────────────────
class _FeatureExtractor(nn.Module):
    def __init__(self, features: nn.Module, pool: nn.Module, trunk: nn.Module):
        super().__init__()
        self.features = features
        self.pool = pool
        self.trunk = trunk

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.trunk(x)


def build_backbone(name: str, pretrained: bool = True) -> tuple["nn.Module", int]:
    """Loads a pretrained backbone and strips its final classification
    layer, keeping the "neck" (hidden linear + activation + dropout, where
    the architecture has one) so every head gets a rich shared feature
    vector rather than raw pooled conv features."""
    if name == "mobilenet_v3_large":
        weights = tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        m = tvm.mobilenet_v3_large(weights=weights)
        feat_dim = m.classifier[0].out_features  # 1280
        trunk = nn.Sequential(m.classifier[0], m.classifier[1], m.classifier[2])
        print(f"  Backbone : MobileNetV3-Large (~5.4M params)")
        return _FeatureExtractor(m.features, m.avgpool, trunk), feat_dim

    elif name == "mobilenet_v3_small":
        weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.mobilenet_v3_small(weights=weights)
        feat_dim = m.classifier[0].out_features  # 1024
        trunk = nn.Sequential(m.classifier[0], m.classifier[1], m.classifier[2])
        print(f"  Backbone : MobileNetV3-Small (~2.5M params)")
        return _FeatureExtractor(m.features, m.avgpool, trunk), feat_dim

    elif name == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.efficientnet_b0(weights=weights)
        feat_dim = m.classifier[1].in_features  # 1280
        trunk = nn.Sequential(m.classifier[0])  # Dropout only
        print(f"  Backbone : EfficientNet-B0 (~5.3M params)")
        return _FeatureExtractor(m.features, m.avgpool, trunk), feat_dim

    else:
        # Default fallback: MobileNetV3-Large
        weights = tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        m = tvm.mobilenet_v3_large(weights=weights)
        feat_dim = m.classifier[0].out_features
        trunk = nn.Sequential(m.classifier[0], m.classifier[1], m.classifier[2])
        print(f"  Backbone : MobileNetV3-Large (default fallback for {name!r})")
        return _FeatureExtractor(m.features, m.avgpool, trunk), feat_dim


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Head Model
# ──────────────────────────────────────────────────────────────────────────────
class MultiHeadModel(nn.Module):
    """One shared backbone + one small linear head per trainable schema
    field: `regression` -> Linear(feat_dim, 1); `ordinal` -> Linear(feat_dim,
    levels); `categorical` -> Linear(feat_dim, len(classes)). `meta` fields
    are skipped entirely (see trainable_fields())."""

    def __init__(self, backbone_name: str, schema: list, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone_name
        self.schema = trainable_fields(schema)
        self.feature_extractor, self.feat_dim = build_backbone(backbone_name, pretrained)

        heads = {}
        for f in self.schema:
            name = f["name"]
            if f["type"] == "regression":
                heads[name] = nn.Linear(self.feat_dim, 1)
            elif f["type"] == "ordinal":
                heads[name] = nn.Linear(self.feat_dim, f["levels"])
            elif f["type"] == "categorical":
                heads[name] = nn.Linear(self.feat_dim, len(f["classes"]))
        self.heads = nn.ModuleDict(heads)

    def forward(self, x) -> dict:
        feat = self.feature_extractor(x)
        return {name: head(feat) for name, head in self.heads.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Masked multi-head loss
# ──────────────────────────────────────────────────────────────────────────────
def compute_losses(outputs: dict, targets: dict, masks: dict, schema: list):
    """
    outputs: dict[name] -> (B, C) logits, or (B, 1) for regression
    targets: dict[name] -> (B,) tensor — float for regression, long index otherwise
    masks:   dict[name] -> (B,) float tensor, 1.0 where this head is
             supervised for that sample, 0.0 where it must not contribute
             (real-world samples: every head except `score` is masked to
             0 — see 05_finetune_real_world.py's RealWorldScoreDataset).

    The per-head loss is a MASKED MEAN, i.e. (raw_per_sample * mask).sum()
    / mask.sum() — not `.sum() / batch_size`. Dividing by the effective
    (unmasked) count instead of the raw batch size means an attribute
    head's loss magnitude does not shrink just because a mixed real+
    synthetic batch happens to contain mostly real (masked-out) samples;
    it is computed as if the head were trained only on the samples that
    actually supervise it. `weight` is schema['loss_weight'] read
    verbatim from label_schema_<Category>.json — score is 1.0, everything
    else is 0.3 (0.5 for `formality`) by construction of the schema, never
    re-hardcoded here.

    Returns (total_loss: scalar tensor, per_head: dict[name] -> float).
    """
    total = None
    per_head = {}
    for f in schema:
        if f["type"] == "meta":
            continue
        name = f["name"]
        weight = f["loss_weight"]
        out = outputs[name]
        tgt = targets[name]
        m = masks[name].to(out.dtype)

        if f["type"] == "regression":
            raw = F.smooth_l1_loss(out.squeeze(-1), tgt, reduction="none")
        else:
            raw = F.cross_entropy(out, tgt, reduction="none")

        denom = m.sum().clamp(min=1.0)
        head_loss = (raw * m).sum() / denom
        per_head[name] = head_loss.detach().item()
        term = weight * head_loss
        total = term if total is None else total + term
    return total, per_head


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic Qwen dataset (CSV + shared images/ dir)
# ──────────────────────────────────────────────────────────────────────────────
class SyntheticCsvDataset(Dataset):
    """One row per generated image. Every trainable schema field gets a
    real target and mask=1.0 — full multi-head supervision, this is the
    "replay" data Phase B mixes back in."""

    def __init__(self, rows: list, images_dir: Path, schema: list, transform):
        self.rows = rows
        self.images_dir = Path(images_dir)
        self.schema = trainable_fields(schema)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(self.images_dir / row["filename"]).convert("RGB")
        image = self.transform(image)

        targets, masks = {}, {}
        for f in self.schema:
            name = f["name"]
            raw = row.get(name, "")
            if raw == "" or raw is None:
                val = 0.0 if f["type"] == "regression" else 0
            else:
                val = encode_target(f, raw)
            if f["type"] == "regression":
                targets[name] = torch.tensor(float(val), dtype=torch.float32)
            else:
                targets[name] = torch.tensor(int(val), dtype=torch.long)
            masks[name] = torch.tensor(1.0, dtype=torch.float32)
        return image, targets, masks


# ──────────────────────────────────────────────────────────────────────────────
# Real-world dataset — tier -> score anchor, score head only
# ──────────────────────────────────────────────────────────────────────────────
class RealWorldScoreDataset(Dataset):
    """Wraps a flat list of (image_path, tier) samples pooled across all
    age-demographics for one gender+stream category (see
    05_finetune_real_world.py's discover_real_samples()). Real photos only
    carry a tier label, never the synthetic pipeline's rich attribute
    labels, so every sample supervises ONLY the `score` head; every other
    head is masked to 0.0 (present in the target dict purely so a batch
    can collate/concatenate uniformly with synthetic replay samples in
    the same forward/backward pass — see ReplayMixedLoader).

    Score anchors are DERIVED from taxonomy.SCORE_BANDS (not
    hand-duplicated numbers) so Phase A's synthetic score scale and Phase
    B's real-tier score scale are the same continuous 1-10 scale:
      1_Needs_Improvement -> combined flaw_severe ∪ flaw_mild band (1.0, 3.0)
      2_Average            -> exactly taxonomy.SCORE_BANDS["average"]  (4.0, 6.0)
      3_Polished            -> exactly taxonomy.SCORE_BANDS["polished"] (7.0, 10.0)
    """
    TIER_SCORE_RANGES = {
        "1_Needs_Improvement": (tx.SCORE_BANDS["flaw_severe"][0], tx.SCORE_BANDS["flaw_mild"][1]),
        "2_Average": tx.SCORE_BANDS["average"],
        "3_Polished": tx.SCORE_BANDS["polished"],
    }

    def __init__(self, samples: list, schema: list, transform, seed: int = 0):
        self.samples = samples  # list of (Path, tier_str)
        self.schema = trainable_fields(schema)
        self.transform = transform
        self._seed = seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, tier = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        # Deterministic-but-varied per (idx, epoch-independent) draw: seed a
        # local RNG from the sample index so re-reading the same sample
        # within an epoch (num_workers>0 re-invoking __getitem__) is stable
        # while different samples get different anchors.
        rng = random.Random(self._seed * 1_000_003 + idx)
        lo, hi = self.TIER_SCORE_RANGES[tier]
        score = round(rng.uniform(lo, hi), 1)

        targets, masks = {}, {}
        for f in self.schema:
            name = f["name"]
            if name == "score":
                targets[name] = torch.tensor(score, dtype=torch.float32)
                masks[name] = torch.tensor(1.0, dtype=torch.float32)
            elif f["type"] == "regression":
                targets[name] = torch.tensor(0.0, dtype=torch.float32)
                masks[name] = torch.tensor(0.0, dtype=torch.float32)
            else:
                targets[name] = torch.tensor(0, dtype=torch.long)
                masks[name] = torch.tensor(0.0, dtype=torch.float32)
        return image, targets, masks


# ──────────────────────────────────────────────────────────────────────────────
# Replay batch mixer — combines a real loader and a synthetic loader into
# one stream of mixed batches (real-majority, e.g. 70/30).
# ──────────────────────────────────────────────────────────────────────────────
def _merge_batches(b1, b2):
    img1, t1, m1 = b1
    img2, t2, m2 = b2
    images = torch.cat([img1, img2], dim=0)
    targets = {k: torch.cat([t1[k], t2[k]], dim=0) for k in t1}
    masks = {k: torch.cat([m1[k], m2[k]], dim=0) for k in m1}
    return images, targets, masks


class ReplayMixedLoader:
    """Yields batches built from `real_n` real samples + `synth_n`
    synthetic replay samples concatenated together (real_n + synth_n ==
    batch_size), so a single forward/backward pass sees both real
    score-only supervision AND full synthetic multi-head supervision —
    this is the mechanism that fights catastrophic forgetting of the
    attribute heads (real data alone can never train them; see
    RealWorldScoreDataset). The synthetic side is cycled (repeated) for
    the length of one epoch over the (usually larger) real dataset.
    If `synth_dataset` is empty, falls back to real-only batches.
    """

    def __init__(self, real_dataset, synth_dataset, batch_size: int,
                 replay_ratio: float, shuffle: bool = True, num_workers: int = 2):
        self.synth_n = (max(1, round(batch_size * replay_ratio))
                        if (replay_ratio > 0 and len(synth_dataset)) else 0)
        self.real_n = batch_size - self.synth_n
        assert self.real_n > 0, "replay_ratio too high — no room left for real samples in the batch"

        self.real_loader = DataLoader(real_dataset, batch_size=self.real_n, shuffle=shuffle,
                                       num_workers=num_workers, drop_last=True, pin_memory=True)
        self.synth_loader = None
        if self.synth_n > 0:
            self.synth_loader = DataLoader(synth_dataset, batch_size=self.synth_n, shuffle=shuffle,
                                            num_workers=num_workers, drop_last=True, pin_memory=True)

    def __len__(self):
        return len(self.real_loader)

    def __iter__(self):
        synth_iter = self._cycle(self.synth_loader) if self.synth_loader is not None else None
        for real_batch in self.real_loader:
            if synth_iter is None:
                yield real_batch
            else:
                yield _merge_batches(real_batch, next(synth_iter))

    @staticmethod
    def _cycle(loader):
        while True:
            for b in loader:
                yield b


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader, schema: list, device) -> dict:
    """Runs a full pass, returning total masked loss, score MAE/RMSE (over
    samples where the score head was actually supervised), and per-head
    classification accuracy (None for a head with zero supervised samples
    in this loader — e.g. every attribute head on a real-only val set)."""
    model.eval()
    trainable = trainable_fields(schema)
    total_loss, total_n = 0.0, 0
    score_abs_errs, score_sq_errs = [], []
    correct = {f["name"]: 0 for f in trainable if f["type"] != "regression"}
    counts = {f["name"]: 0 for f in trainable if f["type"] != "regression"}

    with torch.no_grad():
        for images, targets, masks in loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            masks = {k: v.to(device) for k, v in masks.items()}
            outputs = model(images)
            loss, _ = compute_losses(outputs, targets, masks, schema)

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_n += bs

            for f in trainable:
                name = f["name"]
                m = masks[name]
                valid = m > 0
                if not valid.any():
                    continue
                if f["type"] == "regression":
                    err = (outputs[name].squeeze(-1) - targets[name])[valid]
                    score_abs_errs.extend(err.abs().cpu().tolist())
                    score_sq_errs.extend((err ** 2).cpu().tolist())
                else:
                    preds = outputs[name].argmax(dim=1)
                    correct[name] += (preds[valid] == targets[name][valid]).sum().item()
                    counts[name] += int(valid.sum().item())

    result = {"loss": total_loss / total_n if total_n else 0.0}
    if score_abs_errs:
        result["score_mae"] = sum(score_abs_errs) / len(score_abs_errs)
        result["score_rmse"] = (sum(score_sq_errs) / len(score_sq_errs)) ** 0.5
    result["head_accuracy"] = {
        name: (correct[name] / counts[name] if counts[name] else None) for name in correct
    }
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Data discovery — never raises. Both Phase A and Phase B need --dry-run
# (and even non-dry-run) to report "0 images found" cleanly rather than
# crash, since the 28,000-image synthetic run may not have happened yet
# (see ML/README.md's documented gap).
# ──────────────────────────────────────────────────────────────────────────────
def discover_synthetic_source(category: str, qa_dir: Path, raw_dir: Path) -> dict:
    """Prefers qa_dir (qa_processed/ — has a qa_pass column written by
    extract_measured_labels.py; trains only on qa_pass==1), falls back to
    raw_dir (raw_generated/, unfiltered) with a warning. Returns a dict
    that is always safe to inspect even when nothing was found:
      {available, source, schema, schema_path, images_dir, csv_path,
       rows (already qa-filtered when applicable), total_rows, warning}
    """
    result = {
        "available": False, "source": None, "schema": None, "schema_path": None,
        "images_dir": None, "csv_path": None, "rows": [], "total_rows": 0, "warning": None,
    }

    def _try(base_dir: Path, label: str):
        if not base_dir.exists():
            return None
        schema_path = base_dir / f"label_schema_{category}.json"
        images_dir = base_dir / "images"
        csv_candidates = [
            base_dir / f"labels_{category}_measured.csv",
            base_dir / f"labels_{category}.csv",
        ]
        csv_path = next((p for p in csv_candidates if p.exists()), None)
        if not schema_path.exists() or csv_path is None or not images_dir.exists():
            return None
        schema = load_schema(schema_path)
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        return {"schema": schema, "schema_path": schema_path, "images_dir": images_dir,
                "csv_path": csv_path, "rows": rows, "source": label}

    found = _try(qa_dir, "qa_processed")
    if found is None:
        found = _try(raw_dir, "raw_generated")
        if found is not None:
            result["warning"] = (
                f"{category}: qa_processed/ has no usable data — falling back to "
                f"raw_generated/ (unfiltered, no qa_pass gate). Run "
                f"extract_measured_labels.py once the synthetic generation run "
                f"is complete for cleaner training data."
            )
    if found is None:
        result["warning"] = (
            f"{category}: no synthetic data found in qa_processed/ or raw_generated/ "
            f"— 0 images available. Has dataset_generator/full_run.py been run yet?"
        )
        return result

    rows = found["rows"]
    total_rows = len(rows)
    if found["source"] == "qa_processed" and rows and "qa_pass" in rows[0]:
        filtered = [r for r in rows if str(r.get("qa_pass", "")).strip() == "1"]
    else:
        filtered = rows

    result.update({
        "available": len(filtered) > 0,
        "source": found["source"],
        "schema": found["schema"],
        "schema_path": found["schema_path"],
        "images_dir": found["images_dir"],
        "csv_path": found["csv_path"],
        "rows": filtered,
        "total_rows": total_rows,
    })
    if not result["available"] and result["warning"] is None:
        result["warning"] = (
            f"{category}: found {total_rows} row(s) in {found['source']} but 0 usable "
            f"(all failed QA or the CSV was empty)."
        )
    return result


def discover_real_samples(category: str, training_data_dir: Path,
                           category_to_stream: dict, demographics: list, tiers: list) -> dict:
    """Pools all 3 age-bucket demographics for this category's gender under
    TRAINING_DATA_DIR/<stream>/<demographic>/<tier>/*.jpg into one flat
    list of (path, tier) samples — see config.py's CATEGORY_TO_STREAM."""
    gender_prefix = category.split("_")[0]  # "Men" / "Women"
    stream = category_to_stream[category]
    matched_demographics = [d for d in demographics if d.startswith(gender_prefix)]

    samples = []
    per_tier_counts = {t: 0 for t in tiers}
    for demo in matched_demographics:
        for tier in tiers:
            tier_dir = training_data_dir / stream / demo / tier
            if not tier_dir.exists():
                continue
            for f in tier_dir.glob("*"):
                if f.is_file() and not f.name.startswith("."):
                    samples.append((f, tier))
                    per_tier_counts[tier] += 1

    return {
        "samples": samples, "stream": stream, "matched_demographics": matched_demographics,
        "per_tier_counts": per_tier_counts, "total": len(samples),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CoreML Export — multi-output (score + one output per attribute head)
# ──────────────────────────────────────────────────────────────────────────────
def export_to_coreml(model: "nn.Module", schema: list, image_size: int, category: str,
                      backbone_name: str, output_path: Path, extra_metadata: dict = None) -> Path:
    """Traces the model and converts to .mlpackage with one NAMED output
    per trainable schema head: the score head is exported as a raw scalar
    regression output; every categorical/ordinal head is exported through
    a softmax so iOS gets a probability distribution rather than raw
    logits. user_defined_metadata carries the full label schema (as JSON)
    so the iOS side can interpret which output is which without
    hand-maintaining a duplicate list of head names/types."""
    if not HAS_CT:
        raise RuntimeError("coremltools is not installed — cannot export to CoreML.")

    model = model.eval().cpu()
    trainable = trainable_fields(schema)

    class _ExportWrapper(nn.Module):
        def __init__(self, inner, fields):
            super().__init__()
            self.inner = inner
            self.fields = fields

        def forward(self, x):
            outs = self.inner(x)
            results = []
            for f in self.fields:
                o = outs[f["name"]]
                if f["type"] == "regression":
                    o = o.squeeze(-1)
                else:
                    o = torch.softmax(o, dim=1)
                results.append(o)
            return tuple(results)

    wrapper = _ExportWrapper(model, trainable).eval()
    example_input = torch.zeros(1, 3, image_size, image_size)
    traced = torch.jit.trace(wrapper, example_input)

    output_names = [f["name"] for f in trainable]
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
        outputs=[ct.TensorType(name=name) for name in output_names],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )

    mlmodel.author = "LookMax ML Pipeline"
    mlmodel.license = "Proprietary — NexurTech"
    mlmodel.short_description = (
        f"LookMax {category} effort model. Input: {image_size}x{image_size} RGB image. "
        f"Outputs: 'score' (1-10 regression) plus one probability distribution per "
        f"attribute head — see user_defined_metadata['label_schema']."
    )
    mlmodel.version = "1.0"
    mlmodel.user_defined_metadata["category"] = category
    mlmodel.user_defined_metadata["backbone"] = backbone_name
    mlmodel.user_defined_metadata["input_size"] = str(image_size)
    mlmodel.user_defined_metadata["output_names"] = ", ".join(output_names)
    mlmodel.user_defined_metadata["label_schema"] = json.dumps(schema)
    if extra_metadata:
        for k, v in extra_metadata.items():
            mlmodel.user_defined_metadata[k] = str(v)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    return output_path
