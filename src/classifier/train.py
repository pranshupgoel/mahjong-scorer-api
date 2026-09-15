"""
Train the tile classifier (MobileNetV3-Small backbone, fine-tuned) on the
canonical 42-class tile taxonomy.

The labeled dataset (mahjong-dataset-master) is small -- ~628 images across
42 classes, some bonus-tile classes as low as 7-9 examples in the train
split. That's the main risk here, not architecture: augmentation is
deliberately aggressive (no horizontal flip though -- tile faces have
directional numerals/kanji, flipping them would create labels that don't
look like real tiles), class weighting compensates for the uneven counts,
and early stopping on val accuracy guards against overfitting a model this
size to a dataset this small. Run src/classifier/evaluate.py on the held-out
test split once training finishes -- val accuracy on ~1-2 images/class is
too noisy to trust by itself.

See README.md "On-device inference" for the follow-up work (quantization /
export to TFLite or Core ML) once accuracy is validated.

Usage:
    python -m src.classifier.train --splits-dir Datasets/classifier-splits \
        --epochs 30 --batch-size 32 --out checkpoints/classifier.pt
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from src.classifier.dataset import TileDataset
from src.common.labels import CANONICAL_CLASSES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(train: bool):
    if train:
        ops = [
            # Mild random crop instead of a hard resize -- helps the model
            # not depend on tiles being perfectly centered/cropped by the
            # upstream detector.
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomAffine(degrees=12, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.02),
            # No horizontal/vertical flip: tile faces have directional
            # numerals/kanji/wind glyphs, so a flipped tile is not a valid
            # example of its class.
        ]
    else:
        ops = [transforms.Resize((224, 224))]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if train:
        # Random erasing (applied post-normalize, on the tensor) simulates
        # partial occlusion/glare on a physical tile photo -- cheap extra
        # regularization when there are only ~10-15 images per class.
        ops.append(transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)))
    return transforms.Compose(ops)


def build_model(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def compute_class_weights(train_ds: TileDataset, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights so the ~7-image bonus-tile classes aren't
    drowned out by the ~16-image suit classes during training."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for label in train_ds.df["canonical_label"]:
        from src.common.labels import CANONICAL_TO_IDX
        counts[CANONICAL_TO_IDX[label]] += 1
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss, correct, n = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        n += imgs.size(0)
    return total_loss / n, correct / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default="Datasets/classifier-splits")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8, help="early stop after this many epochs with no val_acc improvement")
    ap.add_argument("--freeze-backbone", action="store_true", help="only fine-tune the classifier head -- try this first given how little data there is")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints/classifier.pt")
    ap.add_argument("--log-csv", default="checkpoints/train_log.csv")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir = Path(args.splits_dir)

    train_ds = TileDataset(splits_dir / "train.csv", build_transforms(train=True))
    val_ds = TileDataset(splits_dir / "val.csv", build_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(len(CANONICAL_CLASSES), freeze_backbone=args.freeze_backbone).to(device)
    class_weights = compute_class_weights(train_ds, len(CANONICAL_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    epochs_since_improve = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        print(f"epoch {epoch:02d}  train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}"])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_since_improve = 0
            torch.save({"model_state": model.state_dict(), "classes": CANONICAL_CLASSES}, out_path)
            print(f"  -> saved new best to {out_path} (val_acc={val_acc:.3f})")
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"  no val_acc improvement in {args.patience} epochs -- stopping early at epoch {epoch}")
                break

    print(f"done. best val_acc={best_val_acc:.3f}. Now run src/classifier/evaluate.py on the test split "
          f"for a real read -- val here is only ~{len(val_ds)} images across 42 classes, too small to trust alone.")


if __name__ == "__main__":
    main()
