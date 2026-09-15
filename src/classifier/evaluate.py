"""
Evaluate a trained classifier checkpoint on the held-out test split and
report per-class precision/recall/F1 plus a confusion matrix -- this is
the "does it actually work" check to run once training finishes, separate
from the train/val loop in train.py.

Usage:
    python -m src.classifier.evaluate --splits-dir Datasets/classifier-splits \
        --checkpoint checkpoints/classifier.pt --out-dir eval_out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.classifier.dataset import TileDataset
from src.classifier.train import build_model, build_transforms


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = logits.argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
    return all_labels, all_preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default="Datasets/classifier-splits")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--checkpoint", default="checkpoints/classifier.pt")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-dir", default="eval_out")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt["classes"]

    ds = TileDataset(Path(args.splits_dir) / f"{args.split}.csv", build_transforms(train=False))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])

    labels, preds = run_inference(model, loader, device)

    report_dict = classification_report(
        labels, preds, labels=list(range(len(classes))), target_names=classes,
        zero_division=0, output_dict=True,
    )
    report_txt = classification_report(
        labels, preds, labels=list(range(len(classes))), target_names=classes,
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.split}_report.txt").write_text(report_txt)
    (out_dir / f"{args.split}_report.json").write_text(json.dumps(report_dict, indent=2))

    # Confusion matrix as PNG if matplotlib is available; always dump raw counts too.
    (out_dir / f"{args.split}_confusion_matrix.json").write_text(
        json.dumps({"classes": classes, "matrix": cm.tolist()}, indent=2)
    )
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=6)
        ax.set_yticklabels(classes, fontsize=6)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix ({args.split}), overall accuracy={report_dict['accuracy']:.3f}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"{args.split}_confusion_matrix.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass

    print(report_txt)
    print(f"overall accuracy: {report_dict['accuracy']:.4f}")
    print(f"saved report + confusion matrix to {out_dir}/")

    # Flag the classes that are worst-performing -- useful to know which
    # tiles need more training data or better augmentation.
    per_class = [(name, report_dict[name]["f1-score"], report_dict[name]["support"])
                 for name in classes]
    per_class.sort(key=lambda x: x[1])
    print("\nworst 5 classes by F1:")
    for name, f1, support in per_class[:5]:
        print(f"  {name}: f1={f1:.3f} (support={int(support)})")


if __name__ == "__main__":
    main()
