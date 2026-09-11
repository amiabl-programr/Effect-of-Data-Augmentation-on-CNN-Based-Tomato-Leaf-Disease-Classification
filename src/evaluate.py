"""
Final test-set evaluation for a trained checkpoint (sections 3.13-3.14).

Usage:
    python src/evaluate.py --checkpoint results/checkpoints/baseline_seed42.pt --config configs/experiments/baseline.yaml
"""
import argparse
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

from dataset import build_dataloaders
from model import SmallCNN


def evaluate(checkpoint_path, config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    classes = ckpt["classes"]

    _, _, test_loader, _ = build_dataloaders(
        root=cfg["data_root"], condition=cfg["condition"], seed=cfg["seed"],
        batch_size=cfg["batch_size"], aug_params=cfg.get("aug_params", {}),
    )

    model = SmallCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    print(f"\n=== Test results: {cfg['condition']} (seed {cfg['seed']}) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro Precision: {precision:.4f} | Macro Recall: {recall:.4f} | Macro F1: {f1:.4f}\n")
    print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    os.makedirs("results/confusion_matrices", exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {cfg['condition']} (seed {cfg['seed']})")
    plt.tight_layout()
    out_path = f"results/confusion_matrices/{cfg['condition']}_seed{cfg['seed']}.png"
    plt.savefig(out_path)
    print(f"Confusion matrix saved to {out_path}")

    return {"accuracy": acc, "macro_precision": precision, "macro_recall": recall, "macro_f1": f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.config)