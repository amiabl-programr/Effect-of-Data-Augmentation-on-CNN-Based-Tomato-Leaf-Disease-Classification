"""
Train one experimental condition (baseline / flip / rotation / brightness).

Usage:
    python src/train.py --config configs/baseline.yaml

Each run logs to W&B under the group named by the augmentation condition
and saves a checkpoint under results/.
"""
import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam

from dataset import build_dataloaders, compute_class_weights, stratified_split
from model import SmallCNN
from utils import set_seed


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main(config_path):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    train_loader, val_loader, test_loader, classes = build_dataloaders(
        root=cfg["data_root"],
        condition=cfg["condition"],
        seed=cfg["seed"],
        batch_size=cfg["batch_size"],
        aug_params=cfg.get("aug_params", {}),
    )

    # --- Optional class weighting, applied consistently if enabled ---
    criterion_weight = None
    if cfg.get("use_class_weights", False):
        train_samples, _, _, _, _ = stratified_split(cfg["data_root"], cfg["seed"])
        criterion_weight = compute_class_weights(train_samples, len(classes)).to(device)

    # --- Model ---
    model = SmallCNN(num_classes=len(classes), dropout=cfg.get("dropout", 0.3)).to(device)
    criterion = nn.CrossEntropyLoss(weight=criterion_weight)
    optimizer = Adam(model.parameters(), lr=cfg["learning_rate"])

    # --- W&B ---
    use_wandb = cfg.get("use_wandb", True)
    if use_wandb:
        import wandb
        wandb.init(
            entity="oluwayemivictor15-independent-researcher",
            project=cfg.get("wandb_project", "tomato-leaf-augmentation"),
            group=cfg["condition"],
            name=f"{cfg['condition']}_seed{cfg['seed']}",
            config=cfg,
        )
        wandb.config.update({"num_params": model.count_parameters(), "classes": classes})

    # --- Training loop ---
    best_val_acc = 0.0
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["checkpoint_dir"], f"{cfg['condition']}_seed{cfg['seed']}.pt")

    for epoch in range(cfg["epochs"]):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(f"Epoch {epoch+1}/{cfg['epochs']} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc,
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(), "classes": classes, "cfg": cfg}, ckpt_path)

    print(f"Best val_acc={best_val_acc:.4f}. Checkpoint saved to {ckpt_path}")
    if use_wandb:
        wandb.summary["best_val_acc"] = best_val_acc
        wandb.finish()

    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()
    main(args.config)
