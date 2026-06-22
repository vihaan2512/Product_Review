import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as tvm
from PIL import Image
from sklearn.model_selection import train_test_split
from loguru import logger

from src.utils.logger import setup_logger
from src.utils.metrics import evaluate_defect
from src.utils.tracker import ExperimentTracker


# ── Dataset ───────────────────────────────────────────────────

class MVTecDataset(Dataset):
    """
    PyTorch Dataset for MVTec images.
    Loads images from the metadata CSV created in Week 1.
    """

    def __init__(self, dataframe: pd.DataFrame, transform=None, augment: bool = False):
        self.data      = dataframe.reset_index(drop=True)
        self.transform = transform
        self.augment   = augment

        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]) if augment else None

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row   = self.data.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label"])

        if self.augment and self.aug_transform:
            image = self.aug_transform(image)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# ── Training ──────────────────────────────────────────────────

def train_resnet(args) -> None:
    setup_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    # ── Load metadata ─────────────────────────────────────────
    meta_path = Path("data/processed/mvtec_metadata.csv")
    if not meta_path.exists():
        logger.error(
            "MVTec metadata not found.\n"
            "Run: python src/utils/preprocess.py first."
        )
        return

    df = pd.read_csv(meta_path)
    logger.info(f"Total images: {len(df)} | Defects: {df['label'].sum()}")

    test_df = df[df["split"] == "test"].reset_index(drop=True)
    train_meta, val_meta = train_test_split(
        test_df, test_size=0.2, random_state=42, stratify=test_df["label"]
    )

    train_good = df[(df["split"] == "train") & (df["label"] == 0)]
    train_meta = pd.concat([train_meta, train_good]).reset_index(drop=True)

    logger.info(
        f"Fine-tune split — Train: {len(train_meta)} | Val: {len(val_meta)}\n"
        f"Train defects: {train_meta['label'].sum()} | "
        f"Train normal: {(train_meta['label']==0).sum()}"
    )

    # ── Transforms ────────────────────────────────────────────
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    # ── Datasets & DataLoaders ────────────────────────────────
    train_dataset = MVTecDataset(train_meta, transform=base_transform, augment=True)
    val_dataset   = MVTecDataset(val_meta,   transform=base_transform, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── Model — Transfer Learning ─────────────────────────────
    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)

    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    model.fc = nn.Linear(2048, 2)
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({trainable/total:.1%})")

    # ── Class weights for imbalance ───────────────────────────
    n_normal   = (train_meta["label"] == 0).sum()
    n_defect   = (train_meta["label"] == 1).sum()
    w_normal   = len(train_meta) / (2 * n_normal)
    w_defect   = len(train_meta) / (2 * n_defect)
    weights    = torch.tensor([w_normal, w_defect], dtype=torch.float).to(device)
    criterion  = nn.CrossEntropyLoss(weight=weights)
    logger.info(f"Class weights — normal: {w_normal:.2f} | defective: {w_defect:.2f}")

    # ── Optimizer — only update unfrozen params ───────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    # ── Training loop ─────────────────────────────────────────
    tracker = ExperimentTracker("defect_module")
    best_auroc = 0.0
    best_path  = Path("models/defect_resnet_best.pt")
    best_path.parent.mkdir(exist_ok=True)

    with tracker.start_run("resnet50_finetune"):
        tracker.log_params({
            "epochs": args.epochs, "lr": args.lr,
            "batch_size": args.batch_size, "trainable_params": trainable,
        })

        for epoch in range(1, args.epochs + 1):
            # Train
            model.train()
            train_loss, correct, total_n = 0, 0, 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss   = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                correct    += (logits.argmax(1) == labels).sum().item()
                total_n    += labels.size(0)

            scheduler.step()
            train_acc = correct / total_n

            # Validate
            model.eval()
            y_true, y_scores = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    probs  = torch.softmax(model(images), dim=1)[:, 1]
                    y_true.extend(labels.numpy())
                    y_scores.extend(probs.cpu().numpy())

            val_metrics = evaluate_defect(y_true, y_scores)

            logger.info(
                f"Epoch {epoch}/{args.epochs} | "
                f"Train Loss: {train_loss/len(train_loader):.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val AUROC: {val_metrics['auroc']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f}"
            )

            tracker.log_metrics({
                "train_loss": train_loss / len(train_loader),
                "train_acc":  train_acc,
                "val_auroc":  val_metrics["auroc"],
                "val_f1":     val_metrics["f1"],
            }, step=epoch)

            if val_metrics["auroc"] > best_auroc:
                best_auroc = val_metrics["auroc"]
                torch.save(model.state_dict(), best_path)
                logger.info(f"Best model saved — Val AUROC: {best_auroc:.4f}")

        logger.success(
            f"Training complete.\n"
            f"Best Val AUROC: {best_auroc:.4f}\n"
            f"Model saved -> {best_path}"
        )
        tracker.log_metrics({"best_val_auroc": best_auroc})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int,   default=16)
    args = parser.parse_args()
    train_resnet(args)