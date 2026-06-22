import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from loguru import logger

from src.utils.logger import setup_logger
from src.utils.tracker import ExperimentTracker
from src.utils.metrics import evaluate_sentiment, plot_confusion_matrix
from src.sentiment.dataset import create_dataloaders
from src.sentiment.model import SentimentClassifier


ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning(
            "No GPU found — using CPU. Training will be slow (~2-3 hrs per epoch). "
            "Consider using Google Colab for free GPU."
        )
    return device


def compute_class_weights(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    labels = train_df["label_id"].values
    classes = np.unique(labels)
    
    # Calculate balanced weights: total_samples / (n_classes * class_samples)
    calculated_weights = compute_class_weight("balanced", classes=classes, y=labels)
    
    logger.info(
        f"Dynamic balanced class weights — "
        f"negative: {calculated_weights[0]:.3f} | "
        f"neutral: {calculated_weights[1]:.3f} | "
        f"positive: {calculated_weights[2]:.3f}"
    )
    return torch.tensor(calculated_weights, dtype=torch.float).to(device)


class FocalLoss(nn.Module):
    
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(inputs, dim=-1)
        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        probs = torch.exp(target_log_probs)
        
        focal_weight = (1.0 - probs) ** self.gamma
        loss = -focal_weight * target_log_probs
        
        if self.alpha is not None:
            alpha_factor = self.alpha.gather(dim=0, index=targets)
            loss = loss * alpha_factor
            
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def train_one_epoch(
    model: SentimentClassifier,
    loader,
    optimizer,
    scheduler,
    criterion,
    device: torch.device,
    epoch: int,
) -> dict:
    
    model.train()  
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)

        loss = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Acc: {correct/total:.4f}"
            )

    return {
        "train_loss": total_loss / len(loader),
        "train_acc":  correct / total,
    }


def evaluate(
    model: SentimentClassifier,
    loader,
    criterion,
    device: torch.device,
) -> dict:
   
    model.eval()   
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():   
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    pred_labels  = [ID2LABEL[p] for p in all_preds]
    true_labels  = [ID2LABEL[l] for l in all_labels]

    metrics = evaluate_sentiment(true_labels, pred_labels)
    metrics["val_loss"] = total_loss / len(loader)

    return metrics, all_preds, all_labels


def train(args) -> None:
    setup_logger()
    device = get_device()

    processed_dir = Path("data/processed")
    logger.info("Loading processed data...")

    train_df = pd.read_parquet(processed_dir / "reviews_train.parquet")
    val_df   = pd.read_parquet(processed_dir / "reviews_val.parquet")
    test_df  = pd.read_parquet(processed_dir / "reviews_test.parquet")

    before = len(train_df)
    train_df = train_df.drop_duplicates(subset="clean_text").reset_index(drop=True)
    logger.info(f"Removed {before - len(train_df)} duplicate reviews from training set")

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    logger.info(f"Train label distribution:\n{train_df['label'].value_counts()}")

    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    model = SentimentClassifier(
        model_name=args.model_name,
        num_classes=3,
        dropout_rate=args.dropout,
    ).to(device)

    params = model.get_param_count()
    logger.info(f"Model params — total: {params['total']:,} | trainable: {params['trainable']:,}")

    class_weights = compute_class_weights(train_df, device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    optimizer = AdamW([
        {"params": model.bert.parameters(),       "lr": args.lr},
        {"params": model.classifier.parameters(), "lr": args.lr * 10},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    warmup_ratio = getattr(args, "warmup_ratio", 0.1)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    logger.info(f"Total training steps: {total_steps} | Warmup steps: {warmup_steps} (ratio: {warmup_ratio})")

    tracker = ExperimentTracker("sentiment_module")
    run_name = f"distilbert_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}"

    with tracker.start_run(run_name=run_name):
        tracker.log_params({
            "model_name":  args.model_name,
            "epochs":      args.epochs,
            "batch_size":  args.batch_size,
            "max_length":  args.max_length,
            "learning_rate": args.lr,
            "dropout":     args.dropout,
            "warmup_ratio": getattr(args, "warmup_ratio", 0.1),
            "train_size":  len(train_df),
            "val_size":    len(val_df),
        })
        tracker.log_model_summary(model)

        best_val_f1 = 0.0
        patience = 4
        epochs_no_improve = 0
        best_model_path = Path("models/sentiment_best.pt")
        best_model_path.parent.mkdir(exist_ok=True)

        for epoch in range(1, args.epochs + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"EPOCH {epoch}/{args.epochs}")
            logger.info(f"{'='*50}")

            start_time = time.time()

            train_metrics = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                criterion, device, epoch
            )

            val_metrics, val_preds, val_labels = evaluate(
                model, val_loader, criterion, device
            )

            elapsed = time.time() - start_time
            logger.info(
                f"Epoch {epoch} complete in {elapsed:.0f}s | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Macro F1: {val_metrics['macro_f1']:.4f}"
            )

            tracker.log_metrics({
                "train_loss":  train_metrics["train_loss"],
                "train_acc":   train_metrics["train_acc"],
                "val_loss":    val_metrics["val_loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }, step=epoch)

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best model saved — Val Macro F1: {best_val_f1:.4f}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                logger.info(f"No validation improvement for {epochs_no_improve} consecutive epoch(s).")
                if epochs_no_improve >= patience:
                    logger.warning(f"Early stopping triggered! Training stopped after {epoch} epochs.")
                    break

        logger.info("\nLoading best model for final test evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        test_metrics, test_preds, test_labels = evaluate(
            model, test_loader, criterion, device
        )

        logger.info(
            f"\nFINAL TEST RESULTS:\n"
            f"  Accuracy:    {test_metrics['accuracy']:.4f}\n"
            f"  Macro F1:    {test_metrics['macro_f1']:.4f}\n"
            f"  Weighted F1: {test_metrics['weighted_f1']:.4f}"
        )

        tracker.log_metrics({
            "test_accuracy":    test_metrics["accuracy"],
            "test_macro_f1":    test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
        })

        logger.success(f"Training complete. Best val Macro F1: {best_val_f1:.4f}")
        logger.info("View results: mlflow ui --backend-store-uri logs/mlruns")


def get_default_hparams(epochs: int = 10, dropout: float = 0.2) -> dict:
    from src.sentiment.tune import load_best_hparams
    return load_best_hparams({
        "lr": 2e-5,
        "dropout": dropout,
        "batch_size": 32,
        "max_length": 256,
        "warmup_ratio": 0.1,
    })


def train_with_best_hparams(epochs: int) -> None:
    import argparse
    best = get_default_hparams()

    args = argparse.Namespace(
        model_name   = "distilbert-base-uncased",
        epochs       = epochs,
        lr           = best.get("lr"),
        dropout      = best.get("dropout"),
        batch_size   = best.get("batch_size"),
        max_length   = best.get("max_length"),
        warmup_ratio = best.get("warmup_ratio"),
        weight_decay  = best.get("weight_decay"),
        max_grad_norm = best.get("max_grad_norm", 1.0),
    )
    logger.info(f"Training with Optuna best params: {vars(args)}")
    train(args)


if __name__ == "__main__":
    best_hparams = get_default_hparams()

    parser = argparse.ArgumentParser(description="Train sentiment classifier")
    parser.add_argument("--model_name",   default="distilbert-base-uncased")
    parser.add_argument("--epochs",       type=int,   default=best_hparams.get("epochs", 10))
    parser.add_argument("--batch_size",   type=int,   default=best_hparams.get("batch_size", 64))
    parser.add_argument("--max_length",   type=int,   default=best_hparams.get("max_length"))
    parser.add_argument("--lr",           type=float, default=best_hparams.get("lr"))
    parser.add_argument("--dropout",      type=float, default=best_hparams.get("dropout"))
    parser.add_argument("--warmup_ratio", type=float, default=best_hparams.get("warmup_ratio"))
    parser.add_argument("--weight_decay", type=float, default=best_hparams.get("weight_decay"))
    parser.add_argument("--max_grad_norm", type=float, default=best_hparams.get("max_grad_norm", 1.0))
    args = parser.parse_args()

    logger.info(f"Training config: {vars(args)}")
    train(args)