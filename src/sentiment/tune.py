import argparse
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import optuna
from optuna.integration import MLflowCallback
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from loguru import logger

from src.utils.logger import setup_logger
from src.utils.tracker import ExperimentTracker
from src.sentiment.dataset import create_dataloaders
from src.sentiment.model import SentimentClassifier


ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

def objective(trial: optuna.Trial, train_df: pd.DataFrame,
              val_df: pd.DataFrame, device: torch.device) -> float:

    lr = trial.suggest_float(
        "lr",
        low=1e-5,     
        high=5e-5,    
        log=True,     
    )
    dropout = trial.suggest_float(
        "dropout",
        low=0.1,
        high=0.5,
        step=0.1,     
    )
    batch_size = trial.suggest_categorical(
        "batch_size",
        choices=[32, 64],   
    )
    max_length = trial.suggest_categorical(
        "max_length",
        choices=[128, 256],
    )
    warmup_ratio = trial.suggest_float(
        "warmup_ratio",
        low=0.0,
        high=0.1,
        step=0.05,
    )
    epochs = 3
    weight_decay = trial.suggest_float(
        "weight_decay",
        low=0.0,
        high=0.1
    )
    max_grad_norm = 1.0

    logger.info(
        f"\nTrial {trial.number} — "
        f"lr={lr:.2e} | dropout={dropout} | "
        f"batch={batch_size} | max_len={max_length} | warmup={warmup_ratio} | epochs={epochs} | weight_decay={weight_decay:.4f} | max_grad_norm={max_grad_norm}"
    )

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    tune_size = min(20000, len(train_df))
    tune_df   = train_df.sample(n=tune_size, random_state=42).reset_index(drop=True)

    train_loader, val_loader, _ = create_dataloaders(
        tune_df, val_df, val_df,   
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=0,
    )

    model = SentimentClassifier(
        model_name="distilbert-base-uncased",
        num_classes=3,
        dropout_rate=dropout,
    ).to(device)

    labels  = tune_df["label_id"].values
    weights = compute_class_weight("balanced", classes=np.array([0,1,2]), y=labels)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float).to(device)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_epochs    = epochs
    total_steps = len(train_loader) * n_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()
    for batch_idx, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch   = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        if (batch_idx + 1) % 100 == 0:
            intermediate_loss = loss.item()
            trial.report(intermediate_loss, step=batch_idx)

            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at batch {batch_idx}")
                raise optuna.TrialPruned()

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch   = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            preds  = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    from sklearn.metrics import f1_score
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    logger.info(f"Trial {trial.number} — Val Macro F1: {macro_f1:.4f}")
    return macro_f1


def run_tuning(n_trials: int = 20) -> dict:
    setup_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Tuning on: {device}")

    if not torch.cuda.is_available():
        logger.warning(
            "No GPU detected. Tuning will be very slow on CPU.\n"
            "Strongly recommend running this on Google Colab with GPU."
        )

    processed_dir = Path("data/processed")
    train_df = pd.read_parquet(processed_dir / "reviews_train.parquet")
    val_df   = pd.read_parquet(processed_dir / "reviews_val.parquet")

    train_df = train_df.drop_duplicates(subset="clean_text").reset_index(drop=True)
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)}")

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = optuna.pruners.MedianPruner(
        n_startup_trials=5,    
        n_warmup_steps=50,     
    )

    study = optuna.create_study(
        study_name="sentiment_tuning_v2",
        direction="maximize",   
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///logs/optuna_sentiment_v2.db",  
        load_if_exists=True,    
    )

    logger.info(f"Starting Optuna study — {n_trials} trials")
    logger.info("Progress saved to logs/optuna_sentiment_v2.db")
    logger.info("You can safely interrupt and resume with load_if_exists=True")

    study.optimize(
        lambda trial: objective(trial, train_df, val_df, device),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_trial = study.best_trial
    logger.success(
        f"\nBest trial: #{best_trial.number}\n"
        f"  Val Macro F1: {best_trial.value:.4f}\n"
        f"  Hyperparameters:"
    )
    for k, v in best_trial.params.items():
        logger.info(f"    {k}: {v}")

    best_params = best_trial.params
    output_path = Path("configs/best_hparams.yaml")
    with open(output_path, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)
    logger.success(f"Best hyperparameters saved → {output_path}")

    try:
        import plotly
        Path("outputs").mkdir(exist_ok=True)

        importance_fig = optuna.visualization.plot_param_importances(study)
        importance_fig.write_image("outputs/optuna_param_importance.png")

        history_fig = optuna.visualization.plot_optimization_history(study)
        history_fig.write_image("outputs/optuna_history.png")

        logger.info("Charts saved → outputs/optuna_param_importance.png")
        logger.info("             → outputs/optuna_history.png")
    except Exception as e:
        logger.warning(f"Could not save charts: {e}")

    return best_params

def load_best_hparams(fallback: dict) -> dict:
    hparams_path = Path("configs/best_hparams.yaml")

    if hparams_path.exists():
        with open(hparams_path) as f:
            params = yaml.safe_load(f)
        logger.info(f"Loaded Optuna best hyperparameters from {hparams_path}")
        return params
    else:
        logger.info("No Optuna results found — using default hyperparameters")
        return fallback


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=20,
                        help="Number of Optuna trials (more = better but slower)")
    args = parser.parse_args()

    best = run_tuning(n_trials=args.n_trials)

    print("\n" + "="*50)
    print("NEXT STEP: Train final model with best hyperparameters")
    print("  python src/sentiment/train.py")
    print("  (train.py auto-loads configs/best_hparams.yaml)")
    print("="*50)