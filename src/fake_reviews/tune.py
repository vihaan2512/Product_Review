import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from loguru import logger

from src.utils.logger import setup_logger
from src.fake_reviews.features import LinguisticFeatureExtractor


def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    n_estimators = trial.suggest_int("n_estimators_gb", 50, 300, step=50)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
    max_depth = trial.suggest_int("max_depth", 3, 8)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 20)

    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    
    return float(np.mean(scores))


def load_best_fake_hparams(fallback: dict) -> dict:
    hparams_path = Path("configs/best_fake_hparams.yaml")
    if hparams_path.exists():
        with open(hparams_path) as f:
            params = yaml.safe_load(f)
        logger.info(f"Loaded Optuna best fake detector hyperparameters from {hparams_path}")
        return params
    else:
        logger.info("No Optuna results found for fake review detector - using default hyperparameters")
        return fallback


def run_tuning(csv_path: str, n_trials: int = 20) -> dict:
    setup_logger()
    logger.info("Starting hyperparameter tuning for Gradient Boosting fake review classifier...")

    df = pd.read_csv(csv_path)
    
    text_col = next((c for c in df.columns if any(k in c.lower() for k in ["text", "review"])), None)
    label_col = next((c for c in df.columns if any(k in c.lower() for k in ["label", "deceptive", "fake"])), None)

    if not text_col or not label_col:
        raise ValueError(f"Could not find text/label columns. Available: {df.columns.tolist()}")

    df = df.rename(columns={text_col: "clean_text", label_col: "true_label"})
    if df["true_label"].dtype == object:
        df["true_label"] = df["true_label"].map(
            {"deceptive": 1, "truthful": 0, "fake": 1, "real": 0}
        )

    df["true_label"] = df["true_label"].astype(int)

    logger.info("Extracting features from deceptive reviews...")
    fe = LinguisticFeatureExtractor()
    features_df = fe.extract_batch(df["clean_text"].tolist())
    X = features_df.values.astype(np.float32)
    y = df["true_label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        study_name="fake_reviews_gb_tuning",
        direction="maximize",
        sampler=sampler
    )

    study.optimize(
        lambda trial: objective(trial, X_scaled, y),
        n_trials=n_trials,
        show_progress_bar=True
    )

    best_params = study.best_trial.params
    logger.success(f"Best trial parameters: {best_params}")

    hparams_path = Path("configs/best_fake_hparams.yaml")
    hparams_path.parent.mkdir(exist_ok=True)
    with open(hparams_path, "w") as f:
        yaml.dump(best_params, f)
    logger.info(f"Saved best parameters to {hparams_path}")

    return best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="data/raw/amazon_fake/deceptive_reviews.csv")
    parser.add_argument("--n_trials", type=int, default=20)
    args = parser.parse_args()

    run_tuning(args.csv_path, n_trials=args.n_trials)