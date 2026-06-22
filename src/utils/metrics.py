import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score,
)
from loguru import logger


# ─────────────────────────────────────────────────────────────
# Module 1: Sentiment Analysis
# ─────────────────────────────────────────────────────────────

def evaluate_sentiment(y_true: list, y_pred: list, labels: list = None) -> dict:

    if labels is None:
        labels = ["negative", "neutral", "positive"]

    metrics = {
        "accuracy":       accuracy_score(y_true, y_pred),
        "macro_f1":       f1_score(y_true, y_pred, average="macro"),
        "weighted_f1":    f1_score(y_true, y_pred, average="weighted"),
        "macro_precision":precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall":   recall_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class_f1":   dict(zip(
            labels,
            f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        )),
    }

    logger.info("=== Module 1 — Sentiment Analysis ===")
    logger.info(f"Accuracy:       {metrics['accuracy']:.4f}")
    logger.info(f"Macro F1:       {metrics['macro_f1']:.4f}")
    logger.info(f"Weighted F1:    {metrics['weighted_f1']:.4f}")
    logger.info(f"Per-class F1:   {metrics['per_class_f1']}")
    logger.info(f"\n{classification_report(y_true, y_pred, target_names=labels)}")

    return metrics


def plot_confusion_matrix(y_true: list, y_pred: list,
                          labels: list, title: str = "",
                          save_path: str = None) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title or "Confusion Matrix", fontsize=13)

    for ax, data, fmt, t in zip(
        axes, [cm, cm_norm], ["d", ".2f"],
        ["Raw counts", "Normalised (recall)"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
            linewidths=0.5, linecolor="white",
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(t)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
# Module 2: Defect Detection
# ─────────────────────────────────────────────────────────────

def evaluate_defect(y_true: list, y_pred_proba: list,
                    threshold: float = 0.5) -> dict:
    y_pred = (np.array(y_pred_proba) >= threshold).astype(int)

    metrics = {
        "accuracy":   accuracy_score(y_true, y_pred),
        "precision":  precision_score(y_true, y_pred, zero_division=0),
        "recall":     recall_score(y_true, y_pred, zero_division=0),
        "f1":         f1_score(y_true, y_pred, zero_division=0),
        "auroc":      roc_auc_score(y_true, y_pred_proba),
        "avg_precision": average_precision_score(y_true, y_pred_proba),
    }

    logger.info("=== Module 2 — Defect Detection ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    return metrics


# ─────────────────────────────────────────────────────────────
# Module 3: Fake Review Detection
# ─────────────────────────────────────────────────────────────

def evaluate_fake_reviews(y_true: list, y_scores: list,
                          threshold: float = 0.5) -> dict:
    y_pred = (np.array(y_scores) >= threshold).astype(int)

    k_values = [10, 50, 100]
    precision_at_k = {}
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_arr = np.array(y_true)
    for k in k_values:
        top_k = sorted_indices[:k]
        precision_at_k[f"precision@{k}"] = y_true_arr[top_k].mean()

    metrics = {
        "accuracy":   accuracy_score(y_true, y_pred),
        "precision":  precision_score(y_true, y_pred, zero_division=0),
        "recall":     recall_score(y_true, y_pred, zero_division=0),
        "f1":         f1_score(y_true, y_pred, zero_division=0),
        "auroc":      roc_auc_score(y_true, y_scores),
        **precision_at_k,
    }

    logger.info("=== Module 3 — Fake Review Detection ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    return metrics


# ─────────────────────────────────────────────────────────────
# Module 4: Aspect-Based Sentiment (ABSA)
# ─────────────────────────────────────────────────────────────

def evaluate_absa(y_true_per_aspect: dict, y_pred_per_aspect: dict) -> dict:
    results = {}
    for aspect in y_true_per_aspect:
        if aspect not in y_pred_per_aspect:
            continue
        f1 = f1_score(
            y_true_per_aspect[aspect], y_pred_per_aspect[aspect],
            average="macro", zero_division=0
        )
        results[aspect] = f1

    mean_f1 = np.mean(list(results.values())) if results else 0.0
    results["mean_aspect_f1"] = mean_f1

    logger.info("=== Module 4 — Aspect-Based Sentiment ===")
    for aspect, score in results.items():
        logger.info(f"  {aspect}: {score:.4f}")

    return results


# ─────────────────────────────────────────────────────────────
# Module 5: Quality Score
# ─────────────────────────────────────────────────────────────

def evaluate_quality_score(y_true_scores: list, y_pred_scores: list) -> dict:
    from scipy.stats import spearmanr, pearsonr

    y_true = np.array(y_true_scores)
    y_pred = np.array(y_pred_scores)

    spearman_corr, spearman_p = spearmanr(y_true, y_pred)
    pearson_corr,  pearson_p  = pearsonr(y_true, y_pred)
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    metrics = {
        "spearman_corr": spearman_corr,
        "spearman_p":    spearman_p,
        "pearson_corr":  pearson_corr,
        "mae":           mae,
        "rmse":          rmse,
    }

    logger.info("=== Module 5 — Quality Score ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    return metrics