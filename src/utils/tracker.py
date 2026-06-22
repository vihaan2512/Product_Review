import os
from pathlib import Path
from contextlib import contextmanager
from typing import Any

import mlflow
from loguru import logger

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


class ExperimentTracker:

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "logs/mlruns",
    ):
        Path(tracking_uri).mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        logger.info(f"MLflow tracking -> {tracking_uri}  |  experiment: {experiment_name}")

    @contextmanager
    def start_run(self, run_name: str = None, tags: dict = None):
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            logger.info(f"MLflow run started: {run.info.run_id}  ({run_name})")
            yield run
            logger.info(f"MLflow run ended: {run.info.run_id}")

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)
        logger.debug(f"Logged params: {params}")

    def log_metrics(self, metrics: dict, step: int = None) -> None:
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged metrics (step={step}): {metrics}")

    def log_artifact(self, path: str) -> None:
        if Path(path).exists():
            mlflow.log_artifact(path)
            logger.debug(f"Logged artifact: {path}")
        else:
            logger.warning(f"Artifact not found, skipping: {path}")

    def log_model_summary(self, model, framework: str = "pytorch") -> None:
        try:
            if framework == "pytorch":
                total = sum(p.numel() for p in model.parameters())
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                mlflow.log_metrics({
                    "total_params": total,
                    "trainable_params": trainable,
                })
                logger.info(f"Model params — total: {total:,} | trainable: {trainable:,}")
        except Exception as e:
            logger.warning(f"Could not log model summary: {e}")

    @staticmethod
    def get_best_run(experiment_name: str, metric: str, mode: str = "max") -> dict:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"Experiment '{experiment_name}' not found.")
            return {}

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'DESC' if mode == 'max' else 'ASC'}"],
        )
        if runs.empty:
            return {}

        best = runs.iloc[0]
        logger.info(
            f"Best run — ID: {best['run_id']} | "
            f"{metric}: {best.get(f'metrics.{metric}', 'N/A'):.4f}"
        )
        return best.to_dict()