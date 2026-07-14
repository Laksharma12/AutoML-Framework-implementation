"""Shared utility helpers for the AutoML framework."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not exist."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(log_file: Path) -> logging.Logger:
    """Configure application logging."""

    ensure_directory(log_file.parent)
    logger = logging.getLogger("automl")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def detect_problem_type(target: pd.Series) -> str:
    """Infer whether the task is classification or regression."""

    non_null = target.dropna()
    if non_null.empty:
        return "classification"

    unique_values = non_null.nunique()
    if pd.api.types.is_numeric_dtype(non_null) and unique_values > 20:
        return "regression"
    return "classification"


def safe_predict_proba(model, features):
    """Return probability estimates when the estimator supports them."""

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities.ravel()

    if hasattr(model, "decision_function"):
        decision_scores = model.decision_function(features)
        if decision_scores.ndim > 1:
            decision_scores = decision_scores[:, 0]
        return 1.0 / (1.0 + np.exp(-decision_scores))

    return None


def classification_metrics(y_true, y_pred, y_score=None) -> Dict[str, float]:
    """Compute the classification metrics required by the framework."""

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_score is not None:
        try:
            metrics["ROC AUC"] = roc_auc_score(y_true, y_score)
        except ValueError:
            metrics["ROC AUC"] = np.nan
    else:
        metrics["ROC AUC"] = np.nan

    return metrics


def get_feature_names(preprocessor):
    """Extract feature names from a fitted ColumnTransformer."""

    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        return np.array([])


def save_dataframe(dataframe: pd.DataFrame, path: Path) -> None:
    """Persist a dataframe as CSV."""

    ensure_directory(path.parent)
    dataframe.to_csv(path, index=False)
