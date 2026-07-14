"""Model evaluation and result aggregation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.utils import classification_metrics, safe_predict_proba


@dataclass
class EvaluationResult:
    """Container for a fitted model and its metrics."""

    model_name: str
    estimator: object
    metrics: Dict[str, float]
    training_time: float
    confusion_matrix: np.ndarray
    classification_report: str
    y_pred: np.ndarray
    y_score: np.ndarray | None


class Evaluator:
    """Train models, evaluate them, and create a sortable results table."""

    def evaluate_models(
        self,
        models: Dict[str, object],
        X_train,
        X_test,
        y_train,
        y_test,
        target_names: List[str] | None = None,
    ) -> Tuple[pd.DataFrame, Dict[str, EvaluationResult]]:
        """Fit all models and collect the required metrics."""

        rows: List[Dict[str, object]] = []
        results: Dict[str, EvaluationResult] = {}

        for model_name, estimator in models.items():
            start_time = time.time()
            estimator.fit(X_train, y_train)
            training_time = time.time() - start_time

            y_pred = estimator.predict(X_test)
            y_score = safe_predict_proba(estimator, X_test)
            metrics = classification_metrics(y_test, y_pred, y_score)
            matrix = confusion_matrix(y_test, y_pred)
            report = classification_report(
                y_test,
                y_pred,
                target_names=target_names,
                zero_division=0,
            )

            row = {
                "Model Name": model_name,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1"],
                "ROC AUC": metrics["ROC AUC"],
                "Training Time": training_time,
            }
            rows.append(row)

            results[model_name] = EvaluationResult(
                model_name=model_name,
                estimator=estimator,
                metrics=metrics,
                training_time=training_time,
                confusion_matrix=matrix,
                classification_report=report,
                y_pred=y_pred,
                y_score=y_score,
            )

        results_df = pd.DataFrame(rows).sort_values(by="F1", ascending=False).reset_index(drop=True)
        return results_df, results
