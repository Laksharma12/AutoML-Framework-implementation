"""Visualization utilities for model diagnostics and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix
from sklearn.model_selection import learning_curve

from src.utils import ensure_directory, get_feature_names


@dataclass
class Visualizer:
    """Create and save evaluation plots."""

    output_dir: Path

    def __post_init__(self) -> None:
        ensure_directory(self.output_dir)
        sns.set_style("whitegrid")

    def plot_confusion_matrix(self, y_true, y_pred, model_name: str) -> Path:
        """Save a confusion matrix figure."""

        figure_path = self.output_dir / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(figure_path, dpi=300)
        plt.close()
        return figure_path

    def plot_roc_curve(self, y_true, y_score, model_name: str) -> Optional[Path]:
        """Save a ROC curve figure when probabilities are available."""

        if y_score is None:
            return None

        figure_path = self.output_dir / f"{model_name.lower().replace(' ', '_')}_roc_curve.png"
        plt.figure(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_true, y_score)
        plt.title(f"ROC Curve - {model_name}")
        plt.tight_layout()
        plt.savefig(figure_path, dpi=300)
        plt.close()
        return figure_path

    def plot_precision_recall_curve(self, y_true, y_score, model_name: str) -> Optional[Path]:
        """Save a precision-recall curve figure when probabilities are available."""

        if y_score is None:
            return None

        figure_path = self.output_dir / f"{model_name.lower().replace(' ', '_')}_precision_recall_curve.png"
        plt.figure(figsize=(6, 5))
        PrecisionRecallDisplay.from_predictions(y_true, y_score)
        plt.title(f"Precision-Recall Curve - {model_name}")
        plt.tight_layout()
        plt.savefig(figure_path, dpi=300)
        plt.close()
        return figure_path

    def plot_model_comparison(self, results_df: pd.DataFrame) -> Path:
        """Save a comparison chart for the candidate models."""

        figure_path = self.output_dir / "model_comparison.png"
        plt.figure(figsize=(12, 6))
        ordered = results_df.sort_values("F1", ascending=True)
        sns.barplot(data=ordered, x="F1", y="Model Name", palette="viridis")
        plt.title("Model Comparison by F1 Score")
        plt.xlabel("F1 Score")
        plt.ylabel("Model Name")
        plt.tight_layout()
        plt.savefig(figure_path, dpi=300)
        plt.close()
        return figure_path

    def plot_feature_importance(self, pipeline, top_n: int = 15) -> Optional[Path]:
        """Plot feature importance or coefficient magnitude for fitted models."""

        if not hasattr(pipeline, "named_steps"):
            return None

        model = pipeline.named_steps.get("model")
        preprocessor = pipeline.named_steps.get("preprocessor")
        feature_names = get_feature_names(preprocessor)
        if feature_names.size == 0:
            return None

        values = None
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        elif hasattr(model, "coef_"):
            values = np.abs(model.coef_).ravel()

        if values is None:
            return None

        importance_frame = pd.DataFrame({"feature": feature_names, "importance": values})
        importance_frame = importance_frame.sort_values("importance", ascending=False).head(top_n)

        figure_path = self.output_dir / "feature_importance.png"
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_frame, x="importance", y="feature", palette="magma")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(figure_path, dpi=300)
        plt.close()
        return figure_path

    def plot_learning_curve(self, estimator, X_train, y_train, model_name: str) -> Optional[Path]:
        """Save a learning curve for the selected estimator."""

        try:
            train_sizes, train_scores, validation_scores = learning_curve(
                estimator,
                X_train,
                y_train,
                cv=5,
                n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 5),
                scoring="f1",
            )
        except Exception:
            return None

        figure_path = self.output_dir / f"{model_name.lower().replace(' ', '_')}_learning_curve.png"
        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, train_scores.mean(axis=1), label="Training score")
        plt.plot(train_sizes, validation_scores.mean(axis=1), label="Cross-validation score")
        plt.fill_between(
            train_sizes,
            train_scores.mean(axis=1) - train_scores.std(axis=1),
            train_scores.mean(axis=1) + train_scores.std(axis=1),
            alpha=0.2,
        )
        plt.fill_between(
            train_sizes,
            validation_scores.mean(axis=1) - validation_scores.std(axis=1),
            validation_scores.mean(axis=1) + validation_scores.std(axis=1),
            alpha=0.2,
        )
        plt.title(f"Learning Curve - {model_name}")
        plt.xlabel("Training Samples")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_path, dpi=300)
        plt.close()
        return figure_path
