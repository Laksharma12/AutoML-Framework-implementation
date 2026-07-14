"""Central configuration for the AutoML framework."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class AppConfig:
    """Application-wide paths and runtime settings."""

    project_root: Path = Path(__file__).resolve().parents[1]
    data_path: Path = project_root / "datasets" / "adult.csv"
    models_dir: Path = project_root / "models"
    reports_dir: Path = project_root / "reports"
    figures_dir: Path = reports_dir / "figures"
    plots_dir: Path = reports_dir / "plots"
    logs_dir: Path = project_root / "logs"
    results_path: Path = reports_dir / "model_results.csv"
    best_model_path: Path = models_dir / "best_model.pkl"
    target_column: str = "income"
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    tuning_iterations: int = 20
    scoring: str = "f1"


def model_configurations(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """Return default model hyperparameters and search spaces."""

    return {
        "Logistic Regression": {
            "estimator": "logistic_regression",
            "params": {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__penalty": ["l2"],
                "model__solver": ["lbfgs", "liblinear"],
            },
        },
        "Decision Tree": {
            "estimator": "decision_tree",
            "params": {
                "model__max_depth": [None, 5, 10, 15, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        },
        "Random Forest": {
            "estimator": "random_forest",
            "params": {
                "model__n_estimators": [100, 200, 300, 500],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
        },
        "Gradient Boosting": {
            "estimator": "gradient_boosting",
            "params": {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.7, 0.85, 1.0],
            },
        },
        "XGBoost": {
            "estimator": "xgboost",
            "params": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [3, 4, 6, 8],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
                "model__reg_alpha": [0.0, 0.1, 0.5],
                "model__reg_lambda": [1.0, 1.5, 2.0],
            },
        },
        "LightGBM": {
            "estimator": "lightgbm",
            "params": {
                "model__n_estimators": [100, 200, 300],
                "model__num_leaves": [15, 31, 63],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
                "model__min_child_samples": [10, 20, 30],
            },
        },
        "SVM": {
            "estimator": "svm",
            "params": {
                "model__C": [0.1, 1.0, 10.0],
                "model__kernel": ["rbf", "linear"],
                "model__gamma": ["scale", "auto"],
            },
        },
        "KNN": {
            "estimator": "knn",
            "params": {
                "model__n_neighbors": [3, 5, 7, 9, 11],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        },
        "Naive Bayes": {
            "estimator": "naive_bayes",
            "params": {},
        },
    }
