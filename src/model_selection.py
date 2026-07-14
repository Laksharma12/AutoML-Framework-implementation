"""Model factory and base estimator creation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None


@dataclass
class ModelSelector:
    """Construct model pipelines for classification tasks."""

    random_state: int = 42

    def build_models(self, preprocessor) -> Dict[str, Pipeline]:
        """Create model pipelines with a shared preprocessor."""

        models: Dict[str, Pipeline] = {
            "Logistic Regression": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=self.random_state,
                        ),
                    ),
                ]
            ),
            "Decision Tree": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        DecisionTreeClassifier(
                            class_weight="balanced",
                            random_state=self.random_state,
                        ),
                    ),
                ]
            ),
            "Random Forest": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=200,
                            class_weight="balanced",
                            random_state=self.random_state,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "Gradient Boosting": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", GradientBoostingClassifier(random_state=self.random_state)),
                ]
            ),
            "SVM": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        SVC(
                            probability=True,
                            class_weight="balanced",
                            random_state=self.random_state,
                        ),
                    ),
                ]
            ),
            "KNN": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", KNeighborsClassifier()),
                ]
            ),
            "Naive Bayes": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", GaussianNB()),
                ]
            ),
        }

        if XGBClassifier is not None:
            models["XGBoost"] = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        XGBClassifier(
                            objective="binary:logistic",
                            eval_metric="logloss",
                            random_state=self.random_state,
                            n_estimators=200,
                            learning_rate=0.05,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )

        if LGBMClassifier is not None:
            models["LightGBM"] = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        LGBMClassifier(
                            random_state=self.random_state,
                            n_estimators=200,
                            learning_rate=0.05,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )

        return models
