"""Hyperparameter tuning with RandomizedSearchCV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from sklearn.model_selection import RandomizedSearchCV

from src.config import model_configurations


@dataclass
class HyperparameterTuner:
    """Tune selected models with cross-validated random search."""

    cv: int = 5
    scoring: str = "f1"
    n_iter: int = 20
    random_state: int = 42

    def tune(
        self,
        models: Dict[str, object],
        X_train,
        y_train,
        model_names_to_tune: Optional[Iterable[str]] = None,
    ) -> Dict[str, object]:
        """Tune the configured subset of models and return the best estimators."""

        tuned_models = dict(models)
        configs = model_configurations(self.random_state)
        target_models = {"Random Forest", "XGBoost", "LightGBM"}
        if model_names_to_tune is not None:
            target_models = target_models.intersection(set(model_names_to_tune))

        for model_name in target_models:
            if model_name not in models or model_name not in configs:
                continue

            search_space = configs[model_name]["params"]
            if not search_space:
                continue

            search = RandomizedSearchCV(
                estimator=models[model_name],
                param_distributions=search_space,
                n_iter=self.n_iter,
                scoring=self.scoring,
                cv=self.cv,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0,
            )
            search.fit(X_train, y_train)
            tuned_models[model_name] = search.best_estimator_

        return tuned_models
