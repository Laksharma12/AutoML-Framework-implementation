"""Model persistence utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib

from src.utils import ensure_directory


@dataclass
class ModelSaver:
    """Save and load trained estimators."""

    def save(self, model, path: Path) -> Path:
        """Persist a model to disk with joblib."""

        ensure_directory(path.parent)
        joblib.dump(model, path)
        return path

    def load(self, path: Path):
        """Load a persisted model from disk."""

        return joblib.load(path)
