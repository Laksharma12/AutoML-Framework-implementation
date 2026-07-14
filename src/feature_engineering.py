"""Feature engineering helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FeatureEngineer:
    """Create simple, low-risk derived features."""

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Add deterministic engineered features."""

        transformed = dataframe.copy()

        if {"capital.gain", "capital.loss"}.issubset(transformed.columns):
            transformed["capital.net"] = transformed["capital.gain"] - transformed["capital.loss"]
            transformed["has_capital_activity"] = (
                (transformed["capital.gain"].fillna(0) > 0)
                | (transformed["capital.loss"].fillna(0) > 0)
            ).astype(int)

        if "hours.per.week" in transformed.columns:
            transformed["hours_per_week_squared"] = transformed["hours.per.week"] ** 2

        return transformed
