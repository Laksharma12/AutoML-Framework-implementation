"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


class DataLoader:
    """Load and inspect tabular datasets for the AutoML framework."""

    def __init__(self, csv_path: Path, target_column: str = "income") -> None:
        self.csv_path = csv_path
        self.target_column = target_column

    def load_data(self) -> pd.DataFrame:
        """Load the dataset and normalize missing values."""

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.csv_path}. Place adult.csv in datasets/."
            )

        dataframe = pd.read_csv(
            self.csv_path,
            na_values=["?"],
            skipinitialspace=True,
        )
        dataframe = dataframe.replace("?", pd.NA)
        return dataframe

    def display_shape(self, dataframe: pd.DataFrame) -> None:
        """Print the dataset shape."""

        print(f"Dataset shape: {dataframe.shape}")

    def show_dtypes(self, dataframe: pd.DataFrame) -> pd.Series:
        """Return and print the dataset dtypes."""

        dtypes = dataframe.dtypes
        print("Data types:")
        print(dtypes)
        return dtypes

    def detect_numerical_columns(self, dataframe: pd.DataFrame) -> List[str]:
        """Identify numerical feature columns."""

        return dataframe.select_dtypes(include=["number"]).columns.tolist()

    def detect_categorical_columns(self, dataframe: pd.DataFrame) -> List[str]:
        """Identify categorical feature columns."""

        categorical_columns = dataframe.select_dtypes(include=["object", "category"]).columns.tolist()
        return [column for column in categorical_columns if column != self.target_column]

    def detect_target_column(self, dataframe: pd.DataFrame) -> str:
        """Validate that the target column exists."""

        if self.target_column not in dataframe.columns:
            raise KeyError(f"Target column '{self.target_column}' not found in dataset.")
        return self.target_column

    def print_summary(self, dataframe: pd.DataFrame) -> None:
        """Print a concise dataset summary."""

        print("Dataset summary:")
        print(dataframe.describe(include="all").transpose())
        print("Missing values:")
        print(dataframe.isna().sum())
