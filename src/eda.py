"""Automatic exploratory data analysis for the dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils import ensure_directory


@dataclass
class EDA:
    """Generate summary statistics and diagnostic figures."""

    figures_dir: Path

    def __post_init__(self) -> None:
        ensure_directory(self.figures_dir)
        sns.set_style("whitegrid")

    def run(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        numerical_columns: List[str],
        categorical_columns: List[str],
    ) -> None:
        """Execute the full EDA workflow."""

        self.save_statistics(dataframe)
        self.save_missing_values_report(dataframe)
        self.plot_class_distribution(dataframe, target_column)
        self.plot_correlation_matrix(dataframe, numerical_columns)
        self.plot_histograms(dataframe, numerical_columns)
        self.plot_boxplots(dataframe, numerical_columns)
        self.plot_countplots(dataframe, categorical_columns, target_column)
        self.plot_pairplot(dataframe, numerical_columns, target_column)

    def save_statistics(self, dataframe: pd.DataFrame) -> Path:
        """Persist summary statistics for the report."""

        statistics_path = self.figures_dir / "dataset_statistics.csv"
        dataframe.describe(include="all").transpose().to_csv(statistics_path)
        return statistics_path

    def save_missing_values_report(self, dataframe: pd.DataFrame) -> Path:
        """Persist a missing value report."""

        missing_report = dataframe.isna().sum().reset_index()
        missing_report.columns = ["feature", "missing_values"]
        report_path = self.figures_dir / "missing_values_report.csv"
        missing_report.to_csv(report_path, index=False)
        return report_path

    def plot_class_distribution(self, dataframe: pd.DataFrame, target_column: str) -> Path:
        """Plot the target class distribution."""

        figure_path = self.figures_dir / "class_distribution.png"
        plt.figure(figsize=(6, 4))
        sns.countplot(data=dataframe, x=target_column, palette="Set2")
        plt.title("Class Distribution")
        plt.tight_layout()
        plt.savefig(figure_path, dpi=300)
        plt.close()
        return figure_path

    def plot_correlation_matrix(self, dataframe: pd.DataFrame, numerical_columns: List[str]) -> Path | None:
        """Plot the correlation heatmap for numerical features."""

        if not numerical_columns:
            return None

        figure_path = self.figures_dir / "correlation_matrix.png"
        plt.figure(figsize=(10, 8))
        sns.heatmap(dataframe[numerical_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(figure_path, dpi=300)
        plt.close()
        return figure_path

    def plot_histograms(self, dataframe: pd.DataFrame, numerical_columns: List[str]) -> None:
        """Plot histograms for numerical features."""

        for column in numerical_columns:
            figure_path = self.figures_dir / f"hist_{column}.png"
            plt.figure(figsize=(6, 4))
            sns.histplot(dataframe[column].dropna(), kde=True, bins=30, color="steelblue")
            plt.title(f"Histogram: {column}")
            plt.tight_layout()
            plt.savefig(figure_path, dpi=300)
            plt.close()

    def plot_boxplots(self, dataframe: pd.DataFrame, numerical_columns: List[str]) -> None:
        """Plot boxplots for numerical features."""

        for column in numerical_columns:
            figure_path = self.figures_dir / f"box_{column}.png"
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=dataframe[column], color="lightcoral")
            plt.title(f"Boxplot: {column}")
            plt.tight_layout()
            plt.savefig(figure_path, dpi=300)
            plt.close()

    def plot_countplots(self, dataframe: pd.DataFrame, categorical_columns: List[str], target_column: str) -> None:
        """Plot count plots for categorical features."""

        for column in categorical_columns:
            figure_path = self.figures_dir / f"count_{column}.png"
            plt.figure(figsize=(8, 4))
            sns.countplot(data=dataframe, x=column, hue=target_column)
            plt.xticks(rotation=45, ha="right")
            plt.title(f"Count Plot: {column}")
            plt.tight_layout()
            plt.savefig(figure_path, dpi=300)
            plt.close()

    def plot_pairplot(self, dataframe: pd.DataFrame, numerical_columns: List[str], target_column: str) -> Path | None:
        """Plot a pairplot for a small numerical subset."""

        if len(numerical_columns) < 2:
            return None

        selected_columns = numerical_columns[:5]
        plot_data = dataframe[selected_columns + [target_column]].dropna()
        if plot_data.empty:
            return None

        figure_path = self.figures_dir / "pairplot.png"
        sample_size = min(1000, len(plot_data))
        if sample_size < len(plot_data):
            plot_data = plot_data.sample(n=sample_size, random_state=42)
        pairplot = sns.pairplot(plot_data, hue=target_column, corner=True)
        pairplot.fig.suptitle("Pairplot", y=1.02)
        pairplot.fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close("all")
        return figure_path
