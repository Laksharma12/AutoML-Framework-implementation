"""Entry point for the AutoML framework."""

from __future__ import annotations

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import AppConfig
from src.data_loader import DataLoader
from src.eda import EDA
from src.evaluation import Evaluator
from src.feature_engineering import FeatureEngineer
from src.hyperparameter_tuning import HyperparameterTuner
from src.model_selection import ModelSelector
from src.preprocessing import build_preprocessor
from src.save_model import ModelSaver
from src.utils import detect_problem_type, ensure_directory, save_dataframe, setup_logging
from src.visualization import Visualizer


def print_banner() -> None:
    """Print the framework banner."""

    print("==================================")
    print("AutoML Framework Started")
    print("==================================")


def main() -> None:
    """Run the complete AutoML pipeline."""

    config = AppConfig()
    for path in [config.models_dir, config.reports_dir, config.figures_dir, config.plots_dir, config.logs_dir]:
        ensure_directory(path)

    logger = setup_logging(config.logs_dir / "app.log")
    print_banner()

    loader = DataLoader(config.data_path, config.target_column)
    dataframe = loader.load_data()
    logger.info("Dataset Loaded")
    print("Dataset Loaded")

    loader.display_shape(dataframe)
    loader.show_dtypes(dataframe)
    loader.detect_target_column(dataframe)
    loader.print_summary(dataframe)

    problem_type = detect_problem_type(dataframe[config.target_column])
    if problem_type != "classification":
        raise ValueError("This implementation currently focuses on classification tasks.")

    feature_engineer = FeatureEngineer()
    dataframe = feature_engineer.transform(dataframe)

    numerical_features = loader.detect_numerical_columns(dataframe)
    categorical_features = loader.detect_categorical_columns(dataframe)

    eda = EDA(figures_dir=config.figures_dir)
    eda.run(dataframe, config.target_column, numerical_features, categorical_features)
    logger.info("EDA Completed")
    print("EDA Completed")

    target = dataframe[config.target_column].astype(str).str.strip().str.replace(".", "", regex=False)
    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(target)

    features = dataframe.drop(columns=[config.target_column])
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        encoded_target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=encoded_target,
    )

    preprocessor = build_preprocessor(numerical_features, categorical_features)
    print("Preprocessing Completed")
    logger.info("Preprocessing Completed")

    selector = ModelSelector(random_state=config.random_state)
    base_models = selector.build_models(preprocessor)

    print("Training Models...")
    logger.info("Training Started")
    tuner = HyperparameterTuner(
        cv=config.cv_folds,
        scoring=config.scoring,
        n_iter=config.tuning_iterations,
        random_state=config.random_state,
    )
    tuned_models = tuner.tune(base_models, X_train, y_train)

    print("Evaluating Models...")
    evaluator = Evaluator()
    target_names = list(label_encoder.classes_)
    results_df, fitted_results = evaluator.evaluate_models(
        tuned_models,
        X_train,
        X_test,
        y_train,
        y_test,
        target_names=target_names,
    )
    logger.info("Training Finished")

    print("Selecting Best Model...")
    best_row = results_df.iloc[0]
    best_model_name = best_row["Model Name"]
    best_result = fitted_results[best_model_name]

    print("Saving Model...")
    saver = ModelSaver()
    saver.save(best_result.estimator, config.best_model_path)
    logger.info("Model Saved")

    print("Generating Report...")
    save_dataframe(results_df, config.results_path)

    visualizer = Visualizer(output_dir=config.plots_dir)
    visualizer.plot_model_comparison(results_df)
    visualizer.plot_confusion_matrix(y_test, best_result.y_pred, best_model_name)
    visualizer.plot_roc_curve(y_test, best_result.y_score, best_model_name)
    visualizer.plot_precision_recall_curve(y_test, best_result.y_score, best_model_name)
    visualizer.plot_feature_importance(best_result.estimator)
    visualizer.plot_learning_curve(best_result.estimator, X_train, y_train, best_model_name)

    print("Done!")
    print("==================================")
    print("Best Model:")
    print(f"Accuracy: {best_row['Accuracy']:.4f}")
    print(f"F1 Score: {best_row['F1']:.4f}")
    print(f"ROC AUC: {best_row['ROC AUC']:.4f}")
    print("==================================")


if __name__ == "__main__":
    main()
