# AutoML Framework

A modular, production-oriented AutoML framework for binary classification on the Adult Income dataset.

## Features

- Data loading with automatic `?` to `NaN` replacement
- Automatic EDA and figure generation
- sklearn preprocessing pipelines
- Multiple baseline and tuned models
- Evaluation, visualization, logging, and model persistence
- Optional SHAP, learning curves, and Streamlit dashboard

## Project Structure

- `datasets/adult.csv`
- `models/`
- `reports/`
- `logs/`
- `src/`
- `main.py`

## Setup

```bash
pip install -r requirements.txt
```

If `xgboost`, `lightgbm`, or `shap` are unavailable in your environment, the framework will skip those optional components gracefully.

## Data

Place `adult.csv` in `datasets/adult.csv`.

The target column is expected to be `income`.
Missing values represented by `?` are automatically converted to `NaN`.

## Run

```bash
python main.py
```

## Output

- Figures: `reports/figures/`
- Plots: `reports/plots/`
- Results CSV: `reports/model_results.csv`
- Best model: `models/best_model.pkl`
- Logs: `logs/app.log`

## Streamlit Dashboard

Optional dashboard entry point is available at `streamlit_app.py` if Streamlit is installed.
