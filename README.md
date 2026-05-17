# Telecom Churn MLOps Project

This project contains a notebook-driven telecom churn prediction workflow built with Python, XGBoost, MLflow, and FastAPI. It covers data preparation, model training, model registry usage, offline evaluation, and a small prediction API.

## Project Overview

The main workflow appears to be:

1. Load and prepare telecom churn data from `telecom_churn.csv`.
2. Train an XGBoost churn model in `model_churn.ipynb`.
3. Save model artifacts under `models/v1/`.
4. Register and track experiments with MLflow in `mlruns/`.
5. Evaluate the registered model with `6_evaluate_model.py`.
6. Serve predictions through `serve_model.py`.

## Repository Structure

```text
MLOPS/
|-- model_churn.ipynb         # Main notebook for data prep, training, MLflow logging, and drift checks
|-- 6_evaluate_model.py       # Loads a registered model and evaluates it on test data
|-- serve_model.py            # FastAPI inference endpoint
|-- telecom_churn.csv         # Source dataset used by the notebook
|-- data/
|   `-- churn_test_v1.parquet # Test dataset consumed by the evaluation script
|-- models/
|   `-- v1/                   # Saved model and preprocessing artifacts
|-- mlruns/                   # Local MLflow tracking store
|-- mlartifacts/              # MLflow artifact storage
`-- mlflow.db                 # Local MLflow database/state
```

## Main Files

### `model_churn.ipynb`

The notebook is the core of the project. Based on the current contents, it handles:

- loading the telecom dataset
- rebuilding the churn label with business logic
- train/validation/test splitting
- feature engineering
- XGBoost model training
- artifact export to `models/v1/`
- MLflow experiment logging
- model registration
- drift metric logging
- creation of `data/churn_test_v1.parquet`

### `6_evaluate_model.py`

This script:

- connects to a local MLflow tracking directory (`file:./mlruns`)
- loads the `telecom-churn-xgb` model from the `Production` stage
- reads `data/churn_test_v1.parquet`
- calculates AUC, F1, precision, recall, and accuracy
- logs evaluation results back to MLflow
- promotes or archives the current model version based on thresholds

### `serve_model.py`

This exposes a FastAPI endpoint:

- `POST /predict`

It expects a payload like:

```json
{
  "features": {
    "telecom_partner": "Airtel",
    "gender": "Male",
    "age": 35
  }
}
```

Current note: `serve_model.py` imports `load_model` from `inference`, but no `inference.py` file is present in this repository snapshot. The API may need that module restored or updated before it can run successfully.

## Setup

This repository does not currently include a root `requirements.txt` or environment file. The code and tracked environments indicate these packages are likely required:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `mlflow`
- `fastapi`
- `uvicorn`
- `pyarrow`
- `pydantic`
- `joblib`

Example setup:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn xgboost mlflow fastapi uvicorn pyarrow pydantic joblib
```

## How To Run

### 1. Train and log the model

Open and run the notebook:

```powershell
jupyter notebook model_churn.ipynb
```

### 2. Evaluate the registered model

Run:

```powershell
python 6_evaluate_model.py
```

The script expects:

- a local MLflow tracking directory at `./mlruns`
- a registered model named `telecom-churn-xgb`
- test data at `data/churn_test_v1.parquet`

### 3. Start the API

Run:

```powershell
uvicorn serve_model:app --reload
```

If startup fails, verify that the missing `inference` module has been added back or that `serve_model.py` is updated to load the model directly from `models/v1/` or MLflow.

## MLflow Notes

This repository already contains local MLflow tracking output, including:

- experiment metadata under `mlruns/`
- registered model metadata for `telecom-churn-xgb`
- a current `Production` model version in the local registry

These folders are generated runtime state and usually should not be committed in a clean source repository unless the goal is to preserve a fixed local demo snapshot.

## Git Hygiene

The project currently contains generated files and folders such as:

- virtual environments (`venv/`, `myenv/`)
- notebook checkpoints
- Python bytecode caches
- MLflow outputs
- local databases
- model binaries
- parquet data outputs

The included `.gitignore` is set up to exclude those items going forward.

## Suggested Next Improvements

- add a real `requirements.txt` or `environment.yml`
- add `inference.py` or update the API to use the saved model directly
- separate source code into modules instead of keeping most logic in the notebook
- add tests for preprocessing and inference
- keep raw data, model artifacts, and MLflow state out of version control unless intentionally publishing a reproducible snapshot
