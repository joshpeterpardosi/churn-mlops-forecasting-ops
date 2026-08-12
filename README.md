# churn-mlops-forecasting-ops

![CI](https://github.com/joshpeterpardosi/churn-mlops-forecasting-ops/actions/workflows/ci.yml/badge.svg)

An end-to-end MLOps loop for a SaaS subscription business: customer churn
prediction and MRR/revenue forecasting, built as one pipeline — data,
train, track, serve, monitor, retrain.

## Architecture

```
[Kaggle churn CSV] --+
                      |
[Synthetic MRR series] +--> Dagster asset pipeline
                      |
                 raw -> validate -> feature table
                      |
            +---------+---------+
            |                   |
     churn model           forecast model
     (classification)      (MRR / revenue, time series)
            |                   |
        MLflow tracking + model registry (both models)
            |
     FastAPI serving layer (Docker)
            |
     predictions log --> Evidently AI drift / perf reports
            |
     Dagster sensor watches drift threshold --> retrain trigger (loops back to training)
```

## What's here

| Stage | What it does |
|---|---|
| Data layer | Dagster assets: raw ingestion, validation, feature table join |
| Churn model | LightGBM classifier, tracked and auto-promoted via MLflow |
| Forecast model | LightGBM regression on lag features, same MLflow promotion path |
| Serving | FastAPI: `POST /predict/churn`, `GET /forecast/mrr?horizon=N` |
| Monitoring | Evidently drift reports + a Dagster sensor that triggers retraining |
| CI/CD | GitHub Actions: lint (ruff), test (pytest), Docker build |

Design docs and implementation plans for each stage are in
[`docs/planning/`](docs/planning/).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate  # .venv/bin/activate on macOS/Linux
pip install -r requirements-lock.txt
pip install --no-deps -e .
```

## Running it

```bash
# Orchestration UI — materialize assets, inspect the pipeline graph
dagster dev

# Serving API
uvicorn churn_mlops.serving.app:app --reload

# Tests
pytest
```

## Requirements

Python 3.13. Dependencies are pinned in `requirements-lock.txt` (the exact
versions this project is tested against); `pyproject.toml` documents the
direct dependency ranges.
