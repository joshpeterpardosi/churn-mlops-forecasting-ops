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

# activate the venv:
source .venv/Scripts/activate   # git-bash on Windows, or macOS/Linux (.venv/bin/activate)
.venv\Scripts\activate          # cmd.exe / PowerShell on Windows

pip install -r requirements-lock.txt
pip install --no-deps -e .
```

## Prerequisites

Before `dagster dev` or the serving API will actually work end to end, a
fresh clone needs two things that are gitignored (never source-controlled)
and so aren't present out of the box:

1. **The source dataset.** Download the Kaggle
   ["Telco Customer Churn"](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
   dataset CSV and save it to `data/source/telco_churn.csv` — this is the
   exact path the `raw_churn` asset expects
   (see `src/churn_mlops/assets/churn.py`).
2. **Trained, promoted models.** The serving API loads models from MLflow's
   `@production` alias; until the pipeline has run once, `/predict/churn`
   and `/forecast/mrr` return `503`. Materialize the full asset pipeline
   first, either from the `dagster dev` UI ("Materialize all") or from the
   command line:

   ```bash
   dagster asset materialize --select "*" -m churn_mlops.definitions
   ```

   This trains and promotes both the churn and forecast models before you
   start the API.

Want to try the project without any of that setup? **`pytest` works
standalone on a bare clone**, with nothing beyond `pip install` above — all
101 tests use `tmp_path` and mocks, no real dataset or trained models
required. It's the fastest way to see the project actually run.

## Running it

```bash
# Orchestration UI — materialize assets, inspect the pipeline graph
dagster dev

# Serving API (after the pipeline has been materialized — see Prerequisites)
uvicorn churn_mlops.serving.app:app --reload

# Tests — no prerequisites needed
pytest
```

## Requirements

Python 3.13. Dependencies are pinned in `requirements-lock.txt` (the exact
versions this project is tested against); `pyproject.toml` documents the
direct dependency ranges.
