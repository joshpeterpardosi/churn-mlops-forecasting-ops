# Churn Prediction & Revenue Forecasting — MLOps Pipeline

![CI](https://github.com/joshpeterpardosi/churn-mlops-forecasting-ops/actions/workflows/ci.yml/badge.svg)

An end-to-end MLOps loop for a SaaS subscription business: customer churn
prediction and MRR/revenue forecasting, built as one pipeline — data,
train, track, serve, monitor, retrain.

## Headline result

| Metric | Value |
|---|---|
| Churn model | LightGBM, ROC-AUC **0.8401** |
| Decision threshold | 0.25, chosen by expected cost at a 10:1 miss-to-false-alarm ratio — not the 0.5 default |
| Recall at the chosen point | **0.8021**, against 0.5053 at the default threshold |
| Precision at the chosen point | 0.5000, held at the floor that keeps the retention list credible |
| Serving | FastAPI, live predictions from the MLflow production-aliased model |
| Retraining | Evidently drift sensor triggers a Dagster retrain run |
| Tests | 101 standalone tests in CI, no dataset or trained model required |

Moving the threshold from the library default to the cost-derived point
converts 59% more churners caught, paid for with precision the retention
team can still act on. The threshold is stored with the model, so serving
applies the point training chose instead of re-deciding it — the reasoning,
including the rule that was tried first and rejected, is in
[`docs/design/2026-08-16-churn-operating-point.md`](docs/design/2026-08-16-churn-operating-point.md).

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
| Churn model | LightGBM classifier, tracked and auto-promoted via MLflow. Decision threshold chosen from the cost of a missed churner, not left at 0.5 |
| Forecast model | LightGBM regression on lag features, same MLflow promotion path |
| Serving | FastAPI: `POST /predict/churn`, `GET /forecast/mrr?horizon=N`, plus an interactive demo UI at `GET /` |
| Monitoring | Evidently drift reports + a Dagster sensor that triggers retraining |
| CI/CD | GitHub Actions: lint (ruff), test (pytest), Docker build |

The design brief and a per-stage design spec for each of the six stages are in
[`docs/design/`](docs/design/).

### Operating point

A classifier returns a probability; 0.5 is a library default, not a decision. For
churn the errors cost very different amounts — a missed churner loses the account,
a false alarm costs a phone call — so the threshold is chosen by minimising
expected cost at a 10:1 ratio, subject to a precision floor that keeps the
retention list credible. Full reasoning, including the rule that was tried first
and rejected, is in
[`docs/design/2026-08-16-churn-operating-point.md`](docs/design/2026-08-16-churn-operating-point.md).

| | at 0.50 (default) | at 0.25 (chosen) |
|---|---|---|
| recall | 0.5053 | **0.8021** |
| precision | 0.6517 | 0.5000 |
| roc_auc | 0.8401 | 0.8401 |

The threshold travels with the model, so serving applies the point that training
chose rather than re-deciding it.

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

Once the API is running, open `http://localhost:8000/` in a browser for an
interactive demo — real forms for both endpoints, wired to the live models,
no separate setup needed.

## Requirements

Python 3.13. Dependencies are pinned in `requirements-lock.txt` (the exact
versions this project is tested against); `pyproject.toml` documents the
direct dependency ranges.
