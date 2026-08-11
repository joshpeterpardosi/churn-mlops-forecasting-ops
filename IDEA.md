# churn-mlops-forecasting-ops — Project Idea

**Status:** Idea / design, not yet implemented
**Type:** Portfolio project (MLOps)

## Goal

Demonstrate a full MLOps loop by combining two SaaS subscription models —
customer churn prediction and MRR/revenue forecasting — in one pipeline,
end to end: data → train → track → serve → monitor → retrain.

## Domain & Data

- **Domain:** SaaS subscription business
- **Churn data:** Kaggle "Telco Customer Churn" dataset (pinned) — provides
  churn labels and customer attributes
- **Forecast data:** Synthetic MRR/revenue time series generated to align
  with the churn dataset's customer cohorts and churn timeline

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

## Components

- **Orchestration — Dagster**: asset-based DAG, runs locally (`dagster dev`),
  built-in UI shows asset graph/lineage/run history. Sensor polls the
  Evidently drift report and fires a retrain job when drift crosses threshold.
- **Two models**, trained as separate Dagster assets, sharing overlapping
  features (tenure, plan, usage) from one feature table:
  - Churn model: binary classification
  - Forecast model: MRR/revenue time series
- **Experiment tracking — MLflow**: logs runs/params/metrics for both models,
  registers and promotes the best run to "production" stage.
- **Serving — FastAPI + Docker**: two endpoints —
  `POST /predict/churn` and `GET /forecast/mrr?horizon=N` — containerized,
  bundled via `docker-compose` alongside Dagster, MLflow, and Evidently
  services so the whole stack runs locally with one command.
- **Monitoring — Evidently AI**: generates data drift, target drift, and
  model performance HTML reports by comparing live prediction logs against
  the training baseline.
- **CI/CD — GitHub Actions**: on push, lint (ruff), run pytest, build the
  Docker image. Repo creation and pushes handled via `gh` CLI during the
  implementation session.

## Data Flow

1. Raw CSV (churn) + synthetic series (MRR) ingested
2. Validation asset: schema, null, and range checks — pipeline fails fast
   on bad data, does not propagate downstream
3. Feature table built from validated data
4. Both models trained, every run logged to MLflow
5. Best run per model registered/promoted
6. FastAPI loads registered models, serves predictions
7. Predictions logged
8. Evidently compares live predictions vs. training baseline, generates
   drift/perf reports
9. Dagster sensor reads drift score; above threshold → triggers retrain job,
   loop repeats

## Error Handling & Testing

- **pytest**: data validation checks, model output shape/range assertions,
  API request/response contract tests
- **Dagster asset checks**: validation asset failure halts the pipeline
  before training runs on bad data
- **API**: Pydantic input validation (422 on bad payload), 503 if model
  not loaded

## Open Items for Implementation Phase

- ~~Pin exact churn dataset source~~ — **decided: Kaggle Telco Customer
  Churn**
- ~~Define synthetic MRR generation logic~~ — **decided: step drop to 0 at
  churn_date per customer (from Telco churn label), flat MRR before that,
  plus mild noise + monthly seasonality overlay**
- ~~Pick forecast model family~~ — **decided: gradient boosting w/ lag
  features**
- ~~Define concrete drift threshold for retrain trigger~~ — **decided:
  Evidently `DataDriftPreset` default — dataset drift flagged when share of
  drifted columns ≥ 0.5; sensor reads this result directly**
- Project folder location, git init, `gh` repo setup — to happen in the
  desktop implementation session, not here
