# Monitoring — Design Spec

**Date:** 2026-08-12
**Status:** Approved, pending implementation plan
**Parent spec:** [0000-project-brief.md](0000-project-brief.md)
**Sub-project:** 5 of 6 (monitoring — consumes serving's live predictions and the data layer's `feature_table`)

## Purpose

Log every `POST /predict/churn` prediction, generate an Evidently data-drift
+ prediction-drift report comparing live predictions against the training
baseline, and wire a Dagster sensor that triggers a `churn_model` retrain
when drift crosses the project brief's pinned threshold.

## Decisions carried from the project brief / prior sub-projects

- Drift trigger: Evidently `DataDriftPreset` default, drifted-column share
  ≥ 0.5 (pinned in the data layer spec, sub-project 1)
- `promote_if_better` already exists (`lib/mlflow_registry.py`) and
  auto-promotes a retrained `churn_model` if it beats the current
  Production `roc_auc` — this sub-project only needs to trigger the
  retrain, not touch promotion logic
- Storage: local files only, no cloud, no new external service — same
  pattern as sub-projects 1-4

## Decisions made this session

- **Scope: churn model only.** Data drift + target drift + performance is
  the well-supported Evidently workflow for a classifier (true label
  eventually exists, clear presets). Forecast/regression drift monitoring
  is a different Evidently workflow and out of scope here — can be added
  later as its own increment if wanted.
- **Prediction logging: append to a local Parquet log
  (`data/predictions/churn_predictions.parquet`).** Same storage
  convention as the rest of the project. No native Parquet append mode —
  read-existing + concat + write, acceptable at this data volume.
- **Current-data window: all logged predictions since the log file's
  creation, no rolling window.** Simplest option, matches the "no extra
  state" pattern, and is also the more statistically meaningful choice at
  this data volume — a small rolling window would just be noisy with
  demo-scale traffic.
- **No model-performance report.** Live predictions have no ground-truth
  label (nobody knows yet if a customer actually churned), so true
  performance monitoring isn't computable without a label-feedback loop
  this project doesn't have. Scope is **data drift** (input feature
  distributions vs training baseline) **+ prediction drift** (predicted
  probability/label distribution vs training baseline's actual `Churn`
  distribution, substituting for "target drift" since no live ground
  truth exists). This is a deliberate, documented scope reduction from
  the project brief's original "data drift, target drift, model performance" list —
  full performance monitoring would need customers' actual churn outcomes
  to arrive later, which is a different sub-project's worth of plumbing
  (out of scope for this repo).
- **Architecture: Dagster asset generates the report, a sensor watches its
  metadata.** `drift_report` asset does the actual Evidently computation
  and persists an HTML report + drift-share metadata; the sensor is a
  thin trigger layer reading that asset's latest materialization metadata
  — not where drift math lives. Matches this repo's asset-per-artifact
  pattern (data layer, both models) rather than computing drift inline on
  every sensor tick.

## Architecture

```
POST /predict/churn --logs--> data/predictions/churn_predictions.parquet
                                        │
                                        ▼
                              drift_report Dagster asset
                     (reference: feature_table: current: churn_predictions)
                                        │
                                        ▼
                  reports/churn_drift_report.html
                  + metadata {drift_share, dataset_drift_detected}
                                        │
                                        ▼
                              retrain_sensor (Dagster)
                    watches drift_report's latest materialization
                    drift_share ≥ 0.5 → RunRequest(churn_model)
```

- `lib/prediction_logging.py` — pure-ish logging function, isolated file
  I/O, no Dagster/Evidently imports
- `serving/app.py` — `predict_churn` calls the logger after building its
  response; a logging failure must never break the HTTP response
- `lib/drift_report.py` — pure function wrapping Evidently, no Dagster
  imports
- `assets/drift_report.py` — Dagster asset: reads `feature_table` +
  `churn_predictions.parquet`, calls the pure function, writes the HTML
  report to disk, returns drift metadata
- `sensors/retrain_sensor.py` — Dagster sensor, fires `RunRequest` for
  `churn_model` when the latest `drift_report` materialization's
  `drift_share` metadata ≥ 0.5, using a cursor so it never re-fires for
  the same materialization twice

## Components

1. **`log_prediction(row: dict, log_path=DEFAULT_PREDICTIONS_PATH) -> None`**
   (`lib/prediction_logging.py`) — appends one row (the request's raw
   feature values + `churn_probability` + `churn_prediction` + a
   timestamp) to the Parquet log. Creates the file if it doesn't exist
   yet. Read-existing + concat + write.
2. **`predict_churn` (modified)** — after building `ChurnPredictResponse`,
   calls `log_prediction(...)` inside a try/except that only logs a
   warning on failure — the HTTP response must succeed regardless of
   whether logging worked.
3. **`build_drift_report(reference_df, current_df) -> evidently.Report`**
   (`lib/drift_report.py`) — builds an Evidently report combining a data
   drift preset (on the churn model's `FEATURE_COLUMNS`) and a
   prediction-drift check (comparing the current data's predicted
   `Churn`-equivalent column against the reference's actual `Churn`
   column). Caller extracts `drift_share` and `dataset_drift_detected`
   from the report's dict/JSON output.
4. **`drift_report` asset** — reads `feature_table` (reference, restricted
   to `FEATURE_COLUMNS`) and `data/predictions/churn_predictions.parquet`
   (current). If the predictions log doesn't exist yet (no requests
   served yet — a valid, expected state early on), skips computation and
   returns `drift_share=None`, `dataset_drift_detected=False` rather than
   failing. Otherwise calls `build_drift_report`, writes the HTML report
   to `reports/churn_drift_report.html`, returns
   `{"drift_share": float, "dataset_drift_detected": bool}` as both the
   Dagster output and materialization metadata (so it's visible in the
   Dagster UI without opening the HTML file).
5. **`retrain_sensor`** — a Dagster asset sensor on `drift_report`. On
   each evaluation, reads the latest materialization's
   `dataset_drift_detected` metadata; if `True`, yields a `RunRequest`
   that materializes `churn_model`. Uses the materialization's own
   identity (e.g. its run ID or storage ID) as the sensor cursor so the
   same drift result never triggers two retrains.

## Data Flow

1. Each `POST /predict/churn` call logs its input features + prediction
   to `churn_predictions.parquet` (best-effort, never blocks the
   response)
2. `drift_report` asset (materialized manually or on Dagster's normal
   schedule) reads `feature_table` as reference and
   `churn_predictions.parquet` as current, computes the Evidently report,
   writes the HTML, returns drift metadata
3. `retrain_sensor` evaluates on each tick, reads `drift_report`'s latest
   materialization metadata, fires a `churn_model` retrain if
   `dataset_drift_detected` is `True` and this materialization hasn't
   already triggered one
4. The retrained `churn_model` auto-promotes via the existing
   `promote_if_better` if it beats current Production — no new promotion
   logic needed here

## Error Handling

- `log_prediction` failures are caught in `predict_churn` and logged as a
  warning — never surfaced to the HTTP caller as an error, since
  monitoring is a side-concern to serving a prediction
- `drift_report` asset handles a missing/empty predictions log gracefully
  (no data yet is expected, not an error) rather than raising
- No blocking `asset_check` on `drift_report` — unlike the data layer's
  validation checks, drift is informational, not a data-quality gate

## Testing

- **pytest**, pure-function level:
  - `log_prediction` — writes the correct row shape; appends to an
    existing file; creates the file if missing; multiple calls produce
    multiple rows
  - `build_drift_report` — synthetic reference/current DataFrames with
    injected drift (e.g. `current`'s `MonthlyCharges` shifted well
    outside `reference`'s range) produces `dataset_drift_detected=True`;
    identical distributions produce `dataset_drift_detected=False`
- **Asset-level tests**: `drift_report` materializes correctly against a
  real-shaped fixture predictions log; materializes gracefully (no crash,
  `drift_share=None`) when the predictions log doesn't exist; the HTML
  file is actually written to disk
- **Sensor-level tests**: `retrain_sensor` yields a `RunRequest` when the
  latest `drift_report` materialization shows drift detected; yields
  nothing when it doesn't; does not re-fire for a materialization it's
  already processed (cursor behavior)
- **Real-data sanity check**: make a handful of real `curl` requests
  against the running `POST /predict/churn` endpoint to populate the log,
  materialize `drift_report` for real via
  `python -m dagster asset materialize --select feature_table,drift_report -m churn_mlops.definitions`,
  inspect the actual HTML output and metadata — not deferred, same
  pattern as every prior sub-project's real-data verification.

## Out of Scope (future sub-projects / explicitly deferred)

- Forecast model drift monitoring (regression-specific Evidently preset,
  different workflow) — could be a future increment, not required here
- True model-performance monitoring (needs a ground-truth feedback loop —
  actual customer churn outcomes arriving later — that doesn't exist in
  this project)
- CI/CD (GitHub Actions), `gh` repo push (sub-project 6)
- Docker containerization of the Dagster sensor/scheduler process — the
  existing `docker-compose.yml` only runs the FastAPI `api` service;
  running Dagster's sensor daemon in Docker is not required by this spec
