# Forecast Model — Design Spec

**Date:** 2026-08-11
**Status:** Approved, pending implementation plan
**Parent spec:** [../../../IDEA.md](../../../IDEA.md)
**Sub-project:** 3 of 6 (forecast model — consumes the data layer's `validated_mrr` and `feature_table`)

## Purpose

Train a per-customer MRR forecast model using lag features on top of
`validated_mrr`, track every run in MLflow, and auto-promote the best run
to Production — same operational pattern as the churn model
(sub-project 2), sharing its MLflow promotion logic instead of
duplicating it a second time.

## Decisions carried from IDEA.md / prior sub-projects

- Forecast model family: gradient boosting w/ lag features (IDEA.md)
- MLflow tracking: local SQLite-backed store (`sqlite:///mlflow.db`), no
  `mlflow server` process required — same as sub-project 2
- Promotion: modern alias API (`set_registered_model_alias` /
  `get_model_version_by_alias`, alias `"production"`)
- `validated_mrr`'s terminal churn-zero row exists specifically so the
  forecast model can use it (data layer spec, Components §3) — this spec
  is where that intent gets used

## Decisions made this session

- **Forecast framing:** per-customer next-month-style regression using
  each customer's own lag history, not an aggregate calendar-time
  forecast. `validated_mrr`'s `month` column is relative to each
  customer's signup, not a shared calendar axis, so a calendar-aggregate
  forecast would need synthetic signup dates first — out of scope. This
  framing is also the natural fit for "gradient boosting w/ lag features"
  (lags of each customer's own series), where a calendar-aggregate
  forecast would more naturally suit ARIMA/Prophet, which IDEA.md already
  passed over.
- **Train/test split:** time-based, not random. For each customer, the
  last valid row (full 3-month lag history + target) is held out as test;
  everything earlier is train. Prevents a customer's later month leaking
  into training for their own earlier month.
- **Lag depth:** 3 lags (`lag_1`, `lag_2`, `lag_3`) plus their rolling
  3-month mean. Matches IDEA.md's "lag features" wording directly; deep
  enough for signal, shallow enough that most customers retain usable
  history.
- **Metric:** RMSE drives promotion (penalizes large misses more than
  MAE, which matters more for revenue than for churn classification);
  MAE also logged for context. MAPE rejected — undefined/unstable near
  MRR≈0, which is common in this dataset (new signups, churned
  customers).
- **Feature scope:** lags + static features from `feature_table`
  (`Contract`, `InternetService`, `PaymentMethod`, `TechSupport`), not
  lags alone. Matches IDEA.md's "share overlapping features" framing for
  the two models, likely improves accuracy (contract type affects revenue
  stability), and reuses already-validated columns — no new validation
  logic needed.
- **Shared MLflow promotion helper:** extracted now, on this second model,
  rather than deferred to a third. `lib/mlflow_registry.py`'s
  `promote_if_better` replaces both this model's and (via refactor) the
  churn model's inline promotion comparison.
- **Library:** LightGBM again (`LGBMRegressor`), for consistency with the
  churn model's `LGBMClassifier` — one gradient boosting library across
  both models, one dependency story.

## Architecture

```
validated_mrr ─┐
                ├─→ forecast_features → forecast_model
feature_table ─┘
```

- `build_forecast_features` (pure, `src/churn_mlops/lib/forecast_features.py`)
  — no MLflow/Dagster imports
- `forecast_features` asset (`src/churn_mlops/assets/forecast_features.py`)
  — Parquet via the existing `feature_io_manager` (same stage as
  `feature_table`, no new IOManager instance)
- `train_forecast_model` (pure, `src/churn_mlops/lib/forecast_model.py`)
- **New:** `src/churn_mlops/lib/mlflow_registry.py` —
  `promote_if_better(client, model_name, metric_name, value, higher_is_better) -> bool`
- `forecast_model` asset (`src/churn_mlops/assets/forecast_model.py`) —
  all MLflow interaction lives here, same split as the churn model
- **Refactor:** `churn_model` asset (`src/churn_mlops/assets/churn_model.py`)
  switches its inline promotion comparison to call the shared
  `promote_if_better`; `get_production_roc_auc` is retired

## Components

1. **`build_forecast_features(mrr_df, feature_df) -> pd.DataFrame`** —
   sorts each customer's rows by `month`; computes `lag_1`, `lag_2`,
   `lag_3` (`MRR.shift(1/2/3)` within customer group) and
   `rolling_3mo_mean` (mean of the three lags); drops rows where any lag
   is missing (a customer's first 3 months); joins `Contract`,
   `InternetService`, `PaymentMethod`, `TechSupport` from `feature_table`
   on `customerID`. Output columns: `customerID`, `month`, `lag_1`,
   `lag_2`, `lag_3`, `rolling_3mo_mean`, the 4 static columns,
   `target_mrr` (that row's own `MRR`). A churned customer's terminal
   zero-MRR row is a normal, valid `target_mrr=0` row here — not dropped,
   not special-cased.
2. **`forecast_features` asset** — wraps the above,
   `io_manager_key="feature_io_manager"`.
3. **`train_forecast_model(forecast_df, params=DEFAULT_PARAMS) -> (model, metrics, X_test)`**
   — time-split: groupby `customerID`, last row → test, rest → train
   (customers with only one valid row after lag-dropping go entirely to
   train, contribute no test row); casts the 4 categoricals to pandas
   `category` dtype; trains `LGBMRegressor(**params)`; evaluates `rmse`
   and `mae` on the held-out rows; returns `X_test` from the start (the
   signature-logging gap fixed in sub-project 2's final review doesn't
   need re-discovering here).
4. **`promote_if_better(client, model_name, metric_name, value, higher_is_better) -> bool`**
   (`lib/mlflow_registry.py`) — fetches the current Production version's
   `metric_name` via the alias API; if none exists or `value` beats it in
   the specified direction, promotes and returns `True`; else returns
   `False`. Generalizes sub-project 2's `get_production_roc_auc` from a
   fixed-metric lookup into a metric-agnostic one.
5. **`forecast_model` asset** — loads `forecast_features`, calls
   `train_forecast_model`, opens an MLflow run under experiment
   `"forecast_model"`, logs params/metrics/model with signature + input
   example (via `mlflow.models.infer_signature` on `X_test`), calls
   `promote_if_better(client, "forecast_model", "rmse", metrics["rmse"], higher_is_better=False)`,
   returns `{"run_id", "version", "metrics", "promoted"}`.
6. **`churn_model` asset refactor** — replaces its inline
   `current_production_roc_auc is None or metrics["roc_auc"] > current_production_roc_auc`
   logic with
   `promote_if_better(client, "churn_model", "roc_auc", metrics["roc_auc"], higher_is_better=True)`.
   External behavior must stay identical — its existing test suite
   (`tests/test_churn_model_asset.py`) passes unchanged, proving the
   refactor didn't alter behavior.

## Data Flow

1. `forecast_features` asset loads `validated_mrr` + `feature_table`,
   builds lag/rolling features, joins static columns, drops rows lacking
   full 3-month lag history
2. `forecast_model` asset loads `forecast_features`, time-splits (last
   valid row per customer → test), trains `LGBMRegressor`, evaluates
   `rmse`/`mae`
3. Asset logs params/metrics/model+signature to MLflow, calls
   `promote_if_better(..., higher_is_better=False)`
4. Asset returns run metadata as its Dagster output

## Error Handling

- Same posture as the churn model: training/MLflow failures propagate as
  normal exceptions; no custom handling; no blocking `asset_check` on
  either new asset — upstream data already passed its checks in the data
  layer.
- Documented behavior, not an error: a customer with fewer than 4 months
  of history (3 lags + 1 target row) contributes zero rows to
  `forecast_features` — they don't appear, rather than producing NaN-lag
  rows or crashing.

## Testing

- **pytest**, pure-function level:
  - `build_forecast_features` — lag values correct for a multi-month
    synthetic customer; a customer with <4 valid months contributes zero
    rows; static features joined correctly; a churned customer's
    terminal zero-MRR row appears as a valid `target_mrr=0` row
  - `train_forecast_model` — `metrics` has `rmse`/`mae` keys, both ≥0;
    returned model is a fitted `LGBMRegressor`; time-split respects order
    (every test row's `month` ≥ that customer's max train-row `month`)
  - `promote_if_better` — both directions tested with a mocked client
    (`higher_is_better=True` and `=False`); the no-Production-yet case
- **Asset-level tests** (real MLflow client, `tmp_path`-scoped,
  `monkeypatch.chdir` per sub-project 2's hermeticity fix): first run
  promotes, better RMSE promotes and replaces, worse RMSE registers but
  doesn't promote — same 3-scenario pattern as the churn model
- **Regression check on the `churn_model` refactor**: its existing test
  suite must pass unchanged — same external behavior required, no new
  tests needed for this part
- **Real-data sanity run**:
  `python -m dagster asset materialize --select feature_table,forecast_features,forecast_model -m churn_mlops.definitions`
  against the real Kaggle-derived data already on disk — not deferred,
  same as sub-project 2's Task 5

## Out of Scope (future sub-projects)

- FastAPI serving layer (sub-project 4) — will need to load both
  `models:/churn_model@production` and `models:/forecast_model@production`
- Evidently monitoring + Dagster retrain sensor (sub-project 5)
- CI/CD (GitHub Actions), `gh` repo push (sub-project 6)
- Calendar-aggregate portfolio-level revenue forecasting (would need
  synthetic signup dates assigned to `validated_mrr`'s relative month
  axis) — not attempted here
