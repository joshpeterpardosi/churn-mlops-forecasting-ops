# Churn Model — Design Spec

**Date:** 2026-08-11
**Status:** Approved, implemented, amended post-review (see below)
**Parent spec:** [../../../IDEA.md](../../../IDEA.md)
**Sub-project:** 2 of 6 (churn model — consumes the data layer's `feature_table`)

## Amendment (2026-08-11, post-implementation review)

The final whole-branch review found two Important gaps versus this spec's
original intent, both fixed in code and reflected here:

1. **No logged model signature.** The spec's stated purpose ("a later
   serving layer can load 'the current model'") wasn't actually true as
   first implemented — the registered model had no MLflow signature or
   input example, and predicting on a raw `feature_table` frame raised
   `ValueError: train and valid dataset categorical_feature do not match`
   (LightGBM requires callers to independently reproduce the training-time
   categorical casting). Fixed by having `train_churn_model` also return
   the transformed test feature frame, and having the `churn_model` asset
   log a signature + input example via `mlflow.models.infer_signature`.
   **Correction (sub-project 3's final review):** this only fixed half the
   problem. Logging a signature records the expected schema but doesn't
   remove the categorical-cast obligation — the model's own logged
   `input_example` still raised the same `ValueError` when predicted
   against, because MLflow round-trips the example through JSON and loses
   the `category` dtype. The actual fix (applied in sub-project 3) wraps
   the model in an `mlflow.pyfunc.PythonModel` that casts the categorical
   columns internally before predicting, so the logged signature is
   honest and callers pass plain values with no hidden preprocessing
   contract to remember.
2. **Features selected by subtraction, not by an explicit allowlist.** Any
   future column added to `feature_table` would have silently become a
   model feature — the same risk class the data layer's label-leakage bug
   came from. Fixed with an explicit `FEATURE_COLUMNS` list in
   `train_churn_model`.
3. Artifact root corrected from `./mlartifacts/` to `./mlruns/` (MLflow
   3.x's actual SQLite-backed default) — see the Architecture section.

## Purpose

Train a churn classifier on `feature_table` (produced by sub-project 1),
track every run in MLflow, and automatically promote the best run to
Production so a later serving layer can load "the current model" without
knowing which run produced it.

## Decisions carried from IDEA.md / sub-project 1

- Dataset: Kaggle Telco Customer Churn, via `feature_table`
  (`docs/planning/specs/2026-08-11-data-layer-design.md`) — 7043 rows, 10
  columns, `current_mrr`/`trailing_3mo_avg_mrr` leak-free
- Two models trained as separate Dagster assets, MLflow tracking +
  registry for both (this spec covers the churn model only; the forecast
  model is sub-project 3)

## Decisions made this session

- Model family: gradient boosting classifier, LightGBM
  (`lightgbm.LGBMClassifier`) — native categorical support avoids manual
  one-hot encoding of `Contract`/`InternetService`/`PaymentMethod`/`TechSupport`
- Primary metric: ROC-AUC (threshold-independent, standard for binary
  classification under mild class imbalance — 26.5% churn rate). Accuracy,
  precision, recall, and F1 are also logged, but ROC-AUC drives promotion
- MLflow backend: local SQLite-backed tracking store
  (`sqlite:///mlflow.db`, repo root, gitignored) with a local artifact root
  (`./mlruns/`, also gitignored — MLflow 3.x's actual default for a
  SQLite-backed store; this spec originally said `./mlartifacts/`, amended
  post-implementation to match reality). This is queried directly by the
  MLflow Python client — **no `mlflow server` process is required** for
  the Dagster asset to run. A `docker-compose`-managed `mlflow server` (per
  IDEA.md) is for the UI and for other services to reach later; it is not
  a dependency of this asset
- Promotion rule: auto-promote a newly registered model version to
  Production if its ROC-AUC beats the current Production version's, or if
  no Production version exists yet
- Dagster asset granularity: one asset (`churn_model`) does the full
  train → log → register → promote sequence
- Hyperparameters: fixed, reasonable defaults for LightGBM
  (`n_estimators`, `max_depth`, `learning_rate`) — no search. Logged to
  MLflow as params regardless, so they're inspectable per run

## Architecture

```
feature_table → churn_model (train, log to MLflow, register, promote-if-better)
```

- `train_churn_model` (pure function, `src/churn_mlops/lib/churn_model.py`):
  no MLflow imports, unit-testable standalone
- `churn_model` asset (`src/churn_mlops/assets/churn_model.py`): all
  MLflow interaction lives here
- Output is run metadata (`run_id`, `version`, `metrics`, `promoted`), not
  tabular data — uses Dagster's default IOManager, not the data layer's
  `ParquetIOManager`

## Components

1. **`train_churn_model(feature_df, params=DEFAULT_PARAMS) -> (model, metrics, X_test)`**
   (amended post-implementation: also returns the transformed test feature
   frame, so the asset can log an MLflow model signature — see Amendment
   below)
   — drops `customerID` (join key, not a feature); casts `Contract`,
   `InternetService`, `PaymentMethod`, `TechSupport` to pandas `category`
   dtype; stratified 80/20 train/test split on `Churn`
   (`random_state=42`); trains `LGBMClassifier(**params)`; evaluates on
   the held-out test set; returns `metrics = {"roc_auc", "accuracy",
   "precision", "recall", "f1"}` (all floats in `[0, 1]`).
2. **`get_production_roc_auc(client, model_name) -> float | None`** —
   takes an MLflow client (mockable in tests); looks up the current
   Production version's logged `roc_auc`; returns `None` if no Production
   version exists.
3. **`churn_model` asset** — loads `feature_table`; calls
   `train_churn_model`; opens an MLflow run under experiment
   `"churn_model"`; logs params + metrics; logs and registers the model
   (`registered_model_name="churn_model"`); calls
   `get_production_roc_auc`; promotes the new version to Production if it
   wins or nothing is registered yet; returns
   `{"run_id", "version", "metrics", "promoted"}`.

## Data Flow

1. `churn_model` asset loads `feature_table` (already materialized)
2. `train_churn_model` splits, trains, evaluates — pure, no I/O
3. Asset opens an MLflow run, logs params + metrics, logs the model
   artifact, registers a new version under `"churn_model"`
4. Asset queries the current Production version's `roc_auc` via
   `get_production_roc_auc`
5. If the new version wins (or none exists), asset promotes it to
   Production via the MLflow client
6. Asset returns run metadata as its Dagster output

## Error Handling

- Training failures (bad data shape, single-class target) propagate as
  normal exceptions — Dagster surfaces the failed run; no custom handling
  needed at this scale.
- No blocking `asset_check` on this asset (unlike the data layer) —
  `feature_table` already passed its own upstream checks, and
  model-quality gating is the promotion comparison itself, not a separate
  check.
- MLflow client failures (registration, promotion) fail the asset run
  directly — no custom retry logic for a portfolio-scale project.

## Testing

- **pytest**, pure-function level:
  - `train_churn_model` — synthetic `feature_table`-shaped fixture;
    asserts all 5 metric keys present, each in `[0, 1]`, returned model is
    a fitted `LGBMClassifier`
  - `get_production_roc_auc` — `None` when no Production version exists
    (mocked client); correct float when one does
- **Asset-level tests** — real MLflow client against
  `sqlite:///{tmp_path}/mlflow.db` (no server process, per Architecture):
  - first-ever run registers a version and promotes it (no prior
    Production)
  - a subsequent run with better ROC-AUC promotes and replaces Production
  - a subsequent run with worse ROC-AUC registers a new version but does
    **not** promote — prior Production version stays current
- A real-data training run (via `dagster asset materialize`, same pattern
  as the data layer's smoke check) is worth doing once implemented, as a
  sanity check — not a required gate, since `feature_table` already proved
  itself against real data in sub-project 1.

## Out of Scope (future sub-projects)

- Forecast model training / MLflow tracking (sub-project 3)
- FastAPI serving layer (sub-project 4)
- Evidently monitoring + Dagster retrain sensor (sub-project 5)
- CI/CD (GitHub Actions), `gh` repo push (sub-project 6)
