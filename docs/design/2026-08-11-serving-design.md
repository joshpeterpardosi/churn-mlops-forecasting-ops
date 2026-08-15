# Serving — Design Spec

**Date:** 2026-08-11
**Status:** Approved, pending implementation plan
**Parent spec:** [0000-project-brief.md](0000-project-brief.md)
**Sub-project:** 4 of 6 (serving — consumes both models' `@production` aliases and `forecast_features`)

## Purpose

Expose the churn and forecast models as a FastAPI service: `POST
/predict/churn` for per-customer churn risk, `GET /forecast/mrr?horizon=N`
for an N-month-ahead portfolio MRR forecast — containerized via Docker,
consistent with the project brief's "everything local" pattern.

## Decisions carried from the project brief / prior sub-projects

- Endpoints: `POST /predict/churn`, `GET /forecast/mrr?horizon=N`
  (the project brief)
- Pydantic input validation → 422 on bad payload; 503 if model not loaded
  (the project brief)
- Both models already predict correctly on plain-dtype input via
  `CategoricalCastingModel` (`src/churn_mlops/lib/serving.py`, fixed in
  sub-project 3's final review) — no manual categorical casting needed at
  the API layer
- Load via the alias URI: `mlflow.pyfunc.load_model("models:/churn_model@production")`
  / `"models:/forecast_model@production"`

## Decisions made this session

- **Horizon semantics:** `forecast_model` predicts one month ahead per
  customer from lag features — it has no native multi-step or aggregate
  capability. `horizon=N` is served by recursively predicting each active
  customer N steps ahead (feeding each prediction back in as the new
  `lag_1`, shifting the lag window), then summing the customers'
  month-`N` predictions into a portfolio total. Standard technique for
  extending a single-step lag model — matches "gradient boosting w/ lag
  features" rather than requiring a different model family.
- **Forecast seed data:** read `data/features/forecast_features.parquet`
  directly (each customer's latest row = starting lag state) rather than
  requiring the caller to submit history. Zero new infra, consistent with
  the local-file pattern from sub-projects 1-3; freshness = whenever
  Dagster last materialized `forecast_features`.
- **Churn request shape:** caller submits raw feature values in the POST
  body (not a `customerID` lookup) — works for any customer including
  hypothetical/new ones not yet in `feature_table`, standard REST
  prediction-endpoint pattern.
- **Model loading:** both models loaded once at FastAPI startup, cached in
  `app.state`. Load failure (e.g. no `production` alias yet) doesn't
  crash the app — endpoints return 503 until a working model exists.
  Rejected per-request loading: adds MLflow round-trip latency to every
  call, and "503 if not loaded" stops being a real startup state.
- **Docker data access:** bind-mount `mlflow.db`, `mlruns/`, `data/` into
  the container rather than baking them into the image — the container
  always sees current local state; retraining doesn't require a rebuild.
- **Churn response:** both `churn_probability` (float) and
  `churn_prediction` (bool), not just the hard label — ROC-AUC (the
  model's own training metric) is probability-based, so exposing the
  probability matches what the model was actually optimized for. Requires
  extending `CategoricalCastingModel` with a `predict_proba` path.
- **Horizon bound:** `horizon` validated to `1–24` (422 outside that
  range) — recursive compute cost scales linearly with
  `horizon × n_customers`; an unbounded horizon is a local-DoS footgun
  even for a portfolio demo.

## Architecture

```
FastAPI app (src/churn_mlops/serving/app.py)
  ├─ startup: load models:/churn_model@production + models:/forecast_model@production
  │           via mlflow.pyfunc.load_model, cache in app.state; failure → not loaded, not fatal
  ├─ POST /predict/churn  → Pydantic request → churn_model → {churn_probability, churn_prediction}
  └─ GET /forecast/mrr?horizon=N → read data/features/forecast_features.parquet →
                                    recursive_forecast() → {horizon, forecast_total_mrr}
```

- `src/churn_mlops/lib/recursive_forecast.py` — pure function, no
  FastAPI/MLflow imports
- `src/churn_mlops/serving/schemas.py` — Pydantic `ChurnPredictRequest` /
  `ChurnPredictResponse`
- `src/churn_mlops/lib/serving.py`'s `CategoricalCastingModel` gains a
  `predict_proba` method
- `Dockerfile` (repo root) + `docker-compose.yml` (new, one `api` service
  for now — extensible for `dagster`/`mlflow`/`evidently` services in
  later sub-projects per the project brief)

## Components

1. **`CategoricalCastingModel.predict_proba`** (extends
   `lib/serving.py`) — casts categorical columns internally, then calls
   `self.model.predict_proba(df)[:, 1]`. Only meaningful for the churn
   classifier. Accessing it after `mlflow.pyfunc.load_model(...)`
   requires `.unwrap_python_model()` — the loaded pyfunc wrapper only
   exposes `.predict()` by default; `unwrap_python_model()` returns the
   actual `CategoricalCastingModel` instance with its custom methods.
2. **`recursive_forecast(seed_df, horizon, predict_fn) -> float`**
   (`lib/recursive_forecast.py`) — for each of `horizon` steps: call
   `predict_fn` on current per-customer feature rows, shift
   `lag_1 → lag_2 → lag_3`, insert the new prediction as `lag_1`,
   recompute `rolling_3mo_mean`; after `horizon` steps, sum the final
   predictions across all customers. `predict_fn` is injected so tests
   can use a fake instead of a real MLflow model.
3. **`app.py`** — FastAPI lifespan handler loads both models at startup
   (failure → `None`, logged, non-fatal); `POST /predict/churn`
   (Pydantic-validated body → `predict_proba` → `{churn_probability,
   churn_prediction}`, threshold 0.5 for the bool); `GET
   /forecast/mrr?horizon=N` (`horizon` validated `1–24` via FastAPI
   `Query`, reads `forecast_features.parquet`, takes each customer's
   latest row, calls `recursive_forecast`).
4. **Docker** — `Dockerfile` (`pip install -e .`, `CMD uvicorn
   churn_mlops.serving.app:app --host 0.0.0.0 --port 8000`),
   `docker-compose.yml` with one `api` service, bind-mounting
   `mlflow.db`/`mlruns/`/`data/`.

## Data Flow

1. App startup: attempt to load both production models via
   `unwrap_python_model()`; store in `app.state`; failure → `None`,
   logged, app still starts
2. `POST /predict/churn`: Pydantic validates body (422 on bad
   shape/type) → if model not loaded → 503 → else build a 1-row
   DataFrame → `predict_proba` → response
3. `GET /forecast/mrr?horizon=N`: `horizon` validated `1–24` (422 if out
   of range) → if model not loaded → 503 → else read
   `forecast_features.parquet`, take latest row per `customerID` →
   `recursive_forecast` → response

## Error Handling

- **422**: Pydantic validation failure on the churn request body;
  `horizon` outside `1–24`
- **503**: model not loaded (startup failed or still pending); also
  returned if `forecast_features.parquet` is missing (e.g. Dagster never
  run) — same "dependency not ready" semantics as model-not-loaded, not a
  500
- Any other unexpected exception during inference: no custom handling —
  FastAPI's default 500, consistent with "no error handling beyond what's
  needed" from prior specs

## Testing

- **pytest**, pure-function level:
  - `recursive_forecast` — fake `predict_fn`, verify lag-shift logic
    across steps, sum-across-customers correct, `horizon=1` reduces to a
    single predict call
  - `CategoricalCastingModel.predict_proba` — casts correctly, returns
    probability array in `[0, 1]`
- **API-level tests** (FastAPI `TestClient`):
  - `POST /predict/churn` valid body → 200, correct shape, probability
    in `[0, 1]`
  - `POST /predict/churn` invalid body (missing field/wrong type) → 422
  - `GET /forecast/mrr?horizon=N` valid → 200
  - `GET /forecast/mrr?horizon=0` or `horizon=25` → 422
  - Model-not-loaded (mocked `app.state`) → both endpoints 503
- **Real-data sanity check**: run the API locally (or via `docker-compose
  up`) against the real registered models + real
  `forecast_features.parquet` already on disk, `curl` both endpoints,
  confirm sane responses — not deferred, same pattern as sub-projects
  1-3's real-data runs.

## Out of Scope (future sub-projects)

- Evidently monitoring + Dagster retrain sensor (sub-project 5)
- CI/CD (GitHub Actions), `gh` repo push (sub-project 6)
- Authentication/rate limiting — not requested by the project brief, portfolio-scale
  demo
- `docker-compose` orchestration of `dagster dev` / `mlflow server` /
  Evidently services together — this sub-project adds only the `api`
  service; the full multi-service compose file is implied by the project brief but
  not required until later sub-projects need it
