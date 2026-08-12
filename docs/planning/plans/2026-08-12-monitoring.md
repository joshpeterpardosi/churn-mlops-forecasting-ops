# Monitoring Implementation Plan

**Goal:** Log every `POST /predict/churn` prediction, add a Dagster `drift_report` asset that runs Evidently data-drift + prediction-drift against the training baseline, and a `retrain_sensor` that fires a `churn_model` retrain when drift is detected.

**Architecture:** `predict_churn` appends each prediction to a local Parquet log (pure logging function, best-effort — never breaks the HTTP response). `drift_report` (Dagster asset) reads that log plus `feature_table`, runs an Evidently report via a pure wrapper function, writes an HTML report, and returns drift metadata. `retrain_sensor` reads `drift_report`'s latest materialization metadata from the Dagster event log and fires a `RunRequest` for `churn_model` when drift was detected, using the materialization's own storage ID as a dedup cursor.

**Tech Stack:** Evidently (`DataDriftPreset`, `TargetDriftPreset`), Dagster (asset + sensor + `define_asset_job`), pandas, FastAPI (existing), pytest.

## Global Constraints

- Scope: churn model only — no forecast-model drift monitoring, no model-performance report (no ground-truth feedback loop exists) (spec: Decisions made this session)
- Prediction logging: append to `data/predictions/churn_predictions.parquet`; a logging failure must never break the `POST /predict/churn` HTTP response (spec: Components §2, Error Handling)
- Current-data window for drift: all logged predictions, no rolling window (spec: Decisions made this session)
- Drift trigger: Evidently `DataDriftPreset`'s own default `dataset_drift` boolean (internally computed at drifted-column share ≥ 0.5) — do not re-implement the threshold comparison manually, read the field Evidently already computes (spec: carried from data layer spec, sub-project 1)
- `drift_report` asset must handle a missing predictions log gracefully (no requests served yet is an expected state, not an error) — no exception, return `dataset_drift_detected=False` (spec: Error Handling)
- No blocking `asset_check` on `drift_report` — drift is informational, not a data-quality gate (spec: Error Handling)
- `retrain_sensor` must never fire twice for the same `drift_report` materialization (spec: Architecture)

## Prerequisite

None — `feature_table.parquet` and the real registered `churn_model` already exist locally from prior sub-projects, so this plan's final task can run its real-data sanity check directly.

---

## File Structure

```
pyproject.toml                                  # + evidently dependency
.gitignore                                       # + reports/
src/churn_mlops/
  lib/
    prediction_logging.py                        # log_prediction
    drift_report.py                               # build_drift_report, extract_drift_summary
  serving/
    app.py                                        # MODIFY: predict_churn logs each prediction
  assets/
    drift_report.py                                # drift_report Dagster asset
  sensors/
    __init__.py                                    # NEW
    retrain_sensor.py                               # retrain_sensor, churn_retrain_job
  definitions.py                                   # MODIFY: + drift_report asset, + retrain_sensor
tests/
  test_prediction_logging.py
  test_app.py                                       # MODIFY: + logging tests
  test_drift_report.py
  test_drift_report_asset.py
  test_retrain_sensor.py
  test_definitions.py                                # MODIFY: 9 assets + sensor
data/predictions/                                    # generated at runtime, covered by existing `data/` gitignore entry
reports/                                              # generated at runtime, needs new gitignore entry
```

---

### Task 1: `log_prediction` pure function

**Files:**
- Create: `src/churn_mlops/lib/prediction_logging.py`
- Test: `tests/test_prediction_logging.py`

**Interfaces:**
- Produces: `DEFAULT_PREDICTIONS_PATH: str`,
  `log_prediction(row: dict, log_path: str = DEFAULT_PREDICTIONS_PATH) -> None`.
  Appends one row to a Parquet log, creating the file and its parent
  directory if they don't exist. Consumed by Task 2's `predict_churn`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prediction_logging.py
import pandas as pd

from churn_mlops.lib.prediction_logging import log_prediction


def test_log_prediction_creates_file_with_one_row(tmp_path):
    log_path = tmp_path / "predictions.parquet"
    log_prediction({"tenure": 5, "churn_probability": 0.7}, log_path=str(log_path))

    df = pd.read_parquet(log_path)
    assert len(df) == 1
    assert df.iloc[0]["tenure"] == 5
    assert df.iloc[0]["churn_probability"] == 0.7
    assert "logged_at" in df.columns


def test_log_prediction_appends_to_existing_file(tmp_path):
    log_path = tmp_path / "predictions.parquet"
    log_prediction({"tenure": 5, "churn_probability": 0.7}, log_path=str(log_path))
    log_prediction({"tenure": 10, "churn_probability": 0.3}, log_path=str(log_path))

    df = pd.read_parquet(log_path)
    assert len(df) == 2
    assert list(df["tenure"]) == [5, 10]


def test_log_prediction_creates_parent_directory(tmp_path):
    log_path = tmp_path / "nested" / "dir" / "predictions.parquet"
    log_prediction({"tenure": 5}, log_path=str(log_path))

    assert log_path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_prediction_logging.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_mlops.lib.prediction_logging'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn_mlops/lib/prediction_logging.py
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_PREDICTIONS_PATH = "data/predictions/churn_predictions.parquet"


def log_prediction(row: dict, log_path: str = DEFAULT_PREDICTIONS_PATH) -> None:
    row = dict(row)
    row["logged_at"] = datetime.now(timezone.utc).isoformat()

    new_row_df = pd.DataFrame([row])

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing_df = pd.read_parquet(path)
        combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    else:
        combined_df = new_row_df

    combined_df.to_parquet(path, index=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_prediction_logging.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/churn_mlops/lib/prediction_logging.py tests/test_prediction_logging.py
git commit -m "feat: add log_prediction pure function"
```

---

### Task 2: Wire logging into `POST /predict/churn`

**Files:**
- Modify: `src/churn_mlops/serving/app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `log_prediction` (Task 1)
- Produces: module constant `PREDICTIONS_LOG_PATH: str` in `app.py` (read
  fresh inside `predict_churn` each call, same late-binding pattern
  already used for `FORECAST_FEATURES_PATH` in this file — required so
  `monkeypatch.setattr` works in tests)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_app.py — add to the existing file
import pandas as pd


def test_predict_churn_logs_prediction_to_configured_path(tmp_path, monkeypatch):
    app.state.churn_model = _FakeChurnModel()
    log_path = tmp_path / "predictions.parquet"
    monkeypatch.setattr(app_module, "PREDICTIONS_LOG_PATH", str(log_path))
    client = TestClient(app)

    response = client.post("/predict/churn", json=_valid_churn_payload())

    assert response.status_code == 200
    df = pd.read_parquet(log_path)
    assert len(df) == 1
    assert df.iloc[0]["churn_probability"] == 0.73
    assert df.iloc[0]["tenure"] == 5


def test_predict_churn_still_succeeds_if_logging_fails(monkeypatch):
    app.state.churn_model = _FakeChurnModel()

    def _raise(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(app_module, "log_prediction", _raise)
    client = TestClient(app)

    response = client.post("/predict/churn", json=_valid_churn_payload())

    assert response.status_code == 200
    assert response.json()["churn_probability"] == 0.73
```

Note: `app_module` and `_FakeChurnModel`/`_valid_churn_payload` already
exist in this test file from earlier tasks (`import churn_mlops.serving.app
as app_module` and a `_FakeChurnModel` class with a `predict_proba`
returning `[0.73] * len(df)`) — reuse them, don't redefine.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_app.py -v`
Expected: FAIL — `AttributeError: module 'churn_mlops.serving.app' has no attribute 'PREDICTIONS_LOG_PATH'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn_mlops/serving/app.py — add this import near the other churn_mlops imports
from churn_mlops.lib.prediction_logging import log_prediction

# add this constant near FORECAST_FEATURES_PATH
PREDICTIONS_LOG_PATH = "data/predictions/churn_predictions.parquet"
```

Modify the `predict_churn` function body:

```python
@app.post("/predict/churn", response_model=ChurnPredictResponse)
def predict_churn(request: ChurnPredictRequest):
    if app.state.churn_model is None:
        raise HTTPException(status_code=503, detail="churn_model not loaded")

    row = pd.DataFrame([request.model_dump()])[CHURN_FEATURE_COLUMNS]
    probability = float(app.state.churn_model.predict_proba(None, row)[0])
    prediction = probability >= 0.5

    try:
        log_prediction(
            {
                **request.model_dump(),
                "churn_probability": probability,
                "churn_prediction": prediction,
            },
            log_path=PREDICTIONS_LOG_PATH,
        )
    except Exception:
        logger.warning("Failed to log prediction", exc_info=True)

    return ChurnPredictResponse(
        churn_probability=probability,
        churn_prediction=prediction,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_app.py -v`
Expected: PASS (all tests in this file, including the 2 new ones)

- [ ] **Step 5: Commit**

```bash
git add src/churn_mlops/serving/app.py tests/test_app.py
git commit -m "feat: log every churn prediction for drift monitoring"
```

---

### Task 3: `build_drift_report` + `extract_drift_summary` pure functions

**Files:**
- Modify: `pyproject.toml`
- Create: `src/churn_mlops/lib/drift_report.py`
- Test: `tests/test_drift_report.py`

**Interfaces:**
- Produces: `REPORT_COLUMNS: list[str]`,
  `build_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> evidently.report.Report`,
  `extract_drift_summary(report) -> dict` with keys `{"drift_share": float, "dataset_drift_detected": bool}`.
  Both DataFrames must contain `churn_mlops.lib.churn_model.FEATURE_COLUMNS`
  plus a `Churn` column (0/1) — for `current_df`, `Churn` holds the
  model's *predicted* label, standing in for a true target since no
  ground truth exists yet (this is `TargetDriftPreset` used as a
  prediction-drift proxy, per the spec). Consumed by Task 4's
  `drift_report` asset.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_drift_report.py
import numpy as np
import pandas as pd

from churn_mlops.lib.drift_report import build_drift_report, extract_drift_summary


def _synthetic_df(n=200, seed=0, monthly_charges_shift=0.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "tenure": rng.integers(0, 72, size=n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
        "MonthlyCharges": rng.uniform(20, 120, size=n) + monthly_charges_shift,
        "TotalCharges": rng.uniform(0, 8000, size=n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size=n),
        "PaymentMethod": rng.choice(
            ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"], size=n
        ),
        "TechSupport": rng.choice(["No", "No internet service", "Yes"], size=n),
        "current_mrr": rng.uniform(0, 120, size=n),
        "trailing_3mo_avg_mrr": rng.uniform(0, 120, size=n),
        "Churn": rng.integers(0, 2, size=n),
    })


def test_no_drift_when_distributions_match():
    reference_df = _synthetic_df(seed=0)
    current_df = _synthetic_df(seed=0)  # identical (same seed)

    report = build_drift_report(reference_df, current_df)
    summary = extract_drift_summary(report)

    assert summary["dataset_drift_detected"] is False


def test_drift_detected_when_distribution_shifts():
    reference_df = _synthetic_df(seed=0)
    current_df = _synthetic_df(seed=1, monthly_charges_shift=500.0)

    report = build_drift_report(reference_df, current_df)
    summary = extract_drift_summary(report)

    assert summary["drift_share"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_drift_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_mlops.lib.drift_report'` (and `evidently` not yet a declared dependency)

- [ ] **Step 3: Add the dependency and write minimal implementation**

```toml
# pyproject.toml — add to the existing dependencies list
dependencies = [
  "dagster>=1.7",
  "dagster-webserver>=1.7",
  "pandas>=2.2",
  "pyarrow>=16.0",
  "numpy>=1.26",
  "lightgbm>=4.3",
  "scikit-learn>=1.4",
  "mlflow>=2.14",
  "fastapi>=0.110",
  "uvicorn>=0.29",
  "httpx>=0.27",
  "evidently>=0.4.30,<0.5.0",
]
```

```python
# src/churn_mlops/lib/drift_report.py
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

from churn_mlops.lib.churn_model import FEATURE_COLUMNS

REPORT_COLUMNS = FEATURE_COLUMNS + ["Churn"]


def build_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Report:
    column_mapping = ColumnMapping(target="Churn")

    report = Report(metrics=[DataDriftPreset(columns=FEATURE_COLUMNS), TargetDriftPreset()])
    report.run(
        reference_data=reference_df[REPORT_COLUMNS],
        current_data=current_df[REPORT_COLUMNS],
        column_mapping=column_mapping,
    )
    return report


def extract_drift_summary(report: Report) -> dict:
    result = report.as_dict()
    data_drift_result = result["metrics"][0]["result"]
    return {
        "drift_share": data_drift_result["share_of_drifted_columns"],
        "dataset_drift_detected": bool(data_drift_result["dataset_drift"]),
    }
```

**Known risk:** the exact dict keys inside `report.as_dict()["metrics"][0]["result"]`
depend on the installed `evidently` version. If `share_of_drifted_columns`
or `dataset_drift` raise a `KeyError`, run
`python -c "from churn_mlops.lib.drift_report import build_drift_report; import pandas as pd, json; ..."`
with two small DataFrames to print `report.as_dict()` and inspect the
actual key names for the first metric's result — adapt
`extract_drift_summary` to match, and note in your report which keys the
installed version actually used. This is expected troubleshooting for a
fast-moving library, not something to treat as BLOCKED.

- [ ] **Step 4: Install the dependency and run tests to verify they pass**

Run: `pip install -e ".[dev]"` then `pytest tests/test_drift_report.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/churn_mlops/lib/drift_report.py tests/test_drift_report.py
git commit -m "feat: add build_drift_report and extract_drift_summary"
```

---

### Task 4: `drift_report` Dagster asset

**Files:**
- Modify: `.gitignore`
- Create: `src/churn_mlops/assets/drift_report.py`
- Test: `tests/test_drift_report_asset.py`

**Interfaces:**
- Consumes: `build_drift_report`, `extract_drift_summary` (Task 3)
- Produces: asset `drift_report` (no `io_manager_key` — returns a
  `MaterializeResult`, not a DataFrame). Module constants
  `PREDICTIONS_LOG_PATH: str`, `DRIFT_REPORT_HTML_PATH: str` (both read
  fresh at call time, same late-binding convention as elsewhere in this
  codebase). Consumed by Task 5's `retrain_sensor` (via the Dagster event
  log, not a direct import) and Task 6's real-data run.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_drift_report_asset.py
import numpy as np
import pandas as pd
from dagster import asset, materialize

import churn_mlops.assets.drift_report as drift_report_module
from churn_mlops.assets.drift_report import drift_report


def _synthetic_feature_table(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "customerID": [f"c{i}" for i in range(n)],
        "tenure": rng.integers(0, 72, size=n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
        "MonthlyCharges": rng.uniform(20, 120, size=n),
        "TotalCharges": rng.uniform(0, 8000, size=n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size=n),
        "PaymentMethod": rng.choice(
            ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"], size=n
        ),
        "TechSupport": rng.choice(["No", "No internet service", "Yes"], size=n),
        "Churn": rng.integers(0, 2, size=n),
        "current_mrr": rng.uniform(0, 120, size=n),
        "trailing_3mo_avg_mrr": rng.uniform(0, 120, size=n),
    })


@asset(name="feature_table")
def _stub_feature_table():
    return _synthetic_feature_table()


def test_drift_report_materializes_gracefully_with_no_predictions_log(tmp_path, monkeypatch):
    monkeypatch.setattr(drift_report_module, "PREDICTIONS_LOG_PATH", str(tmp_path / "does_not_exist.parquet"))
    monkeypatch.setattr(drift_report_module, "DRIFT_REPORT_HTML_PATH", str(tmp_path / "report.html"))

    result = materialize([_stub_feature_table, drift_report])

    assert result.success
    mats = result.asset_materializations_for_node("drift_report")
    assert mats[0].metadata["dataset_drift_detected"].value is False


def test_drift_report_materializes_and_writes_html_with_predictions(tmp_path, monkeypatch):
    predictions_path = tmp_path / "predictions.parquet"
    html_path = tmp_path / "report.html"
    monkeypatch.setattr(drift_report_module, "PREDICTIONS_LOG_PATH", str(predictions_path))
    monkeypatch.setattr(drift_report_module, "DRIFT_REPORT_HTML_PATH", str(html_path))

    rng = np.random.default_rng(1)
    n = 50
    predictions_df = pd.DataFrame({
        "tenure": rng.integers(0, 72, size=n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
        "MonthlyCharges": rng.uniform(20, 120, size=n),
        "TotalCharges": rng.uniform(0, 8000, size=n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size=n),
        "PaymentMethod": rng.choice(
            ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"], size=n
        ),
        "TechSupport": rng.choice(["No", "No internet service", "Yes"], size=n),
        "current_mrr": rng.uniform(0, 120, size=n),
        "trailing_3mo_avg_mrr": rng.uniform(0, 120, size=n),
        "churn_probability": rng.uniform(0, 1, size=n),
        "churn_prediction": rng.integers(0, 2, size=n),
        "logged_at": ["2026-08-12T00:00:00+00:00"] * n,
    })
    predictions_df.to_parquet(predictions_path)

    result = materialize([_stub_feature_table, drift_report])

    assert result.success
    assert html_path.exists()
    mats = result.asset_materializations_for_node("drift_report")
    assert "drift_share" in mats[0].metadata
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_drift_report_asset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_mlops.assets.drift_report'`

- [ ] **Step 3: Add the gitignore entry and write minimal implementation**

```
# .gitignore — add
reports/
```

```python
# src/churn_mlops/assets/drift_report.py
import os

import pandas as pd
from dagster import MaterializeResult, MetadataValue, asset

from churn_mlops.lib.drift_report import build_drift_report, extract_drift_summary

PREDICTIONS_LOG_PATH = "data/predictions/churn_predictions.parquet"
DRIFT_REPORT_HTML_PATH = "reports/churn_drift_report.html"


@asset
def drift_report(feature_table: pd.DataFrame) -> MaterializeResult:
    if not os.path.exists(PREDICTIONS_LOG_PATH):
        return MaterializeResult(
            metadata={
                "drift_share": MetadataValue.float(0.0),
                "dataset_drift_detected": MetadataValue.bool(False),
                "note": MetadataValue.text("No predictions logged yet"),
            }
        )

    current_df = pd.read_parquet(PREDICTIONS_LOG_PATH)
    current_df = current_df.rename(columns={"churn_prediction": "Churn"})
    current_df["Churn"] = current_df["Churn"].astype(int)

    report = build_drift_report(feature_table, current_df)

    os.makedirs(os.path.dirname(DRIFT_REPORT_HTML_PATH), exist_ok=True)
    report.save_html(DRIFT_REPORT_HTML_PATH)

    summary = extract_drift_summary(report)

    return MaterializeResult(
        metadata={
            "drift_share": MetadataValue.float(summary["drift_share"]),
            "dataset_drift_detected": MetadataValue.bool(summary["dataset_drift_detected"]),
        }
    )
```

If `result.asset_materializations_for_node(...)` or `.metadata[...]` don't
match the installed Dagster version's exact API shape, inspect
`result.all_events` for the `ASSET_MATERIALIZATION` event type and adapt
— same documented-troubleshooting approach as Task 3's Evidently risk.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_drift_report_asset.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add .gitignore src/churn_mlops/assets/drift_report.py tests/test_drift_report_asset.py
git commit -m "feat: add drift_report Dagster asset"
```

---

### Task 5: `retrain_sensor` + Definitions wiring

**Files:**
- Create: `src/churn_mlops/sensors/__init__.py`
- Create: `src/churn_mlops/sensors/retrain_sensor.py`
- Modify: `src/churn_mlops/definitions.py`
- Test: `tests/test_retrain_sensor.py`
- Test: `tests/test_definitions.py`

**Interfaces:**
- Consumes: `churn_model` asset (`churn_mlops.assets.churn_model`,
  existing), the Dagster event log for `AssetKey("drift_report")`
  (Task 4's asset, consumed via the instance's event records, not a
  direct Python import)
- Produces: `churn_retrain_job` (a `define_asset_job` targeting
  `churn_model`), `retrain_sensor` (a `@sensor` bound to that job,
  `default_status=DefaultSensorStatus.STOPPED`). Consumed by Task 6 and
  `definitions.py`'s `sensors=[...]` list.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_retrain_sensor.py
from dagster import (
    MaterializeResult,
    MetadataValue,
    RunRequest,
    SkipReason,
    asset,
    build_sensor_context,
    instance_for_test,
    materialize,
)

from churn_mlops.sensors.retrain_sensor import retrain_sensor


@asset(name="drift_report")
def _stub_drift_report_drifted():
    return MaterializeResult(
        metadata={"drift_share": MetadataValue.float(0.6), "dataset_drift_detected": MetadataValue.bool(True)}
    )


@asset(name="drift_report")
def _stub_drift_report_clean():
    return MaterializeResult(
        metadata={"drift_share": MetadataValue.float(0.1), "dataset_drift_detected": MetadataValue.bool(False)}
    )


def test_sensor_fires_when_drift_detected():
    with instance_for_test() as instance:
        materialize([_stub_drift_report_drifted], instance=instance)

        context = build_sensor_context(instance=instance)
        result = retrain_sensor(context)

        assert isinstance(result, RunRequest)


def test_sensor_skips_when_no_drift():
    with instance_for_test() as instance:
        materialize([_stub_drift_report_clean], instance=instance)

        context = build_sensor_context(instance=instance)
        result = retrain_sensor(context)

        assert isinstance(result, SkipReason)


def test_sensor_does_not_refire_for_same_materialization():
    with instance_for_test() as instance:
        materialize([_stub_drift_report_drifted], instance=instance)

        context1 = build_sensor_context(instance=instance)
        first_result = retrain_sensor(context1)
        assert isinstance(first_result, RunRequest)

        context2 = build_sensor_context(instance=instance, cursor=context1.cursor)
        second_result = retrain_sensor(context2)

        assert isinstance(second_result, SkipReason)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retrain_sensor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_mlops.sensors.retrain_sensor'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn_mlops/sensors/__init__.py
```

```python
# src/churn_mlops/sensors/retrain_sensor.py
from dagster import (
    AssetKey,
    DagsterEventType,
    DefaultSensorStatus,
    EventRecordsFilter,
    RunRequest,
    SensorEvaluationContext,
    SkipReason,
    define_asset_job,
    sensor,
)

from churn_mlops.assets.churn_model import churn_model

DRIFT_REPORT_ASSET_KEY = AssetKey("drift_report")

churn_retrain_job = define_asset_job("churn_retrain_job", selection=[churn_model])


@sensor(job=churn_retrain_job, default_status=DefaultSensorStatus.STOPPED)
def retrain_sensor(context: SensorEvaluationContext):
    event_records = context.instance.get_event_records(
        EventRecordsFilter(
            event_type=DagsterEventType.ASSET_MATERIALIZATION,
            asset_key=DRIFT_REPORT_ASSET_KEY,
        ),
        limit=1,
    )

    if not event_records:
        return SkipReason("No drift_report materialization found yet")

    latest_record = event_records[0]
    materialization = latest_record.event_log_entry.dagster_event.event_specific_data.materialization
    metadata = materialization.metadata

    drift_detected_metadata = metadata.get("dataset_drift_detected")
    if drift_detected_metadata is None or not drift_detected_metadata.value:
        return SkipReason("No drift detected in latest drift_report materialization")

    cursor_key = str(latest_record.storage_id)
    if context.cursor == cursor_key:
        return SkipReason("Already triggered a retrain for this drift_report materialization")

    context.update_cursor(cursor_key)
    return RunRequest(run_key=cursor_key)
```

**Known risk:** `latest_record.event_log_entry.dagster_event.event_specific_data.materialization`
is the general-purpose path to a materialization event's data, but some
Dagster versions also expose a more direct `latest_record.asset_materialization`
convenience property. If the written path raises an `AttributeError`, try
`latest_record.asset_materialization` instead and note in your report
which one the installed version needed.

Wire into `definitions.py`:

```python
# src/churn_mlops/definitions.py
from dagster import Definitions
from churn_mlops.assets import (
    churn,
    mrr,
    features,
    churn_model as churn_model_assets,
    forecast_features as forecast_features_assets,
    forecast_model as forecast_model_assets,
    drift_report as drift_report_assets,
)
from churn_mlops.io.parquet_io_manager import ParquetIOManager
from churn_mlops.sensors.retrain_sensor import retrain_sensor

defs = Definitions(
    assets=[
        churn.raw_churn,
        churn.validated_churn,
        mrr.synthetic_mrr_raw,
        mrr.validated_mrr,
        features.feature_table,
        churn_model_assets.churn_model,
        forecast_features_assets.forecast_features,
        forecast_model_assets.forecast_model,
        drift_report_assets.drift_report,
    ],
    asset_checks=[churn.validated_churn_check, mrr.validated_mrr_check],
    sensors=[retrain_sensor],
    resources={
        "raw_io_manager": ParquetIOManager(base_dir="data/raw"),
        "validated_io_manager": ParquetIOManager(base_dir="data/validated"),
        "feature_io_manager": ParquetIOManager(base_dir="data/features"),
    },
)
```

```python
# tests/test_definitions.py — replace the existing assertion
from churn_mlops.definitions import defs


def test_definitions_registers_nine_assets():
    asset_graph = defs.get_repository_def().asset_graph
    asset_names = {key.path[-1] for key in asset_graph.get_all_asset_keys()}
    assert asset_names == {
        "raw_churn", "validated_churn", "synthetic_mrr_raw", "validated_mrr",
        "feature_table", "churn_model", "forecast_features", "forecast_model",
        "drift_report",
    }


def test_definitions_registers_retrain_sensor():
    sensor_names = {s.name for s in defs.get_repository_def().sensor_defs}
    assert "retrain_sensor" in sensor_names
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retrain_sensor.py tests/test_definitions.py -v`
Expected: PASS (all tests in both files)

- [ ] **Step 5: Commit**

```bash
git add src/churn_mlops/sensors/ src/churn_mlops/definitions.py tests/test_retrain_sensor.py tests/test_definitions.py
git commit -m "feat: add retrain_sensor and wire drift_report + sensor into Definitions"
```

---

### Task 6: Real-data sanity check

**Files:** None — verification only.

**Interfaces:** None — this task consumes the full pipeline built in Tasks 1-5.

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v`
Expected: PASS, all tests green (existing + new from Tasks 1-5)

- [ ] **Step 2: Populate the predictions log with real requests**

Start the API:

```bash
python -m uvicorn churn_mlops.serving.app:app --port 8000
```

In another terminal, send a handful of real requests (repeat with 3-5
different payloads, varying values):

```bash
curl -s -X POST http://localhost:8000/predict/churn -H "Content-Type: application/json" -d "{\"tenure\": 5, \"Contract\": \"Month-to-month\", \"MonthlyCharges\": 50.0, \"TotalCharges\": 250.0, \"InternetService\": \"DSL\", \"PaymentMethod\": \"Electronic check\", \"TechSupport\": \"No\", \"current_mrr\": 50.0, \"trailing_3mo_avg_mrr\": 48.0}"
```

Confirm `data/predictions/churn_predictions.parquet` now exists and has
one row per request:

```bash
python -c "import pandas as pd; print(pd.read_parquet('data/predictions/churn_predictions.parquet'))"
```

Stop the uvicorn server and confirm port 8000 is free before continuing.

- [ ] **Step 3: Materialize `drift_report` against real data**

```bash
python -m dagster asset materialize --select feature_table,drift_report -m churn_mlops.definitions
```

Expected: run succeeds, `reports/churn_drift_report.html` exists and is a
non-trivial HTML file (open it or check its size — should be well over a
few KB, an actual Evidently report, not an empty shell). Confirm the
materialization's logged metadata (visible in the run's stdout or by
querying it, e.g. via a short Python snippet using
`DagsterInstance.get()` and the same `EventRecordsFilter` pattern Task 5
uses) shows a `drift_share` value.

- [ ] **Step 4: Sanity-check the sensor definition loads**

```bash
python -c "from churn_mlops.definitions import defs; print([s.name for s in defs.get_repository_def().sensor_defs])"
```

Expected: prints `['retrain_sensor']`, confirming the sensor is wired
into `Definitions` without import errors. Do not attempt to actually
start the sensor daemon or `dagster dev` for this check — a static import
sanity check is sufficient given `default_status=STOPPED`.

---

## Self-Review Notes

- **Spec coverage:** Architecture (Task 5 Definitions wiring), all 5
  components (Tasks 1, 2, 3, 4, 5), data flow (Task 2/4/5 bodies), error
  handling (Task 2's try/except never breaks the response, Task 4's
  missing-log graceful path, no blocking asset_check anywhere), testing
  (every scenario from the spec's Testing section has a task: pure
  logging tests in Task 1, wiring tests in Task 2, drift-computation
  tests in Task 3, asset-level tests in Task 4, sensor-level tests in
  Task 5, real-data sanity check in Task 6) — all covered.
- **Placeholder scan:** none found — every step has real code, including
  documented fallback guidance for the two named version-uncertainty
  risks (Evidently's `as_dict()` key names in Task 3, Dagster's event
  record attribute path in Task 5) rather than vague "handle it"
  language.
- **Type consistency:** `log_prediction(row: dict, log_path: str = ...)`
  used identically in Task 1's definition and Task 2's call site;
  `build_drift_report`/`extract_drift_summary` signatures identical
  across Task 3's definition and Task 4's usage; `PREDICTIONS_LOG_PATH`
  used as the same string value in Task 2 (`app.py`) and Task 4
  (`assets/drift_report.py`) — these are two independent module
  constants with the same value by design (one is the serving layer's
  write path, the other is the Dagster asset's read path — both point at
  the same file, but there is deliberately no shared import between
  `serving/` and `assets/` here, matching this codebase's existing
  pattern of small, independent modules); `AssetKey("drift_report")` in
  Task 5 matches the asset name Task 4 defines (`def drift_report(...)`
  under `@asset`, no explicit `name=` override, so Dagster's default
  asset key is the function name).
