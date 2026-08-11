# Forecast Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `forecast_features` and `forecast_model` Dagster assets that train a per-customer MRR regression on lag features, track/register/promote it via a new shared MLflow promotion helper, and refactor `churn_model` onto that same helper.

**Architecture:** `forecast_features` (pure function + asset) builds lag/rolling/static features from `validated_mrr` + `feature_table`. `forecast_model` (pure function + asset) trains `LGBMRegressor` on a time-based split and tracks it in MLflow. A new `lib/mlflow_registry.py` centralizes the promote-if-better comparison, used by both `forecast_model` and a refactored `churn_model`.

**Tech Stack:** LightGBM (`lightgbm.LGBMRegressor`), scikit-learn (metrics), MLflow (tracking + registry, same SQLite-backed local store as sub-project 2), Dagster, pytest. No new dependencies — `lightgbm`, `scikit-learn`, `mlflow` are already installed from sub-project 2.

## Global Constraints

- Forecast framing: per-customer regression using each customer's own lag history, not calendar-aggregate forecasting (spec: Decisions made this session)
- Train/test split: time-based — each customer's last valid row is test, earlier rows are train; a customer with only one valid row contributes no test row (spec: Decisions made this session)
- Lag depth: `lag_1`, `lag_2`, `lag_3` (previous 3 months' MRR) plus `rolling_3mo_mean` (spec: Decisions made this session)
- Metric: RMSE drives promotion (`higher_is_better=False`); MAE also logged (spec: Decisions made this session)
- Feature scope: lags + `Contract`, `InternetService`, `PaymentMethod`, `TechSupport` from `feature_table` (spec: Decisions made this session)
- Library: LightGBM (`LGBMRegressor`), same as the churn model's `LGBMClassifier` (spec: Decisions made this session)
- Fixed hyperparameters, no search: `n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42` — same values as the churn model, for consistency
- Promotion: modern alias API (`set_registered_model_alias` / `get_model_version_by_alias`, alias `"production"`) via the new shared `promote_if_better` helper — not the deprecated `stages` API
- `churn_model`'s external behavior must be unchanged after its refactor — its existing asset-level tests pass without modification (spec: Testing)

## Prerequisite

None — `data/validated/validated_mrr.parquet` and `data/features/feature_table.parquet` already exist locally from prior sub-projects, so this plan's final task can run its real-data sanity check directly.

---

## File Structure

```
src/churn_mlops/
  lib/
    forecast_features.py       # build_forecast_features
    forecast_model.py          # train_forecast_model, FEATURE_COLUMNS, DEFAULT_PARAMS
    mlflow_registry.py         # NEW: get_production_metric, promote_if_better (shared)
    churn_model.py             # MODIFY: remove get_production_roc_auc (retired)
  assets/
    forecast_features.py       # forecast_features asset
    forecast_model.py          # forecast_model asset
    churn_model.py             # MODIFY: use promote_if_better instead of inline comparison
  definitions.py                # MODIFY: + forecast_features, forecast_model assets
tests/
  test_forecast_features.py
  test_forecast_features_asset.py
  test_forecast_model.py
  test_forecast_model_asset.py
  test_mlflow_registry.py
  test_churn_model.py          # MODIFY: remove get_production_roc_auc tests
  test_definitions.py           # MODIFY: 8 assets now
```

---

### Task 1: `build_forecast_features` pure function

**Files:**
- Create: `src/churn_mlops/lib/forecast_features.py`
- Test: `tests/test_forecast_features.py`

**Interfaces:**
- Produces: `STATIC_FEATURE_COLUMNS: list[str]`,
  `build_forecast_features(mrr_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame`
  with output columns `customerID`, `month`, `lag_1`, `lag_2`, `lag_3`,
  `rolling_3mo_mean`, `Contract`, `InternetService`, `PaymentMethod`, `TechSupport`,
  `target_mrr`. Consumed by Task 4's `forecast_features` asset.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_forecast_features.py
import pandas as pd
from churn_mlops.lib.forecast_features import build_forecast_features


def _mrr_df():
    return pd.DataFrame({
        "customerID": ["1"] * 6,
        "month": [0, 1, 2, 3, 4, 5],
        "MRR": [50.0, 50.0, 52.0, 51.0, 53.0, 0.0],
    })


def _feature_df():
    return pd.DataFrame({
        "customerID": ["1"],
        "Contract": ["Month-to-month"],
        "InternetService": ["DSL"],
        "PaymentMethod": ["Electronic check"],
        "TechSupport": ["No"],
    })


def test_lag_values_correct_for_multi_month_customer():
    result = build_forecast_features(_mrr_df(), _feature_df())
    row = result[result["month"] == 3].iloc[0]
    assert row["lag_1"] == 52.0
    assert row["lag_2"] == 50.0
    assert row["lag_3"] == 50.0
    assert row["rolling_3mo_mean"] == (52.0 + 50.0 + 50.0) / 3
    assert row["target_mrr"] == 51.0


def test_customer_with_fewer_than_four_months_contributes_no_rows():
    mrr_df = pd.DataFrame({
        "customerID": ["2", "2"],
        "month": [0, 1],
        "MRR": [30.0, 30.0],
    })
    feature_df = pd.DataFrame({
        "customerID": ["2"], "Contract": ["One year"], "InternetService": ["No"],
        "PaymentMethod": ["Mailed check"], "TechSupport": ["Yes"],
    })
    result = build_forecast_features(mrr_df, feature_df)
    assert len(result) == 0


def test_static_features_joined_correctly():
    result = build_forecast_features(_mrr_df(), _feature_df())
    row = result[result["month"] == 3].iloc[0]
    assert row["Contract"] == "Month-to-month"
    assert row["TechSupport"] == "No"


def test_churned_customer_terminal_zero_row_is_valid_target():
    result = build_forecast_features(_mrr_df(), _feature_df())
    row = result[result["month"] == 5].iloc[0]
    assert row["target_mrr"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_forecast_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_mlops.lib.forecast_features'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn_mlops/lib/forecast_features.py
import pandas as pd

STATIC_FEATURE_COLUMNS = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]


def build_forecast_features(mrr_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    mrr_sorted = mrr_df.sort_values(["customerID", "month"]).copy()

    grouped = mrr_sorted.groupby("customerID")["MRR"]
    mrr_sorted["lag_1"] = grouped.shift(1)
    mrr_sorted["lag_2"] = grouped.shift(2)
    mrr_sorted["lag_3"] = grouped.shift(3)
    mrr_sorted["rolling_3mo_mean"] = mrr_sorted[["lag_1", "lag_2", "lag_3"]].mean(axis=1)
    mrr_sorted["target_mrr"] = mrr_sorted["MRR"]

    result = mrr_sorted.dropna(subset=["lag_1", "lag_2", "lag_3"]).copy()

    static_features = feature_df[["customerID"] + STATIC_FEATURE_COLUMNS]
    result = result.merge(static_features, on="customerID", how="left")

    return result[
        ["customerID", "month", "lag_1", "lag_2", "lag_3", "rolling_3mo_mean"]
        + STATIC_FEATURE_COLUMNS
        + ["target_mrr"]
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_forecast_features.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/churn_mlops/lib/forecast_features.py tests/test_forecast_features.py
git commit -m "feat: add build_forecast_features pure function"
```

---

### Task 2: `promote_if_better` shared MLflow helper

**Files:**
- Create: `src/churn_mlops/lib/mlflow_registry.py`
- Test: `tests/test_mlflow_registry.py`

**Interfaces:**
- Produces: `get_production_metric(client, model_name: str, metric_name: str) -> float | None`,
  `promote_if_better(client, model_name: str, metric_name: str, value: float, version: str, higher_is_better: bool) -> bool`.
  Consumed by Task 3's `churn_model` refactor and Task 6's `forecast_model` asset.
  Note: `version` is the model version string to promote TO if the comparison
  wins — required to call `client.set_registered_model_alias`, and not
  optional despite being a late addition to this plan's signature versus the
  spec's prose description.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mlflow_registry.py
from unittest.mock import MagicMock

from mlflow.exceptions import MlflowException

from churn_mlops.lib.mlflow_registry import get_production_metric, promote_if_better


def test_get_production_metric_returns_none_when_no_production_version():
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = MlflowException("not found")
    assert get_production_metric(client, "m", "roc_auc") is None


def test_get_production_metric_returns_value_when_production_exists():
    client = MagicMock()
    version = MagicMock(run_id="run1")
    client.get_model_version_by_alias.return_value = version
    run = MagicMock()
    run.data.metrics = {"roc_auc": 0.9}
    client.get_run.return_value = run
    assert get_production_metric(client, "m", "roc_auc") == 0.9


def test_promote_if_better_promotes_when_no_production_exists():
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = MlflowException("not found")
    promoted = promote_if_better(client, "m", "roc_auc", 0.8, "1", higher_is_better=True)
    assert promoted is True
    client.set_registered_model_alias.assert_called_once_with("m", "production", "1")


def test_promote_if_better_higher_is_better_promotes_when_new_value_wins():
    client = MagicMock()
    version = MagicMock(run_id="run1")
    client.get_model_version_by_alias.return_value = version
    run = MagicMock()
    run.data.metrics = {"roc_auc": 0.7}
    client.get_run.return_value = run
    promoted = promote_if_better(client, "m", "roc_auc", 0.8, "2", higher_is_better=True)
    assert promoted is True
    client.set_registered_model_alias.assert_called_once_with("m", "production", "2")


def test_promote_if_better_higher_is_better_skips_when_new_value_loses():
    client = MagicMock()
    version = MagicMock(run_id="run1")
    client.get_model_version_by_alias.return_value = version
    run = MagicMock()
    run.data.metrics = {"roc_auc": 0.9}
    client.get_run.return_value = run
    promoted = promote_if_better(client, "m", "roc_auc", 0.8, "2", higher_is_better=True)
    assert promoted is False
    client.set_registered_model_alias.assert_not_called()


def test_promote_if_better_lower_is_better_promotes_when_new_value_wins():
    client = MagicMock()
    version = MagicMock(run_id="run1")
    client.get_model_version_by_alias.return_value = version
    run = MagicMock()
    run.data.metrics = {"rmse": 10.0}
    client.get_run.return_value = run
    promoted = promote_if_better(client, "m", "rmse", 8.0, "2", higher_is_better=False)
    assert promoted is True


def test_promote_if_better_lower_is_better_skips_when_new_value_loses():
    client = MagicMock()
    version = MagicMock(run_id="run1")
    client.get_model_version_by_alias.return_value = version
    run = MagicMock()
    run.data.metrics = {"rmse": 8.0}
    client.get_run.return_value = run
    promoted = promote_if_better(client, "m", "rmse", 10.0, "2", higher_is_better=False)
    assert promoted is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mlflow_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_mlops.lib.mlflow_registry'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn_mlops/lib/mlflow_registry.py
from mlflow.exceptions import MlflowException


def get_production_metric(client, model_name: str, metric_name: str) -> float | None:
    try:
        version = client.get_model_version_by_alias(model_name, "production")
    except MlflowException:
        return None
    run = client.get_run(version.run_id)
    return float(run.data.metrics[metric_name])


def promote_if_better(
    client,
    model_name: str,
    metric_name: str,
    value: float,
    version: str,
    higher_is_better: bool,
) -> bool:
    current = get_production_metric(client, model_name, metric_name)

    if current is None:
        should_promote = True
    elif higher_is_better:
        should_promote = value > current
    else:
        should_promote = value < current

    if should_promote:
        client.set_registered_model_alias(model_name, "production", version)

    return should_promote
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mlflow_registry.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/churn_mlops/lib/mlflow_registry.py tests/test_mlflow_registry.py
git commit -m "feat: add shared promote_if_better MLflow helper"
```

---

### Task 3: Refactor `churn_model` onto `promote_if_better`

**Files:**
- Modify: `src/churn_mlops/lib/churn_model.py` — remove `get_production_roc_auc`
- Modify: `src/churn_mlops/assets/churn_model.py` — use `promote_if_better`
- Modify: `tests/test_churn_model.py` — remove the two `get_production_roc_auc` tests

**Interfaces:**
- Consumes: `promote_if_better` (Task 2)
- No new interfaces produced — this task changes internals only. External
  behavior (the `churn_model` asset's return shape and promotion outcomes)
  must be unchanged.

- [ ] **Step 1: Read the current files**

Read `src/churn_mlops/lib/churn_model.py`, `src/churn_mlops/assets/churn_model.py`,
and `tests/test_churn_model.py` in full before making any change — this task
edits real, already-merged code, not a fresh file.

- [ ] **Step 2: Remove `get_production_roc_auc` from `lib/churn_model.py`**

Delete the function `get_production_roc_auc(client, model_name)` and its
`from mlflow.exceptions import MlflowException` import (remove the import
only if nothing else in the file still uses `MlflowException` — check
before deleting the import line). Leave `FEATURE_COLUMNS`,
`CATEGORICAL_COLUMNS`, `DEFAULT_PARAMS`, and `train_churn_model` untouched.

- [ ] **Step 3: Remove its tests from `tests/test_churn_model.py`**

Delete `test_get_production_roc_auc_returns_none_when_no_production_version`
and `test_get_production_roc_auc_returns_metric_when_production_exists`.
Remove the now-unused `from unittest.mock import MagicMock` and
`from mlflow.exceptions import MlflowException` imports from this test file
only if nothing else in it still uses them.

- [ ] **Step 4: Refactor the promotion logic in `assets/churn_model.py`**

Replace the current lookup-and-compare block (something like
`current_production_roc_auc = get_production_roc_auc(client, MODEL_NAME)`
followed by an inline `promoted = current_production_roc_auc is None or
metrics["roc_auc"] > current_production_roc_auc` and a conditional
`client.set_registered_model_alias(...)` call) with a single call to the
shared helper:

```python
from churn_mlops.lib.mlflow_registry import promote_if_better
# (remove the now-unused `from churn_mlops.lib.churn_model import get_production_roc_auc`
# import if the asset imported it separately from train_churn_model)

# ... inside the asset function, after `version = model_info.registered_model_version`:
promoted = promote_if_better(
    client, MODEL_NAME, "roc_auc", metrics["roc_auc"], version, higher_is_better=True
)
```

The asset's overall structure (config class, `train_churn_model` call,
`mlflow.start_run()` block, `log_params`/`log_metrics`/`log_model` with
signature, return dict shape) stays exactly as it is today — only the
promotion-comparison block changes. If the current file's exact variable
names or formatting differ from what's shown above, apply the same
transformation (replace inline comparison + conditional alias-set with one
`promote_if_better(...)` call) to what's actually there rather than forcing
a literal match to this snippet.

- [ ] **Step 5: Run the full test suite to verify nothing broke**

Run: `pytest tests/test_churn_model.py tests/test_churn_model_asset.py -v`
Expected: PASS — `tests/test_churn_model_asset.py`'s existing tests
(`test_first_run_registers_and_promotes`, `test_worse_run_does_not_promote`,
`test_better_run_promotes_and_replaces`, `test_logged_model_has_signature`)
must pass **unmodified**. If any fails, the refactor changed external
behavior — fix the refactor, not the test.

- [ ] **Step 6: Commit**

```bash
git add src/churn_mlops/lib/churn_model.py src/churn_mlops/assets/churn_model.py tests/test_churn_model.py
git commit -m "refactor: churn_model uses shared promote_if_better helper"
```

---

### Task 4: `forecast_features` Dagster asset

**Files:**
- Create: `src/churn_mlops/assets/forecast_features.py`
- Test: `tests/test_forecast_features_asset.py`

**Interfaces:**
- Consumes: `build_forecast_features` (Task 1), `ParquetIOManager` (existing,
  `src/churn_mlops/io/parquet_io_manager.py`)
- Produces: asset `forecast_features` (`io_manager_key="feature_io_manager"`).
  Consumed by Task 6's `forecast_model` asset.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forecast_features_asset.py
import pandas as pd
from dagster import asset, materialize

from churn_mlops.assets.forecast_features import forecast_features
from churn_mlops.io.parquet_io_manager import ParquetIOManager


@asset(name="validated_mrr")
def _stub_validated_mrr():
    return pd.DataFrame({
        "customerID": ["1"] * 6,
        "month": [0, 1, 2, 3, 4, 5],
        "MRR": [50.0, 50.0, 52.0, 51.0, 53.0, 0.0],
    })


@asset(name="feature_table")
def _stub_feature_table():
    return pd.DataFrame({
        "customerID": ["1"],
        "Contract": ["Month-to-month"],
        "InternetService": ["DSL"],
        "PaymentMethod": ["Electronic check"],
        "TechSupport": ["No"],
    })


def test_forecast_features_materializes(tmp_path):
    result = materialize(
        [_stub_validated_mrr, _stub_feature_table, forecast_features],
        resources={"feature_io_manager": ParquetIOManager(base_dir=str(tmp_path / "features"))},
    )
    assert result.success
    df = result.output_for_node("forecast_features")
    assert len(df) == 3
    assert "target_mrr" in df.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_forecast_features_asset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_mlops.assets.forecast_features'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn_mlops/assets/forecast_features.py
import pandas as pd
from dagster import asset

from churn_mlops.lib.forecast_features import build_forecast_features


@asset(io_manager_key="feature_io_manager")
def forecast_features(validated_mrr: pd.DataFrame, feature_table: pd.DataFrame) -> pd.DataFrame:
    return build_forecast_features(validated_mrr, feature_table)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_forecast_features_asset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/churn_mlops/assets/forecast_features.py tests/test_forecast_features_asset.py
git commit -m "feat: add forecast_features Dagster asset"
```

---

### Task 5: `train_forecast_model` pure function

**Files:**
- Create: `src/churn_mlops/lib/forecast_model.py`
- Test: `tests/test_forecast_model.py`

**Interfaces:**
- Produces: `CATEGORICAL_COLUMNS: list[str]`, `FEATURE_COLUMNS: list[str]`,
  `DEFAULT_PARAMS: dict`,
  `train_forecast_model(forecast_df: pd.DataFrame, params: dict | None = None) -> tuple[LGBMRegressor, dict[str, float], pd.DataFrame]`
  where the metrics dict has keys `{"rmse", "mae"}` and the third element is
  the held-out test feature frame (`X_test`). Consumed by Task 6's
  `forecast_model` asset.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_forecast_model.py
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from churn_mlops.lib.forecast_model import train_forecast_model


def _synthetic_forecast_df(n_customers=30, months_per_customer=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_customers):
        base = rng.uniform(20, 100)
        for m in range(months_per_customer):
            rows.append({
                "customerID": f"c{c}",
                "month": m + 3,
                "lag_1": base + rng.normal(0, 2),
                "lag_2": base + rng.normal(0, 2),
                "lag_3": base + rng.normal(0, 2),
                "rolling_3mo_mean": base,
                "Contract": rng.choice(["Month-to-month", "One year", "Two year"]),
                "InternetService": rng.choice(["DSL", "Fiber optic", "No"]),
                "PaymentMethod": rng.choice(["Electronic check", "Mailed check"]),
                "TechSupport": rng.choice(["Yes", "No"]),
                "target_mrr": base + rng.normal(0, 2),
            })
    return pd.DataFrame(rows)


def test_train_forecast_model_returns_expected_metrics():
    df = _synthetic_forecast_df()
    model, metrics, X_test = train_forecast_model(df)
    assert isinstance(model, LGBMRegressor)
    assert set(metrics.keys()) == {"rmse", "mae"}
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0


def test_time_split_respects_order_per_customer():
    df = _synthetic_forecast_df()
    model, metrics, X_test = train_forecast_model(df)
    sorted_df = df.sort_values(["customerID", "month"])
    expected_last_indices = sorted_df.groupby("customerID").tail(1).index
    assert set(X_test.index) == set(expected_last_indices)


def test_single_row_customer_contributes_no_test_row():
    multi_df = _synthetic_forecast_df(n_customers=5, months_per_customer=8)
    single_row = pd.DataFrame([{
        "customerID": "solo",
        "month": 3,
        "lag_1": 40.0, "lag_2": 40.0, "lag_3": 40.0, "rolling_3mo_mean": 40.0,
        "Contract": "One year", "InternetService": "DSL",
        "PaymentMethod": "Mailed check", "TechSupport": "No",
        "target_mrr": 40.0,
    }])
    df = pd.concat([multi_df, single_row], ignore_index=True)
    model, metrics, X_test = train_forecast_model(df)
    solo_index = df[df["customerID"] == "solo"].index[0]
    assert solo_index not in X_test.index
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_forecast_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_mlops.lib.forecast_model'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn_mlops/lib/forecast_model.py
from typing import Any

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

CATEGORICAL_COLUMNS = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]
FEATURE_COLUMNS = ["lag_1", "lag_2", "lag_3", "rolling_3mo_mean"] + CATEGORICAL_COLUMNS

DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.05,
    "random_state": 42,
}


def train_forecast_model(
    forecast_df: pd.DataFrame, params: dict[str, Any] | None = None
) -> tuple[LGBMRegressor, dict[str, float], pd.DataFrame]:
    params = dict(DEFAULT_PARAMS) if params is None else params

    df = forecast_df.sort_values(["customerID", "month"]).copy()
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")

    group_sizes = df.groupby("customerID")["customerID"].transform("size")
    is_last_row = df.groupby("customerID").cumcount(ascending=False) == 0
    is_test = is_last_row & (group_sizes > 1)

    train_df = df[~is_test]
    test_df = df[is_test]

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["target_mrr"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["target_mrr"]

    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_test, y_pred)),
    }
    return model, metrics, X_test
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_forecast_model.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/churn_mlops/lib/forecast_model.py tests/test_forecast_model.py
git commit -m "feat: add train_forecast_model pure function"
```

---

### Task 6: `forecast_model` Dagster asset — first run + signature

**Files:**
- Create: `src/churn_mlops/assets/forecast_model.py`
- Test: `tests/test_forecast_model_asset.py`

**Interfaces:**
- Consumes: `train_forecast_model` (Task 5), `promote_if_better` (Task 2)
- Produces: asset `forecast_model` (Dagster `@asset`, config class
  `ForecastMlflowConfig` with field `tracking_uri: str = "sqlite:///mlflow.db"`),
  returning `dict` with keys `{"run_id", "version", "metrics", "promoted"}`.
  Module-level constant: `MODEL_NAME = "forecast_model"`. Consumed by
  Task 7's promotion tests and Task 8's `Definitions` wiring.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_forecast_model_asset.py
import numpy as np
import pandas as pd
from dagster import RunConfig, asset, materialize
from mlflow.tracking import MlflowClient

from churn_mlops.assets.forecast_model import ForecastMlflowConfig, forecast_model


def _synthetic_forecast_table(n_customers=30, months_per_customer=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_customers):
        base = rng.uniform(20, 100)
        for m in range(months_per_customer):
            rows.append({
                "customerID": f"c{c}",
                "month": m + 3,
                "lag_1": base + rng.normal(0, 2),
                "lag_2": base + rng.normal(0, 2),
                "lag_3": base + rng.normal(0, 2),
                "rolling_3mo_mean": base,
                "Contract": rng.choice(["Month-to-month", "One year", "Two year"]),
                "InternetService": rng.choice(["DSL", "Fiber optic", "No"]),
                "PaymentMethod": rng.choice(["Electronic check", "Mailed check"]),
                "TechSupport": rng.choice(["Yes", "No"]),
                "target_mrr": base + rng.normal(0, 2),
            })
    return pd.DataFrame(rows)


@asset(name="forecast_features")
def _stub_forecast_features():
    return _synthetic_forecast_table()


def test_first_run_registers_and_promotes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    result = materialize(
        [_stub_forecast_features, forecast_model],
        run_config=RunConfig(ops={"forecast_model": ForecastMlflowConfig(tracking_uri=tracking_uri)}),
    )

    assert result.success
    output = result.output_for_node("forecast_model")
    assert output["promoted"] is True

    client = MlflowClient(tracking_uri=tracking_uri)
    production_version = client.get_model_version_by_alias("forecast_model", "production")
    assert production_version.version == output["version"]


def test_logged_model_has_signature(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    result = materialize(
        [_stub_forecast_features, forecast_model],
        run_config=RunConfig(ops={"forecast_model": ForecastMlflowConfig(tracking_uri=tracking_uri)}),
    )
    assert result.success
    output = result.output_for_node("forecast_model")

    import mlflow
    mlflow.set_tracking_uri(tracking_uri)
    model_info = mlflow.models.get_model_info(f"models:/forecast_model/{output['version']}")
    assert model_info.signature is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_forecast_model_asset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_mlops.assets.forecast_model'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn_mlops/assets/forecast_model.py
import mlflow
import mlflow.lightgbm
import pandas as pd
from dagster import Config, asset
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from churn_mlops.lib.forecast_model import train_forecast_model
from churn_mlops.lib.mlflow_registry import promote_if_better

MODEL_NAME = "forecast_model"


class ForecastMlflowConfig(Config):
    tracking_uri: str = "sqlite:///mlflow.db"


@asset
def forecast_model(config: ForecastMlflowConfig, forecast_features: pd.DataFrame) -> dict:
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment("forecast_model")

    model, metrics, X_eval = train_forecast_model(forecast_features)

    with mlflow.start_run() as run:
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)

        y_pred = model.predict(X_eval)
        signature = infer_signature(X_eval, y_pred)
        model_info = mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=X_eval.head(5),
        )

        client = MlflowClient(tracking_uri=config.tracking_uri)
        version = model_info.registered_model_version
        promoted = promote_if_better(
            client, MODEL_NAME, "rmse", metrics["rmse"], version, higher_is_better=False
        )

        return {
            "run_id": run.info.run_id,
            "version": version,
            "metrics": metrics,
            "promoted": promoted,
        }
```

If `model_info.registered_model_version` doesn't work on the installed MLflow
version, fall back to `client.search_model_versions(f"name='{MODEL_NAME}'")`
sorted by version descending, taking the first — same fallback used
successfully in sub-project 2's churn model asset.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_forecast_model_asset.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/churn_mlops/assets/forecast_model.py tests/test_forecast_model_asset.py
git commit -m "feat: add forecast_model Dagster asset with MLflow tracking and promotion"
```

---

### Task 7: Promotion comparison — better run promotes, worse run doesn't

**Files:**
- Test: `tests/test_forecast_model_asset.py`

**Interfaces:**
- Consumes: `forecast_model` asset (Task 6), patches its module-level
  `train_forecast_model` reference
  (`churn_mlops.assets.forecast_model.train_forecast_model`) to control
  metrics deterministically.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_forecast_model_asset.py — add to the existing file
from unittest.mock import patch

import churn_mlops.assets.forecast_model as forecast_model_module


def _fitted_stub_regressor():
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(n_estimators=2, max_depth=2)
    model.fit(pd.DataFrame({"a": [0, 1, 0, 1]}), [10.0, 20.0, 10.0, 20.0])
    return model


def _fixed_metrics(rmse):
    return {"rmse": rmse, "mae": rmse}


def test_worse_run_does_not_promote(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_config = RunConfig(ops={"forecast_model": ForecastMlflowConfig(tracking_uri=tracking_uri)})

    with patch.object(
        forecast_model_module, "train_forecast_model",
        return_value=(_fitted_stub_regressor(), _fixed_metrics(10.0), pd.DataFrame({"a": [0, 1]})),
    ):
        first = materialize([_stub_forecast_features, forecast_model], run_config=run_config)
    assert first.success
    first_version = first.output_for_node("forecast_model")["version"]

    with patch.object(
        forecast_model_module, "train_forecast_model",
        return_value=(_fitted_stub_regressor(), _fixed_metrics(15.0), pd.DataFrame({"a": [0, 1]})),
    ):
        second = materialize([_stub_forecast_features, forecast_model], run_config=run_config)
    assert second.success
    output = second.output_for_node("forecast_model")
    assert output["promoted"] is False

    client = MlflowClient(tracking_uri=tracking_uri)
    production_version = client.get_model_version_by_alias("forecast_model", "production")
    assert production_version.version == first_version
    assert production_version.version != output["version"]


def test_better_run_promotes_and_replaces(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_config = RunConfig(ops={"forecast_model": ForecastMlflowConfig(tracking_uri=tracking_uri)})

    with patch.object(
        forecast_model_module, "train_forecast_model",
        return_value=(_fitted_stub_regressor(), _fixed_metrics(10.0), pd.DataFrame({"a": [0, 1]})),
    ):
        materialize([_stub_forecast_features, forecast_model], run_config=run_config)

    with patch.object(
        forecast_model_module, "train_forecast_model",
        return_value=(_fitted_stub_regressor(), _fixed_metrics(5.0), pd.DataFrame({"a": [0, 1]})),
    ):
        second = materialize([_stub_forecast_features, forecast_model], run_config=run_config)
    assert second.success
    output = second.output_for_node("forecast_model")
    assert output["promoted"] is True

    client = MlflowClient(tracking_uri=tracking_uri)
    production_version = client.get_model_version_by_alias("forecast_model", "production")
    assert production_version.version == output["version"]
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_forecast_model_asset.py -v`
Expected: PASS (4 tests total: first-run, signature, worse-doesn't-promote,
better-promotes) — this task is primarily verification, since Task 2's
`promote_if_better` already implements the correct comparison direction for
`higher_is_better=False`. If `test_worse_run_does_not_promote` fails because
the second (worse, higher RMSE) run still promotes, check that
`higher_is_better=False` is actually passed at the `forecast_model` asset's
`promote_if_better(...)` call site (Task 6, Step 3) — not `True`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_forecast_model_asset.py
git commit -m "test: verify promotion comparison for forecast_model better/worse runs"
```

---

### Task 8: Wire into Definitions + real-data sanity run

**Files:**
- Modify: `src/churn_mlops/definitions.py`
- Modify: `tests/test_definitions.py`

**Interfaces:**
- Consumes: `forecast_features` (Task 4), `forecast_model` (Task 6)
- Produces: updated `defs` registering 8 assets total (the prior 6 from
  sub-projects 1-2 plus `forecast_features` and `forecast_model`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_definitions.py — replace the existing assertion
from churn_mlops.definitions import defs


def test_definitions_registers_eight_assets():
    asset_graph = defs.get_repository_def().asset_graph
    asset_names = {key.path[-1] for key in asset_graph.get_all_asset_keys()}
    assert asset_names == {
        "raw_churn", "validated_churn", "synthetic_mrr_raw", "validated_mrr",
        "feature_table", "churn_model", "forecast_features", "forecast_model",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_definitions.py -v`
Expected: FAIL — `asset_names` is missing `"forecast_features"` and `"forecast_model"`

- [ ] **Step 3: Write minimal implementation**

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
)
from churn_mlops.io.parquet_io_manager import ParquetIOManager

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
    ],
    asset_checks=[churn.validated_churn_check, mrr.validated_mrr_check],
    resources={
        "raw_io_manager": ParquetIOManager(base_dir="data/raw"),
        "validated_io_manager": ParquetIOManager(base_dir="data/validated"),
        "feature_io_manager": ParquetIOManager(base_dir="data/features"),
    },
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_definitions.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/churn_mlops/definitions.py tests/test_definitions.py
git commit -m "feat: wire forecast_features and forecast_model assets into Definitions"
```

- [ ] **Step 6: Real-data sanity run**

`data/validated/validated_mrr.parquet` and `data/features/feature_table.parquet`
already exist locally. Run:

```bash
python -m dagster asset materialize --select feature_table,forecast_features,forecast_model -m churn_mlops.definitions
```

Expected: run succeeds, logs show `forecast_features` and `forecast_model`
materializing, and the final output metrics dict has `rmse`/`mae` values
that are finite, non-negative, and small relative to typical MRR values
(tens of dollars, not thousands) — a sanity check that the model learned
something real, not a formal gate. Confirm the model was registered and
promoted (first run for `forecast_model`, so `promoted` should be `True`).

---

## Self-Review Notes

- **Spec coverage:** Architecture (Task 8 Definitions), all 6 components
  (Tasks 1, 2, 3, 4, 5, 6-7), data flow (Task 4/6 asset bodies), error
  handling (no custom handling added, matching spec), testing (every
  required test scenario has a task: pure-function tests in 1/2/5,
  asset-level promotion scenarios in 6-7, churn_model regression check in
  3, real-data sanity check in 8) — all covered.
- **Placeholder scan:** none found — every step has real code, including
  the Task 3 refactor's explicit fallback guidance for when the current
  file's exact formatting differs from the snippet shown (not a vague
  "handle it appropriately" — a concrete instruction to apply the same
  transformation).
- **Type consistency:** `train_forecast_model`'s 3-tuple return
  (`model, metrics, X_test`) matches `train_churn_model`'s post-fix shape
  from sub-project 2 and is used consistently in Tasks 5, 6, 7;
  `promote_if_better`'s signature (including the `version` parameter
  needed to actually call `set_registered_model_alias`, filled in beyond
  the spec's shorter prose description) is identical across Tasks 2, 3,
  6; `FEATURE_COLUMNS`/`CATEGORICAL_COLUMNS` naming in
  `lib/forecast_model.py` mirrors the churn model's naming convention;
  `MODEL_NAME = "forecast_model"` used consistently for registration,
  promotion, and the signature test's `models:/forecast_model/{version}`
  URI in Task 6.
