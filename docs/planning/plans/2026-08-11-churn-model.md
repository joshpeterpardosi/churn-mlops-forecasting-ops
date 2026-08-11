# Churn Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `churn_model` Dagster asset that trains a LightGBM classifier on `feature_table`, tracks the run in MLflow, and auto-promotes it to Production when it beats the current Production version's ROC-AUC.

**Architecture:** One pure function (`train_churn_model`) does the ML work with zero MLflow imports; one Dagster asset (`churn_model`) owns all MLflow interaction (logging, registration, promotion comparison). Same split as the data layer's `lib/`/`assets/` pattern.

**Tech Stack:** LightGBM (`lightgbm.LGBMClassifier`), scikit-learn (split + metrics), MLflow (tracking + Model Registry, SQLite-backed, no server process needed), Dagster, pytest.

## Global Constraints

- Primary metric: ROC-AUC drives promotion; accuracy/precision/recall/F1 also logged (spec: Decisions made this session)
- MLflow backend: `sqlite:///mlflow.db` local file, queried directly by the MLflow Python client — no `mlflow server` process required (spec: Architecture)
- Promotion: use MLflow's modern **alias** API (`set_registered_model_alias` / `get_model_version_by_alias`, alias name `"production"`) — not the deprecated `stages` API (implementation detail resolved this plan, consistent with spec's "production" language)
- Promote a new version if its ROC-AUC beats the current Production version's, or if no Production version exists yet (spec: Components §3)
- `train_churn_model` has no MLflow imports and is unit-testable standalone (spec: Architecture)
- Categorical columns cast to pandas `category` dtype for LightGBM's native handling: `Contract`, `InternetService`, `PaymentMethod`, `TechSupport` (spec: Components §1)
- Stratified 80/20 train/test split on `Churn`, `random_state=42` (spec: Components §1)
- Fixed hyperparameters, no search: `n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42` (spec: Decisions made this session)

## Prerequisite

None — `data/source/telco_churn.csv` and `feature_table.parquet` already exist locally from sub-project 1's real-data smoke check, so this plan's final manual step can run against real data directly (no Kaggle download needed this time).

---

## File Structure

```
pyproject.toml                              # + lightgbm, scikit-learn, mlflow deps
src/churn_mlops/
  lib/
    churn_model.py                          # train_churn_model, get_production_roc_auc
  assets/
    churn_model.py                          # churn_model Dagster asset
  definitions.py                            # + churn_model asset registration
tests/
  test_churn_model.py                       # pure-function tests
  test_churn_model_asset.py                 # asset-level MLflow tests
  test_definitions.py                       # extended: 6 assets now
mlflow.db, mlartifacts/                     # gitignored, created at runtime
```

---

### Task 1: Add ML dependencies + `train_churn_model` pure function

**Files:**
- Modify: `pyproject.toml`
- Create: `src/churn_mlops/lib/churn_model.py`
- Test: `tests/test_churn_model.py`

**Interfaces:**
- Produces: `DEFAULT_PARAMS: dict`, `CATEGORICAL_COLUMNS: list[str]`,
  `train_churn_model(feature_df: pd.DataFrame, params: dict | None = None) -> tuple[LGBMClassifier, dict[str, float]]`
  where the returned dict has keys `{"roc_auc", "accuracy", "precision", "recall", "f1"}`.
  Consumed by Task 5's `churn_model` asset.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_churn_model.py
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from churn_mlops.lib.churn_model import train_churn_model


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
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n
        ),
        "TechSupport": rng.choice(["Yes", "No"], size=n),
        "Churn": rng.integers(0, 2, size=n),
        "current_mrr": rng.uniform(0, 120, size=n),
        "trailing_3mo_avg_mrr": rng.uniform(0, 120, size=n),
    })


def test_train_churn_model_returns_expected_metrics():
    df = _synthetic_feature_table()
    model, metrics = train_churn_model(df)
    assert isinstance(model, LGBMClassifier)
    assert set(metrics.keys()) == {"roc_auc", "accuracy", "precision", "recall", "f1"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_train_churn_model_uses_default_params_when_none_given():
    df = _synthetic_feature_table()
    model, _ = train_churn_model(df)
    params = model.get_params()
    assert params["n_estimators"] == 100
    assert params["max_depth"] == 5
    assert params["learning_rate"] == 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_churn_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_mlops.lib.churn_model'` (and `lightgbm` not yet a declared dependency)

- [ ] **Step 3: Add dependencies and write minimal implementation**

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
]
```

```python
# src/churn_mlops/lib/churn_model.py
from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

CATEGORICAL_COLUMNS = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]

DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.05,
    "random_state": 42,
}


def train_churn_model(
    feature_df: pd.DataFrame, params: dict[str, Any] | None = None
) -> tuple[LGBMClassifier, dict[str, float]]:
    params = dict(DEFAULT_PARAMS) if params is None else params

    df = feature_df.drop(columns=["customerID"]).copy()
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    return model, metrics
```

- [ ] **Step 4: Install deps and run tests to verify they pass**

Run: `pip install -e ".[dev]"` then `pytest tests/test_churn_model.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/churn_mlops/lib/churn_model.py tests/test_churn_model.py
git commit -m "feat: add train_churn_model pure function"
```

---

### Task 2: `get_production_roc_auc` pure-ish helper

**Files:**
- Modify: `src/churn_mlops/lib/churn_model.py`
- Test: `tests/test_churn_model.py`

**Interfaces:**
- Produces: `get_production_roc_auc(client, model_name: str) -> float | None`.
  Consumed by Task 5's `churn_model` asset. Takes any object exposing MLflow's
  `MlflowClient.get_model_version_by_alias` / `get_run` interface — tests pass a mock.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_churn_model.py — add to the existing file
from unittest.mock import MagicMock

from mlflow.exceptions import MlflowException

from churn_mlops.lib.churn_model import get_production_roc_auc


def test_get_production_roc_auc_returns_none_when_no_production_version():
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = MlflowException("not found")
    assert get_production_roc_auc(client, "churn_model") is None


def test_get_production_roc_auc_returns_metric_when_production_exists():
    client = MagicMock()
    version = MagicMock(run_id="run123")
    client.get_model_version_by_alias.return_value = version
    run = MagicMock()
    run.data.metrics = {"roc_auc": 0.87}
    client.get_run.return_value = run

    assert get_production_roc_auc(client, "churn_model") == 0.87
    client.get_model_version_by_alias.assert_called_once_with("churn_model", "production")
    client.get_run.assert_called_once_with("run123")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_churn_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'get_production_roc_auc'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn_mlops/lib/churn_model.py — add to the existing file
from mlflow.exceptions import MlflowException


def get_production_roc_auc(client, model_name: str) -> float | None:
    try:
        version = client.get_model_version_by_alias(model_name, "production")
    except MlflowException:
        return None
    run = client.get_run(version.run_id)
    return float(run.data.metrics["roc_auc"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_churn_model.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/churn_mlops/lib/churn_model.py tests/test_churn_model.py
git commit -m "feat: add get_production_roc_auc helper"
```

---

### Task 3: `churn_model` Dagster asset — first run registers and promotes

**Files:**
- Create: `src/churn_mlops/assets/churn_model.py`
- Test: `tests/test_churn_model_asset.py`

**Interfaces:**
- Consumes: `train_churn_model`, `get_production_roc_auc` (Task 1/2)
- Produces: asset `churn_model` (Dagster `@asset`, config class `MlflowConfig` with field
  `tracking_uri: str = "sqlite:///mlflow.db"`), returning
  `dict` with keys `{"run_id", "version", "metrics", "promoted"}`.
  Consumed by Task 5's `Definitions` wiring.
- Module-level constant: `MODEL_NAME = "churn_model"` (the MLflow registered model name).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_churn_model_asset.py
import numpy as np
import pandas as pd
from dagster import RunConfig, asset, materialize
from mlflow.tracking import MlflowClient

from churn_mlops.assets.churn_model import MlflowConfig, churn_model


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
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n
        ),
        "TechSupport": rng.choice(["Yes", "No"], size=n),
        "Churn": rng.integers(0, 2, size=n),
        "current_mrr": rng.uniform(0, 120, size=n),
        "trailing_3mo_avg_mrr": rng.uniform(0, 120, size=n),
    })


@asset(name="feature_table")
def _stub_feature_table():
    return _synthetic_feature_table()


def test_first_run_registers_and_promotes(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    result = materialize(
        [_stub_feature_table, churn_model],
        run_config=RunConfig(ops={"churn_model": MlflowConfig(tracking_uri=tracking_uri)}),
    )

    assert result.success
    output = result.output_for_node("churn_model")
    assert output["promoted"] is True

    client = MlflowClient(tracking_uri=tracking_uri)
    production_version = client.get_model_version_by_alias("churn_model", "production")
    assert production_version.version == output["version"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_churn_model_asset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_mlops.assets.churn_model'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn_mlops/assets/churn_model.py
import mlflow
import mlflow.lightgbm
import pandas as pd
from dagster import Config, asset
from mlflow.tracking import MlflowClient

from churn_mlops.lib.churn_model import get_production_roc_auc, train_churn_model

MODEL_NAME = "churn_model"


class MlflowConfig(Config):
    tracking_uri: str = "sqlite:///mlflow.db"


@asset
def churn_model(config: MlflowConfig, feature_table: pd.DataFrame) -> dict:
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment("churn_model")

    model, metrics = train_churn_model(feature_table)

    with mlflow.start_run() as run:
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        model_info = mlflow.lightgbm.log_model(
            model, artifact_path="model", registered_model_name=MODEL_NAME
        )

        client = MlflowClient(tracking_uri=config.tracking_uri)
        version = model_info.registered_model_version
        current_production_roc_auc = get_production_roc_auc(client, MODEL_NAME)

        promoted = current_production_roc_auc is None or metrics["roc_auc"] > current_production_roc_auc
        if promoted:
            client.set_registered_model_alias(MODEL_NAME, "production", version)

        return {
            "run_id": run.info.run_id,
            "version": version,
            "metrics": metrics,
            "promoted": promoted,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_churn_model_asset.py -v`
Expected: PASS

If `model_info.registered_model_version` doesn't exist on the installed MLflow
version's `ModelInfo` return type, fall back to reading the version from
`client.search_model_versions(f"name='{MODEL_NAME}'")` sorted by version
descending, taking the first — note which approach worked in the report.

- [ ] **Step 5: Commit**

```bash
git add src/churn_mlops/assets/churn_model.py tests/test_churn_model_asset.py
git commit -m "feat: add churn_model Dagster asset with MLflow tracking and promotion"
```

---

### Task 4: Promotion comparison — better run promotes, worse run doesn't

**Files:**
- Test: `tests/test_churn_model_asset.py`

**Interfaces:**
- Consumes: `churn_model` asset (Task 3), patches its module-level `train_churn_model`
  reference (`churn_mlops.assets.churn_model.train_churn_model`) to control metrics
  deterministically instead of relying on LightGBM's real (noisy) output on random labels.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_churn_model_asset.py — add to the existing file
from unittest.mock import patch

import churn_mlops.assets.churn_model as churn_model_module


def _fitted_stub_model():
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(n_estimators=2, max_depth=2)
    model.fit(pd.DataFrame({"a": [0, 1, 0, 1]}), [0, 1, 0, 1])
    return model


def _fixed_metrics(roc_auc):
    return {"roc_auc": roc_auc, "accuracy": roc_auc, "precision": roc_auc, "recall": roc_auc, "f1": roc_auc}


def test_worse_run_does_not_promote(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_config = RunConfig(ops={"churn_model": MlflowConfig(tracking_uri=tracking_uri)})

    with patch.object(
        churn_model_module, "train_churn_model",
        return_value=(_fitted_stub_model(), _fixed_metrics(0.70)),
    ):
        first = materialize([_stub_feature_table, churn_model], run_config=run_config)
    assert first.success
    first_version = first.output_for_node("churn_model")["version"]

    with patch.object(
        churn_model_module, "train_churn_model",
        return_value=(_fitted_stub_model(), _fixed_metrics(0.60)),
    ):
        second = materialize([_stub_feature_table, churn_model], run_config=run_config)
    assert second.success
    output = second.output_for_node("churn_model")
    assert output["promoted"] is False

    client = MlflowClient(tracking_uri=tracking_uri)
    production_version = client.get_model_version_by_alias("churn_model", "production")
    assert production_version.version == first_version
    assert production_version.version != output["version"]


def test_better_run_promotes_and_replaces(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    run_config = RunConfig(ops={"churn_model": MlflowConfig(tracking_uri=tracking_uri)})

    with patch.object(
        churn_model_module, "train_churn_model",
        return_value=(_fitted_stub_model(), _fixed_metrics(0.70)),
    ):
        materialize([_stub_feature_table, churn_model], run_config=run_config)

    with patch.object(
        churn_model_module, "train_churn_model",
        return_value=(_fitted_stub_model(), _fixed_metrics(0.85)),
    ):
        second = materialize([_stub_feature_table, churn_model], run_config=run_config)
    assert second.success
    output = second.output_for_node("churn_model")
    assert output["promoted"] is True

    client = MlflowClient(tracking_uri=tracking_uri)
    production_version = client.get_model_version_by_alias("churn_model", "production")
    assert production_version.version == output["version"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_churn_model_asset.py -v`
Expected: FAIL — at this point Task 3's asset always promotes on the first run, but the
promotion-comparison logic already written in Task 3 should make `test_worse_run_does_not_promote`
pass and `test_better_run_promotes_and_replaces` pass too, UNLESS `model_info.registered_model_version`
or the alias calls behave differently under repeated registration — run this step for real and
report exactly what fails, since Task 3's implementation is meant to already satisfy this
task's requirements and this is primarily a verification task.

- [ ] **Step 3: Fix implementation if needed**

If both tests pass immediately, there is no implementation change to make — the promotion
logic from Task 3 already satisfies this task. If `test_worse_run_does_not_promote` fails
because the second run still promotes, check the comparison operator in
`src/churn_mlops/assets/churn_model.py` (`metrics["roc_auc"] > current_production_roc_auc`,
strictly greater-than, not `>=`) and that `current_production_roc_auc` is being read fresh
on each run (not cached across materializations).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_churn_model_asset.py -v`
Expected: PASS (3 tests: first-run, worse-doesn't-promote, better-promotes)

- [ ] **Step 5: Commit**

```bash
git add tests/test_churn_model_asset.py
git commit -m "test: verify promotion comparison for better/worse runs"
```

(If Step 3 required a code change, also `git add src/churn_mlops/assets/churn_model.py`
in this commit.)

---

### Task 5: Wire into Definitions + real-data sanity run

**Files:**
- Modify: `src/churn_mlops/definitions.py`
- Modify: `tests/test_definitions.py`

**Interfaces:**
- Consumes: `churn_model` asset (Task 3)
- Produces: updated `defs` registering 6 assets total (5 from the data layer + `churn_model`)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_definitions.py — replace the existing assertion
from churn_mlops.definitions import defs


def test_definitions_registers_six_assets():
    asset_graph = defs.get_repository_def().asset_graph
    asset_names = {key.path[-1] for key in asset_graph.get_all_asset_keys()}
    assert asset_names == {
        "raw_churn", "validated_churn", "synthetic_mrr_raw", "validated_mrr",
        "feature_table", "churn_model",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_definitions.py -v`
Expected: FAIL — `asset_names` is missing `"churn_model"`

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn_mlops/definitions.py
from dagster import Definitions
from churn_mlops.assets import churn, mrr, features, churn_model as churn_model_assets
from churn_mlops.io.parquet_io_manager import ParquetIOManager

defs = Definitions(
    assets=[
        churn.raw_churn,
        churn.validated_churn,
        mrr.synthetic_mrr_raw,
        mrr.validated_mrr,
        features.feature_table,
        churn_model_assets.churn_model,
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
git commit -m "feat: wire churn_model asset into Definitions"
```

- [ ] **Step 6: Real-data sanity run**

`data/source/telco_churn.csv` already exists locally (from sub-project 1). Run:

```bash
python -m dagster asset materialize --select feature_table,churn_model -m churn_mlops.definitions
```

Expected: run succeeds, logs show `churn_model` materializing, and the final output
metrics dict has `roc_auc` meaningfully above 0.5 (real signal, unlike the random-label
synthetic fixtures used in tests). Confirm a `mlflow.db` file and `mlartifacts/` directory
now exist at the repo root, and that they're covered by `.gitignore` (add `mlflow.db` and
`mlartifacts/` to `.gitignore` if not already covered by an existing pattern — check first).

---

## Self-Review Notes

- **Spec coverage:** Architecture (Task 5 Definitions), all 3 components (Tasks 1-2, 3, 3-4),
  data flow (Task 3's asset body), error handling (no custom handling added anywhere,
  matching the spec's "let exceptions propagate" call), testing (every required test
  scenario from the spec's Testing section has a task: pure-function tests in Task 1-2,
  the three asset-level promotion scenarios in Task 3-4, real-data sanity check in Task 5) — all covered.
- **Placeholder scan:** none found — every step has real code, including the fallback
  guidance in Task 3 Step 4 for a possible MLflow API signature mismatch (named
  explicitly, not a vague "handle errors").
- **Type consistency:** `train_churn_model` signature and return shape identical across
  Task 1's definition and Task 3/4's usage; `get_production_roc_auc(client, model_name)`
  identical across Task 2's definition and Task 3's usage; `MlflowConfig.tracking_uri`
  used consistently in all `RunConfig` calls across Task 3 and Task 4's tests; `MODEL_NAME`
  constant used consistently for both registration and alias lookups.
