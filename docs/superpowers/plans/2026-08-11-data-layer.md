# Data Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Dagster asset pipeline (raw_churn → validated_churn → synthetic_mrr_raw → validated_mrr → feature_table) that turns the Telco Customer Churn CSV + a generated MRR series into a single feature table.

**Architecture:** Five Dagster assets in a linear chain, each backed by a pure, independently-testable function. Three `ParquetIOManager` instances (raw/validated/features) persist assets to local Parquet. Two blocking `asset_check`s (on `validated_churn` and `validated_mrr`) halt the pipeline on bad data.

**Tech Stack:** Python 3.11, Dagster (`dagster`, `dagster-webserver`), pandas, pyarrow, numpy, pytest.

## Global Constraints

- Storage format: Parquet on local disk only — `data/raw/`, `data/validated/`, `data/features/` (spec: Architecture)
- Asset checks use `blocking=True` (severity ERROR equivalent) — never `WARN` (spec: Error Handling)
- Curated feature columns only: `tenure`, `Contract`, `MonthlyCharges`, `TotalCharges`, `InternetService`, `PaymentMethod`, `TechSupport`, `Churn` (spec: Components §5)
- MRR generation: flat at `MonthlyCharges` from month 0 to `tenure-1`; drop to 0 at month `tenure` only if `Churn==1`; Gaussian noise + monthly seasonality overlay, MRR never negative (spec: Components §3)
- All asset logic lives in pure functions (DataFrame in, DataFrame out), separate from Dagster IO boilerplate (spec: Error Handling)

## Prerequisite (manual, not a task)

Download the Kaggle "Telco Customer Churn" CSV and save it at
`data/source/telco_churn.csv`. This requires a Kaggle account/login, so it
cannot be automated here. Tasks 4, 7, and 9 use a small fixture CSV for
tests and don't require the real file — only the final manual smoke check
in Task 10 does.

---

## File Structure

```
pyproject.toml
data/
  source/               # manual: telco_churn.csv goes here
  raw/                  # ParquetIOManager output
  validated/            # ParquetIOManager output
  features/             # ParquetIOManager output
src/churn_mlops/
  __init__.py
  io/
    __init__.py
    parquet_io_manager.py
  lib/
    __init__.py
    churn_validation.py
    mrr_generation.py
    mrr_validation.py
    feature_join.py
  assets/
    __init__.py
    churn.py
    mrr.py
    features.py
  definitions.py
tests/
  __init__.py
  fixtures/
    tiny_churn_valid.csv
    tiny_churn_invalid.csv
  test_package.py
  test_parquet_io_manager.py
  test_churn_validation.py
  test_mrr_generation.py
  test_mrr_validation.py
  test_feature_join.py
  test_churn_assets.py
  test_mrr_assets.py
  test_feature_table_asset.py
  test_definitions.py
```

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/churn_mlops/__init__.py`
- Create: `tests/__init__.py`
- Test: `tests/test_package.py`

**Interfaces:**
- Produces: `churn_mlops.__version__` (str) — used by no later task, sanity check only

- [ ] **Step 1: Write the failing test**

```python
# tests/test_package.py
import churn_mlops

def test_package_version():
    assert churn_mlops.__version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_package.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_mlops'`

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[project]
name = "churn-mlops-forecasting-ops"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "dagster>=1.7",
  "dagster-webserver>=1.7",
  "pandas>=2.2",
  "pyarrow>=16.0",
  "numpy>=1.26",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
where = ["src"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

```python
# src/churn_mlops/__init__.py
__version__ = "0.1.0"
```

```python
# tests/__init__.py
```

- [ ] **Step 4: Install package in editable mode and run test to verify it passes**

Run: `pip install -e ".[dev]"` then `pytest tests/test_package.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/churn_mlops/__init__.py tests/__init__.py tests/test_package.py
git commit -m "chore: scaffold churn_mlops package"
```

---

### Task 2: Parquet IOManager

**Files:**
- Create: `src/churn_mlops/io/__init__.py`
- Create: `src/churn_mlops/io/parquet_io_manager.py`
- Test: `tests/test_parquet_io_manager.py`

**Interfaces:**
- Produces: `ParquetIOManager(base_dir: str)` — Dagster `IOManager` subclass with `handle_output(context, obj: pd.DataFrame)` and `load_input(context) -> pd.DataFrame`. Consumed by Task 4, 7, 9 as `io_manager_key` resources.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parquet_io_manager.py
import pandas as pd
from dagster import build_output_context, build_input_context
from churn_mlops.io.parquet_io_manager import ParquetIOManager

def test_round_trip(tmp_path):
    manager = ParquetIOManager(base_dir=str(tmp_path))
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    out_context = build_output_context(name="my_asset")
    manager.handle_output(out_context, df)

    in_context = build_input_context(upstream_output=out_context)
    result = manager.load_input(in_context)

    pd.testing.assert_frame_equal(result, df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parquet_io_manager.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_mlops.io'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn_mlops/io/__init__.py
```

```python
# src/churn_mlops/io/parquet_io_manager.py
import os
import pandas as pd
from dagster import IOManager, InputContext, OutputContext


class ParquetIOManager(IOManager):
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def _path_for_name(self, name: str) -> str:
        return os.path.join(self.base_dir, f"{name}.parquet")

    def handle_output(self, context: OutputContext, obj: pd.DataFrame):
        path = self._path_for_name(context.name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        obj.to_parquet(path, index=False)

    def load_input(self, context: InputContext) -> pd.DataFrame:
        path = self._path_for_name(context.upstream_output.name)
        return pd.read_parquet(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_parquet_io_manager.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/churn_mlops/io/ tests/test_parquet_io_manager.py
git commit -m "feat: add ParquetIOManager for Dagster asset storage"
```

---

### Task 3: Churn validation pure functions

**Files:**
- Create: `src/churn_mlops/lib/__init__.py`
- Create: `src/churn_mlops/lib/churn_validation.py`
- Test: `tests/test_churn_validation.py`

**Interfaces:**
- Produces: `clean_total_charges(df: pd.DataFrame) -> pd.DataFrame`, `map_churn_label(df: pd.DataFrame) -> pd.DataFrame`, `validate_churn(df: pd.DataFrame) -> list[str]` — consumed by Task 4's `validated_churn` asset and its asset check.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_churn_validation.py
import pandas as pd
from churn_mlops.lib.churn_validation import (
    clean_total_charges,
    map_churn_label,
    validate_churn,
)

def _valid_df():
    return pd.DataFrame({
        "customerID": ["1", "2"],
        "tenure": [5, 10],
        "MonthlyCharges": [50.0, 70.0],
        "TotalCharges": ["250.0", "700.0"],
        "Churn": ["Yes", "No"],
    })

def test_clean_total_charges_handles_blank_strings():
    df = _valid_df()
    df.loc[0, "TotalCharges"] = "  "
    result = clean_total_charges(df)
    assert pd.isna(result.loc[0, "TotalCharges"])
    assert result.loc[1, "TotalCharges"] == 700.0

def test_map_churn_label_converts_yes_no_to_int():
    df = _valid_df()
    result = map_churn_label(df)
    assert result["Churn"].tolist() == [1, 0]

def test_validate_churn_passes_on_clean_data():
    df = map_churn_label(clean_total_charges(_valid_df()))
    errors = validate_churn(df)
    assert errors == []

def test_validate_churn_catches_missing_columns():
    df = _valid_df().drop(columns=["tenure"])
    errors = validate_churn(df)
    assert any("missing columns" in e for e in errors)

def test_validate_churn_catches_negative_tenure():
    df = map_churn_label(clean_total_charges(_valid_df()))
    df.loc[0, "tenure"] = -1
    errors = validate_churn(df)
    assert any("tenure" in e for e in errors)

def test_validate_churn_catches_nulls():
    df = map_churn_label(clean_total_charges(_valid_df()))
    df.loc[0, "TotalCharges"] = None
    errors = validate_churn(df)
    assert any("TotalCharges" in e for e in errors)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_churn_validation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_mlops.lib'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn_mlops/lib/__init__.py
```

```python
# src/churn_mlops/lib/churn_validation.py
import pandas as pd

REQUIRED_COLUMNS = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
    df["TotalCharges"] = df["TotalCharges"].replace("", None)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def map_churn_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def validate_churn(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"missing columns: {missing_cols}")
        return errors

    for col in ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]:
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            errors.append(f"{col} has {n_null} null values")

    if (df["tenure"] < 0).any():
        errors.append("tenure has negative values")
    if (df["MonthlyCharges"] < 0).any():
        errors.append("MonthlyCharges has negative values")
    if (df["TotalCharges"] < 0).any():
        errors.append("TotalCharges has negative values")
    if not df["Churn"].isin([0, 1]).all():
        errors.append("Churn has values outside {0, 1}")

    return errors
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_churn_validation.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/churn_mlops/lib/__init__.py src/churn_mlops/lib/churn_validation.py tests/test_churn_validation.py
git commit -m "feat: add churn cleaning and validation functions"
```

---

### Task 4: raw_churn + validated_churn Dagster assets

**Files:**
- Create: `src/churn_mlops/assets/__init__.py`
- Create: `src/churn_mlops/assets/churn.py`
- Create: `tests/fixtures/tiny_churn_valid.csv`
- Create: `tests/fixtures/tiny_churn_invalid.csv`
- Test: `tests/test_churn_assets.py`

**Interfaces:**
- Consumes: `ParquetIOManager` (Task 2), `clean_total_charges`/`map_churn_label`/`validate_churn` (Task 3)
- Produces: assets `raw_churn`, `validated_churn`; asset check `validated_churn_check`. Consumed by Task 7 (`synthetic_mrr_raw` takes `validated_churn` as input) and Task 9 (`feature_table`).

- [ ] **Step 1: Create fixture CSVs**

```csv
# tests/fixtures/tiny_churn_valid.csv
customerID,tenure,MonthlyCharges,TotalCharges,Contract,InternetService,PaymentMethod,TechSupport,Churn
1,5,50.0,250.0,Month-to-month,DSL,Electronic check,No,Yes
2,10,70.0,700.0,One year,Fiber optic,Mailed check,Yes,No
3,1,30.0, ,Month-to-month,DSL,Electronic check,No,No
```

```csv
# tests/fixtures/tiny_churn_invalid.csv
customerID,tenure,MonthlyCharges,TotalCharges,Contract,InternetService,PaymentMethod,TechSupport,Churn
1,-3,50.0,250.0,Month-to-month,DSL,Electronic check,No,Yes
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_churn_assets.py
from dagster import materialize, RunConfig
from churn_mlops.io.parquet_io_manager import ParquetIOManager
from churn_mlops.assets.churn import raw_churn, validated_churn, validated_churn_check, RawChurnConfig

def _resources(tmp_path):
    return {
        "raw_io_manager": ParquetIOManager(base_dir=str(tmp_path / "raw")),
        "validated_io_manager": ParquetIOManager(base_dir=str(tmp_path / "validated")),
    }

def test_validated_churn_passes_on_valid_fixture(tmp_path):
    result = materialize(
        [raw_churn, validated_churn, validated_churn_check],
        resources=_resources(tmp_path),
        run_config=RunConfig(
            ops={"raw_churn": RawChurnConfig(source_path="tests/fixtures/tiny_churn_valid.csv")}
        ),
    )
    assert result.success
    check_result = result.get_asset_check_evaluations()[0]
    assert check_result.passed

    df = result.output_for_node("validated_churn")
    assert df["Churn"].tolist() == [1, 0, 0]
    assert df.loc[df["customerID"] == "3", "TotalCharges"].isna().all()

def test_validated_churn_check_fails_on_invalid_fixture(tmp_path):
    result = materialize(
        [raw_churn, validated_churn, validated_churn_check],
        resources=_resources(tmp_path),
        run_config=RunConfig(
            ops={"raw_churn": RawChurnConfig(source_path="tests/fixtures/tiny_churn_invalid.csv")}
        ),
        raise_on_error=False,
    )
    assert not result.success
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_churn_assets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_mlops.assets'`

- [ ] **Step 4: Write minimal implementation**

```python
# src/churn_mlops/assets/__init__.py
```

```python
# src/churn_mlops/assets/churn.py
import pandas as pd
from dagster import asset, asset_check, AssetCheckResult, Config
from churn_mlops.lib.churn_validation import clean_total_charges, map_churn_label, validate_churn


class RawChurnConfig(Config):
    source_path: str = "data/source/telco_churn.csv"


@asset(io_manager_key="raw_io_manager")
def raw_churn(config: RawChurnConfig) -> pd.DataFrame:
    return pd.read_csv(config.source_path)


@asset(io_manager_key="validated_io_manager")
def validated_churn(raw_churn: pd.DataFrame) -> pd.DataFrame:
    df = clean_total_charges(raw_churn)
    df = map_churn_label(df)
    return df


@asset_check(asset=validated_churn, blocking=True)
def validated_churn_check(validated_churn: pd.DataFrame) -> AssetCheckResult:
    errors = validate_churn(validated_churn)
    return AssetCheckResult(passed=len(errors) == 0, metadata={"errors": errors})
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_churn_assets.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add src/churn_mlops/assets/__init__.py src/churn_mlops/assets/churn.py tests/fixtures/tiny_churn_valid.csv tests/fixtures/tiny_churn_invalid.csv tests/test_churn_assets.py
git commit -m "feat: add raw_churn and validated_churn Dagster assets"
```

---

### Task 5: MRR generation pure function

**Files:**
- Create: `src/churn_mlops/lib/mrr_generation.py`
- Test: `tests/test_mrr_generation.py`

**Interfaces:**
- Produces: `generate_mrr_series(churn_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame` with columns `customerID`, `month`, `MRR`. Consumed by Task 7's `synthetic_mrr_raw` asset.
- Consumes: DataFrame with columns `customerID`, `tenure`, `MonthlyCharges`, `Churn` (int 0/1) — the shape produced by Task 3/4's `validated_churn`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mrr_generation.py
import pandas as pd
from churn_mlops.lib.mrr_generation import generate_mrr_series

def test_churned_customer_drops_to_zero_at_tenure_month():
    df = pd.DataFrame({
        "customerID": ["1"], "tenure": [3], "MonthlyCharges": [50.0], "Churn": [1],
    })
    result = generate_mrr_series(df)
    last_row = result[result["customerID"] == "1"].sort_values("month").iloc[-1]
    assert last_row["month"] == 3
    assert last_row["MRR"] == 0.0

def test_active_customer_never_drops_to_zero():
    df = pd.DataFrame({
        "customerID": ["2"], "tenure": [4], "MonthlyCharges": [70.0], "Churn": [0],
    })
    result = generate_mrr_series(df)
    cust = result[result["customerID"] == "2"]
    assert cust["month"].max() == 3
    assert (cust["MRR"] > 0).all()

def test_mrr_never_negative_across_many_customers():
    df = pd.DataFrame({
        "customerID": [str(i) for i in range(50)],
        "tenure": [6] * 50,
        "MonthlyCharges": [10.0] * 50,
        "Churn": [0] * 50,
    })
    result = generate_mrr_series(df, seed=7)
    assert (result["MRR"] >= 0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mrr_generation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_mlops.lib.mrr_generation'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn_mlops/lib/mrr_generation.py
import numpy as np
import pandas as pd


def generate_mrr_series(churn_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for row in churn_df.itertuples(index=False):
        tenure = max(int(row.tenure), 1)
        monthly = float(row.MonthlyCharges)
        churned = int(row.Churn) == 1

        for month in range(tenure):
            seasonality = 1 + 0.03 * np.sin(2 * np.pi * month / 12)
            noise = 1 + rng.normal(0, 0.02)
            mrr = max(monthly * seasonality * noise, 0.0)
            rows.append({"customerID": row.customerID, "month": month, "MRR": mrr})

        if churned:
            rows.append({"customerID": row.customerID, "month": tenure, "MRR": 0.0})

    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mrr_generation.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/churn_mlops/lib/mrr_generation.py tests/test_mrr_generation.py
git commit -m "feat: add synthetic MRR series generation"
```

---

### Task 6: MRR validation pure function

**Files:**
- Create: `src/churn_mlops/lib/mrr_validation.py`
- Test: `tests/test_mrr_validation.py`

**Interfaces:**
- Produces: `validate_mrr(df: pd.DataFrame) -> list[str]`. Consumed by Task 7's `validated_mrr_check` asset check.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mrr_validation.py
import pandas as pd
from churn_mlops.lib.mrr_validation import validate_mrr

def test_validate_mrr_passes_on_clean_data():
    df = pd.DataFrame({"customerID": ["1", "1"], "month": [0, 1], "MRR": [10.0, 10.0]})
    assert validate_mrr(df) == []

def test_validate_mrr_catches_missing_columns():
    df = pd.DataFrame({"customerID": ["1"], "month": [0]})
    errors = validate_mrr(df)
    assert any("missing columns" in e for e in errors)

def test_validate_mrr_catches_negative_values():
    df = pd.DataFrame({"customerID": ["1"], "month": [0], "MRR": [-5.0]})
    errors = validate_mrr(df)
    assert any("negative" in e for e in errors)

def test_validate_mrr_catches_nulls():
    df = pd.DataFrame({"customerID": ["1"], "month": [0], "MRR": [None]})
    errors = validate_mrr(df)
    assert any("null" in e for e in errors)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mrr_validation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_mlops.lib.mrr_validation'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn_mlops/lib/mrr_validation.py
import pandas as pd

REQUIRED_COLUMNS = ["customerID", "month", "MRR"]


def validate_mrr(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"missing columns: {missing_cols}")
        return errors

    if df["MRR"].isna().any():
        errors.append("MRR has null values")
    if df["month"].isna().any():
        errors.append("month has null values")
    if (df["MRR"].dropna() < 0).any():
        errors.append("MRR has negative values")

    return errors
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mrr_validation.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/churn_mlops/lib/mrr_validation.py tests/test_mrr_validation.py
git commit -m "feat: add MRR series validation function"
```

---

### Task 7: synthetic_mrr_raw + validated_mrr Dagster assets

**Files:**
- Create: `src/churn_mlops/assets/mrr.py`
- Test: `tests/test_mrr_assets.py`

**Interfaces:**
- Consumes: `validated_churn` asset (Task 4), `generate_mrr_series` (Task 5), `validate_mrr` (Task 6)
- Produces: assets `synthetic_mrr_raw`, `validated_mrr`; asset check `validated_mrr_check`. Consumed by Task 9 (`feature_table`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mrr_assets.py
from dagster import materialize, RunConfig
from churn_mlops.io.parquet_io_manager import ParquetIOManager
from churn_mlops.assets.churn import raw_churn, validated_churn, validated_churn_check, RawChurnConfig
from churn_mlops.assets.mrr import synthetic_mrr_raw, validated_mrr, validated_mrr_check

def _resources(tmp_path):
    return {
        "raw_io_manager": ParquetIOManager(base_dir=str(tmp_path / "raw")),
        "validated_io_manager": ParquetIOManager(base_dir=str(tmp_path / "validated")),
    }

def test_mrr_pipeline_materializes_from_validated_churn(tmp_path):
    result = materialize(
        [raw_churn, validated_churn, validated_churn_check, synthetic_mrr_raw, validated_mrr, validated_mrr_check],
        resources=_resources(tmp_path),
        run_config=RunConfig(
            ops={"raw_churn": RawChurnConfig(source_path="tests/fixtures/tiny_churn_valid.csv")}
        ),
    )
    assert result.success
    df = result.output_for_node("validated_mrr")
    assert set(df.columns) == {"customerID", "month", "MRR"}
    assert (df["MRR"] >= 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mrr_assets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_mlops.assets.mrr'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn_mlops/assets/mrr.py
import pandas as pd
from dagster import asset, asset_check, AssetCheckResult
from churn_mlops.lib.mrr_generation import generate_mrr_series
from churn_mlops.lib.mrr_validation import validate_mrr


@asset(io_manager_key="raw_io_manager")
def synthetic_mrr_raw(validated_churn: pd.DataFrame) -> pd.DataFrame:
    return generate_mrr_series(validated_churn)


@asset(io_manager_key="validated_io_manager")
def validated_mrr(synthetic_mrr_raw: pd.DataFrame) -> pd.DataFrame:
    return synthetic_mrr_raw


@asset_check(asset=validated_mrr, blocking=True)
def validated_mrr_check(validated_mrr: pd.DataFrame) -> AssetCheckResult:
    errors = validate_mrr(validated_mrr)
    return AssetCheckResult(passed=len(errors) == 0, metadata={"errors": errors})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mrr_assets.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/churn_mlops/assets/mrr.py tests/test_mrr_assets.py
git commit -m "feat: add synthetic_mrr_raw and validated_mrr Dagster assets"
```

---

### Task 8: Feature table join pure function

**Files:**
- Create: `src/churn_mlops/lib/feature_join.py`
- Test: `tests/test_feature_join.py`

**Interfaces:**
- Produces: `build_feature_table(churn_df: pd.DataFrame, mrr_df: pd.DataFrame) -> pd.DataFrame` with columns `customerID`, `tenure`, `Contract`, `MonthlyCharges`, `TotalCharges`, `InternetService`, `PaymentMethod`, `TechSupport`, `Churn`, `current_mrr`, `trailing_3mo_avg_mrr`. Consumed by Task 9's `feature_table` asset.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_feature_join.py
import pandas as pd
from churn_mlops.lib.feature_join import build_feature_table

def _churn_df():
    return pd.DataFrame({
        "customerID": ["1", "2"],
        "tenure": [3, 2],
        "Contract": ["Month-to-month", "One year"],
        "MonthlyCharges": [50.0, 70.0],
        "TotalCharges": [150.0, 140.0],
        "InternetService": ["DSL", "Fiber optic"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "TechSupport": ["No", "Yes"],
        "Churn": [1, 0],
    })

def _mrr_df():
    return pd.DataFrame({
        "customerID": ["1", "1", "1", "2", "2"],
        "month": [0, 1, 2, 0, 1],
        "MRR": [50.0, 50.0, 0.0, 70.0, 70.0],
    })

def test_no_duplicate_customer_ids():
    result = build_feature_table(_churn_df(), _mrr_df())
    assert result["customerID"].is_unique

def test_row_count_matches_churn_input():
    result = build_feature_table(_churn_df(), _mrr_df())
    assert len(result) == 2

def test_current_mrr_uses_latest_month():
    result = build_feature_table(_churn_df(), _mrr_df())
    row = result[result["customerID"] == "1"].iloc[0]
    assert row["current_mrr"] == 0.0

def test_trailing_avg_computed():
    result = build_feature_table(_churn_df(), _mrr_df())
    row = result[result["customerID"] == "1"].iloc[0]
    assert row["trailing_3mo_avg_mrr"] == (50.0 + 50.0 + 0.0) / 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_feature_join.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_mlops.lib.feature_join'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn_mlops/lib/feature_join.py
import pandas as pd

CHURN_FEATURE_COLUMNS = [
    "customerID", "tenure", "Contract", "MonthlyCharges", "TotalCharges",
    "InternetService", "PaymentMethod", "TechSupport", "Churn",
]


def build_feature_table(churn_df: pd.DataFrame, mrr_df: pd.DataFrame) -> pd.DataFrame:
    churn_features = churn_df[CHURN_FEATURE_COLUMNS].copy()

    sorted_mrr = mrr_df.sort_values("month")

    latest_mrr = (
        sorted_mrr.groupby("customerID")
        .tail(1)[["customerID", "MRR"]]
        .rename(columns={"MRR": "current_mrr"})
    )

    trailing_avg = (
        sorted_mrr.groupby("customerID")["MRR"]
        .apply(lambda s: s.tail(3).mean())
        .reset_index(name="trailing_3mo_avg_mrr")
    )

    result = churn_features.merge(latest_mrr, on="customerID", how="left")
    result = result.merge(trailing_avg, on="customerID", how="left")
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_feature_join.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/churn_mlops/lib/feature_join.py tests/test_feature_join.py
git commit -m "feat: add feature table join function"
```

---

### Task 9: feature_table Dagster asset (full pipeline)

**Files:**
- Create: `src/churn_mlops/assets/features.py`
- Test: `tests/test_feature_table_asset.py`

**Interfaces:**
- Consumes: `validated_churn` (Task 4), `validated_mrr` (Task 7), `build_feature_table` (Task 8)
- Produces: asset `feature_table`. Terminal asset for this sub-project — consumed by churn/forecast model training in later sub-projects (out of scope here).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_feature_table_asset.py
from dagster import materialize, RunConfig
from churn_mlops.io.parquet_io_manager import ParquetIOManager
from churn_mlops.assets.churn import raw_churn, validated_churn, validated_churn_check, RawChurnConfig
from churn_mlops.assets.mrr import synthetic_mrr_raw, validated_mrr, validated_mrr_check
from churn_mlops.assets.features import feature_table

ALL_ASSETS = [
    raw_churn, validated_churn, validated_churn_check,
    synthetic_mrr_raw, validated_mrr, validated_mrr_check,
    feature_table,
]

def _resources(tmp_path):
    return {
        "raw_io_manager": ParquetIOManager(base_dir=str(tmp_path / "raw")),
        "validated_io_manager": ParquetIOManager(base_dir=str(tmp_path / "validated")),
        "feature_io_manager": ParquetIOManager(base_dir=str(tmp_path / "features")),
    }

def test_full_pipeline_materializes_feature_table(tmp_path):
    result = materialize(
        ALL_ASSETS,
        resources=_resources(tmp_path),
        run_config=RunConfig(
            ops={"raw_churn": RawChurnConfig(source_path="tests/fixtures/tiny_churn_valid.csv")}
        ),
    )
    assert result.success
    df = result.output_for_node("feature_table")
    assert len(df) == 3
    assert "current_mrr" in df.columns
    assert "trailing_3mo_avg_mrr" in df.columns
    assert df["customerID"].is_unique
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_feature_table_asset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_mlops.assets.features'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn_mlops/assets/features.py
import pandas as pd
from dagster import asset
from churn_mlops.lib.feature_join import build_feature_table


@asset(io_manager_key="feature_io_manager")
def feature_table(validated_churn: pd.DataFrame, validated_mrr: pd.DataFrame) -> pd.DataFrame:
    return build_feature_table(validated_churn, validated_mrr)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_feature_table_asset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/churn_mlops/assets/features.py tests/test_feature_table_asset.py
git commit -m "feat: add feature_table Dagster asset"
```

---

### Task 10: Definitions wiring + manual smoke check

**Files:**
- Create: `src/churn_mlops/definitions.py`
- Test: `tests/test_definitions.py`

**Interfaces:**
- Consumes: all assets from Task 4, 7, 9; `ParquetIOManager` (Task 2)
- Produces: `defs` (Dagster `Definitions`) — the entry point `dagster dev` loads. Terminal deliverable of this sub-project.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_definitions.py
from churn_mlops.definitions import defs

def test_definitions_registers_five_assets():
    asset_graph = defs.get_repository_def().asset_graph
    asset_names = {key.path[-1] for key in asset_graph.get_all_asset_keys()}
    assert asset_names == {
        "raw_churn", "validated_churn", "synthetic_mrr_raw", "validated_mrr", "feature_table",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_definitions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'churn_mlops.definitions'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/churn_mlops/definitions.py
from dagster import Definitions
from churn_mlops.assets import churn, mrr, features
from churn_mlops.io.parquet_io_manager import ParquetIOManager

defs = Definitions(
    assets=[
        churn.raw_churn,
        churn.validated_churn,
        mrr.synthetic_mrr_raw,
        mrr.validated_mrr,
        features.feature_table,
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
git commit -m "feat: wire Dagster Definitions for data layer pipeline"
```

- [ ] **Step 6: Manual smoke check (requires the real Kaggle CSV from the Prerequisite section)**

Run: `dagster dev -m churn_mlops.definitions`
Expected: Dagster UI opens at `localhost:3000`, asset graph shows all 5 assets + 2 checks; materializing `feature_table` from the UI succeeds end-to-end against the real Telco CSV placed at `data/source/telco_churn.csv`.

---

## Self-Review Notes

- **Spec coverage:** Architecture (Task 10 Definitions), all 5 components (Tasks 3-4, 5-7, 8-9), data flow (Tasks 4/7/9 chain), error handling / blocking checks (Tasks 4, 7), testing (every task is TDD + Task 9 covers the join-level pytest requirement from spec) — all covered.
- **Placeholder scan:** none found — every step has real code.
- **Type consistency:** `RawChurnConfig` (Task 4) reused identically in Tasks 4, 7, 9 tests; `ParquetIOManager` (Task 2) reused identically across Tasks 4, 7, 9, 10; column names (`customerID`, `MRR`, `current_mrr`, `trailing_3mo_avg_mrr`) consistent between Task 5/6/8 pure functions and their asset wiring in Task 7/9.
