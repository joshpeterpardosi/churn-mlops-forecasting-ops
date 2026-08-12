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
    n = 150
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
        "churn_prediction": rng.integers(0, 2, size=n).astype(bool),
        "logged_at": ["2026-08-12T00:00:00+00:00"] * n,
    })
    predictions_df.to_parquet(predictions_path)

    result = materialize([_stub_feature_table, drift_report])

    assert result.success
    assert html_path.exists()
    mats = result.asset_materializations_for_node("drift_report")
    assert "drift_share" in mats[0].metadata
