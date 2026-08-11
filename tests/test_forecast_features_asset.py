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
