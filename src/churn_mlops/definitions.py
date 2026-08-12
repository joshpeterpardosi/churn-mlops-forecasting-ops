from dagster import Definitions

from churn_mlops.assets import (
    churn,
    churn_model as churn_model_assets,
    drift_report as drift_report_assets,
    features,
    forecast_features as forecast_features_assets,
    forecast_model as forecast_model_assets,
    mrr,
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
