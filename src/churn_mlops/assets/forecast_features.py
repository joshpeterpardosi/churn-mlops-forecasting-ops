import pandas as pd
from dagster import asset

from churn_mlops.lib.forecast_features import build_forecast_features


@asset(io_manager_key="feature_io_manager")
def forecast_features(validated_mrr: pd.DataFrame, feature_table: pd.DataFrame) -> pd.DataFrame:
    return build_forecast_features(validated_mrr, feature_table)
