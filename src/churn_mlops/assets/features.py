import pandas as pd
from dagster import asset

from churn_mlops.lib.feature_join import build_feature_table


@asset(io_manager_key="feature_io_manager")
def feature_table(validated_churn: pd.DataFrame, validated_mrr: pd.DataFrame) -> pd.DataFrame:
    return build_feature_table(validated_churn, validated_mrr)
