import pandas as pd
from dagster import AssetCheckResult, Config, asset, asset_check

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
