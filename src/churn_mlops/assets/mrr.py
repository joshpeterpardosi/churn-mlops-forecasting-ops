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
