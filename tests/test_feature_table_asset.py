from dagster import RunConfig, materialize

from churn_mlops.assets.churn import RawChurnConfig, raw_churn, validated_churn, validated_churn_check
from churn_mlops.assets.features import feature_table
from churn_mlops.assets.mrr import synthetic_mrr_raw, validated_mrr, validated_mrr_check
from churn_mlops.io.parquet_io_manager import ParquetIOManager

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
