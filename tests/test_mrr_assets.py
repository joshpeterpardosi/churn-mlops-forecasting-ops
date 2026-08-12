from dagster import RunConfig, materialize

from churn_mlops.assets.churn import RawChurnConfig, raw_churn, validated_churn, validated_churn_check
from churn_mlops.assets.mrr import synthetic_mrr_raw, validated_mrr, validated_mrr_check
from churn_mlops.io.parquet_io_manager import ParquetIOManager


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
