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
    assert df.loc[df["customerID"] == 3, "TotalCharges"].tolist() == [30.0]


def test_validated_churn_check_passes_on_tenure_zero_blank_total_charges(tmp_path):
    result = materialize(
        [raw_churn, validated_churn, validated_churn_check],
        resources=_resources(tmp_path),
        run_config=RunConfig(
            ops={"raw_churn": RawChurnConfig(source_path="tests/fixtures/tiny_churn_tenure_zero.csv")}
        ),
    )
    assert result.success
    check_result = result.get_asset_check_evaluations()[0]
    assert check_result.passed

    df = result.output_for_node("validated_churn")
    row = df[df["customerID"] == 1].iloc[0]
    assert row["tenure"] == 0
    assert row["TotalCharges"] == 0.0


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
