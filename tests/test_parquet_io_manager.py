import pandas as pd
from dagster import build_output_context, build_input_context
from churn_mlops.io.parquet_io_manager import ParquetIOManager

def test_round_trip(tmp_path):
    manager = ParquetIOManager(base_dir=str(tmp_path))
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    out_context = build_output_context(name="my_asset", asset_key="my_asset")
    manager.handle_output(out_context, df)

    in_context = build_input_context(upstream_output=out_context)
    result = manager.load_input(in_context)

    pd.testing.assert_frame_equal(result, df)
