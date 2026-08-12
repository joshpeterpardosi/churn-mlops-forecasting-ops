import pandas as pd
from dagster import build_input_context, build_output_context

from churn_mlops.io.parquet_io_manager import ParquetIOManager


def test_round_trip(tmp_path):
    manager = ParquetIOManager(base_dir=str(tmp_path))
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    # name ("result") deliberately differs from asset_key ("my_asset") to
    # mirror real asset materialization, where the Dagster output name
    # defaults to "result" for every asset while asset_key carries the
    # asset's real identity.
    out_context = build_output_context(name="result", asset_key="my_asset")
    manager.handle_output(out_context, df)

    in_context = build_input_context(upstream_output=out_context)
    result = manager.load_input(in_context)

    pd.testing.assert_frame_equal(result, df)


def test_sibling_assets_sharing_output_name_do_not_collide(tmp_path):
    # Regression test for the bug this IOManager originally had: every
    # single-output @asset shares the same Dagster output name ("result"),
    # so keying the parquet path on the output name (instead of the asset's
    # own asset_key) makes any two assets that share an IOManager instance
    # overwrite each other's file. This is exactly what happened with
    # validated_churn and validated_mrr sharing validated_io_manager.
    manager = ParquetIOManager(base_dir=str(tmp_path))
    df_a = pd.DataFrame({"a": [1, 2]})
    df_b = pd.DataFrame({"b": [3, 4, 5]})

    out_a = build_output_context(name="result", asset_key="asset_a")
    out_b = build_output_context(name="result", asset_key="asset_b")

    manager.handle_output(out_a, df_a)
    manager.handle_output(out_b, df_b)

    in_a = build_input_context(upstream_output=out_a)
    in_b = build_input_context(upstream_output=out_b)

    pd.testing.assert_frame_equal(manager.load_input(in_a), df_a)
    pd.testing.assert_frame_equal(manager.load_input(in_b), df_b)
