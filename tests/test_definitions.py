from churn_mlops.definitions import defs


def test_definitions_registers_eight_assets():
    asset_graph = defs.get_repository_def().asset_graph
    asset_names = {key.path[-1] for key in asset_graph.get_all_asset_keys()}
    assert asset_names == {
        "raw_churn", "validated_churn", "synthetic_mrr_raw", "validated_mrr",
        "feature_table", "churn_model", "forecast_features", "forecast_model",
    }
