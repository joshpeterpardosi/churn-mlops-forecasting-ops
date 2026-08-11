import os
import pandas as pd
from dagster import IOManager, InputContext, OutputContext


class ParquetIOManager(IOManager):
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def _path_for_name(self, name: str) -> str:
        return os.path.join(self.base_dir, f"{name}.parquet")

    def handle_output(self, context: OutputContext, obj: pd.DataFrame):
        path = self._path_for_name(context.asset_key.path[-1])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        obj.to_parquet(path, index=False)

    def load_input(self, context: InputContext) -> pd.DataFrame:
        path = self._path_for_name(context.upstream_output.asset_key.path[-1])
        return pd.read_parquet(path)
