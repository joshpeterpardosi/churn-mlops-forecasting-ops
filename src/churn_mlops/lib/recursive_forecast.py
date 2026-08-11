from typing import Callable

import pandas as pd


def recursive_forecast(
    seed_df: pd.DataFrame, horizon: int, predict_fn: Callable[[pd.DataFrame], object]
) -> float:
    df = seed_df.copy().reset_index(drop=True)

    for _ in range(horizon):
        predictions = list(predict_fn(df))
        df["lag_3"] = df["lag_2"]
        df["lag_2"] = df["lag_1"]
        df["lag_1"] = predictions
        df["rolling_3mo_mean"] = df[["lag_1", "lag_2", "lag_3"]].mean(axis=1)

    return float(df["lag_1"].sum())
