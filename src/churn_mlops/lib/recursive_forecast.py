from collections.abc import Callable

import pandas as pd


def recursive_forecast(
    seed_df: pd.DataFrame, horizon: int, predict_fn: Callable[[pd.DataFrame], object]
) -> float:
    """Roll a per-customer MRR forecast forward `horizon` months and sum the final step.

    `seed_df` must carry `lag_1`/`lag_2`/`lag_3`, where `lag_1` is each row's most
    recent month's MRR. On each step, `predict_fn` is called once with the current
    state of the (in-place mutated) dataframe to predict next month's MRR; the lags
    then shift forward (`lag_1` -> `lag_2` -> `lag_3`, dropping the oldest) and the
    new prediction becomes the new `lag_1`, before `rolling_3mo_mean` is recomputed
    from the shifted lags.

    The return value is the SUM across all rows of only the FINAL step's prediction
    (i.e. the portfolio-wide total MRR at month `horizon`), not a cumulative sum of
    predictions across all intermediate steps.
    """
    df = seed_df.copy().reset_index(drop=True)

    for _ in range(horizon):
        predictions = list(predict_fn(df))
        df["lag_3"] = df["lag_2"]
        df["lag_2"] = df["lag_1"]
        df["lag_1"] = predictions
        df["rolling_3mo_mean"] = df[["lag_1", "lag_2", "lag_3"]].mean(axis=1)

    return float(df["lag_1"].sum())
