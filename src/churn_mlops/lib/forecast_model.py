from typing import Any

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

CATEGORICAL_COLUMNS = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]
FEATURE_COLUMNS = ["lag_1", "lag_2", "lag_3", "rolling_3mo_mean", *CATEGORICAL_COLUMNS]

DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.05,
    "random_state": 42,
}


def train_forecast_model(
    forecast_df: pd.DataFrame, params: dict[str, Any] | None = None
) -> tuple[LGBMRegressor, dict[str, float], pd.DataFrame]:
    params = dict(DEFAULT_PARAMS) if params is None else params

    df = forecast_df.sort_values(["customerID", "month"]).copy()
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")

    group_sizes = df.groupby("customerID")["customerID"].transform("size")
    is_last_row = df.groupby("customerID").cumcount(ascending=False) == 0
    is_test = is_last_row & (group_sizes > 1)

    train_df = df[~is_test]
    test_df = df[is_test]

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["target_mrr"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["target_mrr"]

    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    baseline_rmse = float(mean_squared_error(y_test, X_test["lag_1"]) ** 0.5)
    metrics = {
        "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "baseline_rmse": baseline_rmse,
    }
    return model, metrics, X_test
