import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from churn_mlops.lib.forecast_model import train_forecast_model


def _synthetic_forecast_df(n_customers=30, months_per_customer=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_customers):
        base = rng.uniform(20, 100)
        for m in range(months_per_customer):
            rows.append({
                "customerID": f"c{c}",
                "month": m + 3,
                "lag_1": base + rng.normal(0, 2),
                "lag_2": base + rng.normal(0, 2),
                "lag_3": base + rng.normal(0, 2),
                "rolling_3mo_mean": base,
                "Contract": rng.choice(["Month-to-month", "One year", "Two year"]),
                "InternetService": rng.choice(["DSL", "Fiber optic", "No"]),
                "PaymentMethod": rng.choice(["Electronic check", "Mailed check"]),
                "TechSupport": rng.choice(["Yes", "No"]),
                "target_mrr": base + rng.normal(0, 2),
            })
    return pd.DataFrame(rows)


def test_train_forecast_model_returns_expected_metrics():
    df = _synthetic_forecast_df()
    model, metrics, _X_test = train_forecast_model(df)
    assert isinstance(model, LGBMRegressor)
    assert set(metrics.keys()) == {"rmse", "mae", "baseline_rmse"}
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0
    assert metrics["baseline_rmse"] >= 0


def test_time_split_respects_order_per_customer():
    df = _synthetic_forecast_df()
    _model, _metrics, X_test = train_forecast_model(df)
    sorted_df = df.sort_values(["customerID", "month"])
    expected_last_indices = sorted_df.groupby("customerID").tail(1).index
    assert set(X_test.index) == set(expected_last_indices)


def test_single_row_customer_contributes_no_test_row():
    multi_df = _synthetic_forecast_df(n_customers=5, months_per_customer=8)
    single_row = pd.DataFrame([{
        "customerID": "solo",
        "month": 3,
        "lag_1": 40.0, "lag_2": 40.0, "lag_3": 40.0, "rolling_3mo_mean": 40.0,
        "Contract": "One year", "InternetService": "DSL",
        "PaymentMethod": "Mailed check", "TechSupport": "No",
        "target_mrr": 40.0,
    }])
    df = pd.concat([multi_df, single_row], ignore_index=True)
    _model, _metrics, X_test = train_forecast_model(df)
    solo_index = df[df["customerID"] == "solo"].index[0]
    assert solo_index not in X_test.index
