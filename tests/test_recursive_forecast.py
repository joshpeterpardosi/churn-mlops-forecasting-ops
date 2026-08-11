import pandas as pd

from churn_mlops.lib.recursive_forecast import recursive_forecast


def _seed_df():
    return pd.DataFrame({
        "customerID": ["a", "b"],
        "lag_1": [50.0, 30.0],
        "lag_2": [48.0, 29.0],
        "lag_3": [47.0, 28.0],
        "rolling_3mo_mean": [48.33, 29.0],
        "Contract": ["Month-to-month", "One year"],
        "InternetService": ["DSL", "Fiber optic"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "TechSupport": ["No", "Yes"],
    })


def test_horizon_one_calls_predict_once_and_sums():
    df = _seed_df()
    calls = []

    def fake_predict(features_df):
        calls.append(features_df.copy())
        return [100.0, 200.0]

    total = recursive_forecast(df, horizon=1, predict_fn=fake_predict)

    assert len(calls) == 1
    assert total == 300.0


def test_horizon_shifts_lags_between_steps():
    df = _seed_df()
    predictions_by_call = [[60.0, 31.0], [70.0, 32.0]]
    calls = []

    def fake_predict(features_df):
        calls.append(features_df.copy())
        return predictions_by_call[len(calls) - 1]

    total = recursive_forecast(df, horizon=2, predict_fn=fake_predict)

    second_call_df = calls[1]
    assert list(second_call_df["lag_1"]) == [60.0, 31.0]
    assert total == 70.0 + 32.0


def test_horizon_zero_returns_seed_lag1_sum():
    df = _seed_df()
    total = recursive_forecast(df, horizon=0, predict_fn=lambda d: [999.0, 999.0])
    assert total == 50.0 + 30.0
