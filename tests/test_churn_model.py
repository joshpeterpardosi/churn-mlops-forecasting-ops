import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from churn_mlops.lib.churn_model import train_churn_model


def _synthetic_feature_table(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "customerID": [f"c{i}" for i in range(n)],
        "tenure": rng.integers(0, 72, size=n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
        "MonthlyCharges": rng.uniform(20, 120, size=n),
        "TotalCharges": rng.uniform(0, 8000, size=n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size=n),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n
        ),
        "TechSupport": rng.choice(["Yes", "No"], size=n),
        "Churn": rng.integers(0, 2, size=n),
        "current_mrr": rng.uniform(0, 120, size=n),
        "trailing_3mo_avg_mrr": rng.uniform(0, 120, size=n),
    })


def test_train_churn_model_returns_expected_metrics():
    df = _synthetic_feature_table()
    model, metrics = train_churn_model(df)
    assert isinstance(model, LGBMClassifier)
    assert set(metrics.keys()) == {"roc_auc", "accuracy", "precision", "recall", "f1"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_train_churn_model_uses_default_params_when_none_given():
    df = _synthetic_feature_table()
    model, _ = train_churn_model(df)
    params = model.get_params()
    assert params["n_estimators"] == 100
    assert params["max_depth"] == 5
    assert params["learning_rate"] == 0.05
