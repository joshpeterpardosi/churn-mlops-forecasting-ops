import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score

from churn_mlops.lib.churn_model import (
    FEATURE_COLUMNS,
    MIN_PRECISION,
    select_operating_point,
    train_churn_model,
)


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
    model, metrics, X_test = train_churn_model(df)
    assert isinstance(model, LGBMClassifier)
    assert set(metrics.keys()) == {
        "roc_auc",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "precision_at_default",
        "recall_at_default",
    }
    for value in metrics.values():
        assert 0.0 <= value <= 1.0
    assert list(X_test.columns) == FEATURE_COLUMNS


def test_train_churn_model_uses_default_params_when_none_given():
    df = _synthetic_feature_table()
    model, _, _ = train_churn_model(df)
    params = model.get_params()
    assert params["n_estimators"] == 100
    assert params["max_depth"] == 5
    assert params["learning_rate"] == 0.05


def test_train_churn_model_ignores_unexpected_extra_columns():
    df = _synthetic_feature_table()
    df["some_future_column"] = 1
    model, _, X_test = train_churn_model(df)
    assert "some_future_column" not in model.feature_name_
    assert "some_future_column" not in X_test.columns


def test_operating_point_respects_the_precision_floor():
    # A model with real signal: probability tracks the label closely, with
    # enough overlap that the threshold actually matters.
    rng = np.random.default_rng(1)
    y_true = pd.Series(rng.integers(0, 2, size=400))
    y_proba = np.clip(y_true * 0.45 + rng.normal(0.25, 0.18, size=400), 0.0, 1.0)

    threshold = select_operating_point(y_true, y_proba)
    precision = precision_score(y_true, y_proba >= threshold, zero_division=0)
    assert precision >= MIN_PRECISION


def test_operating_point_beats_the_default_on_recall_when_misses_are_costly():
    rng = np.random.default_rng(2)
    y_true = pd.Series(rng.integers(0, 2, size=400))
    y_proba = np.clip(y_true * 0.45 + rng.normal(0.25, 0.18, size=400), 0.0, 1.0)

    threshold = select_operating_point(y_true, y_proba)
    chosen = recall_score(y_true, y_proba >= threshold, zero_division=0)
    default = recall_score(y_true, y_proba >= 0.5, zero_division=0)
    # Pricing a missed churner at 10x a false alarm can only push the threshold
    # down, never up, so recall must not regress against the library default.
    assert threshold <= 0.5
    assert chosen >= default


def test_operating_point_falls_back_when_no_threshold_clears_the_floor():
    # Pure noise: precision hovers around the base rate and never reaches 0.5,
    # so the selector must still return a usable threshold rather than None.
    rng = np.random.default_rng(3)
    y_true = pd.Series(rng.integers(0, 2, size=200))
    y_proba = rng.uniform(0, 1, size=200)

    threshold = select_operating_point(y_true, y_proba, min_precision=0.99)
    assert 0.0 < threshold < 1.0
