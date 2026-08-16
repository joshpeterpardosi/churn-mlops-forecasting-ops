from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

CATEGORICAL_COLUMNS = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]

# Explicit feature allowlist: every feature_table column the model is deliberately
# trained on (everything except customerID and the Churn label). Adding a column to
# feature_table should never silently become a model feature — it must be added here.
FEATURE_COLUMNS = [
    "tenure",
    "Contract",
    "MonthlyCharges",
    "TotalCharges",
    "InternetService",
    "PaymentMethod",
    "TechSupport",
    "current_mrr",
    "trailing_3mo_avg_mrr",
]

DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.05,
    "random_state": 42,
}

# --- Operating point -------------------------------------------------------
#
# A classifier returns a probability; turning it into a decision needs a
# threshold, and 0.5 is a library default, not a business choice. For churn the
# two errors cost very different amounts:
#
#   false negative  a churner we never contacted, so we lose the account
#   false positive  a healthy customer who got an unnecessary retention call
#
# COST_FN_TO_FP prices the first relative to the second. Ten is deliberately
# conservative: a retention call is minutes of an agent's time, while a lost
# subscription is months of revenue.
COST_FN_TO_FP = 10.0

# Cost alone is not enough. Minimising it unconstrained drives the threshold
# toward zero — on the Telco data the unconstrained optimum sits near 0.05 and
# flags roughly two thirds of the healthy customer base. A retention list that
# large is not worked, it is ignored, so the saving is imaginary. This floor
# keeps the list credible: at least half of the flagged accounts must be real
# churners.
MIN_PRECISION = 0.50

# Kept only to report what the library default would have produced, so the
# chosen point can be compared against it.
DEFAULT_THRESHOLD = 0.5


def select_operating_point(
    y_true: pd.Series,
    y_proba: np.ndarray,
    cost_fn_to_fp: float = COST_FN_TO_FP,
    min_precision: float = MIN_PRECISION,
) -> float:
    """Choose the probability threshold that minimises expected cost, subject to
    a precision floor.

    Returns the chosen threshold. Falls back to the highest-precision candidate
    if no threshold clears the floor, which happens when the model has no useful
    signal at all — in that case the caller should not be shipping the model.
    """
    y_true = np.asarray(y_true)
    candidates = np.round(np.arange(0.01, 1.00, 0.01), 2)

    best_threshold = None
    best_cost = np.inf
    fallback_threshold = DEFAULT_THRESHOLD
    fallback_precision = -1.0

    for threshold in candidates:
        y_pred = y_proba >= threshold
        false_neg = int(((~y_pred) & (y_true == 1)).sum())
        false_pos = int((y_pred & (y_true == 0)).sum())
        cost = cost_fn_to_fp * false_neg + false_pos
        precision = precision_score(y_true, y_pred, zero_division=0)

        if precision > fallback_precision:
            fallback_precision = precision
            fallback_threshold = float(threshold)

        if precision >= min_precision and cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)

    return best_threshold if best_threshold is not None else fallback_threshold


def _metrics_at(y_true: pd.Series, y_proba: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = y_proba >= threshold
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def train_churn_model(
    feature_df: pd.DataFrame, params: dict[str, Any] | None = None
) -> tuple[LGBMClassifier, dict[str, float], pd.DataFrame]:
    params = dict(DEFAULT_PARAMS) if params is None else params

    df = feature_df.copy()
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")

    X = df[FEATURE_COLUMNS]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = select_operating_point(y_test, y_proba)

    chosen = _metrics_at(y_test, y_proba, threshold)
    default = _metrics_at(y_test, y_proba, DEFAULT_THRESHOLD)

    # precision/recall/f1/accuracy are reported at the chosen operating point,
    # because that is the point the service actually runs at. The *_at_default
    # pair is kept so the trade the threshold bought stays visible in MLflow.
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "threshold": float(threshold),
        **chosen,
        "precision_at_default": default["precision"],
        "recall_at_default": default["recall"],
    }
    return model, metrics, X_test
