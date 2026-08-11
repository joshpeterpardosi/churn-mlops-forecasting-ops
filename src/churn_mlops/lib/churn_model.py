from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

CATEGORICAL_COLUMNS = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]

DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.05,
    "random_state": 42,
}


def train_churn_model(
    feature_df: pd.DataFrame, params: dict[str, Any] | None = None
) -> tuple[LGBMClassifier, dict[str, float]]:
    params = dict(DEFAULT_PARAMS) if params is None else params

    df = feature_df.drop(columns=["customerID"]).copy()
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    return model, metrics
