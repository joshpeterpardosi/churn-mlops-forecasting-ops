import pandas as pd

STATIC_FEATURE_COLUMNS = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]


def build_forecast_features(mrr_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    mrr_sorted = mrr_df.sort_values(["customerID", "month"]).copy()

    grouped = mrr_sorted.groupby("customerID")["MRR"]
    mrr_sorted["lag_1"] = grouped.shift(1)
    mrr_sorted["lag_2"] = grouped.shift(2)
    mrr_sorted["lag_3"] = grouped.shift(3)
    mrr_sorted["rolling_3mo_mean"] = mrr_sorted[["lag_1", "lag_2", "lag_3"]].mean(axis=1)
    mrr_sorted["target_mrr"] = mrr_sorted["MRR"]

    result = mrr_sorted.dropna(subset=["lag_1", "lag_2", "lag_3"]).copy()

    static_features = feature_df[["customerID", *STATIC_FEATURE_COLUMNS]]
    result = result.merge(static_features, on="customerID", how="left")

    return result[
        ["customerID", "month", "lag_1", "lag_2", "lag_3", "rolling_3mo_mean", *STATIC_FEATURE_COLUMNS, "target_mrr"]
    ]
