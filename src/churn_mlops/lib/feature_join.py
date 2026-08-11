import pandas as pd

CHURN_FEATURE_COLUMNS = [
    "customerID", "tenure", "Contract", "MonthlyCharges", "TotalCharges",
    "InternetService", "PaymentMethod", "TechSupport", "Churn",
]


def build_feature_table(churn_df: pd.DataFrame, mrr_df: pd.DataFrame) -> pd.DataFrame:
    churn_features = churn_df[CHURN_FEATURE_COLUMNS].copy()

    sorted_mrr = mrr_df.sort_values("month")

    latest_mrr = (
        sorted_mrr.groupby("customerID")
        .tail(1)[["customerID", "MRR"]]
        .rename(columns={"MRR": "current_mrr"})
    )

    trailing_avg = (
        sorted_mrr.groupby("customerID")["MRR"]
        .apply(lambda s: s.tail(3).mean())
        .reset_index(name="trailing_3mo_avg_mrr")
    )

    result = churn_features.merge(latest_mrr, on="customerID", how="left")
    result = result.merge(trailing_avg, on="customerID", how="left")
    return result
