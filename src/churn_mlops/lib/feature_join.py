import pandas as pd

CHURN_FEATURE_COLUMNS = [
    "customerID", "tenure", "Contract", "MonthlyCharges", "TotalCharges",
    "InternetService", "PaymentMethod", "TechSupport", "Churn",
]


def build_feature_table(churn_df: pd.DataFrame, mrr_df: pd.DataFrame) -> pd.DataFrame:
    churn_features = churn_df[CHURN_FEATURE_COLUMNS].copy()

    # Exclude each churned customer's terminal churn-month zero row before
    # aggregating. Including it would make current_mrr == 0 a perfect proxy
    # for Churn == 1 (label leakage) — see spec Components §5. The zero row
    # itself is left untouched in validated_mrr; only this aggregation
    # excludes it, by using only months strictly before `tenure` for
    # customers who churned.
    tenure_lookup = churn_df[["customerID", "tenure", "Churn"]]
    mrr_with_tenure = mrr_df.merge(tenure_lookup, on="customerID", how="left")
    terminal_churn_row = (mrr_with_tenure["Churn"] == 1) & (
        mrr_with_tenure["month"] == mrr_with_tenure["tenure"]
    )
    pre_churn_mrr = mrr_with_tenure.loc[~terminal_churn_row, ["customerID", "month", "MRR"]]

    sorted_mrr = pre_churn_mrr.sort_values("month")

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
