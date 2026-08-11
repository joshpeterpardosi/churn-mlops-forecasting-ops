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
    # excludes it.
    #
    # The terminal row is identified structurally — "the last row, by month,
    # within a churned customer's group" — rather than by recomputing
    # `month == tenure`. generate_mrr_series floors tenure to
    # max(tenure, 1) when placing the terminal row (mrr_generation.py), so
    # for a tenure == 0 churned customer the real observation lands at
    # month 0 and the terminal zero lands at month 1 (floored) — a raw
    # tenure comparison drops the wrong row and reintroduces the leak for
    # that subset. "Last row in the group" holds regardless of any
    # tenure-flooring quirks upstream, since the generator always appends
    # the churn-zero row after every pre-churn month it emits.
    churn_lookup = churn_df[["customerID", "Churn"]]
    mrr_with_churn = mrr_df.merge(churn_lookup, on="customerID", how="left")
    sorted_by_group = mrr_with_churn.sort_values(["customerID", "month"]).reset_index(drop=True)
    is_last_in_group = sorted_by_group.groupby("customerID").cumcount(ascending=False) == 0
    terminal_churn_row = (sorted_by_group["Churn"] == 1) & is_last_in_group
    pre_churn_mrr = sorted_by_group.loc[~terminal_churn_row, ["customerID", "month", "MRR"]]

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
