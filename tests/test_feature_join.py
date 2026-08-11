import pandas as pd
from churn_mlops.lib.feature_join import build_feature_table
from churn_mlops.lib.mrr_generation import generate_mrr_series

def _churn_df():
    return pd.DataFrame({
        "customerID": ["1", "2"],
        "tenure": [3, 2],
        "Contract": ["Month-to-month", "One year"],
        "MonthlyCharges": [50.0, 70.0],
        "TotalCharges": [150.0, 140.0],
        "InternetService": ["DSL", "Fiber optic"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "TechSupport": ["No", "Yes"],
        "Churn": [1, 0],
    })

def _mrr_df():
    # Mirrors real generate_mrr_series output: customer "1" churns at
    # tenure=3, so months 0-2 are pre-churn observations and month 3 is the
    # terminal zero row. Customer "2" is active (Churn=0, tenure=2), so it
    # only has months 0-1 and no terminal row.
    return pd.DataFrame({
        "customerID": ["1", "1", "1", "1", "2", "2"],
        "month": [0, 1, 2, 3, 0, 1],
        "MRR": [50.0, 50.0, 45.0, 0.0, 70.0, 70.0],
    })

def test_no_duplicate_customer_ids():
    result = build_feature_table(_churn_df(), _mrr_df())
    assert result["customerID"].is_unique

def test_row_count_matches_churn_input():
    result = build_feature_table(_churn_df(), _mrr_df())
    assert len(result) == 2

def test_current_mrr_excludes_terminal_churn_zero_row():
    result = build_feature_table(_churn_df(), _mrr_df())
    row = result[result["customerID"] == "1"].iloc[0]
    assert row["current_mrr"] == 45.0

def test_trailing_avg_computed():
    result = build_feature_table(_churn_df(), _mrr_df())
    row = result[result["customerID"] == "1"].iloc[0]
    assert row["trailing_3mo_avg_mrr"] == (50.0 + 50.0 + 45.0) / 3

def test_current_mrr_is_not_a_perfect_proxy_for_churn():
    # The bug this guards against: current_mrr == 0 iff Churn == 1. Here
    # customer "1" churned but had a nonzero pre-churn MRR — current_mrr
    # must reflect that value, not the terminal zero row.
    result = build_feature_table(_churn_df(), _mrr_df())
    churned_row = result[result["customerID"] == "1"].iloc[0]
    assert churned_row["Churn"] == 1
    assert churned_row["current_mrr"] != 0.0
    assert churned_row["current_mrr"] == 45.0

def test_active_customer_current_mrr_unaffected():
    result = build_feature_table(_churn_df(), _mrr_df())
    active_row = result[result["customerID"] == "2"].iloc[0]
    assert active_row["Churn"] == 0
    assert active_row["current_mrr"] == 70.0

def test_zero_tenure_churned_customer_current_mrr_is_not_leaked():
    # Regression: generate_mrr_series floors tenure to max(tenure, 1) when
    # placing the terminal churn-zero row (mrr_generation.py:9). For a
    # churned customer with raw tenure == 0, the real MRR observation lands
    # at month 0 and the terminal zero row lands at month 1 (floored) — not
    # month 0. A tenure-based exclusion (month == tenure) would drop the
    # real observation and keep the terminal zero, reintroducing the leak
    # for this subset. The structural "last row in the group" exclusion
    # must get this right without any tenure comparison.
    churn_df = pd.DataFrame({
        "customerID": ["9"],
        "tenure": [0],
        "Contract": ["Month-to-month"],
        "MonthlyCharges": [80.0],
        "TotalCharges": [0.0],
        "InternetService": ["DSL"],
        "PaymentMethod": ["Electronic check"],
        "TechSupport": ["No"],
        "Churn": [1],
    })
    mrr_df = generate_mrr_series(churn_df, seed=1)

    rows = mrr_df.sort_values("month").reset_index(drop=True)
    assert rows["month"].tolist() == [0, 1]
    assert rows.loc[0, "MRR"] > 0.0
    assert rows.loc[1, "MRR"] == 0.0

    result = build_feature_table(churn_df, mrr_df)
    row = result.iloc[0]
    assert row["current_mrr"] == rows.loc[0, "MRR"]
    assert row["current_mrr"] != 0.0
