import pandas as pd
from churn_mlops.lib.feature_join import build_feature_table

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
    return pd.DataFrame({
        "customerID": ["1", "1", "1", "2", "2"],
        "month": [0, 1, 2, 0, 1],
        "MRR": [50.0, 50.0, 0.0, 70.0, 70.0],
    })

def test_no_duplicate_customer_ids():
    result = build_feature_table(_churn_df(), _mrr_df())
    assert result["customerID"].is_unique

def test_row_count_matches_churn_input():
    result = build_feature_table(_churn_df(), _mrr_df())
    assert len(result) == 2

def test_current_mrr_uses_latest_month():
    result = build_feature_table(_churn_df(), _mrr_df())
    row = result[result["customerID"] == "1"].iloc[0]
    assert row["current_mrr"] == 0.0

def test_trailing_avg_computed():
    result = build_feature_table(_churn_df(), _mrr_df())
    row = result[result["customerID"] == "1"].iloc[0]
    assert row["trailing_3mo_avg_mrr"] == (50.0 + 50.0 + 0.0) / 3
