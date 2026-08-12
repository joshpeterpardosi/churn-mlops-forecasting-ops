import pandas as pd

from churn_mlops.lib.mrr_generation import generate_mrr_series


def test_churned_customer_drops_to_zero_at_tenure_month():
    df = pd.DataFrame({
        "customerID": ["1"], "tenure": [3], "MonthlyCharges": [50.0], "Churn": [1],
    })
    result = generate_mrr_series(df)
    last_row = result[result["customerID"] == "1"].sort_values("month").iloc[-1]
    assert last_row["month"] == 3
    assert last_row["MRR"] == 0.0


def test_active_customer_never_drops_to_zero():
    df = pd.DataFrame({
        "customerID": ["2"], "tenure": [4], "MonthlyCharges": [70.0], "Churn": [0],
    })
    result = generate_mrr_series(df)
    cust = result[result["customerID"] == "2"]
    assert cust["month"].max() == 3
    assert (cust["MRR"] > 0).all()


def test_mrr_never_negative_across_many_customers():
    df = pd.DataFrame({
        "customerID": [str(i) for i in range(50)],
        "tenure": [6] * 50,
        "MonthlyCharges": [10.0] * 50,
        "Churn": [0] * 50,
    })
    result = generate_mrr_series(df, seed=7)
    assert (result["MRR"] >= 0).all()
