import pandas as pd

from churn_mlops.lib.mrr_validation import validate_mrr


def test_validate_mrr_passes_on_clean_data():
    df = pd.DataFrame({"customerID": ["1", "1"], "month": [0, 1], "MRR": [10.0, 10.0]})
    assert validate_mrr(df) == []

def test_validate_mrr_catches_missing_columns():
    df = pd.DataFrame({"customerID": ["1"], "month": [0]})
    errors = validate_mrr(df)
    assert any("missing columns" in e for e in errors)

def test_validate_mrr_catches_negative_values():
    df = pd.DataFrame({"customerID": ["1"], "month": [0], "MRR": [-5.0]})
    errors = validate_mrr(df)
    assert any("negative" in e for e in errors)

def test_validate_mrr_catches_nulls():
    df = pd.DataFrame({"customerID": ["1"], "month": [0], "MRR": [None]})
    errors = validate_mrr(df)
    assert any("null" in e for e in errors)

def test_validate_mrr_catches_month_gap():
    df = pd.DataFrame({"customerID": ["1", "1"], "month": [0, 2], "MRR": [10.0, 10.0]})
    errors = validate_mrr(df)
    assert any("coverage" in e for e in errors)

def test_validate_mrr_catches_duplicate_month():
    df = pd.DataFrame({"customerID": ["1", "1"], "month": [0, 0], "MRR": [10.0, 10.0]})
    errors = validate_mrr(df)
    assert any("duplicate" in e for e in errors)

def test_validate_mrr_passes_contiguous_months_per_customer():
    df = pd.DataFrame({
        "customerID": ["1", "1", "1", "2", "2"],
        "month": [0, 1, 2, 0, 1],
        "MRR": [10.0, 10.0, 10.0, 5.0, 5.0],
    })
    assert validate_mrr(df) == []
