import pandas as pd

from churn_mlops.lib.churn_validation import (
    clean_total_charges,
    map_churn_label,
    validate_churn,
)


def _valid_df():
    return pd.DataFrame({
        "customerID": ["1", "2"],
        "tenure": [5, 10],
        "MonthlyCharges": [50.0, 70.0],
        "TotalCharges": ["250.0", "700.0"],
        "Contract": ["Month-to-month", "One year"],
        "InternetService": ["DSL", "Fiber optic"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "TechSupport": ["No", "Yes"],
        "Churn": ["Yes", "No"],
    })

def test_clean_total_charges_handles_blank_strings():
    df = _valid_df()
    df.loc[0, "TotalCharges"] = "  "
    result = clean_total_charges(df)
    assert pd.isna(result.loc[0, "TotalCharges"])
    assert result.loc[1, "TotalCharges"] == 700.0

def test_map_churn_label_converts_yes_no_to_int():
    df = _valid_df()
    result = map_churn_label(df)
    assert result["Churn"].tolist() == [1, 0]

def test_validate_churn_passes_on_clean_data():
    df = map_churn_label(clean_total_charges(_valid_df()))
    errors = validate_churn(df)
    assert errors == []

def test_validate_churn_catches_missing_columns():
    df = _valid_df().drop(columns=["tenure"])
    errors = validate_churn(df)
    assert any("missing columns" in e for e in errors)

def test_validate_churn_catches_negative_tenure():
    df = map_churn_label(clean_total_charges(_valid_df()))
    df.loc[0, "tenure"] = -1
    errors = validate_churn(df)
    assert any("tenure" in e for e in errors)

def test_validate_churn_catches_nulls():
    df = map_churn_label(clean_total_charges(_valid_df()))
    df.loc[0, "TotalCharges"] = None
    errors = validate_churn(df)
    assert any("TotalCharges" in e for e in errors)

def test_validate_churn_catches_missing_feature_join_columns():
    df = _valid_df().drop(columns=["Contract"])
    errors = validate_churn(df)
    assert any("missing columns" in e and "Contract" in e for e in errors)

def test_clean_total_charges_imputes_zero_for_tenure_zero_blank():
    df = _valid_df()
    df.loc[0, "tenure"] = 0
    df.loc[0, "TotalCharges"] = "  "
    result = clean_total_charges(df)
    assert result.loc[0, "TotalCharges"] == 0.0

def test_tenure_zero_blank_total_charges_passes_validate_churn():
    df = _valid_df()
    df.loc[0, "tenure"] = 0
    df.loc[0, "TotalCharges"] = "  "
    df = map_churn_label(clean_total_charges(df))
    errors = validate_churn(df)
    assert errors == []

def test_blank_total_charges_on_positive_tenure_row_is_still_rejected():
    # Regression guard: the tenure==0 imputation must not overcorrect and
    # mask genuinely bad data on rows where tenure > 0.
    df = _valid_df()
    assert (df["tenure"] > 0).all()
    df.loc[0, "TotalCharges"] = "  "
    df = map_churn_label(clean_total_charges(df))
    errors = validate_churn(df)
    assert any("TotalCharges" in e for e in errors)
