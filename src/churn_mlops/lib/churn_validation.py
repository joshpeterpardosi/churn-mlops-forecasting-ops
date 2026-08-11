import pandas as pd

REQUIRED_COLUMNS = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]

# Columns feature_table needs that only require a presence check here (no
# null/range validation) — see spec Components §2.
PRESENCE_ONLY_COLUMNS = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
    df["TotalCharges"] = df["TotalCharges"].replace("", None)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Real Telco data has blank TotalCharges on every tenure == 0 row
    # (brand-new customers with no billing history yet). Impute those to
    # 0.0; every other blank stays null and is rejected by validate_churn.
    zero_tenure_blank = (df["tenure"] == 0) & df["TotalCharges"].isna()
    df.loc[zero_tenure_blank, "TotalCharges"] = 0.0
    return df


def map_churn_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def validate_churn(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing_cols = [c for c in REQUIRED_COLUMNS + PRESENCE_ONLY_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"missing columns: {missing_cols}")
        return errors

    for col in ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]:
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            errors.append(f"{col} has {n_null} null values")

    if (df["tenure"] < 0).any():
        errors.append("tenure has negative values")
    if (df["MonthlyCharges"] < 0).any():
        errors.append("MonthlyCharges has negative values")
    if (df["TotalCharges"] < 0).any():
        errors.append("TotalCharges has negative values")
    if not df["Churn"].isin([0, 1]).all():
        errors.append("Churn has values outside {0, 1}")

    return errors
