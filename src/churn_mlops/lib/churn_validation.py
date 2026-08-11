import pandas as pd

REQUIRED_COLUMNS = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
    df["TotalCharges"] = df["TotalCharges"].replace("", None)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def map_churn_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def validate_churn(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
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
