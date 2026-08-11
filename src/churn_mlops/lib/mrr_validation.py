import pandas as pd

REQUIRED_COLUMNS = ["customerID", "month", "MRR"]


def validate_mrr(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"missing columns: {missing_cols}")
        return errors

    if df["MRR"].isna().any():
        errors.append("MRR has null values")
    if df["month"].isna().any():
        errors.append("month has null values")
    if (df["MRR"].dropna() < 0).any():
        errors.append("MRR has negative values")

    return errors
