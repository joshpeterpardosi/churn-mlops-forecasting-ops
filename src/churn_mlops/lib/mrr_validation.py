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

    duplicate_customers = []
    gap_customers = []
    for customer_id, group in df.groupby("customerID"):
        months = sorted(int(m) for m in group["month"].dropna())
        if len(months) != len(set(months)):
            duplicate_customers.append(customer_id)
            continue
        if months != list(range(len(months))):
            gap_customers.append(customer_id)

    if duplicate_customers:
        errors.append(f"duplicate month values for customers: {duplicate_customers}")
    if gap_customers:
        errors.append(f"non-contiguous month coverage for customers: {gap_customers}")

    return errors
