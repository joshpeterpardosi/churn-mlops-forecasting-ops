import numpy as np
import pandas as pd


def generate_mrr_series(churn_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for row in churn_df.itertuples(index=False):
        tenure = max(int(row.tenure), 1)
        monthly = float(row.MonthlyCharges)
        churned = int(row.Churn) == 1

        for month in range(tenure):
            seasonality = 1 + 0.03 * np.sin(2 * np.pi * month / 12)
            noise = 1 + rng.normal(0, 0.02)
            mrr = max(monthly * seasonality * noise, 0.0)
            rows.append({"customerID": row.customerID, "month": month, "MRR": mrr})

        if churned:
            rows.append({"customerID": row.customerID, "month": tenure, "MRR": 0.0})

    return pd.DataFrame(rows)
