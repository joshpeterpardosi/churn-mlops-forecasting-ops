import numpy as np
import pandas as pd

from churn_mlops.lib.drift_report import build_drift_report, extract_drift_summary


def _synthetic_df(n=200, seed=0, monthly_charges_shift=0.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "tenure": rng.integers(0, 72, size=n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
        "MonthlyCharges": rng.uniform(20, 120, size=n) + monthly_charges_shift,
        "TotalCharges": rng.uniform(0, 8000, size=n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size=n),
        "PaymentMethod": rng.choice(
            ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"], size=n
        ),
        "TechSupport": rng.choice(["No", "No internet service", "Yes"], size=n),
        "current_mrr": rng.uniform(0, 120, size=n),
        "trailing_3mo_avg_mrr": rng.uniform(0, 120, size=n),
        "Churn": rng.integers(0, 2, size=n),
    })


def test_no_drift_when_distributions_match():
    reference_df = _synthetic_df(seed=0)
    current_df = _synthetic_df(seed=0)  # identical (same seed)

    report = build_drift_report(reference_df, current_df)
    summary = extract_drift_summary(report)

    assert summary["dataset_drift_detected"] is False


def test_drift_detected_when_distribution_shifts():
    reference_df = _synthetic_df(seed=0)
    current_df = _synthetic_df(seed=1, monthly_charges_shift=500.0)

    report = build_drift_report(reference_df, current_df)
    summary = extract_drift_summary(report)

    assert summary["drift_share"] > 0
