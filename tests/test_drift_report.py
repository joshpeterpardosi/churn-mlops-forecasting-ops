import numpy as np
import pandas as pd

from churn_mlops.lib.drift_report import build_drift_report, extract_drift_summary


def _synthetic_df(n=1500, seed=0, shift=0.0):
    # n=1500 (>= Evidently's internal reference-size cutoff of 1000) so these tests exercise
    # the Wasserstein/Jensen-Shannon distance-metric code path that actually runs in production
    # against the real 7043-row feature_table, rather than the small-sample K-S/chi-square
    # hypothesis-test path.
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "tenure": rng.integers(0, 72, size=n) + shift,
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
        "MonthlyCharges": rng.uniform(20, 120, size=n) + shift,
        "TotalCharges": rng.uniform(0, 8000, size=n) + shift * 50,
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], size=n),
        "PaymentMethod": rng.choice(
            ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"], size=n
        ),
        "TechSupport": rng.choice(["No", "No internet service", "Yes"], size=n),
        "current_mrr": rng.uniform(0, 120, size=n) + shift,
        "trailing_3mo_avg_mrr": rng.uniform(0, 120, size=n) + shift,
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
    # Shifting 5 of the 9 feature+target-adjacent numeric columns (tenure, MonthlyCharges,
    # TotalCharges, current_mrr, trailing_3mo_avg_mrr) by a large margin. Confirmed empirically
    # (via a throwaway run of build_drift_report/extract_drift_summary at this n and shift):
    # share_of_drifted_columns=0.5556 (5/9), dataset_drift=True — comfortably past Evidently's
    # default 0.5 dataset-drift-share threshold, so this actually exercises the real
    # drift-detected branch instead of only asserting drift_share > 0.
    current_df = _synthetic_df(seed=1, shift=500.0)

    report = build_drift_report(reference_df, current_df)
    summary = extract_drift_summary(report)

    assert summary["dataset_drift_detected"] is True
