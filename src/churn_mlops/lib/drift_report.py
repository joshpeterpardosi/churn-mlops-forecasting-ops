import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

from churn_mlops.lib.churn_model import FEATURE_COLUMNS

REPORT_COLUMNS = [*FEATURE_COLUMNS, "Churn"]


def build_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Report:
    # current_df["Churn"] here is actually the model's *predicted* label (proxying for a
    # true label, since no ground truth exists in production). This introduces a systematic
    # offset versus the reference's true label distribution because of the 0.5 classification
    # threshold — not a bug, but worth knowing if TargetDriftPreset's drift signal looks
    # noisier than expected.
    column_mapping = ColumnMapping(target="Churn")

    report = Report(metrics=[DataDriftPreset(columns=FEATURE_COLUMNS), TargetDriftPreset()])
    report.run(
        reference_data=reference_df[REPORT_COLUMNS],
        current_data=current_df[REPORT_COLUMNS],
        column_mapping=column_mapping,
    )
    return report


def extract_drift_summary(report: Report) -> dict:
    result = report.as_dict()
    data_drift_result = result["metrics"][0]["result"]
    # Note: Evidently's own result dict also contains a key literally named "drift_share" at
    # this same level, which is the *threshold* (default 0.5), not the actual share. The local
    # name "drift_share" used in this function's return dict refers to share_of_drifted_columns,
    # not to Evidently's own "drift_share" field — a real naming collision worth knowing about.
    return {
        "drift_share": data_drift_result["share_of_drifted_columns"],
        "dataset_drift_detected": bool(data_drift_result["dataset_drift"]),
    }
