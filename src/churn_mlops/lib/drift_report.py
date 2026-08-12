import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

from churn_mlops.lib.churn_model import FEATURE_COLUMNS

REPORT_COLUMNS = FEATURE_COLUMNS + ["Churn"]


def build_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Report:
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
    return {
        "drift_share": data_drift_result["share_of_drifted_columns"],
        "dataset_drift_detected": bool(data_drift_result["dataset_drift"]),
    }
