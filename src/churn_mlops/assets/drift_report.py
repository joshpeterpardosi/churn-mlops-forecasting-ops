import os

import pandas as pd
from dagster import MaterializeResult, MetadataValue, asset

from churn_mlops.lib.drift_report import build_drift_report, extract_drift_summary

PREDICTIONS_LOG_PATH = "data/predictions/churn_predictions.parquet"
DRIFT_REPORT_HTML_PATH = "reports/churn_drift_report.html"
MIN_DRIFT_SAMPLE_SIZE = 150


@asset
def drift_report(feature_table: pd.DataFrame) -> MaterializeResult:
    if not os.path.exists(PREDICTIONS_LOG_PATH):
        return MaterializeResult(
            metadata={
                "dataset_drift_detected": MetadataValue.bool(False),
                "note": MetadataValue.text("No predictions logged yet"),
            }
        )

    current_df = pd.read_parquet(PREDICTIONS_LOG_PATH)

    if len(current_df) < MIN_DRIFT_SAMPLE_SIZE:
        return MaterializeResult(
            metadata={
                "dataset_drift_detected": MetadataValue.bool(False),
                "note": MetadataValue.text(
                    f"Only {len(current_df)} predictions logged; need at least "
                    f"{MIN_DRIFT_SAMPLE_SIZE} to compute drift reliably"
                ),
            }
        )

    current_df = current_df.rename(columns={"churn_prediction": "Churn"})
    current_df["Churn"] = current_df["Churn"].astype(int)

    report = build_drift_report(feature_table, current_df)

    os.makedirs(os.path.dirname(DRIFT_REPORT_HTML_PATH) or ".", exist_ok=True)
    report.save_html(DRIFT_REPORT_HTML_PATH)

    summary = extract_drift_summary(report)

    return MaterializeResult(
        metadata={
            "drift_share": MetadataValue.float(summary["drift_share"]),
            "dataset_drift_detected": MetadataValue.bool(summary["dataset_drift_detected"]),
        }
    )
