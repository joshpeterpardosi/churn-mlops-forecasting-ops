from dagster import (
    AssetKey,
    DagsterEventType,
    DefaultSensorStatus,
    EventRecordsFilter,
    RunRequest,
    SensorEvaluationContext,
    SkipReason,
    define_asset_job,
    sensor,
)

from churn_mlops.assets.churn_model import churn_model

DRIFT_REPORT_ASSET_KEY = AssetKey("drift_report")

churn_retrain_job = define_asset_job("churn_retrain_job", selection=[churn_model])


@sensor(job=churn_retrain_job, default_status=DefaultSensorStatus.STOPPED)
def retrain_sensor(context: SensorEvaluationContext):
    event_records = context.instance.get_event_records(
        EventRecordsFilter(
            event_type=DagsterEventType.ASSET_MATERIALIZATION,
            asset_key=DRIFT_REPORT_ASSET_KEY,
        ),
        limit=1,
    )

    if not event_records:
        return SkipReason("No drift_report materialization found yet")

    latest_record = event_records[0]
    materialization = latest_record.event_log_entry.dagster_event.event_specific_data.materialization
    metadata = materialization.metadata

    drift_detected_metadata = metadata.get("dataset_drift_detected")
    if drift_detected_metadata is None or not drift_detected_metadata.value:
        return SkipReason("No drift detected in latest drift_report materialization")

    cursor_key = str(latest_record.storage_id)
    if context.cursor == cursor_key:
        return SkipReason("Already triggered a retrain for this drift_report materialization")

    context.update_cursor(cursor_key)
    return RunRequest(run_key=cursor_key)
