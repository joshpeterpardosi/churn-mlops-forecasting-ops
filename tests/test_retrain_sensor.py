from dagster import (
    MaterializeResult,
    MetadataValue,
    RunRequest,
    SkipReason,
    asset,
    build_sensor_context,
    instance_for_test,
    materialize,
)

from churn_mlops.sensors.retrain_sensor import retrain_sensor


@asset(name="drift_report")
def _stub_drift_report_drifted():
    return MaterializeResult(
        metadata={"drift_share": MetadataValue.float(0.6), "dataset_drift_detected": MetadataValue.bool(True)}
    )


@asset(name="drift_report")
def _stub_drift_report_clean():
    return MaterializeResult(
        metadata={"drift_share": MetadataValue.float(0.1), "dataset_drift_detected": MetadataValue.bool(False)}
    )


def test_sensor_fires_when_drift_detected():
    with instance_for_test() as instance:
        materialize([_stub_drift_report_drifted], instance=instance)

        context = build_sensor_context(instance=instance)
        result = retrain_sensor(context)

        assert isinstance(result, RunRequest)


def test_sensor_skips_when_no_drift():
    with instance_for_test() as instance:
        materialize([_stub_drift_report_clean], instance=instance)

        context = build_sensor_context(instance=instance)
        result = retrain_sensor(context)

        assert isinstance(result, SkipReason)


def test_sensor_does_not_refire_for_same_materialization():
    with instance_for_test() as instance:
        materialize([_stub_drift_report_drifted], instance=instance)

        context1 = build_sensor_context(instance=instance)
        first_result = retrain_sensor(context1)
        assert isinstance(first_result, RunRequest)

        context2 = build_sensor_context(instance=instance, cursor=context1.cursor)
        second_result = retrain_sensor(context2)

        assert isinstance(second_result, SkipReason)
