from mlflow.exceptions import MlflowException


def get_production_metric(client, model_name: str, metric_name: str) -> float | None:
    try:
        version = client.get_model_version_by_alias(model_name, "production")
    except MlflowException:
        return None
    run = client.get_run(version.run_id)
    return float(run.data.metrics[metric_name])


def promote_if_better(
    client,
    model_name: str,
    metric_name: str,
    value: float,
    version: str,
    higher_is_better: bool,
) -> bool:
    current = get_production_metric(client, model_name, metric_name)

    if current is None:
        should_promote = True
    elif higher_is_better:
        should_promote = value > current
    else:
        should_promote = value < current

    if should_promote:
        client.set_registered_model_alias(model_name, "production", version)

    return should_promote
