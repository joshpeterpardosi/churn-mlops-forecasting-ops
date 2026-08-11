import mlflow
import mlflow.lightgbm
import pandas as pd
from dagster import Config, asset
from mlflow.tracking import MlflowClient

from churn_mlops.lib.churn_model import get_production_roc_auc, train_churn_model

MODEL_NAME = "churn_model"


class MlflowConfig(Config):
    tracking_uri: str = "sqlite:///mlflow.db"


@asset
def churn_model(config: MlflowConfig, feature_table: pd.DataFrame) -> dict:
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment("churn_model")

    model, metrics = train_churn_model(feature_table)

    with mlflow.start_run() as run:
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        model_info = mlflow.lightgbm.log_model(
            model, artifact_path="model", registered_model_name=MODEL_NAME
        )

        client = MlflowClient(tracking_uri=config.tracking_uri)
        version = model_info.registered_model_version
        current_production_roc_auc = get_production_roc_auc(client, MODEL_NAME)

        promoted = current_production_roc_auc is None or metrics["roc_auc"] > current_production_roc_auc
        if promoted:
            client.set_registered_model_alias(MODEL_NAME, "production", version)

        return {
            "run_id": run.info.run_id,
            "version": version,
            "metrics": metrics,
            "promoted": promoted,
        }
