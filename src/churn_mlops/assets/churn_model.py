import mlflow
import mlflow.lightgbm
import pandas as pd
from dagster import Config, asset
from mlflow.tracking import MlflowClient

from churn_mlops.lib.churn_model import train_churn_model
from churn_mlops.lib.mlflow_registry import promote_if_better

MODEL_NAME = "churn_model"


class MlflowConfig(Config):
    tracking_uri: str = "sqlite:///mlflow.db"


@asset
def churn_model(config: MlflowConfig, feature_table: pd.DataFrame) -> dict:
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment("churn_model")

    model, metrics, X_eval = train_churn_model(feature_table)

    with mlflow.start_run() as run:
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)

        input_example = X_eval.head(5)
        signature = mlflow.models.infer_signature(X_eval, model.predict(X_eval))
        model_info = mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=input_example,
        )

        client = MlflowClient(tracking_uri=config.tracking_uri)
        version = model_info.registered_model_version
        promoted = promote_if_better(
            client, MODEL_NAME, "roc_auc", metrics["roc_auc"], version, higher_is_better=True
        )

        return {
            "run_id": run.info.run_id,
            "version": version,
            "metrics": metrics,
            "promoted": promoted,
        }
