import mlflow
import mlflow.lightgbm
import pandas as pd
from dagster import Config, asset
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from churn_mlops.lib.forecast_model import train_forecast_model
from churn_mlops.lib.mlflow_registry import promote_if_better

MODEL_NAME = "forecast_model"


class ForecastMlflowConfig(Config):
    tracking_uri: str = "sqlite:///mlflow.db"


@asset
def forecast_model(config: ForecastMlflowConfig, forecast_features: pd.DataFrame) -> dict:
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment("forecast_model")

    model, metrics, X_eval = train_forecast_model(forecast_features)

    with mlflow.start_run() as run:
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)

        y_pred = model.predict(X_eval)
        signature = infer_signature(X_eval, y_pred)
        model_info = mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=X_eval.head(5),
        )

        client = MlflowClient(tracking_uri=config.tracking_uri)
        version = model_info.registered_model_version
        promoted = promote_if_better(
            client, MODEL_NAME, "rmse", metrics["rmse"], version, higher_is_better=False
        )

        return {
            "run_id": run.info.run_id,
            "version": version,
            "metrics": metrics,
            "promoted": promoted,
        }
