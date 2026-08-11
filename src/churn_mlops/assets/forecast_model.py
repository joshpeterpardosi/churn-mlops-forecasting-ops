import mlflow
import mlflow.pyfunc
import pandas as pd
from dagster import Config, asset
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from churn_mlops.lib.forecast_model import CATEGORICAL_COLUMNS, train_forecast_model
from churn_mlops.lib.mlflow_registry import promote_if_better
from churn_mlops.lib.serving import CategoricalCastingModel

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

        # See churn_model asset for why the signature/example must be built
        # from plain-dtype data rather than the category-cast X_eval used for
        # fitting: MLflow's signature has no native "category" type, and the
        # logged input_example loses its category dtype on the JSON
        # round-trip, so a naturally plain-dtype caller (and the model's own
        # logged input_example) would otherwise fail LightGBM's categorical
        # dtype check at predict time.
        cols_to_uncast = [col for col in CATEGORICAL_COLUMNS if col in X_eval.columns]
        plain_dtype_example = X_eval.head(5).astype({col: "object" for col in cols_to_uncast})
        wrapped = CategoricalCastingModel(model, CATEGORICAL_COLUMNS)
        signature = infer_signature(plain_dtype_example, wrapped.predict(None, plain_dtype_example))
        model_info = mlflow.pyfunc.log_model(
            python_model=wrapped,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=plain_dtype_example,
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
