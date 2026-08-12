import mlflow
import mlflow.pyfunc
import pandas as pd
from dagster import Config, asset
from mlflow.tracking import MlflowClient

from churn_mlops.lib.churn_model import CATEGORICAL_COLUMNS, train_churn_model
from churn_mlops.lib.mlflow_registry import promote_if_better
from churn_mlops.lib.serving import CategoricalCastingModel

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

        # X_eval has the categorical columns cast to pandas "category" dtype
        # (required by LightGBM at fit time). A real caller naturally has
        # plain object/string dtype data, so the logged signature/example must
        # describe that plain-dtype contract, not the category-cast one -
        # otherwise the model's own logged input_example fails to predict
        # (MLflow's signature has no native "category" type, and the
        # input_example loses its category dtype on the JSON round-trip
        # anyway). The wrapper below casts internally so callers don't need
        # to know about the category-dtype requirement at all.
        cols_to_uncast = [col for col in CATEGORICAL_COLUMNS if col in X_eval.columns]
        plain_dtype_example = X_eval.head(5).astype(dict.fromkeys(cols_to_uncast, "object"))
        wrapped = CategoricalCastingModel(model, CATEGORICAL_COLUMNS)
        signature = mlflow.models.infer_signature(
            plain_dtype_example, wrapped.predict(None, plain_dtype_example)
        )
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
            client, MODEL_NAME, "roc_auc", metrics["roc_auc"], version, higher_is_better=True
        )

        return {
            "run_id": run.info.run_id,
            "version": version,
            "metrics": metrics,
            "promoted": promoted,
        }
