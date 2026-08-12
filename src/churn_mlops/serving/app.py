import logging
import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from churn_mlops.lib.churn_model import FEATURE_COLUMNS as CHURN_FEATURE_COLUMNS
from churn_mlops.lib.forecast_model import FEATURE_COLUMNS as FORECAST_FEATURE_COLUMNS
from churn_mlops.lib.prediction_logging import log_prediction
from churn_mlops.lib.recursive_forecast import recursive_forecast
from churn_mlops.serving.schemas import ChurnPredictRequest, ChurnPredictResponse

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
CHURN_MODEL_URI = "models:/churn_model@production"
FORECAST_MODEL_URI = "models:/forecast_model@production"
FORECAST_FEATURES_PATH = "data/features/forecast_features.parquet"
PREDICTIONS_LOG_PATH = "data/predictions/churn_predictions.parquet"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.churn_model = None
    app.state.forecast_model = None

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        loaded = mlflow.pyfunc.load_model(CHURN_MODEL_URI)
        app.state.churn_model = loaded.unwrap_python_model()
    except Exception:
        logger.warning("Failed to load churn_model@production", exc_info=True)

    try:
        loaded = mlflow.pyfunc.load_model(FORECAST_MODEL_URI)
        app.state.forecast_model = loaded.unwrap_python_model()
    except Exception:
        logger.warning("Failed to load forecast_model@production", exc_info=True)

    yield


app = FastAPI(lifespan=lifespan)


@app.post("/predict/churn", response_model=ChurnPredictResponse)
def predict_churn(request: ChurnPredictRequest):
    if app.state.churn_model is None:
        raise HTTPException(status_code=503, detail="churn_model not loaded")

    row = pd.DataFrame([request.model_dump()])[CHURN_FEATURE_COLUMNS]
    probability = float(app.state.churn_model.predict_proba(None, row)[0])
    prediction = probability >= 0.5

    try:
        log_prediction(
            {
                **request.model_dump(),
                "churn_probability": probability,
                "churn_prediction": prediction,
            },
            log_path=PREDICTIONS_LOG_PATH,
        )
    except Exception:
        logger.warning("Failed to log prediction", exc_info=True)

    return ChurnPredictResponse(
        churn_probability=probability,
        churn_prediction=prediction,
    )


@app.get("/forecast/mrr")
def forecast_mrr(horizon: int = Query(ge=1, le=24)):
    if app.state.forecast_model is None:
        raise HTTPException(status_code=503, detail="forecast_model not loaded")

    try:
        forecast_features = pd.read_parquet(FORECAST_FEATURES_PATH)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="forecast_features data not available") from None

    seed_df = (
        forecast_features.sort_values("month")
        .groupby("customerID")
        .tail(1)
        .reset_index(drop=True)
    )
    # A customer's terminal row has target_mrr == 0 only at the month they churned
    # (per the data layer's mrr_generation.py design). Excluding those rows keeps
    # already-churned customers from being forecast forward as if still active.
    seed_df = seed_df[seed_df["target_mrr"] != 0].reset_index(drop=True)

    def predict_fn(df: pd.DataFrame):
        return app.state.forecast_model.predict(None, df[FORECAST_FEATURE_COLUMNS])

    total = recursive_forecast(seed_df, horizon=horizon, predict_fn=predict_fn)

    return {"horizon": horizon, "forecast_total_mrr": total}
