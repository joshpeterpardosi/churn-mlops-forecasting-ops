import logging
from contextlib import asynccontextmanager

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from churn_mlops.lib.churn_model import FEATURE_COLUMNS as CHURN_FEATURE_COLUMNS
from churn_mlops.lib.forecast_model import FEATURE_COLUMNS as FORECAST_FEATURE_COLUMNS
from churn_mlops.lib.recursive_forecast import recursive_forecast
from churn_mlops.serving.schemas import ChurnPredictRequest, ChurnPredictResponse

logger = logging.getLogger(__name__)

CHURN_MODEL_URI = "models:/churn_model@production"
FORECAST_MODEL_URI = "models:/forecast_model@production"
FORECAST_FEATURES_PATH = "data/features/forecast_features.parquet"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.churn_model = None
    app.state.forecast_model = None

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

    return ChurnPredictResponse(
        churn_probability=probability,
        churn_prediction=probability >= 0.5,
    )


@app.get("/forecast/mrr")
def forecast_mrr(horizon: int = Query(ge=1, le=24)):
    if app.state.forecast_model is None:
        raise HTTPException(status_code=503, detail="forecast_model not loaded")

    try:
        forecast_features = pd.read_parquet(FORECAST_FEATURES_PATH)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="forecast_features data not available")

    seed_df = (
        forecast_features.sort_values("month")
        .groupby("customerID")
        .tail(1)
        .reset_index(drop=True)
    )

    def predict_fn(df: pd.DataFrame):
        return app.state.forecast_model.predict(None, df[FORECAST_FEATURE_COLUMNS])

    total = recursive_forecast(seed_df, horizon=horizon, predict_fn=predict_fn)

    return {"horizon": horizon, "forecast_total_mrr": total}
