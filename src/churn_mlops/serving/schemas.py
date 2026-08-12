from typing import Literal

from pydantic import BaseModel, Field


class ChurnPredictRequest(BaseModel):
    tenure: int = Field(ge=0)
    Contract: Literal["Month-to-month", "One year", "Two year"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    InternetService: Literal["DSL", "Fiber optic", "No"]
    PaymentMethod: Literal[
        "Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"
    ]
    TechSupport: Literal["No", "No internet service", "Yes"]
    current_mrr: float = Field(ge=0)
    trailing_3mo_avg_mrr: float = Field(ge=0)


class ChurnPredictResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
