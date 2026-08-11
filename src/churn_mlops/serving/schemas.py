from pydantic import BaseModel, Field


class ChurnPredictRequest(BaseModel):
    tenure: int = Field(ge=0)
    Contract: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    InternetService: str
    PaymentMethod: str
    TechSupport: str
    current_mrr: float = Field(ge=0)
    trailing_3mo_avg_mrr: float = Field(ge=0)


class ChurnPredictResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
