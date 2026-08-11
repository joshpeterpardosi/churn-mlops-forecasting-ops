import pytest
from pydantic import ValidationError

from churn_mlops.serving.schemas import ChurnPredictRequest


def _valid_kwargs():
    return dict(
        tenure=5, Contract="Month-to-month", MonthlyCharges=50.0, TotalCharges=250.0,
        InternetService="DSL", PaymentMethod="Electronic check", TechSupport="No",
        current_mrr=50.0, trailing_3mo_avg_mrr=48.0,
    )


def test_valid_request_parses():
    req = ChurnPredictRequest(**_valid_kwargs())
    assert req.tenure == 5
    assert req.Contract == "Month-to-month"


def test_negative_tenure_rejected():
    kwargs = _valid_kwargs()
    kwargs["tenure"] = -1
    with pytest.raises(ValidationError):
        ChurnPredictRequest(**kwargs)


def test_missing_field_rejected():
    with pytest.raises(ValidationError):
        ChurnPredictRequest(tenure=5, Contract="Month-to-month")


def test_wrong_type_rejected():
    kwargs = _valid_kwargs()
    kwargs["MonthlyCharges"] = "not-a-number"
    with pytest.raises(ValidationError):
        ChurnPredictRequest(**kwargs)
