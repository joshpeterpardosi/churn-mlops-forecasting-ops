import pandas as pd

from churn_mlops.lib.serving import CategoricalCastingModel


class _RecordingModel:
    """Stub standing in for a fitted LightGBM model: records the dtype of each
    column it was asked to predict on, so tests can assert the cast happened."""

    def __init__(self):
        self.seen_dtypes = None

    def predict(self, df):
        self.seen_dtypes = dict(df.dtypes)
        return df.shape[0] * [1]


def test_predict_casts_categorical_columns_before_delegating():
    inner = _RecordingModel()
    wrapped = CategoricalCastingModel(inner, categorical_columns=["Contract", "TechSupport"])

    plain_input = pd.DataFrame({
        "Contract": ["Month-to-month", "One year"],
        "TechSupport": ["Yes", "No"],
        "tenure": [1, 24],
    })
    assert plain_input["Contract"].dtype == object

    result = wrapped.predict(None, plain_input)

    assert inner.seen_dtypes["Contract"].name == "category"
    assert inner.seen_dtypes["TechSupport"].name == "category"
    assert inner.seen_dtypes["tenure"].name != "category"
    assert len(result) == 2

    # original caller's DataFrame is untouched (predict copies, doesn't mutate)
    assert plain_input["Contract"].dtype == object


def test_predict_does_not_mutate_caller_dataframe():
    inner = _RecordingModel()
    wrapped = CategoricalCastingModel(inner, categorical_columns=["Contract"])
    plain_input = pd.DataFrame({"Contract": ["Month-to-month", "One year"]})

    wrapped.predict(None, plain_input)

    assert plain_input["Contract"].dtype == object


def test_predict_skips_categorical_columns_absent_from_input():
    """A categorical column configured on the wrapper but missing from a given
    input frame (e.g. a narrower stub in a test) should not raise a KeyError."""
    inner = _RecordingModel()
    wrapped = CategoricalCastingModel(inner, categorical_columns=["Contract", "NotPresent"])
    plain_input = pd.DataFrame({"Contract": ["Month-to-month"], "tenure": [5]})

    result = wrapped.predict(None, plain_input)

    assert inner.seen_dtypes["Contract"].name == "category"
    assert "NotPresent" not in inner.seen_dtypes
    assert len(result) == 1
