import numpy as np
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


class _RecordingProbaModel:
    """Stub standing in for a fitted LightGBM classifier's predict_proba."""

    def __init__(self):
        self.seen_dtypes = None

    def predict_proba(self, df):
        self.seen_dtypes = dict(df.dtypes)
        n = df.shape[0]
        return np.column_stack([np.full(n, 0.3), np.full(n, 0.7)])


def test_predict_proba_casts_categorical_columns_and_returns_positive_class():
    inner = _RecordingProbaModel()
    wrapped = CategoricalCastingModel(inner, categorical_columns=["Contract"])
    plain_input = pd.DataFrame({"Contract": ["Month-to-month", "One year"], "tenure": [1, 2]})

    result = wrapped.predict_proba(None, plain_input)

    assert inner.seen_dtypes["Contract"].name == "category"
    assert list(result) == [0.7, 0.7]


def test_predict_applies_the_configured_threshold():
    """The stub always returns probability 0.7. A threshold below it must flag,
    a threshold above it must not — so the wrapper is genuinely using the value
    it was given rather than LightGBM's implicit 0.5."""
    plain_input = pd.DataFrame({"Contract": ["Month-to-month"], "tenure": [1]})

    flags = CategoricalCastingModel(
        _RecordingProbaModel(), categorical_columns=["Contract"], threshold=0.25
    ).predict(None, plain_input)
    assert list(flags) == [True]

    holds = CategoricalCastingModel(
        _RecordingProbaModel(), categorical_columns=["Contract"], threshold=0.9
    ).predict(None, plain_input)
    assert list(holds) == [False]


def test_predict_delegates_to_the_model_when_no_threshold_is_set():
    """Regressors have no predict_proba, and the forecast model shares this
    wrapper. With threshold left as None, predict must pass straight through."""
    inner = _RecordingModel()
    wrapped = CategoricalCastingModel(inner, categorical_columns=["Contract"])
    plain_input = pd.DataFrame({"Contract": ["Month-to-month"], "tenure": [5]})

    result = wrapped.predict(None, plain_input)

    assert list(result) == [1]
    assert inner.seen_dtypes["Contract"].name == "category"


def test_predict_tolerates_a_model_pickled_before_threshold_existed():
    """Models registered before the decision threshold was added unpickle
    without a `threshold` attribute — unpickling restores `__dict__` directly
    and never runs `__init__`. Serving loads those artifacts from the registry,
    so predict must still delegate rather than raise AttributeError."""
    inner = _RecordingModel()
    wrapped = CategoricalCastingModel(inner, categorical_columns=["Contract"])
    del wrapped.threshold  # what an artifact pickled before the change carries
    plain_input = pd.DataFrame({"Contract": ["Month-to-month"], "tenure": [5]})

    result = wrapped.predict(None, plain_input)

    assert list(result) == [1]
    assert inner.seen_dtypes["Contract"].name == "category"
