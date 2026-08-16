import mlflow.pyfunc
import pandas as pd


class CategoricalCastingModel(mlflow.pyfunc.PythonModel):
    """Wraps a fitted LightGBM model that was trained on pandas ``category``-dtype
    columns so it can be served with a plain-dtype (object/string) input contract.

    LightGBM requires the exact same categorical dtype at predict time as at fit
    time. MLflow's signature inference has no native "category" type (it reports
    such columns as plain ``string``), and the ``input_example`` logged alongside
    a model round-trips through JSON, losing any ``category`` dtype on reload. A
    caller following the logged signature naturally passes plain-dtype data, which
    then fails LightGBM's dtype check. This wrapper accepts that plain-dtype input
    and casts the categorical columns internally before delegating to the
    underlying fitted model, so the logged signature is honest and callers need no
    hidden preprocessing step.

    For classifiers it also carries the decision threshold chosen at training
    time. LightGBM's own ``predict`` would silently apply 0.5; the operating
    point is a business decision, so it travels with the model rather than being
    re-decided by whatever happens to call it. Leave ``threshold`` as ``None``
    for regressors and for any classifier that really does want the default —
    the wrapper then delegates to the underlying ``predict`` untouched.
    """

    def __init__(
        self, model, categorical_columns: list[str], threshold: float | None = None
    ):
        self.model = model
        self.categorical_columns = categorical_columns
        self.threshold = threshold

    def _cast(self, model_input: pd.DataFrame) -> pd.DataFrame:
        df = model_input.copy()
        for col in self.categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df

    def predict(self, context, model_input: pd.DataFrame, params=None):
        df = self._cast(model_input)
        if self.threshold is None:
            return self.model.predict(df)
        return self.model.predict_proba(df)[:, 1] >= self.threshold

    def predict_proba(self, context, model_input: pd.DataFrame, params=None):
        return self.model.predict_proba(self._cast(model_input))[:, 1]
