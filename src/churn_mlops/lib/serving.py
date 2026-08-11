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
    """

    def __init__(self, model, categorical_columns: list[str]):
        self.model = model
        self.categorical_columns = categorical_columns

    def predict(self, context, model_input: pd.DataFrame, params=None):
        df = model_input.copy()
        for col in self.categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return self.model.predict(df)

    def predict_proba(self, context, model_input: pd.DataFrame, params=None):
        df = model_input.copy()
        for col in self.categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype("category")
        return self.model.predict_proba(df)[:, 1]
