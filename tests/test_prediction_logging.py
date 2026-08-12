import pandas as pd

from churn_mlops.lib.prediction_logging import log_prediction


def test_log_prediction_creates_file_with_one_row(tmp_path):
    log_path = tmp_path / "predictions.parquet"
    log_prediction({"tenure": 5, "churn_probability": 0.7}, log_path=str(log_path))

    df = pd.read_parquet(log_path)
    assert len(df) == 1
    assert df.iloc[0]["tenure"] == 5
    assert df.iloc[0]["churn_probability"] == 0.7
    assert "logged_at" in df.columns


def test_log_prediction_appends_to_existing_file(tmp_path):
    log_path = tmp_path / "predictions.parquet"
    log_prediction({"tenure": 5, "churn_probability": 0.7}, log_path=str(log_path))
    log_prediction({"tenure": 10, "churn_probability": 0.3}, log_path=str(log_path))

    df = pd.read_parquet(log_path)
    assert len(df) == 2
    assert list(df["tenure"]) == [5, 10]


def test_log_prediction_creates_parent_directory(tmp_path):
    log_path = tmp_path / "nested" / "dir" / "predictions.parquet"
    log_prediction({"tenure": 5}, log_path=str(log_path))

    assert log_path.exists()
