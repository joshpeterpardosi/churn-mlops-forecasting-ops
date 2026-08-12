from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

DEFAULT_PREDICTIONS_PATH = "data/predictions/churn_predictions.parquet"


def log_prediction(row: dict, log_path: str = DEFAULT_PREDICTIONS_PATH) -> None:
    row = dict(row)
    row["logged_at"] = datetime.now(UTC).isoformat()

    new_row_df = pd.DataFrame([row])

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing_df = pd.read_parquet(path)
        combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    else:
        combined_df = new_row_df

    combined_df.to_parquet(path, index=False)
