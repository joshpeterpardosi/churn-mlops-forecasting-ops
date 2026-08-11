# Data Layer — Design Spec

**Date:** 2026-08-11
**Status:** Approved, pending implementation plan
**Parent spec:** [../../../IDEA.md](../../../IDEA.md)
**Sub-project:** 1 of 6 (data layer — foundation for churn model, forecast model,
serving, monitoring, CI/CD)

## Purpose

Build the Dagster asset pipeline that ingests the Telco churn CSV and a
synthetic MRR series, validates both, and produces a single feature table
consumed by the churn classification model and the MRR forecast model.

## Decisions carried from IDEA.md

- Churn dataset: Kaggle Telco Customer Churn
- MRR generation: step drop to 0 at churn month, flat before that, plus
  noise + monthly seasonality
- Forecast model family (out of scope here, downstream consumer): gradient
  boosting w/ lag features
- Drift trigger (out of scope here): Evidently `DataDriftPreset` default,
  drifted-column share ≥ 0.5

## Architecture

Five Dagster assets, sequential:

```
raw_churn → validated_churn → synthetic_mrr_raw → validated_mrr → feature_table
```

- Storage: Parquet files on local disk — `data/raw/`, `data/validated/`,
  `data/features/`
- Dagster `asset_checks` attached to `validated_churn` and `validated_mrr`,
  severity `ERROR` — a failed check blocks materialization of everything
  downstream. Chosen over `WARN` because IDEA.md requires the pipeline to
  fail fast and never propagate bad data; `WARN` would let bad rows slip
  through silently.
- `feature_table` joins `validated_churn` + `validated_mrr` on `customerID`

## Components

1. **raw_churn** — loads `data/source/telco_churn.csv` as-is into
   `raw_churn.parquet`. No transformation.
2. **validated_churn** — schema check (expected columns/types), null check
   (`tenure`, `MonthlyCharges`, `TotalCharges`, `Churn` required),
   range check (`tenure ≥ 0`, charges `≥ 0`). Cleans `TotalCharges` blank
   strings (Telco's known dirty field). Maps `Churn` Yes/No → 1/0.
3. **synthetic_mrr_raw** — generates a monthly MRR series per customer from
   `validated_churn`:
   - MRR = `MonthlyCharges`, flat from month 0 to `tenure - 1`
   - if `Churn == 1`: MRR drops to 0 at month `tenure`
   - if `Churn == 0`: MRR stays flat through the observation window
   - overlay: small Gaussian noise + monthly seasonality (sine wave, small
     amplitude), both bounded so MRR never goes negative
4. **validated_mrr** — schema/null/range check on the generated series
   (`MRR ≥ 0`, no NaNs, full date coverage per customer).
5. **feature_table** — joins `validated_churn` (curated subset: `tenure`,
   `Contract`, `MonthlyCharges`, `TotalCharges`, `InternetService`,
   `PaymentMethod`, `TechSupport`, `Churn`) with MRR-derived features
   (current MRR, trailing-3-month average) on `customerID`.

## Data Flow

1. `raw_churn` loads Telco CSV unmodified
2. `validated_churn` runs schema/null/range checks, cleans `TotalCharges`,
   maps `Churn` — halts pipeline on check failure
3. `synthetic_mrr_raw` generates the MRR series from validated churn data
4. `validated_mrr` runs schema/null/range checks on the generated series —
   halts pipeline on check failure
5. `feature_table` joins both validated sources into the output consumed by
   the churn and forecast models (out of scope for this spec)

## Error Handling

- Dagster `asset_checks`, severity `ERROR`, on `validated_churn` and
  `validated_mrr` — failure blocks downstream materialization; failed run
  is visible in the Dagster UI.
- Asset logic is written as pure functions (DataFrame in, DataFrame out),
  separated from Dagster IO boilerplate, so it is testable without running
  the pipeline.

## Testing

- **pytest**, pure-function level (no Dagster run required):
  - `validated_churn` logic — catches malformed `TotalCharges`, negative
    `tenure`, missing `Churn`
  - `synthetic_mrr_raw` logic — churned customer hits 0 exactly at the
    `tenure` month; active customer stays flat through the window;
    noise/seasonality never pushes MRR negative
  - `validated_mrr` logic — same check pattern as churn validation
  - `feature_table` join — no duplicate `customerID`, row count matches
    `validated_churn`, no unexpected nulls post-join
- **Dagster asset checks** — same rules wired into the live pipeline,
  catching bad real data at run time, not just bad test fixtures.

## Out of Scope (future sub-projects)

- Churn model training / MLflow tracking
- Forecast model training / MLflow tracking
- FastAPI serving layer
- Evidently monitoring + Dagster retrain sensor
- CI/CD (GitHub Actions), `gh` repo push
