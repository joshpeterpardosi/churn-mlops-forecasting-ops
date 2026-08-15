# Data Layer — Design Spec

**Date:** 2026-08-11
**Status:** Approved, implemented, amended post-review (see below)
**Parent spec:** [0000-project-brief.md](0000-project-brief.md)
**Sub-project:** 1 of 6 (data layer — foundation for churn model, forecast model,
serving, monitoring, CI/CD)

## Amendment (2026-08-11, post-implementation review)

The final whole-branch review found two Critical defects inherited from
this spec as originally written, both fixed here plus in code:

1. **`TotalCharges` null rule contradicted the real dataset.** The original
   §Components 2 required both "cleans blank `TotalCharges` strings" and an
   unconditional null check on `TotalCharges` — but the real Kaggle Telco
   CSV has blank `TotalCharges` on every `tenure == 0` row (customers with
   no billing history yet), so the blocking check rejected every real run.
   Fixed by imputing `TotalCharges = 0.0` for `tenure == 0` during cleaning,
   keeping the null check strict for every other case.
2. **`current_mrr` leaked the `Churn` label.** MRR drops to exactly 0 at
   the churn month only when `Churn == 1`, and the original §Components 5
   defined `current_mrr` as the latest MRR value including that terminal
   row — making `current_mrr == 0` a perfect proxy for `Churn == 1`. Fixed
   by computing `current_mrr` / `trailing_3mo_avg_mrr` from months before
   the churn event only; the terminal zero row stays in `validated_mrr`
   for the forecast model, which needs it.

## Purpose

Build the Dagster asset pipeline that ingests the Telco churn CSV and a
synthetic MRR series, validates both, and produces a single feature table
consumed by the churn classification model and the MRR forecast model.

## Decisions carried from the project brief

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
  downstream. Chosen over `WARN` because the project brief requires the pipeline to
  fail fast and never propagate bad data; `WARN` would let bad rows slip
  through silently.
- `feature_table` joins `validated_churn` + `validated_mrr` on `customerID`

## Components

1. **raw_churn** — loads `data/source/telco_churn.csv` as-is into
   `raw_churn.parquet`. No transformation.
2. **validated_churn** — schema check (expected columns/types, covering every
   column `feature_table` depends on: `tenure`, `MonthlyCharges`,
   `TotalCharges`, `Churn`, `Contract`, `InternetService`, `PaymentMethod`,
   `TechSupport`), null check (`tenure`, `MonthlyCharges`, `Churn` required;
   `TotalCharges` required except when `tenure == 0`, where a blank value is
   imputed to `0.0` during cleaning rather than treated as missing —
   brand-new customers have no billing history yet), range check
   (`tenure ≥ 0`, charges `≥ 0`). Cleans `TotalCharges` blank strings
   (Telco's known dirty field; `tenure == 0` rows imputed to `0.0`, all
   other blanks remain null and fail the check). Maps `Churn` Yes/No → 1/0.
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
   computed from months *before* the churn event only (current MRR = latest
   MRR value at month `< tenure`, trailing-3-month average over the same
   pre-churn window) on `customerID`. The terminal churn-month zero row in
   `validated_mrr` is deliberately excluded from these features — including
   it would make `current_mrr` a perfect proxy for `Churn` (`current_mrr ==
   0` iff `Churn == 1`), leaking the training target into the feature set.
   The zero row stays in `validated_mrr` itself for the forecast model.

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
