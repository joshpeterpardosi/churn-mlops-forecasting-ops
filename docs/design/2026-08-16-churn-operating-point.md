# Churn Operating Point — Design Note

**Parent spec:** [0000-project-brief.md](0000-project-brief.md)

Follows the churn model sub-project. Changes how a probability becomes a
decision; does not retrain anything.

## Problem

`train_churn_model` reported precision and recall from `model.predict`, which
applies scikit-learn's implicit 0.5 threshold. On the Telco test split that gave:

| | |
|---|---|
| precision | 0.6517 |
| recall | 0.5053 |

Recall of 0.51 means the pipeline misses roughly half the customers who actually
churn. Nothing in the code chose 0.5 — it is a library default that arrived by
omission, and 0.5 is only correct when the two kinds of error cost the same.

For churn they do not:

- **False negative** — a churner nobody contacted. The account is lost, and with
  it the remaining subscription revenue.
- **False positive** — a healthy customer got an unnecessary retention call.
  A few minutes of an agent's time.

## What was tried first, and why it failed

The textbook move is to minimise expected cost: pick the threshold that
minimises `C_fn × FN + C_fp × FP`. Pricing a miss at ten times a false alarm and
sweeping thresholds gives this:

| threshold | precision | recall | FP | FN | cost | share of healthy base flagged |
|---|---|---|---|---|---|---|
| 0.05 | 0.3489 | 0.9786 | 683 | 8 | 763 | 66.0% |
| 0.07 | 0.3723 | 0.9626 | 607 | 14 | **747** | 58.6% |
| 0.09 | 0.3940 | 0.9439 | 543 | 21 | 753 | 52.5% |
| 0.15 | 0.4437 | 0.8957 | 420 | 39 | 810 | 40.6% |
| 0.25 | 0.5000 | 0.8021 | 300 | 74 | 1040 | 29.0% |
| 0.50 | 0.6517 | 0.5053 | 101 | 185 | 1951 | 10.6% |

The unconstrained optimum sits at **0.07**, where the model flags **58.6% of the
healthy customer base**. A retention list naming three fifths of all customers
does not get worked, it gets ignored — so the modelled saving never arrives. The
rule is correct arithmetic and a useless instruction.

## Decision

Minimise expected cost **subject to a precision floor**.

```
COST_FN_TO_FP = 10.0   # a missed churner is worth ten false alarms
MIN_PRECISION = 0.50   # at least half the flagged accounts must be real churners
```

The floor is what makes the list credible enough to be acted on. It is not a
statistical criterion and it is not pretending to be one — it is the point below
which the retention team stops believing the output.

Selection lives in `select_operating_point`, sweeps thresholds from 0.01 to 0.99
in 0.01 steps, and falls back to the highest-precision candidate if nothing
clears the floor. That fallback only fires when the model has no usable signal,
in which case it should not be shipped at all.

## Result

Chosen threshold: **0.25**

| metric | at 0.50 (default) | at 0.25 (chosen) |
|---|---|---|
| recall | 0.5053 | **0.8021** |
| precision | 0.6517 | 0.5000 |
| f1 | 0.5693 | 0.6160 |
| accuracy | 0.7970 | 0.7346 |
| roc_auc | 0.8401 | 0.8401 |

Recall rises 0.2968 in absolute terms, **58.7% relative**. Precision falls to the
floor by construction. Accuracy drops, and that is the expected trade — accuracy
rewards predicting the majority class on a 26.5%-positive problem, which is
exactly the behaviour we are trying to move away from.

`roc_auc` is unchanged because it is threshold-free. That matters for promotion:
the registry still ranks versions by `roc_auc`, so choosing an operating point
cannot game the promotion gate.

## Where the threshold lives

It travels with the model. `CategoricalCastingModel` takes a `threshold`
argument, and its `predict` applies it instead of delegating to LightGBM's
implicit 0.5. The serving endpoint reads the threshold off the loaded model
rather than hardcoding a number, so re-tuning is a training-time change that
propagates on the next promotion with no edit to the API.

`threshold=None` keeps the old pass-through behaviour, which the forecast model
needs — `LGBMRegressor` has no `predict_proba`, and both models share this
wrapper.

## Non-goals

- No retraining, no hyperparameter search, no resampling. This note is only
  about turning a probability into a decision.
- The 10:1 cost ratio is an assumption, not a measurement. Nobody has given us
  the real cost of a retention call or the real value of a retained account.
  When those numbers exist, `COST_FN_TO_FP` is the one line to change.
- No per-segment thresholds. A high-value customer probably deserves a lower bar
  than a low-value one, but that needs revenue per account in the feature table.

## Tests

- The chosen point respects the precision floor.
- The chosen threshold never exceeds 0.5 and never reduces recall against the
  default, because pricing misses above false alarms can only push it down.
- The selector still returns a usable threshold when no candidate clears the
  floor.
- The wrapper applies the threshold it was given, and passes straight through
  when it has none.
