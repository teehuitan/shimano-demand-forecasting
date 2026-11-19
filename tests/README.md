# Verification suites

Regression gate for the daily pipeline: the day/month as-of refactor (`cutoff_grain`), the
PlanModelCode grain, the MLflow registry and `main.py`'s daily flow. Run before and after any edit
to `order_book.py`, the feature builders, the `ctx` and skeleton builders, `model_registry.py`,
`db_writer.py` or `main.py`.

```
python tests/run_all.py          # one line per suite
python tests/run_all.py -v       # every individual check
```

Plain scripts, not pytest. Each exits non-zero on failure, so `run_all.py` works as a gate.

**Nothing real is touched.** `tests/no_db.py`, imported first by every suite, makes every database
engine raise on use, sets `LOG_DB_ENABLED=0`, points MLflow at a throwaway local sqlite store
(never the tracking server), and stops `main.py`'s `load_dotenv(override=True)` from putting
`.env`'s real values back. Orders come from `tests/synthetic_fabric.py`, whose PlanModelCodes never
equal their SalesModelCode and where C1/C2 share `PNC`, so grain mistakes fail instead of passing
by coincidence. The one optional real input is `data/STJ/train7_*.parquet` in
`verify_grain_contract`, skipped when absent.

## What each suite defends

| Suite | Protects |
|---|---|
| `verify_order_book` | A day-grain book queried at a month's last day returns exactly what the month-grain book returns at that month; a mid-month as-of is strictly smaller; a grain mismatch raises. |
| `verify_grain_contract` | `cutoff_grain` survives save → MLflow version tag → load; versions registered before the tag read as `'month'`; a month-trained model scored on day rows fails loudly instead of silently under-forecasting. |
| `verify_known_order` | `attach_known_order_features` on the month path reproduces the pre-refactor function exactly, all 13 columns. |
| `verify_ctx_routing` | With the day as-of pinned to month end, every column matches the month panel; `Lag*`/`ADI`/`HistVisibility_*` never move mid-month. |
| `verify_day_skeleton` | Day 0 is the last day of the origin month; short months de-duplicate; inference rejects a missing or out-of-window `cutoff_date`; day panels get their own `train7_*_day` / `inf7_*_day` file names. |
| `verify_folds` | `FoldSequence` cuts frames identical to the eager list it replaced, stays re-iterable, keeps fold count at day grain, and leaks no as-of date past the test origin's month end. A day panel's day-0 slice pools to the month panel's metrics. |
| `verify_intramonth` | The intra-month family is absent from a month panel and exactly added on a day panel; two day-0 identities (`HistVisibility_SameDayOfMonth` = `HistVisibility_AllHistory`, `Accumulation_sinceMonthStart` = `Accumulation_1mo`). |
| `verify_daily_cadence` | The orchestrator's cadence matrix: one run per new source file; retrain when the file's settled month passes `trained_through`; predict every file at day grain, at month grain only if the month-end forecast isn't written yet (or just retrained); `DAILY_FORCE_RUN` / `DAILY_FORCE_RETRAIN`. Why staleness reads MLflow, not `PredictionResult.OriginDate`. What each writer sends (`write_prediction_result`, `month_start_write_clear`, `write_outstanding_order`), checked against `create_result_tables.sql`, whose indexes must match the live database's (none on the write-only Log). |
| `verify_grain_lifecycle` | A deployment keeps its grain across every retrain, through the real `run_daily` → `train_deployment` → MLflow → `deployed_state` → `_prediction_phase` → `predict`. Day and month deployments over two month closes; the first write of each month clears last month; the same file twice does nothing unless forced; mixed grains are refused and repairable only with an explicit grain; a first deployment has no default grain; `train.py`'s CLI resolves grain the same way. |
| `verify_atomic_deploy` | A deployment changes as one unit. One panel build and one Fabric load per run; both halves from one load; a failed fit registers nothing; a save that dies part-way leaves every alias as it was, and its orphan versions are never served (even for a brand-new label); a failed alias move is rolled back; saving one half at another grain is refused; a failed retrain or grain switch never leaves the system stuck. |
| `verify_net_orders` | KnownCanceledOrder is read at the same as-of point as KnownOrder, checked row by row against ground truth from the order rows, at both grains and for every plant, through real training and the real `_prediction_phase`. |
| `verify_no_lookahead` | A day-grain training row equals the row inference builds on that cutoff date from a feed complete through that date only. An injected one-day leak is caught. |
| `verify_truncation_invariance` | The leakage gate: every feature is unchanged when every order after T is cut off, at day and month grain, at month boundaries and leap days. Injected look-aheads (day and month) are caught. Also the HasSuccessor rule. |
| `verify_cny_calendar` | `Is_CNY_Month`/`Months_to_CNY`/`CutoffMonthIsCNY` follow the February rule every year; `Months_to_CNY` is always 0..11. |
| `verify_input_coverage` | **B2:** `load_io` drops missing, pre-2010 and future dates with a WARNING per reason. **B3:** messy TFGMaterial ProductCodes are cleaned by `load_tfg_material` and resolve to the same ranges and skeleton; ordered codes with no TFG range are named in a WARNING ("will not be trained or forecast"). |
| `verify_forecast_eligibility` | **B4:** production scores exactly the rows every backtest population scores, via one rule (`meets_min_launch_age` over `ForecastEligibilityAge`), which never reaches a model input. |
| `verify_deployment_refit` | **B10:** the deployed model is refitted on all settled rows (train + val) for the round count early stopping chose, for every plant's real spec. **B17:** a silent XGBoost fit evaluates only the validation set. |
| `verify_run_resilience` | Against `synthetic_fabric.TableEngine` (in-memory tables, real commit/rollback). **B5:** OutstandingOrder keeps one snapshot per file; a rerun replaces only its own. **B6:** PredictionResult keeps the month's days; a same-day rerun replaces its own; `month_start_write_clear` is all-or-nothing, so a failure leaves the actual missing and the next file retries. **B7:** a failed retrain still scores with the incumbent; OutstandingOrder only beside a current forecast; a processed file does nothing; `main()` runs every plant and then exits non-zero. **B8:** newest parseable file name; an origin behind `trained_through` is refused. **B13:** the Log is write-only: every write appends all its rows, a rerun included. **B18:** a broken or empty run writes nothing. **B19:** predicted net never below 0; the manual benchmark is not floored. |

## The two invariants these exist to protect

1. **Day 0 is the last day of the ORIGIN month**, not day-of-next-month. It is the monthly
   pipeline's own as-of point and the only row comparable to the monthly panel.
2. **`shift_asof` preserves position in the month** — a month-end as-of shifts to the earlier
   month's month-end (Feb 28 → Jan 31, not Jan 28).

Together: **a day panel's month-end slice reproduces the monthly panel on every column.**
