# STJ Sales Forecast — Pipeline Guide

This document explains how the demand-forecasting pipeline works end to end: what runs every day,
how the data flows from Fabric to SQL Server, and what each module in `adj_forecast_lib/` is
responsible for. It's written for someone picking up this codebase for the first time.

## 1. What this system does

Every day, for each plant (STJ = `3300`, SPL = `2100`), the pipeline forecasts **Adjustment** —
the gap between what's already been ordered (`KnownOrder`) and what will actually be ordered
(`ActualOrder`) — for each product (`PlanModelCode`), for 1–6 months ahead (`Horizon`). The
final number shown to the business is:

```
Predicted = KnownOrder + PredictedAdjustment
```

Every plant is scored with **`model_{label}`** — a trained ML model (one per horizon) predicting
Adjustment. The model type (XGBoost/CatBoost) and hyperparameters per horizon come from
experimentation, not a fixed choice (`BEST_PARAMS_BY_PLANT`), and specific PartCodes can route to
a separately-tuned quantile variant instead of the standard model (`QUANTILE_PARTCODES_BY_PLANT`).
For plants where `PLANT_INCLUDES_MANUAL_METHOD` is set (STJ only,
since that's the only plant whose business process has an existing Excel method to compare
against), a second row is also written: **`manual_{window}`**, which reproduces the business's
Excel "recent incoming trend" method (`ManualSim{window}`), so the model can be compared against
the status quo.

Results land in SQL Server (`AI.DemandForecast` schema), which a dashboard layer reads and the dashboard lives in a different repo.

## 2. Deployment & scheduling

- `deploy.ps1` runs in Bamboo. It writes `.env` and two batch files (`run.bat`, `start.bat`) onto
  the target server from Bamboo variables, then SSHes in to run `start.bat` once.
- `start.bat` does first-time/redeploy setup: verifies `uv` is on PATH, rebuilds `.venv` if it's
  stale, runs `uv sync --locked`, then (re)registers a Windows Scheduled Task that runs `run.bat`
  daily at **11:00**, plus once on every system startup.
- `run.bat` is the actual scheduled command: `uv run python main.py`. It runs once and exits —
  there's no long-lived server process to babysit.
- **`start.bat` also runs `run.bat` once immediately, at the end of every deploy** (after the
  Scheduled Task is (re)registered), so a deploy doesn't have to wait for the next 11:00 trigger
  or a reboot to take effect. The next *scheduled* run is still 11:00 the following day.

## 3. `main.py` — the daily orchestrator

`main()` reads every `DAILY_*` env var (see `.env.example`), loops over `DAILY_PLANTS`, and calls
`run_daily(plant, ...)` for each. There are no CLI args — everything is environment-driven, so an
unattended scheduled run behaves identically to a manual one, and a missing/misconfigured key
fails loudly (`KeyError`) instead of silently running on a stale default.

`run_daily` runs once per new source file, in three phases:

```
run_daily(plant):
    if the file already has OutstandingOrder rows: stop       # already processed (unless DAILY_FORCE_*)
    retrain    if the file's settled month > MLflow trained_through, a grain change, or FORCE_RETRAIN
    predict    day grain: every file; month grain: only if this month's forecast isn't written yet
               writes ProductLifecycle, then PredictionResult (+Log) for this OriginDate; the first
               write of a month also writes ActualOrder for N and clears last month, in one transaction
    OutstandingOrder   only when the forecast is current (scored now, or already written)
```

**Retrain** fires once a month settles: when the settled month the file implies is newer than the
deployment's `trained_through` in MLflow. A failed retrain leaves the previous deployment in use,
and prediction still scores with it.

**OutstandingOrder** is refreshed only beside a current forecast, so the two tables always describe
the same as-of point: if prediction fails, the previous snapshot is kept. Writing it is also what
marks the file as processed, so a failed run is retried on the next trigger.

`latest_source_file_date()` (`preprocessing.py`) recognizes two `_source_file` naming
conventions: plain `YYYYMMDD.xlsx`, and SPL's `YYYYMMDD - YYYYMMDD.xlsx` range format, whose
effective as-of date is the day *after* the range's end date. A filename matching neither pattern
is logged and treated as `(None, None)` — settled-month detection simply doesn't advance that day.

There is no separate `RunLog` table — each run's console output is also written to a timestamped
file under `logs/{PlantName}/`, and that log file *is* the audit trail.

Every record is additionally mirrored into `[AI].AI.AIAPPLOGS` (the shared PP-Dashboard log table)
as `ApplicationName='PP-Dashboard'`, `agent_id='Demand-forecast-ML'`. That copy is for querying,
not for audit — it's written asynchronously and drops records rather than ever failing a run, so
the log file above remains the authoritative record. Each `configure_logging` call mints a fresh
`CorrelationId`, so one plant's run is `WHERE CorrelationId = ...`; the plant, purpose and log file
path are in the `Properties` JSON. Set `LOG_DB_ENABLED=0` to turn the mirror off.

A failed phase is logged with its traceback and the run carries on with the phases that can still
run; the run ends with one summary line per plant, then raises `DailyRunError`. `main()` still runs
every remaining plant and exits non-zero at the end, so the scheduled task sees the failure.

## 4. Core vocabulary (read this before anything else makes sense)

| Term | Meaning |
|---|---|
| `PlanModelCode` | The product-level entity being forecast — the grain of every table and the training panel (= SIC's `Plan Model ID`). |
| `SalesModelCode` | The sales model a `PlanModelCode` belongs to. Kept as a feature and on `ProductLifecycle`/`PredictionResultLog`; `ProductSuccessor` pairs are stored at this grain. |
| `PartCode` | A coarser grouping above both (a product family/type). |
| `MonthIdx` | A monotonic integer month counter (`year*12 + month`), because `YYYYMM` ints aren't arithmetic-safe across year boundaries (`202511 + 3` isn't a real month). Converted via `config.to_month_idx` / `month_idx_to_yyyymm`. |
| `Origin` | The month *as of which* a forecast is made — "today," in model terms. |
| `Target` | The month *being forecast*. |
| `Horizon` | `Target - Origin`, in months (1–6). |
| `KnownOrder` | Cumulative order quantity already booked for `Target`, as of `Origin` — this is the order book's running total, not the eventual total. Equivalent to "Request Order" in the UI, whether expressed as net or gross. |
| `ActualOrder` | The *settled*, final total for a month, once it's fully in the past. |
| `Adjustment` | `ActualOrder - KnownOrder` — **this is the model's training target**, not the raw order volume. The model is only trying to predict how much more will come in on top of what's already known. |
| Settled vs. pending | A month is "settled" once enough time has passed that its orders are no longer changing. `max_settled_idx` is the boundary; everything after it is still "pending." |

Forecasting `Adjustment` instead of `ActualOrder` directly is the central design choice: it lets
`KnownOrder` act as a strong, leak-safe baseline, and the model only needs to learn the residual.

## 5. End-to-end data flow

```
Fabric (Warehouse_AI.dbo.IncomingOrder_GoogleLooker, via io.sql)
  │
  ▼
load_io()  (preprocessing.py)              — clean, type-cast, flag cancellations
  │
  ▼
build_order_book_context()  (preprocessing.py)
  ├─ build_order_book()  (order_book.py)   — cumulative KnownOrder book per (code, RequestMonth, EntryMonth)
  ├─ resolve_effective_ranges_with_tfg()   — launch/end dates from TFGMaterial (MDM)
  ├─ get_predecessor_successor_flags_from_db()  — replacement relationships
  └─ → one `ctx` dict, shared across training AND prediction for this run
  │
  ├─→ build_full_panel()      (training)   → attach_* feature families → Adjustment label
  │      │
  │      ▼
  │   train_and_save()  (train.py)  → 6 horizon models × (standard + quantile) → MLflow registry
  │
  └─→ build_inference_skeleton() (live scoring) → same attach_* feature families, no label
         │
         ▼
      predict() / predict_manual()  (predict.py)  → Predicted = KnownOrder + PredictedAdjustment
         │
         ▼
      db_writer.py  → SQL Server: PredictionResult, PredictionResultLog, ActualOrder,
                       ProductLifecycle, OutstandingOrder
```

The critical property here: **training and inference call the exact same feature-attaching
functions** (`_attach_all_features` in `preprocessing.py`). This guarantees a feature computed at
training time and at prediction time can never silently drift apart into two different
implementations.

## 6. Module-by-module reference (`adj_forecast_lib/`)

### Live daily-pipeline modules (everything `main.py` touches)

- **`config.py`** — All environment-sourced constants (plant names, excluded part codes, external
  event date ranges, COVID window, MLflow URI, etc.) plus the `MonthIdx` conversion helpers. Every
  constant the live pipeline needs is `os.environ[...]` with **no default** — a missing `.env` key
  fails at import time, not silently mid-run.
- **`order_book.py`** — The core primitive underlying `KnownOrder`. Builds a cumulative-by-
  `EntryMonthIdx` table per group, then `query_known_order` does an `merge_asof` "as of this
  cutoff month, what had been booked for that target month" lookup. Also used (with a different
  cutoff column) to compute `KnownCanceledOrder`.
- **`preprocessing.py`** — The biggest file; orchestrates everything from a Fabric query to a
  model-ready DataFrame:
  - `load_io()` — loads + cleans one plant's incoming orders, flags known-valid cancellations,
    drops excluded part codes and negative lead times (screened on the raw
    `Customer_Request_Date`, before the `Ship_To_Code` bring-forward), then the brought-forward
    rows whose request month had already closed when they were entered. STJ `SLH_Order='Y'` rows
    are kept (§13).
  - `build_order_book_context()` — builds the shared `ctx` dict (order books, launch dates,
    lifecycle ranges, predecessor/successor flags, the settled-month boundary). Everything
    downstream reads from this one dict instead of re-querying.
  - `_attach_all_features()` — the fixed sequence of feature-family calls (calendar → history →
    lag-events → trend/ADI → known-order → historical visibility → group-level → lifecycle →
    external factor). Shared by both training and inference.
  - `build_full_panel()` — full historical training panel (one row per code × target × horizon),
    with `ActualOrder`/`Adjustment` labels attached.
  - `build_inference_skeleton()` — the live, label-free equivalent for one Origin month; also
    reconstructs just enough historical rows (`build_adjustment_history_lookback`) so
    `Adjustment`-dependent features can self-join without needing the full training panel loaded.
- **`features.py`** — The feature-family builders themselves. Each function is documented with
  *why it's anchored where it is* (Origin vs. Target) — this is what keeps every feature
  leakage-safe for whichever horizon it's used at. Families include:
  - Calendar (`Year`/`Month`/`Month_sin`/`cos`/`Quarter`/CNY proximity)
  - History/lag (`Lag0..Lag23`, trailing means, trend slope, volatility, z-scores)
  - Known-order momentum/accumulation/YoY
  - Historical visibility (how much of a past month's eventual total was knowable that far out)
  - ADI (Average Demand Interval — intermittency signal)
  - Lag-event counts (hand-specified external event windows)
  - `AdjTrailMean{w}` / `ManualSim{w}` — label-dependent, must run *after* `Adjustment` exists.
- **`lifecycle_features.py`** — Cold-start / product-age features: months since launch, months
  since last demand, consecutive demand streak, predecessor/successor flags, and an entity's
  demand share relative to its `PartCode` family.
- **`lifecycle_cleaning.py`** — Resolves each `PlanModelCode`'s effective launch/end date from
  `TFGMaterial` (MDM): every ProductCode under the code counts, ordered or not. Also resolves
  predecessor/successor relationships from `AI.DemandForecast.ProductSuccessor` (stored as
  `SalesModelCode` pairs, mapped onto `PlanModelCode` at read time). The Excel-based path
  (`get_predecessor_successor_flags_from_excel`) still exists but the live pipeline uses the
  DB-backed one.
- **`train.py`** — `train_and_save()` trains one model per horizon (`model_params.BEST_PARAMS_BY_PLANT[plant][h]` picks the
  model family + hyperparameters — horizon 1 uses CatBoost/MAE, horizons 2–6 use XGBoost with a
  Tweedie objective) and saves both the standard model and a `{label}_quantile` variant (only
  meaningfully different for the `SL`/`RD` part codes — see `predict.py`'s routing). No held-out
  test set in production — only a recent validation tail for early stopping.
- **`models.py`** — Thin adapters so XGBoost/LightGBM/CatBoost can be trained/predicted through one
  uniform interface (`ADAPTERS`, `PREDICT_FUNCS`), despite each library having a different fit/
  predict API (CatBoost in particular needs categoricals cast to strings).
- **`model_registry.py`** — Persists trained models via the MLflow Model Registry (not a local
  file registry) — one registered name per `(plant, horizon, label)`, versions accumulate,
  `load_model()` always takes the latest version. Sets
  `MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT=True` before importing `mlflow`, because the Bamboo
  agent's console is cp1252 and can't encode the emoji MLflow prints after every run — left
  unset, that print crashes the run.
- **`predict.py`** — Scores a live origin month with either the trained model
  (`_score_routed`, splitting `SL`/`RD` part codes off to the quantile-labeled models) or the
  manual `ManualSim{window}` method. Output is long-format (`Method` distinguishes rows, not
  columns), so both methods coexist in the same table.
- **`collate_actual_order.py`** — Once a month is settled, computes its final gross/net
  `ActualOrder` per code, for every month in that code's active lifespan.
- **`collate_outstanding_order.py`** — Computes same-day `OutstandingOrder`: orders already
  requested before today's source file's date, excluding the N+1 target month (that's already
  covered by `KnownOrder`) and net of known-valid cancellations. Runs only when that day's
  forecast is current (see §3), so it always sits beside the forecast it belongs to.
- **`db_writer.py`** — All SQL Server writes, against the schema defined in
  `create_result_tables.sql`. The engine is created with `fast_executemany=True` (pyodbc's
  batched-insert mode), since these writes are large multi-row DataFrames:
  - `PredictionResult` — **current month only**: one set of rows per OriginDate (a same-day rerun
    replaces its own), cleared once when the month's first write lands.
  - `PredictionResultLog` — append-only history of every prediction ever written (never cleared).
    Write-only: never read back, so it has no index; a same-day rerun appends its rows again.
  - `ActualOrder` — upserted (insert new months, correct existing ones in place).
  - `ProductLifecycle` — upserted, never wiped (other reads depend on it always being populated).
  - `OutstandingOrder` — one snapshot per source file (a rerun replaces its own), cleared in the
    same transaction as `PredictionResult` at the month's first write.
  - `ProductSuccessor` — maintained manually via `lifecycle_cleaning.py`'s CLI, not touched by
    `main.py`'s daily run.
- **`logging_setup.py`** — `configure_logging(purpose, plant)` sets up one console + one
  timestamped file handler under `logs/{PlantName}/`, plus the `AIAPPLOGS` handler below. Called
  once per plant per run in `main.py`.
- **`db_logging.py`** — The `[AI].AI.AIAPPLOGS` handler. Queues records and writes them in batches
  from a daemon thread (its own small engine, so log writes never contend with `db_writer`'s pool),
  and swallows every DB error — an unreachable log DB costs you the mirror, never the run.
- **`wipe_test_run.py`** — Manual cleanup CLI: deletes a test label's rows from `PredictionResult`,
  `PredictionResultLog`, `ActualOrder`, and `OutstandingOrder`, plus its MLflow registered models.
  Excludes `ProductLifecycle`/`ProductSuccessor` — both are upserted, never wiped, and not
  test-run-specific. Dry-run by default (`--confirm` to actually delete).
- **`data_loading.py`** — `clean_code` normalizes product codes (uppercase, strip non-alphanumerics).

### Supporting / offline modules (not called by `main.py`, but used for model development)

- **`folds.py`** — Fold generators for backtesting (`fixed_split_*`, `walk_forward_*`) and for the
  no-test-set deployment split (`deployment_split_horizon`, what `train.py` actually uses).
- **`evaluation.py`** — MAE/RMSE/MAPE/WAPE/bias metrics and sample-weighting helpers (down-weight
  COVID-era targets, up-weight young SKUs, up-weight external-event targets). `train.py` uses this
  for the deployment fit.
- **`replacement_workbook.py`** — Parses the STJ replacement Excel workbook. The daily run reads
  predecessor/successor pairs from `ProductSuccessor`; this module is what
  `lifecycle_cleaning.py`'s CLI uses when those pairs are loaded from the workbook instead.
- **`preprocessing.preprocess_sic_plan`** — Reads the business's own SIC demand plan (a separate
  manual forecast) from Fabric `Warehouse_AI.dbo.SIC_DemandPlan`, joined to the panel on SIC's
  `Plan Model ID` (= our `PlanModelCode`), for comparison against the model. Read through the same
  service-account Fabric connection as the incoming orders.

## 7. Feature families at a glance

`preprocessing._feature_families()` is the authoritative list. Broadly:

| Family | What it captures |
|---|---|
| Calendar | Month/quarter/seasonality, both at Origin and Target |
| LegacyHistory | Lag0–23, trailing means/ratios, trend slope, volatility, z-score, ADI, stability |
| KnownOrder | Momentum/accumulation of the order book itself, at 1/3/6 month and YoY offsets |
| HistoricalVisibility | How much of a past month's eventual total was visible that many months out |
| GroupLevel | Same history features, rolled up to `PartCode` |
| MarketShock | Plant-wide order volume shock (month-on-month change, 3-month z-score) |
| CountryVisibility | Country mix of the order book: the focus country's (CN by default) known vs. usual share, and the visibility / implied adjustment that mix suggests |
| IntraMonth | Day grain only: where the as-of date sits in the month, arrival velocity, same-day-of-month visibility |
| ExternalFactor | `HasExternalEvent`: hand-specified event windows, point-in-time - an event counts only from origins at or after its start month (end announced: targets up to it; end unknown: while it runs, every target) |
| MovingAverageBaselines | `AdjTrailMean{w}` and `ManualSim{w}` — label-dependent |
| Lifecycle | Age since launch, demand streaks, predecessor/successor, relative demand share |

`MarketShock`, `CountryVisibility` and `IntraMonth` were added in 2026-09, validated, and are part
of the deployed feature set (`preprocessing.NEW_FEATURE_FAMILIES` names them, so an experiment can
rebuild the feature set from before them). `_feature_families()` is the list the model actually
trains on. Code comments call an older column set "the reference panel". It was the notebook-era
panel then named `train10`, and has nothing to do with the `data/*/train10_*.parquet` /
`train11_*.parquet` files the CLI writes (train11 = point-in-time `HasExternalEvent`).
`data/STJ/train12_*` was a one-off comparison panel with every event range on the end-known list (hindsight about
each event's end); it is never a baseline, see §14.

## 8. Models

- One model per horizon (1–6), not one shared model — `model_params.BEST_PARAMS_BY_PLANT` picks a different model
  family/hyperparameters per horizon (CatBoost/MAE for horizon 1, XGBoost/Tweedie for 2–6).
- A parallel `BEST_PARAMS_QUANTILE` spec trains a quantile-regression variant, saved under
  `{label}_quantile`. `predict.py` always routes `SL`/`RD` part codes to the quantile models and
  everything else to the standard ones — both are always trained, every day.
- Models are registered in MLflow under `{PlantName}_h{horizon}_{label}`; `load_model()` always
  loads the newest version for that name.
- Sample weighting (`evaluation.py`) down-weights COVID-era targets, and can up-weight young SKUs
  or external-event targets — configured per-horizon in `BEST_PARAMS`.

## 9. Environment configuration

Everything is driven by `.env` (see `.env.example` for the full key list). Broad groups:

- `QA_DB_*` / `FABRIC_DB_*` — SQL Server connection strings (QA/AI database vs. Fabric warehouse).
- `ODBC_DRIVER` — ODBC driver name for every SQL Server connection (e.g. `ODBC Driver 17 for SQL Server`).
- `IO_SQL_PATH` / `IO_OUTSTANDING_SQL_PATH` — paths to the `.sql` files that load incoming orders
  (`io.sql`, `io_outstanding.sql`).
- Pipeline config — plant names/exclusions, external event ranges, COVID window, CNY months,
  MLflow tracking URI, XGBoost training size knobs.
- `DAILY_*` — everything `main.py`'s `main()` reads to drive the scheduled run (plants, label,
  covid weight, validation months, horizons, min-months-since-launch, quantile part codes, manual
  window).

There is deliberately **no default** for any of these — a missing key is a loud `KeyError` at
import time, not a silent fallback to a stale value in production.

## 10. Database schema (`create_result_tables.sql`)

All tables live under `AI.DemandForecast`:

- **`PredictionResult`** — current month's predictions only, one set per OriginDate; cleared at
  the month's first write. This is what a dashboard reads for "latest."
- **`PredictionResultLog`** — same shape plus `SalesModelCode` (frozen as of the prediction),
  append-only, never cleared — the full history. Write-only (no index beyond the primary key); after
  a same-day rerun, the latest row per forecast and OriginDate (by `Id`) is the current one.
- **`ActualOrder`** — settled totals plus `SalesResult`, upserted (unique on
  Plant/PlanModelCode/TargetYearMonth).
- **`ProductLifecycle`** — resolved launch/end dates per `PlanModelCode` (unique on
  Plant/PlanModelCode), upserted, never wiped. The other tables get `SalesModelCode`/
  `PlanModelText` by joining here on (Plant, PlanModelCode).
- **`OutstandingOrder`** — one snapshot of unfulfilled orders per source file this month; cleared
  with `PredictionResult` at the month's first write.
- **`ProductSuccessor`** — predecessor/successor relationship pairs, maintained manually via
  `lifecycle_cleaning.py`'s own CLI, not touched by the daily run.

## 11. Directory layout

```
main.py                    Entry point / daily orchestrator
adj_forecast_lib/          All pipeline logic (see §6)
io.sql / io_outstanding.sql  Fabric queries for incoming orders
create_result_tables.sql   SQL Server schema (run manually against a fresh DB)
amend_schema.sql           Run-once SalesModelCode -> PlanModelCode migration for an existing DB (deletes all rows)
deploy.ps1                 Bamboo deployment script (writes .env/run.bat/start.bat, SSHes to run)
notebooks/                 Exploratory/experiment notebooks (not part of the live pipeline)
logs/{Plant}/              One timestamped log file per run
data/, results/, models/   Local scratch output from CLI tools (not written by main.py directly)
```

## 12. Running things manually

Every live module also has its own CLI (`python -m adj_forecast_lib.<module> --help`), useful for
backfills or debugging without running the full daily job:

```bash
# Build/save a training panel to parquet
python -m adj_forecast_lib.preprocessing --type train --plant 3300

# Train one label's models (both standard and quantile need running separately)
python -m adj_forecast_lib.train --plant 3300 --label v1 --params standard
python -m adj_forecast_lib.train --plant 3300 --label v1 --params quantile

# Score a specific origin month without touching the DB
python -m adj_forecast_lib.predict --plant 3300 --method model --label v1 --origin 202607

# Recompute settled ActualOrder for one month (add --write-to-db for a real backfill)
python -m adj_forecast_lib.collate_actual_order --plant 3300 --month 202608

# Recompute today's OutstandingOrder snapshot
python -m adj_forecast_lib.collate_outstanding_order --plant 3300

# Clean up everything written by a test run (dry-run by default)
python -m adj_forecast_lib.wipe_test_run --plant 3300 --label qa_v1
```

## 13. Design decisions worth knowing (so you don't "fix" them)

- **`Adjustment`, not `ActualOrder`, is the training target.** The model predicts the residual on
  top of `KnownOrder`, not the raw total.
- **Origin-anchored vs. Target-anchored features are anchored deliberately**, not arbitrarily —
  anything anchored at Origin is safe at every horizon; anything at Target usually isn't.
  `HasExternalEvent` used to be Target-only and leaked: it flagged an event month up to 6 months before
  anyone knew of the event. Since train11 it is point-in-time (`config.is_known_event_idx`), and its
  hindsight twin `EventTargetHindsight` - not a feature - only drives the external-event sample weight.
  `LagEventCount` (event months in the trailing window) is Origin-anchored and was always safe.
- **Only PlanModelCodes that have ever been ordered are in scope.** Lifecycle, `ActualOrder`,
  training and prediction all cover just the codes in that plant's (PartCode-filtered) order
  book. Codes that are in TFGMaterial under the plant but were never ordered are deliberately
  left out. They'd have no `PartCode` (it comes from orders, and the DB columns are `NOT NULL`),
  and `collate_actual_order` would pad them with zero actuals.
- **`ctx` is built once per plant per run and threaded through everything** (lifecycle write,
  training, prediction) — never rebuilt mid-run, so the settled-month boundary can't drift between
  steps of the same run.
- **`PredictionResult`/`OutstandingOrder` are cleared monthly; `PredictionResultLog`/`ActualOrder`/
  `ProductLifecycle` are upsert-or-append.** Don't add a clear step to the latter three.
- **Retrain fires only on a new settled month; `OutstandingOrder` refreshes only beside a current
  forecast.** A failed prediction keeps the previous snapshot, so the two tables never disagree.
- **`np.minimum`/`np.fmax`/`np.fmin` choices in `preprocessing.py`/`lifecycle_cleaning.py` are
  deliberate** (NaN-propagating vs. NaN-ignoring, chosen per-field) — don't casually swap one for
  the other without checking the surrounding comment.
- **`Ship_To_Code=3003328` orders have their `Customer_Request_Date` brought forward one month**
  (`_bring_forward_ship_to_request_date` in `preprocessing.py`) — business-confirmed data-entry
  quirk, STJ-only. No SPL order has ever carried this code, so the fix isn't plant-gated in code;
  instead `load_io` logs `plant` on every call. If that log ever shows a nonzero count for
  plant=2100, the rule is firing somewhere it shouldn't — investigate before trusting the output.
  The negative-lead-time drop runs **before** this shift. After it, a row the shift moved into a
  request month that had **already closed when the order was entered** (a negative lead time
  created by the bring-forward) is dropped, with its own WARNING (count + units). Business usually
  counts these orders, but they are a tiny share of STJ volume, they would rewrite a settled month's
  ActualOrder after it was written and leak into closed-month features (found by
  `tests/verify_truncation_real.py`), and the business dashboards read SalesResult, not
  ActualOrder - so only the model is affected. **Be aware: the model does not predict these
  orders**, which can be a sizeable share of a single code's volume. A shift that stays within or after the entry month is
  kept. OutstandingOrder applies the same bring-forward, so the dropped orders (their request month
  is before N+1) show there as outstanding until sold or cancelled.
- **STJ `SLH_Order='Y'` rows are kept as orders and excluded only from `OutstandingOrder`** -
  business-confirmed. Each month an "SLH yyyymm" file lists SLH orders already recognised as sales
  in the previous month. An SLH order isn't marked complete until it ships out of SLH, so it carries
  into the next month's incoming orders with status Shipped, and those rows get `SLH_Order='Y'`. The
  flag stops the *sale* being counted twice, but neither side needs it here: IO keeps one row per
  order line, and `SalesResult` comes from `HistoricalSales_LocalAndExport`, which already handles
  it. The row is real demand in its request month. Don't filter orders on the flag: it's set only
  after shipment (the first `'Y'` rows appear in `20260831.xlsx`), so the filter dropped orders that
  were already in `KnownOrder`. That is why Sep-2026 STJ `ActualOrder` came out below the logged
  28 Sep `KnownGrossOrder` for many codes, and never above it. `OutstandingOrder` still drops
  `'Y'` rows, because they have already shipped. If the carry-over rewrites the request month,
  that's the general request-date issue in §14, in either direction; `OutstandingOrder` is
  unaffected since `'Y'` rows are excluded.
- **STJ periods covered by both a converted Shimano GO file and a direct download use both, on
  purpose.** The primary source is the Shimano GO direct download in Looker schema (e.g. `20251231
  (20170101 - 20251231) - Direct Download Looker Schema Version.xlsx`, `20260828 (20260101 - 20260828).xlsx`).
  The raw Excel exports we were originally given contain extra orders that are not in the direct
  download, and those orders improve training. Nobody knows why they are missing (possibly
  cancellations), but they are treated as real demand. So the raw exports are converted to the Looker
  schema (`data/raw/incoming_orders/Raw_IO_Basic_Cleaning.ipynb`; currently `20170101 - 20251031` and
  `20251101 - 20260601 - Convert From Shimano GO to Looker Schema.xlsx`) and loaded alongside, with
  duplicates removed. This applies to any period where both kinds of file exist, not only 2017-2025.
  Details:
  - **Dedup key:** order line = Shimano_Order_Number + Seq + Seq_2 + MaterialCode (business-confirmed); the same order
    and Seq with a different material is a different order line.
  - **Which copy wins:** the upsert keeps the newest `FileDate`, which is the *start* date in the file name. So where
    a line is in both files, the direct-download version wins, and the converted file only supplies the lines the
    download lacks. As of 2026-10, `20170101 - 20251031` supplies 161k lines: 38-60% of 2021-2023 volume, 16% of 2020,
    18% of 2024, under 10% in other years. `20251101 - 20260601` has 21 lines left; newer direct downloads replaced the
    rest.
  - **Customer identity on converted lines:** there is no `Customer_Code` or customer group, because the raw GO customer
    code is a different code system with no cross-reference. `Customer_Short_Name_1` is the raw GO short name, which
    can differ from the Looker short name of the same customer (e.g. TANDM vs TIGER), so per-customer history for STJ
    2020-2024 can show one customer under two names.
  - SPL has no converted files; its history is yearly Looker-schema files.

## 14. Known limitations

- **Forecasts made during an event with an unknown end treat the months after it as "still going" (until the
  external-event rework).** `*_EXTERNAL_EVENT_END_KNOWN` in `.env` lists only events whose end was known when they
  started (the 2026 tariff: surcharge end set in advance). For the others (COVID, Douyin), every forecast made during
  the event flags all its targets as "still going", including months after the actual end. Measured on Douyin
  (2026-10-16): this did *not* push the after-months high on balance (0.7 WAPE better, as they were under-forecast),
  though it can for single codes. The after-event over-forecast seen at h4-h6 is the model carrying the event level
  forward, and it is the same with or without this flag.
  - *Not fixed by listing every event as end-known:* that was tried (train12, 2026-10-14) and reverted 2026-10-15.
    It makes the after-months look better only because the backtest is told the end date (hindsight), which live
    nobody knows.
  - *Live:* enter a new event without an end-known pair unless its end is genuinely announced.
  - Details: `notebooks/forecast_improvement/README.md` step 0c; the real fix is step 5 in `BACKLOG.md`.

- **⚠ Backtests assume old orders are never edited — unverified.** Every training panel is rebuilt
  from today's IO extract, so historical order-book features (`KnownOrder`, lag/momentum features,
  and the daily-refresh notebook's `known_order_as_of_day` etc.) see each order's *latest* `Qty`/
  `EntryDate`, not what the book showed at that origin or snapshot day. This only matters if IO
  edits **overwrite the existing row in place**. Edits recorded as new rows (an amendment line dated
  when the change was made, or cancel + re-enter) land in the book at the edit date, so `Adjustment`
  carries them and there is no issue. Cancellations are already in the safe form: they stay as gross
  rows flagged `IsCanceled`. If edits are in place:
  - **Backtest WAPE is optimistic.** Past rows already contain edits made after their cutoff — a
    mild feature leak that no fold design can remove.
  - **Models never learn what a later edit looks like, at any horizon.** Example: an order entered
    at T−6 for 100 is edited in place to 60 at T−2. Today's IO shows `EntryDate=T−6, Qty=60`, so
    training `KnownOrder` is 60 at every horizon and `Adjustment` is 0 throughout. Live at h=6, the
    book showed 100 and the true adjustment was −40. Moving from h6 to h5 doesn't reveal it either,
    since both come from the same final row. Live error will be higher than the backtest by that gap.

  Production itself is point-in-time: each monthly run's `KnownGrossOrder` is kept unchanged in
  `PredictionResultLog`. **To size the gap,** compare those rows against `KnownOrder` recomputed from
  the current book for the same (code, origin, target). Matching values mean there are no in-place
  edits (either none happen or they're stored as new rows) and this item can be closed. **If they do,** start archiving a daily order-book snapshot so later
  training can use true point-in-time features. The history before that can't be recovered.

  Notes for running the check:
  - The Log holds one as-of point per run (monthly runs → month ends only), not every day. A sample
    of logged origins is enough to size the gap; once daily runs are live, daily points accrue too.
  - Coverage starts when the Log started being written, and only for codes that were scored.
  - Key on `PlanModelCode`; for runs before the grain change use the Log's frozen `SalesModelCode`.
  - STJ rows logged before the SLH fix (2026-10-09, §13) left out SLH orders already flagged `'Y'`,
    so their `KnownGrossOrder` is too low for SLH-heavy codes. Today's book includes those orders,
    so the comparison shows a gap there that is not an in-place edit. Exclude those rows, or
    subtract the `'Y'` lines, before reading the gap.
  - Compare totals per (TargetYearMonth, Horizon). If the gap is a few percent, accept it and close
    this item - archiving a snapshot is only worth its cost if the gap is large.

- **⚠ Customer request dates can change after the month closes — frequency unknown.** IO overwrites
  the row, so a moved request date moves the line's `Qty` from one target month to another in
  every panel rebuilt afterwards, including months already settled. Seen once: SPL order `1259051`
  (Seq 1 `PN0HT00`, Seq 2 `PN5W8C0`, 5,000 each, entered 2026-08-07) moved from 202609 to
  2027-01-04 in `20261009.xlsx`, after Sep-2026 had settled.
  - **Production is unaffected.** `main.py` writes `ActualOrder` once, for month N, at the first run
    of N+1, and later runs never forecast a month before their origin. The table therefore holds
    the actual as it stood when the forecast was scored, which is the intended comparison. Only a
    manual `collate_actual_order` rerun overwrites it with today's value.
  - **Training is affected.** Panels always use the latest request date, so a past month's
    `ActualOrder`/`Adjustment` (the target) and the `KnownOrder` features for both months reflect
    the move. Like the in-place edits above, the model never sees the original request month.
  - **To size it,** compare `ActualOrder` table rows (frozen at month start) against `ActualOrder`
    recomputed from today's book for the same (PlanModelCode, TargetYearMonth). The gap is exactly
    what later edits did to settled months. No fix for now; the daily snapshot above would cover
    it too.

- **A re-coded product's whole history moves to its newest PlanModelCode.**
  `resolve_plan_model_codes` gives a ProductCode with several codes its latest order's, so training
  rows before the re-coding sit under a code production didn't yet use. Mild backtest optimism for
  re-coded products only - not target leakage.

- **IntraMonth working-day features count Mon–Fri only.** `CutoffWorkingDays*`,
  `CutoffMonthProgress` and `ArrivalRate_*` (`intramonth_features.py`) use no holiday or CNY
  calendar, so CNY weeks count as working days. `ArrivalRate_7d` also divides by a constant 5.

- **Future improvement: `_attach_same_day_visibility` repeats its book queries.** It crosses every
  day-grain as-of point (code, origin, horizon, cutoff date) with up to `lookback_months` prior
  months, about 26M candidate rows over 6.1M as-of points on STJ. The counts are expected, but most
  lookups repeat: `Denom` depends only on (code, M′, OriginIdx), so it is recomputed for every
  horizon × cutoff day, and `Numer` depends only on (code, M′, Horizon, DayIndex), so it repeats for
  each origin that looks back at the same M′. Running each query on its unique keys and merging the
  results back would cut runtime and memory. Only worth doing if this step becomes slow or close to
  running out of memory.

- **Boundary: purely external events can't be forecast before they become known.** A shock nobody saw
  coming (a tariff announced overnight, COVID, a viral livestream) leaves no trace in the order data or any
  feature until it happens, so the far horizons will miss the start of such an event. That is a data limit, not a
  model fault, and only outside information dated by when it became known (announcements, news) could move it.
  Decisions taken in response are different: a price rise decided today for a future date is known from the
  decision date, so it can be a feature from then on, even though the change itself is still ahead.

- **Future improvement: external drivers are barely modelled.**
  - **Events and campaigns.** The only event inputs are hand-specified major-event windows
    (`*_EXTERNAL_EVENT_RANGES` in `.env`, giving `HasExternalEvent` / `LagEventCount*`) plus COVID
    phase. Internal (Shimano) and external (customer / brand) marketing campaigns, promotions and
    launches are not considered at all. Two things are missing: a source for this information, and a
    way to backtest it fairly. A backtest needs to know when each campaign became known, not just when
    it ran, or the feature leaks.
  - **Price.** One test of a unit-price feature barely improved accuracy. That is probably because
    it was not explored enough, not because price doesn't matter: there should be a correlation, even
    a strong one. The trial used the order lines' own `UnitPrice` per currency. That is the realised
    price on orders, which may be the wrong signal. It is not part of the live feature set. Open questions:
    - **Source:** the realised order-line price, or the announced / list price?
    - **Currency:** customers buy in several currencies, so decide whether to convert to one, and how
      to include FX movement itself.
    - **Timing:** when a price change becomes known to customers, not just when it takes effect.
      Announced increases such as tariffs and oil surcharges make customers bring procurement forward
      ahead of the effective date, or hold it until after. Capturing that needs announcement dates
      (e.g. "months until an announced change" and its size), anchored so a backtest only sees
      announcements made before each as-of date.

- **Future decision: training-panel size vs. server memory.** The day-grain training panel grows
  every month: each settled month adds about (live codes × 6 horizons × ~30 cutoff days) rows, and
  new codes widen it further. As of 2026-10 it is already large: STJ is 1.46M rows × 157 columns
  (~200 MB parquet, several times that in memory), and SPL about 6M rows (~0.9 GB parquet). The
  monthly retrain loads the full panel and fits the standard and quantile models for every horizon
  inside the 11:00 scheduled run (§2). That server also hosts other applications, so a growing
  memory peak can affect them. Peak memory of a retrain has not been measured yet; measure it before
  choosing among these levers:
  - **Feature pruning.** Drop features that add little, e.g. by SHAP importance. MLflow has "SHAP
    prune" runs from the two-stage daily-refresh experiment only; nothing similar has been tried on
    the production day-grain models.
  - **Fixed vs. expanding training window.** Today the window keeps expanding from the dataset floor.
    A fixed window (e.g. the last N years) caps the panel size, but may lose old seasonality and
    COVID-era signal. Decide by backtest, comparing a few N against expanding.
  - **Fewer cutoff days in training.** Training on every k-th cutoff day instead of all ~30 (the
    two-stage experiment used k = 4) shrinks training rows roughly k-fold, while still scoring every
    day.
  - **Scheduling.** If memory stays high, move the retrain-month run outside busy hours, or onto a
    separate machine.

- **UI assumptions (the UI is maintained by another team, not in this repo).** The dashboard
  hardcodes two things based on this pipeline's output, so a change here, or in the source data,
  can break it silently:
  - **The PIC adjustment is the `manual_6` method.** The UI treats `PredictionResult` rows with
    `Method = 'manual_6'` as the PIC (manual) adjustment. The `6` comes from `DAILY_MANUAL_WINDOW`
    in `.env`. Changing that window changes the `Method` value (e.g. `manual_3`), and the UI will no
    longer find the PIC rows. Tell the UI team before changing it.
  - **STJ shows `PlanModelText`, not `PlanModelCode`.** The STJ user asked for the text only.
    Naming is confusing across systems: our `PlanModelCode` is SIC demand plan's `Plan Model ID`,
    and our `PlanModelText` is SIC's `Plan Model Code`. One text can belong to more than one code.
    The known case is `FDTY300`, which has two `PlanModelCode`s; the UI merges them, hardcoded at
    the user's request. The user says new codes will be created with distinct texts, but nothing
    in our data enforces that. **If users report a duplicated Plan Model Text in the UI, first check
    whether it maps to more than one `PlanModelCode`** (`ProductLifecycle` has both columns). If
    it does, it is a new case like `FDTY300`, not a pipeline bug. SPL has no such issue yet, because
    what SPL displays has not been confirmed with the user; it may appear once it is.

- **Open question for users: should production requirement leave out orders already fulfilled?**
  `PredictedNetOrder` is total demand for the target month, and a line stays in it after it ships
  (only a cancellation removes it). So `PredictedNetOrder + OutstandingOrder` is the month's demand
  plus unshipped backlog. For request months N+1 and later, part of the book can already be
  fulfilled: in `20261009.xlsx`, ~350k units requested for 202610 (mostly `Shipped(SLH)`) and ~107k
  for 202611 had already shipped. Confirm with users whether these should be subtracted
  (`PredictedNetOrder − AlreadyFulfilled + OutstandingOrder`, floored at 0 per code) or the current
  figure is what they want.
  - **If subtracted, compute it from current order status, outside the model.** Read it off the full
    book (not today's file only: a line that shipped earlier stays in an older file) for request
    months >= N+1. It needs no as-of history, since only today's figure is displayed.
  - **Either way, keep it out of the demand forecast.** Shipping changes what is left to produce,
    not demand. Removing shipped lines from orders, KnownOrder or the label would use information
    only known after the fact - the same leak as the old `SLH_Order` filter (§13) - and make actuals
    shrink after they are written.
  - **Also to agree with users:** which `Order_Status_Text` values count as fulfilled (e.g.
    `Shipped(SLH)`, `Sold`, `Shipped`), and whether the same rule then replaces `SLH_Order='Y'` in
    `OutstandingOrder`. The two must use one rule, or a line is "done" in one figure and not in the
    other. STJ currently keeps `Sold`/`Shipped` lines with `SLH_Order='N'` as outstanding
    (business-confirmed), which a status-based rule would change.
