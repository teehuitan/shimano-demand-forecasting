"""Daily pipeline orchestrator, configured by DAILY_* env keys: retrain on a new settled month,
predict daily at day grain, refresh OutstandingOrder only alongside a successful forecast.
"""
import logging
import os
import time

import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)  # before adj_forecast_lib imports - db_writer/preprocessing build
# their engines from os.environ at their own import time.

from adj_forecast_lib import db_writer
from adj_forecast_lib.config import CUTOFF_GRAINS, month_idx_to_yyyymm, PLANT_INCLUDES_MANUAL_METHOD
from adj_forecast_lib.logging_setup import configure_logging, prune_old_logs
from adj_forecast_lib.model_registry import deployed_state, resolve_deployment_grain
from adj_forecast_lib.preprocessing import (
    _settled_month_for_file_date, build_order_book_context, ctx_cutoff_grain,
    cutoff_date_from_file_date, latest_source_file_date, preprocess_for_inference,
)
from adj_forecast_lib.collate_actual_order import collate_actual_order
from adj_forecast_lib.collate_outstanding_order import _load_orders_for_file, collate_outstanding_order
from adj_forecast_lib.predict import score_manual, score_models
from adj_forecast_lib.train import DEFAULT_COVID_WEIGHT, train_deployment

log = logging.getLogger(__name__)


class DailyRunError(RuntimeError):
    """Raised after all runnable phases finish; `failures` holds (phase, exception) pairs."""

    def __init__(self, plant, failures):
        self.failures = list(failures)
        super().__init__(f"daily run for plant {plant} failed: " + '; '.join(
            f"{phase} - {type(e).__name__}: {e}" for phase, e in self.failures))


def _elapsed(t0: float) -> str:
    """Seconds since time.perf_counter() value `t0`, for the run summary line."""
    return f'{time.perf_counter() - t0:.0f}s'


def _is_stale(new_settled_idx, state: dict) -> bool:
    """True when the file's settled month is newer than the registry's trained_through."""
    if new_settled_idx is None:
        return False
    if state is None or state['trained_through'] is None:
        return True  # nothing deployed under this label yet
    return new_settled_idx > state['trained_through']


def _already_done(check, plant, *args) -> bool:
    """`check(plant, *args)`; a failed DB read counts as not done."""
    try:
        return check(plant, *args)
    except Exception:
        log.warning("%s FAILED for plant %s - treating it as not done.", check.__name__, plant,
            exc_info=True)
        return False


def _build_run_ctx(plant, *, cutoff_grain, settled_idx):
    """The ctx this run trains on, pinned to run_daily's `settled_idx`; prediction reuses it."""
    max_settled_target = None if settled_idx is None else int(month_idx_to_yyyymm([settled_idx])[0])
    return build_order_book_context(plant, max_settled_target=max_settled_target,
        cutoff_grain=cutoff_grain)


def _train_phase(plant, label, covid_weight, val_months, ctx, *, cutoff_grain):
    """Trains the full deployment at one grain on run_daily's `ctx`."""
    train_deployment(plant, label, cutoff_grain=cutoff_grain, covid_weight=covid_weight,
        val_months=val_months, ctx=ctx)


def _prediction_phase(plant, label, horizons, min_origin_months_since_launch, quantile_partcodes,
        manual_window, *, cutoff_grain, file_date, trained_through=None, ctx=None):
    """Scores model (+ manual method) at the origin `file_date` implies, then writes the results.
    All checks run before the first write, so a refused run writes nothing. `ctx` is this run's
    retrain ctx; None builds a fresh one."""
    if file_date is None:
        raise ValueError(f"no parseable source file for plant {plant} - can't pin the origin or "
                         "the cutoff date, so nothing is scored.")
    settled_idx = _settled_month_for_file_date(file_date)
    origin_yyyymm = int(month_idx_to_yyyymm([settled_idx])[0])
    cutoff_date = cutoff_date_from_file_date(file_date) if cutoff_grain == 'day' else None
    log.info("Scoring plant %s at origin %s (%s grain%s).", plant, origin_yyyymm, cutoff_grain,
        f', as of {cutoff_date.date()}' if cutoff_date is not None else '')

    horizons = list(horizons)
    skeleton, ctx = preprocess_for_inference(plant, origin_yyyymm, horizons,
        min_origin_months_since_launch=min_origin_months_since_launch,
        cutoff_grain=cutoff_grain, cutoff_date=cutoff_date, ctx=ctx)
    _check_origin_not_behind(plant, ctx['max_settled_idx'], trained_through)
    results = [score_models(skeleton, horizons, label, plant, quantile_partcodes)]
    if PLANT_INCLUDES_MANUAL_METHOD[plant]:
        results.append(score_manual(skeleton, manual_window))
    result = pd.concat(results, ignore_index=True)
    _check_single_asof(result, origin_yyyymm)

    lifecycle_df = ctx['effective_ranges'][['PartCode', 'SalesModelCode', 'PlanModelCode', 'PlanModelText',
        'StartMonthIdx', 'EndMonthIdx']].copy()
    lifecycle_df['StartYearMonth'] = month_idx_to_yyyymm(lifecycle_df['StartMonthIdx'].to_numpy())
    lifecycle_df['EndYearMonth'] = month_idx_to_yyyymm(lifecycle_df['EndMonthIdx'].to_numpy())
    origin_date = _origin_date_for(ctx['max_settled_idx'], cutoff_grain, cutoff_date)
    # New month = no ActualOrder for N yet. Called directly, not via _already_done: a failed read
    # must fail this phase, not wipe the month as if it were new.
    is_new_month = not db_writer.has_actual_for_month(plant, origin_yyyymm)
    # Built before the first write, so a failure here leaves every table as it was.
    actual_df = collate_actual_order(ctx, plant, month_idx=ctx['max_settled_idx']) if is_new_month else None

    db_writer.write_product_lifecycle(lifecycle_df, plant)
    if is_new_month:
        log.info("New month: writing ActualOrder + PredictionResult, clearing last month (one "
            "transaction): plant=%s origin=%s", plant, origin_date.date())
        db_writer.month_start_write_clear(actual_df, result, plant, origin_date)
    else:
        log.info("Writing PredictionResult: plant=%s origin=%s", plant, origin_date.date())
        db_writer.write_prediction_result(result, plant, origin_date)
    return ctx


def _check_origin_not_behind(plant, origin_idx, trained_through) -> None:
    """Refuses to score an origin earlier than the deployment's trained_through."""
    if trained_through is not None and origin_idx < trained_through:
        origin, through = (int(m) for m in month_idx_to_yyyymm([origin_idx, trained_through]))
        raise ValueError(
            f"refusing to score plant {plant} at origin {origin}: the deployment is trained "
            f"through {through}, so the settled month has gone BACKWARDS - the newest source file "
            "was removed, or an older one re-ingested. Scoring would replace the current forecast "
            "with an older one.")


def _check_single_asof(result: pd.DataFrame, settled_yyyymm: int) -> None:
    """Refuses an empty result or rows scored at more than one origin/cutoff day."""
    if result.empty:
        raise ValueError(
            f"scoring produced no rows at all for origin {settled_yyyymm} - no PlanModelCode was "
            "eligible, or every one was filtered out (see the inference skeleton's log lines "
            "above). Nothing is written; the previous forecast stays.")
    origins = sorted(pd.unique(result['OriginYYYYMM']).tolist())
    days = sorted(pd.unique(result['CutoffDayIndex']).tolist())
    if origins != [settled_yyyymm] or len(days) != 1:
        raise ValueError(
            f"prediction rows span origins {origins} / cutoff days {days}, but this run stamps "
            f"one OriginDate for origin {settled_yyyymm} on all of them - refusing to write rows "
            "scored at a different as-of point.")


def _origin_date_for(max_settled_idx, cutoff_grain, cutoff_date):
    """OriginDate: settled month-end at month grain, the cutoff date at day grain."""
    if cutoff_grain == 'day' and cutoff_date is not None:
        return pd.Timestamp(cutoff_date)
    yyyymm = int(month_idx_to_yyyymm([max_settled_idx])[0])
    return pd.Timestamp(year=yyyymm // 100, month=yyyymm % 100, day=1) + pd.offsets.MonthEnd(0)


def _outstanding_order_phase(plant: int, file_date, source_file, settled_month_idx):
    """Writes today's OutstandingOrder snapshot for the forecast's settled month."""
    if file_date is None:
        log.warning("No parseable source file for plant %s - skipping today's OutstandingOrder run.",
            plant)
        return
    orders = _load_orders_for_file(plant, source_file)
    result = collate_outstanding_order(orders, file_date, source_file, plant,
        settled_month_idx=settled_month_idx)
    db_writer.write_outstanding_order(result, plant, source_file)


def _describe_deployment(state) -> str:
    if state is None or state['cutoff_grain'] is None or state['trained_through'] is None:
        return 'there is none that can be scored from, so nothing is scored today'
    return (f"it stays in use ({state['cutoff_grain']} grain, trained through "
            f"{int(month_idx_to_yyyymm([state['trained_through']])[0])})")


def run_daily(plant, label, covid_weight=DEFAULT_COVID_WEIGHT, val_months=6, horizons=range(1, 7),
        min_origin_months_since_launch=3, quantile_partcodes=None, manual_window=6, cutoff_grain=None,
        force_run=False, force_retrain=False):
    """Runs retrain, prediction and OutstandingOrder for one plant, once per new source file.
    `force_run` reruns an already-processed file; `force_retrain` also retrains without a new
    settled month. `cutoff_grain=None` keeps the deployed grain."""
    file_date, source_file = latest_source_file_date(plant)     # read ONCE for the whole run
    new_settled_idx = _settled_month_for_file_date(file_date)
    log.info("Daily run start: plant=%s source_file=%s", plant, source_file)
    if (file_date is not None and not (force_run or force_retrain)
            and _already_done(db_writer.has_outstanding_for_file, plant, source_file)):
        log.info("Source file %s for plant %s is already processed - nothing to do (set "
            "DAILY_FORCE_RUN=1 to rerun it).", source_file, plant)
        return

    explicit = cutoff_grain is not None
    state = deployed_state(plant=plant, label=label, allow_mixed=explicit)
    grain = resolve_deployment_grain(state, cutoff_grain)
    deployed_grain = state['cutoff_grain'] if state else None
    grain_change = state is not None and deployed_grain != grain
    needs_retrain = _is_stale(new_settled_idx, state)
    failures = []
    # phase -> "OK (12s)" / "FAILED (3s) ..." / "skipped (...)", logged as one summary line below.
    status = {'retrain': 'skipped (no new settled month)'}
    retrained = False
    ctx = None

    if needs_retrain or grain_change or force_retrain:
        if grain_change:
            log.warning("Deployment %r for plant %s is at %s grain but %r was requested - "
                "retraining at %r now.", label, plant, deployed_grain or state['labels'], grain, grain)
        else:
            log.info("%s for plant %s - retraining at %r grain.", 'New settled month detected'
                if needs_retrain else 'DAILY_FORCE_RETRAIN set', plant, grain)
        t0 = time.perf_counter()
        try:
            # Assigned before training, so a failed fit still leaves ctx for prediction to reuse.
            ctx = _build_run_ctx(plant, cutoff_grain=grain, settled_idx=new_settled_idx)
            _train_phase(plant, label, covid_weight, val_months, ctx, cutoff_grain=grain)
        except Exception as e:
            log.exception("Retraining %r for plant %s at %r grain FAILED. The previous deployment "
                "is untouched: %s. The next run retries.", label, plant, grain, _describe_deployment(state))
            failures.append(('retrain', e))
            status['retrain'] = (f"FAILED ({_elapsed(t0)}) - kept previous deployment trained through "
                f"{state['trained_through'] if state else 'n/a'}")
            # Score exactly as if no retrain had been due: the incumbent, at its own grain.
            grain, grain_change = deployed_grain, False
            if ctx is not None and ctx_cutoff_grain(ctx) != grain:
                ctx = None  # a failed grain change - prediction rebuilds at the incumbent's grain
        else:
            status['retrain'] = f'OK ({_elapsed(t0)})'
            state = deployed_state(plant=plant, label=label)
            # Post-condition: the deployment left behind must be at the resolved grain and trained
            # through this run's settled month, or nothing is scored from it.
            if (state is None or state['cutoff_grain'] != grain
                    or (new_settled_idx is not None and state['trained_through'] != new_settled_idx)):
                failures.append(('retrain', RuntimeError(
                    f"retrained deployment {label!r} for plant {plant} at {grain!r} grain through "
                    f"settled month {new_settled_idx}, but the registry now reports "
                    f"{state and (state['labels'], state['cutoff_grain'], state['trained_through'])} "
                    "- refusing to score from it.")))
                log.error("%s", failures[-1][1])
                status['retrain'] = f'FAILED ({_elapsed(t0)}) - deployment left at the wrong grain or month'
                state = None
            else:
                retrained = True

    forecast_current = False
    if state is None or grain is None:
        log.warning("No deployment under %r for plant %s can be scored from (%s) - skipping "
            "prediction.", label, plant, 'see the retrain failure above' if failures
            else 'nothing is deployed, and there is no settled month to train on yet')
        status['prediction'] = 'skipped (no usable deployment)'
    elif (grain == 'month' and not (retrained or force_run) and new_settled_idx is not None
            and _already_done(db_writer.has_prediction_at, plant,
                              _origin_date_for(new_settled_idx, 'month', None))):
        # Month grain scores as of the month end, so a new file mid-month gives identical numbers.
        log.info("Plant %s's month-grain forecast for this settled month is already written - "
            "skipping prediction.", plant)
        status['prediction'] = 'skipped (month grain, already predicted)'
        forecast_current = True
    else:
        t0 = time.perf_counter()
        try:
            ctx = _prediction_phase(plant, label, horizons, min_origin_months_since_launch,
                quantile_partcodes, manual_window, cutoff_grain=grain, file_date=file_date,
                trained_through=state['trained_through'], ctx=ctx)
        except Exception as e:
            log.exception("Scoring plant %s FAILED - PredictionResult keeps the previous forecast, "
                "and OutstandingOrder keeps the snapshot that matches it.", plant)
            failures.append(('prediction', e))
            status['prediction'] = f'FAILED ({_elapsed(t0)}) - previous forecast kept'
        else:
            status['prediction'] = f'OK ({_elapsed(t0)})'
            forecast_current = True

    # OutstandingOrder must stay aligned with the forecast it sits beside: refreshed only when this
    # run scored, or the forecast for this month is already current. Writing it marks the file done.
    if not forecast_current:
        log.warning("Skipping OutstandingOrder for plant %s - prediction did not succeed, so the "
            "previous snapshot is kept to stay aligned with the previous forecast.", plant)
        status['OutstandingOrder'] = 'skipped (prediction did not succeed)'
    else:
        t0 = time.perf_counter()
        try:
            _outstanding_order_phase(plant, file_date, source_file,
                settled_month_idx=ctx['max_settled_idx'] if ctx else new_settled_idx)
        except Exception as e:
            log.exception("Refreshing OutstandingOrder for plant %s FAILED - the previous snapshot stays.",
                plant)
            failures.append(('OutstandingOrder', e))
            status['OutstandingOrder'] = f'FAILED ({_elapsed(t0)}) - previous snapshot kept'
        else:
            status['OutstandingOrder'] = f'OK ({_elapsed(t0)})'

    log.log(logging.WARNING if failures else logging.INFO, "Daily run summary, plant %s: %s",
        plant, ' | '.join(f'{phase}={s}' for phase, s in status.items()))
    if failures:
        raise DailyRunError(plant, failures) from failures[0][1]


def main():
    """Runs run_daily for each plant in DAILY_PLANTS; one plant's failure doesn't stop the rest."""
    plants = [int(p) for p in os.environ['DAILY_PLANTS'].split(',') if p.strip()]
    label = os.environ['DAILY_LABEL']
    covid_weight = float(os.environ['DAILY_COVID_WEIGHT'])
    val_months = int(os.environ['DAILY_VAL_MONTHS'])
    horizons = [int(h) for h in os.environ['DAILY_HORIZONS'].split(',') if h.strip()]
    min_origin_months_since_launch = int(os.environ['DAILY_MIN_ORIGIN_MONTHS_SINCE_LAUNCH'])
    manual_window = int(os.environ['DAILY_MANUAL_WINDOW'])
    cutoff_grain = os.environ.get('DAILY_CUTOFF_GRAIN', '').strip() or None
    if cutoff_grain is not None and cutoff_grain not in CUTOFF_GRAINS:
        raise ValueError(f"DAILY_CUTOFF_GRAIN={cutoff_grain!r} - expected one of {list(CUTOFF_GRAINS)} or empty")
    # Manual reruns only - leave unset for the scheduled run.
    force_run = os.environ.get('DAILY_FORCE_RUN', '').strip() == '1'
    force_retrain = os.environ.get('DAILY_FORCE_RETRAIN', '').strip() == '1'

    failed = []
    for plant in plants:
        configure_logging('daily', plant)
        prune_old_logs(plant)
        try:
            run_daily(plant, label, covid_weight=covid_weight, val_months=val_months,
                horizons=horizons, min_origin_months_since_launch=min_origin_months_since_launch,
                manual_window=manual_window, cutoff_grain=cutoff_grain, force_run=force_run,
                force_retrain=force_retrain)
        except Exception:
            log.exception("run_daily FAILED: plant=%s - continuing with remaining plants.", plant)
            failed.append(plant)
    # Raised only after every plant ran, so the scheduler sees a non-zero exit.
    if failed:
        raise SystemExit(f"daily run failed for plant(s) {failed} - see the log above.")


if __name__ == '__main__':
    main()
