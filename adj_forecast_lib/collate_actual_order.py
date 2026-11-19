"""Collates settled ActualOrder per (PartCode, SalesModelCode, PlanModelCode, TargetYYYYMM)."""
import argparse
import logging
import os

import numpy as np
import pandas as pd

from . import db_writer
from .config import month_idx_to_yyyymm, plant_name, to_month_idx
from .logging_setup import configure_logging
from .preprocessing import build_order_book_context, load_sales_result

log = logging.getLogger(__name__)


def collate_actual_order(ctx: dict, plant: int, month_idx: int = None, max_settled: int = None) -> pd.DataFrame:
    """month_idx=None returns full settled history up to `max_settled`; an int returns that month only."""
    # Lifecycle widened to the settled demand span, so every real order is shown. From launch_by_code,
    # not first_actual_order_by_code - that one moves with extend_to_orders. No launch -> floor.
    floor = ctx['dataset_floor_idx']
    start_by_code = np.fmax(np.fmin(ctx['launch_by_code'].fillna(floor), ctx['first_demand_by_code']), floor)
    end_by_code = ctx['effective_ranges'].set_index('PlanModelCode')['EndMonthIdx']
    max_settled_idx = int(to_month_idx(max_settled)) if max_settled is not None else ctx['max_settled_idx']

    # No resolved EndMonthIdx (still active / TFGMaterial has no end date) runs to max_settled_idx.
    end_by_code = np.fmin(np.fmax(end_by_code.reindex(start_by_code.index).fillna(max_settled_idx),
        ctx['last_demand_by_code']), max_settled_idx)

    frames = []
    for code, start in start_by_code.items():
        end = end_by_code[code]
        if pd.isna(start) or pd.isna(end) or start > end:
            continue  # no valid settled window for this code (e.g. resolved end predates the dataset floor)
        if month_idx is None:
            target_idx = np.arange(int(start), int(end) + 1)
        elif start <= month_idx <= end:
            target_idx = np.array([month_idx])
        else:
            continue  # this code's window doesn't cover month_idx (e.g. already ended, or not launched yet)
        frames.append(pd.DataFrame({'PlanModelCode': code, 'TargetIdx': target_idx}))
    if not frames:
        # pd.concat([]) raises "No objects to concatenate". No settled month is not an error:
        # there is simply nothing to collate, and write_actual_order skips an empty frame.
        log.warning("collate_actual_order: no PlanModelCode has a settled window%s - nothing to "
            "collate.", '' if month_idx is None
            else f" covering {int(month_idx_to_yyyymm([month_idx])[0])}")
        return pd.DataFrame(columns=['PartCode', 'SalesModelCode', 'PlanModelCode', 'TargetYYYYMM',
                                     'ActualGrossOrder', 'ActualNetOrder', 'SalesResult'])
    grid = pd.concat(frames, ignore_index=True)

    grid = grid.merge(
        ctx['actual_order'].rename(columns={'RequestMonthIdx': 'TargetIdx', 'ActualOrder': 'ActualGrossOrder'}),
        on=['PlanModelCode', 'TargetIdx'], how='left',
    )
    grid = grid.merge(
        ctx['actual_order_net'].rename(columns={'RequestMonthIdx': 'TargetIdx'}),
        on=['PlanModelCode', 'TargetIdx'], how='left',
    )
    grid['ActualGrossOrder'] = grid['ActualGrossOrder'].fillna(0.0)
    grid['ActualNetOrder'] = grid['ActualNetOrder'].fillna(0.0)

    sales_result = load_sales_result(plant, ctx['product_ranges'].set_index('ProductCode')['PlanModelCode'])
    grid = grid.merge(sales_result, on=['PlanModelCode', 'TargetIdx'], how='left')
    # 0 only within HistoricalSales_LocalAndExport's loaded Period range - outside it (not yet
    # loaded, or before its history starts) sales are unknown, left NaN -> NULL.
    covered = grid['TargetIdx'].between(sales_result['TargetIdx'].min(), sales_result['TargetIdx'].max())
    grid.loc[covered, 'SalesResult'] = grid.loc[covered, 'SalesResult'].fillna(0.0)
    grid['TargetYYYYMM'] = month_idx_to_yyyymm(grid['TargetIdx'].to_numpy())
    grid = grid.join(ctx['code_map'], on='PlanModelCode')

    return grid[['PartCode', 'SalesModelCode', 'PlanModelCode', 'TargetYYYYMM', 'ActualGrossOrder', 'ActualNetOrder',
        'SalesResult']].sort_values(['PlanModelCode', 'TargetYYYYMM']).reset_index(drop=True)


def _parse_args():
    p = argparse.ArgumentParser(description="Collate settled ActualOrder per (PlanModelCode, TargetYYYYMM).")
    p.add_argument('--plant', type=int, required=True, help="Plant code (e.g. 2100=SPL, 3300=STJ).")
    p.add_argument('--max-settled', type=int, default=None,
        help="YYYYMM - caps the full-history output as of an earlier settled boundary "
             "(defaults to the latest completed month). Ignored if --month is given.")
    p.add_argument('--month', type=int, default=None,
        help="YYYYMM - restrict to just this one month instead of the full settled history.")
    p.add_argument('--write-to-db', action='store_true',
        help="Also upsert the result into DemandForecast.ActualOrder - normally this CLI only "
             "saves a local parquet (main.py is the only regular writer). Use for a deliberate "
             "one-time backfill/correction.")
    return p.parse_args()


def main():
    args = _parse_args()
    configure_logging('collate_actual_order', args.plant)

    # Actuals are settled-month quantities, identical at either grain - month skips the day books.
    ctx = build_order_book_context(args.plant, cutoff_grain='month')
    month_idx = int(to_month_idx(args.month)) if args.month is not None else None
    result = collate_actual_order(ctx, args.plant, month_idx=month_idx, max_settled=args.max_settled)

    if args.month is not None:
        origin_yyyymm = args.month  # single-month mode: the filename names that one month
    else:
        max_settled_idx = int(to_month_idx(args.max_settled)) if args.max_settled is not None else ctx['max_settled_idx']
        origin_yyyymm = int(month_idx_to_yyyymm(np.array([max_settled_idx]))[0])

    out_path = f'results/{plant_name(args.plant)}/actual_order_upto{origin_yyyymm}.parquet'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.to_parquet(out_path, index=False)
    log.info("Saved %d row(s) to %s", len(result), out_path)
    log.info("%d PlanModelCode(s), TargetYYYYMM range %s-%s",
        result['PlanModelCode'].nunique(), result['TargetYYYYMM'].min(), result['TargetYYYYMM'].max())

    if args.write_to_db:
        n = db_writer.write_actual_order(result, args.plant)
        log.info("write_to_db: upserted %d row(s) into ActualOrder for plant %s", n, args.plant)


if __name__ == '__main__':
    main()
