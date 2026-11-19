"""Real-data truncation-invariance check (manual - needs Fabric; NOT part of run_all.py).

verify_truncation_invariance.py proves the code has no look-ahead on synthetic data. This runs the
same rule on the REAL order feed, for every plant: build its day-grain training panel from today's orders, then
again from the orders as they looked at T (entered <= T, cancellations known after T reverted), and
every feature on every row with as-of <= T must be identical.

Read-only: loads orders, TFGMaterial and ProductSuccessor once, writes nothing to any database or
to MLflow. Differences go to a CSV in the review folder.

    python tests/verify_truncation_real.py [--plants 3300 2100] [--dates 2025-06-30 2026-03-14 ...]

What it does and does not prove:
  - Both builds use TODAY's row for each order (the source is an upsert, one row per order line),
    so an order edited in place after T looks the same in both. Restatement is out of scope.
  - TFGMaterial and ProductSuccessor are today's snapshot in both builds (ProductSuccessor is
    static for now), so a launch date or pair recorded after T is not tested.
  - resolve_plan_model_codes gives a ProductCode its LATEST order's PlanModelCode, so a product
    re-coded after T sits under a different code in the two builds. Those codes are reported
    separately: expected, not a code leak (README section 14).
  - Predecessor/successor flags follow .env (PLANT_INCLUDES_PRED_SUCC), as the production run does.
"""
import argparse
import logging
import os
import pathlib
import sys
import gc
import tempfile
import time

os.environ['LOG_DB_ENABLED'] = '0'            # never write run logs to the AI database
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from adj_forecast_lib import preprocessing as pp  # noqa: E402
from adj_forecast_lib.config import (  # noqa: E402
    CUTOFF_DATE_COL, PLANT_INCLUDES_PRED_SUCC, PLANT_NAMES, PLANT_SUBMODEL_PLAN_MODEL_OVERRIDES,
    month_idx_to_yyyymm,
)
from adj_forecast_lib.lifecycle_cleaning import resolve_plan_model_codes  # noqa: E402
from adj_forecast_lib.lifecycle_features import FORECAST_ELIGIBILITY_AGE_COL  # noqa: E402

LABELS = {'ActualOrder', 'Adjustment'}
KEY = ['PlanModelCode', 'TargetIdx', 'Horizon', CUTOFF_DATE_COL]
ATOL = 1e-9
OUT_DIR = pathlib.Path.home() / 'Downloads' / 'grand_review_2026-10-05'


def feed_as_of(orders: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """The feed a file dated asof+1 holds: nothing entered later, no cancellation known later."""
    o = orders[orders['EntryDate'] <= asof].copy()
    later = o['IsCanceled'] & (o['CancelKnownDate'] > asof)
    o.loc[later, 'IsCanceled'] = False
    o.loc[later, 'CancelMonthIdx'] = np.nan
    o.loc[later, ['CancelKnownDate', 'CancelDate']] = pd.NaT
    return o


def build(plant, orders, tfg, file_date, max_settled_target=None):
    """The real training panel, with only the three Fabric reads replaced by the frames given."""
    saved = {n: getattr(pp, n) for n in ('load_io', 'load_tfg_material', 'latest_source_file_date')}
    pp.load_io = lambda p, exclude_partcodes=None: orders.copy()
    pp.load_tfg_material = lambda: tfg.copy()
    pp.latest_source_file_date = lambda p: (file_date, file_date.strftime('%Y%m%d') + '.xlsx')
    try:
        panel, _f, feature_cols, ctx = pp.preprocess_for_training(
            plant, max_settled_target=max_settled_target, cutoff_grain='day')
    finally:
        for n, f in saved.items():
            setattr(pp, n, f)
    return panel, feature_cols, ctx


def recoded_codes(plant, orders, tfg, asof) -> set:
    """PlanModelCodes whose products resolve differently with only the orders known at `asof`."""
    over = PLANT_SUBMODEL_PLAN_MODEL_OVERRIDES[plant]
    full, _ = resolve_plan_model_codes(orders, tfg, submodel_overrides=over)
    trunc, _ = resolve_plan_model_codes(feed_as_of(orders, asof), tfg, submodel_overrides=over)
    a = full.groupby('ProductCode')['PlanModelCode'].first()
    b = trunc.groupby('ProductCode')['PlanModelCode'].first()
    common = a.index.intersection(b.index)
    moved = common[(a[common] != b[common]).to_numpy()]
    return set(a[moved]) | set(b[moved])


def save_full(full, cols, path) -> list:
    """Writes the full panel's key + compared columns to `path` and returns its column list, so it
    can be freed: SPL's day panel is ~6M rows, and two in memory don't fit."""
    full = full[KEY + cols].assign(PlanModelCode=full['PlanModelCode'].astype(str))
    for c in cols:
        if str(full[c].dtype) == 'category':
            full[c] = full[c].astype(str)
    full.to_parquet(path, index=False)
    return list(full.columns)


def compare(full_path, full_cols, trunc, cols, asof):
    """Compares `trunc` with the saved full panel, rows with as-of <= `asof`, one column at a time."""
    rows = [(CUTOFF_DATE_COL, '<=', asof)]
    # Same dtypes on both sides, or a parquet round-trip (ns -> us dates) would match no rows at all.
    norm = lambda k: k.astype({'PlanModelCode': str, 'TargetIdx': 'int64', 'Horizon': 'int64',
                               CUTOFF_DATE_COL: 'datetime64[ns]'})
    a = pd.MultiIndex.from_frame(norm(pd.read_parquet(full_path, columns=KEY, filters=rows)))
    t = trunc[trunc[CUTOFF_DATE_COL] <= asof]
    b = pd.MultiIndex.from_frame(norm(t[KEY]))
    out = {'dup': (int(a.duplicated().sum()), int(b.duplicated().sum())),
           'only_full': a.difference(b), 'only_trunc': b.difference(a),
           # every saved (key + compared) column must exist in the truncated build too
           'cols': (sorted(set(full_cols) - set(trunc.columns)), [])}
    common = a.intersection(b)
    pos_a, pos_b = a.get_indexer(common), b.get_indexer(common)
    diffs, worst = [], 0.0
    for c in cols:
        x = pd.read_parquet(full_path, columns=[c], filters=rows)[c].iloc[pos_a].reset_index(drop=True)
        y = t[c].iloc[pos_b].reset_index(drop=True)
        if x.dtype.kind in 'fiub' and y.dtype.kind in 'fiub':
            xv, yv = x.to_numpy('float64'), y.to_numpy('float64')
            # equal (incl. +inf == +inf, a legacy code's eligibility age) or both NaN -> no gap
            gap = np.where((xv == yv) | (np.isnan(xv) & np.isnan(yv)), 0.0, np.abs(xv - yv))
            gap = np.where(np.isnan(gap), np.inf, gap)
            bad = gap > ATOL
            worst = max(worst, float(np.nanmax(np.where(np.isinf(gap), 0, gap))) if len(gap) else 0.0)
        else:
            bad = x.astype(str).to_numpy() != y.astype(str).to_numpy()
        if bad.any():
            diffs.append(pd.DataFrame({'column': c, 'full': x[bad].astype(str).to_numpy(),
                                       'truncated': y[bad].astype(str).to_numpy()},
                                      index=common[bad]).reset_index())
    out['n'] = len(common)
    out['worst'] = worst
    out['diffs'] = pd.concat(diffs, ignore_index=True) if diffs else pd.DataFrame()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--plants', type=int, nargs='+', default=sorted(PLANT_NAMES))
    p.add_argument('--dates', nargs='*', default=None,
        help='as-of dates T (YYYY-MM-DD). Default: day 0, a mid-month and a day 30 in the last year.')
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')
    tfg = pp.load_tfg_material()
    failed = {plant: run_plant(plant, tfg, args.dates) for plant in args.plants}
    print('\n' + '=' * 78)
    for plant, failures in failed.items():
        print(f'{plant} ({PLANT_NAMES[plant]}): ' + (f'{len(failures)} as-of date(s) FAILED: {failures}'
              if failures else 'ALL AS-OF DATES PASSED (outside re-coded codes)'))
    sys.exit(1 if any(failed.values()) else 0)


def run_plant(plant, tfg, date_args) -> list:
    """Runs every as-of date for one plant; returns the dates that failed."""
    print(f'\n#################### {plant} ({PLANT_NAMES[plant]}), pred/succ '
          f'{"on" if PLANT_INCLUDES_PRED_SUCC[plant] else "off"} (per .env) ####################')
    t0 = time.time()
    orders = pp.load_io(plant)
    file_date, source_file = pp.latest_source_file_date(plant)
    data_asof = pp.cutoff_date_from_file_date(file_date)
    print(f'plant {plant}: {len(orders):,} orders, data through {data_asof.date()} '
          f'({source_file}), loaded in {time.time() - t0:.0f}s')

    t0 = time.time()
    full, feature_cols, ctx = build(plant, orders, tfg, file_date)
    settled = int(month_idx_to_yyyymm([ctx['max_settled_idx']])[0])
    # Key columns are matched row by row, not compared - and must not appear twice in the saved frame.
    cols = [c for c in feature_cols if c not in LABELS and c not in KEY] + [FORECAST_ELIGIBILITY_AGE_COL]
    print(f'full day panel {full.shape}, settled {settled}, {len(cols)} columns compared '
          f'({time.time() - t0:.0f}s)')
    full_path = pathlib.Path(tempfile.gettempdir()) / f'real_truncation_full_{plant}.parquet'
    full_cols = save_full(full, cols, full_path)
    del full, ctx
    gc.collect()

    if date_args:
        dates = [pd.Timestamp(d) for d in date_args]
    else:
        last_settled_end = pd.Timestamp(f'{settled // 100}-{settled % 100:02d}-01') + pd.offsets.MonthEnd(0)
        dates = [last_settled_end - pd.offsets.MonthEnd(12),                        # day 0, a year ago
                 last_settled_end - pd.offsets.MonthEnd(6) + pd.Timedelta(days=14),  # mid-month
                 last_settled_end - pd.offsets.MonthEnd(2) + pd.Timedelta(days=30),  # day 30
                 last_settled_end]                                                  # latest day 0

    failures, frames = [], []
    for asof in dates:
        t0 = time.time()
        entered = int((orders['EntryDate'] == asof).sum())
        week_after = int(((orders['EntryDate'] > asof) & (orders['EntryDate'] <= asof + pd.Timedelta(days=7))).sum())
        trunc, _fc, _ctx = build(plant, feed_as_of(orders, asof), tfg, asof + pd.Timedelta(days=1),
                                 max_settled_target=settled)
        r = compare(full_path, full_cols, trunc, cols, asof)
        del trunc
        gc.collect()
        recoded = recoded_codes(plant, orders, tfg, asof)
        d = r['diffs']
        real = d[~d['PlanModelCode'].isin(recoded)] if len(d) else d
        print(f'\n-- as-of {asof.date()} ({asof.day_name()[:3]}): {entered} orders entered on T, '
              f'{week_after} in the week after; {r["n"]:,} rows compared ({time.time() - t0:.0f}s)')
        same_features = list(_fc) == list(feature_cols)
        print(f'   keys unique: {r["dup"] == (0, 0)} {r["dup"]}; same feature list: {same_features}; '
              f'compared columns all present: {not r["cols"][0]} {r["cols"][0] or ""}')
        # Rows only in full = codes first ordered after T: production at T could not know them yet
        # (the known "rows before the first order" effect, not a feature leak).
        print(f'   rows only in full: {len(r["only_full"])} (codes first ordered after T), '
              f'only in truncated: {len(r["only_trunc"])}')
        print(f'   PlanModelCodes re-coded after T (expected, reported apart): {sorted(recoded) or "none"}')
        if len(d):
            d = d.assign(asof=asof.date(), recoded=d['PlanModelCode'].isin(recoded))
            frames.append(d)
        if len(real):
            per_col = real.groupby('column').size().sort_values(ascending=False)
            print(f'   LEAK? {len(real):,} differing cell(s) outside re-coded codes, in {len(per_col)} '
                  f'column(s): {per_col.head(10).to_dict()}')
            failures.append(asof.date())
        else:
            print(f'   PASS: every compared feature identical outside re-coded codes '
                  f'(max |diff| {r["worst"]:.3g}; {len(d)} cell(s) on re-coded codes)')
        if r['dup'] != (0, 0) or r['cols'][0] or not same_features or len(r['only_trunc']):
            failures.append(asof.date())

    if frames:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f'real_truncation_{plant}.csv'
        pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        print(f'\ndifferences written to {out}')
    full_path.unlink(missing_ok=True)
    return sorted(set(failures))


if __name__ == '__main__':
    main()
