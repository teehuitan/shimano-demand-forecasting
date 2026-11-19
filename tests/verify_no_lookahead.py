"""No look-ahead, end to end: a training row is exactly what a live run on its as-of date sees.

Every other suite pins one invariant of one family. This one asks the question they all serve, for
every feature at once: take a row of the DAY-GRAIN TRAINING panel, built from the full order
history, and rebuild it the way production would on that row's as-of date D - from a feed that
only contains what existed on D (nothing entered later, no cancellation recorded later), through
the real preprocess_for_inference. Every feature column must come back identical.

A difference means one of two defects, and both are fatal to daily refresh:
  - look-ahead: the training row read something after D, so offline accuracy is optimistic;
  - train/serve skew: the live path computes the column differently from the training path, so
    the deployed model is fed a distribution it never saw.

AdjTrailMean*/ManualSim* are included although they are month-grain baselines, because their
training history (the panel's own day-0 rows) and their live history
(build_adjustment_history_lookback) are built by different code - exactly where skew hides.

The as-of points cover the edges of the schedule: day 0 (the origin month's end), day 1, a
mid-month day, and the last day of a window (Feb 27, Jan 30) - the day before the next origin's
day 0.

Section 2 injects a one-day leak into the day-grain KnownOrder query and requires this suite to
catch it, so "no differences" can't pass vacuously.
"""
import os
import pathlib
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'tests'))
os.chdir(_REPO_ROOT)        # load_io's SQL and the replacement workbook are cwd-relative
import no_db  # noqa: E402,F401 - every DB engine raises from here on (see no_db.py)

from synthetic_fabric import FabricStub, make_orders  # noqa: E402
from adj_forecast_lib import features  # noqa: E402
from adj_forecast_lib import preprocessing as pp  # noqa: E402
from adj_forecast_lib.config import CUTOFF_DATE_COL, PLANT_NAMES, to_month_idx  # noqa: E402
from adj_forecast_lib.lifecycle_features import FORECAST_ELIGIBILITY_AGE_COL  # noqa: E402

FAILURES = []


def check(label, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' - ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(label)


ORDERS = make_orders()
KEY = ['SalesModelCode', 'TargetIdx', 'Horizon']
BASELINES = ['AdjTrailMean3', 'AdjTrailMean6', 'AdjTrailMean12',
             'ManualSim3', 'ManualSim6', 'ManualSim12']
# (origin YYYYMM, as-of date): see the module docstring for why these.
POINTS = [(202601, '2026-01-31'), (202601, '2026-02-01'), (202601, '2026-02-10'),
          (202601, '2026-02-27'), (202512, '2026-01-30'), (202603, '2026-04-17')]


def feed_as_of(orders: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """What a feed complete through `asof` contains (a source file dated asof+1): no order entered
    later, and no cancellation recorded later - such an order is simply still open on that date."""
    o = orders[orders['EntryDate'] <= asof].copy()
    later = o['CancelDate'] > asof
    o.loc[later, ['IsCanceled', 'CancelKnownValid']] = False
    o.loc[later, 'CancelMonthIdx'] = np.nan
    o.loc[later, 'CancelDate'] = pd.NaT
    return o


def diff_columns(train_rows: pd.DataFrame, live_rows: pd.DataFrame, cols: list) -> dict:
    """{column: description} for every column that differs on the rows both sides have."""
    a = train_rows.assign(SalesModelCode=train_rows['SalesModelCode'].astype(str)) \
        .set_index(KEY).sort_index()
    b = live_rows.assign(SalesModelCode=live_rows['SalesModelCode'].astype(str)) \
        .set_index(KEY).sort_index()
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    bad = {}
    for c in cols:
        if c not in a.columns or c not in b.columns:
            bad[c] = 'present on one side only'
            continue
        x, y = a[c], b[c]
        if x.dtype.kind in 'fiub' and y.dtype.kind in 'fiub':
            xv, yv = x.to_numpy('float64'), y.to_numpy('float64')
            differs = ~((np.isnan(xv) & np.isnan(yv)) | np.isclose(xv, yv, rtol=1e-5, atol=1e-6))
        else:
            differs = x.astype(str).to_numpy() != y.astype(str).to_numpy()
        if differs.any():
            i = np.flatnonzero(differs)[0]
            bad[c] = (f'{int(differs.sum())}/{len(differs)} row(s), e.g. train={x.iloc[i]!r} '
                      f'live={y.iloc[i]!r} at {common[i]}')
    return bad


def compare(plant: int, panel: pd.DataFrame, cols: list) -> dict:
    """{(origin, date): (n shared rows, n train rows, {col: diff})} across POINTS."""
    out = {}
    for origin, date in POINTS:
        date = pd.Timestamp(date)
        train_rows = panel[(panel['OriginIdx'] == int(to_month_idx(origin)))
                           & (panel[CUTOFF_DATE_COL] == date)]
        # the file that yields cutoff D is dated D+1 (complete through D only)
        with FabricStub(feed_as_of(ORDERS, date), file_date=date + pd.Timedelta(days=1)):
            live, _ctx = pp.preprocess_for_inference(plant, origin, range(1, 7),
                cutoff_grain='day', cutoff_date=date, min_origin_months_since_launch=None)
        shared = len(train_rows.set_index(KEY).index.intersection(
            live.assign(SalesModelCode=live['SalesModelCode'].astype(str)).set_index(KEY).index))
        out[(origin, date)] = (shared, len(train_rows), diff_columns(train_rows, live, cols))
    return out


def training_panel(plant: int):
    with FabricStub(ORDERS):
        panel, _fams, feature_cols, _ctx = pp.preprocess_for_training(plant, cutoff_grain='day')
    panel = panel.assign(SalesModelCode=panel['SalesModelCode'].astype(str))
    # The eligibility age isn't a feature, but it decides which rows are scored, so it must agree too.
    cols = sorted(set(feature_cols + BASELINES + [FORECAST_ELIGIBILITY_AGE_COL])
                  - {'SalesModelCode', 'PartCode'})
    return panel, cols


for plant in sorted(PLANT_NAMES):
    tag = f'{plant} ({PLANT_NAMES[plant]})'
    print(f'\n=== 1. {tag}: training rows vs a live run on their own as-of date ===')
    panel, cols = training_panel(plant)
    for (origin, date), (shared, n_train, bad) in compare(plant, panel, cols).items():
        check(f'{tag} origin {origin} as of {date.date()}: every training row has a live twin',
              n_train > 0 and shared == n_train, f'{shared} of {n_train} matched')
        check(f'{tag} origin {origin} as of {date.date()}: all {len(cols)} columns identical',
              not bad, '; '.join(f'{c}: {m}' for c, m in list(bad.items())[:4]))

print('\n=== 2. the suite catches a one-day leak (so section 1 is not vacuous) ===')
real_query = features.query_known_order


def peeks_one_day_ahead(q, book, group_cols, arrival_col='EntryMonthIdx', asof_col='CutoffMonthIdx'):
    if arrival_col == 'EntryDate':
        q = q.copy()
        q[asof_col] = pd.to_datetime(q[asof_col]) + pd.Timedelta(days=1)
    return real_query(q, book, group_cols, arrival_col=arrival_col, asof_col=asof_col)


plant = sorted(PLANT_NAMES)[0]
features.query_known_order = peeks_one_day_ahead
try:
    leaky_panel, cols = training_panel(plant)
finally:
    features.query_known_order = real_query
flagged = {c for _s, _n, bad in compare(plant, leaky_panel, cols).values() for c in bad}
check('a KnownOrder query peeking one day ahead is flagged, on KnownOrder itself and the families '
      'built on it', {'KnownOrder', 'Accumulation_1mo', 'ManualSim6'} <= flagged,
      f'flagged only {sorted(flagged)}')

print('\n' + '=' * 78)
print(f'{len(FAILURES)} FAILURE(S): {FAILURES}' if FAILURES else 'ALL CHECKS PASSED')
sys.exit(1 if FAILURES else 0)
