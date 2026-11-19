"""Truncation-invariance leakage test for the full training panel, at PlanModelCode grain.

The rule: a feature on a row whose as-of point is T may only use what was known at T. So build the
panel twice - once from the full synthetic feed, once from the feed as it looked at T (orders entered
after T removed, cancellations known after T reverted to open) - and every feature column on every
row with as-of <= T must be identical, on identical row sets. Any difference is a leak.

Sections:
  1. day grain, swept over the as-of dates most likely to break (day 0, day 1, weekends, the
     clamped days 28-30, February, leap day)
  2. month grain, at month ends
  3. sentinels: a one-day (day grain) and one-month (month grain) look-ahead injected into every
     query_known_order caller must be caught - proves 1-2 are not vacuous
  4. HasSuccessor rule: flagged exactly when the successor's start is known and <= OriginIdx
  5. HasExternalEvent rule: equals the flag from the event config as it could have been written at
     each origin (hand-written config, so 1-3 can't see it); config rejects a stray END_KNOWN pair
  6. no look-ahead through the inference path: a training row equals a live
     preprocess_for_inference run on its own as-of date (ported from verify_no_lookahead.py)
  7. informational: a ProductCode re-coded to a new PlanModelCode after T - resolve_plan_model_codes
     moves its whole history to the LATEST order's code, so the full build reads the future

The synthetic feed is synthetic_fabric's, mapped onto PlanModelCode here: two multi-SalesModelCode
PlanModelCodes, blank PlanModelCodes filled from the product's other orders, and two successors -
one with a TFG start, one without (first-order fallback).

Known blind spots - what passing does NOT prove:
  - Restated source rows (Qty / request-date edits, deleted orders): one row per order here.
  - TFGMaterial and ProductSuccessor are snapshots in production; here the TFG is rebuilt from the
    truncated feed, so a launch date known only after T is out of scope.
  - The truncated build is forced to the full build's max_settled_target, so both span the same
    targets; production at T would settle earlier. Only rows with as-of <= T are compared.
  - Month grain: T must be a month end - an OriginIdx is only fully known once its month closes.
  - The compared columns come from the full build's feature_cols (plus ForecastEligibilityAge); a
    column only the truncated build has is caught by the column-set check, not compared.
  - The sentinel patches features, intramonth_features, country_visibility and order_book - every
    module calling order_book.query_known_order on the panel path. Re-check this list when a
    caller is added.
  - Section 5 proves the flag follows the knowability rule, not that .env is truthful: whether an
    END_KNOWN end really was announced when the event started is a business fact.
"""
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'tests'))
os.chdir(_REPO_ROOT)
import no_db  # noqa: E402,F401

from synthetic_fabric import FILE_DATE, FabricStub, make_orders  # noqa: E402
from adj_forecast_lib import (  # noqa: E402
    country_visibility, features, intramonth_features, order_book,
)
from adj_forecast_lib import preprocessing as pp  # noqa: E402
from adj_forecast_lib.config import (  # noqa: E402
    CUTOFF_DATE_COL, PLANT_EXTERNAL_EVENT_END_KNOWN, PLANT_EXTERNAL_EVENT_RANGES, PLANT_INCLUDES_PRED_SUCC,
    PLANT_NAMES, is_event_month_idx, month_idx_to_yyyymm, to_month_idx,
)
from adj_forecast_lib.lifecycle_features import FORECAST_ELIGIBILITY_AGE_COL  # noqa: E402

PLANTS = sorted(PLANT_NAMES)       # every configured plant: STJ and SPL
CURRENT = {'plant': PLANTS[0]}     # the plant sections 1-4 are building for right now
TAG = lambda: f"{CURRENT['plant']} ({PLANT_NAMES[CURRENT['plant']]})"
LABELS = {'ActualOrder', 'Adjustment'}
ATOL = 1e-9                      # exact comparison: the same orders summed in the same order
DAY_KEY = ['PlanModelCode', 'TargetIdx', 'Horizon', CUTOFF_DATE_COL]
MONTH_KEY = ['PlanModelCode', 'TargetIdx', 'Horizon']

DAY_DATES = [
    '2024-02-29',  # leap-year month end (day 0)
    '2024-03-01',  # day 1 after a leap February
    '2024-03-30',  # day 30: shift_asof(-1 month) clamps onto Feb 29, a closed month
    '2025-01-31',  # day 0, 31-day month
    '2025-02-01',  # day 1, a Saturday
    '2025-02-27',  # last day of a non-leap February window
    '2025-03-15',  # mid-month Saturday
    '2025-03-30',  # day 30, clamps onto Feb 28
    '2025-04-30',  # day 0, 30-day month
    '2026-02-27',
    '2026-08-31',  # last settled month end in the synthetic feed
]
MONTH_DATES = ['2024-02-29', '2025-06-30', '2026-03-31']

# SalesModelCode -> PlanModelCode. PNA and PNB carry two SalesModelCodes each.
SMC_TO_PMC = {'A1': 'PNA', 'A2': 'PNA', 'B1': 'PNB', 'B2': 'PNB', 'C1': 'PNC1', 'C2': 'PNC2',
              'KNOWN1': 'PNK', 'Z1': 'PNZ1', 'Y1': 'PNY1'}
# ProductSuccessor pairs (SalesModelCode level, as the table stores them). Y1 has no TFG start, so
# its start falls back to its first order's month.
SUCCESSOR_PAIRS = pd.DataFrame({'SalesModelCode': ['C1', 'C2'], 'SuccessorCode': ['Z1', 'Y1']})
NO_TFG_START = {'Y1'}

FAILURES = []


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f'  [{detail}]' if detail and not ok else ''))
    if not ok:
        FAILURES.append(name)


def _copy_as(orders, source, start, code):
    """`source`'s orders entered on or after `start`, relabelled as a new SalesModelCode `code`."""
    o = orders[(orders['SalesModelCode'] == source) & (orders['EntryDate'] >= start)].copy()
    return o.assign(SalesModelCode=code, ProductCode=code + 'P')


def make_pmc_orders() -> pd.DataFrame:
    """synthetic_fabric's orders plus two successors, mapped onto PlanModelCode; ~5% of rows blank,
    to be filled from their ProductCode's other orders."""
    o = make_orders()
    o = pd.concat([o, _copy_as(o, 'A1', '2023-06-20', 'Z1'), _copy_as(o, 'B1', '2023-10-01', 'Y1')],
                  ignore_index=True)
    o['PlanModelCode'] = o['SalesModelCode'].map(SMC_TO_PMC)
    blank = np.random.default_rng(11).random(len(o)) < 0.05
    o.loc[blank, 'PlanModelCode'] = np.nan
    return o


def make_pmc_tfg(orders: pd.DataFrame) -> pd.DataFrame:
    """One TFGMaterial row per ProductCode, launched 5 days before its first order."""
    first = orders.groupby('ProductCode')['EntryDate'].min()
    smc = orders.groupby('ProductCode')['SalesModelCode'].first().reindex(first.index)
    start = (first - pd.Timedelta(days=5)).where(~smc.isin(NO_TFG_START))
    return pd.DataFrame({'ProductCode': first.index, 'SalesModelCode': smc.to_numpy(),
                         'SubModelCode': (smc + 'S').to_numpy(), 'ProductAbbreviation': smc.to_numpy(),
                         'ProductStartDate': start.to_numpy(), 'ProductEndDate': pd.NaT})


class PmcFabricStub(FabricStub):
    """FabricStub with a PlanModelCode-aware TFG and SUCCESSOR_PAIRS as the ProductSuccessor table."""

    def _load_tfg(self):
        self.calls['load_tfg_material'] += 1
        return make_pmc_tfg(self.orders)

    def _successor_pairs(self):
        return SUCCESSOR_PAIRS


def feed_as_of(orders, asof):
    """The feed a file dated asof+1 holds: nothing entered later, no cancellation known later."""
    o = orders[orders['EntryDate'] <= asof].copy()
    later = o['IsCanceled'] & (o['CancelKnownDate'] > asof)
    o.loc[later, 'IsCanceled'] = False
    o.loc[later, 'CancelMonthIdx'] = np.nan
    o.loc[later, ['CancelKnownDate', 'CancelDate']] = pd.NaT
    return o


def build(orders, file_date, grain, max_settled_target=None, plant=None):
    plant = CURRENT['plant'] if plant is None else plant
    with PmcFabricStub(orders, file_date=file_date):
        ctx = pp.build_order_book_context(plant, max_settled_target=max_settled_target,
                                          cutoff_grain=grain)
        panel, _fams, feature_cols = pp.build_full_panel(ctx)
    return panel.assign(PlanModelCode=panel['PlanModelCode'].astype(str)), feature_cols, ctx


def compared_cols(feature_cols):
    # SalesModelCode/PartCode stay in: they are the PlanModelCode's FIRST order's codes, so a
    # truncated feed could pick a different one.
    return sorted(set(feature_cols + [FORECAST_ELIGIBILITY_AGE_COL]) - {'PlanModelCode'} - LABELS)


def compare_cols(a, b, cols):
    """({column: description} for every differing column, largest absolute numeric difference)."""
    bad, worst = {}, 0.0
    for c in cols:
        x, y = a[c], b[c]
        if x.dtype.kind in 'fiub' and y.dtype.kind in 'fiub':
            xv, yv = x.to_numpy('float64'), y.to_numpy('float64')
            one_nan = np.isnan(xv) != np.isnan(yv)
            gap = np.where(np.isnan(xv) | np.isnan(yv), 0.0, np.abs(xv - yv))
            worst = max(worst, float(gap.max(initial=0.0)))
            d = one_nan | (gap > ATOL)
        else:
            d = x.astype(str).to_numpy() != y.astype(str).to_numpy()
        if d.any():
            i = np.flatnonzero(d)[0]
            bad[c] = f'{int(d.sum())}/{len(d)} rows, e.g. full={x.iloc[i]!r} trunc={y.iloc[i]!r} at {a.index[i]}'
    return bad, worst


def restrict(panel, grain, asof):
    if grain == 'day':
        return panel[panel[CUTOFF_DATE_COL] <= asof].set_index(DAY_KEY).sort_index()
    if not asof.is_month_end:
        raise ValueError(f'month-grain as-of {asof.date()} must be a month end')
    origin_cap = to_month_idx(int(asof.strftime('%Y%m')))
    return panel[panel['OriginIdx'] <= origin_cap].set_index(MONTH_KEY).sort_index()


def truncation_diff(full, cols, grain, asof, settled, orders):
    trunc, _, _ = build(feed_as_of(orders, asof), asof + pd.Timedelta(days=1), grain,
                        max_settled_target=settled)
    a, b = restrict(full, grain, asof), restrict(trunc, grain, asof)
    common = a.index.intersection(b.index)
    bad, worst = compare_cols(a.loc[common], b.loc[common], cols)
    return {'bad': bad, 'worst': worst, 'n': len(common),
            'only_full': a.index.difference(b.index), 'only_trunc': b.index.difference(a.index),
            'dup_full': int(a.index.duplicated().sum()), 'dup_trunc': int(b.index.duplicated().sum()),
            'cols_full_only': sorted(set(full.columns) - set(trunc.columns)),
            'cols_trunc_only': sorted(set(trunc.columns) - set(full.columns))}


def order_activity(orders, asof):
    """Orders entered on T, cancellations known on T, and orders entered in the week after T."""
    entered = int((orders['EntryDate'] == asof).sum())
    cancelled = int((orders['IsCanceled'] & (orders['CancelKnownDate'] == asof)).sum())
    week_after = int(((orders['EntryDate'] > asof)
                      & (orders['EntryDate'] <= asof + pd.Timedelta(days=7))).sum())
    return entered, cancelled, week_after


def run_sweep(grain, dates, orders):
    full, feature_cols, ctx = build(orders, FILE_DATE, grain)
    settled = int(month_idx_to_yyyymm([ctx['max_settled_idx']])[0])
    cols = compared_cols(feature_cols)
    print(f'  full {grain} panel {full.shape}, {len(cols)} columns compared, settled {settled}, '
          f'{full["PlanModelCode"].nunique()} PlanModelCodes')
    for d in dates:
        asof = pd.Timestamp(d)
        t0 = time.time()
        entered, cancelled, week_after = order_activity(orders, asof)
        print(f'  -- as-of {d} ({asof.day_name()[:3]}): {entered} order(s) entered on T, '
              f'{cancelled} cancellation(s) known on T, {week_after} entered in the week after T')
        r = truncation_diff(full, cols, grain, asof, settled, orders)
        check(f'{TAG()} {grain} {d}: keys unique in both builds', not r['dup_full'] and not r['dup_trunc'],
              f"dup_full={r['dup_full']} dup_trunc={r['dup_trunc']}")
        check(f'{TAG()} {grain} {d}: column sets equal', not r['cols_full_only'] and not r['cols_trunc_only'],
              f"full_only={r['cols_full_only']} trunc_only={r['cols_trunc_only']}")
        check(f'{TAG()} {grain} {d}: {r["n"]} rows, row sets equal',
              not len(r['only_full']) and not len(r['only_trunc']),
              f"only_full={len(r['only_full'])} only_trunc={len(r['only_trunc'])}")
        check(f'{TAG()} {grain} {d}: all {len(cols)} columns identical, max |diff| {r["worst"]:.3g} '
              f'({time.time() - t0:.0f}s)',
              not r['bad'], '; '.join(f'{c}: {m}' for c, m in list(r['bad'].items())[:6]))
    return full, cols, settled, ctx


def event_flag_as_written_at(origin, target, ranges, end_known):
    """HasExternalEvent from the config as it could have been written at each origin: an event not
    yet started is absent, and an end neither announced nor passed is open."""
    origin, target = np.asarray(origin), np.asarray(target)
    flag = np.zeros(len(origin), dtype=bool)
    for lo, hi in ranges:
        start, end = to_month_idx(lo), to_month_idx(hi)
        visible_end = np.where(((lo, hi) in end_known) | (origin > end), end, np.iinfo(np.int64).max)
        flag |= (origin >= start) & (target >= start) & (target <= visible_end)
    return flag


def check_event_flag(panel, grain, feature_cols):
    origin, target = panel['OriginIdx'].to_numpy(), panel['TargetIdx'].to_numpy()
    expected = event_flag_as_written_at(origin, target, TEST_EVENT_RANGES, TEST_EVENT_END_KNOWN)
    got = panel['HasExternalEvent'].to_numpy() == 1
    check(f'{TAG()} {grain}: HasExternalEvent uses only events known at the origin',
          bool((got == expected).all()), f'{int((got != expected).sum())} row(s) differ')
    hindsight = is_event_month_idx(target, TEST_EVENT_RANGES)
    check(f'{TAG()} {grain}: the target-anchored hindsight flag would fail this check (not vacuous)',
          bool((hindsight != expected).any()))
    for lo, hi in TEST_EVENT_RANGES:
        in_event = (origin >= to_month_idx(lo)) & (origin <= to_month_idx(hi))
        check(f'{TAG()} {grain}: event {lo}-{hi} flags some rows (not vacuous)', bool(got[in_event].any()))
    check(f'{TAG()} {grain}: EventTargetHindsight is the hindsight flag (sample weight only)',
          bool((panel['EventTargetHindsight'].to_numpy() == hindsight).all()))
    check(f'{TAG()} {grain}: HasExternalEvent is a feature, EventTargetHindsight is not',
          'HasExternalEvent' in feature_cols and 'EventTargetHindsight' not in feature_cols
          and 'EventTargetHindsight' not in pp.full_feature_cols())


def with_patched_query(fake, fn):
    """Runs fn() with every panel-path query_known_order caller routed through `fake`."""
    real = order_book.query_known_order
    callers = [features, intramonth_features, country_visibility, order_book]
    for m in callers:
        m.query_known_order = fake
    try:
        return fn()
    finally:
        for m in callers:
            m.query_known_order = real


_real_query = order_book.query_known_order


def peeks_one_day_ahead(q, book, group_cols, arrival_col='EntryMonthIdx', asof_col='CutoffMonthIdx'):
    if arrival_col == 'EntryDate':
        q = q.copy()
        q[asof_col] = pd.to_datetime(q[asof_col]) + pd.Timedelta(days=1)
    return _real_query(q, book, group_cols, arrival_col=arrival_col, asof_col=asof_col)


def peeks_one_month_ahead(q, book, group_cols, arrival_col='EntryMonthIdx', asof_col='CutoffMonthIdx'):
    if arrival_col == 'EntryMonthIdx':
        q = q.copy()
        q[asof_col] = q[asof_col] + 1
    return _real_query(q, book, group_cols, arrival_col=arrival_col, asof_col=asof_col)


# On here for every plant, whatever .env says, so HasPredecessor/HasSuccessor are always tested.
# Same for the external events: two inside the synthetic feed's span, one whose end was announced
# up front and one whose end only became known once it passed.
TEST_EVENT_RANGES = [(202403, 202405), (202509, 202512)]
TEST_EVENT_END_KNOWN = [(202509, 202512)]
for _plant in PLANT_NAMES:
    PLANT_INCLUDES_PRED_SUCC[_plant] = True
    PLANT_EXTERNAL_EVENT_RANGES[_plant] = TEST_EVENT_RANGES
    PLANT_EXTERNAL_EVENT_END_KNOWN[_plant] = TEST_EVENT_END_KNOWN

ORDERS = make_pmc_orders()

for plant in PLANTS:
    CURRENT['plant'] = plant
    print(f'\n#################### {TAG()} ####################')
    print('=== 1. day grain: features invariant to truncating the feed at the as-of date ===')
    full_day, day_cols, day_settled, day_ctx = run_sweep('day', DAY_DATES, ORDERS)

    print('\n=== 2. month grain: same, at month ends ===')
    full_month, month_cols, month_settled, _ = run_sweep('month', MONTH_DATES, ORDERS)

    print('\n=== 3. sentinels: an injected look-ahead is caught ===')
    leaky_day, _, _ = with_patched_query(peeks_one_day_ahead, lambda: build(ORDERS, FILE_DATE, 'day'))
    expected = {'KnownOrder', 'Accumulation_7d', 'KnownCNShare'} & set(day_cols)
    for d in ['2025-03-14', '2025-04-30']:      # mid-month, and a month boundary (day 0)
        bad = truncation_diff(leaky_day, day_cols, 'day', pd.Timestamp(d), day_settled, ORDERS)['bad']
        check(f'{TAG()} day {d}: one-day look-ahead flagged on {sorted(expected)}', expected <= set(bad),
              f'flagged only {sorted(bad)}')
        print(f'    (leak reached {len(bad)} columns)')
    leaky_month, _, _ = with_patched_query(peeks_one_month_ahead,
                                           lambda: build(ORDERS, FILE_DATE, 'month'))
    bad = truncation_diff(leaky_month, month_cols, 'month', pd.Timestamp('2025-06-30'), month_settled,
                          ORDERS)['bad']
    check(f'{TAG()} month 2025-06-30: one-month look-ahead flagged on KnownOrder', 'KnownOrder' in bad,
          f'flagged only {sorted(bad)}')
    print(f'    (leak reached {len(bad)} columns: {sorted(bad)})')

    print('\n=== 4. HasSuccessor: flagged exactly when the successor start is known and <= OriginIdx ===')
    succ_start = day_ctx['pred_succ_flags'].set_index('PlanModelCode')['SuccessorStart']
    start = full_day['PlanModelCode'].map(succ_start).astype('float64')
    flagged = full_day['HasSuccessor'] == 1
    due = np.isfinite(start) & (start <= full_day['OriginIdx'])
    check(f'{TAG()} every flagged row has a finite successor start <= OriginIdx',
          bool(np.isfinite(start[flagged]).all()
               and (start[flagged] <= full_day.loc[flagged, 'OriginIdx']).all()),
          f'{int((flagged & ~due).sum())} row(s) flagged early or without a start')
    check(f'{TAG()} every row past its successor start is flagged', bool(flagged[due].all()),
          f'{int((due & ~flagged).sum())} row(s) missed')
    check(f'{TAG()} both successor pairs flag some rows (not vacuous)',
          set(full_day.loc[flagged, 'PlanModelCode']) == {'PNC1', 'PNC2'},
          f"flagged codes {sorted(set(full_day.loc[flagged, 'PlanModelCode']))}")
    first_entry = ORDERS.groupby('SalesModelCode')['EntryMonthIdx'].min()
    check(f'{TAG()} a successor without a TFG start falls back to its first order month',
          succ_start.get('PNC2') == first_entry['Y1'],
          f"PNC2 start {succ_start.get('PNC2')} vs Y1 first order {first_entry['Y1']}")

    print('\n=== 5. HasExternalEvent: only events known at the origin (config knowability) ===')
    # Truncating orders can't catch this: the event ranges are hand-written config, not data.
    check_event_flag(full_day, 'day', day_cols)
    check_event_flag(full_month, 'month', month_cols)
    name = PLANT_NAMES[plant]
    bad_env = dict(os.environ, **{f'{name}_EXTERNAL_EVENT_END_KNOWN': '[[209901, 209912]]'})
    r = subprocess.run([sys.executable, '-c', 'import adj_forecast_lib.config'], env=bad_env,
                       cwd=_REPO_ROOT, capture_output=True, text=True)
    check(f'{TAG()} config rejects an END_KNOWN pair that is not an event range',
          r.returncode != 0 and f'{name}_EXTERNAL_EVENT_END_KNOWN' in r.stderr, r.stderr[-300:])

print('\n=== 6. no look-ahead through the inference path: training row == live run on its date ===')
NL_KEY = ['PlanModelCode', 'TargetIdx', 'Horizon']
NL_BASELINES = ['AdjTrailMean3', 'AdjTrailMean6', 'AdjTrailMean12',
                'ManualSim3', 'ManualSim6', 'ManualSim12']
NL_POINTS = [(202601, '2026-01-31'), (202601, '2026-02-01'), (202601, '2026-02-10'),
             (202601, '2026-02-27'), (202512, '2026-01-30'), (202603, '2026-04-17')]


def live_vs_training(plant, panel, cols):
    """{(origin, date): (n shared rows, n training rows, {col: diff})} across NL_POINTS."""
    out = {}
    for origin, date in NL_POINTS:
        date = pd.Timestamp(date)
        train_rows = panel[(panel['OriginIdx'] == int(to_month_idx(origin)))
                           & (panel[CUTOFF_DATE_COL] == date)].set_index(NL_KEY).sort_index()
        with PmcFabricStub(feed_as_of(ORDERS, date), file_date=date + pd.Timedelta(days=1)):
            live, _ctx = pp.preprocess_for_inference(plant, origin, range(1, 7), cutoff_grain='day',
                cutoff_date=date, min_origin_months_since_launch=None)
        live = live.assign(PlanModelCode=live['PlanModelCode'].astype(str)).set_index(NL_KEY).sort_index()
        common = train_rows.index.intersection(live.index)
        missing = [c for c in cols if c not in live.columns]
        bad, _ = compare_cols(train_rows.loc[common], live.loc[common], [c for c in cols if c not in missing])
        bad.update({c: 'absent from the live rows' for c in missing})
        out[(origin, date)] = (len(common), len(train_rows), bad)
    return out


def training_panel(plant):
    panel, feature_cols, _ = build(ORDERS, FILE_DATE, 'day', plant=plant)
    return panel, sorted(set(compared_cols(feature_cols)) | set(NL_BASELINES))


for plant in sorted(PLANT_NAMES):
    tag = f'{plant} ({PLANT_NAMES[plant]})'
    panel, cols = training_panel(plant)
    for (origin, date), (shared, n_train, bad) in live_vs_training(plant, panel, cols).items():
        check(f'{tag} origin {origin} as of {date.date()}: all {n_train} training rows have a live twin',
              n_train > 0 and shared == n_train, f'{shared} of {n_train} matched')
        check(f'{tag} origin {origin} as of {date.date()}: all {len(cols)} columns identical',
              not bad, '; '.join(f'{c}: {m}' for c, m in list(bad.items())[:4]))

plant = sorted(PLANT_NAMES)[0]
leaky_panel, cols = with_patched_query(peeks_one_day_ahead, lambda: training_panel(plant))
flagged_cols = {c for _s, _n, bad in live_vs_training(plant, leaky_panel, cols).values() for c in bad}
check('inference path: a one-day look-ahead is flagged on KnownOrder and the families built on it',
      {'KnownOrder', 'Accumulation_1mo', 'ManualSim6'} <= flagged_cols,
      f'flagged only {sorted(flagged_cols)}')

print('\n=== 7. informational: a ProductCode re-coded to a new PlanModelCode after T ===')
recoded = ORDERS.copy()
moved = (recoded['ProductCode'] == 'C2P') & (recoded['EntryDate'] >= '2025-06-01') \
    & recoded['PlanModelCode'].notna()
recoded.loc[moved, 'PlanModelCode'] = 'PNC2X'
full_r, cols_r, ctx_r = build(recoded, FILE_DATE, 'day')
asof = pd.Timestamp('2025-03-31')
r = truncation_diff(full_r, compared_cols(cols_r), 'day', asof,
                    int(month_idx_to_yyyymm([ctx_r['max_settled_idx']])[0]), recoded)
by_code = lambda idx: sorted({k[0] for k in idx})
print(f'  as-of {asof.date()}, C2P re-coded PNC2 -> PNC2X from 2025-06-01: '
      f"{len(r['only_full'])} row(s) only in the full build {by_code(r['only_full'])}, "
      f"{len(r['only_trunc'])} only in the truncated build {by_code(r['only_trunc'])}, "
      f"{len(r['bad'])} column(s) differ on shared rows {sorted(r['bad'])[:6]}")
print('  (non-zero = resolve_plan_model_codes reads the latest order\'s code; not a failure here)')

print('\n' + '=' * 78)
print(f'{len(FAILURES)} FAILURE(S): {FAILURES}' if FAILURES else 'ALL CHECKS PASSED')
sys.exit(1 if FAILURES else 0)
