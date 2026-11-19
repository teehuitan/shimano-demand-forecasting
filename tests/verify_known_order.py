"""Edit-3 verification: attach_known_order_features across both as-of grains.

The load-bearing check is REGRESSION - the month path must reproduce the pre-edit function
exactly, all 13 columns, since every current caller relies on it. The original is reimplemented
verbatim below and diffed against, rather than eyeballed.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'tests'))
import no_db  # noqa: E402,F401 - every DB engine raises from here on (see no_db.py)

from adj_forecast_lib.config import month_idx_to_yyyymm, to_month_idx
from adj_forecast_lib.features import attach_known_order_features, safe_ratio
from adj_forecast_lib.order_book import (
    build_order_book, query_known_order, resolve_known_order_gaps, shift_asof,
)

FAILURES = []
OUT_COLS = ['KnownOrder', 'KnownOrder_prevOrigin', 'Accumulation_1mo', 'AccumulationPct_1mo',
            'KnownOrder_3moAgo', 'Momentum_3mo', 'Accumulation_3mo',
            'KnownOrder_6moAgo', 'Momentum_6mo', 'Accumulation_6mo',
            'KnownOrder_YoY_prior', 'KnownOrder_YoY', 'KnownOrder_x_Horizon']


def check(label, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' - ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(label)


def same(a, b):
    """NaN-aware elementwise equality."""
    a, b = np.asarray(a, dtype='float64'), np.asarray(b, dtype='float64')
    return ((np.isnan(a) & np.isnan(b)) | (a == b)).all()


def month_end(month_idx):
    yyyymm = int(month_idx_to_yyyymm(np.array([month_idx]))[0])
    return pd.Timestamp(year=yyyymm // 100, month=yyyymm % 100, day=1) + pd.offsets.MonthEnd(0)


# ---------------------------------------------------------------- the pre-edit implementation
def attach_known_order_features_ORIGINAL(panel, order_book_code, first_entry_by_code,
                                         group_col='PlanModelCode'):
    panel = panel.copy()
    q = panel.rename(columns={'TargetIdx': 'RequestMonthIdx', 'OriginIdx': 'CutoffMonthIdx'})
    raw = query_known_order(q[[group_col, 'RequestMonthIdx', 'CutoffMonthIdx']],
                            order_book_code, [group_col])
    panel['KnownOrder'] = np.nan_to_num(raw, nan=0.0)
    first_entry = panel[group_col].map(first_entry_by_code).to_numpy()

    def _known_offset(offset):
        qq = panel[[group_col, 'TargetIdx']].rename(columns={'TargetIdx': 'RequestMonthIdx'})
        qq['CutoffMonthIdx'] = panel['OriginIdx'].to_numpy() - offset
        raw_o = query_known_order(qq, order_book_code, [group_col])
        return resolve_known_order_gaps(raw_o, qq['CutoffMonthIdx'].to_numpy(), first_entry)

    panel['KnownOrder_prevOrigin'] = _known_offset(1)
    panel['Accumulation_1mo'] = panel['KnownOrder'] - panel['KnownOrder_prevOrigin']
    panel['AccumulationPct_1mo'] = safe_ratio(panel['Accumulation_1mo'], panel['KnownOrder_prevOrigin'])
    for k in (3, 6):
        panel[f'KnownOrder_{k}moAgo'] = _known_offset(k)
        panel[f'Momentum_{k}mo'] = safe_ratio(panel['KnownOrder'], panel[f'KnownOrder_{k}moAgo'])
        panel[f'Accumulation_{k}mo'] = panel['KnownOrder'] - panel[f'KnownOrder_{k}moAgo']
    yoy_q = panel[[group_col, 'TargetIdx', 'OriginIdx']].copy()
    yoy_q['RequestMonthIdx'] = yoy_q['TargetIdx'] - 12
    yoy_q['CutoffMonthIdx'] = yoy_q['OriginIdx'] - 12
    yoy_raw = query_known_order(yoy_q[[group_col, 'RequestMonthIdx', 'CutoffMonthIdx']],
                                order_book_code, [group_col])
    panel['KnownOrder_YoY_prior'] = resolve_known_order_gaps(
        yoy_raw, yoy_q['CutoffMonthIdx'].to_numpy(), first_entry)
    panel['KnownOrder_YoY'] = safe_ratio(panel['KnownOrder'], panel['KnownOrder_YoY_prior'])
    panel['KnownOrder_x_Horizon'] = panel['KnownOrder'] * panel['Horizon']
    return panel


# ---------------------------------------------------------------------------------- fixtures
rng = np.random.default_rng(7)
N = 6000
entry_date = pd.Timestamp('2022-01-01') + pd.to_timedelta(rng.integers(0, 1400, N), unit='D')
entry_midx = to_month_idx((entry_date.year * 100 + entry_date.month).to_numpy())
orders = pd.DataFrame({
    'PlanModelCode': rng.choice(list('ABCD'), N),
    'RequestMonthIdx': (entry_midx + rng.integers(0, 7, N)).astype('int32'),
    'EntryMonthIdx': entry_midx.astype('int32'),
    'EntryDate': entry_date,
    'Qty': rng.integers(1, 400, N).astype('float32'),
})
book_m = build_order_book(orders, ['PlanModelCode'])
book_d = build_order_book(orders, ['PlanModelCode'], arrival_col='EntryDate')
first_entry = orders.groupby('PlanModelCode')['EntryMonthIdx'].min()

origins = sorted(orders['EntryMonthIdx'].unique())[14:-2]
panel = pd.DataFrame([(g, o, h) for g in list('ABCD') for o in origins for h in (1, 3, 6)],
                     columns=['PlanModelCode', 'OriginIdx', 'Horizon'])
panel['TargetIdx'] = panel['OriginIdx'] + panel['Horizon']
print(f"orders={len(orders)} panel={len(panel)} origins={len(origins)}\n")

print('1. REGRESSION - month path identical to the pre-edit function')
ref = attach_known_order_features_ORIGINAL(panel, book_m, first_entry)
new = attach_known_order_features(panel, book_m, first_entry)
mismatched = [c for c in OUT_COLS if not same(ref[c], new[c])]
check(f'all {len(OUT_COLS)} columns identical', not mismatched, f'differ: {mismatched}')
check('no columns added or dropped', set(new.columns) == set(ref.columns),
      str(set(new.columns) ^ set(ref.columns)))
check('row order and count preserved', len(new) == len(panel) and new.index.equals(panel.index))
check('output is non-trivial', np.isfinite(new['KnownOrder']).any() and new['KnownOrder'].sum() > 0)

print('\n2. day path at month-end reproduces KnownOrder exactly')
panel_d = panel.copy()
panel_d['CutoffDate'] = [month_end(o) for o in panel['OriginIdx']]
day = attach_known_order_features(panel_d, book_d, first_entry,
                                  asof_col='CutoffDate', arrival_col='EntryDate')
check('KnownOrder identical (the unshifted query)', same(new['KnownOrder'], day['KnownOrder']))
check('KnownOrder_x_Horizon identical', same(new['KnownOrder_x_Horizon'], day['KnownOrder_x_Horizon']))

print('\n3. shifted features: equal only where the calendar shift lands on a month-end')
# shift_asof is same-day-of-month, so month_end(m) - k months == month_end(m-k) only when the
# source month is not SHORTER than the target month. Feb 28 - 1mo = Jan 28, not Jan 31.
for k, col in ((1, 'KnownOrder_prevOrigin'), (3, 'KnownOrder_3moAgo'), (6, 'KnownOrder_6moAgo'),
               (12, 'KnownOrder_YoY_prior')):
    shifted = pd.to_datetime(shift_asof(panel_d['CutoffDate'].to_numpy(), k))
    expected = pd.to_datetime([month_end(o - k) for o in panel['OriginIdx']])
    lands_clean = np.asarray(shifted == expected)
    agree = np.array([same([a], [b]) for a, b in zip(ref[col], day[col])])
    check(f'{col}: matches wherever the shift lands on a month-end '
          f'({lands_clean.sum()}/{len(lands_clean)} rows)',
          agree[lands_clean].all() if lands_clean.any() else True)
    n_div = int((~lands_clean).sum())
    if n_div:
        print(f'         {n_div} row(s) shift to a non-month-end (short-month clamp) - '
              f'{"divergent as expected" if not agree[~lands_clean].all() else "coincidentally equal"}')

print('\n4. mid-month as-of resolves finer')
panel_mid = panel.copy()
panel_mid['CutoffDate'] = [month_end(o) - pd.Timedelta(days=18) for o in panel['OriginIdx']]
mid = attach_known_order_features(panel_mid, book_d, first_entry,
                                  asof_col='CutoffDate', arrival_col='EntryDate')
check('mid-month KnownOrder <= month-end KnownOrder',
      (mid['KnownOrder'].to_numpy() <= day['KnownOrder'].to_numpy()).all())
check('mid-month KnownOrder strictly smaller somewhere',
      (mid['KnownOrder'].to_numpy() < day['KnownOrder'].to_numpy()).any())
check('Accumulation_1mo still finite and varying at day grain',
      np.isfinite(mid['Accumulation_1mo']).any() and mid['Accumulation_1mo'].nunique() > 1)

print('\n5. grain guard reaches through the feature layer')
try:
    attach_known_order_features(panel_d, book_d, first_entry, asof_col='CutoffDate')  # month book default
    check('day asof + default month arrival_col raises', False, 'no exception')
except Exception as e:
    check('day asof + default month arrival_col raises', 'grain mismatch' in str(e) or
          'EntryMonthIdx' in str(e), f'{type(e).__name__}: {str(e)[:110]}')
try:
    attach_known_order_features(panel, book_d, first_entry, arrival_col='EntryDate')  # month asof
    check('month asof + day book raises', False, 'no exception')
except Exception as e:
    check('month asof + day book raises', 'grain mismatch' in str(e), f'{type(e).__name__}: {str(e)[:110]}')

print('\n6. downstream callers still import and run')
try:
    from adj_forecast_lib.preprocessing import build_adjustment_history_lookback  # noqa: F401
    import adj_forecast_lib.country_visibility  # noqa: F401
    check('preprocessing + country_visibility import', True)
except Exception as e:
    check('preprocessing + country_visibility import', False, f'{type(e).__name__}: {e}')

print('\n' + '=' * 64)
print(f"{len(FAILURES)} FAILURE(S): {FAILURES}" if FAILURES else 'ALL CHECKS PASSED')
sys.exit(1 if FAILURES else 0)
