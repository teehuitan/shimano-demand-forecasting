"""Edit-1 verification for adj_forecast_lib/order_book.py.

Central claim: a DAY-grain book queried at a month's LAST DAY must return exactly what the
MONTH-grain book returns at that month's ordinal. If that holds, the day-grain path is a strict
refinement of the monthly one and nothing downstream changes when the as-of point lands on a
month boundary. Also checks backward compatibility, the grain guard, and the two as-of helpers.

Synthetic data on purpose: load_io needs an interactive Fabric login, and the property under test
is about the primitive's arithmetic, not about STJ's particular orders.
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
from adj_forecast_lib.order_book import (
    build_order_book, query_known_order, shift_asof, asof_month_idx, _check_grain,
)

FAILURES = []


def check(label, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' - ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(label)


def month_end(month_idx):
    yyyymm = int(month_idx_to_yyyymm(np.array([month_idx]))[0])
    return pd.Timestamp(year=yyyymm // 100, month=yyyymm % 100, day=1) + pd.offsets.MonthEnd(0)


def make_orders(seed=0, n=4000):
    """Order lines with BOTH grains present and mutually consistent - EntryMonthIdx is derived
    from EntryDate exactly as load_io derives it, which is what makes the equivalence meaningful.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp('2024-01-01')
    entry_date = start + pd.to_timedelta(rng.integers(0, 730, n), unit='D')
    entry_midx = to_month_idx((entry_date.year * 100 + entry_date.month).to_numpy())
    # request month >= entry month, mirroring load_io's negative-lead drop
    request_midx = entry_midx + rng.integers(0, 7, n)
    return pd.DataFrame({
        'SalesModelCode': rng.choice(list('ABCDE'), n),
        'RequestMonthIdx': request_midx.astype('int32'),
        'EntryMonthIdx': entry_midx.astype('int32'),
        'EntryDate': entry_date,
        'Qty': rng.integers(1, 500, n).astype('float32'),
    })


orders = make_orders()
book_m = build_order_book(orders, ['SalesModelCode'])                      # default arrival_col
book_d = build_order_book(orders, ['SalesModelCode'], arrival_col='EntryDate')
print(f"orders={len(orders)}  month-book rows={len(book_m)}  day-book rows={len(book_d)}")

# Query grid: every (group, request month, as-of month) combo in range.
months = sorted(orders['EntryMonthIdx'].unique())
grid = pd.DataFrame([(g, r, c) for g in list('ABCDE')
                     for r in sorted(orders['RequestMonthIdx'].unique())[:18]
                     for c in months[::3]],
                    columns=['SalesModelCode', 'RequestMonthIdx', 'CutoffMonthIdx'])
print(f"query grid rows={len(grid)}\n")

print('1. month-end equivalence (the core claim)')
res_m = query_known_order(grid, book_m, ['SalesModelCode'])

grid_d = grid.copy()
grid_d['CutoffMonthIdx'] = [month_end(c) for c in grid['CutoffMonthIdx']]
res_d = query_known_order(grid_d, book_d, ['SalesModelCode'], arrival_col='EntryDate')

both_nan = np.isnan(res_m) & np.isnan(res_d)
equal = both_nan | (res_m == res_d)
check('day-grain @ month-end == month-grain @ month-ordinal',
      equal.all(), f'{(~equal).sum()} of {len(equal)} rows differ')
check('NaN pattern identical (warm-up gaps line up)',
      (np.isnan(res_m) == np.isnan(res_d)).all())
check('grid is non-trivial (some non-NaN, some NaN)',
      np.isfinite(res_m).any() and np.isnan(res_m).any(),
      f'finite={np.isfinite(res_m).sum()} nan={np.isnan(res_m).sum()}')

print('\n2. mid-month as-of actually resolves finer (not just re-deriving the month)')
grid_mid = grid.copy()
grid_mid['CutoffMonthIdx'] = (pd.to_datetime([month_end(c) for c in grid['CutoffMonthIdx']])
                             - pd.Timedelta(days=20))
res_mid = query_known_order(grid_mid, book_d, ['SalesModelCode'], arrival_col='EntryDate')
finite = np.isfinite(res_m) & np.isfinite(res_mid)
check('mid-month <= month-end everywhere (cumulative is monotone)',
      (res_mid[finite] <= res_m[finite]).all())
check('mid-month is STRICTLY smaller somewhere (finer resolution is real)',
      (res_mid[finite] < res_m[finite]).any(),
      'day grain returned the same value everywhere - not resolving intra-month')

print('\n3. backward compatibility - every existing call site is 3 positional args')
res_compat = query_known_order(grid, book_m, ['SalesModelCode'])
check('query_known_order(q, book, group_cols) unchanged', np.array_equal(
    np.nan_to_num(res_compat, nan=-1), np.nan_to_num(res_m, nan=-1)))
book_default = build_order_book(orders, ['SalesModelCode'])
check('build_order_book(orders, group_cols) still month-grain',
      'EntryMonthIdx' in book_default.columns and book_default['EntryMonthIdx'].dtype.kind in 'iu')
check('book column set unchanged', set(book_default.columns) ==
      {'SalesModelCode', 'RequestMonthIdx', 'EntryMonthIdx', 'Qty', 'CumQty'},
      str(sorted(book_default.columns)))

print('\n4. grain guard - the hazard that would silently return KnownOrder=0')
try:
    query_known_order(grid, book_d, ['SalesModelCode'], arrival_col='EntryDate')
    check('month ordinals vs day book raises', False, 'no exception')
except ValueError as e:
    check('month ordinals vs day book raises ValueError', 'grain mismatch' in str(e))
try:
    query_known_order(grid_d, book_m, ['SalesModelCode'])
    check('dates vs month book raises', False, 'no exception')
except ValueError as e:
    check('dates vs month book raises ValueError', 'grain mismatch' in str(e))
# and confirm the matched pairs do NOT raise
try:
    _check_grain(grid, book_m, 'EntryMonthIdx', 'CutoffMonthIdx')
    _check_grain(grid_d, book_d, 'EntryDate', 'CutoffMonthIdx')
    check('matched grains pass the guard', True)
except ValueError as e:
    check('matched grains pass the guard', False, str(e))

print('\n5. shift_asof / asof_month_idx across both grains')
m_vals = np.array([24320, 24321], dtype='int64')
check('shift_asof month: plain subtraction', list(shift_asof(m_vals, 3)) == [24317, 24318])
d_vals = pd.to_datetime(['2026-03-31', '2026-08-15']).to_numpy()
shifted = pd.to_datetime(shift_asof(d_vals, 1))
check('shift_asof day: same-day-of-month, clamped (31 Mar -> 28 Feb)',
      list(shifted.strftime('%Y-%m-%d')) == ['2026-02-28', '2026-07-15'],
      str(list(shifted.strftime('%Y-%m-%d'))))
check('asof_month_idx month: identity', list(asof_month_idx(m_vals)) == [24320, 24321])
check('asof_month_idx day: date -> ordinal',
      list(asof_month_idx(d_vals)) == [int(to_month_idx(202603)), int(to_month_idx(202608))],
      str(list(asof_month_idx(d_vals))))
# pd.to_datetime(...).to_numpy(), NOT np.array([Timestamp]) - the latter builds an OBJECT-dtype
# array, which _is_day_grain correctly rejects as neither datetime64 nor an ordinal. Real callers
# always pass a DataFrame column or shift_asof output, both of which are datetime64.
check('asof_month_idx(day) round-trips to the month the date falls in',
      int(asof_month_idx(pd.to_datetime([month_end(24320)]).to_numpy())[0]) == 24320)

print('\n6. module graph still imports')
try:
    import adj_forecast_lib.features          # noqa: F401
    import adj_forecast_lib.country_visibility  # noqa: F401
    import adj_forecast_lib.lifecycle_features  # noqa: F401
    check('features / country_visibility / lifecycle_features import', True)
except Exception as e:
    check('features / country_visibility / lifecycle_features import', False, f'{type(e).__name__}: {e}')

print('\n' + ('=' * 60))
print(f"{len(FAILURES)} FAILURE(S): {FAILURES}" if FAILURES else "ALL CHECKS PASSED")
sys.exit(1 if FAILURES else 0)
