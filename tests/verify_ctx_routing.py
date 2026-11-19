"""Edit-4a verification: dual books in ctx + per-family routing in _attach_all_features.

The decisive check is SCOPE. Running the identical skeleton through both grains (with the day
as-of pinned to month-end) must leave every non-day-sensitive column bit-identical - Lag*,
Group_*, ADI, trend, HistVisibility*, lifecycle, calendar, market shock. If any of those moved,
routing leaked into a family that should have stayed monthly.

Synthetic ctx, assembled with the real builders, so no Fabric login is needed.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'tests'))
import no_db  # noqa: E402,F401 - every DB engine raises from here on (see no_db.py)

from adj_forecast_lib.config import CUTOFF_DATE_COL, month_idx_to_yyyymm, to_month_idx
from adj_forecast_lib.country_visibility import (
    build_country_visibility, build_focus_country_order_book, build_sku_country_mix,
)
from adj_forecast_lib.features import build_market_shock_monthly
from adj_forecast_lib.intramonth_features import INTRAMONTH_FEATURE_COLS
from adj_forecast_lib.order_book import build_order_book
from adj_forecast_lib.preprocessing import _attach_all_features, _day_grain_routing

FAILURES = []


def check(label, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' - ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(label)


def month_end(month_idx):
    yyyymm = int(month_idx_to_yyyymm(np.array([month_idx]))[0])
    return pd.Timestamp(year=yyyymm // 100, month=yyyymm % 100, day=1) + pd.offsets.MonthEnd(0)


def col_same(a, b):
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    if a.dtype.kind in 'fiu' and b.dtype.kind in 'fiu':
        av, bv = a.to_numpy('float64'), b.to_numpy('float64')
        return bool(((np.isnan(av) & np.isnan(bv)) | (av == bv)).all())
    return bool(a.equals(b))


# ------------------------------------------------------------------- synthetic orders + ctx
rng = np.random.default_rng(11)
N = 12000
entry_date = pd.Timestamp('2021-06-01') + pd.to_timedelta(rng.integers(0, 1600, N), unit='D')
entry_midx = to_month_idx((entry_date.year * 100 + entry_date.month).to_numpy())
codes = rng.choice(['A1', 'A2', 'B1', 'B2', 'C1'], N)
orders = pd.DataFrame({
    'PlanModelCode': codes,
    'PartCode': pd.Series(codes).str[0].to_numpy(),
    'CustomerCountry': rng.choice(['CN', 'DE', 'US', 'JP'], N, p=[.45, .25, .2, .1]),
    'RequestMonthIdx': (entry_midx + rng.integers(0, 7, N)).astype('int32'),
    'EntryMonthIdx': entry_midx.astype('int32'),
    'EntryDate': entry_date,
    'Qty': rng.integers(1, 300, N).astype('float32'),
})

floor = int(orders['EntryMonthIdx'].min())
max_settled = int(orders['EntryMonthIdx'].max()) - 1
part_map = orders.groupby('PlanModelCode')['PartCode'].first()
actual_order = orders.groupby(['PlanModelCode', 'RequestMonthIdx'])['Qty'].sum() \
    .rename('ActualOrder').reset_index()
actual_part = orders.groupby(['PartCode', 'RequestMonthIdx'])['Qty'].sum() \
    .rename('ActualOrder').reset_index()
launch_by_code = orders.groupby('PlanModelCode')['EntryMonthIdx'].min()
launch_by_part = orders.groupby('PartCode')['EntryMonthIdx'].min()
panel_origins = range(floor, max_settled + 1)


def make_ctx(grain):
    return {
        'cutoff_grain': grain,
        'include_cny': True,
        'event_ranges': [(202001, 202012)],
        'order_book_code': build_order_book(orders, ['PlanModelCode']),
        'order_book_code_daily': (build_order_book(orders, ['PlanModelCode'],
                                                   arrival_col='EntryDate')
                                  if grain == 'day' else None),
        'order_book_part': build_order_book(orders, ['PartCode']),
        'actual_order': actual_order,
        'actual_part': actual_part,
        'launch_by_code': launch_by_code,
        'eligibility_launch_by_code': launch_by_code,   # every launch here is known
        'first_actual_order_by_code': np.fmax(launch_by_code, floor),
        'first_actual_order_by_part': np.fmax(launch_by_part, floor),
        'first_known_order_by_code': np.fmax(launch_by_code, floor + 1),
        'known_history_start_by_code': np.fmax(launch_by_code, floor + 1),  # as the real ctx
        'first_known_order_by_part': np.fmax(launch_by_part, floor + 1),
        'effective_ranges': pd.DataFrame({'PlanModelCode': launch_by_code.index,
                                          'StartMonthIdx': launch_by_code.to_numpy(),
                                          'EndMonthIdx': np.nan}),
        'part_code_map': part_map,
        'pred_succ_flags': None,
        'market_shock_monthly': build_market_shock_monthly(orders),
        'include_country_visibility': True,
        'country_visibility': build_country_visibility(orders, panel_origins),
        'sku_country_mix': build_sku_country_mix(orders, panel_origins),
        'focus_country_order_book': build_focus_country_order_book(orders, 'CN'),
        'focus_country_order_book_daily': (build_focus_country_order_book(
            orders, 'CN', arrival_col='EntryDate') if grain == 'day' else None),
    }


origins = list(range(floor + 26, max_settled - 6))
skeleton = pd.DataFrame([(c, o, h) for c in launch_by_code.index for o in origins for h in (1, 3, 6)],
                        columns=['PlanModelCode', 'OriginIdx', 'Horizon'])
skeleton['TargetIdx'] = skeleton['OriginIdx'] + skeleton['Horizon']
skeleton['PartCode'] = skeleton['PlanModelCode'].map(part_map)
print(f"orders={len(orders)}  skeleton={len(skeleton)}  origins={len(origins)}\n")

print('1. routing table')
ctx_m, ctx_d = make_ctx('month'), make_ctx('day')
r_m, r_d = _day_grain_routing(ctx_m), _day_grain_routing(ctx_d)
check('month routing == every function default',
      r_m['asof_col'] is None and r_m['arrival_col'] == 'EntryMonthIdx'
      and r_m['known_order_book'] is ctx_m['order_book_code'])
check('day routing picks the daily books',
      r_d['asof_col'] == CUTOFF_DATE_COL and r_d['arrival_col'] == 'EntryDate'
      and r_d['known_order_book'] is ctx_d['order_book_code_daily']
      and r_d['focus_order_book'] is ctx_d['focus_country_order_book_daily'])
try:
    _day_grain_routing({**ctx_m, 'cutoff_grain': 'day'})
    check("grain='day' with no daily book raises", False, 'no exception')
except ValueError as e:
    check("grain='day' with no daily book raises", 'order_book_code_daily' in str(e))
# It used to be read as month. That's a silent fallback: every builder records the grain, so a ctx
# without one was hand-assembled, and guessing its grain is how month rows end up in a day run.
try:
    _day_grain_routing({k: v for k, v in ctx_m.items() if k != 'cutoff_grain'})
    check('ctx with no cutoff_grain key raises instead of defaulting to month', False, 'no exception')
except ValueError as e:
    check('ctx with no cutoff_grain key raises instead of defaulting to month',
          "ctx['cutoff_grain']" in str(e))

print('\n2. both grains build a panel')
panel_m = _attach_all_features(skeleton, ctx_m)
sk_day = skeleton.copy()
sk_day[CUTOFF_DATE_COL] = [month_end(o) for o in skeleton['OriginIdx']]
panel_d = _attach_all_features(sk_day, ctx_d)
check('month panel built', len(panel_m) == len(skeleton))
check('day panel built', len(panel_d) == len(skeleton))
# Since step 6 the day panel also carries the intra-month family, whose columns exist ONLY at day
# grain. So the equivalence contract below is over the SHARED columns; the shape contract is that
# the difference is exactly the as-of column plus that family, and nothing else has crept in.
check('day panel is the month panel + the as-of column + the intra-month family',
      set(panel_d.columns) - set(panel_m.columns) == {CUTOFF_DATE_COL} | set(INTRAMONTH_FEATURE_COLS),
      str(set(panel_d.columns) ^ set(panel_m.columns)
          - {CUTOFF_DATE_COL} - set(INTRAMONTH_FEATURE_COLS)))

print('\n3. SCOPE - which columns actually moved (day pinned to month-end)')
shared = [c for c in panel_m.columns if c in panel_d.columns]
moved = [c for c in shared if not col_same(panel_m[c], panel_d[c])]
# KnownOrder itself and KnownCNShare must be IDENTICAL at month-end; the shifted ones legitimately
# differ where a same-day-of-month shift lands off a month-end (see step 3).
expected_identical = ['KnownOrder', 'KnownOrder_x_Horizon', 'KnownCNShare', 'CNShareGap',
                      'ImpliedAdjustment_Mix', 'BaselineCNShare', 'ExpectedVisibility']
allowed_to_move = {'KnownOrder_prevOrigin', 'Accumulation_1mo', 'AccumulationPct_1mo',
                   'KnownOrder_3moAgo', 'Momentum_3mo', 'Accumulation_3mo',
                   'KnownOrder_6moAgo', 'Momentum_6mo', 'Accumulation_6mo',
                   'KnownOrder_YoY_prior', 'KnownOrder_YoY'}
for c in expected_identical:
    check(f'{c} identical at month-end', c in shared and col_same(panel_m[c], panel_d[c]))
leaked = sorted(set(moved) - allowed_to_move)
check('NO non-day-sensitive family moved', not leaked, f'leaked: {leaked}')
print(f'         moved: {sorted(moved)}')
print(f'         unchanged: {len(shared) - len(moved)} of {len(shared)} columns')

print('\n4. mid-month as-of actually differs from month-end')
sk_mid = skeleton.copy()
sk_mid[CUTOFF_DATE_COL] = [month_end(o) - pd.Timedelta(days=17) for o in skeleton['OriginIdx']]
panel_mid = _attach_all_features(sk_mid, ctx_d)
check('KnownOrder mid-month <= month-end',
      (panel_mid['KnownOrder'].to_numpy() <= panel_d['KnownOrder'].to_numpy()).all())
check('KnownOrder mid-month strictly smaller somewhere',
      (panel_mid['KnownOrder'].to_numpy() < panel_d['KnownOrder'].to_numpy()).any())
lag_cols = [f'Lag{k}' for k in range(24)] + [f'Group_Lag{k}' for k in range(24)]
check('Lag/Group_Lag spine UNCHANGED even mid-month (closed months are final)',
      all(col_same(panel_m[c], panel_mid[c]) for c in lag_cols if c in shared))
check('HistVisibility_* unchanged mid-month',
      all(col_same(panel_m[c], panel_mid[c]) for c in shared if c.startswith('HistVisibility')
          or c.startswith('PartLevel_HistVisibility')))

print('\n5. build_order_book_context signature carries the grain')
import inspect
from adj_forecast_lib.preprocessing import build_order_book_context
sig = inspect.signature(build_order_book_context)
# Required since the net-order fix: every caller states the grain it builds for, so a context
# can't be month grain merely because nobody said otherwise.
check("cutoff_grain is keyword-only and required - no month default",
      sig.parameters['cutoff_grain'].kind is inspect.Parameter.KEYWORD_ONLY
      and sig.parameters['cutoff_grain'].default is inspect.Parameter.empty)

print('\n' + '=' * 64)
print(f"{len(FAILURES)} FAILURE(S): {FAILURES}" if FAILURES else 'ALL CHECKS PASSED')
sys.exit(1 if FAILURES else 0)
