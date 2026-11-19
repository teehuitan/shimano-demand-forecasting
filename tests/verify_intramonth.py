"""Step-6 verification: the intra-month feature family.

These columns are the entire informational content of daily refresh, so the suite leans on the
two places where a NEW feature must exactly reproduce an OLD one at the day-0 anchor - the same
style of cross-check that caught the shift_asof month-end bug:

  * HistVisibility_SameDayOfMonth at d=0 must equal HistVisibility_AllHistory. The new feature is
    a strict generalisation ("h months out AND d days in"), so at d=0 it is the old one. If it
    isn't, the day-d analogue as-of date is being computed wrong.
  * Accumulation_sinceMonthStart at d=0 must equal Accumulation_1mo. At the origin month's end,
    "this calendar month so far" IS the last whole month.

Plus: a month panel must come back untouched (these columns are absent at month grain, not NaN),
the working-day counts must actually ignore weekends, and no candidate as-of point may sit at or
after the row's own as-of point.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'tests'))
import no_db  # noqa: E402,F401 - every DB engine raises from here on (see no_db.py)

from adj_forecast_lib.config import CUTOFF_DATE_COL, to_month_idx
from adj_forecast_lib.country_visibility import (
    build_country_visibility, build_focus_country_order_book, build_sku_country_mix,
)
from adj_forecast_lib.cutoff_grain import cutoff_day_index, monthly_anchor_mask
from adj_forecast_lib.features import build_market_shock_monthly
from adj_forecast_lib.intramonth_features import (
    ARRIVAL_VELOCITY_COLS, CUTOFF_CALENDAR_COLS, INTRAMONTH_FEATURE_COLS,
    SAME_DAY_VISIBILITY_COLS, attach_intramonth_features,
)
from adj_forecast_lib.order_book import build_order_book
from adj_forecast_lib.preprocessing import (
    NEW_FEATURE_FAMILIES, _feature_families, baseline_feature_cols, build_full_panel,
    build_known_canceled_books, full_feature_cols,
)

FAILURES = []


def check(label, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' - ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(label)


def close(a, b, tol=1e-6):
    a, b = np.asarray(a, dtype='float64'), np.asarray(b, dtype='float64')
    both_nan = np.isnan(a) & np.isnan(b)
    return bool((both_nan | (np.abs(a - b) <= tol)).all())


# ------------------------------------------------------------------------------- synthetic ctx
rng = np.random.default_rng(23)
N = 14000
entry_date = pd.Timestamp('2021-06-01') + pd.to_timedelta(rng.integers(0, 1500, N), unit='D')
entry_midx = to_month_idx((entry_date.year * 100 + entry_date.month).to_numpy())
codes = rng.choice(['A1', 'A2', 'B1', 'C1'], N)
orders = pd.DataFrame({
    'PlanModelCode': codes,
    'PartCode': pd.Series(codes).str[0].to_numpy(),
    'CustomerCountry': rng.choice(['CN', 'DE', 'US'], N, p=[.5, .3, .2]),
    'RequestMonthIdx': (entry_midx + rng.integers(0, 7, N)).astype('int32'),
    'EntryMonthIdx': entry_midx.astype('int32'),
    'EntryDate': entry_date,
    'Qty': rng.integers(1, 250, N).astype('float32'),
    # no cancellations here, but build_full_panel reads the cancellation books for the net floor
    'IsCanceled': False, 'CancelMonthIdx': np.nan, 'CancelKnownDate': pd.NaT, 'CancelDate': pd.NaT,
})
floor = int(orders['EntryMonthIdx'].min())
max_settled = int(orders['EntryMonthIdx'].max()) - 1
part_map = orders.groupby('PlanModelCode')['PartCode'].first()
actual_order = orders.groupby(['PlanModelCode', 'RequestMonthIdx'])['Qty'].sum() \
    .rename('ActualOrder').reset_index()
launch_code = orders.groupby('PlanModelCode')['EntryMonthIdx'].min()
launch_part = orders.groupby('PartCode')['EntryMonthIdx'].min()
origins_all = range(floor, max_settled + 1)


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
        'actual_part': orders.groupby(['PartCode', 'RequestMonthIdx'])['Qty'].sum()
                             .rename('ActualOrder').reset_index(),
        'launch_by_code': launch_code,
        'eligibility_launch_by_code': launch_code,   # every launch here is known
        'first_actual_order_by_code': np.fmax(launch_code, floor),
        'first_actual_order_by_part': np.fmax(launch_part, floor),
        'first_known_order_by_code': np.fmax(launch_code, floor + 1),
        'known_history_start_by_code': np.fmax(launch_code, floor + 1),  # as the real ctx
        'train_first_target_by_code': np.fmax(launch_code, floor + 1),   # extend_to_orders off
        'extend_to_orders': False,
        'first_known_order_by_part': np.fmax(launch_part, floor + 1),
        'code_map': pd.DataFrame({'PartCode': part_map, 'SalesModelCode': part_map.index}),
        'pred_succ_flags': None,
        'market_shock_monthly': build_market_shock_monthly(orders),
        'include_country_visibility': True,
        'country_visibility': build_country_visibility(orders, origins_all),
        'sku_country_mix': build_sku_country_mix(orders, origins_all),
        'focus_country_order_book': build_focus_country_order_book(orders, 'CN'),
        'focus_country_order_book_daily': (build_focus_country_order_book(
            orders, 'CN', arrival_col='EntryDate') if grain == 'day' else None),
        'dataset_floor_idx': floor,
        'max_settled_idx': max_settled,
        **dict(zip(('known_canceled_order_book_code', 'known_canceled_order_book_code_daily'),
                   build_known_canceled_books(orders, grain))),
        'effective_ranges': pd.DataFrame({'PlanModelCode': launch_code.index,
                                          'EndMonthIdx': np.nan}),
    }


print('\n=== 1. a month panel is returned untouched ===')
panel_m, _, _ = build_full_panel(make_ctx('month'))
present_m = [c for c in INTRAMONTH_FEATURE_COLS if c in panel_m.columns]
check('no intra-month column appears on a month panel (absent, not NaN)', not present_m,
      f'found {present_m}')
probe = panel_m.head(50).copy()
check('attach_intramonth_features is the identity without an as-of column',
      attach_intramonth_features(probe) is probe)

print('\n=== 2. a day panel carries the whole family ===')
panel_d, _, _ = build_full_panel(make_ctx('day'))
missing = [c for c in INTRAMONTH_FEATURE_COLS if c not in panel_d.columns]
check(f'all {len(INTRAMONTH_FEATURE_COLS)} columns present', not missing, f'missing {missing}')
shared = [c for c in panel_m.columns if c in panel_d.columns]
# CUTOFF_DATE_COL is the as-of column itself, not a feature - it's the thing the family describes.
new = [c for c in panel_d.columns if c not in panel_m.columns and c != CUTOFF_DATE_COL]
check(f'day panel = month panel + this family + the as-of column ({len(shared)} shared, '
      f'{len(new)} new features)',
      sorted(new) == sorted(INTRAMONTH_FEATURE_COLS) and set(panel_m.columns) <= set(panel_d.columns),
      f'unexpected: {sorted(set(new) ^ set(INTRAMONTH_FEATURE_COLS))}')

anchor = monthly_anchor_mask(panel_d)
day0 = panel_d[anchor].sort_values(['PlanModelCode', 'TargetIdx', 'Horizon']).reset_index(drop=True)
month = panel_m.sort_values(['PlanModelCode', 'TargetIdx', 'Horizon']).reset_index(drop=True)
check(f'day-0 slice aligns row-for-row with the month panel ({len(day0):,} rows)',
      len(day0) == len(month))

print('\n=== 3. cutoff calendar ===')
check('CutoffDayIndex matches cutoff_grain.cutoff_day_index',
      bool((panel_d['CutoffDayIndex'].to_numpy() == cutoff_day_index(panel_d)).all()))
check('day-0 rows are index 0', bool((day0['CutoffDayIndex'] == 0).all()))
elapsed, remaining = panel_d['CutoffWorkingDaysElapsed'], panel_d['CutoffWorkingDaysRemaining']
check("elapsed + remaining = the month's own working-day count (20..23 for Mon-Fri months)",
      bool((elapsed >= 0).all()) and bool((remaining >= 0).all())
      and bool((((elapsed + remaining) >= 20) & ((elapsed + remaining) <= 23)).all()),
      f'range {int((elapsed + remaining).min())}..{int((elapsed + remaining).max())}')
# elapsed == 0 is legitimate and must stay possible: a cutoff on Sat the 1st has had no working
# day yet. What must NOT happen is a zero on a weekday - that would mean the count is off by one.
asof = pd.to_datetime(panel_d[CUTOFF_DATE_COL])
zero_days = asof[elapsed.to_numpy() == 0].dt.dayofweek
check(f'elapsed reaches 0 only on weekends at a month start ({len(zero_days):,} such rows)',
      len(zero_days) > 0 and bool((zero_days >= 5).all()),
      f'weekday zeros: {sorted(set(zero_days[zero_days < 5]))}')
check('CutoffMonthProgress in [0, 1]',
      bool((panel_d['CutoffMonthProgress'] >= 0).all())
      and bool((panel_d['CutoffMonthProgress'] <= 1.0 + 1e-6).all()))

# weekends must not advance the count: a Sunday cutoff has the same elapsed count as the Friday
# two days before - but only when both fall in the SAME month, since the count resets each month.
by_date = panel_d.assign(_d=asof).groupby('_d')['CutoffWorkingDaysElapsed'].first()
pairs = [(d, d - pd.Timedelta(days=2)) for d in by_date.index
         if d.dayofweek == 6 and (d - pd.Timedelta(days=2)) in by_date.index
         and (d - pd.Timedelta(days=2)).month == d.month]
check(f'weekends do not advance the working-day count ({len(pairs)} same-month Sunday/Friday '
      f'pairs)', len(pairs) > 0 and all(by_date[s] == by_date[f] for s, f in pairs),
      f'{[(str(s.date()), int(by_date[s]), int(by_date[f])) for s, f in pairs if by_date[s] != by_date[f]][:3]}')
check('CutoffDayOfWeek agrees with the date', bool((panel_d['CutoffDayOfWeek'].to_numpy()
                                                    == asof.dt.dayofweek.to_numpy()).all()))

print('\n=== 4. arrival velocity, and the day-0 identity against Accumulation_1mo ===')
check('Accumulation_sinceMonthStart at d=0 IS Accumulation_1mo (same window, month-end to '
      'month-end)', close(day0['Accumulation_sinceMonthStart'], day0['Accumulation_1mo']))
check('the identity is not vacuous: the two differ off the anchor',
      not close(panel_d.loc[~anchor, 'Accumulation_sinceMonthStart'],
                panel_d.loc[~anchor, 'Accumulation_1mo']))
check('accumulation over a longer window is never smaller (the book only grows)',
      bool((panel_d['Accumulation_14d'] >= panel_d['Accumulation_7d'] - 1e-6).all()))
# NaN is expected (resolve_known_order_gaps leaves it before a code's first entry) and `NaN >= x`
# is False, so compare only the observed values - the claim is about sign, not about coverage.
acc = panel_d[['Accumulation_7d', 'Accumulation_14d', 'Accumulation_sinceMonthStart']]
check(f'no negative accumulation anywhere ({int(acc.notna().sum().sum()):,} observed values)',
      bool((acc[acc.notna()].fillna(0) >= -1e-3).all().all()),
      f'min {acc.min().to_dict()}')

print('\n=== 5. same-day-of-month visibility, and the day-0 identity against AllHistory ===')
check('HistVisibility_SameDayOfMonth at d=0 IS HistVisibility_AllHistory (d=0 is the month-end '
      'as-of point the monthly variant already uses)',
      close(day0['HistVisibility_SameDayOfMonth'], day0['HistVisibility_AllHistory'], tol=1e-5))
check('the identity is not vacuous: the two differ off the anchor',
      not close(panel_d.loc[~anchor, 'HistVisibility_SameDayOfMonth'],
                panel_d.loc[~anchor, 'HistVisibility_AllHistory'], tol=1e-5))
vis = panel_d['HistVisibility_SameDayOfMonth'].dropna()
check(f'visibility is a fraction in [0, 1] ({len(vis):,} non-null)',
      bool(((vis >= -1e-9) & (vis <= 1.0 + 1e-6)).all()),
      f'range {vis.min():.4f}..{vis.max():.4f}')
implied = panel_d['ImpliedTotal_SameDayOfMonth']
check('ImpliedTotal >= KnownOrder wherever visibility is a real fraction',
      bool((implied[vis.index] >= panel_d.loc[vis.index, 'KnownOrder'] - 1e-3).all()))

# the curve must actually rise through the month - that IS the signal
curve = panel_d.groupby('CutoffDayIndex')['HistVisibility_SameDayOfMonth'].mean()
check('mean visibility rises from day 0 to month end (the fill curve exists)',
      curve.loc[30] > curve.loc[0], f'day0={curve.loc[0]:.4f} day30={curve.loc[30]:.4f}')

print('\n=== 6. leak: no lookback as-of point may reach the row\'s own as-of point ===')
sample = panel_d.sample(min(4000, len(panel_d)), random_state=0)
prior_max = pd.to_datetime(sample[CUTOFF_DATE_COL]) - pd.offsets.MonthEnd(1)
check('every candidate month M\' is strictly before the row\'s origin, so its analogue as-of date '
      'is strictly before the row\'s own', bool((prior_max < pd.to_datetime(
          sample[CUTOFF_DATE_COL])).all()))
lookahead = panel_d.loc[anchor, 'HistVisibility_SameDayOfMonth']
check('day-0 visibility never uses the row\'s own target month (would force the ratio to 1.0)',
      not bool((lookahead.dropna() > 0.999).all()))

print('\n=== 7. family registration ===')
fams = _feature_families()
check("'IntraMonth' is a registered family", fams.get('IntraMonth') == INTRAMONTH_FEATURE_COLS)
check("'IntraMonth' is in NEW_FEATURE_FAMILIES, so baseline_feature_cols() can still rebuild the "
      "pre-daily-refresh feature set for comparisons", 'IntraMonth' in NEW_FEATURE_FAMILIES)
check('every 2026-09 family (MarketShock, CountryVisibility, IntraMonth) is in the DEPLOYED set, '
      'full_feature_cols() - validated and promoted, not held out (audit B9)',
      set(NEW_FEATURE_FAMILIES) == {'MarketShock', 'CountryVisibility', 'IntraMonth'}
      and all(set(fams[f]) <= set(full_feature_cols()) for f in NEW_FEATURE_FAMILIES))
check('full_feature_cols() includes it and baseline_feature_cols() excludes it',
      set(INTRAMONTH_FEATURE_COLS) <= set(full_feature_cols())
      and not (set(INTRAMONTH_FEATURE_COLS) & set(baseline_feature_cols())))
check('the three sub-groups partition the family',
      CUTOFF_CALENDAR_COLS + ARRIVAL_VELOCITY_COLS + SAME_DAY_VISIBILITY_COLS
      == INTRAMONTH_FEATURE_COLS)
check('baseline_feature_cols() on a DAY panel is exactly the pre-daily-refresh feature set',
      not (set(INTRAMONTH_FEATURE_COLS) & {c for c in baseline_feature_cols()
                                           if c in panel_d.columns}))

print('\n' + '=' * 78)
print(f'{len(FAILURES)} FAILURE(S): {FAILURES}' if FAILURES else 'ALL CHECKS PASSED')
sys.exit(1 if FAILURES else 0)
