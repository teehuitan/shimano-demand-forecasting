"""Edit-4b verification: origin-day fan-out in the training skeleton + a dated inference skeleton.

Three things matter here:
  * the fan-out - each origin gets every date from its own month end to the day before the next
    month's end (that date is the next origin's day 0), so no date belongs to two origins;
  * the duplicate-index landmine in AdjTrailMean/ManualSim must be DEFUSED, not merely avoided
    by luck - their lookup is indexed on (code, OriginIdx), which a fanned panel repeats;
  * the month-end slice of a day panel must still reproduce the monthly panel's KnownOrder.
"""
import pathlib
import re
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'tests'))
import no_db  # noqa: E402,F401 - every DB engine raises from here on (see no_db.py)

from adj_forecast_lib.config import (
    CUTOFF_DATE_COL, month_idx_to_yyyymm, to_month_idx,
)
from adj_forecast_lib.country_visibility import (
    build_country_visibility, build_focus_country_order_book, build_sku_country_mix,
)
from adj_forecast_lib.features import build_market_shock_monthly
from adj_forecast_lib.intramonth_features import INTRAMONTH_FEATURE_COLS
from adj_forecast_lib.order_book import build_order_book, frame_cutoff_grain
from adj_forecast_lib.preprocessing import (
    _expand_to_cutoff_days, _lifespan_skeleton, monthly_anchor_mask, build_full_panel,
    build_inference_skeleton, build_known_canceled_books, cutoff_dates_for_origins, cutoff_window,
)

FAILURES = []


def check(label, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' - ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(label)


def col_same(a, b):
    a, b = np.asarray(a, dtype='float64'), np.asarray(b, dtype='float64')
    return bool(((np.isnan(a) & np.isnan(b)) | (a == b)).all())


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
    # no cancellations in this suite - their maths is verify_net_orders' job - but the columns a
    # real load_io row carries must be there for the cancellation books to build
    'IsCanceled': False, 'CancelMonthIdx': np.nan, 'CancelKnownDate': pd.NaT,
    'CancelDate': pd.NaT,
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
        # a source file dated at the end of the last entry month: data current through then
        'data_asof': orders['EntryDate'].max() + pd.offsets.MonthEnd(0),
        **dict(zip(('known_canceled_order_book_code', 'known_canceled_order_book_code_daily'),
                   build_known_canceled_books(orders, grain))),
        'effective_ranges': pd.DataFrame({'PlanModelCode': launch_code.index,
                                          'EndMonthIdx': np.nan}),
    }


print('1. cutoff_dates_for_origins')
feb_origin = int(to_month_idx(202601)) - 0   # origin Jan 2026 -> cutoff month Feb 2026 (28 days)
o = np.array([feb_origin])
check('lands in the month AFTER the origin',
      pd.Timestamp(cutoff_dates_for_origins(o, 1)[0]) == pd.Timestamp('2026-02-01'))
check('day=0 is the origin month last day',
      pd.Timestamp(cutoff_dates_for_origins(o, 0)[0]) == pd.Timestamp("2026-01-31"))
check('day 31 clamps into a 28-day February',
      pd.Timestamp(cutoff_dates_for_origins(o, 31)[0]) == pd.Timestamp('2026-02-28'))
jan_origin = int(to_month_idx(202512))
check('day 31 is a real 31st in a 31-day month',
      pd.Timestamp(cutoff_dates_for_origins(np.array([jan_origin]), 31)[0])
      == pd.Timestamp('2026-01-31'))

print('\n2. _expand_to_cutoff_days follows the real calendar')
base = pd.DataFrame({'PlanModelCode': ['A1', 'A1'], 'TargetIdx': [feb_origin + 1, jan_origin + 1],
                     'Horizon': [1, 1], 'OriginIdx': [feb_origin, jan_origin], 'PartCode': ['A', 'A']})
exp = _expand_to_cutoff_days(base)
feb_rows = exp[exp['OriginIdx'] == feb_origin]
jan_rows = exp[exp['OriginIdx'] == jan_origin]
check('February yields 27 in-month dates + 1 anchor (Feb 28 is the next origin\'s day 0)',
      len(feb_rows) == 28, str(len(feb_rows)))
check('January yields 30 + 1 anchor', len(jan_rows) == 31, str(len(jan_rows)))
check('no duplicate (code, target, horizon, cutoff)',
      not exp.duplicated(['PlanModelCode', 'TargetIdx', 'Horizon', CUTOFF_DATE_COL]).any())
check('exactly one month-end row per origin',
      monthly_anchor_mask(exp).sum() == 2, str(monthly_anchor_mask(exp).sum()))
check('frame reads as day grain', frame_cutoff_grain(exp) == 'day')
jan_last = pd.to_datetime(jan_rows[CUTOFF_DATE_COL]).max()
check('an origin\'s window stops the day before the next month\'s end - Jan 31 belongs to the '
      'Jan origin only', jan_last == pd.Timestamp('2026-01-30'), str(jan_last))
check('consecutive origins never share an as-of date',
      set(cutoff_window(jan_origin)).isdisjoint(cutoff_window(jan_origin + 1)))

print('\n3. _lifespan_skeleton fans out only under day grain')
sk_m = _lifespan_skeleton(make_ctx('month'))
sk_d = _lifespan_skeleton(make_ctx('day'))
check('month skeleton has no cutoff column', CUTOFF_DATE_COL not in sk_m.columns)
check('day skeleton has one', CUTOFF_DATE_COL in sk_d.columns)
ratio = len(sk_d) / len(sk_m)
check(f'row count grew ~31x (actual {ratio:.1f}x: {len(sk_m):,} -> {len(sk_d):,})',
      30 <= ratio <= 32)
check('day skeleton month-end slice == month skeleton row count',
      int(monthly_anchor_mask(sk_d).sum()) == len(sk_m),
      f'{int(monthly_anchor_mask(sk_d).sum())} vs {len(sk_m)}')

print('\n4. build_full_panel under day grain - the landmine')
panel_m = build_full_panel(make_ctx('month'))[0]
try:
    panel_d, _fam, _cols = build_full_panel(make_ctx('day'))
    check('AdjTrailMean/ManualSim self-join does NOT raise on the fanned panel', True)
except Exception as e:
    check('AdjTrailMean/ManualSim self-join does NOT raise on the fanned panel', False,
          f'{type(e).__name__}: {str(e)[:150]}')
    panel_d = None

if panel_d is not None:
    # Since step 6, also the intra-month family - day-grain-only columns by construction.
    check('day panel == month panel columns + cutoff + the intra-month family',
          set(panel_d.columns) - set(panel_m.columns)
          == {CUTOFF_DATE_COL} | set(INTRAMONTH_FEATURE_COLS),
          str(set(panel_d.columns) ^ set(panel_m.columns)
              - {CUTOFF_DATE_COL} - set(INTRAMONTH_FEATURE_COLS)))
    me = panel_d[monthly_anchor_mask(panel_d)].sort_values(
        ['PlanModelCode', 'TargetIdx', 'Horizon']).reset_index(drop=True)
    ref = panel_m.sort_values(['PlanModelCode', 'TargetIdx', 'Horizon']).reset_index(drop=True)
    check('month-end slice has the same row count as the monthly panel', len(me) == len(ref),
          f'{len(me)} vs {len(ref)}')
    if len(me) == len(ref):
        check('KnownOrder identical on the month-end slice',
              col_same(me['KnownOrder'], ref['KnownOrder']))
        check('Adjustment identical on the month-end slice',
              col_same(me['Adjustment'], ref['Adjustment']))
        check('AdjTrailMean12 identical (month-grain baseline, read off month-end history)',
              col_same(me['AdjTrailMean12'], ref['AdjTrailMean12']))
        check('ManualSim12 identical', col_same(me['ManualSim12'], ref['ManualSim12']))
        check('Lag0 identical', col_same(me['Lag0'], ref['Lag0']))
    # the point of the exercise: mid-month rows must actually differ
    mid = panel_d[~monthly_anchor_mask(panel_d)]
    check('mid-month rows exist and their KnownOrder varies within an origin',
          len(mid) > 0 and mid.groupby(['PlanModelCode', 'TargetIdx', 'Horizon'])['KnownOrder']
                              .nunique().max() > 1)

print('\n5. build_inference_skeleton date handling')
ctx_d = make_ctx('day')
ok = False
try:
    build_inference_skeleton(ctx_d, max_settled, [1, 3], min_origin_months_since_launch=None)
except ValueError as e:
    ok = 'no cutoff_date' in str(e)
check('day grain without a cutoff_date raises', ok)
ok = False
try:
    build_inference_skeleton(ctx_d, max_settled, [1, 3], cutoff_date='2000-01-15',
                             min_origin_months_since_launch=None)
except ValueError as e:
    ok = 'outside the valid as-of window' in str(e)
check('a cutoff_date outside the valid window raises', ok)
good_date = pd.Timestamp(cutoff_dates_for_origins(np.array([max_settled]), 12)[0])
inf_d = build_inference_skeleton(ctx_d, max_settled, [1, 3], cutoff_date=good_date,
                                 min_origin_months_since_launch=None)
check('valid cutoff_date builds a day-grain skeleton',
      frame_cutoff_grain(inf_d) == 'day' and (inf_d[CUTOFF_DATE_COL] == good_date).all())
inf_m = build_inference_skeleton(make_ctx('month'), max_settled, [1, 3],
                                 min_origin_months_since_launch=None)
check('month grain still needs no date and stays month grain',
      frame_cutoff_grain(inf_m) == 'month')

print('\n6. public entry points thread the grain')
import inspect
from adj_forecast_lib.preprocessing import preprocess_for_inference, preprocess_for_training
required = lambda prm: (prm.kind is inspect.Parameter.KEYWORD_ONLY
                        and prm.default is inspect.Parameter.empty)
check('preprocess_for_training(*, cutoff_grain) - required, no month default',
      required(inspect.signature(preprocess_for_training).parameters['cutoff_grain']))
p = inspect.signature(preprocess_for_inference).parameters
check('preprocess_for_inference(*, cutoff_grain, cutoff_date=None) - grain required',
      required(p['cutoff_grain']) and p['cutoff_date'].default is None)

# The CLI is how a panel actually gets built, so the grain has to reach it too - and a day panel
# must not be able to land on a month panel's filename. They differ in row count and in columns,
# and a model trained on one and scored on the other is caught by the registry grain check; letting
# them share a name would turn that loud error into a silently overwritten parquet.
from adj_forecast_lib.preprocessing import _parse_args
_argv = sys.argv
try:
    base = ['x', '--type', 'train', '--plant', '3300', '--month-range', '201701-202608']
    sys.argv = base
    check('CLI defaults to day grain (the deployed grain)', _parse_args().cutoff_grain == 'day')
    sys.argv = base + ['--cutoff-grain', 'month']
    check('CLI accepts --cutoff-grain month', _parse_args().cutoff_grain == 'month')
    sys.argv = base + ['--cutoff-grain', 'day']
    check('CLI accepts --cutoff-grain day', _parse_args().cutoff_grain == 'day')
    try:
        sys.argv = base + ['--cutoff-grain', 'weekly']
        _parse_args()
        check('CLI rejects an unknown grain', False, 'accepted')
    except SystemExit:
        check('CLI rejects an unknown grain', True)
finally:
    sys.argv = _argv

src = (_REPO_ROOT / 'adj_forecast_lib' / 'preprocessing.py').read_text(encoding='utf-8')
check("day panels are written under a distinct filename, never over a month panel's",
      "suffix = '_day' if args.cutoff_grain == 'day' else ''" in src
      # any panel version number - only the {suffix} placement matters
      and re.search(r'train\d+_\{args\.month_range\}\{suffix\}', src) is not None
      and re.search(r'inf\d+_\{args\.origin\}\{suffix\}', src) is not None)

print('\n' + '=' * 66)
print(f"{len(FAILURES)} FAILURE(S): {FAILURES}" if FAILURES else 'ALL CHECKS PASSED')
sys.exit(1 if FAILURES else 0)
