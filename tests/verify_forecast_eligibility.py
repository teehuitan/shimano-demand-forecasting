"""Forecast eligibility (audit B4): at any origin, production and every backtest score the same rows.

What this replaced. min_origin_months_since_launch was tested as `OriginMonthsSinceLaunch >= n` at
six separate sites - inference plus five backtest populations. OriginMonthsSinceLaunch is NaN for a
code whose TFGMaterial record has no ProductStartDate, so every site silently dropped such codes. A
first fix then kept them in inference only ("unknown = mature"): production scored rows no backtest
ever saw, including a brand-new undated code it should have held back for its first three months.

Now there is one test, lifecycle_features.meets_min_launch_age, over one non-feature column,
ForecastEligibilityAge. Its launch is the TFGMaterial launch where there is one, otherwise the
code's first order; an undated code first ordered AT the dataset floor predates the data and is
always eligible. The model feature OriginMonthsSinceLaunch is untouched.

The scenario, on the synthetic feed, for every plant (min age 3), by PlanModelCode:
  PNB1    unknown legacy - no TFG start date, ordered since the dataset floor -> eligible everywhere
  PNNEW1  unknown new    - no TFG start date, first ordered 2025-11            -> eligible from 202602
  PNA2    known young    - TFG launch 2025-12                                   -> eligible from 202603
  rest  known mature   - TFG launches 2022-2023                               -> eligible everywhere

  1. Eligibility only: launch_by_code, OriginMonthsSinceLaunch and the skeleton are untouched, the
     column is in no feature list, and swapping the eligibility launch changes no other column.
  2. For origins 202510..202603, inference (from a feed complete only through that origin) and EACH
     backtest population - walk_forward_horizon, walk_forward_shared, test_rows_for_horizon,
     baseline_folds_for_horizon, all_settled_rows_for_horizon, seed_overall_results - hold exactly
     the same (code, target, horizon) rows, with the three codes in or out as listed. Then the same
     at day grain for one origin and cutoff date.
  3. The two rules this replaced disagree on those rows, so section 2 isn't passing vacuously.
  4. A panel saved before the column existed falls back to the old test, with one warning.
"""
import contextlib
import logging
import os
import pathlib
import sys
import tempfile
import warnings

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'tests'))
os.chdir(_REPO_ROOT)        # the STJ replacement workbook is cwd-relative
import no_db  # noqa: E402,F401 - every DB engine raises from here on (see no_db.py)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from synthetic_fabric import PMC_BY_SMC, FabricStub, make_orders, make_tfg  # noqa: E402
from adj_forecast_lib import folds as fl, lifecycle_features as lf, preprocessing as pp  # noqa: E402
from adj_forecast_lib.config import CUTOFF_DATE_COL, PLANT_NAMES, to_month_idx  # noqa: E402
from adj_forecast_lib.evaluation import seed_overall_results  # noqa: E402
from adj_forecast_lib.lifecycle_features import FORECAST_ELIGIBILITY_AGE_COL as ELIG  # noqa: E402

warnings.filterwarnings('ignore')
LIB = logging.getLogger('adj_forecast_lib')
LIB.addHandler(logging.NullHandler())   # keep the library's warnings off stderr outside captures
FAILURES = []


def check(label, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' - ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(label)


class _Records(logging.Handler):
    def __init__(self, level):
        super().__init__(level)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def captured(level=logging.WARNING):
    handler, old_level = _Records(level), LIB.level
    LIB.addHandler(handler)
    LIB.setLevel(min(level, logging.WARNING))
    try:
        yield handler.messages
    finally:
        LIB.removeHandler(handler)
        LIB.setLevel(old_level)


def logged(messages, *needles) -> bool:
    return any(all(n in m for n in needles) for m in messages)


# An unconfigured plant code can't run these sections yet: build_order_book_context indexes the
# per-plant config dicts directly (audit B15).
PLANTS = sorted(PLANT_NAMES)
MIN_AGE = 3
# YOUNG is A2, not C2: C2 shares PlanModelCode PNC with C1, whose earlier launch would hide its own.
LEGACY_SMC, NEW_SMC, YOUNG_SMC = 'B1', 'NEW1', 'A2'
LEGACY, NEW, YOUNG = PMC_BY_SMC[LEGACY_SMC], 'PNNEW1', PMC_BY_SMC[YOUNG_SMC]
NEW_FIRST_ENTRY, YOUNG_LAUNCH = pd.Timestamp('2025-11-01'), pd.Timestamp('2025-12-01')
ORIGINS = [202510, 202601, 202602, 202603]
ELIGIBLE_AT = {LEGACY: set(ORIGINS), NEW: {202602, 202603}, YOUNG: {202603}}


def scenario_orders() -> pd.DataFrame:
    """The synthetic feed plus NEW1: A1's own orders from 2025-11 on, under a new code."""
    orders = make_orders()
    new = orders[(orders['SalesModelCode'] == 'A1') & (orders['EntryDate'] >= NEW_FIRST_ENTRY)].copy()
    new['SalesModelCode'], new['ProductCode'], new['PlanModelCode'] = NEW_SMC, f'{NEW_SMC}P', NEW
    return pd.concat([orders, new], ignore_index=True)


class ScenarioStub(FabricStub):
    """TFGMaterial with B1 and NEW1 undated and A2 launched 2025-12 - the same master data at every
    as-of point, as the real TFGMaterial is."""
    def _load_tfg(self):
        self.calls['load_tfg_material'] += 1
        tfg = make_tfg(self.orders)
        tfg.loc[tfg['ProductCode'].isin([f'{LEGACY_SMC}P', f'{NEW_SMC}P']), 'ProductStartDate'] = pd.NaT
        tfg.loc[tfg['ProductCode'] == f'{YOUNG_SMC}P', 'ProductStartDate'] = YOUNG_LAUNCH
        return tfg


def feed_as_of(orders: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """What a feed complete through `asof` contains: nothing entered or cancelled later."""
    o = orders[orders['EntryDate'] <= asof].copy()
    later = o['CancelDate'] > asof
    o.loc[later, 'IsCanceled'] = False
    o.loc[later, 'CancelMonthIdx'] = np.nan
    o.loc[later, ['CancelDate', 'CancelKnownDate']] = pd.NaT
    return o


def month_end(yyyymm) -> pd.Timestamp:
    return pd.Timestamp(f'{yyyymm // 100}-{yyyymm % 100:02d}-01') + pd.offsets.MonthEnd(0)


def keyset(df: pd.DataFrame) -> set:
    return set(zip(df['PlanModelCode'].astype(str), df['TargetIdx'].astype(int),
                   df['Horizon'].astype(int)))


def codes(df: pd.DataFrame) -> set:
    return set(df['PlanModelCode'].astype(str))


def rows(df: pd.DataFrame, code: str) -> pd.DataFrame:
    return df[df['PlanModelCode'].astype(str) == code]


def diff(a: set, b: set) -> str:
    return (f'{len(a)} vs {len(b)} rows; only first: {sorted(a - b)[:3]}, '
            f'only second: {sorted(b - a)[:3]}')


def live_rows(plant, origin_yyyymm, max_settled, grain='month', cutoff_date=None, min_age=MIN_AGE):
    """What production scores at this origin, built from a feed complete only through its as-of
    point (a source file dated the day after). Restricted to settled targets - a backtest can only
    score a target whose actuals exist, so that is the part the two can be compared on."""
    asof = cutoff_date if grain == 'day' else month_end(origin_yyyymm)
    with ScenarioStub(feed_as_of(ORDERS, asof), file_date=asof + pd.Timedelta(days=1)):
        sk, _ctx = pp.preprocess_for_inference(plant, origin_yyyymm, range(1, 7),
            cutoff_grain=grain, cutoff_date=cutoff_date, min_origin_months_since_launch=min_age)
    return sk[sk['TargetIdx'] <= max_settled]


def fold_at(fold_seq, origin_idx):
    """The fold whose test origin is `origin_idx`, or None."""
    starts = [spec[0] for spec in fold_seq.specs]
    return fold_seq[starts.index(origin_idx)] if origin_idx in starts else None


def backtest_populations(panel, origin_idx, max_settled, seeded) -> dict:
    """{population: its test rows at origin_idx} for every backtest that applies the age filter."""
    kw = dict(initial_train_origins=1, min_origin_months_since_launch=MIN_AGE)
    pops = {name: [] for name in ('walk_forward_horizon', 'baseline_folds_for_horizon',
                                  'test_rows_for_horizon', 'all_settled_rows_for_horizon')}
    for h in range(1, 7):
        fold = fold_at(fl.walk_forward_horizon(panel[panel['Horizon'] == h], horizon=h,
                                               max_settled_target_idx=max_settled, **kw), origin_idx)
        if fold is not None:
            pops['walk_forward_horizon'].append(fold[2])
        fold = fold_at(fl.baseline_folds_for_horizon(h, panel, max_settled, **kw), origin_idx)
        if fold is not None:
            pops['baseline_folds_for_horizon'].append(fold[1])
        for name, fn in (('test_rows_for_horizon', fl.test_rows_for_horizon),
                         ('all_settled_rows_for_horizon', fl.all_settled_rows_for_horizon)):
            got = fn(h, panel, max_settled, **kw)
            pops[name].append(got[got['OriginIdx'] == origin_idx])
    out = {name: pd.concat(frames, ignore_index=True) for name, frames in pops.items()}
    shared = fold_at(fl.walk_forward_shared(panel, max_settled_target_idx=max_settled, **kw),
                     origin_idx)
    out['walk_forward_shared'] = shared['test']
    out['seed_overall_results'] = seeded[seeded['OriginIdx'] == origin_idx]
    return out


ORDERS = scenario_orders()
FLOOR = int(ORDERS['EntryMonthIdx'].min())
FIRST_ENTRY = ORDERS.groupby('PlanModelCode')['EntryMonthIdx'].min()
check('precondition: B1 is ordered from the dataset floor', FIRST_ENTRY[LEGACY] == FLOOR)
check('precondition: NEW1 is first ordered in 2025-11', FIRST_ENTRY[NEW] == to_month_idx(202511))
TMP = pathlib.Path(tempfile.mkdtemp(prefix='eligibility_'))

for plant in PLANTS:
    tag = f'{plant} ({PLANT_NAMES[plant]})'

    print(f'\n=== 1. {tag}: the fallback reaches eligibility only, never a model input ===')
    with ScenarioStub(ORDERS), captured(logging.INFO) as msgs:
        panel, _fams, feature_cols, ctx = pp.preprocess_for_training(plant, cutoff_grain='month')
    max_settled = ctx['max_settled_idx']
    launch, elig = ctx['launch_by_code'], ctx['eligibility_launch_by_code']
    check(f'{tag}: launch_by_code is still NaN for B1 and NEW1 (the model launch is untouched)',
          np.isnan(launch[LEGACY]) and np.isnan(launch[NEW]))
    check(f'{tag}: eligibility launch - B1 predates the data (-inf), NEW1 its first order, A2 TFG',
          elig[LEGACY] == -np.inf and elig[NEW] == to_month_idx(202511)
          and elig[YOUNG] == to_month_idx(202512), f'{elig[[LEGACY, NEW, YOUNG]].to_dict()}')
    check(f'{tag}: OriginMonthsSinceLaunch (a model feature) is still NaN on every B1/NEW1 row',
          rows(panel, LEGACY)['OriginMonthsSinceLaunch'].isna().all()
          and rows(panel, NEW)['OriginMonthsSinceLaunch'].isna().all())
    check(f'{tag}: the skeleton is untouched - NEW1 still starts at the dataset floor',
          ctx['first_known_order_by_code'][NEW] == FLOOR + 1)
    known = panel['OriginMonthsSinceLaunch'].notna()
    check(f'{tag}: wherever the launch is known, the eligibility age equals OriginMonthsSinceLaunch',
          np.array_equal(panel.loc[known, ELIG].to_numpy(),
                         panel.loc[known, 'OriginMonthsSinceLaunch'].to_numpy()))
    check(f'{tag}: B1 rows are +inf, NEW1 rows count from 202511',
          np.isposinf(rows(panel, LEGACY)[ELIG]).all()
          and np.array_equal(rows(panel, NEW)[ELIG].to_numpy(),
                             (rows(panel, NEW)['OriginIdx'] - to_month_idx(202511)).to_numpy('float64')))
    check(f'{tag}: {ELIG} is in no feature list', ELIG not in feature_cols
          and ELIG not in pp.full_feature_cols() and ELIG not in pp.baseline_feature_cols())
    swapped = dict(ctx, eligibility_launch_by_code=pd.Series(-np.inf, index=elig.index))
    p_real, p_swap = pp.build_full_panel(ctx)[0], pp.build_full_panel(swapped)[0]
    check(f'{tag}: swapping the eligibility launch changes {ELIG} and nothing else',
          p_real.drop(columns=ELIG).equals(p_swap.drop(columns=ELIG))
          and not p_real[ELIG].equals(p_swap[ELIG]))
    check(f'{tag}: a WARNING names NEW1 as aged from its first order',
          logged(msgs, 'no TFGMaterial launch date - forecast eligibility counts their age from '
                       'their first order', NEW), str(msgs))
    check(f'{tag}: B1 is reported as predating the data',
          logged(msgs, 'already ordered when the data starts', LEGACY), str(msgs))

    print(f'\n=== 2. {tag}: inference == every backtest population, origin by origin ===')
    seed_path = TMP / f'panel_{plant}.parquet'
    panel[['PlanModelCode', 'PartCode', 'Horizon', 'OriginIdx', 'TargetIdx', 'ActualOrder',
           'KnownOrder', ELIG]].to_parquet(seed_path, index=False)
    seeded = seed_overall_results(str(seed_path), str(TMP / f'res_{plant}.parquet'),
                                  min_origin_months_since_launch=MIN_AGE)
    for origin in ORIGINS:
        origin_idx = int(to_month_idx(origin))
        live = live_rows(plant, origin, max_settled)
        check(f'{tag} {origin}: membership - B1/NEW1/A2 in exactly where expected',
              all((code in codes(live)) == (origin in ELIGIBLE_AT[code]) for code in ELIGIBLE_AT),
              f'live codes {sorted(codes(live))}')
        for name, pop in backtest_populations(panel, origin_idx, max_settled, seeded).items():
            check(f'{tag} {origin}: {name} == inference ({len(keyset(live))} rows)',
                  keyset(pop) == keyset(live), diff(keyset(pop), keyset(live)))

    print(f'\n=== 2b. {tag}: the same at day grain (origin 202602, as of 2026-03-10) ===')
    with ScenarioStub(ORDERS):
        day_panel, _f, _c, day_ctx = pp.preprocess_for_training(plant, cutoff_grain='day')
    asof = pd.Timestamp('2026-03-10')
    per_row = day_panel.groupby(['PlanModelCode', 'OriginIdx', 'Horizon'], observed=True)[ELIG]
    check(f'{tag}: {ELIG} is the same on every cutoff day of a (code, origin, horizon)',
          bool((per_row.nunique(dropna=False) == 1).all()))
    live_d = live_rows(plant, 202602, day_ctx['max_settled_idx'], grain='day', cutoff_date=asof)
    wf_d = [fold_at(fl.walk_forward_horizon(day_panel[day_panel['Horizon'] == h], horizon=h,
                    initial_train_origins=1, max_settled_target_idx=day_ctx['max_settled_idx'],
                    min_origin_months_since_launch=MIN_AGE), int(to_month_idx(202602)))
            for h in range(1, 7)]
    test_d = pd.concat([f[2] for f in wf_d if f is not None], ignore_index=True)
    test_d = test_d[pd.to_datetime(test_d[CUTOFF_DATE_COL]) == asof]
    check(f'{tag} day: walk_forward_horizon test rows on {asof.date()} == inference',
          len(test_d) > 0 and keyset(test_d) == keyset(live_d), diff(keyset(test_d), keyset(live_d)))
    check(f'{tag} day: NEW1 (age 3 by first order) in, A2 (age 2) out, B1 in',
          {NEW, LEGACY} <= codes(live_d) and YOUNG not in codes(live_d))

    print(f'\n=== 3. {tag}: the rules this replaced disagreed on the same rows ===')
    origin_idx = int(to_month_idx(202601))
    universe = panel[(panel['OriginIdx'] == origin_idx) & (panel['TargetIdx'] <= max_settled)]
    unfiltered = live_rows(plant, 202601, max_settled, min_age=None)
    check(f'{tag}: before any age filter, backtest and inference start from the same rows',
          keyset(universe) == keyset(unfiltered), diff(keyset(universe), keyset(unfiltered)))
    old_backtest = universe[universe['OriginMonthsSinceLaunch'] >= MIN_AGE]
    hotfix_live = unfiltered[~(unfiltered['OriginMonthsSinceLaunch'] < MIN_AGE)]
    gap = keyset(hotfix_live) - keyset(old_backtest)
    check(f'{tag}: old backtest vs hotfix inference differed by exactly the B1 and NEW1 rows',
          keyset(old_backtest) < keyset(hotfix_live) and {c for c, _t, _h in gap} == {LEGACY, NEW},
          diff(keyset(old_backtest), keyset(hotfix_live)))
    shared_rule = live_rows(plant, 202601, max_settled)
    check(f'{tag}: the hotfix forecast NEW1 at age 2 - the shared rule holds it back',
          NEW in codes(hotfix_live) and NEW not in codes(shared_rule))
    check(f'{tag}: the old backtest dropped B1 (mature legacy) - the shared rule keeps it',
          LEGACY not in codes(old_backtest) and LEGACY in codes(shared_rule))

print('\n=== 4. a panel saved before the column existed ===')
legacy_frame = panel.drop(columns=ELIG)
lf._warned_legacy_frame = False
with captured() as msgs:
    first = lf.meets_min_launch_age(legacy_frame, MIN_AGE)
    second = lf.meets_min_launch_age(legacy_frame, MIN_AGE)
check('falls back to the old test exactly: OriginMonthsSinceLaunch >= 3, undated codes excluded',
      np.array_equal(first, (legacy_frame['OriginMonthsSinceLaunch'] >= MIN_AGE).to_numpy())
      and not first[(legacy_frame['PlanModelCode'].astype(str) == LEGACY).to_numpy()].any())
check('and says so, once per process',
      np.array_equal(first, second) and sum('built before it existed' in m for m in msgs) == 1,
      str(msgs))
try:
    lf.meets_min_launch_age(legacy_frame.drop(columns='OriginMonthsSinceLaunch'), MIN_AGE)
    check('a frame with no age column at all raises', False, 'no exception')
except ValueError as e:
    check('a frame with no age column at all raises', 'min_origin_months_since_launch' in str(e))

print('\n' + '=' * 78)
print(f'{len(FAILURES)} FAILURE(S): {FAILURES}' if FAILURES else 'ALL CHECKS PASSED')
sys.exit(1 if FAILURES else 0)
