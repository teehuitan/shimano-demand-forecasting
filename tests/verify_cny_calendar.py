"""CNY calendar features are defined for every year (audit B1).

The bug this pins: Is_CNY_Month / Months_to_CNY were looked up in a hand-kept list of CNY months
that ran from 201702 to 202702. np.clip pinned every later target to Feb 2027, so target 202703 got
Months_to_CNY = -1 (-6 by 202708, -11 by 202801) - a value no model had trained on, and one a tree
split like `Months_to_CNY < 0.5` sends down the CNY-month branch. It first reaches a live row at
origin 202609, h6. Feb 2028 stopped counting as CNY at all, and at the other end Feb 2016 - inside
SPL's history - never did. CutoffMonthIsCNY read the same list.

  1. the rule itself: Is_CNY_Month <=> CNY_MONTH, Months_to_CNY in 0..11, 2010-2040;
  2. unchanged wherever the old list was right (targets 201702..202702), and different exactly
     where it was wrong - the removed code is reproduced here, so the check isn't vacuous;
  3. the real inference path at origin 202609 (h6 -> 202703), both grains, every plant;
  4. CutoffMonthIsCNY on the day path.
"""
import logging
import os
import pathlib
import sys
import warnings

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'tests'))
os.chdir(_REPO_ROOT)        # the STJ replacement workbook is cwd-relative
import no_db  # noqa: E402,F401 - every DB engine raises from here on (see no_db.py)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from synthetic_fabric import FabricStub  # noqa: E402
from adj_forecast_lib import intramonth_features as im, preprocessing as pp  # noqa: E402
from adj_forecast_lib.config import (  # noqa: E402
    CNY_MONTH, CUTOFF_DATE_COL, PLANT_INCLUDES_CNY, PLANT_NAMES, is_cny_month_idx,
    month_idx_to_yyyymm, months_to_cny, to_month_idx,
)
from adj_forecast_lib.features import attach_calendar_features  # noqa: E402

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)
FAILURES = []


def check(label, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' - ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(label)


# An unconfigured plant code can't run section 3 yet: build_order_book_context indexes the
# per-plant config dicts directly (audit B15). Sections 1, 2 and 4 involve no plant.
PLANTS = sorted(PLANT_NAMES)
ALL = np.arange(to_month_idx(201001), to_month_idx(204012) + 1)
MONTHS = month_idx_to_yyyymm(ALL) % 100


def at(yyyymm) -> int:
    return int(np.flatnonzero(ALL == to_month_idx(yyyymm))[0])


print('\n=== 1. the rule, 2010-2040 ===')
mt = months_to_cny(ALL)
check('Is_CNY_Month is exactly CNY_MONTH, in every year',
      np.array_equal(is_cny_month_idx(ALL), MONTHS == CNY_MONTH))
check('Months_to_CNY is always 0..11', mt.min() == 0 and mt.max() == 11, f'{mt.min()}..{mt.max()}')
check('Months_to_CNY is 0 exactly in CNY months', np.array_equal(mt == 0, MONTHS == CNY_MONTH))
check('Months_to_CNY counts down one a month and restarts at 11 after each CNY',
      bool(np.all(np.where(mt[:-1] == 0, mt[1:] == 11, mt[1:] == mt[:-1] - 1))))

print('\n=== 2. the old list, reproduced: identical where it was right, fixed where it was wrong ===')
OLD_CNY_YYYYMM = [201702, 201802, 201902, 202002, 202102, 202202,
                  202302, 202402, 202502, 202602, 202702]
OLD_IDX = np.sort(to_month_idx(np.array(OLD_CNY_YYYYMM)))


def old_calendar(target_idx):
    """The removed features.attach_calendar_features CNY block, verbatim."""
    is_cny = np.isin(target_idx, OLD_IDX).astype(int)
    pos = np.clip(np.searchsorted(OLD_IDX, target_idx, side='left'), 0, len(OLD_IDX) - 1)
    return is_cny, OLD_IDX[pos] - target_idx


new = attach_calendar_features(pd.DataFrame({'TargetIdx': ALL}), 'TargetIdx')
old_is, old_mt = old_calendar(ALL)
span = (ALL >= to_month_idx(201702)) & (ALL <= to_month_idx(202702))
check('every target the old list covered (201702..202702) is unchanged, both columns',
      np.array_equal(new['Is_CNY_Month'].to_numpy()[span], old_is[span])
      and np.array_equal(new['Months_to_CNY'].to_numpy()[span], old_mt[span]))
check('both columns keep their dtypes',
      new['Is_CNY_Month'].dtype == old_is.dtype and new['Months_to_CNY'].dtype == old_mt.dtype,
      f"{new['Is_CNY_Month'].dtype}/{new['Months_to_CNY'].dtype} vs {old_is.dtype}/{old_mt.dtype}")
for yyyymm, was, now in [(202703, -1, 11), (202708, -6, 6), (202801, -11, 1), (201602, 12, 0)]:
    check(f'target {yyyymm}: Months_to_CNY was {was}, is now {now}',
          old_mt[at(yyyymm)] == was and new['Months_to_CNY'].iloc[at(yyyymm)] == now,
          f"old {old_mt[at(yyyymm)]}, new {new['Months_to_CNY'].iloc[at(yyyymm)]}")
check('Feb 2016 and Feb 2028 count as CNY again (the old list said no)',
      old_is[at(201602)] == 0 and old_is[at(202802)] == 0
      and new['Is_CNY_Month'].iloc[at(201602)] == 1 and new['Is_CNY_Month'].iloc[at(202802)] == 1)
check('include_cny=False still attaches neither column',
      not {'Is_CNY_Month', 'Months_to_CNY'} & set(attach_calendar_features(
          pd.DataFrame({'TargetIdx': ALL}), 'TargetIdx', include_cny=False).columns))

print('\n=== 3. the live path at origin 202609: h6 targets 202703 ===')
LIVE_FILE = pd.Timestamp('2026-10-02')         # complete through Oct 1 -> Sep 2026 settles
EXPECTED = {1: 4, 2: 3, 3: 2, 4: 1, 5: 0, 6: 11}   # targets 202610 .. 202703
with FabricStub(file_date=LIVE_FILE):
    for plant in PLANTS:
        tag = f'{plant} ({PLANT_NAMES[plant]})'
        for grain, date in (('month', None), ('day', pd.Timestamp('2026-10-01'))):
            sk, _ctx = pp.preprocess_for_inference(plant, horizons=range(1, 7),
                cutoff_grain=grain, cutoff_date=date)
            check(f'{tag} {grain}: scored at origin 202609, all six horizons',
                  set(sk['OriginIdx']) == {int(to_month_idx(202609))}
                  and set(sk['Horizon']) == set(EXPECTED))
            if not PLANT_INCLUDES_CNY[plant]:
                check(f'{tag} {grain}: CNY columns absent (PLANT_INCLUDES_CNY is off)',
                      not {'Is_CNY_Month', 'Months_to_CNY'} & set(sk.columns))
                continue
            got = sk.groupby('Horizon')['Months_to_CNY'].agg(['min', 'max'])
            check(f'{tag} {grain}: Months_to_CNY per horizon is {EXPECTED} (h6 was -1)',
                  all(got.loc[h, 'min'] == v == got.loc[h, 'max'] for h, v in EXPECTED.items()),
                  got.to_dict('index').__repr__())
            check(f'{tag} {grain}: only the h5 row (target 202702) is a CNY month',
                  set(sk.loc[sk['Is_CNY_Month'] == 1, 'Horizon']) == {5})

print('\n=== 4. CutoffMonthIsCNY on the day path ===')
dates = pd.date_range('2016-01-01', '2030-12-31', freq='5D')
cal = im._attach_cutoff_calendar(pd.DataFrame({CUTOFF_DATE_COL: dates}), CUTOFF_DATE_COL)
check('CutoffMonthIsCNY is 1 exactly for as-of dates in a CNY month, 2016-2030',
      np.array_equal(cal['CutoffMonthIsCNY'].to_numpy() == 1, dates.month == CNY_MONTH))
check('CutoffMonthIsCNY stays int8', cal['CutoffMonthIsCNY'].dtype == 'int8')

print('\n' + '=' * 78)
print(f'{len(FAILURES)} FAILURE(S): {FAILURES}' if FAILURES else 'ALL CHECKS PASSED')
sys.exit(1 if FAILURES else 0)
