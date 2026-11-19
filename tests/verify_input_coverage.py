"""Input guards and SKU coverage (audit B2, B3): two ways rows or whole SalesModelCodes used to
vanish, or corrupt every skeleton, without a line in the log.

  B2  A missing order date became month ordinal -2147483648 (NaN cast to int32). A missing request
      date was then dropped as a "negative lead time"; a missing ENTRY date was kept and became the
      dataset floor, so every range()/arange() built from it ran across billions of months. load_io
      now drops unusable dates with a WARNING per reason, and build_order_book_context refuses a
      floor before MIN_VALID_ORDER_DATE whatever loaded the orders.
  B3  TFGMaterial's ProductCode was matched RAW against the clean_code()d order side, so a master
      record with a dash, space, lowercase or CHAR padding never matched - its SalesModelCode had
      no effective range and was silently never trained or forecast. Both sides are cleaned now,
      and anything still unmatched is logged by name.

Audit B4 (codes with no launch date) is covered by verify_forecast_eligibility.

Every section that builds a context runs the real builders over the synthetic feed for every plant
in config.PLANT_NAMES. No database: tests/no_db.py makes every engine raise, and load_io's one
pd.read_sql is stubbed with a frame shaped like io.sql's output.
"""
import contextlib
import logging
import os
import pathlib
import sys
import warnings

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'tests'))
os.chdir(_REPO_ROOT)        # io.sql and the STJ replacement workbook are cwd-relative
import no_db  # noqa: E402,F401 - every DB engine raises from here on (see no_db.py)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from synthetic_fabric import FabricStub, make_orders, make_tfg  # noqa: E402
from adj_forecast_lib import lifecycle_cleaning as lc, preprocessing as pp  # noqa: E402
from adj_forecast_lib.config import PLANT_NAMES, date_to_month_idx, to_month_idx  # noqa: E402

warnings.filterwarnings('ignore')
LIB = logging.getLogger('adj_forecast_lib')
LIB.addHandler(logging.NullHandler())   # keep the library's warnings off stderr outside captures
FAILURES = []


def check(label, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' - ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(label)


def raises(fn, exc=Exception, needle=''):
    try:
        fn()
    except exc as e:
        return needle in str(e), f'{type(e).__name__}: {str(e)[:160]}'
    except Exception as e:
        return False, f'wrong exception {type(e).__name__}: {str(e)[:160]}'
    return False, 'no exception raised'


class _Records(logging.Handler):
    def __init__(self, level):
        super().__init__(level)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def captured(level=logging.WARNING):
    """The library's log messages at `level` and above, as formatted strings."""
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


# An unconfigured plant code can't run the context-building sections yet: load_io and
# build_order_book_context index the per-plant config dicts directly (audit B15).
PLANTS = sorted(PLANT_NAMES)
# Through resolve_plan_model_codes as build_order_book_context does - the range resolvers read its
# PlanModelCode column.
ORDERS, TFG = lc.resolve_plan_model_codes(make_orders(), make_tfg(make_orders()))

print('\n=== B2.1 _valid_order_dates: one reason per unusable date ===')
NOW = pd.Timestamp('2026-10-01 09:30')
T = pd.Timestamp
entry = pd.Series([T('2024-03-05'), pd.NaT, T('2024-03-05'), T('1900-01-01'), T('2024-03-05'),
                   T('2026-10-02'), T('2026-10-03')])
request = pd.Series([T('2024-05-01'), T('2024-05-01'), pd.NaT, T('2024-05-01'), T('1900-01-01'),
                     T('2026-11-01'), T('2026-11-01')])
with captured() as msgs:
    valid = pp._valid_order_dates(entry, request, now=NOW)
check('valid rows kept; missing, pre-2010 and future dates dropped (one day of slack after now)',
      valid.tolist() == [True, False, False, False, False, True, False], str(valid.tolist()))
for reason in ('no Order_Entry_Date', 'no Customer_Request_Date', 'Order_Entry_Date before 2010-01-01',
               'Customer_Request_Date before 2010-01-01', 'in the future'):
    check(f"WARNING '1 order row(s) dropped - {reason}'", logged(msgs, '1 order row(s) dropped', reason))
corrupt = date_to_month_idx(pd.Series([pd.NaT], dtype='datetime64[ns]')).astype('int32')[0]
check('the failure mode it guards is real: a missing date casts to a garbage month ordinal',
      corrupt < to_month_idx(201001), f'cast gave {corrupt}')

print('\n=== B2.2 load_io end to end (pd.read_sql stubbed) ===')
TODAY = pd.Timestamp.now().normalize()
GOOD = [(T('2024-03-05'), T('2024-05-10')), (T('2024-04-01'), T('2024-04-20')),
        (T('2025-01-15'), T('2025-03-01'))]
BAD = [(None, T('2024-05-10')), (T('2024-03-05'), None), (T('1900-01-01'), T('2024-05-10')),
       (T('2024-03-05'), T('1900-01-01')), (TODAY + pd.Timedelta(days=30), TODAY + pd.Timedelta(days=60))]
ROWS = GOOD + BAD
RAW = pd.DataFrame({
    'MaterialCode': [f'mat-{i}' for i in range(len(ROWS))], 'Product_Type_Code': 'RD',
    'Model_Code': 'rd-m8100', 'Plan_Model_Code': 'pn-rd-m8100', 'Ship_To_Code': 'S1',
    'Class_Text': 'X', 'Currency_Code': 'usd',
    'Customer_Country_Code': 'CN', 'Unit_Price': 10.0, 'Quantity': 5.0,
    'Order_Status_Text': 'Open',
    'Order_Entry_Date Date': pd.Series([e for e, _ in ROWS], dtype='object'),
    'Customer_Request_Date Date': pd.Series([r for _, r in ROWS], dtype='object'),
    'Cancel_Date Date': None,
})
real_read_sql = pd.read_sql
pd.read_sql = lambda *args, **kwargs: RAW.copy()
try:
    for plant in PLANTS:
        tag = f'{plant} ({PLANT_NAMES[plant]})'
        with captured(logging.INFO) as msgs:
            io = pp.load_io(plant)
        check(f'{tag}: the {len(GOOD)} good rows survive, the {len(BAD)} bad ones do not',
              len(io) == len(GOOD), f'{len(io)} rows')
        check(f'{tag}: no garbage month ordinal reaches the frame',
              io['EntryMonthIdx'].min() >= to_month_idx(201001)
              and io['RequestMonthIdx'].min() >= to_month_idx(201001))
        check(f'{tag}: every reason logged as a WARNING',
              sum('order row(s) dropped' in m for m in msgs) == len(BAD))
        check(f'{tag}: a missing/placeholder request date is no longer counted as negative lead',
              logged(msgs, 'Dropped 0 row(s) whose Customer_Request_Date month is before their '
                           'Order_Entry_Date month'))
finally:
    pd.read_sql = real_read_sql

print('\n=== B2.3 build_order_book_context refuses a floor before MIN_VALID_ORDER_DATE ===')
for label, entry_idx, entry_date in (
        ('a 1900 placeholder entry date', to_month_idx(190001), T('1900-01-15')),
        ('the NaN-cast-to-int32 ordinal', np.iinfo('int32').min, T('2024-01-15'))):
    bad = ORDERS.iloc[[0]].copy()
    bad['EntryMonthIdx'] = np.int32(entry_idx)
    bad['EntryDate'] = entry_date
    with FabricStub(pd.concat([ORDERS, bad], ignore_index=True)):
        for plant in PLANTS:
            ok, d = raises(lambda: pp.build_order_book_context(plant, cutoff_grain='month'),
                           ValueError, 'MIN_VALID_ORDER_DATE')
            check(f'{plant} ({PLANT_NAMES[plant]}): {label} is refused before any skeleton is built',
                  ok, d)

print('\n=== B3.1 messy TFGMaterial ProductCodes still match ===')


def messy(tfg: pd.DataFrame) -> pd.DataFrame:
    """Every master-data spelling clean_code() folds back: lowercase, a dash, leading space, padding."""
    styles = [str.lower, lambda c: c[:1] + '-' + c[1:], lambda c: ' ' + c, lambda c: c + '   ']
    return tfg.assign(ProductCode=[styles[i % 4](c) for i, c in enumerate(tfg['ProductCode'])])


_REAL_LOAD_TFG = pp.load_tfg_material      # before any FabricStub replaces it


def load_tfg_from(raw: pd.DataFrame) -> pd.DataFrame:
    """The real load_tfg_material (where codes are cleaned) on `raw` as the Fabric result."""
    real_read_sql, pd.read_sql = pd.read_sql, lambda *a, **k: raw.copy()
    try:
        return _REAL_LOAD_TFG()
    finally:
        pd.read_sql = real_read_sql


RAW_TFG = make_tfg(make_orders())
MESSY = messy(RAW_TFG)
check('the messy master data really differs from the order codes (so this section is not vacuous)',
      not set(MESSY['ProductCode']) & set(ORDERS['ProductCode']))
check('load_tfg_material folds every messy ProductCode back to the clean one',
      sorted(load_tfg_from(MESSY)['ProductCode']) == sorted(load_tfg_from(RAW_TFG)['ProductCode']))
with captured() as msgs:
    resolved = [lc.resolve_plan_model_codes(make_orders(), load_tfg_from(t))[1] for t in (RAW_TFG, MESSY)]
    clean_r, messy_r = (lc.resolve_effective_ranges_with_tfg(ORDERS, t) for t in resolved)
key = lambda df, col: df.sort_values(col).reset_index(drop=True)
check('effective ranges from messy master data equal those from clean master data',
      key(messy_r, 'PlanModelCode').equals(key(clean_r, 'PlanModelCode')))
check('nothing is logged when every code resolves',
      not [m for m in msgs if 'no TFGMaterial record' in m or 'no effective range' in m], str(msgs))

print('\n=== B3.2 the warnings name what is still unmatched ===')
partial = ORDERS.copy()
partial.loc[partial.index[partial['SalesModelCode'] == 'A1'][:50], 'ProductCode'] = 'A1Q'
with captured() as msgs:
    r = lc.resolve_effective_ranges_with_tfg(partial, TFG[TFG['ProductCode'] != 'KNOWN1P'])
check("'2 ProductCodes have orders but no TFGMaterial record', naming A1Q and KNOWN1P",
      logged(msgs, '2 ProductCodes have orders but no TFGMaterial record', 'A1Q', 'KNOWN1P'), str(msgs))
check('the PlanModelCode that lost EVERY ProductCode is named as not trained or forecast',
      logged(msgs, '1 PlanModelCodes have orders but no effective range', 'PNKNOWN1',
             'will not be trained or forecast'), str(msgs))
check('PNA1 keeps its range through its other ProductCode; PNKNOWN1 has none',
      'PNA1' in set(r['PlanModelCode']) and 'PNKNOWN1' not in set(r['PlanModelCode']))


class MessyTfgStub(FabricStub):
    def _load_tfg(self):
        self.calls['load_tfg_material'] += 1
        return load_tfg_from(messy(make_tfg(self.orders)))


print('\n=== B3.3 the live path forecasts the same codes from messy master data ===')
for plant in PLANTS:
    tag = f'{plant} ({PLANT_NAMES[plant]})'
    with FabricStub(ORDERS):
        clean_sk, _ = pp.preprocess_for_inference(plant, horizons=range(1, 7), cutoff_grain='month')
    with MessyTfgStub(ORDERS):
        messy_sk, _ = pp.preprocess_for_inference(plant, horizons=range(1, 7), cutoff_grain='month')
    try:
        pd.testing.assert_frame_equal(messy_sk, clean_sk)
        same, detail = True, ''
    except AssertionError as e:
        same, detail = False, str(e)[:300]
    check(f'{tag}: identical inference skeleton, all {clean_sk["PlanModelCode"].nunique()} codes',
          same, detail)

print('\n' + '=' * 78)
print(f'{len(FAILURES)} FAILURE(S): {FAILURES}' if FAILURES else 'ALL CHECKS PASSED')
sys.exit(1 if FAILURES else 0)
