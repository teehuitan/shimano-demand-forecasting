"""Run resilience and DB-write contracts: what a daily run leaves in the tables when it is rerun,
when a write fails, when a retrain fails, and when the source files misbehave.

  B5  OutstandingOrder keeps one snapshot per source file for the current month. A rerun of the
      same file replaces only that file's rows (it used to append duplicates); a new file adds.
  B6  PredictionResult keeps the month's daily forecasts: a new OriginDate adds, a rerun at the
      same OriginDate replaces its own rows. The first write of a month (no ActualOrder for N yet)
      upserts the actual and clears last month's PredictionResult and OutstandingOrder in the SAME
      transaction as the insert - so a failure leaves everything, the actual included, as it was,
      and the next file retries the month start. Every consistency check runs before any write.
  B7  A failed retrain scores the day with the incumbent, OutstandingOrder refreshes beside it (not
      after a failed scoring), and run_daily still raises DailyRunError; main() exits non-zero
      once every plant has run.
  B8  The settled month comes from the newest PARSEABLE file name; an origin earlier than the
      deployment's trained_through is refused.
  B13 PredictionResultLog is write-only: every write appends all its rows, a rerun included, so the
      latest logged row per forecast and OriginDate (by Id) is what PredictionResult holds.
  B18 A run whose data is empty or broken writes nothing, and says why in plain words.
  B19 A model never forecasts a negative net order; the manual benchmark is left as computed.

The DB layer runs against synthetic_fabric.TableEngine (in-memory tables, real commit/rollback),
so each check reads what the tables HOLD. The registry is the throwaway MLflow store (no_db.py).
DB sections also run for an unconfigured plant code; orchestrator sections only for configured
plants, since _prediction_phase reads per-plant config. No database.
"""
import contextlib
import copy
import logging
import os
import pathlib
import re
import sys
import warnings

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'tests'))
import no_db  # noqa: E402,F401 - every DB engine raises from here on (see no_db.py)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import main as orch  # noqa: E402
from adj_forecast_lib import collate_actual_order as cao  # noqa: E402
from adj_forecast_lib import db_writer, model_registry as mr, predict as pm, preprocessing as pp  # noqa: E402
from adj_forecast_lib.config import PLANT_NAMES, month_idx_to_yyyymm, plant_name, to_month_idx  # noqa: E402
from synthetic_fabric import InjectedFailure, TableEngine  # noqa: E402

warnings.filterwarnings('ignore')
# Several checks make a phase fail on purpose, and run_daily logs each failure with its traceback.
logging.getLogger().addHandler(logging.NullHandler())
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
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def captured(level=logging.WARNING):
    """`level`-and-above records from the library and main.py while the block runs."""
    root = logging.getLogger()
    handler, old_level = _Records(level), root.level
    root.addHandler(handler)
    root.setLevel(min(level, old_level))
    try:
        yield handler.records
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)


def logged(records, level, *needles) -> bool:
    return any(r.levelno >= level and all(n in r.getMessage() for n in needles) for r in records)


@contextlib.contextmanager
def db(engine):
    saved = db_writer._engine
    db_writer._engine = engine
    try:
        yield engine
    finally:
        db_writer._engine = saved


PLANTS = sorted(PLANT_NAMES) + [4400]          # 4400: a plant no config dict knows
RUN_PLANTS = sorted(PLANT_NAMES)
T = int(to_month_idx(202609))                  # the settled month an October 2026 file implies
tag_of = lambda plant: f'{plant} ({plant_name(plant)})'
PR_INSERT = '.PredictionResult\n'              # the PredictionResult INSERT, not its Log's
LOG_INSERT = 'INSERT INTO [AI].DemandForecast.PredictionResultLog'

# ---------------------------------------------------------------------------------------------
print('\n=== B5. OutstandingOrder: one snapshot per source file, a rerun replaces its own ===')


def snapshot(file_yyyymmdd, codes, qty):
    return pd.DataFrame({'PartCode': 'SL', 'SalesModelCode': list(codes),
                         'PlanModelCode': [f'PN{c}' for c in codes],
                         'SourceFileName': f'{file_yyyymmdd}.xlsx', 'TargetYearMonth': '202610',
                         'OutstandingOrder': float(qty)})


def write_snapshot(file_yyyymmdd, codes, qty, plant):
    return db_writer.write_outstanding_order(snapshot(file_yyyymmdd, codes, qty), plant,
                                             f'{file_yyyymmdd}.xlsx')


old = TableEngine()
for day, codes, qty in ((20260914, 'ABC', 5), (20260914, 'ABC', 5), (20260915, 'AB', 7)):
    with old.begin() as conn:      # a pure INSERT: what a rerun did before it deleted its own file
        conn.execute('INSERT INTO [AI].DemandForecast.OutstandingOrder',
                     [dict(r, Plant='3300') for r in snapshot(day, codes, qty).to_dict('records')])
held = old.rows('OutstandingOrder', 3300)
check('a pure INSERT really duplicates a rerun: 3 + 3 + 2 = 8 rows, code A counted 3 times',
      len(held) == 8 and (held['PlanModelCode'] == 'PNA').sum() == 3, f'{len(held)} rows')

eng = TableEngine()
with db(eng):
    for plant in PLANTS:                    # every plant holds a snapshot before any is changed
        write_snapshot(20260914, 'ABC', plant, plant)
    for plant in PLANTS:
        tag = tag_of(plant)
        others = {p: eng.rows('OutstandingOrder', p) for p in PLANTS if p != plant}
        first, begun = eng.rows('OutstandingOrder', plant), eng.begun
        write_snapshot(20260914, 'ABC', plant, plant)
        check(f'{tag}: a same-file rerun leaves the same 3 rows, not 6',
              eng.rows('OutstandingOrder', plant).equals(first), str(len(eng.rows('OutstandingOrder', plant))))
        write_snapshot(20260915, 'AB', plant, plant)
        now = eng.rows('OutstandingOrder', plant)
        check(f"{tag}: the next day's file is ADDED beside it - 3 + 2 rows, one snapshot per file",
              len(now) == 5 and sorted(now['SourceFileName'].unique()) == ['20260914.xlsx', '20260915.xlsx'],
              str(now))
        check(f'{tag}: each write is ONE transaction', eng.begun == begun + 2, str(eng.begun - begun))
        eng.fail_on = 'INSERT INTO'
        ok, d = raises(lambda: write_snapshot(20260915, 'ABC', plant, plant), InjectedFailure)
        eng.fail_on = None
        check(f"{tag}: a failed rerun INSERT rolls back its DELETE - that file's rows stay",
              ok and eng.rows('OutstandingOrder', plant).equals(now), d)
        check(f"{tag}: no other plant's rows were touched",
              all(eng.rows('OutstandingOrder', p).equals(rows) for p, rows in others.items()))

main_src = (_REPO_ROOT / 'main.py').read_text(encoding='utf-8')
check('main.py calls no separate clear - each table is cleared inside the write that refills it',
      not re.search(r'db_writer\.clear_', main_src))

# ---------------------------------------------------------------------------------------------
print('\n=== B6. PredictionResult keeps the month; the month start is one transaction ===')


def forecast(n, known=100.0):
    return pd.DataFrame({'PartCode': 'SL', 'SalesModelCode': [f'C{i}' for i in range(n)],
                         'PlanModelCode': [f'PNC{i}' for i in range(n)],
                         'TargetYYYYMM': 202610, 'Horizon': 1, 'KnownOrder': known,
                         'KnownCanceledOrder': 0.0, 'Method': 'model_production',
                         'Predicted': known + 20.0})


def actual(month_yyyymm=202609):
    return pd.DataFrame({'PartCode': ['SL'], 'PlanModelCode': ['PNC0'], 'TargetYYYYMM': [month_yyyymm],
                         'ActualGrossOrder': [90.0], 'ActualNetOrder': [85.0], 'SalesResult': [np.nan]})


SEP14, SEP15, SEP16, SEP30 = (pd.Timestamp(f'2026-09-{d}') for d in (14, 15, 16, 30))
at = lambda df, origin: df[pd.to_datetime(df['OriginDate']) == origin] if len(df) else df

old = TableEngine()
with db(old):
    db_writer.write_prediction_result(forecast(4), 3300, SEP14)
old.fail_on = 'PredictionResultLog'
with old.begin() as conn:       # a clear in its own transaction, committed
    conn.execute('DELETE FROM [AI].DemandForecast.PredictionResult WHERE Plant = :plant', {'plant': '3300'})
try:
    with old.begin() as conn:   # the refill, in a second transaction that fails
        conn.execute('INSERT INTO [AI].DemandForecast.PredictionResultLog', [{}])
except InjectedFailure:
    pass
check('a clear and refill in two transactions really leaves NO forecast when the refill fails',
      old.rows('PredictionResult', 3300).empty)

eng = TableEngine()
with db(eng):
    for plant in PLANTS:
        db_writer.write_prediction_result(forecast(4), plant, SEP14)
    for plant in PLANTS:
        tag = tag_of(plant)
        others = {p: eng.rows('PredictionResult', p) for p in PLANTS if p != plant}
        begun = eng.begun
        db_writer.write_prediction_result(forecast(3), plant, SEP15)
        cur, hist = eng.rows('PredictionResult', plant), eng.rows('PredictionResultLog', plant)
        check(f"{tag}: a new day's forecast is ADDED beside the month's earlier days (4 + 3), in ONE "
              'transaction', len(at(cur, SEP14)) == 4 and len(at(cur, SEP15)) == 3
              and eng.begun == begun + 1, f'{len(cur)} rows, {eng.begun - begun} transaction(s)')
        db_writer.write_prediction_result(forecast(3, known=110.0), plant, SEP15)
        rerun = eng.rows('PredictionResult', plant)
        check(f'{tag}: a rerun at the same OriginDate replaces only that day\'s rows',
              len(rerun) == 7 and at(rerun, SEP14).equals(at(cur, SEP14))
              and (at(rerun, SEP15)['KnownGrossOrder'] == 110.0).all())
        cur, hist = rerun, eng.rows('PredictionResultLog', plant)
        for where, needle in (('the same-day DELETE', '.PredictionResult WHERE'),
                              ('the PredictionResult INSERT', PR_INSERT),
                              ('the PredictionResultLog INSERT', LOG_INSERT)):
            eng.fail_on = needle
            ok, d = raises(lambda: db_writer.write_prediction_result(forecast(5), plant, SEP16),
                           InjectedFailure)
            eng.fail_on = None
            check(f'{tag}: {where} fails -> the forecast and the Log are exactly as they were',
                  ok and eng.rows('PredictionResult', plant).equals(cur)
                  and eng.rows('PredictionResultLog', plant).equals(hist), d)
        with captured() as recs:
            n = db_writer.write_prediction_result(forecast(0), plant, SEP16)
        check(f'{tag}: an empty result writes nothing and keeps the current forecast, with a WARNING',
              n == 0 and eng.rows('PredictionResult', plant).equals(cur)
              and logged(recs, logging.WARNING, 'previous forecast is kept'))
        check(f"{tag}: no other plant's forecast was touched",
              all(eng.rows('PredictionResult', p).equals(rows) for p, rows in others.items()))

    for plant in PLANTS:
        tag = tag_of(plant)
        write_snapshot(20260915, 'AB', plant, plant)
        others = {p: copy.deepcopy(eng.rows('OutstandingOrder', p)) for p in PLANTS if p != plant}
        before = copy.deepcopy(eng.tables)
        for where, needle in (('the ActualOrder upsert', 'MERGE'),
                              ('the OutstandingOrder clear', '.OutstandingOrder WHERE'),
                              ('the PredictionResult clear', '.PredictionResult WHERE'),
                              ('the PredictionResult INSERT', PR_INSERT)):
            eng.fail_on = needle
            ok, d = raises(lambda: db_writer.month_start_write_clear(actual(), forecast(2), plant, SEP30),
                           InjectedFailure)
            eng.fail_on = None
            check(f'{tag}: month start - {where} fails -> every table as it was, the actual still '
                  'missing, so the next file retries the month start',
                  ok and eng.tables == before and not db_writer.has_actual_for_month(plant, 202609), d)
        begun = eng.begun
        db_writer.month_start_write_clear(actual(), forecast(2), plant, SEP30)
        check(f"{tag}: month start succeeds in ONE transaction: last month's forecast and "
              'OutstandingOrder gone, the new day-0 forecast and the actual in',
              eng.begun == begun + 1 and eng.rows('OutstandingOrder', plant).empty
              and len(eng.rows('PredictionResult', plant)) == 2
              and (pd.to_datetime(eng.rows('PredictionResult', plant)['OriginDate']) == SEP30).all()
              and db_writer.has_actual_for_month(plant, 202609))
        check(f"{tag}: ...and no other plant's OutstandingOrder was touched",
              all(eng.rows('OutstandingOrder', p).equals(rows) for p, rows in others.items()))

# The real _prediction_phase, with only its Fabric-facing calls and the scorers stubbed.
FEED = {'file_date': pd.Timestamp('2026-10-06'), 'split_asof': False, 'empty': False}


def fake_inference(plant, origin_yyyymm=None, horizons=range(1, 7), *a, cutoff_grain,
                   cutoff_date=None, ctx=None, **k):
    origin = int(to_month_idx(origin_yyyymm))
    ctx = {'max_settled_idx': origin, 'cutoff_grain': cutoff_grain, 'effective_ranges': pd.DataFrame({
        'PartCode': ['SL', 'RD'], 'SalesModelCode': ['A1', 'B1'], 'PlanModelCode': ['PNA1', 'PNB1'],
        'PlanModelText': [None, None], 'StartMonthIdx': [origin - 40.0] * 2, 'EndMonthIdx': [np.nan] * 2})}
    return pd.DataFrame({'OriginIdx': [origin]}), ctx


def fake_scores(method):
    def score(skeleton, *a, **k):
        origin = int(skeleton['OriginIdx'].iloc[0])
        out = pd.DataFrame({'PartCode': ['SL', 'RD'], 'SalesModelCode': ['A1', 'B1'],
                            'PlanModelCode': ['PNA1', 'PNB1'],
                            'TargetYYYYMM': int(month_idx_to_yyyymm([origin + 1])[0]), 'Horizon': 1,
                            'KnownOrder': 100.0, 'KnownCanceledOrder': 0.0, 'Method': method,
                            'Predicted': 120.0, 'OriginYYYYMM': int(month_idx_to_yyyymm([origin])[0]),
                            'CutoffDayIndex': 0})
        if FEED['split_asof']:                      # one row scored at another origin
            out.loc[1, 'OriginYYYYMM'] = int(month_idx_to_yyyymm([origin - 1])[0])
        return out.iloc[0:0] if FEED['empty'] else out
    return score


def fake_actuals(ctx, plant, month_idx=None):
    return pd.DataFrame({'PartCode': ['SL'], 'SalesModelCode': ['A1'], 'PlanModelCode': ['PNA1'],
                         'TargetYYYYMM': [int(month_idx_to_yyyymm([month_idx])[0])],
                         'ActualGrossOrder': [90.0], 'ActualNetOrder': [85.0], 'SalesResult': [np.nan]})


@contextlib.contextmanager
def scoring_stubs():
    stubs = {'preprocess_for_inference': fake_inference, 'score_models': fake_scores('model_production'),
             'score_manual': fake_scores('manual_6'), 'collate_actual_order': fake_actuals}
    saved = {n: getattr(orch, n) for n in stubs}
    for n, f in stubs.items():
        setattr(orch, n, f)
    try:
        yield
    finally:
        for n, f in saved.items():
            setattr(orch, n, f)


def phase(plant, grain='month', trained_through=None):
    return orch._prediction_phase(plant, 'production', range(1, 7), 3, ('SL', 'RD'), 6,
                                  cutoff_grain=grain, file_date=FEED['file_date'],
                                  trained_through=trained_through)


for plant in RUN_PLANTS:
    tag = tag_of(plant)
    eng = TableEngine()
    with db(eng), scoring_stubs():
        FEED['file_date'] = pd.Timestamp('2026-10-06')
        phase(plant)
        check(f'{tag}: the first run of a month writes ProductLifecycle, ActualOrder and the forecast',
              {'ProductLifecycle', 'ActualOrder', 'PredictionResult', 'PredictionResultLog'} <= set(eng.tables)
              and db_writer.has_actual_for_month(plant, 202609))
        actual_rows = eng.rows('ActualOrder', plant)
        phase(plant)
        check(f'{tag}: a second run that month leaves ActualOrder alone (written once a month)',
              eng.rows('ActualOrder', plant).equals(actual_rows))
        before, begun = copy.deepcopy(eng.tables), eng.begun
        FEED['split_asof'] = True
        ok, d = raises(lambda: phase(plant), ValueError, 'different as-of point')
        FEED['split_asof'] = False
        check(f'{tag}: rows from two as-of points are refused before ANY table is read or written',
              ok and eng.tables == before and eng.begun == begun, d)
        eng.fail_on = PR_INSERT
        ok, d = raises(lambda: phase(plant), InjectedFailure)
        eng.fail_on = None
        check(f"{tag}: the forecast write fails inside a real run -> the previous forecast is intact",
              ok and eng.rows('PredictionResult', plant).equals(
                  pd.DataFrame([r for r in before['PredictionResult'] if r['Plant'] == str(plant)])), d)

# ---------------------------------------------------------------------------------------------
print('\n=== B13. PredictionResultLog is write-only: every write appends all its rows ===')
KEY = ['PlanModelCode', 'TargetYearMonth', 'Horizon', 'Method']
VALUES = ['PartCode', 'KnownGrossOrder', 'KnownNetOrder', 'PredictedGrossOrder', 'PredictedNetOrder']


def latest_logged(engine, plant, origin):
    """The most recent Log row per forecast at `origin` - insertion order is Id order."""
    return (at(engine.rows('PredictionResultLog', plant), origin).groupby(KEY).tail(1)
            .set_index(KEY)[VALUES].sort_index())


def current(engine, plant, origin):
    return at(engine.rows('PredictionResult', plant), origin).set_index(KEY)[VALUES].sort_index()


changed = forecast(4)
changed.loc[0, 'Predicted'] += 7.0          # one forecast moves - a retrained model, fixed data

eng = TableEngine()
with db(eng):
    for plant in PLANTS:                    # every plant logs the same numbers on the same day
        db_writer.write_prediction_result(forecast(4), plant, SEP15)
    for plant in PLANTS:
        tag = tag_of(plant)
        others = {p: eng.rows('PredictionResultLog', p) for p in PLANTS if p != plant}
        first = eng.rows('PredictionResultLog', plant)
        db_writer.write_prediction_result(forecast(4), plant, SEP15)
        now = eng.rows('PredictionResultLog', plant)
        check(f'{tag}: the same forecast rescored at the same OriginDate (a forced rerun) appends all '
              '4 rows again, and every earlier row is still there, unchanged',
              len(now) == 8 and now.iloc[:4].equals(first)
              and len(eng.rows('PredictionResult', plant)) == 4, str(len(now)))
        db_writer.write_prediction_result(changed, plant, SEP15)
        check(f'{tag}: a rerun that moves ONE forecast appends all 4 rows (12), so the latest logged '
              'row per forecast is what PredictionResult holds for that day',
              len(eng.rows('PredictionResultLog', plant)) == 12
              and latest_logged(eng, plant, SEP15).equals(current(eng, plant, SEP15)))
        db_writer.write_prediction_result(forecast(4), plant, SEP16)
        check(f'{tag}: a NEW OriginDate is logged beside the earlier one - that is the evolution the '
              'Log exists to show', len(at(eng.rows('PredictionResultLog', plant), SEP16)) == 4
              and len(eng.rows('PredictionResultLog', plant)) == 16)
        check(f"{tag}: another plant's identical rows on the same day never stand in for this "
              "plant's, and are never touched",
              all(eng.rows('PredictionResultLog', p).equals(rows) for p, rows in others.items()))

for plant in RUN_PLANTS:
    tag = tag_of(plant)
    eng = TableEngine()
    with db(eng), scoring_stubs():
        FEED['file_date'] = pd.Timestamp('2026-10-06')
        phase(plant, 'day', trained_through=T)                 # the Oct 6 file: cutoff Oct 5
        first_run = eng.rows('PredictionResultLog', plant)
        phase(plant, 'day', trained_through=T)                 # the same file again (a forced rerun)
        rerun_log = eng.rows('PredictionResultLog', plant)
        check(f'{tag}: a day-grain rerun of the same file rescores the same cutoff and appends its rows '
              'again, earlier rows unchanged',
              len(first_run) > 0 and len(rerun_log) == 2 * len(first_run)
              and rerun_log.iloc[:len(first_run)].equals(first_run))
        FEED['file_date'] = pd.Timestamp('2026-10-07')
        phase(plant, 'day', trained_through=T)
        check(f"{tag}: ...and the next file's run is logged as of its own cutoff date, beside Oct 5",
              len(at(eng.rows('PredictionResultLog', plant), pd.Timestamp('2026-10-06'))) == len(first_run)
              and len(at(eng.rows('PredictionResult', plant), pd.Timestamp('2026-10-05'))) == len(first_run))

# ---------------------------------------------------------------------------------------------
print('\n=== B18. a run that fails or scores nothing writes nothing ===')


def collate_fails(ctx, plant, month_idx=None):
    raise KeyError('actual_order_net')         # any data problem while building ActualOrder


for plant in RUN_PLANTS:
    tag = tag_of(plant)
    eng = TableEngine()
    with db(eng), scoring_stubs():
        FEED['file_date'] = pd.Timestamp('2026-10-06')
        orch.collate_actual_order = collate_fails        # scoring_stubs restores the real one
        ok, d = raises(lambda: phase(plant), KeyError)
        check(f'{tag}: ActualOrder fails to build on the month start -> no table is written at all '
              '(ProductLifecycle is no longer written first)', ok and eng.tables == {}, d)
        orch.collate_actual_order = fake_actuals
        phase(plant)
        before = copy.deepcopy(eng.tables)
        FEED['empty'] = True
        ok, d = raises(lambda: phase(plant), ValueError, 'no rows at all')
        FEED['empty'] = False
        check(f'{tag}: a run that scores no rows says so plainly, and writes nothing',
              ok and eng.tables == before, d)

ctx_a = {'dataset_floor_idx': T - 60, 'launch_by_code': pd.Series({'PNA1': float(T - 30)}),
         'first_demand_by_code': pd.Series({'PNA1': float(T - 30)}),
         'last_demand_by_code': pd.Series({'PNA1': float(T - 15)}),
         'effective_ranges': pd.DataFrame({'PlanModelCode': ['PNA1'], 'EndMonthIdx': [float(T - 12)]}),
         'max_settled_idx': T,
         'actual_order': pd.DataFrame({'PlanModelCode': ['PNA1'], 'RequestMonthIdx': [T - 15],
                                       'ActualOrder': [40.0]}),
         'actual_order_net': pd.DataFrame({'PlanModelCode': ['PNA1'], 'RequestMonthIdx': [T - 15],
                                           'ActualNetOrder': [35.0]}),
         'product_ranges': pd.DataFrame({'ProductCode': ['A1P'], 'PlanModelCode': ['PNA1']}),
         'code_map': pd.DataFrame({'PartCode': ['SL'], 'SalesModelCode': ['A1']},
                                  index=pd.Index(['PNA1'], name='PlanModelCode'))}
real_sales = cao.load_sales_result
cao.load_sales_result = lambda plant, product_pmc_map: pd.DataFrame(
    {'PlanModelCode': pd.Series(dtype=object), 'TargetIdx': pd.Series(dtype='int64'),
     'SalesResult': pd.Series(dtype=float)})
try:
    with captured() as recs:
        none_settled = cao.collate_actual_order(ctx_a, 3300, month_idx=T)   # PNA1 ended before T
    one_settled = cao.collate_actual_order(ctx_a, 3300, month_idx=T - 15)
finally:
    cao.load_sales_result = real_sales
check('collate_actual_order with no code settled in the month: an empty frame with its columns and '
      'a WARNING, not pd.concat\'s "No objects to concatenate"',
      none_settled.empty and set(none_settled.columns) <= set(one_settled.columns) | {'SalesResult'}
      and logged(recs, logging.WARNING, 'no PlanModelCode has a settled window'), str(none_settled.columns))
check('...while a month a code does cover still collates (40 gross, 35 net)',
      len(one_settled) == 1 and one_settled['ActualGrossOrder'].iloc[0] == 40.0
      and one_settled['ActualNetOrder'].iloc[0] == 35.0, str(one_settled))
eng = TableEngine()
with db(eng):
    check('...and write_actual_order skips the empty frame without touching the database',
          db_writer.write_actual_order(none_settled, 3300) == 0 and eng.begun == 0)
ctx_s = {'effective_ranges': pd.DataFrame({'PlanModelCode': pd.Series(dtype=str),
                                           'EndMonthIdx': pd.Series(dtype=float)}),
         'max_settled_idx': T, 'train_first_target_by_code': pd.Series(dtype=float),
         'extend_to_orders': False, 'dataset_floor_idx': T - 60,
         'code_map': pd.DataFrame(columns=['PartCode', 'SalesModelCode']), 'cutoff_grain': 'month'}
ok, d = raises(lambda: pp._lifespan_skeleton(ctx_s), ValueError, 'no PlanModelCode has a target month')
check('a training panel with no code to train on says so, instead of "No objects to concatenate"', ok, d)

# ---------------------------------------------------------------------------------------------
print('\n=== B19. a model never forecasts a negative net order ===')


class FixedModel:
    """Stands in for a fitted model: returns preset predicted Adjustments, in row order."""
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, X):
        return self.values[:len(X)]

    def get_booster(self):
        """No categorical features (only 'x'), so models._unseen_categories_to_nan changes nothing."""
        empty = type('Cats', (), {'to_arrow': lambda self: []})()
        return type('Booster', (), {'get_categories': lambda self, export_to_arrow: empty})()


skel = pd.DataFrame({'PlanModelCode': ['PNA1', 'PNB1', 'PNC1', 'PND1'],
                     'SalesModelCode': ['A1', 'B1', 'C1', 'D1'], 'PartCode': 'XX', 'Horizon': 1,
                     'OriginIdx': T, 'TargetIdx': T + 1, 'KnownOrder': [100.0, 50.0, 10.0, 100.0],
                     'KnownCanceledOrder': [0.0, 0.0, 0.0, 80.0], 'x': 0.0,
                     'ManualSim6': [-4.0, 2.0, 0.0, 0.0]})
# A1: net 70 (allowed); C1: net -5; D1: gross 70 but net 20 - 30 = -10 (both impossible)
raw = [-30.0, 0.0, -15.0, -30.0]
loaded = {1: (FixedModel(raw), ['x'], {'model_type': 'xgboost', 'cutoff_grain': 'month',
                                        'label': 'production', 'horizon': 1})}
with captured(logging.INFO) as recs:
    scored = pm._score_with_models(skel, [1], loaded, 'model_production').set_index('PlanModelCode')
check('a negative Adjustment is kept while net stays >= 0 (A1: 100 - 30 = 70); a negative net is '
      'floored to 0 (C1: 10 - 15 -> 0; D1: gross 100 - 20 = 80, net 0); B1 untouched (50)',
      scored['Predicted'].to_dict() == {'PNA1': 70.0, 'PNB1': 50.0, 'PNC1': 0.0, 'PND1': 80.0},
      str(scored['Predicted'].to_dict()))
check('...and every run logs how often it happened: 2 of 4, lowest -10.0, 15 units added',
      logged(recs, logging.INFO, 'model_production h1', '2 of 4', 'lowest -10.0', 'adding 15 unit'),
      str([r.getMessage() for r in recs if 'floored' in r.getMessage()]))
manual = pm.score_manual(skel, 6).set_index('PlanModelCode')
check("the manual benchmark is NOT floored - it reproduces the business's own method as it is "
      '(A1: 100 - 4 = 96)', manual.loc['PNA1', 'Predicted'] == 96.0, str(manual['Predicted'].to_dict()))

# ---------------------------------------------------------------------------------------------
print('\n=== B7. a failed retrain still scores (with the incumbent); OutstandingOrder only beside a '
      'current forecast ===')
from xgboost import XGBRegressor  # noqa: E402
TINY = XGBRegressor(n_estimators=1, max_depth=1).fit(pd.DataFrame({'a': [0.0, 1.0]}), [0.0, 1.0])
CLIENT = mr._client()


def deploy(plant, label, trained_through, grain):
    """Registers both halves of a deployment and points the alias at them; returns the alias map."""
    for lbl in mr.deployment_labels(label):
        for h in range(1, 7):
            e = mr.save_model(TINY, plant=plant, horizon=h, feature_cols=['a'], params={},
                              covid_weight=.1, young_sku_weight=1., external_event_weight=1.,
                              label=lbl, trained_through=trained_through, cutoff_grain=grain,
                              deployment_id='test')
            CLIENT.set_registered_model_alias(e['registered_name'], mr.PRODUCTION_ALIAS, e['version'])
    return aliases(plant, label)


def aliases(plant, label):
    return {(lbl, h): mr._aliased_version(CLIENT, mr._registered_name(plant, h, lbl))
            for lbl in mr.deployment_labels(label) for h in range(1, 7)}


def file_for(settled):
    """A source file implying `settled`: the 6th of the month after it."""
    ym = int(month_idx_to_yyyymm([settled + 1])[0])
    day = pd.Timestamp(f'{ym // 100}-{ym % 100:02d}-06')
    return day, day.strftime('%Y%m%d') + '.xlsx'


def run(plant, label, *, train=None, predict=None, outstanding=None, settled=T, score='stub',
        eng=None, predicted=False, processed=False, **kw):
    """One real run_daily. Retrain, ctx build and OutstandingOrder are stubbed, and so is scoring
    unless score='real'. `train`/`predict`/`outstanding`, if given, run inside the matching stub -
    to raise, or to register a deployment. Returns (error, the phases that ran, WARNING+ records)."""
    ran = []
    file_date, source_file = file_for(settled)
    eng = eng or TableEngine()
    if predicted:     # this month's month-end forecast is already written
        eng.tables.setdefault('PredictionResult', []).append(
            {'Plant': str(plant), 'OriginDate': orch._origin_date_for(settled, 'month', None)})
    if processed:     # this file's OutstandingOrder is already written
        eng.tables.setdefault('OutstandingOrder', []).append(
            {'Plant': str(plant), 'SourceFileName': source_file})

    def train_phase(plant, label, covid_weight, val_months, ctx, *, cutoff_grain):
        ran.append(('train', cutoff_grain))
        if train:
            train(cutoff_grain)

    def prediction_phase(*a, cutoff_grain, trained_through=None, **k):
        ran.append(('predict', cutoff_grain, trained_through))
        if predict:
            predict()
        return {'max_settled_idx': settled}

    def outstanding_phase(plant, file_date, source_file, settled_month_idx):
        ran.append(('outstanding', settled_month_idx))
        if outstanding:
            outstanding()

    stubs = {'_train_phase': train_phase, '_outstanding_order_phase': outstanding_phase,
             '_build_run_ctx': lambda plant, *, cutoff_grain, settled_idx: {
                 'cutoff_grain': cutoff_grain, 'max_settled_idx': settled_idx},
             'latest_source_file_date': lambda p: (file_date, source_file)}
    if score == 'stub':
        stubs['_prediction_phase'] = prediction_phase
    saved = {n: getattr(orch, n) for n in stubs}
    for n, f in stubs.items():
        setattr(orch, n, f)
    err = None
    with captured() as recs, db(eng):
        try:
            orch.run_daily(plant, label, **kw)
        except Exception as e:
            err = e
        finally:
            for n, f in saved.items():
                setattr(orch, n, f)
    return err, ran, recs


def dies(exc):
    def fail(*_a):
        raise exc
    return fail


def failed(err, *phases, cause=None) -> bool:
    return (isinstance(err, orch.DailyRunError) and [p for p, _e in err.failures] == list(phases)
            and (cause is None or isinstance(err.__cause__, cause)))


for plant in RUN_PLANTS:
    tag = tag_of(plant)
    for grain in ('day', 'month'):
        label = f'stale_{grain}'
        reg = deploy(plant, label, T - 1, grain)
        err, ran, recs = run(plant, label, train=dies(MemoryError('simulated OOM in the retrain')))
        check(f'{tag} {grain}: the month-close retrain dies -> the file is still scored by the '
              f'incumbent at {grain} grain (trained through the month before), then OutstandingOrder',
              ran == [('train', grain), ('predict', grain, T - 1), ('outstanding', T)], str(ran))
        check(f'{tag} {grain}: ...and the run still fails: DailyRunError, chained to the MemoryError',
              failed(err, 'retrain', cause=MemoryError), repr(err))
        check(f'{tag} {grain}: the failure is logged at ERROR, with its traceback, when it happens',
              any(r.levelno == logging.ERROR and r.exc_info and 'FAILED' in r.getMessage() for r in recs))
        check(f'{tag} {grain}: the deployment is untouched, so the next file is still stale',
              aliases(plant, label) == reg)
        err, ran, _ = run(plant, label, train=lambda g: deploy(plant, label, T, g))
        check(f'{tag} {grain}: ...the next file retrains, and once that succeeds it scores from the '
              'new deployment', err is None
              and ran == [('train', grain), ('predict', grain, T), ('outstanding', T)], f'{err!r} {ran}')

    reg = deploy(plant, 'switch', T, 'month')
    err, ran, _ = run(plant, 'switch', train=dies(MemoryError('quantile fit OOM')), cutoff_grain='day',
                      predicted=True)
    check(f"{tag}: a month -> day switch that dies: this month's month-grain forecast is already "
          'written, so it is not rescored; OutstandingOrder refreshes, DailyRunError',
          ran == [('train', 'day'), ('outstanding', T)] and failed(err, 'retrain', cause=MemoryError)
          and aliases(plant, 'switch') == reg, f'{ran} {err!r}')

    err, ran, recs = run(plant, 'first', train=dies(ValueError('bad label')), cutoff_grain='day')
    check(f'{tag}: the FIRST deployment fails: nothing to score, so no OutstandingOrder either, DailyRunError',
          ran == [('train', 'day')] and failed(err, 'retrain', cause=ValueError)
          and logged(recs, logging.WARNING, 'see the retrain failure above'), f'{ran} {err!r}')

    deploy(plant, 'wrong', T - 1, 'day')
    err, ran, _ = run(plant, 'wrong', train=lambda g: deploy(plant, 'wrong', T, 'month'))
    check(f'{tag}: a retrain that lands at the wrong grain: nothing is scored from it, so no '
          'OutstandingOrder either, DailyRunError', ran == [('train', 'day')]
          and failed(err, 'retrain') and 'refusing to score' in str(err), f'{ran} {err!r}')

    deploy(plant, 'current', T, 'day')
    err, ran, _ = run(plant, 'current', predict=dies(ConnectionError('Fabric dropped the connection')))
    check(f'{tag}: scoring fails -> OutstandingOrder is NOT refreshed, so it stays aligned with the '
          'previous forecast and the file stays unprocessed (retried next run), DailyRunError',
          ran == [('predict', 'day', T)] and failed(err, 'prediction', cause=ConnectionError), f'{ran} {err!r}')
    err, ran, _ = run(plant, 'current', outstanding=dies(InjectedFailure('OutstandingOrder INSERT failed')))
    check(f'{tag}: OutstandingOrder fails -> the forecast was still written first, DailyRunError',
          ran == [('predict', 'day', T), ('outstanding', T)]
          and failed(err, 'OutstandingOrder', cause=InjectedFailure), f'{ran} {err!r}')
    err, ran, _ = run(plant, 'current')
    check(f'{tag}: a clean file raises nothing', err is None
          and ran == [('predict', 'day', T), ('outstanding', T)], f'{ran} {err!r}')
    err, ran, _ = run(plant, 'current', processed=True)
    check(f'{tag}: a file already processed does nothing at all, and raises nothing',
          err is None and ran == [], f'{ran} {err!r}')

    deploy(plant, 'both', T - 1, 'day')
    err, ran, _ = run(plant, 'both', train=dies(MemoryError('oom')), predict=dies(ConnectionError('down')))
    check(f'{tag}: retrain AND scoring fail -> both reported, chained to the first, no OutstandingOrder',
          failed(err, 'retrain', 'prediction', cause=MemoryError)
          and not any(p[0] == 'outstanding' for p in ran), f'{ran} {err!r}')

print('\n=== B7b. main() runs every plant, then exits non-zero if any failed ===')
env = {'DAILY_PLANTS': ','.join(map(str, RUN_PLANTS)), 'DAILY_LABEL': 'production',
       'DAILY_COVID_WEIGHT': '0.1', 'DAILY_VAL_MONTHS': '6', 'DAILY_HORIZONS': '1,2,3,4,5,6',
       'DAILY_MIN_ORIGIN_MONTHS_SINCE_LAUNCH': '3', 'DAILY_MANUAL_WINDOW': '6'}
saved_env = {k: os.environ.get(k) for k in env}
os.environ.update(env)
attempted = []


def run_daily_first_fails(plant, *a, **k):
    attempted.append(plant)
    if plant == RUN_PLANTS[0]:
        raise orch.DailyRunError(plant, [('retrain', MemoryError('oom'))])


saved_fns = {n: getattr(orch, n) for n in ('run_daily', 'configure_logging')}
orch.run_daily, orch.configure_logging = run_daily_first_fails, lambda *a, **k: None
try:
    try:
        orch.main()
        exit_err = None
    except SystemExit as e:
        exit_err = e
finally:
    for n, f in saved_fns.items():
        setattr(orch, n, f)
    for k, v in saved_env.items():
        os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)
check('the first plant failing does not stop the next one from running',
      attempted == RUN_PLANTS, str(attempted))
check('...and the process exits non-zero naming the failed plant, so the scheduler sees it',
      exit_err is not None and exit_err.code not in (0, None) and str(RUN_PLANTS[0]) in str(exit_err.code),
      repr(exit_err))

# ---------------------------------------------------------------------------------------------
print('\n=== B8.1 the latest source file is the newest parseable NAME ===')
NOW = pd.Timestamp('2026-10-01 09:30')


def files(*rows):
    return pd.DataFrame(list(rows), columns=['_source_file', 'FileDate'])


def old_pick(df):
    """The pre-fix rule: the one newest file by FileDate, used only if its name parses."""
    name = str(df.sort_values('FileDate').iloc[-1]['_source_file'])
    m = re.match(r'^(\d{8})\.xlsx$', name, re.IGNORECASE)
    return (pd.Timestamp(m.group(1)), name) if m else (None, None)


def pick(df, now=NOW):
    with captured() as recs:
        got = pp._pick_latest_source_file(df, now=now)
    return got, recs


DAILY = [('20260929.xlsx', '2026-09-29 06:00'), ('20260930.xlsx', '2026-09-30 06:00'),
         ('20261001.xlsx', '2026-10-01 06:00')]
OCT1 = (pd.Timestamp('2026-10-01'), '20261001.xlsx')

got, recs = pick(files(*DAILY))
check('plain daily files: the newest, no warning - and the old rule agrees',
      got == OCT1 and not recs and old_pick(files(*DAILY)) == got, f'{got} {[r.getMessage() for r in recs]}')
misnamed = files(*DAILY[:2], ('Orders_20261001.xlsx', '2026-10-01 06:00'))
got, recs = pick(misnamed)
check('a misnamed newest file: the old rule found NO date (retraining never fired, day grain raised)',
      old_pick(misnamed) == (None, None))
check('...now the newest parseable name, 20260930.xlsx, and a WARNING names the misnamed file',
      got == (pd.Timestamp('2026-09-30'), '20260930.xlsx')
      and logged(recs, logging.WARNING, 'Orders_20261001.xlsx', "doesn't follow"), str(got))
reingested = files(*DAILY, ('20260915.xlsx', '2026-10-01 11:00'))
got, recs = pick(reingested)
check('an older file re-ingested late: the old rule moved the settled month BACKWARDS, 202609 -> 202608',
      old_pick(reingested)[1] == '20260915.xlsx'
      and pp._settled_month_for_file_date(old_pick(reingested)[0]) == T - 1)
check('...now still 20261001.xlsx (settled 202609), with an out-of-order WARNING naming the file',
      got == OCT1 and pp._settled_month_for_file_date(got[0]) == T
      and logged(recs, logging.WARNING, 'out of order', '20260915.xlsx'), str(got))
typo = files(*DAILY, ('20621001.xlsx', '2026-10-01 07:00'))
got, recs = pick(typo)
check('a name dated 2062 (a typo): the old rule would have settled month 206209',
      old_pick(typo)[0] == pd.Timestamp('2062-10-01'))
check('...now it is ignored with a WARNING, and 20261001.xlsx is used',
      got == OCT1 and logged(recs, logging.WARNING, '20621001.xlsx', 'ignored'), str(got))
got, recs = pick(files(*DAILY, ('20261341.xlsx', '2026-10-01 07:00')))
check("eight digits that aren't a date (month 13) count as an unparseable name, with a WARNING",
      got == OCT1 and logged(recs, logging.WARNING, '20261341.xlsx'), str(got))
got, _ = pick(files(('20260930.xlsx', '2026-09-30'), ('20261001.XLSX', '2026-10-01')))
check('the extension still matches case-insensitively', got == (pd.Timestamp('2026-10-01'), '20261001.XLSX'))
got, _ = pick(files(*DAILY[:2], ('20250101 - 20260930.xlsx', '2026-10-01 06:00')))
check('a range name "YYYYMMDD - YYYYMMDD.xlsx" reads as a file dated its end + 1 (2026-10-01)',
      got == (pd.Timestamp('2026-10-01'), '20250101 - 20260930.xlsx'), str(got))
got, recs = pick(files(('history.xlsx', '2026-01-01'), ('Orders_20261001.xlsx', '2026-10-01')))
check('no parseable name at all: (None, None) with a WARNING',
      got == (None, None) and logged(recs, logging.WARNING, 'No source file follows'))
got, recs = pick(files())
check('no files at all: (None, None) with a WARNING',
      got == (None, None) and logged(recs, logging.WARNING, 'No source files found'))

print('\n=== B8.2 latest_source_file_date and _is_stale end to end (pd.read_sql stubbed) ===')
TODAY = pd.Timestamp.now().normalize()
day_name = lambda days_ago: (TODAY - pd.Timedelta(days=days_ago)).strftime('%Y%m%d') + '.xlsx'
VIEW = {'files': None}
sent = []


def fake_read_sql(sql, engine, params=None, **kwargs):
    sent.append((str(sql), params, engine))
    return VIEW['files'].copy()


real_read_sql = pd.read_sql
pd.read_sql = fake_read_sql
try:
    VIEW['files'] = files((day_name(2), TODAY - pd.Timedelta(days=2)),
                          (day_name(1), TODAY - pd.Timedelta(days=1)), (day_name(0), TODAY),
                          (day_name(20), TODAY + pd.Timedelta(hours=3)))     # re-ingested late
    for plant in PLANTS:
        got = pp.latest_source_file_date(plant)
        sql, params, engine = sent[-1]
        check(f'{tag_of(plant)}: the newest NAME wins over a file re-ingested later; one query, one '
              'row per file, for this plant only', got == (TODAY, day_name(0))
              and 'GROUP BY _source_file' in sql and params == {'plant': plant}
              and engine is pp._fabric_engine, f'{got} {params}')
    VIEW['files'] = files((day_name(1), TODAY - pd.Timedelta(days=1)), ('Orders_' + day_name(0), TODAY))
    settled_now = pp._settled_month_for_file_date(TODAY - pd.Timedelta(days=1))
    behind = {'trained_through': settled_now - 1, 'cutoff_grain': 'day', 'labels': {}, 'horizons': []}
    check('with a misnamed newest file the old rule had no date, so the staleness test said "not stale"',
          old_pick(VIEW['files']) == (None, None))
    for plant in PLANTS:
        file_date, _name = pp.latest_source_file_date(plant)
        check(f'{tag_of(plant)}: ...now a deployment a month behind IS stale, so retraining fires',
              orch._is_stale(pp._settled_month_for_file_date(file_date), behind) is True)
finally:
    pd.read_sql = real_read_sql

print('\n=== B8.3 an origin behind the deployment is refused, before anything is written ===')
for plant in RUN_PLANTS:
    tag = tag_of(plant)
    for grain in ('day', 'month'):
        eng = TableEngine()
        with db(eng), scoring_stubs():
            FEED['file_date'] = pd.Timestamp('2026-10-06')                  # settled 202609
            phase(plant, grain, trained_through=T)
            cur, begun = eng.rows('PredictionResult', plant), eng.begun
            FEED['file_date'] = pd.Timestamp('2026-09-21')                  # the October files are gone
            ok, d = raises(lambda: phase(plant, grain, trained_through=T), ValueError, 'BACKWARDS')
            check(f'{tag} {grain}: origin 202608 under a deployment trained through 202609 is '
                  'refused, and no table is touched', ok and eng.begun == begun
                  and eng.rows('PredictionResult', plant).equals(cur), d)
            phase(plant, grain, trained_through=None)
            check(f'{tag} {grain}: without the guard the older origin really is written (so the check '
                  'above is not vacuous)', (pd.to_datetime(eng.rows('PredictionResult', plant)['OriginDate'])
                                            < pd.to_datetime(cur['OriginDate']).min()).any())
            FEED['file_date'] = pd.Timestamp('2026-10-06')
            ok_same = phase(plant, grain, trained_through=T)['max_settled_idx'] == T
            ok_ahead = phase(plant, grain, trained_through=T - 1)['max_settled_idx'] == T
            check(f'{tag} {grain}: the same month, and one month ahead (a failed retrain), are scored',
                  ok_same and ok_ahead)

    deploy(plant, 'behind', T, 'day')
    eng = TableEngine()
    with db(eng), scoring_stubs():
        FEED['file_date'] = pd.Timestamp('2026-10-06')
        phase(plant, 'day', trained_through=T)
        cur = eng.rows('PredictionResult', plant)
        err, ran, _ = run(plant, 'behind', settled=T - 1, score='real', eng=eng)
    check(f'{tag}: run_daily on a feed gone backwards: no retrain, scoring refused (the Oct 5 '
          'forecast stays) and so does OutstandingOrder, DailyRunError',
          failed(err, 'prediction', cause=ValueError) and ran == []
          and eng.rows('PredictionResult', plant).equals(cur), f'{ran} {err!r}')

print('\n' + '=' * 78)
print(f'{len(FAILURES)} FAILURE(S): {FAILURES}' if FAILURES else 'ALL CHECKS PASSED')
sys.exit(1 if FAILURES else 0)
