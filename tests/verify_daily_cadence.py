"""The orchestrator's cadences, and what each run persists.

run_daily runs once per NEW source file (OutstandingOrder holding rows from it marks it done):
retrain when the file's settled month is newer than the deployment's trained_through, predict
every new file at day grain (at month grain only when that month's forecast isn't written yet, or
just retrained), and refresh OutstandingOrder alongside a current forecast. The first write of a
month (no ActualOrder for N yet) upserts the actual and clears last month's PredictionResult and
OutstandingOrder in the same transaction as the prediction.

Three questions used to read one value, PredictionResult.OriginDate; each now has its own source:

    who asks                        used to read                    now reads
    ----------------------------------------------------------------------------------------
    "is the model stale?"           PredictionResult.OriginDate     MLflow trained_through
    "which month is N+1?"           PredictionResult.OriginDate     explicit settled_month_idx
    "is this a new month?"          a retrain having happened       ActualOrder has rows for N

Section 2 shows why the first matters: under a daily OriginDate the old test says "not stale"
forever. Sections 6-8 cover the DB side: the statements each writer sends, and the SQL files
against what Python INSERTs. No database; the registry is the throwaway MLflow store (no_db.py).
"""
import inspect
import pathlib
import re
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'tests'))
import no_db  # noqa: E402,F401 - every DB engine raises from here on (see no_db.py)

import main as orchestrator  # noqa: E402
from adj_forecast_lib import db_writer, model_registry as mr  # noqa: E402
from adj_forecast_lib.collate_outstanding_order import collate_outstanding_order  # noqa: E402
from adj_forecast_lib.config import month_idx_to_yyyymm, to_month_idx  # noqa: E402
from adj_forecast_lib.cutoff_grain import cutoff_day_index  # noqa: E402

FAILURES = []


def check(label, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' - ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(label)


SETTLED = int(to_month_idx(202608))    # the newest settled month the source file implies
FILE = (pd.Timestamp('2026-09-15'), '20260915.xlsx')

print('\n=== 1. deployed_state reads the MLflow registry ===')
from xgboost import XGBRegressor  # noqa: E402
TINY = XGBRegressor(n_estimators=1, max_depth=1).fit(pd.DataFrame({'a': [0.0, 1.0]}), [0.0, 1.0])
CLIENT = mr._client()


def register(label, rows, plant=3300):
    """Registers one version per (horizon, trained_through, grain) and points the alias at it."""
    for h, through, grain in rows:
        e = mr.save_model(TINY, plant=plant, horizon=h, feature_cols=['a'], params={},
                          covid_weight=.1, young_sku_weight=1., external_event_weight=1.,
                          label=label, trained_through=through, cutoff_grain=grain,
                          deployment_id='test')
        CLIENT.set_registered_model_alias(e['registered_name'], mr.PRODUCTION_ALIAS, e['version'])


register('m1', [(h, SETTLED - 1, 'month') for h in range(1, 7)])
st = mr.deployed_state(plant=3300, label='m1')
check('trained_through + grain + horizons',
      st['trained_through'] == SETTLED - 1 and st['cutoff_grain'] == 'month'
      and st['horizons'] == [1, 2, 3, 4, 5, 6], str(st))
check('unknown label -> None', mr.deployed_state(plant=3300, label='nope') is None)
check("another plant's models are not read -> None", mr.deployed_state(plant=2100, label='m1') is None)
register('m_half', [(1, SETTLED, 'month')] + [(h, SETTLED - 1, 'month') for h in range(2, 7)])
check('a half-failed retrain reports the STALE horizon, not the fresh one',
      mr.deployed_state(plant=3300, label='m_half')['trained_through'] == SETTLED - 1)
register('m_mixed', [(1, SETTLED, 'day'), (2, SETTLED, 'month')])
try:
    mr.deployed_state(plant=3300, label='m_mixed')
    check('mixed grains under one label raises', False, 'no exception')
except ValueError as e:
    check('mixed grains under one label raises', 'mixed as-of grains' in str(e))

print('\n=== 2. the staleness test that daily OriginDate would have broken ===')
# What the OLD check did: month-ordinal of PredictionResult's MAX(OriginDate).
old_is_new = lambda new_idx, origin: new_idx > to_month_idx(origin.year * 100 + origin.month)
settled_yyyymm = int(month_idx_to_yyyymm([SETTLED - 1])[0])
month_end_origin = pd.Timestamp(year=settled_yyyymm // 100, month=settled_yyyymm % 100,
                                day=1) + pd.offsets.MonthEnd(0)
check('OLD test worked while OriginDate was the settled month-end', old_is_new(SETTLED, month_end_origin))
daily_origin = month_end_origin + pd.Timedelta('9D')
check('OLD test reports "not stale" once OriginDate goes daily - retraining would never fire again',
      not old_is_new(SETTLED, daily_origin))
check('NEW test still reports stale, because it asks the model, not the dashboard',
      orchestrator._is_stale(SETTLED, {'trained_through': SETTLED - 1}))
check('NEW test reports fresh once the retrain has landed',
      not orchestrator._is_stale(SETTLED, {'trained_through': SETTLED}))
check('nothing deployed -> stale (train the first deployment)', orchestrator._is_stale(SETTLED, None))
check('no parseable source file -> never stale (nothing to act on)',
      not orchestrator._is_stale(None, {'trained_through': SETTLED - 1}))

print('\n=== 3. the cadence matrix ===')


class FakeDB:
    def __init__(self, processed, predicted):
        self.processed, self.predicted, self.calls = processed, predicted, []

    def has_outstanding_for_file(self, plant, source_file):
        return self.processed

    def has_prediction_at(self, plant, origin_date):
        return self.predicted

    def __getattr__(self, name):
        def rec(*a, **k):
            self.calls.append(name)
            return None
        return rec


def run_case(grain, trained_through, processed=False, predicted=False, **kw):
    """Runs run_daily with every side effect stubbed, and returns what it decided to do."""
    st = {'trained_through': trained_through, 'cutoff_grain': grain, 'horizons': list(range(1, 7)),
          'labels': {'production': grain, 'production_quantile': grain}}
    log, fake = [], FakeDB(processed, predicted)
    names = ('db_writer', 'deployed_state', 'latest_source_file_date', '_build_run_ctx',
             '_train_phase', '_prediction_phase', '_outstanding_order_phase')
    orig = {k: getattr(orchestrator, k) for k in names}

    def train(plant, label, covid_weight, val_months, ctx, *, cutoff_grain):
        # Records the grain the retrain was ASKED for - omitting it once turned day deployments
        # into month ones - and lands the retrain where deployed_state will read it back.
        log.append(f'train:{cutoff_grain}')
        st.update(trained_through=SETTLED, cutoff_grain=cutoff_grain)
    orchestrator.db_writer = fake
    orchestrator.deployed_state = lambda **k: dict(st)
    orchestrator.latest_source_file_date = lambda plant: FILE
    orchestrator._build_run_ctx = lambda plant, *, cutoff_grain, settled_idx: {
        'cutoff_grain': cutoff_grain, 'max_settled_idx': settled_idx}
    orchestrator._train_phase = train
    orchestrator._prediction_phase = lambda *a, **k: (
        log.append(f"predict:{k['cutoff_grain']}") or {'max_settled_idx': SETTLED})
    orchestrator._outstanding_order_phase = lambda plant, file_date, source_file, settled_month_idx: \
        log.append(f'outstanding:{settled_month_idx}')
    try:
        orchestrator.run_daily(3300, 'production', **kw)
    finally:
        for k, v in orig.items():
            setattr(orchestrator, k, v)
    return log, fake.calls


O = f'outstanding:{SETTLED}'
cases = {
    'month, new month': (run_case('month', SETTLED - 1), ['train:month', 'predict:month', O]),
    'month, new month, month-end forecast already there': (
        run_case('month', SETTLED - 1, predicted=True), ['train:month', 'predict:month', O]),
    'month, same month, forecast written': (run_case('month', SETTLED, predicted=True), [O]),
    'month, same month, forecast missing (day 0 failed)': (
        run_case('month', SETTLED, predicted=False), ['predict:month', O]),
    'day, new month': (run_case('day', SETTLED - 1), ['train:day', 'predict:day', O]),
    'day, same month': (run_case('day', SETTLED, predicted=True), ['predict:day', O]),
    'day, file already processed': (run_case('day', SETTLED - 1, processed=True), []),
    'month, file already processed': (run_case('month', SETTLED - 1, processed=True), []),
    'day, file already processed, DAILY_FORCE_RUN': (
        run_case('day', SETTLED, processed=True, force_run=True), ['predict:day', O]),
    'month, forecast written, DAILY_FORCE_RUN': (
        run_case('month', SETTLED, predicted=True, force_run=True), ['predict:month', O]),
    'day, same month, DAILY_FORCE_RETRAIN': (
        run_case('day', SETTLED, processed=True, force_retrain=True), ['train:day', 'predict:day', O]),
}
for name, ((log, _calls), want) in cases.items():
    print(f"         {name:52s} -> {log}")
    check(f'{name}: {want}', log == want, f'got {log}')
check('the settled month reaches OutstandingOrder explicitly, never inferred from a date',
      all(s == O for (log, _c), _w in cases.values() for s in log if s.startswith('outstanding')))
check('no case issues a separate clear - each table is cleared inside its own write',
      not any(n.startswith('clear_') for (_l, calls), _w in cases.values() for n in calls),
      str({k: c for k, ((_l, c), _w) in cases.items()}))

print('\n=== 4. _origin_date_for ===')
cutoff = pd.Timestamp('2026-09-15')
check('day grain stamps the real as-of date, so two days are distinguishable',
      orchestrator._origin_date_for(SETTLED, 'day', cutoff) == cutoff)
check('month grain keeps the settled month-end, unchanged',
      orchestrator._origin_date_for(SETTLED, 'month', None) == pd.Timestamp('2026-08-31'))
check('day grain with no date falls back to the month-end rather than stamping None',
      orchestrator._origin_date_for(SETTLED, 'day', None)
      == orchestrator._origin_date_for(SETTLED, 'month', None))

print('\n=== 5. collate_outstanding_order: N+1 explicit vs inferred ===')
orders = pd.DataFrame({
    'PlanModelCode': ['PNA', 'PNA', 'PNB', 'PNB'], 'SalesModelCode': ['A', 'A', 'B', 'B'],
    'PartCode': ['P'] * 4,
    'RequestDate': pd.to_datetime(['2026-08-10', '2026-09-10', '2026-08-20', '2026-10-05']),
    'Qty': [10.0, 20.0, 30.0, 40.0],
    'IsCanceled': [False] * 4, 'IsSold': [False] * 4, 'CancelDate': pd.NaT,
})
file_date = pd.Timestamp('2026-09-30')
aug = int(to_month_idx(202608))
explicit = collate_outstanding_order(orders, file_date, 'f.xlsx', 3300, settled_month_idx=aug)
check("explicit settled month -> N+1 is Sep, so only orders requested before Sep are outstanding",
      explicit['TargetYearMonth'].iloc[0] == '202609'
      and explicit.set_index('PlanModelCode')['OutstandingOrder'].to_dict() == {'PNA': 10.0, 'PNB': 30.0})
check("no settled month -> N+1 is the file date's own month, the same answer",
      collate_outstanding_order(orders, file_date, 'f.xlsx', 3300)['TargetYearMonth'].iloc[0] == '202609')


def old_inferred_n_plus_1(last_prediction_origin):
    """The removed fallback: N+1 inferred by stepping one day past PredictionResult's MAX(OriginDate)."""
    next_day = last_prediction_origin + pd.Timedelta(days=1)
    return str(int(month_idx_to_yyyymm([to_month_idx(next_day.year * 100 + next_day.month)])[0]))


check('the removed date-inference landed on N+2 on the last day of month N+1',
      old_inferred_n_plus_1(pd.Timestamp('2026-09-30')) == '202610' != explicit['TargetYearMonth'].iloc[0])
check('...and agreed on every other cutoff day, which is how it went unnoticed',
      old_inferred_n_plus_1(pd.Timestamp('2026-09-15'))
      == old_inferred_n_plus_1(pd.Timestamp('2026-08-31')) == '202609')
main_src = (_REPO_ROOT / 'main.py').read_text(encoding='utf-8')
check('the fallback is gone: collate_outstanding_order takes no last_prediction_origin, and main.py '
      'never reads PredictionResult to find N',
      'last_prediction_origin' not in inspect.signature(collate_outstanding_order).parameters
      and 'get_last_prediction_origin' not in main_src)

print('\n=== 6. the day index derives from the cutoff date alone ===')
pairs = [(pd.Timestamp('2026-01-31'), 0),     # a month end = day 0 (that month just settled)
         (pd.Timestamp('2026-02-01'), 1),
         (pd.Timestamp('2026-02-15'), 15),
         (pd.Timestamp('2026-02-27'), 27),    # last day of the Jan origin's window
         (pd.Timestamp('2026-02-28'), 0)]     # Feb 28 is the Feb origin's day 0
got = list(cutoff_day_index(pd.DataFrame({'CutoffDate': [d for d, _w in pairs]})))
for (date, want), g in zip(pairs, got):
    check(f'{date.date()} -> day {want}, no origin needed', g == want, f'got {g}')

print('\n=== 7. what each writer actually sends ===')


class FakeEngine:
    """Records every statement and its parameters; counts transactions."""
    def __init__(self):
        self.sent, self.begun = [], 0

    def begin(self):
        import contextlib
        self.begun += 1
        sink = self.sent

        class R:
            def __init__(self, rows):
                self.rowcount = len(rows) if isinstance(rows, list) else 0

            def fetchall(self):
                return []           # nothing already logged

            def first(self):
                return None

        class Conn:
            def execute(self, stmt, rows=None):
                sink.append((' '.join(str(stmt).split()), rows))
                return R(rows)

        @contextlib.contextmanager
        def cm():
            yield Conn()
        return cm()


def sent_by(fn, *args):
    eng, real = FakeEngine(), db_writer._engine
    db_writer._engine = eng
    try:
        fn(*args)
    finally:
        db_writer._engine = real
    kinds = lambda verb: [(s, p) for s, p in eng.sent if s.startswith(verb)]
    return eng, kinds('DELETE'), kinds('INSERT'), kinds('SELECT'), kinds('MERGE')


result_df = pd.DataFrame({
    'PartCode': ['SL'], 'SalesModelCode': ['A1'], 'PlanModelCode': ['PNA1'], 'TargetYYYYMM': [202610],
    'Horizon': [1], 'KnownOrder': [100.0], 'KnownCanceledOrder': [5.0], 'Method': ['model_v1'],
    'Predicted': [120.0]})
actual_df = pd.DataFrame({'PartCode': ['SL'], 'PlanModelCode': ['PNA1'], 'TargetYYYYMM': [202608],
                          'ActualGrossOrder': [90.0], 'ActualNetOrder': [85.0], 'SalesResult': [np.nan]})
outstanding_df = collate_outstanding_order(orders, file_date, '20260930.xlsx', 3300, settled_month_idx=aug)
day = pd.Timestamp('2026-09-15')

eng, deletes, inserts, selects, _m = sent_by(db_writer.write_prediction_result, result_df, 3300, day)
check('daily write: replaces only rows at THIS OriginDate (a rerun), inserts PredictionResult + '
      'Log, in ONE transaction',
      eng.begun == 1 and len(deletes) == 1
      and deletes[0][0].endswith('.PredictionResult WHERE Plant = :plant AND OriginDate = :origin_date')
      and deletes[0][1] == {'plant': '3300', 'origin_date': day}
      and [re.search(r'\.(\w+) \(', s).group(1) for s, _r in inserts] == ['PredictionResult',
                                                                          'PredictionResultLog'],
      str(eng.sent))
check('...and reads nothing: the Log is write-only', not selects, str(selects))
sent_rows = inserts[0][1]
check('OriginDate is the cutoff date, and nothing derivable from it is stored beside it',
      pd.Timestamp(sent_rows[0]['OriginDate']) == day
      and not {'SettledYearMonth', 'CutoffDayIndex'} & set(sent_rows[0]))
check('Net = Gross minus known-canceled, for both Known and Predicted',
      sent_rows[0]['KnownNetOrder'] == 95.0 and sent_rows[0]['PredictedNetOrder'] == 115.0)

eng, deletes, inserts, _s, merges = sent_by(db_writer.month_start_write_clear, actual_df, result_df,
                                            3300, day)
check('first write of the month: ActualOrder upsert, BOTH tables cleared plant-wide, prediction '
      'inserted - all in ONE transaction',
      eng.begun == 1 and len(merges) == 1 and 'ActualOrder' in merges[0][0]
      and sorted(re.search(r'\.(\w+) WHERE', s).group(1) for s, _p in deletes)
      == ['OutstandingOrder', 'PredictionResult']
      and all(s.endswith('WHERE Plant = :plant') for s, _p in deletes) and len(inserts) == 2,
      str(eng.sent))

eng, deletes, inserts, _s, _m = sent_by(db_writer.write_outstanding_order, outstanding_df, 3300,
                                        '20260930.xlsx')
check("OutstandingOrder: replaces only this file's rows (a rerun), inserts, in ONE transaction",
      eng.begun == 1 and len(deletes) == 1
      and deletes[0][0].endswith('.OutstandingOrder WHERE Plant = :plant AND SourceFileName = :source_file')
      and deletes[0][1] == {'plant': '3300', 'source_file': '20260930.xlsx'} and len(inserts) == 1,
      str(eng.sent))

print('\n=== 8. the SQL files agree with what Python writes ===')
ddl = (_REPO_ROOT / 'create_result_tables.sql').read_text(encoding='utf-8')


def table_columns(sql_text, table):
    body = sql_text.split(f'CREATE TABLE DemandForecast.{table} (', 1)[1].split('\n);', 1)[0]
    return {m.group(1) for line in body.splitlines() if (m := re.match(r'\s{4}(\w+)\s+\w', line))}


def insert_columns(sql):
    return set(re.findall(r'\w+', sql[sql.index('(') + 1:sql.index(')')]))


writes = {}
for fn, args in ((db_writer.write_prediction_result, (result_df, 3300, day)),
                 (db_writer.write_outstanding_order, (outstanding_df, 3300, '20260930.xlsx')),
                 (db_writer.month_start_write_clear, (actual_df, result_df, 3300, day))):
    _e, _d, ins, _s, mer = sent_by(fn, *args)
    for sql, rows in ins + mer:
        table = re.search(r'(?:INTO|MERGE) \S+\.(\w+)', sql).group(1)
        writes[table] = (sql, rows[0] if isinstance(rows, list) else rows)
for table in ('PredictionResult', 'PredictionResultLog', 'OutstandingOrder'):
    sql, row = writes[table]
    cols = table_columns(ddl, table)
    check(f'{table}: every column Python inserts exists in the DDL, and is bound',
          insert_columns(sql) <= cols and insert_columns(sql) <= set(row),
          f'missing from DDL: {sorted(insert_columns(sql) - cols)}; unbound: {sorted(insert_columns(sql) - set(row))}')
check('ActualOrder: every column the MERGE binds exists in the DDL',
      set(writes['ActualOrder'][1]) <= table_columns(ddl, 'ActualOrder'))
check('the Log keeps SalesModelCode (frozen as of the prediction); PredictionResult takes it from '
      'ProductLifecycle', 'SalesModelCode' in insert_columns(writes['PredictionResultLog'][0])
      and 'SalesModelCode' not in insert_columns(writes['PredictionResult'][0]))
for table in ('PredictionResult', 'PredictionResultLog'):
    check(f'{table}: DDL has no columns derivable from OriginDate',
          not {'SettledYearMonth', 'CutoffDayIndex'} & table_columns(ddl, table))

# The non-key indexes the live database holds (QA, checked 2026-10-19). The Log has none: it is
# write-only and never read back, so an index would only cost space and insert time.
LIVE_INDEXES = {'IX_PredictionResult_Plant_PlanModelCode_TargetYearMonth_InsertedAt',
                'IX_OutstandingOrder_Plant_PlanModelCode_TargetYearMonth',
                'IX_ProductSuccessor_SuccessorCode'}
fresh = set(re.findall(r'CREATE INDEX (\w+)', ddl))
check('create_result_tables.sql builds exactly the indexes the live database has',
      fresh == LIVE_INDEXES, f'missing={sorted(LIVE_INDEXES - fresh)} extra={sorted(fresh - LIVE_INDEXES)}')
check('no index is defined on PredictionResultLog', 'ON DemandForecast.PredictionResultLog' not in ddl)
check('create_result_tables.sql warns that it is destructive, since it DROPs the history table',
      'DESTRUCTIVE' in ddl)

print('\n=== 9. the real phases, on the last day of month N+1 ===')
# Sep 30 is the one day the old OriginDate inference got N+1 wrong.
sep30 = pd.Timestamp('2026-09-30')
saved = {k: getattr(orchestrator, k) for k in
         ('db_writer', '_load_orders_for_file', 'preprocess_for_inference')}


class NoOriginDB:
    def __init__(self):
        self.written = []

    def write_outstanding_order(self, df, plant, source_file):
        self.written.append((df, source_file))


db9 = NoOriginDB()
orchestrator.db_writer = db9
orchestrator._load_orders_for_file = lambda plant, source_file: orders
try:
    orchestrator._outstanding_order_phase(3300, sep30, '20260930.xlsx', settled_month_idx=aug)
    check('_outstanding_order_phase: N+1 from the settled month is Sep, not Oct, written under its file',
          len(db9.written) == 1 and db9.written[0][0]['TargetYearMonth'].iloc[0] == '202609'
          and db9.written[0][1] == '20260930.xlsx', str(db9.written))

    asked = {}

    class Stop(Exception):
        pass

    def capture_origin(plant, origin_yyyymm=None, *a, **k):
        asked.update(origin=origin_yyyymm, cutoff_date=k.get('cutoff_date'))
        raise Stop
    orchestrator.preprocess_for_inference = capture_origin
    for grain, cutoff in (('day', sep30 - pd.Timedelta(days=1)), ('month', None)):
        asked.clear()
        try:
            orchestrator._prediction_phase(3300, 'production', range(1, 7), 3, ('SL',), 6,
                                           cutoff_grain=grain, file_date=sep30)
        except Stop:
            pass
        check(f'_prediction_phase ({grain}) pins the origin to the settled month of the SAME file'
              + (', cut off the day before the file date' if cutoff is not None else ', as of month end'),
              asked == {'origin': 202608, 'cutoff_date': cutoff}, f'{asked}')
finally:
    for k, v in saved.items():
        setattr(orchestrator, k, v)

print('\n' + '=' * 78)
print(f'{len(FAILURES)} FAILURE(S): {FAILURES}' if FAILURES else 'ALL CHECKS PASSED')
sys.exit(1 if FAILURES else 0)
