"""A synthetic stand-in for the Fabric order feed, for suites that must run the REAL context,
panel and skeleton builders rather than a hand-assembled ctx.

Only the three functions that touch a database are replaced - preprocessing.load_io,
preprocessing.load_tfg_material and preprocessing.latest_source_file_date. Everything downstream
of them (build_order_book_context and every book, feature, panel and skeleton builder) is the
production code. Rows are pushed through the real _flag_known_cancellations, so cancellation
validity is decided exactly as load_io decides it.

Plant-agnostic by construction: the stubs ignore the plant code, so a suite can run the same
scenario for every plant in config.PLANT_NAMES.

One SKU, 'KNOWN1', carries a hand-built cancellation scenario with a known answer (see
KNOWN_SCENARIO); the rest is random background so every feature family has history to read.
"""
import re

import numpy as np
import pandas as pd

import no_db  # noqa: F401 - every real DB engine raises before any stub here is even installed
from adj_forecast_lib import preprocessing
from adj_forecast_lib.config import date_to_month_idx

FILE_DATE = pd.Timestamp('2026-09-21')      # the latest source file's date
DATA_THROUGH = FILE_DATE - pd.Timedelta(days=1)   # a file dated D is complete through D-1 only
SETTLED_YYYYMM = 202608                     # the month before the file's month

# Target Oct 2026 for KNOWN1. As of 2026-09-15 (day 15 after origin Aug 2026):
#   entered <= Sep 15: 100 + 50 + 80 + 40 = 270 gross
#   cancelled <= Sep 15: o1 (Aug 15) 100 + o3 (Sep 12) 80 = 180   -> net 90
#   o4 is cancelled Sep 18 - AFTER Sep 15 - so it must not count as of Sep 15.
# As of 2026-08-31 (month grain / day 0): gross 150 (o1+o2), cancelled 100 (o1) -> net 50.
KNOWN_SCENARIO = [
    # entry,        qty, cancel date or None
    ('2026-07-10', 100.0, '2026-08-15'),
    ('2026-08-05',  50.0, None),
    ('2026-09-05',  80.0, '2026-09-12'),
    ('2026-09-08',  40.0, '2026-09-18'),
]
KNOWN_SMC, KNOWN_TARGET_YYYYMM = 'KNOWN1', 202610
# PlanModelCode is the forecast grain. Never equal to the SalesModelCode, and C1/C2 share PNC, so code
# reading one SalesModelCode instead of summing per PlanModelCode fails here instead of passing.
PMC_BY_SMC = {'A1': 'PNA1', 'A2': 'PNA2', 'B1': 'PNB1', 'B2': 'PNB2', 'C1': 'PNC', 'C2': 'PNC',
              KNOWN_SMC: 'PNKNOWN1'}
KNOWN_CODE = PMC_BY_SMC[KNOWN_SMC]


def make_orders(seed: int = 7, n: int = 9000, cancel_rate: float = 0.08,
                part_map: dict = None) -> pd.DataFrame:
    """Rows shaped exactly like load_io's output."""
    rng = np.random.default_rng(seed)
    codes = np.array(['A1', 'A2', 'B1', 'B2', 'C1', 'C2'])
    entry = pd.Timestamp('2022-01-03') + pd.to_timedelta(
        rng.integers(0, (DATA_THROUGH - pd.Timestamp('2022-01-03')).days + 1, n), unit='D')
    lead_days = rng.integers(0, 200, n)
    request = entry + pd.to_timedelta(lead_days, unit='D')
    code = rng.choice(codes, n)
    is_cancel = rng.random(n) < cancel_rate
    cancel_after = pd.to_timedelta(rng.integers(0, 60, n), unit='D')
    cancel = pd.Series(pd.NaT, index=range(n), dtype='datetime64[ns]')
    cancel[is_cancel] = (entry + cancel_after)[is_cancel]
    # a snapshot can't contain a cancellation dated after the file it came from
    cancel[cancel > DATA_THROUGH] = pd.NaT
    status = np.where(cancel.notna(), 'Canceled', 'Open')

    raw = pd.DataFrame({'SalesModelCode': code, 'Qty': rng.integers(1, 300, n).astype('float32'),
                        'entry': entry, 'request': request, 'cancel': cancel.to_numpy(),
                        'status': status})
    target_mid = pd.Timestamp(f'{KNOWN_TARGET_YYYYMM // 100}-{KNOWN_TARGET_YYYYMM % 100:02d}-15')
    known = [(KNOWN_SMC, q, pd.Timestamp(e), target_mid, pd.Timestamp(c) if c else pd.NaT,
              'Canceled' if c else 'Open') for e, q, c in KNOWN_SCENARIO]
    # background history for KNOWN1 so it passes the lifespan / months-since-launch filters
    known += [(KNOWN_SMC, 20.0 + i % 7, m + pd.Timedelta(days=9), m + pd.Timedelta(days=40),
               pd.NaT, 'Open')
              for i, m in enumerate(pd.date_range('2023-01-01', '2026-06-01', freq='MS'))]
    known = pd.DataFrame(known, columns=raw.columns).astype({'Qty': 'float32',
                                                            'cancel': 'datetime64[ns]'})
    raw = pd.concat([raw, known], ignore_index=True)
    raw = raw[raw['request'] >= raw['entry']].reset_index(drop=True)
    raw = raw[raw['entry'] <= DATA_THROUGH].reset_index(drop=True)

    io = pd.DataFrame({
        'ProductCode': raw['SalesModelCode'] + 'P',
        'SalesModelCode': raw['SalesModelCode'],
        'PlanModelCode': raw['SalesModelCode'].map(PMC_BY_SMC),
        # A*/B* are the default quantile PartCodes (SL/RD) so both scoring halves have rows;
        # pass part_map to give a plant a different mix (e.g. none of them at all).
        'PartCode': raw['SalesModelCode'].str[0].map(part_map or {'A': 'SL', 'B': 'RD'})
                        .fillna(raw['SalesModelCode'].str[0]),
        'Qty': raw['Qty'].astype('float32'),
        'Currency': 'USD',
        'CustomerCountry': rng.choice(['CN', 'DE', 'US'], len(raw), p=[.5, .3, .2]),
        'UnitPrice': np.float32(10.0),
        'ClassText': 'X',
        'Order_Status_Text': raw['status'],
    })
    entry_d, request_d = pd.to_datetime(raw['entry']), pd.to_datetime(raw['request'])
    cancel_d = pd.to_datetime(raw['cancel'])
    io['EntryMonthIdx'] = date_to_month_idx(entry_d).astype('int32')
    io['RequestMonthIdx'] = date_to_month_idx(request_d).astype('int32')
    io = preprocessing._flag_known_cancellations(io, entry_d, request_d, cancel_d)
    io['EntryDate'], io['RequestDate'], io['CancelDate'] = entry_d, request_d, cancel_d
    return io.drop(columns=['Order_Status_Text'])


def make_tfg(orders: pd.DataFrame) -> pd.DataFrame:
    first = orders.groupby('ProductCode')['EntryDate'].min()
    smc = [c[:-1] for c in first.index]
    return pd.DataFrame({'ProductCode': first.index,
                         'SalesModelCode': smc,
                         # the product's own PlanModelCode, so a suite's extra codes need no table entry
                         'SubModelCode': orders.groupby('ProductCode')['PlanModelCode'].first()
                                               .reindex(first.index).to_numpy(),
                         'ProductAbbreviation': [f'{c} name' for c in smc],
                         'ProductStartDate': (first - pd.Timedelta(days=5)).to_numpy(),
                         'ProductEndDate': pd.NaT})


class FabricStub:
    """Installs the synthetic feed in place of preprocessing's three DB calls, and counts calls -
    the panel-build counter the retrain tests assert on. Use as a context manager.

    Replaces every module-level COPY of those functions too (`from .preprocessing import
    latest_source_file_date` in main.py, and any added later), not just preprocessing's own names:
    a copy left pointing at the real function is exactly how a suite once reached Fabric."""

    _NAMES = ('load_io', 'load_tfg_material', 'latest_source_file_date', 'load_sales_result')

    def __init__(self, orders: pd.DataFrame = None, file_date: pd.Timestamp = FILE_DATE):
        self.orders = make_orders() if orders is None else orders
        self.file_date = file_date
        self.calls = {'load_io': 0, 'load_tfg_material': 0}

    def _load_io(self, plant, exclude_partcodes=None):
        self.calls['load_io'] += 1
        return self.orders[self.orders['EntryDate'] <= self.file_date].copy()

    def _load_tfg(self):
        self.calls['load_tfg_material'] += 1
        return make_tfg(self.orders)

    def _latest(self, plant):
        return self.file_date, self.file_date.strftime('%Y%m%d') + '.xlsx'

    @staticmethod
    def _sales_result(plant, product_pmc_map):
        return pd.DataFrame({'PlanModelCode': pd.Series(dtype=object),
                             'TargetIdx': pd.Series(dtype='int64'),
                             'SalesResult': pd.Series(dtype='float64')})

    def _successor_pairs(self) -> pd.DataFrame:
        """The ProductSuccessor table (SalesModelCode pairs) - read when a plant's
        PLANT_INCLUDES_PRED_SUCC is on, whatever .env says. Empty unless a suite overrides it."""
        return pd.DataFrame({'SalesModelCode': pd.Series(dtype=object),
                             'SuccessorCode': pd.Series(dtype=object)})

    def __enter__(self):
        import sys
        from adj_forecast_lib import lifecycle_cleaning
        stubs = {'load_io': self._load_io, 'load_tfg_material': self._load_tfg,
                 'latest_source_file_date': self._latest, 'load_sales_result': self._sales_result}
        real = {n: getattr(preprocessing, n) for n in self._NAMES}
        self._saved = []
        for module in list(sys.modules.values()):
            for name, fn in real.items():
                if module is not None and getattr(module, name, None) is fn:
                    self._saved.append((module, name, fn))
                    setattr(module, name, stubs[name])
        self._saved.append((lifecycle_cleaning, 'pd', lifecycle_cleaning.pd))
        lifecycle_cleaning.pd = _PandasWithSuccessorTable(self._successor_pairs)
        return self

    def __exit__(self, *exc):
        for module, name, fn in reversed(self._saved):
            setattr(module, name, fn)
        return False


class _PandasWithSuccessorTable:
    """lifecycle_cleaning's `pd`, answering its ProductSuccessor query; anything else is pandas."""

    def __init__(self, pairs):
        self._pairs = pairs

    def read_sql(self, sql, con, **kwargs):
        if 'ProductSuccessor' not in str(sql):
            raise AssertionError(f'unexpected query in lifecycle_cleaning: {str(sql)[:80]}')
        return self._pairs().copy()

    def __getattr__(self, name):
        return getattr(pd, name)


class RecordingEngine:
    """Stands in for db_writer._engine: records every statement and its bound rows, returns nothing
    (a scalar() of None, rowcount 0). Lets a test read exactly what would have been INSERTed."""

    def __init__(self):
        self.sent = []

    def begin(self):
        import contextlib
        sent = self.sent

        class _Result:
            rowcount = 0

            def scalar(self):
                return None

            def fetchall(self):
                return []           # nothing is ever already logged

            def first(self):
                return None         # nothing is ever already written

        class _Conn:
            def execute(self, stmt, rows=None):
                sent.append((str(stmt), rows))
                return _Result()

        @contextlib.contextmanager
        def cm():
            yield _Conn()
        return cm()

    def rows_for(self, table: str) -> pd.DataFrame:
        """The rows INSERTed into `table` (exact name, so 'PredictionResult' excludes its Log)."""
        import re
        out = []
        for sql, rows in self.sent:
            m = re.search(r'INSERT INTO \S+\.(\w+)', sql)
            if m and m.group(1) == table and rows:
                out += rows if isinstance(rows, list) else [rows]
        return pd.DataFrame(out)


class InjectedFailure(RuntimeError):
    """Raised by TableEngine.fail_on - a dropped connection, a constraint violation."""


class TableEngine:
    """Stands in for db_writer._engine and EXECUTES what db_writer sends, against in-memory tables
    with real transaction semantics: a begin() block commits only when it exits cleanly, and an
    exception anywhere inside it rolls back every statement it ran. So a test can assert what the
    tables HOLD after a run, a rerun or a failed write - not only what was sent.

    It understands exactly db_writer's statement shapes: DELETE ... WHERE Plant = :plant [AND <col>
    = :p] (any other scope raises - an unscoped one would wipe every plant), INSERT INTO with one
    row or a list, MERGE as an upsert on (Plant, PlanModelCode[, TargetYearMonth]), and SELECT
    <columns> | TOP 1 1 FROM <table> WHERE <col> = :p AND <col> IN :p ... [ORDER BY Id], answered
    in insertion order from what the transaction can see. Anything else raises.
    `fail_on`, a substring, makes every statement containing it raise InjectedFailure before it
    changes anything. `begun` counts begin() calls.
    """

    _STMT = re.compile(r'^\s*(DELETE FROM|INSERT INTO|MERGE)\s+\S+\.(\w+)')
    # ORDER BY is accepted only as Id, the IDENTITY column, which insertion order already is.
    _SELECT = re.compile(r'^\s*SELECT\s+(?P<cols>.+?)\s+FROM\s+\S+\.(?P<table>\w+)\s+WHERE\s+'
                         r'(?P<where>.+?)(?:\s+ORDER BY Id)?\s*$', re.S)
    _COND = re.compile(r'^(\w+)\s*(=|IN)\s*\(?\s*(?::(\w+)|_*\[POSTCOMPILE_(\w+)\])\s*\)?$')

    def __init__(self):
        self.tables = {}
        self.fail_on = None
        self.begun = 0

    def rows(self, table: str, plant=None) -> pd.DataFrame:
        """What `table` holds now, optionally for one plant only."""
        return pd.DataFrame([r for r in self.tables.get(table, [])
                             if plant is None or r['Plant'] == str(plant)])

    def begin(self):
        import contextlib
        import copy
        engine = self

        class _Result:
            def __init__(self, n, rows=()):
                self.rowcount = n
                self._rows = list(rows)

            def scalar(self):
                return None

            def fetchall(self):
                return self._rows

            def first(self):
                return self._rows[0] if self._rows else None

        class _Conn:
            def __init__(self, work):
                self.work = work

            def execute(self, stmt, params=None):
                sql = str(stmt)
                if engine.fail_on and engine.fail_on in sql:
                    raise InjectedFailure(f'injected failure on: {" ".join(sql.split())[:80]}')
                s = TableEngine._SELECT.match(sql)
                if s:
                    conds = []
                    for cond in re.split(r'\s+AND\s+', s['where']):
                        c = TableEngine._COND.match(cond.strip())
                        if not c:
                            raise AssertionError(f'TableEngine does not understand: {cond!r}')
                        conds.append((c.group(1), c.group(2), params[c.group(3) or c.group(4)]))
                    hits = [r for r in self.work.get(s['table'], [])
                            if all(r[col] in v if op == 'IN' else r[col] == v
                                   for col, op, v in conds)]
                    if s['cols'].strip() == 'TOP 1 1':      # db_writer's has_* existence checks
                        return _Result(len(hits[:1]), [(1,)] * len(hits[:1]))
                    cols = [col.strip() for col in s['cols'].split(',')]
                    return _Result(len(hits), [tuple(r[col] for col in cols) for r in hits])
                m = TableEngine._STMT.match(sql)
                if not m:
                    raise AssertionError(f'TableEngine does not understand: {sql.strip()[:80]}')
                verb, table = m.groups()
                held = self.work.setdefault(table, [])
                if verb == 'DELETE FROM':
                    d = re.search(r'WHERE\s+Plant\s*=\s*:plant(?:\s+AND\s+(\w+)\s*=\s*:(\w+))?\s*$',
                                  sql.strip())
                    if not d:
                        raise AssertionError(f'DELETE not scoped to one plant: {sql.strip()}')
                    col, p = d.groups()
                    self.work[table] = [r for r in held if not (r['Plant'] == params['plant']
                                        and (col is None or r[col] == params[p]))]
                    return _Result(len(held) - len(self.work[table]))
                batch = [dict(r) for r in (params if isinstance(params, list) else [params])]
                if verb == 'INSERT INTO':
                    held.extend(batch)
                else:
                    # ActualOrder's key; ProductLifecycle has no TargetYearMonth, so (Plant, PlanModelCode).
                    key = lambda r: (r['Plant'], r['PlanModelCode'], r.get('TargetYearMonth'))
                    new = {key(r): r for r in batch}
                    self.work[table] = [r for r in held if key(r) not in new] + list(new.values())
                return _Result(len(batch))

        @contextlib.contextmanager
        def cm():
            engine.begun += 1
            work = copy.deepcopy(engine.tables)
            yield _Conn(work)
            engine.tables = work        # reached only when the block exited cleanly: commit
        return cm()


def ground_truth(orders: pd.DataFrame, panel: pd.DataFrame, asof_col: str = None) -> pd.DataFrame:
    """Gross/cancelled/net known for EVERY row of a skeleton, straight from order rows: the as-of
    point is CutoffDate at day grain, else the origin month's last day."""
    from adj_forecast_lib.cutoff_grain import cutoff_dates_for_origins
    asof = (pd.to_datetime(panel[asof_col]) if asof_col is not None
            else pd.to_datetime(cutoff_dates_for_origins(panel['OriginIdx'].to_numpy(), 0)))
    keys = pd.DataFrame({'PlanModelCode': panel['PlanModelCode'].astype(str).to_numpy(),
                         'RequestMonthIdx': panel['TargetIdx'].to_numpy(),
                         'asof': asof.to_numpy(), '_row': np.arange(len(panel))})
    o = orders[['PlanModelCode', 'RequestMonthIdx', 'EntryDate', 'CancelKnownDate',
                'IsCanceled', 'Qty']]
    m = keys.merge(o, on=['PlanModelCode', 'RequestMonthIdx'], how='left')
    m['gross'] = np.where(m['EntryDate'] <= m['asof'], m['Qty'], 0.0)
    m['cancelled'] = np.where(m['IsCanceled'].fillna(False).astype(bool)
                              & (m['CancelKnownDate'] <= m['asof']), m['Qty'], 0.0)
    t = m.groupby('_row')[['gross', 'cancelled']].sum().reindex(range(len(panel)), fill_value=0.0)
    t['net'] = t['gross'] - t['cancelled']
    return t.reset_index(drop=True)


def truth(orders: pd.DataFrame, code: str, target_yyyymm: int, asof: pd.Timestamp) -> dict:
    """Ground truth straight from the order rows - no order book involved - for cross-checking."""
    t = orders[(orders['PlanModelCode'] == code)
               & (orders['RequestMonthIdx'] == int(date_to_month_idx(pd.Series([pd.Timestamp(
                   f'{target_yyyymm // 100}-{target_yyyymm % 100:02d}-01')]))[0]))]
    gross = float(t.loc[t['EntryDate'] <= asof, 'Qty'].sum())
    canc = float(t.loc[t['IsCanceled'] & (t['CancelKnownDate'] <= asof), 'Qty'].sum())
    return {'gross': gross, 'cancelled': canc, 'net': gross - canc}
