"""Log retention: the daily run prunes log files and this pipeline's AIAPPLOGS rows older than
today minus LOG_RETENTION_MONTHS.

Checked on a temp log folder: the cutoff (calendar months, clamped to month end, across a year), which
files go (only <purpose>_YYYYMMDD_HHMMSS.log before the cutoff, only this plant's folder), that empty or
invalid settings prune nothing, and that a locked file or a failing DB never raises. The DB delete runs
against a fake engine: only this pipeline's rows (application, agent, environment), cutoff in UTC.
"""
import inspect
import os
import pathlib
import sys
import tempfile
from datetime import date, datetime, timezone

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'tests'))
import no_db  # noqa: E402,F401 - every DB engine raises from here on (see no_db.py)

import main  # noqa: E402
from adj_forecast_lib import db_logging, logging_setup  # noqa: E402
from adj_forecast_lib.config import plant_name  # noqa: E402

FAILURES = []
PLANT, OTHER = 3300, 2100


def check(label, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' - ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(label)


def log_dir_with(names, plant=PLANT) -> str:
    """A temp logs/ folder holding empty files `names` in the plant's subfolder."""
    root = tempfile.mkdtemp(prefix='log_retention_')
    folder = os.path.join(root, plant_name(plant))
    os.makedirs(folder)
    for n in names:
        open(os.path.join(folder, n), 'w').close()
    return root


def left(root, plant=PLANT) -> set:
    return set(os.listdir(os.path.join(root, plant_name(plant))))


def prune(root, today, months='2'):
    os.environ['LOG_RETENTION_MONTHS'] = months
    logging_setup.prune_old_logs(PLANT, log_dir=root, today=today)


print("\n[1] which files go (today 2026-10-15, 2 months -> cutoff 2026-08-15 00:00)")
names = ['daily_20260814_235959.log',            # before the cutoff -> deleted
         'daily_backtest_20260701_090000.log',   # any purpose -> deleted
         'daily_20260815_000000.log',            # exactly at the cutoff -> kept
         'daily_20261015_080000.log',            # today -> kept
         'daily.log',                            # no timestamp -> kept
         'daily_20261399_000000.log',            # impossible date -> kept (warned)
         'notes_20200101_000000.txt']            # not a .log -> kept
root = log_dir_with(names)
other = os.path.join(root, plant_name(OTHER))
os.makedirs(other)
open(os.path.join(other, 'daily_20200101_000000.log'), 'w').close()
prune(root, date(2026, 10, 15))
check("old .log files deleted, whatever the purpose",
      not {'daily_20260814_235959.log', 'daily_backtest_20260701_090000.log'} & left(root))
check("cutoff day itself kept, newer kept",
      {'daily_20260815_000000.log', 'daily_20261015_080000.log'} <= left(root))
check("files without a valid timestamp, and non-.log files, kept",
      {'daily.log', 'daily_20261399_000000.log', 'notes_20200101_000000.txt'} <= left(root))
check("another plant's folder untouched", os.listdir(other) == ['daily_20200101_000000.log'])

print("\n[2] cutoff: calendar months, clamped to month end, across a year")
for today, gone, kept in [
        (date(2026, 12, 31), 'daily_20261030_120000.log', 'daily_20261031_000000.log'),   # -> 31 Oct
        (date(2026, 4, 30), 'daily_20260227_120000.log', 'daily_20260228_000000.log'),    # -> 28 Feb
        (date(2028, 4, 30), 'daily_20280228_120000.log', 'daily_20280229_000000.log'),    # -> 29 Feb (leap)
        (date(2027, 1, 10), 'daily_20261109_120000.log', 'daily_20261110_000000.log')]:   # -> 10 Nov 2026
    root = log_dir_with([gone, kept])
    prune(root, today)
    check(f"today {today}: {gone[6:14]} deleted, {kept[6:14]} kept", left(root) == {kept}, str(left(root)))

print("\n[3] empty or invalid setting prunes nothing")
for setting in ('', '  ', 'abc', '0', '-1', '1.5'):
    root = log_dir_with(['daily_20200101_000000.log'])
    prune(root, date(2026, 10, 15), months=setting)
    check(f"LOG_RETENTION_MONTHS={setting!r}: nothing deleted", left(root) == {'daily_20200101_000000.log'})

print("\n[4] never raises")
root = log_dir_with(['daily_20200101_000000.log', 'daily_20200102_000000.log'])
real_remove = os.remove
calls = []


def locked(path):
    calls.append(path)
    if path.endswith('daily_20200101_000000.log'):
        raise PermissionError('locked')
    real_remove(path)


os.remove = locked
try:
    prune(root, date(2026, 10, 15))
    check("a locked file is skipped, the rest still deleted",
          left(root) == {'daily_20200101_000000.log'} and len(calls) == 2)
except Exception as e:  # noqa: BLE001
    check("a locked file is skipped, the rest still deleted", False, f"{type(e).__name__}: {e}")
finally:
    os.remove = real_remove
try:
    prune(tempfile.mkdtemp(prefix='log_retention_'), date(2026, 10, 15))
    check("a missing plant folder is fine", True)
except Exception as e:  # noqa: BLE001
    check("a missing plant folder is fine", False, f"{type(e).__name__}: {e}")

print("\n[5] AIAPPLOGS delete: only this pipeline's rows, cutoff in UTC")
check("DB logging off (the test default) -> no delete", db_logging.delete_old_db_logs(datetime(2026, 8, 15)) is None)


class FakeConn:
    def __init__(self, sink):
        self.sink = sink

    def execute(self, stmt, params):
        self.sink.append((str(stmt), params))
        return type('Result', (), {'rowcount': 7})()


class FakeEngine:
    def __init__(self, fail=False):
        self.sink, self.fail = [], fail

    def begin(self):
        engine = self

        class Ctx:
            def __enter__(self):
                if engine.fail:
                    raise ConnectionError('server unreachable')
                return FakeConn(engine.sink)

            def __exit__(self, *exc):
                return False
        return Ctx()


real_engine, real_enabled = db_logging._get_engine, os.environ.get('LOG_DB_ENABLED')
os.environ['LOG_DB_ENABLED'] = '1'
os.environ['LOG_DB_ENVIRONMENT'] = 'QA'
try:
    fake = FakeEngine()
    db_logging._get_engine = lambda: fake
    n = db_logging.delete_old_db_logs(datetime(2026, 8, 15))
    sql, params = fake.sink[0]
    check("returns the deleted row count", n == 7)
    check("filters on application, agent, environment and time",
          all(f"{c} = :{p}" in sql for c, p in (('ApplicationName', 'app'), ('agent_id', 'agent'),
                                                 ('Environment', 'env'))) and 'TimestampUtc < :cutoff' in sql)
    check("values are the handler's own (bound, not formatted into the SQL)",
          params == {'app': db_logging.APPLICATION_NAME, 'agent': db_logging.AGENT_ID, 'env': 'QA',
                     'cutoff': datetime(2026, 8, 15)} and 'Demand-forecast-ML' not in sql)
    db_logging._get_engine = lambda: FakeEngine(fail=True)
    check("a failing DB returns None, never raises", db_logging.delete_old_db_logs(datetime(2026, 8, 15)) is None)

    fake = FakeEngine()
    db_logging._get_engine = lambda: fake
    prune(log_dir_with([]), date(2026, 10, 15))
    expected = datetime(2026, 8, 15).astimezone(timezone.utc).replace(tzinfo=None)
    check("prune_old_logs passes the local-midnight cutoff converted to UTC",
          len(fake.sink) == 1 and fake.sink[0][1]['cutoff'] == expected)
    fake.sink.clear()
    prune(log_dir_with([]), date(2026, 10, 15), months='')
    check("empty setting -> no DB delete either", fake.sink == [])
finally:
    db_logging._get_engine = real_engine
    os.environ['LOG_DB_ENABLED'] = real_enabled if real_enabled is not None else '0'

print("\n[6] the daily run prunes right after configuring each plant's logging")
src = inspect.getsource(main.main)
check("main() calls prune_old_logs(plant) after configure_logging('daily', plant)",
      0 < src.find("configure_logging('daily', plant)") < src.find('prune_old_logs(plant)'))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("ALL CHECKS PASSED")
