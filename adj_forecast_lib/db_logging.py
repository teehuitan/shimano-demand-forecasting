"""Mirrors log records into [AI].AI.AIAPPLOGS via a background queue; DB failures never break the
pipeline.
"""
import atexit
import getpass
import json
import logging
import os
import queue
import socket
import sys
import threading
import urllib
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()  # same pattern as db_writer.py/config.py - lets this module stand alone regardless
# of import order.

LOG_DATABASE = 'AI'
LOG_SCHEMA = 'AI'
LOG_TABLE = 'AIAPPLOGS'

APPLICATION_NAME = 'PP-Dashboard'
AGENT_ID = 'Demand-forecast-ML'

BATCH_SIZE = 100        # rows per INSERT round-trip
FLUSH_INTERVAL = 5.0    # seconds the worker waits for a batch to fill before writing it anyway
QUEUE_SIZE = 20000      # records held in memory before new ones are dropped
CLOSE_TIMEOUT = 10.0    # seconds close() waits for the worker to drain before giving up
LOGIN_TIMEOUT = 5       # seconds to wait for a connection before calling the server unreachable

# Never mirrored: SQLAlchemy/pyodbc chatter is generated *by* our own INSERT, so echoing it back
# would feed the queue from the queue. The thread-local guard below is the real protection (it
# catches anything they emit mid-write); this also keeps their non-write noise out of the table.
_EXCLUDED_LOGGER_PREFIXES = ('sqlalchemy', 'pyodbc', __name__)

# Everything logging puts on a LogRecord itself - anything else was passed by a caller via
# `extra=` and is worth keeping, so it gets folded into Properties.
_STANDARD_ATTRS = frozenset((
    'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename', 'funcName', 'levelname',
    'levelno', 'lineno', 'message', 'module', 'msecs', 'msg', 'name', 'pathname', 'process',
    'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName', 'taskName',
))

_INSERT_SQL = f"""
    INSERT INTO [{LOG_DATABASE}].{LOG_SCHEMA}.{LOG_TABLE}
        (TimestampUtc, LogLevel, Message, Exception, ApplicationName, agent_id, Environment,
         LoggerName, MachineName, ProcessId, ThreadId, UserId, CorrelationId, TraceId, SpanId,
         Properties)
    VALUES
        (:TimestampUtc, :LogLevel, :Message, :Exception, :ApplicationName, :agent_id, :Environment,
         :LoggerName, :MachineName, :ProcessId, :ThreadId, :UserId, :CorrelationId, :TraceId,
         :SpanId, :Properties)
"""

_engine = None
_engine_lock = threading.Lock()
# Queued by close() purely to wake the worker out of its blocking get() - without it every shutdown
# would stall for up to FLUSH_INTERVAL seconds waiting on a timeout that has nothing left to fetch.
_SENTINEL = object()
# Set for the duration of a DB write, on the worker thread only. Any record SQLAlchemy/pyodbc emit
# while we are writing sees this and is dropped, so a write can never enqueue more writes.
_writing = threading.local()


def _get_engine():
    """Lazily-built engine, separate from db_writer's pool."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                odbc_str = (
                    f"DRIVER={{{os.environ['ODBC_DRIVER']}}};"
                    f"SERVER={os.environ['QA_DB_SERVER']};"
                    f"UID={os.environ['QA_DB_UID']};"
                    f"PWD={os.environ['QA_DB_PWD']};"
                )
                _engine = create_engine(
                    "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(odbc_str),
                    pool_pre_ping=True, pool_size=1, max_overflow=1, pool_recycle=1800,
                    # Login timeout, not query timeout. Without it an unreachable server hangs the
                    # worker for the driver's own (much longer) default on every single batch.
                    connect_args={'timeout': LOGIN_TIMEOUT},
                )
    return _engine


def _truncate(value, limit):
    """An over-long value turns a VARCHAR column into a failed INSERT (and so a lost batch), so
    every bounded column goes through this."""
    if value is None:
        return None
    value = str(value)
    return value if len(value) <= limit else value[:limit]


class SqlServerLogHandler(logging.Handler):
    """Queues each record and writes it in batches from a daemon thread."""

    def __init__(self, environment, correlation_id=None, context=None, level=logging.NOTSET):
        super().__init__(level=level)
        self.environment = _truncate(environment, 50)
        self.correlation_id = str(correlation_id or uuid.uuid4())
        self.context = dict(context or {})
        self.machine_name = _truncate(socket.gethostname(), 100)
        self.process_id = os.getpid()
        try:
            self.user_id = _truncate(getpass.getuser(), 100)
        except Exception:
            self.user_id = None  # no resolvable account name (some service contexts) - not fatal.

        self._queue = queue.Queue(maxsize=QUEUE_SIZE)
        self._dropped = 0
        self._failures = 0
        self._stopping = threading.Event()
        self._worker = threading.Thread(target=self._run, name='sqlserver-log-writer', daemon=True)
        self._worker.start()
        atexit.register(self.close)

    # ---- producer side (runs on whatever thread logged) ----

    def emit(self, record):
        """Renders the record on the calling thread and queues it; drops rather than raises."""
        try:
            if getattr(_writing, 'active', False):
                return
            if record.name.startswith(_EXCLUDED_LOGGER_PREFIXES):
                return
            row = self._build_row(record)
            try:
                self._queue.put_nowait(row)
            except queue.Full:
                self._dropped += 1
        except Exception:
            self.handleError(record)

    def _build_row(self, record):
        try:
            message = record.getMessage()
        except Exception as exc:  # bad %-format args shouldn't cost us the record entirely
            message = f"<unformattable log message {record.msg!r}: {exc}>"

        exception_text = None
        if record.exc_info:
            exception_text = record.exc_text or logging.Formatter().formatException(record.exc_info)
        if record.stack_info:
            stack = logging.Formatter().formatStack(record.stack_info)
            exception_text = f"{exception_text}\n{stack}" if exception_text else stack

        properties = dict(self.context)
        properties.update({
            'module': record.module,
            'funcName': record.funcName,
            'lineno': record.lineno,
            'pathname': record.pathname,
            'threadName': record.threadName,
        })
        properties.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS})

        created_utc = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return {
            'TimestampUtc': created_utc.replace(tzinfo=None),  # DATETIME2 is naive - UTC by column name
            'LogLevel': _truncate(record.levelname, 15),
            'Message': message,
            'Exception': exception_text,
            'ApplicationName': APPLICATION_NAME,
            'agent_id': AGENT_ID,
            'Environment': self.environment,
            'LoggerName': _truncate(record.name, 255),
            'MachineName': self.machine_name,
            'ProcessId': self.process_id,
            'ThreadId': record.thread,
            'UserId': self.user_id,
            'CorrelationId': self.correlation_id,
            'TraceId': None,   # no distributed tracing in this pipeline - CorrelationId (one per
            'SpanId': None,    # configure_logging call) is what groups a single run's rows.
            'Properties': json.dumps(properties, default=str),
        }

    # ---- consumer side (runs on the worker thread) ----

    def _run(self):
        """Writes queued records in batches; exits only once stopped and the queue is empty."""
        while True:
            batch, stop = [], False
            try:
                item = self._queue.get(timeout=FLUSH_INTERVAL)
            except queue.Empty:
                # Only way out if close() couldn't queue its sentinel (a full queue).
                if self._stopping.is_set():
                    return
                continue
            stop = item is _SENTINEL
            if not stop:
                batch.append(item)
            while not stop and len(batch) < BATCH_SIZE:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is _SENTINEL:
                    stop = True      # still writes `batch` below - the sentinel is always last in.
                else:
                    batch.append(item)
            if batch:
                self._write(batch)
            if stop:
                return

    def _write(self, rows):
        _writing.active = True
        try:
            with _get_engine().begin() as conn:
                conn.execute(text(_INSERT_SQL), rows)
            self._failures = 0
        except Exception as exc:
            self._failures += 1
            # Only the first failure of a streak is reported: an outage would otherwise print one
            # line per batch for the rest of the run, burying the real console log.
            if self._failures == 1:
                print(f"[db_logging] dropping {len(rows)} log row(s) - DB write failed: {exc}",
                    file=sys.stderr)
            # Back off rather than hammer a sick box - but wait() (not sleep) so a close() landing
            # mid-backoff isn't stuck behind the full delay.
            self._stopping.wait(min(2 ** min(self._failures, 5), 30))
        finally:
            _writing.active = False

    # ---- shutdown ----

    def close(self):
        """Idempotent (atexit and logging.shutdown both call it). Waits up to CLOSE_TIMEOUT for the
        worker to drain, so the last records of a run - usually the ones that matter - still land."""
        if self._stopping.is_set():
            return
        self._stopping.set()
        try:
            self._queue.put_nowait(_SENTINEL)
        except queue.Full:
            pass  # worker falls back to noticing _stopping on its next idle timeout.
        self._worker.join(timeout=CLOSE_TIMEOUT)
        if self._dropped:
            print(f"[db_logging] {self._dropped} log record(s) dropped - queue full.", file=sys.stderr)
        try:
            atexit.unregister(self.close)
        except Exception:
            pass
        super().close()


# `or`, not a .get default: deploy.ps1 writes every key unconditionally, so an undefined Bamboo
# variable reaches us as an empty string rather than as a missing key. Empty means "unset" here
# (mirror on, Environment=QA) - an empty Environment would otherwise be written to the table.
def _db_logging_off() -> bool:
    return (os.environ.get('LOG_DB_ENABLED') or '1').strip().lower() in ('0', 'false', 'no')


def _environment(environment=None) -> str:
    return environment or os.environ.get('LOG_DB_ENVIRONMENT') or 'QA'


def build_db_log_handler(environment=None, correlation_id=None, context=None):
    """Returns a configured handler, or None when DB logging is off (LOG_DB_ENABLED=0) or unconfigurable."""
    if _db_logging_off():
        return None
    try:
        return SqlServerLogHandler(_environment(environment), correlation_id=correlation_id, context=context)
    except Exception as exc:
        print(f"[db_logging] DB logging disabled - handler could not be built: {exc}", file=sys.stderr)
        return None


# Only this pipeline's own rows: the table is shared with other applications (USEC, MA-Monolith).
_DELETE_OLD_SQL = f"""
    DELETE FROM [{LOG_DATABASE}].{LOG_SCHEMA}.{LOG_TABLE}
    WHERE ApplicationName = :app AND agent_id = :agent AND Environment = :env AND TimestampUtc < :cutoff
"""


def delete_old_db_logs(cutoff_utc, environment=None):
    """Deletes this pipeline's AIAPPLOGS rows (same application, agent and environment as the handler
    writes) before the naive-UTC `cutoff_utc`. Returns the row count, or None when DB logging is off
    or the delete failed - never raises."""
    if _db_logging_off():
        return None
    try:
        with _get_engine().begin() as conn:
            return conn.execute(text(_DELETE_OLD_SQL), {
                'app': APPLICATION_NAME, 'agent': AGENT_ID, 'env': _environment(environment),
                'cutoff': cutoff_utc}).rowcount
    except Exception as exc:
        logging.getLogger(__name__).warning("Old AIAPPLOGS rows not deleted - DB delete failed: %s", exc)
        return None
