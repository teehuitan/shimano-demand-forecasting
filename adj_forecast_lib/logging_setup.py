"""Configures console, per-run file and (when reachable) AIAPPLOGS logging for one pipeline run."""
import glob
import logging
import os
import re
import uuid
from datetime import date, datetime, timezone

import pandas as pd

from .config import plant_name
from .db_logging import build_db_log_handler, delete_old_db_logs

LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'
_LOG_FILE_STAMP = re.compile(r'_(\d{8}_\d{6})\.log$')   # configure_logging's <purpose>_YYYYMMDD_HHMMSS.log

log = logging.getLogger(__name__)


def configure_logging(purpose: str, plant: int, log_dir: str = 'logs') -> str:
    """Returns the log file path; each call gets its own file and a fresh CorrelationId."""
    plant_log_dir = os.path.join(log_dir, plant_name(plant))
    os.makedirs(plant_log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(plant_log_dir, f"{purpose}_{timestamp}.log")

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file_path, encoding='utf-8'),
    ]
    # None when DB logging is switched off or the handler couldn't be built - console+file logging
    # is never held hostage to the log DB being reachable.
    db_handler = build_db_log_handler(correlation_id=uuid.uuid4(), context={
        'purpose': purpose,
        'plant': plant,
        'plant_name': plant_name(plant),
        'log_file': log_file_path,
    })
    if db_handler is not None:
        handlers.append(db_handler)

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        force=True,
        handlers=handlers,
    )
    return log_file_path


def retention_months() -> int | None:
    """LOG_RETENTION_MONTHS as a whole number >= 1; None (keep everything) when empty or invalid."""
    raw = (os.environ.get('LOG_RETENTION_MONTHS') or '').strip()
    if not raw:
        return None
    try:
        months = int(raw)
    except ValueError:
        months = 0
    if months < 1:
        log.warning("LOG_RETENTION_MONTHS=%r is not a whole number >= 1 - old logs are kept.", raw)
        return None
    return months


def prune_old_logs(plant: int, log_dir: str = 'logs', today: date | None = None) -> None:
    """Deletes this plant's log files, and this pipeline's AIAPPLOGS rows (all plants), from before
    today minus LOG_RETENTION_MONTHS. Does nothing when that is empty; never raises."""
    months = retention_months()
    if months is None:
        return
    # DateOffset clamps to month end: 31 Dec - 2 months = 31 Oct, 30 Apr - 2 months = 28/29 Feb.
    cutoff = (pd.Timestamp(today or date.today()) - pd.DateOffset(months=months)).to_pydatetime()
    deleted = 0
    for path in glob.glob(os.path.join(log_dir, plant_name(plant), '*.log')):
        stamp = _LOG_FILE_STAMP.search(os.path.basename(path))
        try:
            if not stamp or datetime.strptime(stamp.group(1), '%Y%m%d_%H%M%S') >= cutoff:
                continue
            os.remove(path)
            deleted += 1
        except (ValueError, OSError) as exc:   # an impossible date in the name, or a locked file
            log.warning("Old log %s not deleted: %s", path, exc)
    # File names are local time; TimestampUtc is UTC.
    rows = delete_old_db_logs(cutoff.astimezone(timezone.utc).replace(tzinfo=None))
    db = 'AIAPPLOGS not pruned (DB logging off or delete failed)' if rows is None else f'{rows} AIAPPLOGS row(s)'
    log.info("Log retention %d months: deleted %d log file(s) of plant %s and %s, from before %s.",
             months, deleted, plant_name(plant), db, cutoff.date())
