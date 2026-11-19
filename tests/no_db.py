"""Fail-closed database guard for every verification suite.

Importing this module replaces each database engine the library builds at import time with a
stand-in that raises DatabaseForbidden on first use. A suite that forgets to stub some call then
fails immediately with a clear message instead of opening a connection - which for Fabric
(ActiveDirectoryInteractive) means popping an Azure login, and for the AI database means writing.

Why this exists: a refactor of main._prediction_phase once routed scoring through a function one
suite hadn't stubbed, and that suite fell through to the real load_io. Stubs alone depend on
every suite knowing every call path; this does not.

run_all.py installs it before each suite runs, and every suite imports it too, so running a suite
directly is guarded the same way. Engines are replaced, not the functions that use them, so the
stubs suites install (load_io, db_writer._engine, ...) keep working on top of it.
"""
import os
import pathlib
import sys

# db_logging builds its engine lazily on the first log record, so it can't be swapped below -
# switching it off is the only guard against suites writing to [AI].AI.AIAPPLOGS.
os.environ['LOG_DB_ENABLED'] = '0'
# Suites that run main() would otherwise prune the real logs/ folder; verify_log_retention sets it per check.
os.environ['LOG_RETENTION_MONTHS'] = ''

# MLflow goes to a throwaway sqlite store per process, never the real tracking server.
import tempfile  # noqa: E402
_MLFLOW_DIR = tempfile.mkdtemp(prefix='mlflow_tests_').replace(os.sep, '/')
os.environ['MLFLOW_TRACKING_URI'] = f'sqlite:///{_MLFLOW_DIR}/mlflow.db'
os.environ['_MLFLOW_SERVER_ARTIFACT_ROOT'] = (pathlib.Path(_MLFLOW_DIR) / 'artifacts').resolve().as_uri()

# main.py calls load_dotenv(override=True), which would put .env's real MLFLOW_TRACKING_URI and
# LOG_DB_ENABLED back over the two above. Never override in tests: .env only fills what is unset.
import dotenv  # noqa: E402
_real_load_dotenv = dotenv.load_dotenv
dotenv.load_dotenv = lambda *a, **k: _real_load_dotenv(*a, **{**k, 'override': False})

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.pool import NullPool  # noqa: E402

from adj_forecast_lib import collate_outstanding_order, db_writer, lifecycle_cleaning, preprocessing  # noqa: E402


class DatabaseForbidden(RuntimeError):
    """A test reached a real database engine."""


def _forbidden_engine(name: str):
    """A genuine SQLAlchemy Engine - so pandas.read_sql and db_writer's engine.begin() take their
    normal code path - whose every attempt to open a connection raises DatabaseForbidden. It has
    no URL to a real server, so there is nothing it could connect to even if the check failed."""
    def refuse():
        raise DatabaseForbidden(
            f"a test tried to open a connection through {name} - the verification suites must "
            "never touch a database. Stub the call that got here (see tests/synthetic_fabric.py).")
    engine = create_engine('sqlite://', creator=refuse, poolclass=NullPool)
    engine._no_db_name = name
    return engine


ENGINES = [(preprocessing, '_fabric_engine'), (lifecycle_cleaning, '_ai_engine'),
           (db_writer, '_engine'), (collate_outstanding_order, '_engine')]


def install() -> None:
    for module, name in ENGINES:
        setattr(module, name, _forbidden_engine(f'{module.__name__}.{name}'))


def installed() -> bool:
    from adj_forecast_lib import config
    return (all(getattr(getattr(m, n), '_no_db_name', None) for m, n in ENGINES)
            and config.MLFLOW_TRACKING_URI == os.environ['MLFLOW_TRACKING_URI']
            and _MLFLOW_DIR in config.MLFLOW_TRACKING_URI)


install()
