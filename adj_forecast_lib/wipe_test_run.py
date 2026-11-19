"""Wipes a test run's DB rows (PredictionResult, PredictionResultLog, ActualOrder, OutstandingOrder)
and MLflow registered models (standard + '{label}_quantile', horizons 1-6) for one plant. Dry-run
by default.

Usage:
    python -m adj_forecast_lib.wipe_test_run --plant 3300 --label qa_v1                # dry run
    python -m adj_forecast_lib.wipe_test_run --plant 3300 --label qa_v1 --confirm      # deletes
"""
import argparse
import logging
import os
import urllib.parse

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import mlflow
from mlflow.tracking import MlflowClient

from .config import PLANT_NAMES, MLFLOW_TRACKING_URI
from .logging_setup import configure_logging
from .model_registry import _registered_name

log = logging.getLogger(__name__)

AI_DATABASE = 'AI'
SCHEMA = 'DemandForecast'
# Excludes ProductLifecycle/ProductSuccessor - both upserted, never wiped, not test-run-specific.
_DB_TABLES = ('PredictionResult', 'PredictionResultLog', 'ActualOrder', 'OutstandingOrder')

load_dotenv()


def _db_engine():
    odbc_str = (
        f"DRIVER={{{os.environ['ODBC_DRIVER']}}};"
        f"SERVER={os.environ['QA_DB_SERVER']};"
        f"UID={os.environ['QA_DB_UID']};"
        f"PWD={os.environ['QA_DB_PWD']};"
    )
    return create_engine("mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(odbc_str),
        pool_pre_ping=True)


def wipe_db_tables(plant: int, confirm: bool) -> None:
    # NOT scoped by `label` - deletes every row for `plant`, any Method/label. Only safe on a
    # plant holding nothing but this test run (unlike wipe_mlflow_models below, which IS label-scoped).
    engine = _db_engine()
    with engine.begin() as conn:
        for table in _DB_TABLES:
            count = conn.execute(text(
                f"SELECT COUNT(*) FROM [{AI_DATABASE}].{SCHEMA}.{table} WHERE Plant = :plant"
            ), {'plant': str(plant)}).scalar()
            log.info("%s: %d row(s) for plant=%s", table, count, plant)
            if confirm and count:
                conn.execute(text(
                    f"DELETE FROM [{AI_DATABASE}].{SCHEMA}.{table} WHERE Plant = :plant"
                ), {'plant': str(plant)})
                log.info("%s: deleted", table)


def wipe_mlflow_models(plant: int, label: str, confirm: bool) -> None:
    """Deletes both the registered model (registry entry + versions) and each version's
    underlying run - save_model() creates one run per version, holding the actual model artifact/
    eval_history, which delete_registered_model() alone never touches."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    for lbl in (label, f'{label}_quantile'):
        for h in range(1, 7):
            name = _registered_name(plant, h, lbl)
            versions = client.search_model_versions(f"name='{name}'")
            log.info("%s: %d version(s)", name, len(versions))
            if confirm and versions:
                for mv in versions:
                    client.delete_run(mv.run_id)
                client.delete_registered_model(name)
                log.info("%s: deleted (registry + %d run(s))", name, len(versions))


def _parse_args():
    p = argparse.ArgumentParser(description="Wipe DB + MLflow output from a test run of main.py.")
    p.add_argument('--plant', type=int, required=True)
    p.add_argument('--label', required=True, help="Label used for the test run (e.g. DAILY_LABEL).")
    p.add_argument('--confirm', action='store_true', help="Actually delete. Omit for a dry-run.")
    return p.parse_args()


def main():
    args = _parse_args()
    configure_logging('wipe_test_run', args.plant)
    plant_name = PLANT_NAMES.get(args.plant, args.plant)
    log.info("%s: plant=%s (%s) label=%s",
        "DELETING" if args.confirm else "DRY RUN", args.plant, plant_name, args.label)
    wipe_db_tables(args.plant, args.confirm)
    wipe_mlflow_models(args.plant, args.label, args.confirm)


if __name__ == '__main__':
    main()
