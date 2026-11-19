"""Writes pipeline outputs into the DemandForecast SQL Server schema (create_result_tables.sql)."""
import logging
import os
import urllib

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

AI_DATABASE = 'AI'
SCHEMA = 'DemandForecast'

log = logging.getLogger(__name__)

load_dotenv()
_ai_odbc_str = (
    f"DRIVER={{{os.environ['ODBC_DRIVER']}}};"
    f"SERVER={os.environ['QA_DB_SERVER']};"
    f"UID={os.environ['QA_DB_UID']};"
    f"PWD={os.environ['QA_DB_PWD']};"
)
_engine = create_engine(
    "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(_ai_odbc_str),
    pool_pre_ping=True,
    fast_executemany=True,
)


def _try(step, plant, fn, /, *args, **kwargs):
    """Same pattern as main.py's own _try - lets one step of a bundled transaction (see
    month_start_write_clear) still name itself on failure."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        log.exception("%s FAILED: plant=%s", step, plant)
        raise


def has_outstanding_for_file(plant: int, source_file: str) -> bool:
    """True if OutstandingOrder has rows from `source_file` (the "file done" marker)."""
    with _engine.begin() as conn:
        return conn.execute(text(
            f"SELECT TOP 1 1 FROM [{AI_DATABASE}].{SCHEMA}.OutstandingOrder "
            "WHERE Plant = :plant AND SourceFileName = :source_file"
        ), {'plant': str(plant), 'source_file': source_file}).first() is not None


def has_actual_for_month(plant: int, target_yyyymm: int) -> bool:
    """True if ActualOrder already holds rows for `target_yyyymm`."""
    with _engine.begin() as conn:
        return conn.execute(text(
            f"SELECT TOP 1 1 FROM [{AI_DATABASE}].{SCHEMA}.ActualOrder "
            "WHERE Plant = :plant AND TargetYearMonth = :target"
        ), {'plant': str(plant), 'target': str(int(target_yyyymm))}).first() is not None


def has_prediction_at(plant: int, origin_date) -> bool:
    """True if PredictionResult already holds rows at `origin_date`."""
    with _engine.begin() as conn:
        return conn.execute(text(
            f"SELECT TOP 1 1 FROM [{AI_DATABASE}].{SCHEMA}.PredictionResult "
            "WHERE Plant = :plant AND OriginDate = :origin_date"
        ), {'plant': str(plant), 'origin_date': origin_date}).first() is not None


def _delete_prediction_result(conn, plant: int, origin_date=None) -> int:
    """All of `plant`'s rows, or only those at `origin_date` (a rerun of the same day)."""
    sql = f"DELETE FROM [{AI_DATABASE}].{SCHEMA}.PredictionResult WHERE Plant = :plant"
    params = {'plant': str(plant)}
    if origin_date is not None:
        sql += " AND OriginDate = :origin_date"
        params['origin_date'] = origin_date
    deleted = conn.execute(text(sql), params)
    log.info("clear_prediction_result: removed %d row(s) for plant %s", deleted.rowcount, plant)
    return deleted.rowcount


def clear_prediction_result(plant: int) -> int:
    """Unconditionally deletes every PredictionResult row for `plant`. Own transaction - see
    month_start_write_clear for the atomic version main.py uses."""
    with _engine.begin() as conn:
        return _delete_prediction_result(conn, plant)


def _insert_prediction_result(conn, result_df: pd.DataFrame, plant: int, origin_date) -> int:
    """Inserts `result_df` into PredictionResult and appends the same rows to PredictionResultLog
    (write-only, never read back). Requires KnownCanceledOrder on result_df (Net = Gross minus it)."""
    if result_df.empty:
        log.warning("write_prediction_result: nothing to write for plant %s - the previous "
            "forecast is kept.", plant)
        return 0

    rows = [
        {
            'Plant': str(plant),
            'PartCode': r.PartCode,
            'SalesModelCode': r.SalesModelCode,
            'PlanModelCode': r.PlanModelCode,
            'OriginDate': origin_date,
            'TargetYearMonth': str(r.TargetYYYYMM),
            'Horizon': int(r.Horizon),
            'KnownGrossOrder': float(r.KnownOrder),
            'KnownNetOrder': float(r.KnownOrder - r.KnownCanceledOrder),
            'Method': r.Method,
            'PredictedGrossOrder': float(r.Predicted),
            'PredictedNetOrder': float(r.Predicted - r.KnownCanceledOrder),
        }
        for r in result_df.itertuples()
    ]

    # PredictionResult takes SalesModelCode from ProductLifecycle; the Log keeps it, frozen as of
    # this prediction. The unused SalesModelCode key in `rows` is ignored by the first insert.
    conn.execute(text(f"""
        INSERT INTO [{AI_DATABASE}].{SCHEMA}.PredictionResult
            (Plant, PartCode, PlanModelCode, OriginDate, TargetYearMonth, Horizon,
             KnownGrossOrder, KnownNetOrder, Method, PredictedGrossOrder, PredictedNetOrder)
        VALUES
            (:Plant, :PartCode, :PlanModelCode, :OriginDate, :TargetYearMonth, :Horizon,
             :KnownGrossOrder, :KnownNetOrder, :Method, :PredictedGrossOrder, :PredictedNetOrder)
    """), rows)
    conn.execute(text(f"""
        INSERT INTO [{AI_DATABASE}].{SCHEMA}.PredictionResultLog
            (Plant, PartCode, SalesModelCode, PlanModelCode, OriginDate, TargetYearMonth, Horizon,
             KnownGrossOrder, KnownNetOrder, Method, PredictedGrossOrder, PredictedNetOrder)
        VALUES
            (:Plant, :PartCode, :SalesModelCode, :PlanModelCode, :OriginDate, :TargetYearMonth, :Horizon,
             :KnownGrossOrder, :KnownNetOrder, :Method, :PredictedGrossOrder, :PredictedNetOrder)
    """), rows)
    log.info("write_prediction_result: inserted %d row(s) into PredictionResult and PredictionResultLog "
        "for plant %s", len(rows), plant)
    return len(rows)


def write_prediction_result(result_df: pd.DataFrame, plant: int, origin_date) -> int:
    """Writes PredictionResult (+ Log) in one transaction, replacing any rows at `origin_date`."""
    if result_df.empty:
        log.warning("write_prediction_result: nothing to write for plant %s - the previous "
            "forecast is kept.", plant)
        return 0
    with _engine.begin() as conn:
        _delete_prediction_result(conn, plant, origin_date)
        return _insert_prediction_result(conn, result_df, plant, origin_date)


def write_product_lifecycle(lifecycle_df: pd.DataFrame, plant: int) -> int:
    """Upsert keyed on (Plant, PlanModelCode) - never wiped-and-reinserted, so readers never see
    it momentarily empty mid-refresh."""
    if lifecycle_df.empty:
        log.info("write_product_lifecycle: nothing to write for plant %s", plant)
        return 0
    rows = [
        {
            'Plant': str(plant),
            'PartCode': r.PartCode,
            'SalesModelCode': r.SalesModelCode,
            'PlanModelCode': r.PlanModelCode,
            'PlanModelText': None if pd.isna(r.PlanModelText) else r.PlanModelText,
            'StartYearMonth': None if pd.isna(r.StartYearMonth) else str(int(r.StartYearMonth)),
            'EndYearMonth': None if pd.isna(r.EndYearMonth) else str(int(r.EndYearMonth)),
        }
        for r in lifecycle_df.itertuples()
    ]
    with _engine.begin() as conn:
        conn.execute(text(f"""
            MERGE [{AI_DATABASE}].{SCHEMA}.ProductLifecycle AS tgt
            USING (VALUES (:Plant, :PartCode, :SalesModelCode, :PlanModelCode, :PlanModelText, :StartYearMonth,
                    :EndYearMonth))
                AS src (Plant, PartCode, SalesModelCode, PlanModelCode, PlanModelText, StartYearMonth, EndYearMonth)
                ON tgt.Plant = src.Plant AND tgt.PlanModelCode = src.PlanModelCode
            WHEN MATCHED AND (tgt.PartCode <> src.PartCode OR tgt.SalesModelCode <> src.SalesModelCode
                    OR ISNULL(tgt.PlanModelText, '') <> ISNULL(src.PlanModelText, '')
                    OR ISNULL(tgt.StartYearMonth, '') <> ISNULL(src.StartYearMonth, '')
                    OR ISNULL(tgt.EndYearMonth, '') <> ISNULL(src.EndYearMonth, '')) THEN
                UPDATE SET PartCode = src.PartCode, SalesModelCode = src.SalesModelCode,
                    PlanModelText = src.PlanModelText, StartYearMonth = src.StartYearMonth,
                    EndYearMonth = src.EndYearMonth, UpdatedAt = SYSDATETIME()
            WHEN NOT MATCHED THEN
                INSERT (Plant, PartCode, SalesModelCode, PlanModelCode, PlanModelText, StartYearMonth, EndYearMonth)
                VALUES (src.Plant, src.PartCode, src.SalesModelCode, src.PlanModelCode, src.PlanModelText,
                    src.StartYearMonth, src.EndYearMonth);
        """), rows)
    log.info("write_product_lifecycle: %d row(s) submitted for plant %s (insert/update split not "
        "obtainable from MERGE at this batch size)", len(rows), plant)
    return len(rows)


def _upsert_actual_order(conn, actual_df: pd.DataFrame, plant: int) -> int:
    if actual_df.empty:
        log.info("write_actual_order: nothing to write for plant %s", plant)
        return 0
    rows = [
        {
            'Plant': str(plant),
            'PartCode': r.PartCode,
            'PlanModelCode': r.PlanModelCode,
            'TargetYearMonth': str(int(r.TargetYYYYMM)),
            'ActualGrossOrder': float(r.ActualGrossOrder),
            'ActualNetOrder': float(r.ActualNetOrder),
            'SalesResult': None if pd.isna(r.SalesResult) else float(r.SalesResult),
        }
        for r in actual_df.itertuples()
    ]
    # SalesResult is nullable, so its change check can't be a plain <> (NULL <> x is never true).
    conn.execute(text(f"""
        MERGE [{AI_DATABASE}].{SCHEMA}.ActualOrder AS tgt
        USING (VALUES (:Plant, :PartCode, :PlanModelCode, :TargetYearMonth, :ActualGrossOrder, :ActualNetOrder,
                :SalesResult))
            AS src (Plant, PartCode, PlanModelCode, TargetYearMonth, ActualGrossOrder, ActualNetOrder, SalesResult)
            ON tgt.Plant = src.Plant AND tgt.PlanModelCode = src.PlanModelCode
                AND tgt.TargetYearMonth = src.TargetYearMonth
        WHEN MATCHED AND (tgt.ActualGrossOrder <> src.ActualGrossOrder
                OR tgt.ActualNetOrder <> src.ActualNetOrder
                OR EXISTS (SELECT tgt.SalesResult EXCEPT SELECT src.SalesResult)) THEN
            UPDATE SET ActualGrossOrder = src.ActualGrossOrder, ActualNetOrder = src.ActualNetOrder,
                SalesResult = src.SalesResult, UpdatedAt = SYSDATETIME()
        WHEN NOT MATCHED THEN
            INSERT (Plant, PartCode, PlanModelCode, TargetYearMonth, ActualGrossOrder, ActualNetOrder, SalesResult)
            VALUES (src.Plant, src.PartCode, src.PlanModelCode, src.TargetYearMonth,
                src.ActualGrossOrder, src.ActualNetOrder, src.SalesResult);
    """), rows)
    log.info("write_actual_order: %d row(s) submitted for plant %s (insert/update split not "
        "obtainable from MERGE at this batch size)", len(rows), plant)
    return len(rows)


def write_actual_order(actual_df: pd.DataFrame, plant: int) -> int:
    """Upsert keyed on (Plant, PlanModelCode, TargetYearMonth). Own transaction - see
    month_start_write_clear for the atomic version main.py uses."""
    if actual_df.empty:
        log.info("write_actual_order: nothing to write for plant %s", plant)
        return 0
    with _engine.begin() as conn:
        return _upsert_actual_order(conn, actual_df, plant)


def month_start_write_clear(actual_df: pd.DataFrame, result_df: pd.DataFrame, plant: int,
        origin_date) -> tuple:
    """Everything the new month needs, one transaction: upsert ActualOrder, clear the plant's
    PredictionResult and OutstandingOrder, insert day 0's PredictionResult (+ Log). Refuses an
    empty `result_df`, so a failed run keeps last month's rows."""
    if result_df.empty:
        log.warning("month_start_write_clear: nothing scored for plant %s - nothing "
            "written, the previous forecast is kept.", plant)
        return 0, 0, 0
    with _engine.begin() as conn:
        n_actual = _try('upsert_actual_order', plant, _upsert_actual_order, conn, actual_df, plant)
        _try('clear_outstanding_order', plant, _delete_outstanding_order, conn, plant)
        n_cleared = _try('clear_prediction_result', plant, _delete_prediction_result, conn, plant)
        n_written = _try('write_prediction_result', plant, _insert_prediction_result,
            conn, result_df, plant, origin_date)
    return n_actual, n_cleared, n_written


def _delete_outstanding_order(conn, plant: int, source_file=None) -> int:
    """All of `plant`'s rows, or only those from `source_file` (a rerun of the same file)."""
    sql = f"DELETE FROM [{AI_DATABASE}].{SCHEMA}.OutstandingOrder WHERE Plant = :plant"
    params = {'plant': str(plant)}
    if source_file is not None:
        sql += " AND SourceFileName = :source_file"
        params['source_file'] = source_file
    deleted = conn.execute(text(sql), params)
    log.info("clear_outstanding_order: removed %d row(s) for plant %s", deleted.rowcount, plant)
    return deleted.rowcount


def clear_outstanding_order(plant: int) -> int:
    """Unconditionally deletes every OutstandingOrder row for `plant`. Own transaction - see
    month_start_write_clear for the atomic version main.py uses."""
    with _engine.begin() as conn:
        return _delete_outstanding_order(conn, plant)


def _insert_outstanding_order(conn, outstanding_df: pd.DataFrame, plant: int) -> int:
    if outstanding_df.empty:
        log.info("write_outstanding_order: nothing to write for plant %s", plant)
        return 0
    rows = [
        {
            'Plant': str(plant),
            'PartCode': r.PartCode,
            'PlanModelCode': r.PlanModelCode,
            'SourceFileName': r.SourceFileName,
            'TargetYearMonth': r.TargetYearMonth,
            'OutstandingOrder': float(r.OutstandingOrder),
        }
        for r in outstanding_df.itertuples()
    ]
    conn.execute(text(f"""
        INSERT INTO [{AI_DATABASE}].{SCHEMA}.OutstandingOrder
            (Plant, PartCode, PlanModelCode, SourceFileName, TargetYearMonth, OutstandingOrder)
        VALUES (:Plant, :PartCode, :PlanModelCode, :SourceFileName, :TargetYearMonth, :OutstandingOrder)
    """), rows)
    log.info("write_outstanding_order: inserted %d row(s) for plant %s", len(rows), plant)
    return len(rows)


def write_outstanding_order(outstanding_df: pd.DataFrame, plant: int, source_file: str) -> int:
    """Inserts OutstandingOrder in one transaction, replacing any rows from `source_file` (a rerun)."""
    with _engine.begin() as conn:
        _delete_outstanding_order(conn, plant, source_file)
        return _insert_outstanding_order(conn, outstanding_df, plant)


def upsert_product_successor(pred_succ_df: pd.DataFrame, plant: int) -> int:
    """Upserts (SalesModelCode, SuccessorCode) pairs; `plant` is provenance only, not a match key."""
    exploded = pred_succ_df[['SalesModelCode', 'Successor']].explode('Successor') \
        .dropna(subset=['Successor'])
    rows = [
        {'SourcePlant': str(plant), 'SalesModelCode': r.SalesModelCode, 'SuccessorCode': r.Successor}
        for r in exploded.itertuples()
    ]
    if not rows:
        log.info("upsert_product_successor: nothing to add, submitted by plant %s", plant)
        return 0
    with _engine.begin() as conn:
        conn.execute(text(f"""
            MERGE [{AI_DATABASE}].{SCHEMA}.ProductSuccessor AS tgt
            USING (VALUES (:SourcePlant, :SalesModelCode, :SuccessorCode))
                AS src (SourcePlant, SalesModelCode, SuccessorCode)
                ON tgt.SalesModelCode = src.SalesModelCode AND tgt.SuccessorCode = src.SuccessorCode
            WHEN MATCHED THEN
                UPDATE SET SourcePlant = src.SourcePlant, UpdatedAt = SYSDATETIME()
            WHEN NOT MATCHED THEN
                INSERT (SourcePlant, SalesModelCode, SuccessorCode)
                VALUES (src.SourcePlant, src.SalesModelCode, src.SuccessorCode);
        """), rows)
    log.info("upsert_product_successor: %d pair(s) submitted by plant %s (existing pairs "
        "re-confirmed, new pairs inserted)", len(rows), plant)
    return len(rows)


def delete_product_successor(pred_succ_df: pd.DataFrame, plant: int) -> int:
    """Deletes (SalesModelCode, SuccessorCode) pairs; input shape as upsert_product_successor."""
    exploded = pred_succ_df[['SalesModelCode', 'Successor']].explode('Successor') \
        .dropna(subset=['Successor'])
    rows = [
        {'SalesModelCode': r.SalesModelCode, 'SuccessorCode': r.Successor}
        for r in exploded.itertuples()
    ]
    if not rows:
        log.info("delete_product_successor: nothing to delete, submitted by plant %s", plant)
        return 0
    with _engine.begin() as conn:
        conn.execute(text(f"""
            DELETE FROM [{AI_DATABASE}].{SCHEMA}.ProductSuccessor
            WHERE SalesModelCode = :SalesModelCode AND SuccessorCode = :SuccessorCode
        """), rows)
    log.info("delete_product_successor: %d pair(s) submitted by plant %s (pairs not currently "
        "stored are no-ops)", len(rows), plant)
    return len(rows)
