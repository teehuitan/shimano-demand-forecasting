"""Computes OutstandingOrder per PlanModelCode from the latest daily source file."""
import argparse
import logging
import os
import urllib.parse

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from .config import (
    IO_OUTSTANDING_SQL_PATH, PLANT_EXCLUDE_PARTCODES, PLANT_EXCLUDES_SOLD_FROM_OUTSTANDING,
    date_to_month_idx, to_month_idx, month_idx_to_yyyymm, plant_name,
    PLANT_SUBMODEL_PLAN_MODEL_OVERRIDES,
)
from .data_loading import clean_code
from .logging_setup import configure_logging
from .lifecycle_cleaning import resolve_plan_model_codes
from .preprocessing import (
    _bring_forward_ship_to_request_date, _flag_known_cancellations, latest_source_file_date,
    load_tfg_material,
)

log = logging.getLogger(__name__)

load_dotenv()
_odbc_str = (
    f"DRIVER={{{os.environ['ODBC_DRIVER']}}};"
    f"SERVER={os.environ['FABRIC_DB_SERVER']};"
    f"UID={os.environ['FABRIC_DB_UID']};"
    f"PWD={os.environ['FABRIC_DB_PWD']};"
    "Authentication=ActiveDirectoryPassword;"
    "Encrypt=yes;"
)
_engine = create_engine(
    "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(_odbc_str),
    pool_pre_ping=True,
)


def _load_orders_for_file(plant: int, source_file: str, exclude_partcodes=None) -> pd.DataFrame:
    """Loads one source file with load_io's exclusion rules plus STJ's SLH_Order='Y' drop; raises
    ValueError if it has no rows."""
    query = text(open(IO_OUTSTANDING_SQL_PATH, encoding='utf-8').read())
    io = pd.read_sql(query, _engine, params={'plant': plant, 'source_file': source_file})
    if io.empty:
        raise ValueError(f"No rows found for plant={plant}, source_file={source_file!r} - "
            "check the file actually exists in IncomingOrder_GoogleLooker.")
    log.info("Loaded %d row(s): plant=%s source_file=%s", len(io), plant, source_file)

    if exclude_partcodes is None:
        exclude_partcodes = PLANT_EXCLUDE_PARTCODES[plant]

    io = io.rename(columns={'MaterialCode': 'ProductCode', 'Product_Type_Code': 'PartCode',
        'Model_Code': 'SalesModelCode', 'Plan_Model_Code': 'PlanModelCode', 'Quantity': 'Qty'})
    for col in ['ProductCode', 'SalesModelCode', 'PlanModelCode']:
        io[col] = clean_code(io[col]).replace('', np.nan)
    io, _ = resolve_plan_model_codes(io, load_tfg_material(), entry_col='Order_Entry_Date Date',
        submodel_overrides=PLANT_SUBMODEL_PLAN_MODEL_OVERRIDES[plant])

    before = len(io)
    io = io[~io['PartCode'].isin(list(exclude_partcodes))].copy()
    log.info("Dropped %d row(s) with PartCode in exclude_partcodes=%s.", before - len(io), exclude_partcodes)

    # Outstanding only - load_io keeps these rows as real demand. 'Y' = an SLH order already
    # shipped, so it is no longer outstanding (README §13).
    if plant == 3300:
        before = len(io)
        io = io[io['SLH_Order'] != 'Y'].copy()
        log.info("Dropped %d row(s) flagged SLH_Order='Y'.", before - len(io))
    io = io.drop(columns=['SLH_Order'])

    # Same rule as load_io: a negative line (a return or correction) is not an outstanding order.
    negative = io['Qty'] < 0
    if negative.any():
        log.warning("Dropped %d order row(s) with a negative Qty (total %.0f).", int(negative.sum()),
            float(io.loc[negative, 'Qty'].sum()))
    io = io[~negative].copy()

    entry_date = pd.to_datetime(io['Order_Entry_Date Date'])
    # Same bring-forward as load_io; rows it moves into an already-closed month show as outstanding.
    request_date = _bring_forward_ship_to_request_date(
        io, pd.to_datetime(io['Customer_Request_Date Date']), plant)
    cancel_date = pd.to_datetime(io['Cancel_Date Date'])
    io = _flag_known_cancellations(io, entry_date, request_date, cancel_date)
    # Sold = BOL received and shipment departed from Singapore port; recognized as monthly sales -
    # captured here before Order_Status_Text is dropped below.
    io['IsSold'] = io['Order_Status_Text'] == 'Sold'
    io['RequestDate'] = request_date
    io['CancelDate'] = cancel_date
    io = io.drop(columns=['Order_Entry_Date Date', 'Customer_Request_Date Date', 'Cancel_Date Date',
        'Order_Status_Text', 'Ship_To_Code'])

    return io


def collate_outstanding_order(orders: pd.DataFrame, file_date: pd.Timestamp, source_file: str,
        plant: int, settled_month_idx: int = None) -> pd.DataFrame:
    """Qty of orders requested before target month N+1, excluding cancelled (and per-plant Sold) rows.
    `settled_month_idx` is N; None takes N+1 as file_date's month."""
    exclude_sold = PLANT_EXCLUDES_SOLD_FROM_OUTSTANDING[plant]
    current_month_idx = (int(settled_month_idx) + 1 if settled_month_idx is not None
                         else to_month_idx(file_date.year * 100 + file_date.month))

    request_month_idx = date_to_month_idx(orders['RequestDate'])
    before_current_month = request_month_idx < current_month_idx

    is_sold_excluded = before_current_month & orders['IsSold'] & exclude_sold
    if exclude_sold:
        log.info("OutstandingOrder as of %s: %d row(s) excluded as Sold, totaling %.0f unit(s).",
            file_date.date(), int(is_sold_excluded.sum()), orders.loc[is_sold_excluded, 'Qty'].sum())
    else:
        log.info("OutstandingOrder as of %s: plant %s does not exclude Sold rows "
            "(PLANT_EXCLUDES_SOLD_FROM_OUTSTANDING).", file_date.date(), plant)

    # Every Canceled row is excluded whatever its Cancel_Date - after file_date or missing are
    # anomalies, logged for visibility only.
    cancel_after_file_date = orders['IsCanceled'] & (orders['CancelDate'] > file_date)
    cancel_date_missing = orders['IsCanceled'] & orders['CancelDate'].isna()
    log.info("OutstandingOrder as of %s: %d CanceledOrder(s) have Cancel_Date after file_date and %d "
        "have none - still counted as canceled.", file_date.date(), int(cancel_after_file_date.sum()),
        int(cancel_date_missing.sum()))

    is_canceled_excluded = before_current_month & orders['IsCanceled']
    log.info("OutstandingOrder as of %s: %d row(s) excluded as canceled, totaling %.0f unit(s).",
        file_date.date(), int(is_canceled_excluded.sum()), orders.loc[is_canceled_excluded, 'Qty'].sum())

    is_outstanding_candidate = before_current_month & ~is_sold_excluded & ~is_canceled_excluded
    outstanding = orders[is_outstanding_candidate].groupby('PlanModelCode')['Qty'].sum() \
        .rename('OutstandingOrder').reset_index()
    outstanding = outstanding.join(orders.groupby('PlanModelCode')[['PartCode', 'SalesModelCode']].first(),
        on='PlanModelCode')
    outstanding['SourceFileName'] = source_file
    outstanding['TargetYearMonth'] = str(int(month_idx_to_yyyymm([current_month_idx])[0]))

    return outstanding[['PartCode', 'SalesModelCode', 'PlanModelCode', 'SourceFileName', 'TargetYearMonth',
        'OutstandingOrder']] \
        .sort_values('PlanModelCode').reset_index(drop=True)


def _parse_args():
    p = argparse.ArgumentParser(
        description="Compute OutstandingOrder per PlanModelCode as of a source file's date.")
    p.add_argument('--plant', type=int, required=True, help="Plant code (2100=SPL, 3300=STJ).")
    p.add_argument('--file-date', type=int, default=None,
        help="YYYYMMDD - use this specific source file instead of the latest one (e.g. to "
             "reproduce a past day's run). Omit to auto-detect the latest file.")
    return p.parse_args()


def main():
    args = _parse_args()
    configure_logging('collate_outstanding_order', args.plant)

    if args.file_date is not None:
        file_date = pd.Timestamp(str(args.file_date))
        source_file = f'{args.file_date}.xlsx'
    else:
        file_date, source_file = latest_source_file_date(args.plant)
        if file_date is None:
            log.warning("Latest source file for plant %s doesn't follow the plain YYYYMMDD.xlsx "
                "convention - skipping this run (can't reliably determine a reference date).", args.plant)
            return

    orders = _load_orders_for_file(args.plant, source_file)
    result = collate_outstanding_order(orders, file_date, source_file, args.plant)
    log.info("Computed OutstandingOrder for %d PlanModelCode(s) as of %s (source_file=%s).",
        len(result), file_date.date(), source_file)

    out_path = f'results/{plant_name(args.plant)}/outstanding_order_{file_date.strftime("%Y%m%d")}.parquet'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.to_parquet(out_path, index=False)
    log.info("Saved %d row(s) to %s", len(result), out_path)


if __name__ == '__main__':
    main()
