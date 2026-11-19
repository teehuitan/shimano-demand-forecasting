"""End-to-end panel construction shared by train.py/predict.py/evaluate.py, so train-time and
predict-time feature computation never drift apart."""
import logging
import os
import re
import urllib

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from .config import (
    to_month_idx, month_idx_to_yyyymm, date_to_month_idx, IO_SQL_PATH, PLANT_EXTERNAL_EVENT_RANGES,
    PLANT_EXTERNAL_EVENT_END_KNOWN, PLANT_INCLUDES_CNY, CURRENCY_ALIASES, PLANT_NAMES,
    PLANT_EXCLUDE_PARTCODES, PLANT_INCLUDES_PRED_SUCC, TRAIN_EXTEND_TO_ORDERS,
    PLANT_SUBMODEL_PLAN_MODEL_OVERRIDES,
    FOCUS_COUNTRY, PLANT_INCLUDES_COUNTRY_VISIBILITY, COUNTRY_MIX_WINDOW, COUNTRY_MIX_MIN_SHARE,
    CUTOFF_GRAINS, CUTOFF_DATE_COL, MIN_VALID_ORDER_DATE,
)
from .data_loading import clean_code
from .order_book import build_order_book, known_canceled_order
# Re-exported, not merely used: these four moved to cutoff_grain so feature modules could reach
# them (preprocessing imports features, so it can't be the home). Every existing
# `from .preprocessing import cutoff_dates_for_origins` and the tests keep working.
from .cutoff_grain import (  # noqa: F401
    month_bounds as _month_bounds, cutoff_dates_for_origins, monthly_anchor_mask, cutoff_day_index,
    cutoff_window,
)
from .features import (
    attach_calendar_features, attach_history_features, attach_known_order_features,
    attach_historical_visibility, attach_external_factor_features, attach_adi_features,
    attach_lag_event_features, attach_trend_stability_features,
    attach_adjustment_trailing_features, attach_manual_sim_features,
    build_market_shock_monthly, attach_market_shock_features,
)
from .intramonth_features import attach_intramonth_features, INTRAMONTH_FEATURE_COLS
from .country_visibility import (
    build_country_visibility, build_sku_country_mix, build_focus_country_order_book,
    attach_country_visibility_features, country_visibility_feature_cols,
)
from .lifecycle_features import (
    attach_lifecycle_features, compute_dataset_floor,
    resolve_eligibility_launch, attach_forecast_eligibility_age,
)
from .folds import scoreable_rows
from .lifecycle_cleaning import (
    resolve_effective_ranges_with_tfg, get_predecessor_successor_flags_from_db, resolve_product_code_ranges,
    resolve_plan_model_codes,
)
from .logging_setup import LOG_FORMAT

log = logging.getLogger(__name__)

_ENTITY_TYPES = ('PlanModelCode', 'PartCode')
# SIC's 'Plan Model ID' IS the panel's PlanModelCode - SIC's own 'Plan Model Code' is never a key.
_SIC_ENTITY_TYPES = ('PlanModelCode', 'PartCode')

load_dotenv()

_fabric_odbc_str = (
    f"DRIVER={{{os.environ['ODBC_DRIVER']}}};"
    f"SERVER={os.environ['FABRIC_DB_SERVER']};"
    f"UID={os.environ['FABRIC_DB_UID']};"
    f"PWD={os.environ['FABRIC_DB_PWD']};"
    "Authentication=ActiveDirectoryPassword;"
    "Encrypt=yes;"
)
_fabric_engine = create_engine(
    "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(_fabric_odbc_str),
    pool_pre_ping=True,
)

def _check_entity_type(entity_type):
    if entity_type not in _ENTITY_TYPES:
        raise ValueError(f"entity_type must be one of {_ENTITY_TYPES}, got {entity_type!r}")


# Ship_To_Code whose Customer_Request_Date is one month later than the real requested month -
# business-confirmed, STJ-only. See _bring_forward_ship_to_request_date.
_SHIP_TO_BRING_FORWARD_CODE = '3003328'  # Ship_To_Code comes back from SQL as a string, not int


def _bring_forward_ship_to_request_date(io: pd.DataFrame, request_date: pd.Series, plant: int) -> pd.Series:
    """Shifts Customer_Request_Date back one month for Ship_To_Code=3003328 rows - STJ-only rule;
    logs `plant` every call as a tripwire for a future SPL match (see README §13)."""
    mask = io['Ship_To_Code'].astype(str) == _SHIP_TO_BRING_FORWARD_CODE
    log.info("Ship_To_Code=%s: brought Customer_Request_Date forward one month for %d row(s), "
        "plant=%s - STJ(3300)-only rule; a nonzero count for any other plant means it's firing "
        "somewhere it shouldn't.", _SHIP_TO_BRING_FORWARD_CODE, int(mask.sum()), plant)
    request_date = request_date.copy()
    request_date.loc[mask] = request_date.loc[mask] - pd.DateOffset(months=1)
    return request_date


def _flag_known_cancellations(io: pd.DataFrame, entry_date: pd.Series, request_date: pd.Series,
        cancel_date: pd.Series) -> pd.DataFrame:
    """Adds IsCanceled/CancelMonthIdx/CancelKnownDate; a cancel date is never before the entry."""
    is_canceled = io['Order_Status_Text'] == 'Canceled'
    missing_cancel_date = is_canceled & cancel_date.isna()
    before_entry = is_canceled & cancel_date.notna() & (cancel_date < entry_date)
    after_request = is_canceled & cancel_date.notna() & (cancel_date > request_date)
    log.info("%d CanceledOrder(s) missing Cancel_Date - known as canceled from their entry; "
        "still counted as canceled.", missing_cancel_date.sum())
    log.info("%d CanceledOrder(s) have Cancel_Date before Order_Entry_Date - known as canceled from "
        "their entry; still counted as canceled.", before_entry.sum())
    log.info("%d CanceledOrder(s) have Cancel_Date after Customer_Request_Date - still counted as "
        "canceled; the anomalous Cancel_Date is logged for visibility only.", after_request.sum())

    cancel_known_date = cancel_date.where(~missing_cancel_date & ~before_entry, entry_date)
    io['IsCanceled'] = is_canceled
    io['CancelMonthIdx'] = date_to_month_idx(cancel_known_date).astype('float64')
    io['CancelKnownDate'] = cancel_known_date
    io.loc[~is_canceled, 'CancelMonthIdx'] = np.nan  # NaN for non-canceled rows
    io.loc[~is_canceled, 'CancelKnownDate'] = pd.NaT
    return io


def _valid_order_dates(entry_date: pd.Series, request_date: pd.Series, now=None) -> pd.Series:
    """True for rows with valid entry and request dates; logs a warning per reason for the rest."""
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    earliest = pd.Timestamp(MIN_VALID_ORDER_DATE)
    latest_entry = now.normalize() + pd.Timedelta(days=1)     # one day of slack for time zones
    reasons = {
        'no Order_Entry_Date': entry_date.isna(),
        'no Customer_Request_Date': request_date.isna(),
        f'Order_Entry_Date before {earliest.date()}': entry_date < earliest,
        f'Customer_Request_Date before {earliest.date()}': request_date < earliest,
        f'Order_Entry_Date after {latest_entry.date()}, i.e. in the future': entry_date > latest_entry,
    }
    invalid = pd.Series(False, index=entry_date.index)
    for reason, mask in reasons.items():
        if mask.any():
            log.warning("%d order row(s) dropped - %s.", int(mask.sum()), reason)
        invalid |= mask
    return ~invalid


def load_io(plant: int, exclude_partcodes=None) -> pd.DataFrame:
    """Loads and cleans one plant's incoming orders from Fabric's IncomingOrder_GoogleLooker view."""
    if exclude_partcodes is None:
        exclude_partcodes = PLANT_EXCLUDE_PARTCODES[plant]

    query = text(open(IO_SQL_PATH, encoding='utf-8').read())
    io = pd.read_sql(query, _fabric_engine, params={'plant': plant})
    log.info("Loaded %d row(s) from Fabric: plant=%s", len(io), plant)

    io = io.rename(columns={
        'MaterialCode': 'ProductCode',
        'Model_Code': 'SalesModelCode',
        'Plan_Model_Code': 'PlanModelCode',
        'Product_Type_Code': 'PartCode',
        'Quantity': 'Qty',
        'Currency_Code': 'Currency',
        'Customer_Country_Code': 'CustomerCountry',
        'Unit_Price': 'UnitPrice',
        'Class_Text': 'ClassText',
    })
    for col in ['ProductCode', 'SalesModelCode', 'PlanModelCode']:
        io[col] = clean_code(io[col]).replace('', np.nan)

    before = len(io)
    io = io[~io['PartCode'].isin(list(exclude_partcodes))].copy()
    log.info("Dropped %d row(s) with PartCode in exclude_partcodes=%s.", before - len(io), exclude_partcodes)
    # SLH_Order='Y' rows are NOT dropped: the flag only marks an SLH order shipped (a sales-side
    # dedup), set after the fact - the order is still real demand. See README §13.

    # Every order book is a running total that assumes Qty >= 0 - a negative line (a return or
    # correction) would make it fall, pushing visibility ratios above 1 and KnownOrder below its past.
    negative = io['Qty'] < 0
    if negative.any():
        log.warning("Dropped %d order row(s) with a negative Qty (total %.0f).", int(negative.sum()),
            float(io.loc[negative, 'Qty'].sum()))
    io = io[~negative].copy()

    entry_date = pd.to_datetime(io['Order_Entry_Date Date'])
    request_date = pd.to_datetime(io['Customer_Request_Date Date'])
    # Before any month arithmetic or int32 cast, which would otherwise corrupt a missing date silently.
    valid = _valid_order_dates(entry_date, request_date)
    io, entry_date, request_date = io[valid].copy(), entry_date[valid], request_date[valid]

    # Screened on the raw date, before the Ship_To_Code bring-forward; that rule's own fallout is
    # dropped separately below, so its log names the cause.
    neg_lead_mask = date_to_month_idx(request_date) < date_to_month_idx(entry_date)
    log.info("Dropped %d row(s) whose Customer_Request_Date month is before their Order_Entry_Date "
        "month (negative lead time; same-month back-dated orders are kept), screened on the raw date "
        "before the Ship_To_Code bring-forward.", int(neg_lead_mask.sum()))
    io = io[~neg_lead_mask].copy()
    entry_date = entry_date[~neg_lead_mask]
    request_date = request_date[~neg_lead_mask]

    request_date = _bring_forward_ship_to_request_date(io, request_date, plant)
    # A brought-forward request can land in a month that had already closed when the order was
    # entered: it would rewrite a settled month's actual and leak into closed-month features.
    closed = date_to_month_idx(request_date) < date_to_month_idx(entry_date)
    if closed.any():
        log.warning("Dropped %d Ship_To_Code=%s row(s) (%.0f units) that the bring-forward moved into "
            "a request month already closed when they were entered.", int(closed.sum()),
            _SHIP_TO_BRING_FORWARD_CODE, float(io.loc[closed, 'Qty'].sum()))
    io, entry_date, request_date = io[~closed].copy(), entry_date[~closed], request_date[~closed]
    cancel_date = pd.to_datetime(io['Cancel_Date Date'])
    io['EntryMonthIdx'] = date_to_month_idx(entry_date).astype('int32')
    io['RequestMonthIdx'] = date_to_month_idx(request_date).astype('int32')
    io['Qty'] = io['Qty'].astype('float32')
    io['UnitPrice'] = io['UnitPrice'].astype('float32')
    io['Currency'] = io['Currency'].astype(str).str.strip().str.upper().replace(CURRENCY_ALIASES)
    io = _flag_known_cancellations(io, entry_date, request_date, cancel_date)
    # Day-granularity dates kept alongside the *MonthIdx columns above - the monthly panel/
    # training path only ever reads the MonthIdx versions, but collate_outstanding_order needs real
    # dates (its own cutoff is a specific day, not a month).
    io['EntryDate'] = entry_date
    io['RequestDate'] = request_date
    io['CancelDate'] = cancel_date
    io = io.drop(columns=['Order_Entry_Date Date', 'Customer_Request_Date Date', 'Cancel_Date Date',
        'Order_Status_Text', 'Ship_To_Code'])

    return io


def load_tfg_material() -> pd.DataFrame:
    """Loads Fabric's Warehouse_SAP_MMPP TFGMaterial table (ProductCode -> SalesModelCode/SubModelCode,
    dates, and ProductAbbreviation for display)."""
    tfg_material = pd.read_sql(
        "SELECT ProductCode, SalesModelCode, SubModelCode, ProductAbbreviation, ProductStartDate, ProductEndDate "
        "FROM [Warehouse_SAP_MMPP].[dbo].[TFGMaterial]",
        _fabric_engine,
    )
    for col in ['ProductCode', 'SalesModelCode', 'SubModelCode']:
        tfg_material[col] = clean_code(tfg_material[col]).replace('', np.nan)
    # A display name, so only trimmed and uppercased - not clean_code'd. TFG is almost all caps already.
    tfg_material['ProductAbbreviation'] = tfg_material['ProductAbbreviation'].str.strip().str.upper() \
        .replace('', np.nan)
    return tfg_material.dropna(subset=['ProductCode'])


def load_sales_result(plant: int, product_pmc_map: pd.Series) -> pd.DataFrame:
    """Monthly SalesResult per (PlanModelCode, TargetIdx) from HistoricalSales_LocalAndExport, SLH
    included. ProductCodes missing from `product_pmc_map` are dropped."""
    sales = pd.read_sql(text(
        "SELECT Period, ProductCode, SUM(ProductQuantity) AS SalesResult "
        "FROM Warehouse_AI.dbo.HistoricalSales_LocalAndExport "
        "WHERE Plant = :plant GROUP BY Period, ProductCode"
    ), _fabric_engine, params={'plant': plant})
    log.info("Loaded %d (Period, ProductCode) sales row(s) from Fabric: plant=%s", len(sales), plant)

    sales['PlanModelCode'] = clean_code(sales['ProductCode']).map(product_pmc_map)
    unmapped = sales['PlanModelCode'].isna()
    log.info("Dropped %d sales row(s) (qty %.0f) with no PlanModelCode for their ProductCode.",
        unmapped.sum(), sales.loc[unmapped, 'SalesResult'].sum())
    sales['TargetIdx'] = to_month_idx(sales['Period'].astype(int).to_numpy())
    return sales[~unmapped].groupby(['PlanModelCode', 'TargetIdx'])['SalesResult'].sum().reset_index()


_SOURCE_FILE_DATE_RE = re.compile(r'^(\d{8})\.xlsx$', re.IGNORECASE)
# "YYYYMMDD - YYYYMMDD.xlsx" covers through its end date, so it reads as a file dated end + 1.
_SOURCE_FILE_RANGE_RE = re.compile(r'^\d{8}\s*-\s*(\d{8})\.xlsx$', re.IGNORECASE)


def _source_file_name_date(filename):
    """The file date a YYYYMMDD.xlsx or YYYYMMDD - YYYYMMDD.xlsx name carries, or None - for any
    other name, and for eight digits that aren't a calendar date (20261341.xlsx)."""
    m = _SOURCE_FILE_DATE_RE.match(str(filename))
    is_range = m is None
    m = m or _SOURCE_FILE_RANGE_RE.match(str(filename))
    if not m:
        return None
    parsed = pd.to_datetime(m.group(1), format='%Y%m%d', errors='coerce')
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed) + pd.Timedelta(days=1) if is_range else pd.Timestamp(parsed)


def _pick_latest_source_file(files: pd.DataFrame, now=None) -> tuple:
    """(date, name) of the newest source file by the date in its name, ignoring names after
    tomorrow; (None, None) if none parse."""
    if files.empty:
        log.warning("No source files found - no settled month or cutoff date can be inferred.")
        return None, None
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    latest_allowed = now.normalize() + pd.Timedelta(days=1)     # one day of slack for time zones
    files = files.assign(
        _source_file=files['_source_file'].astype(str),
        FileDate=pd.to_datetime(files['FileDate'], errors='coerce'),
        NameDate=pd.to_datetime(pd.Series([_source_file_name_date(f) for f in files['_source_file']],
                                          index=files.index, dtype='object')))
    newest = files.sort_values('FileDate', na_position='first').iloc[-1]

    future = files['NameDate'] > latest_allowed
    if future.any():
        log.warning("%d source file(s) are named for a date after %s and are ignored: %s",
            int(future.sum()), latest_allowed.date(), sorted(files.loc[future, '_source_file']))
    usable = files[files['NameDate'].notna() & ~future]
    if usable.empty:
        log.warning("No source file follows the plain YYYYMMDD.xlsx convention (the newest by "
            "FileDate is %r) - no settled month or cutoff date can be inferred.",
            newest['_source_file'])
        return None, None
    best = usable.loc[usable['NameDate'].idxmax()]

    if pd.isna(newest['NameDate']):
        log.warning("The newest source file by FileDate, %r, doesn't follow the plain YYYYMMDD.xlsx "
            "convention - the settled month and cutoff date come from %r, the newest file that "
            "does. Rename it if it is a daily delivery.", newest['_source_file'], best['_source_file'])
    elif newest['NameDate'] < best['NameDate']:
        log.warning("Source files arrived out of order: %r has the newest FileDate but %r is named "
            "for a later day - using %r. Was an older file re-ingested?", newest['_source_file'],
            best['_source_file'], best['_source_file'])
    return pd.Timestamp(best['NameDate']), best['_source_file']


def latest_source_file_date(plant: int) -> tuple:
    """(date, filename) of `plant`'s newest source file, or (None, None) if no name parses."""
    files = pd.read_sql(text(
        "SELECT _source_file, MAX(FileDate) AS FileDate "
        "FROM Warehouse_AI.dbo.IncomingOrder_GoogleLooker "
        "WHERE Plant = :plant GROUP BY _source_file"
    ), _fabric_engine, params={'plant': plant})
    return _pick_latest_source_file(files)


def _settled_month_for_file_date(file_date) -> int:
    """Settled month = the month before the file's own month (None for no parseable file)."""
    if file_date is None:
        return None
    return int(to_month_idx(file_date.year * 100 + file_date.month)) - 1


def cutoff_date_from_file_date(file_date):
    """A file dated D only has complete data through D-1, so D-1 (the last completed date) is its
    cutoff date."""
    return None if file_date is None else pd.Timestamp(file_date) - pd.Timedelta(days=1)


def _infer_settled_month_from_source_files(plant: int) -> int:
    """Settled month = the month before the latest valid-format file's own month. Returns a
    month-idx, or None if latest_source_file_date did (falls through to the EntryMonthIdx rule)."""
    file_date, _ = latest_source_file_date(plant)
    return _settled_month_for_file_date(file_date)


def ctx_cutoff_grain(ctx: dict) -> str:
    """The grain a ctx was built at; raises if missing rather than assuming 'month'."""
    grain = ctx.get('cutoff_grain')
    if grain not in CUTOFF_GRAINS:
        raise ValueError(f"ctx['cutoff_grain'] is {grain!r}; expected one of {list(CUTOFF_GRAINS)} "
                         "- build the context with build_order_book_context(cutoff_grain=...).")
    return grain


def build_known_canceled_books(orders: pd.DataFrame, cutoff_grain: str) -> tuple:
    """(month book, day book or None) of cancelled quantity by cancel-known date; day book only at
    day grain."""
    cancelled = orders[orders['IsCanceled']]
    month_book = build_order_book(
        cancelled.assign(CancelMonthIdx=cancelled['CancelMonthIdx'].astype('int32')),
        ['PlanModelCode'], arrival_col='CancelMonthIdx')
    day_book = (build_order_book(cancelled, ['PlanModelCode'], arrival_col='CancelKnownDate')
                if cutoff_grain == 'day' else None)
    return month_book, day_book


def build_order_book_context(plant: int, exclude_partcodes=None, max_settled_target: int = None,
        extend_to_orders: bool = None, *, cutoff_grain: str) -> dict:
    """One plant's order books and lookups as a shared `ctx` dict; 'day' grain also builds the
    day-grain books."""
    if cutoff_grain not in CUTOFF_GRAINS:
        raise ValueError(f"cutoff_grain must be one of {list(CUTOFF_GRAINS)}, got {cutoff_grain!r}")
    if extend_to_orders is None:
        extend_to_orders = TRAIN_EXTEND_TO_ORDERS
    orders, tfg_material = resolve_plan_model_codes(load_io(plant, exclude_partcodes=exclude_partcodes),
        load_tfg_material(), submodel_overrides=PLANT_SUBMODEL_PLAN_MODEL_OVERRIDES[plant])
    file_date, _source_file = latest_source_file_date(plant)
    data_asof = cutoff_date_from_file_date(file_date) if file_date is not None \
        else pd.Timestamp(orders['EntryDate'].max())

    order_book_code = build_order_book(orders, ['PlanModelCode'])
    order_book_part = build_order_book(orders, ['PartCode'])
    order_book_code_daily = (build_order_book(orders, ['PlanModelCode'], arrival_col='EntryDate')
        if cutoff_grain == 'day' else None)
    actual_order = orders.groupby(['PlanModelCode', 'RequestMonthIdx'])['Qty'].sum() \
        .rename('ActualOrder').reset_index()
    actual_part = orders.groupby(['PartCode', 'RequestMonthIdx'])['Qty'].sum() \
        .rename('ActualOrder').reset_index()
    # Net counterpart to `actual_order` above, for ActualNetOrder (collate_actual_order.py) only -
    # `actual_order` itself stays the untouched gross total, since it feeds Adjustment (the
    # model's target).
    actual_order_net = orders[~orders['IsCanceled']].groupby(['PlanModelCode', 'RequestMonthIdx'])['Qty'].sum() \
        .rename('ActualNetOrder').reset_index()

    known_canceled_order_book_code, known_canceled_order_book_code_daily = \
        build_known_canceled_books(orders, cutoff_grain)
    n_smc = orders.groupby('PlanModelCode')['SalesModelCode'].nunique()
    if (n_smc > 1).any():
        log.warning("%d PlanModelCode(s) have >1 SalesModelCode on orders, earliest-entered used: %s",
            (n_smc > 1).sum(), sorted(n_smc[n_smc > 1].index))
    # Earliest-entered order, not load order (io.sql has no ORDER BY): later orders can't change it.
    code_map = orders.sort_values('EntryDate', kind='stable') \
        .groupby('PlanModelCode')[['PartCode', 'SalesModelCode']].first()
    # All-codes order volume by request month - grain-independent, read by the MarketShock family.
    market_shock_monthly = build_market_shock_monthly(orders)

    if max_settled_target is not None:
        max_settled_idx = int(to_month_idx(max_settled_target))
        if _month_bounds([max_settled_idx])[1][0] > data_asof:
            log.warning("max_settled_target=%s ends after the data's as-of date %s - its labels are "
                "incomplete, so training on it understates Adjustment.", max_settled_target,
                data_asof.date())
    else:
        max_settled_idx = _settled_month_for_file_date(file_date)
        if max_settled_idx is None:
            # A month only looks settled once the NEXT month's entries have shown up in the data -
            # one month before the latest EntryMonth present, unless the caller says otherwise.
            max_settled_idx = int(orders['EntryMonthIdx'].max()) - 1
    log.info("Settled boundary resolved: plant=%s max_settled_idx=%s (%s)",
        plant, max_settled_idx, int(month_idx_to_yyyymm(max_settled_idx)))
    dataset_floor_idx = compute_dataset_floor(order_book_code)
    # load_io already drops unusable dates (_valid_order_dates); this stops any other source of
    # orders - a stubbed or injected loader - before the floor feeds range()/arange() below.
    earliest_idx = int(date_to_month_idx(pd.Series([pd.Timestamp(MIN_VALID_ORDER_DATE)]))[0])
    if dataset_floor_idx < earliest_idx:
        raise ValueError(
            f"the earliest EntryMonthIdx in the orders is {dataset_floor_idx}, before "
            f"MIN_VALID_ORDER_DATE={MIN_VALID_ORDER_DATE} (month {earliest_idx}) - a missing or "
            "placeholder entry date reached the order book. Every skeleton is built from this "
            "floor, so refusing here rather than allocating decades of rows.")

    effective_ranges = resolve_effective_ranges_with_tfg(orders, tfg_material)
    effective_ranges = effective_ranges.join(code_map, on='PlanModelCode')
    product_ranges = resolve_product_code_ranges(orders, tfg_material)

    launch_by_code = effective_ranges.set_index('PlanModelCode')['StartMonthIdx']
    launch_by_part = effective_ranges.groupby('PartCode')['StartMonthIdx'].min()

    # np.fmax (NaN-ignoring), not np.maximum: launch_by_code/part is trusted as-is and can be
    # genuinely NaN (TFGMaterial has no ProductStartDate for that code) - the dataset floor is
    # always known, so an unknown launch date falls back to the floor itself rather than
    # propagating NaN into every skeleton/history-feature computation these floors feed.
    first_actual_order_by_code = np.fmax(launch_by_code, dataset_floor_idx)
    first_actual_order_by_part = np.fmax(launch_by_part, dataset_floor_idx)
    first_known_order_by_code = np.fmax(launch_by_code, dataset_floor_idx + 1)
    first_known_order_by_part = np.fmax(launch_by_part, dataset_floor_idx + 1)
    # first_known_order_by_code also picks which rows exist (training and inference), so it is
    # never moved; features read this copy instead.
    known_history_start_by_code = first_known_order_by_code
    if extend_to_orders:
        # History starts at the earlier of launch and first order entry - months in between are
        # real zeros, not "no data". launch_by_code itself is unchanged.
        start_by_code = np.fmin(launch_by_code,
            orders.groupby('PlanModelCode')['EntryMonthIdx'].min().reindex(launch_by_code.index))
        start_by_part = np.fmin(launch_by_part,
            orders.groupby('PartCode')['EntryMonthIdx'].min().reindex(launch_by_part.index))
        first_actual_order_by_code = np.fmax(start_by_code, dataset_floor_idx)
        first_actual_order_by_part = np.fmax(start_by_part, dataset_floor_idx)
        known_history_start_by_code = np.fmax(start_by_code, dataset_floor_idx + 1)
        first_known_order_by_part = np.fmax(start_by_part, dataset_floor_idx + 1)
    # Forecast eligibility only - launch_by_code above stays the model's own launch (it feeds
    # OriginMonthsSinceLaunch and the floors), so filling its gaps here would change model inputs.
    eligibility_launch_by_code = resolve_eligibility_launch(launch_by_code,
        orders.groupby('PlanModelCode')['EntryMonthIdx'].min(), dataset_floor_idx)

    # Settled demand span per code, by request month - always computed: collate_actual_order always
    # widens to it, training only under extend_to_orders.
    demand = actual_order[(actual_order['ActualOrder'] > 0) & (actual_order['RequestMonthIdx'] <= max_settled_idx)] \
        .groupby('PlanModelCode')['RequestMonthIdx'].agg(['min', 'max']).reindex(first_known_order_by_code.index)
    first_demand_by_code, last_demand_by_code = demand['min'], demand['max']
    # First training target per code, read by both _lifespan_skeleton and the inference lookback so
    # their past rows match.
    train_first_target_by_code = first_known_order_by_code
    if extend_to_orders:
        train_first_target_by_code = np.fmax(np.fmin(first_known_order_by_code, first_demand_by_code),
            dataset_floor_idx + 1)

    pred_succ_flags = get_predecessor_successor_flags_from_db(effective_ranges, orders) if PLANT_INCLUDES_PRED_SUCC[plant] else None

    # Country-mix visibility inputs. Origins span the full panel range - _lifespan_skeleton keeps
    # OriginIdx >= dataset_floor_idx, and live inference scores at max_settled_idx - so every row
    # either builder will be asked for is covered. Built once here, like the market-shock
    # monthly, rather than per attach call.
    include_country = PLANT_INCLUDES_COUNTRY_VISIBILITY[plant]
    panel_origins = range(int(dataset_floor_idx), int(max_settled_idx) + 1)
    if include_country:
        country_visibility = build_country_visibility(orders, panel_origins)
        sku_country_mix = build_sku_country_mix(orders, panel_origins,
            window=COUNTRY_MIX_WINDOW, min_share=COUNTRY_MIX_MIN_SHARE, group_col='PlanModelCode')
        focus_country_order_book = build_focus_country_order_book(orders, FOCUS_COUNTRY,
            group_col='PlanModelCode')
        # country_visibility/sku_country_mix stay monthly under BOTH grains: the fill curve reads
        # already-closed prior months, and the trailing mix is a settled-history composition -
        # neither moves intra-month. Only the focus book is queried at a future target.
        focus_country_order_book_daily = (build_focus_country_order_book(orders, FOCUS_COUNTRY,
            group_col='PlanModelCode', arrival_col='EntryDate') if cutoff_grain == 'day' else None)
    else:
        country_visibility = sku_country_mix = focus_country_order_book = None
        focus_country_order_book_daily = None

    return {
        'orders': orders,
        'event_ranges': PLANT_EXTERNAL_EVENT_RANGES[plant],
        'event_end_known': PLANT_EXTERNAL_EVENT_END_KNOWN[plant],
        'include_cny': PLANT_INCLUDES_CNY[plant],
        # The as-of grain everything built from this ctx inherits - _attach_all_features routes
        # off this single value rather than each family deciding for itself.
        'cutoff_grain': cutoff_grain,
        'order_book_code': order_book_code,
        'order_book_code_daily': order_book_code_daily,   # None under cutoff_grain='month'
        'order_book_part': order_book_part,
        'actual_order': actual_order,
        'actual_order_net': actual_order_net,
        'known_canceled_order_book_code': known_canceled_order_book_code,
        'known_canceled_order_book_code_daily': known_canceled_order_book_code_daily,  # None: month
        'data_asof': data_asof,
        'actual_part': actual_part,
        'launch_by_code': launch_by_code,
        'first_actual_order_by_code': first_actual_order_by_code,
        'first_actual_order_by_part': first_actual_order_by_part,
        'first_known_order_by_code': first_known_order_by_code,
        'first_known_order_by_part': first_known_order_by_part,
        'known_history_start_by_code': known_history_start_by_code,
        'train_first_target_by_code': train_first_target_by_code,
        'first_demand_by_code': first_demand_by_code,
        'last_demand_by_code': last_demand_by_code,
        'extend_to_orders': extend_to_orders,
        'code_map': code_map,
        'eligibility_launch_by_code': eligibility_launch_by_code,
        'pred_succ_flags': pred_succ_flags,
        'effective_ranges': effective_ranges,
        'dataset_floor_idx': int(dataset_floor_idx),
        'max_settled_idx': max_settled_idx,
        'product_ranges': product_ranges,
        'market_shock_monthly': market_shock_monthly,
        'include_country_visibility': include_country,
        'country_visibility': country_visibility,
        'sku_country_mix': sku_country_mix,
        'focus_country_order_book': focus_country_order_book,
        'focus_country_order_book_daily': focus_country_order_book_daily,  # None under 'month'
    }


def _day_grain_routing(ctx: dict) -> dict:
    """Book and as-of column for the day-sensitive families (KnownOrder, focus share, cancellations)."""
    if ctx_cutoff_grain(ctx) == 'month':
        return {'asof_col': None, 'arrival_col': 'EntryMonthIdx',
                'known_order_book': ctx['order_book_code'],
                'focus_order_book': ctx.get('focus_country_order_book'),
                'cancel_book': ctx.get('known_canceled_order_book_code'),
                'cancel_arrival_col': 'CancelMonthIdx'}
    if ctx.get('order_book_code_daily') is None:
        raise ValueError(
            "ctx['cutoff_grain'] is 'day' but ctx['order_book_code_daily'] is None - rebuild "
            "the context with build_order_book_context(..., cutoff_grain='day').")
    return {'asof_col': CUTOFF_DATE_COL, 'arrival_col': 'EntryDate',
            'known_order_book': ctx['order_book_code_daily'],
            'focus_order_book': ctx.get('focus_country_order_book_daily'),
            'cancel_book': ctx.get('known_canceled_order_book_code_daily'),
            'cancel_arrival_col': 'CancelKnownDate'}


def _attach_all_features(skeleton: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    """The attach_* sequence shared by training and inference, routed by `ctx['cutoff_grain']`."""
    routing = _day_grain_routing(ctx)
    panel = attach_calendar_features(skeleton, 'TargetIdx', include_cny=ctx['include_cny'])
    panel = attach_calendar_features(panel, 'OriginIdx', prefix='Origin_', include_cny=False)

    # MarketShock: re-enabled 2026-09 and part of the deployed feature set. Attaching a family only
    # puts the COLUMNS on the panel - whether a run uses them is decided by feature_cols (see
    # evaluation.build_add_candidates), so one panel build serves every experiment variant too.
    panel = attach_market_shock_features(panel, ctx['market_shock_monthly'])

    panel = attach_history_features(panel, ctx['order_book_code'], ctx['first_actual_order_by_code'],
        group_col='PlanModelCode', prefix='')
    panel = attach_lag_event_features(panel, ctx['event_ranges'], group_col='PlanModelCode', prefix='')
    panel = attach_trend_stability_features(panel, group_col='PlanModelCode', prefix='')
    panel = attach_adi_features(panel, ctx['actual_order'], ctx['first_actual_order_by_code'],
        group_col='PlanModelCode', prefix='')
    panel = attach_known_order_features(panel, routing['known_order_book'],
        ctx['known_history_start_by_code'], group_col='PlanModelCode',
        asof_col=routing['asof_col'], arrival_col=routing['arrival_col'])
    panel = attach_historical_visibility(panel, ctx['order_book_code'], ctx['known_history_start_by_code'],
        group_col='PlanModelCode', prefix='')
    panel = attach_historical_visibility(panel, ctx['order_book_part'], ctx['first_known_order_by_part'],
        group_col='PartCode', prefix='PartLevel_')
    panel = attach_history_features(panel, ctx['order_book_part'], ctx['first_actual_order_by_part'],
        group_col='PartCode', prefix='Group_')
    panel = attach_lag_event_features(panel, ctx['event_ranges'], group_col='PartCode', prefix='Group_')
    panel = attach_trend_stability_features(panel, group_col='PartCode', prefix='Group_')
    panel = attach_adi_features(panel, ctx['actual_part'], ctx['first_actual_order_by_part'],
        group_col='PartCode', prefix='Group_')
    # Needs both the entity-level AND Group_-level attach_history_features calls above
    # already merged (RecentDemandShare3/6/12 compare own Lag0..k against Group_Lag0..k).
    panel = attach_lifecycle_features(panel, ctx['actual_order'],
        ctx['launch_by_code'], ctx['first_actual_order_by_code'], group_col='PlanModelCode', prefix='',
        group_share_prefix='Group_', pred_succ_flags=ctx.get('pred_succ_flags'),
        end_by_group=ctx['effective_ranges'].set_index('PlanModelCode')['EndMonthIdx'])
    # Not a feature (no family lists it): the age meets_min_launch_age tests, on every training
    # and inference row alike, so backtests and production filter on one value.
    panel = attach_forecast_eligibility_age(panel, ctx['eligibility_launch_by_code'],
        group_col='PlanModelCode')

    # Needs KnownOrder (attach_known_order_features, above) for both the focus-share denominator
    # and the implied-adjustment level - see its own docstring.
    panel = attach_country_visibility_features(panel, ctx['country_visibility'],
        ctx['sku_country_mix'], routing['focus_order_book'], focus_country=FOCUS_COUNTRY,
        group_col='PlanModelCode', include=ctx['include_country_visibility'],
        asof_col=routing['asof_col'], arrival_col=routing['arrival_col'])

    # Last, because it needs KnownOrder (above) and is the only family whose columns exist ONLY at
    # day grain - on a month panel it returns `panel` untouched, so no branch is needed here.
    panel = attach_intramonth_features(panel,
        order_book_code_daily=ctx.get('order_book_code_daily'),
        order_book_code=ctx['order_book_code'],
        first_entry_by_code=ctx['known_history_start_by_code'],
        group_col='PlanModelCode', asof_col=routing['asof_col'] or CUTOFF_DATE_COL,
        include_same_day_visibility=ctx.get('include_same_day_visibility', True))

    panel = attach_external_factor_features(panel, ctx['event_ranges'], ctx.get('event_end_known', ()))
    return panel


def _feature_families() -> dict:
    """FEATURE_FAMILIES, notebook cell 37 - PlanModelCode grain (see _lifespan_skeleton)."""
    families = {
        'Calendar': ['Year', 'Month', 'Month_sin', 'Month_cos', 'Quarter', 'Is_CNY_Month', 'Months_to_CNY',
            'Origin_Year', 'Origin_Month', 'Origin_Month_sin', 'Origin_Month_cos', 'Origin_Quarter'],
        # TrailMean3/6/12, ZScore3/6, Lag0Lag1Ratio/Diff not in the reference panel - added after
        # it was built, not needed to reproduce/compare against it. ZScore12 is the reference panel's
        # 'ZScore' under its new name (same formula, see features.py) - kept active.
        'LegacyHistory': [f'Lag{k}' for k in range(0, 24)] + [
            # 'TrailMean3', 'TrailMean6', 'TrailMean12',
            'Recent3_vs_Prior6', 'TrailMean3_vs_base', 'TrailMean12_vs_base',
            'TrendSlope12_vs_base', 'Volatility6', 'Range12_vs_base',
            'ZScore12',
            # 'ZScore3', 'ZScore6', 'Lag0Lag1Ratio', 'Lag0Lag1Diff',
            'Rolling3ADI', 'Rolling6ADI', 'Rolling12ADI', 'Rolling24ADI', 'ExpandingADI',
            'LagEventCount6', 'LagEventCount12', 'StableRunLength',
        ],
        'KnownOrder': [
            'KnownOrder', 'KnownOrder_prevOrigin', 'Accumulation_1mo', 'AccumulationPct_1mo',
            'KnownOrder_3moAgo', 'Momentum_3mo', 'Accumulation_3mo',
            'KnownOrder_6moAgo', 'Momentum_6mo', 'Accumulation_6mo',
            'KnownOrder_YoY_prior', 'KnownOrder_YoY', 'KnownOrder_x_Horizon',
        ],
        'HistoricalVisibility': [
            'HistVisibility_AllHistory', 'HistVisibility_SameMonth',
            'PartLevel_HistVisibility_AllHistory', 'PartLevel_HistVisibility_SameMonth',
        ],
        'GroupLevel': [f'Group_Lag{k}' for k in range(0, 24)] + [
            # 'Group_TrailMean3', 'Group_TrailMean6', 'Group_TrailMean12',
            'Group_Recent3_vs_Prior6', 'Group_TrailMean3_vs_base', 'Group_TrailMean12_vs_base',
            'Group_TrendSlope12_vs_base', 'Group_Volatility6', 'Group_Range12_vs_base',
            'Group_ZScore12',
            # 'Group_ZScore3', 'Group_ZScore6', 'Group_Lag0Lag1Ratio', 'Group_Lag0Lag1Diff',
            'Group_Rolling3ADI', 'Group_Rolling6ADI', 'Group_Rolling12ADI', 'Group_Rolling24ADI',
            'Group_ExpandingADI', 'Group_LagEventCount6', 'Group_LagEventCount12', 'Group_StableRunLength',
        ],
        # MarketShock, CountryVisibility and IntraMonth were added 2026-09, validated, and are part
        # of the deployed feature set. NEW_FEATURE_FAMILIES below names them, so an experiment can
        # still rebuild the feature set from before they were added.
        'MarketShock': ['MarketPctChange', 'MarketZScore_3m'],
        'CountryVisibility': country_visibility_feature_cols(FOCUS_COUNTRY),
        # Day-grain only - these columns are ABSENT from a month panel, not NaN (see
        # intramonth_features). build_full_panel keeps only the columns a panel has, so a
        # month-grain deployment trains without them and a day-grain one with them.
        'IntraMonth': INTRAMONTH_FEATURE_COLS,
        'ExternalFactor': ['HasExternalEvent'],
        'MovingAverageBaselines': [
            'AdjTrailMean3', 'AdjTrailMean6', 'AdjTrailMean12',
            'ManualSim3', 'ManualSim6', 'ManualSim12',
        ],
        'Lifecycle': [
            'OriginMonthsSinceLaunch', 'TargetMonthsSinceLaunch',
            'MonthsSinceLastDemand', 'ConsecutiveMonthsWithDemand',
            # Pending pdt_lifecycle_eda.ipynb EDA (train_start vs effective_start, and vs
            # ConsecutiveMonthsWithDemand) - remove these two if that EDA doesn't support
            # keeping them as distinct from what's already above.
            # 'MonthsSinceFirstDemand', 'MonthsFromLaunchToFirstDemand',
            'HasPredecessor', 'HasSuccessor',
            'RecentDemandShare3', 'RecentDemandShare6', 'RecentDemandShare12',
            # GrowthRateDiff not in the reference panel - added after it was built, not needed to
            # reproduce/compare against it.
            # 'GrowthRateDiff',
        ],
    }
    return families


def _expand_to_cutoff_days(skeleton: pd.DataFrame) -> pd.DataFrame:
    """Fans each row out across its origin's cutoff_window dates; OriginIdx is unchanged."""
    origin_dtype = skeleton['OriginIdx'].dtype
    windows = [pd.DataFrame({'OriginIdx': np.full(len(w), o, dtype=origin_dtype),
                             CUTOFF_DATE_COL: w, '_pos': np.arange(len(w))})
               for o in skeleton['OriginIdx'].unique() for w in [cutoff_window(o)]]
    out = skeleton.merge(pd.concat(windows, ignore_index=True), on='OriginIdx', how='left')
    # day-major order (every row's day 0, then every row's day 1, ...)
    out = out.sort_values('_pos', kind='stable').drop(columns='_pos').reset_index(drop=True)
    log.info("Expanded skeleton across each origin's cutoff window: %d -> %d row(s).",
        len(skeleton), len(out))
    return out


def _lifespan_skeleton(ctx: dict) -> pd.DataFrame:
    """One row per (PlanModelCode, TargetIdx, Horizon) over each code's lifespan, OriginIdx >=
    dataset floor; pre-launch origins are intentional."""
    end_idx_by_code = ctx['effective_ranges'].set_index('PlanModelCode')['EndMonthIdx']
    end_idx_by_code = np.minimum(end_idx_by_code.fillna(ctx['max_settled_idx']), ctx['max_settled_idx'])
    ranges = pd.DataFrame({'first_target': ctx['train_first_target_by_code'], 'EndMonthIdx': end_idx_by_code})
    if ctx['extend_to_orders']:
        # Wrong TFG dates (e.g. SPL's empty TFG rows, launch 2025-11) still pull in long zero stretches.
        ranges['EndMonthIdx'] = np.fmax(ranges['EndMonthIdx'], ctx['last_demand_by_code'])

    frames = []
    for code, r in ranges.iterrows():
        targets = np.arange(int(r['first_target']), int(r['EndMonthIdx']) + 1)
        frames.append(pd.DataFrame({'PlanModelCode': code, 'TargetIdx': targets}))
    if not sum(len(f) for f in frames):
        # Said plainly, rather than as pd.concat's "No objects to concatenate".
        raise ValueError(
            f"no PlanModelCode has a target month to train on ({len(ranges)} code(s) with an "
            "effective range, none with a settled target in it). There is no training panel to build.")
    skeleton = pd.concat(frames, ignore_index=True)
    skeleton = skeleton.merge(pd.DataFrame({'Horizon': range(1, 7)}), how='cross')
    skeleton['OriginIdx'] = skeleton['TargetIdx'] - skeleton['Horizon']
    before = len(skeleton)
    skeleton = skeleton[skeleton['OriginIdx'] >= ctx['dataset_floor_idx']].reset_index(drop=True)
    log.info("Dropped %d training row(s) with OriginIdx before the dataset floor.", before - len(skeleton))
    skeleton = skeleton.join(ctx['code_map'], on='PlanModelCode')
    if ctx_cutoff_grain(ctx) == 'day':
        skeleton = _expand_to_cutoff_days(skeleton)
    return skeleton


# The families added in 2026-09. They are validated and deployed: full_feature_cols() includes
# them, and so does every training panel. Kept as a named constant only so an experiment can
# rebuild the earlier feature set (baseline_feature_cols) without hardcoding column names. With
# IntraMonth excluded, a day panel's feature set is exactly the one from before daily refresh.
NEW_FEATURE_FAMILIES = ('MarketShock', 'CountryVisibility', 'IntraMonth')


def full_feature_cols() -> list:
    """The complete feature_cols allowlist without needing data loaded, so callers (e.g.
    model_registry) can resolve 'all' without rebuilding the panel."""
    families = _feature_families()
    return ['PlanModelCode', 'SalesModelCode', 'PartCode'] + [c for fam in families.values() for c in fam]


def baseline_feature_cols(exclude_families=NEW_FEATURE_FAMILIES) -> list:
    """full_feature_cols() minus `exclude_families` (default: the 2026-09 additions), for comparisons."""
    families = _feature_families()
    return ['PlanModelCode', 'SalesModelCode', 'PartCode'] + [
        c for name, fam in families.items() if name not in exclude_families for c in fam]


def build_full_panel(ctx: dict):
    """Full training panel: skeleton + features + ActualOrder/Adjustment labels + categorical
    casts + FEATURE_FAMILIES/feature_cols. Returns (panel, FEATURE_FAMILIES, feature_cols)."""
    skeleton = _lifespan_skeleton(ctx)
    panel = _attach_all_features(skeleton, ctx)
    # Not a feature (feature_cols is a family allowlist): the backtest's net floor, read at the same
    # as-of point as KnownOrder, exactly as build_inference_skeleton attaches it for production.
    routing = _day_grain_routing(ctx)
    panel['KnownCanceledOrder'] = known_canceled_order(panel, routing['cancel_book'],
        group_col='PlanModelCode', asof_col=routing['asof_col'],
        arrival_col=routing['cancel_arrival_col'])

    panel = panel.merge(
        ctx['actual_order'].rename(columns={'RequestMonthIdx': 'TargetIdx'}),
        on=['PlanModelCode', 'TargetIdx'], how='left',
    )
    panel['ActualOrder'] = panel['ActualOrder'].fillna(0.0)
    panel['Adjustment'] = panel['ActualOrder'] - panel['KnownOrder']

    # label-dependent features - must run after Adjustment exists, unlike everything in
    # _attach_all_features above. Training panel already has every historical row needed to
    # self-join against (history_source=None, the default), unlike live inference - see
    # build_adjustment_history_lookback/build_inference_skeleton.
    #
    # On a day-grain panel these MONTH-grain baselines must not self-join against the whole panel:
    # their lookup is indexed on (code, OriginIdx), which a fanned-out panel repeats once per cutoff
    # day. The day-0 monthly anchor is the right history - one settled Adjustment per (code, origin,
    # horizon), the value the monthly pipeline would have used.
    history = panel[monthly_anchor_mask(panel)] \
        if ctx_cutoff_grain(ctx) == 'day' else None
    panel = attach_adjustment_trailing_features(panel, group_col='PlanModelCode',
        history_source=history)
    panel = attach_manual_sim_features(panel, group_col='PlanModelCode',
        history_source=history)

    panel['PlanModelCode'] = panel['PlanModelCode'].astype('category')
    # now a real feature (full_feature_cols), not just an internal join key for Group_*/
    # PartLevel_* - must be category dtype, same requirement as PlanModelCode, or
    # XGBoost's enable_categorical=True raises on the raw object dtype.
    panel['PartCode'] = panel['PartCode'].astype('category')
    panel['SalesModelCode'] = panel['SalesModelCode'].astype('category')

    feature_families = _feature_families()
    # Not every column in full_feature_cols() is guaranteed to exist - e.g. Is_CNY_Month/
    # Months_to_CNY only get attached when ctx['include_cny'] is True (see
    # attach_calendar_features), which differs per plant (PLANT_INCLUDES_CNY).
    feature_cols = [c for c in full_feature_cols() if c in panel.columns]

    return panel, feature_families, feature_cols


def build_adjustment_history_lookback(ctx: dict, origin_idx: int, horizons,
        windows: tuple = (3, 6, 12), codes=None) -> pd.DataFrame:
    """Settled history rows the live skeleton's trailing-Adjustment and ManualSim features join against."""
    max_h, max_w = max(horizons), max(windows)
    # Same first target as the training panel, so trailing Adjustment features see the same past rows.
    first_known_order = ctx['train_first_target_by_code']
    order_book = ctx['order_book_code']
    actual = ctx['actual_order']

    # codes=None default: pre-filter to codes whose first_known_order could be reached by SOME
    # (origin, horizon) combo here (max reachable TargetIdx = origin_idx + max_h) - not gated on
    # first_known_order <= origin_idx, which would wrongly exclude a code launching mid-window.
    # The per-row TargetIdx filter below does the exact scoping; this is just a cheap pre-filter.
    active_codes = list(codes) if codes is not None \
        else first_known_order[first_known_order <= origin_idx + max_h].index.tolist()
    lookback_origins = range(origin_idx - max_h - max_w + 1, origin_idx + 1)

    skeleton = pd.DataFrame({'PlanModelCode': active_codes})
    skeleton = skeleton.merge(pd.DataFrame({'OriginIdx': lookback_origins}), how='cross')
    skeleton = skeleton.merge(pd.DataFrame({'Horizon': range(1, max_h + 1)}), how='cross')
    skeleton['TargetIdx'] = skeleton['OriginIdx'] + skeleton['Horizon']
    # same lifespan scoping _lifespan_skeleton applies for training (Target >= first_known_order,
    # Origin >= dataset_floor_idx) - without it, a young entity gets rows for targets before it
    # existed, and attach_known_order_features's "Target is always after the code's own start ->
    # raw NaN is a genuine zero" convention (see its own docstring) silently breaks - confirmed
    # empirically to cause exactly this training-vs-inference mismatch for entities within
    # ~lookback-window months of their own start.
    skeleton = skeleton[(skeleton['TargetIdx'] >= skeleton['PlanModelCode'].map(first_known_order))
        & (skeleton['OriginIdx'] >= ctx['dataset_floor_idx'])]

    history = attach_known_order_features(skeleton, order_book, ctx['known_history_start_by_code'],
        group_col='PlanModelCode')
    history = history.merge(actual.rename(columns={'RequestMonthIdx': 'TargetIdx'}),
        on=['PlanModelCode', 'TargetIdx'], how='left')

    # Accumulation_1mo is cutoff-based (OriginIdx <= origin_idx, enforced by lookback_origins
    # above) so it's safe even for a row whose own TargetIdx is still in the future relative to
    # origin_idx 3- same "advance visibility into a future month" pattern KnownOrder/Momentum_k
    # already rely on everywhere. Adjustment is NOT safe there: it's derived from ActualOrder,
    # which must never reflect a month that hasn't happened yet as of `origin_idx`. Neither
    # attach_adjustment_trailing_features nor attach_manual_sim_features currently reads
    # Adjustment for a future-target row (verified: both only reference Adjustment through
    # offsets that keep TargetIdx <= origin_idx), but explicitly nulling it out here removes the
    # possibility entirely, rather than relying on "nothing calls it that way today" - a real
    # risk if `ctx` is ever built from a dataset that extends past `origin_idx` (e.g. backtesting
    # against a full historical file), where ActualOrder for a future month would otherwise sit
    # in this dataframe as genuine (leaky) data instead of a harmless absence.
    is_settled = (skeleton['TargetIdx'] <= origin_idx).to_numpy()
    history['ActualOrder'] = np.where(is_settled, history['ActualOrder'].fillna(0.0), np.nan)
    history['Adjustment'] = history['ActualOrder'] - history['KnownOrder']
    return history


def build_inference_skeleton(ctx: dict, origin_idx: int, horizons, codes=None,
        min_origin_months_since_launch: int = 3, cutoff_date=None) -> pd.DataFrame:
    """Eligible feature rows (with KnownCanceledOrder) for every PlanModelCode x horizon at one Origin.
    Refuses an as-of point after ctx['data_asof']."""
    grain = ctx_cutoff_grain(ctx)
    if ctx.get('data_asof') is None:
        raise ValueError("ctx has no 'data_asof' - can't verify the as-of point isn't after the "
                         "data. Build it with build_order_book_context.")
    first_known_order = ctx['first_known_order_by_code']
    max_horizon = max(horizons)
    # Cheap pre-filter: a code unreachable by any requested horizon (origin_idx + max_horizon)
    # can't produce a valid row below anyway, so skip it before the cross join.
    candidate_codes = list(codes) if codes is not None \
        else first_known_order[first_known_order <= origin_idx + max_horizon].index.tolist()

    end_idx_by_code = ctx['effective_ranges'].set_index('PlanModelCode')['EndMonthIdx']

    skeleton = pd.DataFrame({'PlanModelCode': candidate_codes})
    skeleton = skeleton.merge(pd.DataFrame({'Horizon': list(horizons)}), how='cross')
    skeleton['OriginIdx'] = origin_idx
    skeleton['TargetIdx'] = skeleton['OriginIdx'] + skeleton['Horizon']
    # Per-(code, horizon) scoping, matching _lifespan_skeleton's own TargetIdx range: launch floor
    # below, confirmed-discontinuation ceiling above (end_idx is genuinely NaN when unconfirmed,
    # so no ceiling applies there - only a real TFG end date excludes a row).
    end_idx = skeleton['PlanModelCode'].map(end_idx_by_code)
    before = len(skeleton)
    skeleton = skeleton[(skeleton['TargetIdx'] >= skeleton['PlanModelCode'].map(first_known_order))
        & (end_idx.isna() | (skeleton['TargetIdx'] <= end_idx))].reset_index(drop=True)
    log.info("Dropped %d (code, horizon) row(s) outside their own launch/end-date window.",
        before - len(skeleton))
    skeleton = skeleton.join(ctx['code_map'], on='PlanModelCode')

    if grain == 'month' and cutoff_date is not None:
        raise ValueError(
            f"cutoff_date={cutoff_date} was given for a month-grain context - its as-of point is "
            "the origin month's end, so a date would be ignored. Build a 'day' context instead.")
    if grain == 'day':
        if cutoff_date is None:
            raise ValueError(
                "ctx['cutoff_grain'] is 'day' but no cutoff_date was given. Pass the date the "
                "data is actually current as of - cutoff_date_from_file_date(latest file date) - not "
                "today's date: a stale source file would otherwise claim knowledge the order "
                "book doesn't have.")
        cutoff_date = pd.Timestamp(cutoff_date)
        window = cutoff_window(origin_idx)
        if cutoff_date not in window:
            raise ValueError(
                f"cutoff_date {cutoff_date.date()} is outside the valid as-of window for "
                f"origin_idx={origin_idx} ({window[0].date()} .. {window[-1].date()}). Earlier "
                "means that origin month hasn't settled; later means another month has closed "
                "and origin_idx should advance.")
        skeleton[CUTOFF_DATE_COL] = cutoff_date
    asof = cutoff_date if grain == 'day' \
        else pd.Timestamp(cutoff_dates_for_origins(np.array([origin_idx]), 0)[0])
    if asof > pd.Timestamp(ctx['data_asof']):
        raise ValueError(
            f"as-of point {asof.date()} ({grain} grain, origin_idx={origin_idx}) is after the data's "
            f"own as-of date {pd.Timestamp(ctx['data_asof']).date()} - nothing entered or cancelled "
            "after that is in the book, so these rows would claim knowledge nobody had yet.")

    panel = _attach_all_features(skeleton, ctx)
    routing = _day_grain_routing(ctx)
    panel['KnownCanceledOrder'] = known_canceled_order(panel, routing['cancel_book'],
        group_col='PlanModelCode', asof_col=routing['asof_col'],
        arrival_col=routing['cancel_arrival_col'])

    # label-dependent features can't self-join against `panel` (it only has rows for this one
    # live OriginIdx) - reconstruct just enough settled history to join against instead. Uses the
    # codes actually surviving the filter above (not candidate_codes), since a candidate that ended
    # up with zero rows needs no history reconstructed for it.
    active_codes = skeleton['PlanModelCode'].unique().tolist()
    history = build_adjustment_history_lookback(ctx, origin_idx, horizons, codes=active_codes)
    panel = attach_adjustment_trailing_features(panel, group_col='PlanModelCode', history_source=history)
    panel = attach_manual_sim_features(panel, group_col='PlanModelCode', history_source=history)

    if min_origin_months_since_launch is not None:
        before = len(panel)
        panel = panel[scoreable_rows(panel, min_origin_months_since_launch)].reset_index(drop=True)
        log.info("Dropped %d row(s) from prediction - outside lifecycle, or below %d month(s) since "
            "launch (TFGMaterial launch, else first order).", before - len(panel),
            min_origin_months_since_launch)

    panel['PlanModelCode'] = panel['PlanModelCode'].astype('category')
    # now a real feature (full_feature_cols), not just an internal join key for Group_*/
    # PartLevel_* - must be category dtype, same requirement as PlanModelCode, or
    # XGBoost's enable_categorical=True raises on the raw object dtype.
    panel['PartCode'] = panel['PartCode'].astype('category')
    panel['SalesModelCode'] = panel['SalesModelCode'].astype('category')
    return panel


def _check_ctx_grain(ctx: dict, cutoff_grain: str) -> None:
    """A pre-built ctx must be at the grain the caller asked for - never silently the other one."""
    if ctx_cutoff_grain(ctx) != cutoff_grain:
        raise ValueError(f"pre-built ctx is at {ctx_cutoff_grain(ctx)!r} grain but {cutoff_grain!r} "
                         "was requested - rebuild it with build_order_book_context(cutoff_grain=...).")


def preprocess_for_training(plant: int, exclude_partcodes=None, max_settled_target: int = None,
        ctx: dict = None, *, cutoff_grain: str):
    """Fabric -> training panel for one plant. Returns (panel, FEATURE_FAMILIES, feature_cols, ctx)."""
    if ctx is None:
        ctx = build_order_book_context(plant, exclude_partcodes=exclude_partcodes,
            max_settled_target=max_settled_target, cutoff_grain=cutoff_grain)
    _check_ctx_grain(ctx, cutoff_grain)
    panel, feature_families, feature_cols = build_full_panel(ctx)
    return panel, feature_families, feature_cols, ctx


def preprocess_for_inference(plant: int, origin_yyyymm: int = None, horizons=range(1, 7), codes=None,
        exclude_partcodes=None, min_origin_months_since_launch: int = 3, ctx: dict = None, *,
        cutoff_grain: str, cutoff_date=None):
    """Fabric -> inference skeleton for one plant/Origin. Returns (skeleton, ctx); at day grain
    cutoff_date=None is the day before the latest file."""
    if ctx is None:
        ctx = build_order_book_context(plant, exclude_partcodes=exclude_partcodes,
            max_settled_target=origin_yyyymm, cutoff_grain=cutoff_grain)
    _check_ctx_grain(ctx, cutoff_grain)
    origin_idx = ctx['max_settled_idx']
    if cutoff_grain == 'day' and cutoff_date is None:
        file_date, source_file = latest_source_file_date(plant)
        cutoff_date = cutoff_date_from_file_date(file_date)
        if cutoff_date is None:
            raise ValueError(
                f"cutoff_grain='day' needs an as-of date, but none of plant {plant}'s source files "
                "follows the plain YYYYMMDD.xlsx convention, so one can't be inferred. "
                "Pass cutoff_date explicitly.")
        log.info("Day-grain inference as of %s (from source file %s).",
            cutoff_date.date(), source_file)
    skeleton = build_inference_skeleton(ctx, origin_idx, horizons, codes=codes,
        min_origin_months_since_launch=min_origin_months_since_launch, cutoff_date=cutoff_date)
    return skeleton, ctx


def preprocess_for_inference_range(plant: int, start_origin_yyyymm: int, end_origin_yyyymm: int,
        horizons, codes=None, exclude_partcodes=None,
        min_origin_months_since_launch: int = 3, *, cutoff_grain: str,
        cutoff_days=None) -> pd.DataFrame:
    """preprocess_for_inference over an origin range, sharing one ctx. `cutoff_days` is required at
    day grain and must be omitted at month grain."""
    if cutoff_grain not in CUTOFF_GRAINS:
        raise ValueError(f"cutoff_grain must be one of {list(CUTOFF_GRAINS)}, got {cutoff_grain!r}")
    if cutoff_grain == 'month' and cutoff_days is not None:
        raise ValueError("cutoff_days applies to day grain only - a month-grain row's as-of point "
                         "is its origin month's end.")
    if cutoff_grain == 'day':
        days = sorted({int(d) for d in (cutoff_days or [])})
        if not days:
            raise ValueError("day-grain range scoring needs cutoff_days, e.g. [0] for the monthly "
                             "anchor or [0, 15, 30] for a fill curve.")
        bad = [d for d in days if not 0 <= d <= 30]
        if bad:
            raise ValueError(f"cutoff_days {bad} must be 0..30")
    ctx = build_order_book_context(plant, exclude_partcodes=exclude_partcodes, cutoff_grain=cutoff_grain)

    start_idx = int(to_month_idx(start_origin_yyyymm))
    end_idx = int(to_month_idx(end_origin_yyyymm))
    if end_idx < start_idx:
        raise ValueError(f"end_origin_yyyymm ({end_origin_yyyymm}) is before "
            f"start_origin_yyyymm ({start_origin_yyyymm})")

    frames = []
    for origin_idx in range(start_idx, end_idx + 1):
        if cutoff_grain == 'month':
            dates = [None]
        else:
            window = cutoff_window(origin_idx)   # window[d] is cutoff day d
            dates = [window[d] for d in days if d < len(window)]
        frames += [build_inference_skeleton(ctx, origin_idx, horizons, codes=codes,
                       min_origin_months_since_launch=min_origin_months_since_launch,
                       cutoff_date=date)
                   for date in dates]
    return pd.concat(frames, ignore_index=True)


# 'YYYYMM-Proj' plans are issued the month before YYYYMM, so OriginIdx = YYYYMM - 2.
SIC_PLAN_ORIGIN_OFFSET = 2


def load_sic_demand_plan(plant: int) -> pd.DataFrame:
    """Warehouse_AI.dbo.SIC_DemandPlan -> one row per (PlanModelCode, OriginIdx, TargetIdx)."""
    with _fabric_engine.connect() as conn:
        sic = pd.read_sql(text(
            "SELECT [Plan Name], [Part Code], [Plan Model ID], [Plan Month], Quantity "
            "FROM Warehouse_AI.dbo.SIC_DemandPlan WHERE Plant = :plant AND Quantity IS NOT NULL"
        ), conn, params={'plant': str(plant)})
    sic = sic.rename(columns={'Plan Name': 'PlanName', 'Part Code': 'PartCode',
        'Plan Model ID': 'PlanModelCode', 'Quantity': 'Predicted'})

    plan_yyyymm = sic['PlanName'].str.extract(r'^(\d{6})-Proj$')[0]
    if plan_yyyymm.isna().any():
        raise ValueError(f"Unrecognised Plan Name(s): {sorted(sic.loc[plan_yyyymm.isna(), 'PlanName'].unique())}")
    sic['OriginIdx'] = to_month_idx(plan_yyyymm.astype(int).to_numpy()) - SIC_PLAN_ORIGIN_OFFSET
    sic['TargetIdx'] = to_month_idx(sic['Plan Month'].astype(int).to_numpy())
    sic['Horizon'] = sic['TargetIdx'] - sic['OriginIdx']

    if sic.duplicated(['PlanModelCode', 'OriginIdx', 'TargetIdx']).any():
        raise ValueError("SIC_DemandPlan has >1 row per (Plan Name, Plan Model ID, Plan Month).")
    return sic[['PartCode', 'PlanModelCode', 'PlanName', 'OriginIdx', 'TargetIdx', 'Horizon', 'Predicted']]


def preprocess_sic_plan(plant: int, entity_type: str = 'PlanModelCode', exclude_partcodes=None,
        horizons=range(1, 7)) -> pd.DataFrame:
    """SIC demand plan for `horizons`, at PlanModelCode grain or summed to PartCode.
    `exclude_partcodes` defaults to PLANT_EXCLUDE_PARTCODES[plant]."""
    if entity_type not in _SIC_ENTITY_TYPES:
        raise ValueError(f"entity_type must be one of {_SIC_ENTITY_TYPES}, got {entity_type!r}")
    if exclude_partcodes is None:
        exclude_partcodes = PLANT_EXCLUDE_PARTCODES[plant]
    sic = load_sic_demand_plan(plant)
    sic = sic[sic['Horizon'].isin(list(horizons)) & ~sic['PartCode'].isin(list(exclude_partcodes))]

    if entity_type == 'PartCode':
        return sic.groupby(['PartCode', 'OriginIdx', 'TargetIdx', 'Horizon'], as_index=False)['Predicted'].sum()
    return sic.reset_index(drop=True)

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Construct Processed DataFrame for Training or Inference")
    p.add_argument('--plant', type=int, required=True, help='Plant code.')
    p.add_argument('--exclude-partcodes', default=None,
        help="Comma-separated PartCodes to exclude. Omit to use PLANT_EXCLUDE_PARTCODES' "
             "per-plant default (see config.py).")
    p.add_argument('--data-dir', default='data')
    p.add_argument('--type', choices=['train', 'inference', 'sic'], required=True,
        help="Whether to build a full training panel or an inference skeleton for one Origin.")
    p.add_argument('--month-range', default=None, help="YYYYMM range for training panel or sic (ignored for inference) e.g. 201701-202606")
    p.add_argument('--max-settled-target', type=int, default=None,
        help="Max settled target month (YYYYMM), only used with --type train. Omit to derive it "
             "from the data (one month before the latest EntryMonth present) - see "
             "build_order_book_context.")
    p.add_argument('--entity-type', default='PlanModelCode', choices=list(_SIC_ENTITY_TYPES),
        help='SIC grain (only used with --type sic).')
    p.add_argument('--cutoff-grain', default='day', choices=list(CUTOFF_GRAINS),
        help="As-of grain. 'month': one row per (code, origin, horizon), as-of the origin month's "
             "end. 'day' (default, the deployed grain) fans each row across every cutoff day (~31x "
             "the rows) and writes a '_day' filename so it can't be confused with a month panel.")
    p.add_argument('--cutoff-date', default=None,
        help="As-of date (YYYY-MM-DD) for --type inference --cutoff-grain day. Defaults to the day "
             "before the latest source file's date.")
    p.add_argument('--origin', type=int, default=None, help="Origin month as YYYYMM, e.g. 202607.")
    p.add_argument('--horizons', default='1,2,3,4,5,6', help="Comma-separated horizons (only used with --origin).")
    p.add_argument('--codes', default=None, help="Comma-separated entity codes (only used with --origin).")
    return p.parse_args()


def main():
    args = _parse_args()
    # None (not passed): each entry point resolves PLANT_EXCLUDE_PARTCODES[plant] itself.
    exclude_partcodes = (tuple(c.strip() for c in args.exclude_partcodes.split(',') if c.strip())
        if args.exclude_partcodes else None)

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)

    plant_name = PLANT_NAMES.get(args.plant, args.plant)
    os.makedirs(f"{args.data_dir}/{plant_name}", exist_ok=True)

    if args.type == 'train':
        panel, _feature_families, feature_cols, _ctx = preprocess_for_training(
            args.plant, exclude_partcodes=exclude_partcodes,
            max_settled_target=args.max_settled_target, cutoff_grain=args.cutoff_grain)
        # Day panels get their own filename: a day panel has ~31x the rows and the IntraMonth family,
        # so sharing a name would silently overwrite one with the other.
        suffix = '_day' if args.cutoff_grain == 'day' else ''
        name = f"train11_{args.month_range}{suffix}.parquet"
        panel.to_parquet(f"{args.data_dir}/{plant_name}/{name}", index=False)
        log.info("Saved as %s/%s/%s", args.data_dir, plant_name, name)
        log.info("panel shape: %s (cutoff_grain=%s)", panel.shape, args.cutoff_grain)
        log.info("feature_cols (%d): %s", len(feature_cols), feature_cols)
        print(panel.head())
    elif args.type == 'sic':
        sic = preprocess_sic_plan(args.plant, entity_type=args.entity_type, exclude_partcodes=exclude_partcodes)
        out_path = f"{args.data_dir}/{plant_name}/{args.entity_type}_sic_{args.month_range}.parquet"
        sic.to_parquet(out_path, index=False)
        log.info("Saved as %s", out_path)
        log.info("sic shape: %s", sic.shape)
        print(sic.head())
    else:
        horizons = [int(h) for h in args.horizons.split(',') if h.strip()]
        codes = [c.strip() for c in args.codes.split(',') if c.strip()] if args.codes else None
        skeleton, _ctx = preprocess_for_inference(args.plant, args.origin, horizons, codes=codes,
            exclude_partcodes=exclude_partcodes, cutoff_grain=args.cutoff_grain,
            cutoff_date=pd.Timestamp(args.cutoff_date) if args.cutoff_date else None)
        suffix = '_day' if args.cutoff_grain == 'day' else ''
        name = f"inf10_{args.origin}{suffix}.parquet"
        skeleton.to_parquet(f"{args.data_dir}/{plant_name}/{name}", index=False)
        log.info("Saved as %s/%s/%s", args.data_dir, plant_name, name)
        log.info("skeleton shape: %s", skeleton.shape)
        print(skeleton.head())


if __name__ == '__main__':
    main()
