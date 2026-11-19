"""Shared constants and month-arithmetic helpers; pipeline constants are required env vars."""
import json
import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()  # same pattern as db_writer.py/preprocessing.py/collate_outstanding_order.py - lets
# config.py work standalone (e.g. imported alone in a notebook) regardless of import order.


def _env_json(key):
    """Parses a required JSON-encoded env var - raises KeyError if unset (no default)."""
    return json.loads(os.environ[key])


def _int_keys(d: dict) -> dict:
    """JSON object keys are always strings - this codebase's plant-keyed dicts use int plant
    codes, so every _env_json'd plant dict is normalized through this right after parsing."""
    return {int(k): v for k, v in d.items()}


def _env_list(key, cast=str):
    """Parses a required comma-separated env var - raises KeyError if unset (no default)."""
    return [cast(x.strip()) for x in os.environ[key].split(',') if x.strip()]


def _env_tuple(key, cast=str):
    return tuple(_env_list(key, cast))


# Excel predecessor/successor workbook for lifecycle_cleaning's CLI. The daily run reads
# ProductSuccessor from the database instead.
MAPPINGS_DIR = 'data/raw/mappings'
REPLACEMENT_XLSX_FILENAME = 'STJ替换型号核对.xlsx'

# Fabric sometimes stores CNY as RMB. load_io normalizes the code before anything else reads it.
CURRENCY_ALIASES = {'RMB': 'CNY'}

# Repo root, not under data_dir - resolved relative to cwd, so it doesn't matter that train.py/
# predict.py themselves now live under adj_forecast_lib/.
IO_SQL_PATH = os.environ['IO_SQL_PATH']
IO_OUTSTANDING_SQL_PATH = os.environ['IO_OUTSTANDING_SQL_PATH']

N_ESTIMATORS = int(os.environ['N_ESTIMATORS'])
EARLY_STOPPING_ROUNDS = int(os.environ['EARLY_STOPPING_ROUNDS'])

# Experimental, default off: train each code from its first to last demand month even outside its
# TFG lifecycle (launch features nulled there); inference rows are unchanged either way.
TRAIN_EXTEND_TO_ORDERS = False

MLFLOW_TRACKING_URI = os.environ['MLFLOW_TRACKING_URI']
# The Bamboo agent's console is cp1252, which can't encode the emoji MLflow prints after every
# run - set before any mlflow.start_run() so it never reaches sys.stdout.write() and crashes the run.
os.environ.setdefault('MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT', 'True')
# MLflow's defaults (7 retries x 120s timeout) take ~20 min to fail one request when the server is
# down. 5 x 60s fails in ~7 min and its backoff (0+4+8+16+32s) still rides out a server restart.
# MLflow reads these per request, so setting them after mlflow is imported still takes effect.
os.environ.setdefault('MLFLOW_HTTP_REQUEST_MAX_RETRIES', '5')
os.environ.setdefault('MLFLOW_HTTP_REQUEST_TIMEOUT', '60')

# Chinese New Year is treated as this calendar month EVERY year - a simplification, since the
# holiday itself moves between late January and mid February. A rule rather than a list of years,
# so no target month ever falls past the end of a list. See is_cny_month_idx/months_to_cny below.
CNY_MONTH = 2

# Earliest Order_Entry_Date / Customer_Request_Date load_io accepts - catches placeholder dates
# such as 1900-01-01, which would otherwise become the dataset floor. Not a way to trim history.
MIN_VALID_ORDER_DATE = '2010-01-01'

# MAX_SETTLED_TARGET/MIN_SETTLED_TARGET are both derived, not constants here - see
# build_order_book_context in preprocessing.py (MAX from the latest EntryMonth present, one
# month back; MIN from lifecycle_features.compute_dataset_floor() + 1). A caller can still pass
# an explicit max_settled_target override there (or via train.py's --max-settled-target).


def to_month_idx(yyyymm):
    """Convert a YYYYMM int (or array of them) to a monotonic month-ordinal, since YYYYMM ints
    aren't arithmetic-safe across year boundaries (e.g. 202511 + 3 isn't a real month)."""
    yyyymm = np.asarray(yyyymm)
    year, month = yyyymm // 100, yyyymm % 100
    return year * 12 + month


def month_idx_to_yyyymm(idx):
    """Inverse of to_month_idx."""
    idx = np.asarray(idx)
    year, month = np.divmod(idx - 1, 12)
    return year * 100 + (month + 1)


def date_to_month_idx(date_series):
    """Convert a datetime Series directly to month-ordinals (see to_month_idx)."""
    yyyymm = date_series.dt.year * 100 + date_series.dt.month
    return to_month_idx(yyyymm.to_numpy())


# The as-of ("what was known, when") grain a panel/skeleton expresses. 'month' means OriginIdx
# alone IS the as-of point - every panel built before daily refresh. 'day' means the rows carry a
# real CUTOFF_DATE_COL date, so the as-of point can fall part-way through a month. The forecast
# TARGET stays monthly in both cases; only the cutoff moves. See adj_forecast_lib.order_book.
CUTOFF_GRAINS = ('month', 'day')
# Column name a day-grain panel/skeleton carries its as-of date in. Its ABSENCE means month
# grain, not "unknown" - so order_book.frame_cutoff_grain reads an old parquet correctly with no
# migration, and a model trained before the field existed keeps its month-grain meaning.
CUTOFF_DATE_COL = 'CutoffDate'

# Plant code -> short name, for building human-readable paths/labels (e.g. data/{name}/...)
# instead of the bare numeric code.
PLANT_NAMES = _int_keys(_env_json('PLANT_NAMES'))


def plant_name(plant) -> str:
    """Plant code -> short name for paths and labels; unknown codes fall back to the code itself."""
    try:
        code = int(plant)
    except (TypeError, ValueError):
        raise ValueError(f"plant must be a numeric plant code (e.g. 2100), got {plant!r}") from None
    return PLANT_NAMES.get(code, str(code))

# Default PartCodes to exclude from `orders`, per plant - see adj_forecast_lib.preprocessing.load_io.
PLANT_EXCLUDE_PARTCODES = {k: tuple(v) for k, v in _int_keys(_env_json('PLANT_EXCLUDE_PARTCODES')).items()}

# Fixed, hand-specified date ranges (business knowledge, not fit from data) - inclusive YYYYMM
# ints. Consumed by adj_forecast_lib.features for the point-in-time HasExternalEvent flag (its
# hindsight twin EventTargetHindsight drives the external-event sample weight) and the
# Origin-anchored lag-window contamination features. One list per plant - external events don't
# necessarily affect both plants the same way. Every event counts as known from its start month.
STJ_EXTERNAL_EVENT_RANGES = [tuple(pair) for pair in _env_json('STJ_EXTERNAL_EVENT_RANGES')]  # Plant 3300
SPL_EXTERNAL_EVENT_RANGES = [tuple(pair) for pair in _env_json('SPL_EXTERNAL_EVENT_RANGES')]  # Plant 2100
PLANT_EXTERNAL_EVENT_RANGES = {2100: SPL_EXTERNAL_EVENT_RANGES, 3300: STJ_EXTERNAL_EVENT_RANGES}
# The subset of those ranges whose END was announced when the event started (e.g. a surcharge set for a
# fixed period). For the others the end only becomes known once it passes. See
# features.attach_external_factor_features.
STJ_EXTERNAL_EVENT_END_KNOWN = [tuple(pair) for pair in _env_json('STJ_EXTERNAL_EVENT_END_KNOWN')]
SPL_EXTERNAL_EVENT_END_KNOWN = [tuple(pair) for pair in _env_json('SPL_EXTERNAL_EVENT_END_KNOWN')]
PLANT_EXTERNAL_EVENT_END_KNOWN = {2100: SPL_EXTERNAL_EVENT_END_KNOWN, 3300: STJ_EXTERNAL_EVENT_END_KNOWN}
# A typo'd END_KNOWN pair would silently match no range, turning that event into an end-unknown one.
for _plant, _end_known in PLANT_EXTERNAL_EVENT_END_KNOWN.items():
    _stray = sorted(set(_end_known) - set(PLANT_EXTERNAL_EVENT_RANGES[_plant]))
    if _stray:
        raise ValueError(f"{plant_name(_plant)}_EXTERNAL_EVENT_END_KNOWN lists {_stray}, "
                         f"not in {plant_name(_plant)}_EXTERNAL_EVENT_RANGES")

# Does Chinese New Year matter as a seasonal driver for this plant's Target-anchored calendar
# features (Is_CNY_Month/Months_to_CNY)? See adj_forecast_lib.features.attach_calendar_features.
PLANT_INCLUDES_CNY = _int_keys(_env_json('PLANT_INCLUDES_CNY'))

# REPLACEMENT_XLSX_FILENAME is STJ-only predecessor/successor data - no SPL code ever matches it.
PLANT_INCLUDES_PRED_SUCC = _int_keys(_env_json('PLANT_INCLUDES_PRED_SUCC'))

# Whether to exclude Sold rows from OutstandingOrder, per plant - see collate_outstanding_order.py.
PLANT_EXCLUDES_SOLD_FROM_OUTSTANDING = _int_keys(_env_json('PLANT_EXCLUDES_SOLD_FROM_OUTSTANDING'))

# Whether main.py's prediction phase also scores predict_manual (the ManualSim{window} Excel-
# replica baseline) - only STJ's business process has that method to compare against.
PLANT_INCLUDES_MANUAL_METHOD = _int_keys(_env_json('PLANT_INCLUDES_MANUAL_METHOD'))

COVID_YYYYMM_RANGE = _env_tuple('COVID_YYYYMM_RANGE', cast=int)

# TFG SubModelCode -> PlanModelCode per plant, filling only SubModelCodes orders don't rename.
# Current values are user-supplied SPL mappings (mapping_spl.xlsx), not in ARAS as of 11 Oct 2026.
PLANT_SUBMODEL_PLAN_MODEL_OVERRIDES = _int_keys(_env_json('PLANT_SUBMODEL_PLAN_MODEL_OVERRIDES'))

# Country-mix visibility family (country_visibility.py). Motivated by the SPL EDA
# (SPL_office_customer_signal_test.ipynb, 2026-09): qty-weighted mean order lead time spans 4.2
# months across customer countries - CN 5.16mo (h=6 book only 29.6% filled) vs DE 6.82mo (44.7%) -
# so a SKU's own country mix predicts how filled its book should be at each horizon. Country beats
# Sales_Office_Name on the same measures (4.20 vs 1.85 months of lead spread); offices aggregate
# countries and wash the contrast out, so the office version is deliberately not built.
FOCUS_COUNTRY = 'CN'
# CN is the single largest (33.5% of SPL Qty) AND latest-arriving segment, so it gets explicit
# share/gap columns on top of the pooled ExpectedVisibility. Change per plant if STJ's mix differs.
PLANT_INCLUDES_COUNTRY_VISIBILITY = {2100: True, 3300: True}
# Trailing months of settled history defining a SKU's "normal" country mix. Trailing, not
# expanding - the mix drifts, and a full-history average keeps a dead composition alive for years.
COUNTRY_MIX_WINDOW = 12
# Countries below this share of a SKU's windowed volume are dropped and the rest renormalised -
# 64 countries is mostly a long tail for any one SKU, and it keeps the mix table to a sane size.
COUNTRY_MIX_MIN_SHARE = 0.01

def is_event_month_idx(month_idx, event_ranges) -> np.ndarray:
    """Boolean mask: does each month-ordinal in `month_idx` fall in one of `event_ranges`?"""
    yyyymm = month_idx_to_yyyymm(np.asarray(month_idx))
    is_event = np.zeros(yyyymm.shape, dtype=bool)
    for lo, hi in event_ranges:
        is_event |= (yyyymm >= lo) & (yyyymm <= hi)
    return is_event

def is_known_event_idx(origin_idx, target_idx, event_ranges, end_known_ranges=()) -> np.ndarray:
    """Boolean mask: is the target month in an event already known at the origin (point-in-time)?"""
    origin_idx, target_idx = np.asarray(origin_idx), np.asarray(target_idx)
    known = np.zeros(origin_idx.shape, dtype=bool)
    for lo, hi in event_ranges:
        start, end = to_month_idx(lo), to_month_idx(hi)
        started = origin_idx >= start                   # known from its start month
        if (lo, hi) in end_known_ranges:
            known |= started & (target_idx <= end)      # end announced: flag targets up to it
        else:
            known |= started & (origin_idx <= end)      # end unknown: still running -> assume it continues
    return known

def is_covid_month_idx(month_idx) -> np.ndarray:
    """Boolean mask: is each month-ordinal in COVID_YYYYMM_RANGE?"""
    yyyymm = month_idx_to_yyyymm(np.asarray(month_idx))
    lo, hi = COVID_YYYYMM_RANGE
    return (yyyymm >= lo) & (yyyymm <= hi)


def _calendar_month(month_idx) -> np.ndarray:
    """Calendar month (1..12) of each month-ordinal - to_month_idx's month part, read back."""
    return (np.asarray(month_idx, dtype='int64') - 1) % 12 + 1


def is_cny_month_idx(month_idx) -> np.ndarray:
    """Boolean mask: is each month-ordinal a CNY month (CNY_MONTH, in any year)?"""
    return _calendar_month(month_idx) == CNY_MONTH


def months_to_cny(month_idx) -> np.ndarray:
    """Months from each month-ordinal forward to the next CNY month: 0 in a CNY month itself, so
    always 0..11, in any year."""
    return (CNY_MONTH - _calendar_month(month_idx)) % 12
