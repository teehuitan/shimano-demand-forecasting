"""Launch/end-date resolution from TFGMaterial and predecessor/successor lookup from the workbook."""
import argparse
import logging
import os
import urllib.parse

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from . import db_writer
from .config import to_month_idx, date_to_month_idx, MAPPINGS_DIR, REPLACEMENT_XLSX_FILENAME
from .logging_setup import configure_logging
from .replacement_workbook import _clean_replacement_sheets, _build_ps_ranges

log = logging.getLogger(__name__)

load_dotenv()
_ai_odbc_str = (
    f"DRIVER={{{os.environ['ODBC_DRIVER']}}};"
    f"SERVER={os.environ['QA_DB_SERVER']};"
    f"UID={os.environ['QA_DB_UID']};"
    f"PWD={os.environ['QA_DB_PWD']};"
)
_ai_engine = create_engine(
    "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(_ai_odbc_str),
    pool_pre_ping=True,
)


def resolve_plan_model_codes(orders: pd.DataFrame, tfg_material: pd.DataFrame,
        entry_col: str = 'EntryDate', submodel_overrides: dict = None) -> tuple:
    """Adds PlanModelCode to both frames, dropping uncoded orders; a ProductCode with several codes
    takes its latest order's."""
    sub = tfg_material.set_index('ProductCode')['SubModelCode']
    by_product = orders.dropna(subset=['PlanModelCode']).groupby('ProductCode')['PlanModelCode']
    multi = by_product.nunique() > 1
    product_pmc = by_product.first()
    if multi.any():
        # Only the multi-code products are re-sorted; one history under the current code, not a split.
        m = orders[orders['ProductCode'].isin(multi[multi].index) & orders['PlanModelCode'].notna()]
        latest = m.assign(_entry=pd.to_datetime(m[entry_col])).sort_values('_entry', kind='stable') \
            .groupby('ProductCode')['PlanModelCode'].last()
        log.warning("%d ProductCode(s) have >1 PlanModelCode on orders, all rows moved to the latest order's: %s",
            multi.sum(), {pc: f"{sorted(codes)} -> {latest[pc]}" for pc, codes in by_product.unique()[multi].items()})
        product_pmc.update(latest)
        orders = orders.assign(PlanModelCode=orders['PlanModelCode'].where(
            ~orders['ProductCode'].isin(latest.index) | orders['PlanModelCode'].isna(),
            orders['ProductCode'].map(latest)))
    # SubModelCode -> the one other PlanModelCode its ordered products carry.
    targets = product_pmc.groupby(product_pmc.index.map(sub)).agg(lambda s: sorted(set(s)))
    renames = targets[targets.str.len() == 1].str[0]
    renames = renames[renames != renames.index]
    # Overrides only for SubModelCodes with no ordered product - orders' own mapping always wins.
    overrides = pd.Series(submodel_overrides or {}, dtype=object)
    renames = pd.concat([renames, overrides[~overrides.index.isin(targets.index)]])

    def from_sub(s: pd.Series) -> pd.Series:
        return s.map(renames).fillna(s)

    pmc = orders['PlanModelCode'].fillna(orders['ProductCode'].map(product_pmc))
    via_sub = pmc.isna()
    pmc = pmc.fillna(from_sub(orders['ProductCode'].map(sub)))
    not_pn = via_sub & pmc.notna() & ~pmc.str.startswith('PN', na=False)
    if not_pn.any():
        log.warning("%d order row(s) got a non-PN PlanModelCode from their SubModelCode (no rename or "
            "override for it): %s", not_pn.sum(), sorted(pmc[not_pn].unique()))
    log.info("PlanModelCode: %d SubModelCode(s) renamed, %d left ambiguous; %d blank row(s) filled, "
        "%d dropped with no code.", len(renames), (targets.str.len() > 1).sum(),
        (orders['PlanModelCode'].isna() & pmc.notna()).sum(), pmc.isna().sum())
    orders = orders.assign(PlanModelCode=pmc).dropna(subset=['PlanModelCode'])
    order_pmc = orders.groupby('ProductCode')['PlanModelCode'].first()
    tfg_material = tfg_material.assign(PlanModelCode=tfg_material['ProductCode'].map(order_pmc)
        .fillna(from_sub(tfg_material['SubModelCode'])))
    # Override rows last, then newest product first (ProductCode breaks ties), so a code's 'first'
    # ProductAbbreviation is its newest pre-override product's - Fabric returns rows in no fixed order.
    tfg_material = tfg_material.assign(
        _override=tfg_material['SubModelCode'].isin(list(submodel_overrides or {})),
        _start=pd.to_datetime(tfg_material['ProductStartDate'], errors='coerce'),
    ).sort_values(['_override', '_start', 'ProductCode'], ascending=[True, False, True], na_position='last') \
        .drop(columns=['_override', '_start'])
    return orders, tfg_material


def _scope_tfg_to_order_codes(orders: pd.DataFrame, tfg_material: pd.DataFrame) -> pd.DataFrame:
    """Every TFGMaterial row under a PlanModelCode in `orders`, including unordered ProductCodes."""
    scoped = tfg_material[tfg_material['PlanModelCode'].isin(orders['PlanModelCode'].unique())].copy()
    for col in ['ProductStartDate', 'ProductEndDate']:
        scoped[col] = pd.to_datetime(scoped[col], errors='coerce')
    return scoped


def _preview(codes, n: int = 10) -> str:
    codes = sorted(codes)
    return ', '.join(codes[:n]) + (f', ... (+{len(codes) - n} more)' if len(codes) > n else '')


def _warn_codes_without_range(orders: pd.DataFrame, tfg_material: pd.DataFrame, result: pd.DataFrame) -> None:
    """Names ordered codes with no TFGMaterial range - otherwise silently neither trained nor forecast."""
    missing = set(orders['ProductCode'].dropna()) - set(tfg_material['ProductCode'])
    if missing:
        log.warning("%d ProductCodes have orders but no TFGMaterial record: %s", len(missing),
            _preview(missing))
    dropped = set(orders['PlanModelCode'].dropna()) - set(result['PlanModelCode'])
    if dropped:
        log.warning("%d PlanModelCodes have orders but no effective range - they will not be "
            "trained or forecast: %s", len(dropped), _preview(dropped))


def resolve_product_code_ranges(orders: pd.DataFrame, tfg_material: pd.DataFrame) -> pd.DataFrame:
    """ProductCode-grain counterpart to resolve_effective_ranges_with_tfg, same scope and
    Start-after-End rule. Callers must exclude a ProductCode with no Start, never default it."""
    tfg_material_scoped = _scope_tfg_to_order_codes(orders, tfg_material)
    # groupby, not set_index: TFGMaterial is one row per ProductCode (verified via EDA) - this
    # is just a defensive fallback if that ever stops being true, not handling a real case today.
    result = tfg_material_scoped.groupby('ProductCode', as_index=False).agg(
        StartMonthIdx=('ProductStartDate', 'min'),
        EndMonthIdx=('ProductEndDate', lambda x: x.max(skipna=False)),
    )
    result['PlanModelCode'] = result['ProductCode'].map(
        tfg_material_scoped.groupby('ProductCode')['PlanModelCode'].first())
    result['StartMonthIdx'] = date_to_month_idx(result['StartMonthIdx'])
    result['EndMonthIdx'] = date_to_month_idx(result['EndMonthIdx'])

    inverted = result['StartMonthIdx'] > result['EndMonthIdx']
    result.loc[inverted, 'StartMonthIdx'] = np.nan
    if inverted.any():
        log.warning("%d ProductCode(s) had Start after End - Start nulled: %s",
            inverted.sum(), sorted(result.loc[inverted, 'ProductCode']))

    return result[['ProductCode', 'PlanModelCode', 'StartMonthIdx', 'EndMonthIdx']]


def resolve_effective_ranges_with_tfg(orders: pd.DataFrame, tfg_material: pd.DataFrame) -> pd.DataFrame:
    """One row per PlanModelCode: earliest Start and latest End (NaN if any ProductCode is ongoing)."""
    tfg_material_scoped = _scope_tfg_to_order_codes(orders, tfg_material)
    result = tfg_material_scoped.groupby('PlanModelCode', as_index=False).agg(
        StartMonthIdx=('ProductStartDate', 'min'),
        EndMonthIdx=('ProductEndDate', lambda x: x.max(skipna=False)),
        PlanModelText=('ProductAbbreviation', 'first'),
    )
    n_text = tfg_material_scoped.groupby('PlanModelCode')['ProductAbbreviation'].nunique()
    if (n_text > 1).any():
        log.warning("%d PlanModelCode(s) have >1 ProductAbbreviation, newest product's used: %s", (n_text > 1).sum(),
            sorted(n_text[n_text > 1].index))
    result['StartMonthIdx'] = date_to_month_idx(result['StartMonthIdx'])
    result['EndMonthIdx'] = date_to_month_idx(result['EndMonthIdx'])

    inverted = result['StartMonthIdx'] > result['EndMonthIdx']
    result.loc[inverted, 'StartMonthIdx'] = np.nan
    if inverted.any():
        log.warning("%d PlanModelCode(s) had Start after End - Start nulled: %s",
            inverted.sum(), sorted(result.loc[inverted, 'PlanModelCode']))

    _warn_codes_without_range(orders, tfg_material, result)
    return result


def get_predecessor_successor_flags_from_excel(replacement_xlsx_path: str, max_settled_target: int,
        effective_ranges: pd.DataFrame) -> pd.DataFrame:
    """Predecessor/successor flags from the Product Successor sheet, with launch dates from
    `effective_ranges`."""
    max_settled_idx = to_month_idx(max_settled_target)
    xl = pd.ExcelFile(replacement_xlsx_path)
    succ_raw = xl.parse('Product Successor')
    lc_raw = xl.parse('Product Lifecycle')
    succ, _, _ = _clean_replacement_sheets(succ_raw, lc_raw)
    code_table, _, _, _ = _build_ps_ranges(succ, max_settled_target, max_settled_idx)
    flags = code_table[['SalesModelCode', 'HasPredecessor', 'Successor']]
    return flags.merge(
        effective_ranges[['SalesModelCode', 'StartMonthIdx']].rename(columns={'StartMonthIdx': 'effective_start'}),
        on='SalesModelCode', how='left')


def get_predecessor_successor_flags_from_db(effective_ranges: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """ProductSuccessor (SalesModelCode pairs) mapped onto PlanModelCode - [PlanModelCode, HasPredecessor, SuccessorStart]."""
    pairs = pd.read_sql(text(
        "SELECT SalesModelCode, SuccessorCode FROM [AI].DemandForecast.ProductSuccessor"
    ), _ai_engine)
    smc = effective_ranges['SalesModelCode']
    # SalesModelCode lifecycle = earliest start of its PlanModelCodes; pairs kept only if both codes are in this plant.
    smc_start = effective_ranges.groupby(smc)['StartMonthIdx'].min()
    pairs = pairs[pairs['SalesModelCode'].isin(smc_start.index) & pairs['SuccessorCode'].isin(smc_start.index)]
    # A successor with no TFG start falls back to its first order entry month - never earlier than it
    # was knowable, unlike treating it as known since the start (a leak). Still NaN -> no successor flag.
    known_start = smc_start.fillna(orders.groupby('SalesModelCode')['EntryMonthIdx'].min())
    succ_start = (pairs.assign(start=pairs['SuccessorCode'].map(known_start))
        .groupby('SalesModelCode')['start'].min())
    flags = effective_ranges[['PlanModelCode']].assign(HasPredecessor=smc.isin(set(pairs['SuccessorCode'])),
        SuccessorStart=smc.map(succ_start))
    log.info("Pred/succ: %d SalesModelCode pair(s), %d PlanModelCode(s) with a predecessor, %d with a successor.",
        len(pairs), flags['HasPredecessor'].sum(), flags['SuccessorStart'].notna().sum())
    return flags


def _parse_args():
    p = argparse.ArgumentParser(
        description="Refresh DemandForecast.ProductSuccessor from the replacement workbook.")
    p.add_argument('--plant', type=int, required=True, help="Plant code (2100=SPL, 3300=STJ).")
    p.add_argument('--replacement-xlsx', default=None,
        help=f"Path to the replacement workbook. Defaults to {MAPPINGS_DIR}/{REPLACEMENT_XLSX_FILENAME}.")
    return p.parse_args()


def main():
    args = _parse_args()
    configure_logging('product_successor_refresh', args.plant)

    replacement_xlsx = args.replacement_xlsx or f'{MAPPINGS_DIR}/{REPLACEMENT_XLSX_FILENAME}'

    # max_settled_target/effective_ranges only feed ps_end/effective_start in the function below -
    # neither of which upsert_product_successor reads (it only writes SalesModelCode/Successor
    # pairs) - so no Fabric/TFG load needed here, just a placeholder max_settled_target and an
    # empty effective_ranges (the left-merge just leaves effective_start all-NaN, unused).
    placeholder_max_settled_target = int(pd.Timestamp.now().strftime('%Y%m'))
    empty_effective_ranges = pd.DataFrame({'SalesModelCode': [], 'StartMonthIdx': []})
    pred_succ_df = get_predecessor_successor_flags_from_excel(replacement_xlsx,
        placeholder_max_settled_target, empty_effective_ranges)

    n = db_writer.upsert_product_successor(pred_succ_df, args.plant)
    log.info("Upserted %d ProductSuccessor row(s) for plant %s", n, args.plant)


if __name__ == '__main__':
    main()