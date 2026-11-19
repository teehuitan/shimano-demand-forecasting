"""Replacement-workbook parsing, still used for predecessor/successor relationships."""
import re

import numpy as np
import pandas as pd

from .config import to_month_idx, date_to_month_idx, month_idx_to_yyyymm
from .data_loading import clean_code
from .lifecycle_features import compute_dataset_floor

#  if it contains a parenthesis or bracket character, or the words "except", "rev." (a revision-note marker), 
# "note", "see", "tbc", or "na"/"n/a" as a whole word 
# — matching case-insensitively and on any single one of these being present anywhere in the string
MALFORMED_CODE_PATTERN = re.compile(r'[()\[\]]|except|rev\.|note|see|tbc|n/?a\b', re.IGNORECASE)
DISCONTINUED_HIGHLIGHT_RGB = 'FFFF9900'
NOSUBSTITUTION = 'NOSUBSTITUTION'
PLACEHOLDER_NEW_KEYS = (NOSUBSTITUTION, 'CANCEL')  # non-product values found in new_key -
# 'no real substitute' / 'cancelled' notes, not real SalesModelCodes. Found by checking for
# old_key/new_key values with no digit at all (every real code has one) - these were the only
# two matches in the current file.


def clean_replacement_code_column(df: pd.DataFrame, raw_col: str, source_label: str) -> tuple:
    """Cleans one raw code column, quarantining free-text rows and splitting '&'-joined cells.
    Returns (cleaned_df, flags_df)."""
    df = df.reset_index(drop=True).copy()
    raw = df[raw_col].astype(str)

    is_malformed = raw.str.contains(MALFORMED_CODE_PATTERN, na=False)
    flags = []
    for idx in df.index[is_malformed]:
        flags.append({'source': source_label, 'column': raw_col, 'row_index': idx,
            'original_value': df.at[idx, raw_col], 'treated_value': None,
            'treatment': 'quarantined (malformed - free-text annotation, not a real code)'})

    is_multi = (~is_malformed) & raw.str.contains('&', na=False)
    for idx in df.index[is_multi]:
        parts = [p.strip() for p in str(df.at[idx, raw_col]).split('&')]
        flags.append({'source': source_label, 'column': raw_col, 'row_index': idx,
            'original_value': df.at[idx, raw_col], 'treated_value': ', '.join(parts),
            'treatment': f"split '&'-joined multi-value cell into {len(parts)} row(s)"})

    cleaned = df[~is_malformed].copy()
    cleaned[raw_col] = cleaned[raw_col].astype(str).str.split('&')
    cleaned = cleaned.explode(raw_col).reset_index(drop=True)
    cleaned[raw_col] = cleaned[raw_col].str.strip()
    cleaned[f'{raw_col}_key'] = clean_code(cleaned[raw_col])

    return cleaned, pd.DataFrame(flags, columns=['source', 'column', 'row_index',
        'original_value', 'treated_value', 'treatment'])


def read_lifecycle_highlight_flags(xlsx_path: str, sheet_name: str = 'Product Lifecycle',
        code_col: str = 'model_name', date_col: str = 'first_active_month',
        highlight_rgb: str = DISCONTINUED_HIGHLIGHT_RGB) -> pd.DataFrame:
    """Reads discontinued Product Lifecycle rows from the workbook's cell highlighting (openpyxl)."""
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb[sheet_name]
    header = [c.value for c in ws[1]]
    name_idx = header.index(code_col)
    date_idx = header.index(date_col)

    rows = []
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        if row[date_idx].value is None:
            continue  # blank row or the legend/key row - not real product data
        fill = row[name_idx].fill
        is_discontinued = bool(fill is not None and fill.patternType == 'solid'
            and fill.fgColor is not None and fill.fgColor.rgb == highlight_rgb)
        rows.append({code_col: row[name_idx].value, 'is_discontinued_per_lifecycle': is_discontinued})
    return pd.DataFrame(rows)


def _clean_replacement_sheets(succ_raw, lc_raw):
    """Cleans both sheets' code columns, tagging self-matches and placeholders. Returns (succ, lc,
    cleaning_flags)."""
    succ_c1, flags_old = clean_replacement_code_column(succ_raw, '老型号model', 'Product Successor')
    succ_c2, flags_new = clean_replacement_code_column(succ_c1, '新型号model', 'Product Successor')
    succ = succ_c2.rename(columns={'老型号model_key': 'old_key', '新型号model_key': 'new_key'})

    lc_c, flags_lc = clean_replacement_code_column(lc_raw, 'model_name', 'Product Lifecycle')
    lc = lc_c.rename(columns={'model_name_key': '_key'}).dropna(subset=['first_active_month'])

    succ['is_self_match'] = succ['old_key'] == succ['new_key']
    for _, r in succ[succ['is_self_match']].iterrows():
        flags_old = pd.concat([flags_old, pd.DataFrame([{
            'source': 'Product Successor', 'column': 'old_key==new_key', 'row_index': None,
            'original_value': f"{r['老型号model']} == {r['新型号model']}", 'treated_value': None,
            'treatment': 'excluded from relationship (self-match artifact); dates kept'}])],
            ignore_index=True)

    succ['is_nosub'] = succ['new_key'].isin(PLACEHOLDER_NEW_KEYS)
    for _, r in succ[succ['is_nosub']].iterrows():
        flags_new = pd.concat([flags_new, pd.DataFrame([{
            'source': 'Product Successor', 'column': 'new_key', 'row_index': None,
            'original_value': r['新型号model'], 'treated_value': None,
            'treatment': f"excluded from relationship ({r['new_key']!r} is a placeholder); "
                "old_key side kept"}])],
            ignore_index=True)

    cleaning_flags = pd.concat([flags_old, flags_new, flags_lc], ignore_index=True)
    return succ, lc, cleaning_flags


def _build_ps_ranges(succ, max_settled_target, max_settled_idx):
    """One row per code from Product Successor: ps_start/ps_end and Predecessor/Successor lists.
    Returns (code_table, pred_status, succ_status, date_mismatches)."""
    long_pred = succ[['old_key', 'new_key', 'pred_first', 'pred_last', 'pred_segment',
        'is_self_match', 'is_nosub']].rename(columns={'old_key': 'code', 'new_key': 'partner',
        'pred_first': 'first', 'pred_last': 'last', 'pred_segment': 'segment'})
    long_pred['role'] = 'predecessor'
    long_succ = succ.loc[~succ['is_nosub'], ['new_key', 'old_key', 'succ_first', 'succ_last',
        'succ_segment', 'is_self_match']].rename(columns={'new_key': 'code', 'old_key': 'partner',
        'succ_first': 'first', 'succ_last': 'last', 'succ_segment': 'segment'})
    long_succ['role'] = 'successor'
    long_succ['is_nosub'] = False
    long_df = pd.concat([long_pred, long_succ], ignore_index=True)

    date_mismatches = []
    rows = []
    for code, g in long_df.groupby('code'):
        if g['first'].dropna().nunique() > 1 or g['last'].dropna().nunique() > 1:
            date_mismatches.append({'check': 'within_sheet_code_date_inconsistent', 'old_key': code,
                'new_key': None, 'detail': 'code has disagreeing dates across its own rows - took the union'})
        if g['segment'].dropna().nunique() > 1:
            date_mismatches.append({'check': 'within_sheet_code_segment_inconsistent', 'old_key': code,
                'new_key': None, 'detail': 'code has disagreeing discontinued/not status across its '
                    'own rows - treated as not discontinued unless every occurrence agrees'})
        discontinued = bool((g['segment'] == 'discontinued').all())

        real = g[~g['is_self_match'] & ~g['is_nosub']]
        successors = sorted(set(real.loc[real['role'] == 'predecessor', 'partner']))
        predecessors = sorted(set(real.loc[real['role'] == 'successor', 'partner']))
        if len(successors) > 1:
            date_mismatches.append({'check': 'within_sheet_old_to_multi_new', 'old_key': code,
                'new_key': ', '.join(successors), 'detail': 'one old model maps to multiple new models'})
        if len(predecessors) > 1:
            date_mismatches.append({'check': 'within_sheet_new_to_multi_old', 'old_key': ', '.join(predecessors),
                'new_key': code, 'detail': 'one new model maps to multiple old models'})
        if successors and predecessors:
            date_mismatches.append({'check': 'within_sheet_multi_role', 'old_key': code, 'new_key': code,
                'detail': 'code is both a predecessor and a successor'})

        rows.append({'SalesModelCode': code, 'ps_start_raw': g['first'].min(), 'ps_end_raw': g['last'].max(),
            'ps_discontinued': discontinued, 'Predecessor': predecessors, 'Successor': successors,
            'HasPredecessor': bool(predecessors), 'HasSuccessor': bool(successors)})
    code_table = pd.DataFrame(rows)
    code_table['ps_start'] = date_to_month_idx(code_table['ps_start_raw'])
    code_table['ps_end'] = np.where(code_table['ps_discontinued'],
        date_to_month_idx(code_table['ps_end_raw']), max_settled_idx)
    code_table = code_table.drop(columns=['ps_start_raw', 'ps_end_raw'])

    # Same (old_key, new_key) pair listed more than once - already harmless (the per-code union
    # above already collapses it), flagged once per pair purely for visibility.
    real_pairs = succ[~succ['is_self_match'] & ~succ['is_nosub']]
    dup_counts = real_pairs.groupby(['old_key', 'new_key']).size()
    for (old_key, new_key) in dup_counts[dup_counts > 1].index:
        date_mismatches.append({'check': 'within_sheet_duplicate_pair', 'old_key': old_key,
            'new_key': new_key, 'detail': 'same (old,new) pair occurs more than once in the sheet'})

    non_discontinued = long_df[long_df['segment'] != 'discontinued'].copy()
    non_discontinued['original_end'] = non_discontinued['last']
    non_discontinued['treated_end'] = max_settled_target
    non_discontinued['reason'] = 'segment != discontinued -> do not trust raw last date, use max settled'
    status_cols = ['code', 'role', 'segment', 'original_end', 'treated_end', 'reason']
    status = non_discontinued[status_cols].rename(columns={'code': 'SalesModelCode'})
    pred_status = status[status['role'] == 'predecessor'].reset_index(drop=True)
    succ_status = status[status['role'] == 'successor'].reset_index(drop=True)

    return code_table, pred_status, succ_status, date_mismatches


# Dead from here through resolve_effective_ranges() below - no live caller anywhere in the
# codebase (get_predecessor_successor_flags_from_excel/_from_db use a simpler, TFG-only path
# instead: lifecycle_cleaning.resolve_effective_ranges_with_tfg). Kept in place rather than
# deleted or relocated - this is a real, more thorough Product Successor/Lifecycle/order-book
# reconciliation (with conflict detection + IsCensored) that may be worth wiring in later.
def _build_pl_ranges(lc, codes, replacement_xlsx_path, max_settled_idx):
    """One row per code from Product Lifecycle, as _build_ps_ranges. Returns (pl_ranges, date_mismatches)."""
    highlight = read_lifecycle_highlight_flags(replacement_xlsx_path)
    highlight['_key'] = clean_code(highlight['model_name'])
    lc = lc.merge(highlight[['_key', 'is_discontinued_per_lifecycle']], on='_key', how='left')

    date_mismatches = []
    rows = []
    for key, g in lc.groupby('_key'):
        if g['first_active_month'].dropna().nunique() > 1 or g['last_active_month'].dropna().nunique() > 1:
            date_mismatches.append({'check': 'within_sheet_code_date_inconsistent', 'old_key': key,
                'new_key': None, 'detail': 'Product Lifecycle has disagreeing dates for this code - took the union'})
        rows.append({'SalesModelCode': key, 'pl_start_raw': g['first_active_month'].min(),
            'pl_end_raw': g['last_active_month'].max(),
            'pl_discontinued': bool(g['is_discontinued_per_lifecycle'].fillna(False).all())})
    pl = pd.DataFrame(rows)
    pl['pl_start'] = date_to_month_idx(pl['pl_start_raw'])
    pl['pl_end'] = np.where(pl['pl_discontinued'], date_to_month_idx(pl['pl_end_raw']), max_settled_idx)
    pl_ranges = pl[pl['SalesModelCode'].isin(codes)][
        ['SalesModelCode', 'pl_start', 'pl_end', 'pl_discontinued']].drop_duplicates('SalesModelCode')
    return pl_ranges, date_mismatches


def _build_train_ranges(actual_order: pd.DataFrame, max_settled_idx):
    """First/last settled RequestMonthIdx with nonzero ActualOrder per SalesModelCode, capped at
    max_settled_idx."""
    nonzero = actual_order[(actual_order['ActualOrder'] > 0) &
        (actual_order['RequestMonthIdx'] <= max_settled_idx)]
    return nonzero.groupby('SalesModelCode')['RequestMonthIdx'].agg(
        train_start='min', train_end='max').reset_index()


def _find_status_conflicts(code_table, pl_ranges, max_settled_idx):
    """Resets both sheets to not-discontinued where they disagree on status. Returns (code_table,
    pl_ranges, status_conflicts)."""
    ps_status = code_table[['SalesModelCode', 'ps_discontinued']]
    both = ps_status.merge(pl_ranges[['SalesModelCode', 'pl_discontinued']], on='SalesModelCode', how='inner')
    status_conflicts = both[both['ps_discontinued'] != both['pl_discontinued']].copy()
    conflict_codes = set(status_conflicts['SalesModelCode'])

    code_table = code_table.copy()
    code_table['status_conflict'] = code_table['SalesModelCode'].isin(conflict_codes)
    code_table.loc[code_table['status_conflict'], 'ps_discontinued'] = False
    code_table.loc[code_table['status_conflict'], 'ps_end'] = max_settled_idx

    pl_ranges = pl_ranges.copy()
    pl_is_conflicted = pl_ranges['SalesModelCode'].isin(conflict_codes)
    pl_ranges.loc[pl_is_conflicted, 'pl_discontinued'] = False
    pl_ranges.loc[pl_is_conflicted, 'pl_end'] = max_settled_idx

    return code_table, pl_ranges, status_conflicts


def _yyyymm(month_idx):
    """int month-idx -> int YYYYMM, for human-readable date_mismatch detail strings."""
    return int(month_idx_to_yyyymm(np.array([month_idx]))[0])


def _reject_ps_contradicted(code_table, codes, pl_ranges, train_ranges):
    """Rejects ps_start/ps_end when Product Lifecycle or settled demand falls outside it. Returns
    (ps_ranges, date_mismatches)."""
    ps_ranges = code_table[code_table['SalesModelCode'].isin(codes)][
        ['SalesModelCode', 'ps_start', 'ps_end', 'status_conflict']].drop_duplicates('SalesModelCode').copy()
    ps_ranges = ps_ranges.merge(pl_ranges[['SalesModelCode', 'pl_start', 'pl_end']],
        on='SalesModelCode', how='left')
    ps_ranges = ps_ranges.merge(train_ranges[['SalesModelCode', 'train_start', 'train_end']],
        on='SalesModelCode', how='left')

    reject_start = (ps_ranges['pl_start'] < ps_ranges['ps_start']) | (ps_ranges['train_start'] < ps_ranges['ps_start'])
    reject_end = (ps_ranges['pl_end'] > ps_ranges['ps_end']) | (ps_ranges['train_end'] > ps_ranges['ps_end'])
    ps_reject_mask = reject_start | reject_end

    date_mismatches = []
    for _, r in ps_ranges[ps_reject_mask].iterrows():
        reasons = []
        if r['pl_start'] < r['ps_start']:
            reasons.append(f"Product Lifecycle active from {_yyyymm(r['pl_start'])}, "
                f"before ps_start={_yyyymm(r['ps_start'])}")
        if r['train_start'] < r['ps_start']:
            reasons.append(f"real settled demand from {_yyyymm(r['train_start'])}, "
                f"before ps_start={_yyyymm(r['ps_start'])}")
        if r['pl_end'] > r['ps_end']:
            reasons.append(f"Product Lifecycle active until {_yyyymm(r['pl_end'])}, "
                f"after ps_end={_yyyymm(r['ps_end'])}")
        if r['train_end'] > r['ps_end']:
            reasons.append(f"real settled demand until {_yyyymm(r['train_end'])}, "
                f"after ps_end={_yyyymm(r['ps_end'])}")
        date_mismatches.append({'check': 'cross_sheet_ps_contradicted', 'old_key': r['SalesModelCode'],
            'new_key': None, 'detail': '; '.join(reasons) +
                ' - rejected, falls through to Product Lifecycle/train5'})

    ps_ranges = ps_ranges[~ps_reject_mask][['SalesModelCode', 'ps_start', 'ps_end', 'status_conflict']]

    return ps_ranges, date_mismatches


def _assemble_effective_ranges(codes, ps_ranges, pl_ranges, code_table, first_entry_by_code,
        last_entry_by_code, train_ranges, floor, max_settled_idx, censor_grace_months):
    """Combines PS/PL/order-book sources into one effective range per SalesModelCode. Returns (out,
    date_mismatches)."""
    all_codes = pd.DataFrame({'SalesModelCode': sorted(codes)})
    out = all_codes.merge(ps_ranges, on='SalesModelCode', how='left')
    out = out.merge(pl_ranges, on='SalesModelCode', how='left')
    out = out.merge(code_table[['SalesModelCode', 'Predecessor', 'Successor',
        'HasPredecessor', 'HasSuccessor']], on='SalesModelCode', how='left')
    out['Predecessor'] = out['Predecessor'].apply(lambda v: v if isinstance(v, list) else [])
    out['Successor'] = out['Successor'].apply(lambda v: v if isinstance(v, list) else [])
    out['HasPredecessor'] = out['HasPredecessor'].map(lambda v: v is True)
    out['HasSuccessor'] = out['HasSuccessor'].map(lambda v: v is True)
    out['first_entry'] = out['SalesModelCode'].map(first_entry_by_code)
    out['last_entry'] = out['SalesModelCode'].map(last_entry_by_code)
    out = out.merge(train_ranges, on='SalesModelCode', how='left')

    has_ps = out['ps_start'].notna()
    has_pl = out['pl_start'].notna()
    # Only matters in the has_ps=False branch, so this must read Product Lifecycle's OWN
    # discontinued flag, not ps_discontinued (which is only ever populated when has_ps=True -
    # the opposite case, always NaN/missing here, so gating on it would silently disable the
    # whole PL-fallback branch below rather than restrict it).
    pl_is_discontinued = out['pl_discontinued'].map(lambda v: v is True)

    # effective_start: Product Successor's own claimed start if it has one; otherwise the
    # earlier of train5's own first_entry and Product Lifecycle's first_active_month (see the
    # np.fmin note above - this is unconditional on discontinued status, since a start date is
    # informative regardless of whether the product has since ended).
    fallback_start = np.fmin(out['first_entry'].to_numpy(dtype='float64'), out['pl_start'].to_numpy(dtype='float64'))
    fallback_start = np.maximum(fallback_start, floor)
    out['effective_start'] = np.where(has_ps, out['ps_start'], fallback_start)
    out['start_source'] = np.where(has_ps, 'product_successor', 'first_entry_or_pl_active_min')

    # effective_end: Product Successor's own claimed end if it has one; otherwise Product
    # Lifecycle's end IS ONLY trusted when Product Lifecycle itself confirms the code is
    # discontinued (a real, complete start-to-end lifecycle) - a PL entry that's still active
    # tells us nothing about when/whether this code will end, so it falls to the max-settled
    # default just like a code with no PL entry at all. Also maxed against train_end (settled
    # demand's own last month, capped at max_settled_idx by _build_train_ranges) - empirically,
    # EVERY code on this fallback path had real settled demand recorded after max(last_entry,
    # pl_end) alone, by anywhere from 1 to 37 months, so last_entry/pl_end alone were too early.
    fallback_end = np.fmax(np.fmax(out['last_entry'], out['pl_end']), out['train_end'])
    out['effective_end'] = np.select([has_ps, (~has_ps) & has_pl & pl_is_discontinued],
        [out['ps_end'], fallback_end], default=max_settled_idx)
    out['end_source'] = np.select([has_ps, (~has_ps) & has_pl & pl_is_discontinued],
        ['product_successor', 'last_entry_or_pl_end_max'], default='max_settled_no_info')
    is_conflicted = out['status_conflict'].map(lambda v: v is True)
    out.loc[is_conflicted, 'end_source'] = 'max_settled_ps_pl_conflict'

    # Sanity check: effective_start for non-PS codes is min(first_entry, pl_start), so it should
    # be impossible for real demand to exist before it (first_entry IS the first-ever order by
    # definition) - if this fires, that's a real bug worth investigating.
    date_mismatches = []
    for _, r in out[out['start_source'] == 'first_entry_or_pl_active_min'].iterrows():
        code = r['SalesModelCode']
        code_orders = train_ranges[(train_ranges['SalesModelCode'] == code) &
            (train_ranges['train_start'] < r['effective_start'])]
        if len(code_orders):
            date_mismatches.append({'check': 'UNEXPECTED_demand_before_min_start',
                'old_key': code, 'new_key': 'first_entry_or_pl_active_min',
                'detail': f"real settled demand found before effective_start="
                          f"{_yyyymm(r['effective_start'])} despite using min(first_entry, pl_start) "
                          "- this should not be possible, investigate"})

    # Symmetric sanity check for the end side: fallback_end already maxes in train_end, so this
    # should never fire - if it does, that's a real bug worth investigating.
    for _, r in out[out['end_source'] == 'last_entry_or_pl_end_max'].iterrows():
        code = r['SalesModelCode']
        code_orders = train_ranges[(train_ranges['SalesModelCode'] == code) &
            (train_ranges['train_end'] > r['effective_end'])]
        if len(code_orders):
            date_mismatches.append({'check': 'UNEXPECTED_demand_after_max_end',
                'old_key': code, 'new_key': 'last_entry_or_pl_end_max',
                'detail': f"real settled demand found after effective_end="
                          f"{_yyyymm(r['effective_end'])} despite using max(last_entry, pl_end, "
                          "train_end) - this should not be possible, investigate"})

    out['IsCensored'] = out['effective_start'] <= (floor + censor_grace_months)

    n_per_code = out.groupby('SalesModelCode').size()
    assert (n_per_code == 1).all(), \
        f"expected exactly one row per SalesModelCode, found duplicates: {n_per_code[n_per_code > 1].index.tolist()}"
    assert out['effective_start'].notna().all() and out['effective_end'].notna().all(), \
        "every SalesModelCode must have a resolved effective_start/effective_end (train5 fallback should guarantee this)"

    return out, date_mismatches


def resolve_effective_ranges(orders: pd.DataFrame, actual_order: pd.DataFrame,
        sales_model_codes, replacement_xlsx_path: str,
        max_settled_target: int, censor_grace_months: int = 6) -> dict:
    """One effective (start, end) per SalesModelCode: Product Successor, then Product Lifecycle, then
    the order book."""
    codes = set(sales_model_codes)
    floor = compute_dataset_floor(orders)
    first_entry_by_code = orders.groupby('SalesModelCode')['EntryMonthIdx'].min()
    last_entry_by_code = orders.groupby('SalesModelCode')['EntryMonthIdx'].max()
    max_settled_idx = to_month_idx(max_settled_target)

    xl = pd.ExcelFile(replacement_xlsx_path)
    succ_raw = xl.parse('Product Successor')
    lc_raw = xl.parse('Product Lifecycle')

    succ, lc, cleaning_flags = _clean_replacement_sheets(succ_raw, lc_raw)
    code_table, pred_status, succ_status, within_sheet_mismatches = \
        _build_ps_ranges(succ, max_settled_target, max_settled_idx)
    pl_ranges, pl_mismatches = _build_pl_ranges(lc, codes, replacement_xlsx_path, max_settled_idx)
    train_ranges = _build_train_ranges(actual_order, max_settled_idx)

    # Resolve PS-vs-PL status conflicts FIRST - this corrects code_table's and pl_ranges' own
    # discontinued/end values, so every check and output built from them below (the boundary
    # check, the crosswalk, effective_ranges) already reflects the correction.
    code_table, pl_ranges, status_conflicts = _find_status_conflicts(code_table, pl_ranges, max_settled_idx)

    ps_ranges, ps_mismatches = _reject_ps_contradicted(code_table, codes, pl_ranges, train_ranges)

    # The predecessor/successor crosswalk - every (old_key, new_key) pair implied by code_table's
    # Successor lists (real relationships only: self-match/placeholder already excluded). NOT
    # restricted to codes whose PS date survived _reject_ps_contradicted - that check validates
    # whether a specific code's date is trustworthy, not whether the relationship itself is real
    # (e.g. "does this successor have a known predecessor" doesn't require the predecessor's own
    # date to have survived that check).
    crosswalk = code_table[code_table['Successor'].map(len) > 0][
        ['SalesModelCode', 'Successor', 'ps_discontinued']].explode('Successor').rename(
        columns={'SalesModelCode': 'old_key', 'Successor': 'new_key', 'ps_discontinued': 'pred_discontinued'})
    crosswalk['succ_discontinued'] = crosswalk['new_key'].map(
        code_table.set_index('SalesModelCode')['ps_discontinued'])
    crosswalk = crosswalk.reset_index(drop=True)

    out, assemble_mismatches = _assemble_effective_ranges(
        codes, ps_ranges, pl_ranges, code_table, first_entry_by_code, last_entry_by_code,
        train_ranges, floor, max_settled_idx, censor_grace_months)

    date_mismatches = within_sheet_mismatches + ps_mismatches + pl_mismatches + assemble_mismatches

    # Codes with no appearance anywhere in Product Successor at all (not even a self-match or
    # placeholder row) - code_table only ever has a row for a code mentioned in some role, so
    # this is just the modeling universe minus whatever code_table covers.
    not_in_product_successor = pd.DataFrame(
        {'SalesModelCode': sorted(codes - set(code_table['SalesModelCode']))})

    return {
        'effective_ranges': out,
        'crosswalk': crosswalk,
        'cleaning_flags': cleaning_flags,
        'date_mismatches': pd.DataFrame(date_mismatches),
        'pred_status': pred_status,
        'succ_status': succ_status,
        'status_conflicts': status_conflicts,
        'not_in_product_successor': not_in_product_successor,
    }

