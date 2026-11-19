"""Product lifecycle / cold-start feature family; launch features always read a code's OWN dates -
a predecessor's history must never be spliced into a successor's own age."""
import logging

import numpy as np
import pandas as pd

from .features import safe_ratio, entity_monthly_actual

log = logging.getLogger(__name__)

# Each row's age for FORECAST ELIGIBILITY (min_origin_months_since_launch) - carried on every
# training and inference row, but deliberately in no feature family, so never a model input.
FORECAST_ELIGIBILITY_AGE_COL = 'ForecastEligibilityAge'


def compute_dataset_floor(order_book: pd.DataFrame) -> int:
    """Earliest EntryMonthIdx in the order book - the left-censoring boundary."""
    return int(order_book['EntryMonthIdx'].min())


def resolve_eligibility_launch(launch_by_group: pd.Series, first_entry_by_group: pd.Series,
        dataset_floor_idx: int) -> pd.Series:
    """Eligibility launch month: TFGMaterial launch, else first order entry (-inf if at the dataset
    floor). Does not change `launch_by_group`."""
    first_entry = first_entry_by_group.reindex(launch_by_group.index).astype('float64')
    undated = launch_by_group.isna()
    predates_data = undated & (first_entry <= dataset_floor_idx)
    from_first_order = undated & (first_entry > dataset_floor_idx)
    launch = launch_by_group.astype('float64')
    launch[from_first_order] = first_entry[from_first_order]
    launch[predates_data] = -np.inf
    if from_first_order.any():
        log.warning("%d PlanModelCode(s) have no TFGMaterial launch date - forecast eligibility "
            "counts their age from their first order instead: %s", int(from_first_order.sum()),
            sorted(launch.index[from_first_order.to_numpy()].astype(str))[:10])
    if predates_data.any():
        log.info("%d PlanModelCode(s) have no TFGMaterial launch date but were already ordered "
            "when the data starts - eligible as mature: %s", int(predates_data.sum()),
            sorted(launch.index[predates_data.to_numpy()].astype(str))[:10])
    return launch


def attach_forecast_eligibility_age(panel: pd.DataFrame, eligibility_launch_by_group: pd.Series,
        group_col: str = 'PlanModelCode') -> pd.DataFrame:
    """FORECAST_ELIGIBILITY_AGE_COL = OriginIdx minus the eligibility launch month."""
    panel = panel.copy()
    launch = panel[group_col].map(eligibility_launch_by_group).astype('float64')
    panel[FORECAST_ELIGIBILITY_AGE_COL] = panel['OriginIdx'].to_numpy(dtype='float64') - launch.to_numpy()
    return panel


_warned_legacy_frame = False


def meets_min_launch_age(df: pd.DataFrame, min_months) -> np.ndarray:
    """The single forecast-eligibility test shared by inference and backtests: age >= `min_months`.
    Older panels without the column fall back to OriginMonthsSinceLaunch."""
    global _warned_legacy_frame
    if FORECAST_ELIGIBILITY_AGE_COL in df.columns:
        return (df[FORECAST_ELIGIBILITY_AGE_COL] >= min_months).to_numpy()
    if 'OriginMonthsSinceLaunch' not in df.columns:
        raise ValueError(f"frame has neither {FORECAST_ELIGIBILITY_AGE_COL!r} nor "
            "'OriginMonthsSinceLaunch' - can't apply min_origin_months_since_launch.")
    if not _warned_legacy_frame:
        log.warning("Frame has no %r column (built before it existed) - applying the old "
            "OriginMonthsSinceLaunch >= %s test, which excludes every code with no TFGMaterial "
            "launch date. Rebuild the panel to use the current eligibility rule.",
            FORECAST_ELIGIBILITY_AGE_COL, min_months)
        _warned_legacy_frame = True
    return (df['OriginMonthsSinceLaunch'] >= min_months).to_numpy()


def _attach_launch_features(panel, group_col, launch_by_group, prefix, end_by_group=None):
    """Origin/TargetMonthsSinceLaunch; null when TargetIdx is outside [launch, end]."""
    launch = panel[group_col].map(launch_by_group)
    panel[f'{prefix}OriginMonthsSinceLaunch'] = (panel['OriginIdx'] - launch).astype('float64')
    panel[f'{prefix}TargetMonthsSinceLaunch'] = (panel['TargetIdx'] - launch).astype('float64')
    outside = panel['TargetIdx'] < launch
    if end_by_group is not None:
        outside |= panel['TargetIdx'] > panel[group_col].map(end_by_group)
    panel.loc[outside, [f'{prefix}OriginMonthsSinceLaunch', f'{prefix}TargetMonthsSinceLaunch']] = np.nan
    panel[f'{prefix}InLifecycle'] = ~outside  # not a feature - filters scored rows (folds.scoreable_rows)
    return panel


def _attach_first_demand_features(panel, group_col, launch_by_group, first_demand_by_group, prefix):
    """MonthsSinceFirstDemand and MonthsFromLaunchToFirstDemand; null before first demand."""
    launch = panel[group_col].map(launch_by_group)
    first_demand = panel[group_col].map(first_demand_by_group)

    months_since = (panel['OriginIdx'] - first_demand).astype('float64')
    not_yet_or_unknown = launch.isna() | first_demand.isna() | (panel['OriginIdx'] < first_demand)
    months_since[not_yet_or_unknown] = np.nan
    panel[f'{prefix}MonthsSinceFirstDemand'] = months_since

    # Static per-entity gap, but must still be gated on OriginIdx >= first_demand: before first
    # demand has actually happened, "how long from launch to first demand" isn't knowable yet -
    # it would leak the future timing of that first order into every pre-demand row.
    from_launch = (first_demand - launch).astype('float64')
    from_launch[not_yet_or_unknown] = np.nan
    panel[f'{prefix}MonthsFromLaunchToFirstDemand'] = from_launch
    return panel


def _attach_replacement_flags(panel, group_col, has_predecessor_by_group, successor_start_by_group, prefix):
    """HasPredecessor (static) and HasSuccessor (true once OriginIdx reaches the successor's start)."""
    panel[f'{prefix}HasPredecessor'] = panel[group_col].map(has_predecessor_by_group).fillna(False).astype(int)
    successor_start = panel[group_col].map(successor_start_by_group)
    panel[f'{prefix}HasSuccessor'] = (panel['OriginIdx'] >= successor_start).astype(int)
    return panel


def _attach_consecutive_months_with_demand(unique_oc, monthly, group_col, prefix):
    """ConsecutiveMonthsWithDemand: uncapped run of settled-demand months ending at Origin."""
    m = monthly.sort_values([group_col, 'RequestMonthIdx']).reset_index(drop=True)
    has_demand = m['ActualOrder'] > 0
    streak_group = (~has_demand).groupby(m[group_col]).cumsum()
    run_length = np.where(has_demand, m.groupby([group_col, streak_group]).cumcount() + 1, 0)
    m[f'{prefix}ConsecutiveMonthsWithDemand'] = run_length
    return unique_oc.merge(
        m[[group_col, 'RequestMonthIdx', f'{prefix}ConsecutiveMonthsWithDemand']]
            .rename(columns={'RequestMonthIdx': 'OriginIdx'}),
        on=[group_col, 'OriginIdx'], how='left')


def _attach_months_since_last_demand(unique_oc, monthly, group_col, prefix):
    """MonthsSinceLastDemand as of Origin; NaN before first-ever demand."""
    m = monthly.sort_values([group_col, 'RequestMonthIdx']).reset_index(drop=True)
    has_demand = m['ActualOrder'] > 0
    last_demand_month = m['RequestMonthIdx'].where(has_demand).groupby(m[group_col]).ffill()
    m[f'{prefix}MonthsSinceLastDemand'] = m['RequestMonthIdx'] - last_demand_month
    return unique_oc.merge(
        m[[group_col, 'RequestMonthIdx', f'{prefix}MonthsSinceLastDemand']]
            .rename(columns={'RequestMonthIdx': 'OriginIdx'}),
        on=[group_col, 'OriginIdx'], how='left')


def _attach_group_relative_demand_features(panel, unique_oc, group_col, prefix, group_share_prefix,
        windows=(3, 6, 12)):
    """RecentDemandShare{w}: entity's share of its PartCode family's recent demand; needs the
    Group_ history features on `panel`."""
    max_w = max(windows)
    group_lag_cols = [f'{group_share_prefix}Lag{k}' for k in range(max_w)]
    own_lag_cols = [f'{prefix}Lag{k}' for k in range(max_w)]
    # GrowthRateDiff (and its Lag0Lag1Ratio inputs) not in the reference panel - added after it was
    # built, not needed to reproduce/compare against it. Lag0Lag1Ratio computation is disabled
    # in attach_history_features (features.py), kept in sync with this.
    # growth_cols = [f'{prefix}Lag0Lag1Ratio', f'{group_share_prefix}Lag0Lag1Ratio']
    growth_cols = []
    missing = [c for c in group_lag_cols + own_lag_cols + growth_cols if c not in panel.columns]
    if missing:
        raise ValueError(
            f"group_share_prefix={group_share_prefix!r} but {missing} not found on panel - "
            "run both attach_history_features calls (own AND Group_-level) before "
            "attach_lifecycle_features.")
    share_oc = panel[[group_col, 'OriginIdx'] + own_lag_cols + group_lag_cols + growth_cols] \
        .drop_duplicates(subset=[group_col, 'OriginIdx']).reset_index(drop=True)

    new_col_names = []
    for w in windows:
        own_recent = share_oc[own_lag_cols[:w]].mean(axis=1, skipna=True)
        group_recent = share_oc[group_lag_cols[:w]].mean(axis=1, skipna=True)
        col = f'{prefix}RecentDemandShare{w}'
        share_oc[col] = safe_ratio(own_recent.to_numpy(), group_recent.to_numpy())
        new_col_names.append(col)

    # growth_col = f'{prefix}GrowthRateDiff'
    # share_oc[growth_col] = share_oc[growth_cols[0]] - share_oc[growth_cols[1]]
    # new_col_names.append(growth_col)

    unique_oc = unique_oc.merge(
        share_oc[[group_col, 'OriginIdx'] + new_col_names],
        on=[group_col, 'OriginIdx'], how='left')
    return unique_oc, new_col_names


def attach_lifecycle_features(panel: pd.DataFrame, actual_order: pd.DataFrame,
        launch_by_group: pd.Series, first_actual_order_by_group: pd.Series,
        group_col: str = 'PlanModelCode', prefix: str = '', group_share_prefix: str = None,
        pred_succ_flags: pd.DataFrame = None, end_by_group: pd.Series = None) -> pd.DataFrame:
    """Orchestrates the lifecycle feature groups above. `pred_succ_flags` is PlanModelCode-only,
    so `group_col` must be 'PlanModelCode' whenever it's given."""
    panel = panel.copy()
    panel = _attach_launch_features(panel, group_col, launch_by_group, prefix, end_by_group)

    if pred_succ_flags is not None:
        flags = pred_succ_flags.set_index(group_col)
        panel = _attach_replacement_flags(panel, group_col, flags['HasPredecessor'], flags['SuccessorStart'], prefix)

    unique_oc = panel[[group_col, 'OriginIdx']].drop_duplicates().reset_index(drop=True)
    max_origin_by_group = unique_oc.groupby(group_col)['OriginIdx'].max()
    monthly = entity_monthly_actual(actual_order, group_col, first_actual_order_by_group, max_origin_by_group)
    unique_oc = _attach_consecutive_months_with_demand(unique_oc, monthly, group_col, prefix)
    unique_oc = _attach_months_since_last_demand(unique_oc, monthly, group_col, prefix)
    new_cols = [f'{prefix}ConsecutiveMonthsWithDemand', f'{prefix}MonthsSinceLastDemand']
    if group_share_prefix is not None:
        unique_oc, share_cols = _attach_group_relative_demand_features(
            panel, unique_oc, group_col, prefix, group_share_prefix)
        new_cols += share_cols

    return panel.merge(unique_oc[[group_col, 'OriginIdx'] + new_cols], on=[group_col, 'OriginIdx'], how='left')
