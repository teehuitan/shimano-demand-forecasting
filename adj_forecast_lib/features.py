"""Feature-family builders for the order-adjustment forecasting panel; anchor choice (Origin vs
Target) is what keeps each feature leakage-safe for its horizon."""
import warnings

import numpy as np
import pandas as pd

from .config import (
    month_idx_to_yyyymm, is_event_month_idx, is_known_event_idx, is_covid_month_idx, is_cny_month_idx,
    months_to_cny,
)
from .order_book import (
    asof_month_idx, asof_points, query_known_order, resolve_known_order_gaps, shift_asof,
)


def entity_monthly_actual(actual_order: pd.DataFrame, group_col: str,
        first_entry_by_group: pd.Series, max_month_by_group: pd.Series) -> pd.DataFrame:
    """Per-entity gapless monthly settled ActualOrder (absent month = 0), first entry to ceiling."""
    settled = actual_order.groupby([group_col, 'RequestMonthIdx'])['ActualOrder'].sum()
    frames = []
    for group, first_entry in first_entry_by_group.items():
        max_month = max_month_by_group.get(group)
        if max_month is None or pd.isna(first_entry):
            continue
        months = np.arange(int(first_entry), int(max_month) + 1)
        qty = settled.reindex(pd.MultiIndex.from_product([[group], months])).fillna(0.0).to_numpy()
        frames.append(pd.DataFrame({group_col: group, 'RequestMonthIdx': months, 'ActualOrder': qty}))
    if not frames:
        return pd.DataFrame(columns=[group_col, 'RequestMonthIdx', 'ActualOrder'])
    return pd.concat(frames, ignore_index=True)


def safe_ratio(numer, denom, eps=1e-6):
    """Returns NaN (XGBoost-native missing) rather than inf/0 on near-zero denominators."""
    numer = np.asarray(numer, dtype='float64')
    denom = np.asarray(denom, dtype='float64')
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numer / denom
    result = np.where(np.abs(denom) < eps, np.nan, result)
    return result


def attach_calendar_features(df: pd.DataFrame, month_idx_col: str, prefix: str = '',
        include_cny: bool = True) -> pd.DataFrame:
    """Year, Month, Month_sin/cos, Quarter (+ CNY features if include_cny) for a month-idx column."""
    df = df.copy()
    target_idx = df[month_idx_col].to_numpy()
    yyyymm = month_idx_to_yyyymm(target_idx)
    year, month = yyyymm // 100, yyyymm % 100
    df[f'{prefix}Year'] = year
    df[f'{prefix}Month'] = month
    df[f'{prefix}Month_sin'] = np.sin(2 * np.pi * month / 12)
    df[f'{prefix}Month_cos'] = np.cos(2 * np.pi * month / 12)
    df[f'{prefix}Quarter'] = (month - 1) // 3 + 1

    if include_cny:
        df[f'{prefix}Is_CNY_Month'] = is_cny_month_idx(target_idx).astype(int)
        df[f'{prefix}Months_to_CNY'] = months_to_cny(target_idx)
    return df


def attach_history_features(skeleton: pd.DataFrame, order_book: pd.DataFrame,
    first_entry_by_group: pd.Series, group_col: str = 'PlanModelCode',
    max_lag: int = 24, prefix: str = '', zscore_windows: tuple = (12,)) -> pd.DataFrame:
    # zscore_windows was (3, 6, 12) - 3/6 not in the reference panel (README §7), added after it;
    # restore the full tuple to bring ZScore3/ZScore6 back.
    """Origin-anchored lags, ratios, trailing means, trend, volatility, range and z-scores."""
    unique_oc = skeleton[[group_col, 'OriginIdx']].drop_duplicates().reset_index(drop=True)

    # --- old Lag1..Lag24 window, kept for tracking (superseded by Lag0..Lag23 below) ---
    # lag_q = unique_oc.merge(pd.DataFrame({'k': range(1, max_lag + 1)}), how='cross')

    lag_q = unique_oc.merge(pd.DataFrame({'k': range(0, max_lag)}), how='cross')
    lag_q['RequestMonthIdx'] = lag_q['OriginIdx'] - lag_q['k']
    lag_q['CutoffMonthIdx'] = lag_q['OriginIdx']

    raw = query_known_order(lag_q[[group_col, 'RequestMonthIdx', 'CutoffMonthIdx']],
        order_book, [group_col])
    first_entry = lag_q[group_col].map(first_entry_by_group).to_numpy()
    lag_q['KnownOrder'] = resolve_known_order_gaps(raw, lag_q['RequestMonthIdx'].to_numpy(), first_entry)

    lags_wide = lag_q.pivot(index=[group_col, 'OriginIdx'], columns='k', values='KnownOrder')
    lags_wide.columns = [f'{prefix}Lag{k}' for k in lags_wide.columns]
    lags_wide = lags_wide.reset_index()

    # lag_cols = [f'{prefix}Lag{k}' for k in range(1, max_lag + 1)]  # old, k=1..24
    lag_cols = [f'{prefix}Lag{k}' for k in range(0, max_lag)]  # new, k=0..23
    L = lags_wide[lag_cols].to_numpy(dtype='float64')

    # A young product's very first origin(s) can have ALL lags NaN (nothing exists to
    # aggregate yet) - numpy's nanmean/nanstd correctly return NaN there but warn about it;
    # the warning is cosmetic (expected, not a bug), suppressed for clean notebook output.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        trailmean12_level = np.nanmean(L[:, 0:12], axis=1)
        recent3 = np.nanmean(L[:, 0:3], axis=1)
        prior6 = np.nanmean(L[:, 3:9], axis=1)
        lags_wide[f'{prefix}Recent3_vs_Prior6'] = safe_ratio(recent3, prior6)
        # base: plain trailing 6 months ending at Origin (Lag0-Lag5) - deliberately short/recent
        # rather than a full-year average, so the "_vs_base" ratios stay sensitive to near-term with this
        # shifts. Must never be the same window as its own numerator 
        base = np.nanmean(L[:, 0:6], axis=1)
        # TrailMean3/6/12 (raw levels) not in the reference panel - added after it was built, not
        # needed to reproduce/compare against it. recent3/base/trailmean12_level still computed
        # above regardless, for the _vs_base ratios below.
        # lags_wide[f'{prefix}TrailMean3'] = recent3
        # lags_wide[f'{prefix}TrailMean6'] = base
        # lags_wide[f'{prefix}TrailMean12'] = trailmean12_level
        lags_wide[f'{prefix}TrailMean3_vs_base'] = safe_ratio(recent3, base)
        lags_wide[f'{prefix}TrailMean12_vs_base'] = safe_ratio(trailmean12_level, base)
        lags_wide[f'{prefix}TrendSlope12_vs_base'] = safe_ratio(_ols_slope(L[:, 0:12]), base)
        lags_wide[f'{prefix}Volatility6'] = safe_ratio(np.nanstd(L[:, 0:6], axis=1), np.nanmean(L[:, 0:6], axis=1))
        lags_wide[f'{prefix}Range12_vs_base'] = safe_ratio(
            np.nanmax(L[:, 0:12], axis=1) - np.nanmin(L[:, 0:12], axis=1), base)
        for w in zscore_windows:
            lags_wide[f'{prefix}ZScore{w}'] = safe_ratio(
                L[:, 0] - np.nanmean(L[:, 0:w], axis=1), np.nanstd(L[:, 0:w], axis=1))
        # Fastest-reacting signal available - ratio+diff pair for the same near-zero-divisor
        # reason as Momentum_k/Accumulation_k (attach_known_order_features).
        # Lag0Lag1Ratio/Diff not in the reference panel - added after it was built, not needed to
        # reproduce/compare against it. GrowthRateDiff (lifecycle_features.py) depends on
        # Lag0Lag1Ratio - its computation is disabled there too, kept in sync with this.
        # lags_wide[f'{prefix}Lag0Lag1Ratio'] = safe_ratio(L[:, 0], L[:, 1])
        # lags_wide[f'{prefix}Lag0Lag1Diff'] = L[:, 0] - L[:, 1]

    return skeleton.merge(lags_wide, on=[group_col, 'OriginIdx'], how='left')


def attach_trend_stability_features(panel: pd.DataFrame, group_col: str = 'PlanModelCode',
        max_run: int = 11, threshold: float = 0.25, prefix: str = '') -> pd.DataFrame:
    """StableRunLength: months back from Lag0 with MoM change within `threshold`; needs the Lag columns."""
    lag_cols = [f'{prefix}Lag{k}' for k in range(max_run + 2)]
    unique_oc = panel[[group_col, 'OriginIdx'] + lag_cols].drop_duplicates(subset=[group_col, 'OriginIdx']) \
        .reset_index(drop=True)
    L = unique_oc[lag_cols].to_numpy(dtype='float64')

    cur, prev = L[:, :-1], L[:, 1:]
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_change = np.abs(cur - prev) / np.abs(prev)
    # 0/0 (both months genuinely zero, common for intermittent demand) is a real no-change, not
    # missing data - only an actual NaN lag (young entity, nothing to compare) breaks the run.
    both_zero = (cur == 0) & (prev == 0)
    is_missing = np.isnan(cur) | np.isnan(prev)
    is_stable_step = np.where(is_missing, False, np.where(both_zero, True, pct_change <= threshold))

    run_length = np.zeros(len(unique_oc), dtype=int)
    still_running = np.ones(len(unique_oc), dtype=bool)
    for k in range(is_stable_step.shape[1]):
        still_running &= is_stable_step[:, k]
        run_length += still_running.astype(int)

    unique_oc[f'{prefix}StableRunLength'] = run_length
    return panel.merge(unique_oc[[group_col, 'OriginIdx', f'{prefix}StableRunLength']],
        on=[group_col, 'OriginIdx'], how='left')


def attach_adi_features(panel: pd.DataFrame, actual_order: pd.DataFrame,
    first_entry_by_group: pd.Series, group_col: str = 'PlanModelCode',
    max_lag: int = 24, prefix: str = '') -> pd.DataFrame:
    """Rolling and expanding ADI on settled ActualOrder; must run after attach_history_features."""
    lag_cols = [f'{prefix}Lag{k}' for k in range(max_lag)]
    unique_oc = panel[[group_col, 'OriginIdx'] + lag_cols].drop_duplicates(subset=[group_col, 'OriginIdx']) \
        .sort_values([group_col, 'OriginIdx']).reset_index(drop=True)

    L = unique_oc[lag_cols].to_numpy(dtype='float64')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        for months in (3, 6, 12, 24):
            w = L[:, 0:months]
            observed = ~np.isnan(w)
            n_nonzero = (observed & (w > 0)).sum(axis=1)
            unique_oc[f'{prefix}Rolling{months}ADI'] = safe_ratio(observed.sum(axis=1), n_nonzero)

    max_origin_by_group = unique_oc.groupby(group_col)['OriginIdx'].max()
    monthly = entity_monthly_actual(actual_order, group_col, first_entry_by_group, max_origin_by_group)
    monthly = monthly.sort_values([group_col, 'RequestMonthIdx']).reset_index(drop=True)
    has_demand = monthly['ActualOrder'] > 0
    n_periods_exp = monthly.groupby(group_col).cumcount() + 1
    n_nonzero_exp = has_demand.groupby(monthly[group_col]).cumsum()
    monthly[f'{prefix}ExpandingADI'] = safe_ratio(n_periods_exp.to_numpy(), n_nonzero_exp.to_numpy())

    unique_oc = unique_oc.merge(
        monthly[[group_col, 'RequestMonthIdx', f'{prefix}ExpandingADI']]
            .rename(columns={'RequestMonthIdx': 'OriginIdx'}),
        on=[group_col, 'OriginIdx'], how='left')

    out_cols = ([group_col, 'OriginIdx'] + [f'{prefix}Rolling{m}ADI' for m in (3, 6, 12, 24)]
        + [f'{prefix}ExpandingADI'])
    return panel.merge(unique_oc[out_cols], on=[group_col, 'OriginIdx'], how='left')


def attach_known_order_features(panel: pd.DataFrame, order_book_code: pd.DataFrame,
        first_entry_by_code: pd.Series, group_col: str = 'PlanModelCode',
        asof_col: str = None, arrival_col: str = 'EntryMonthIdx') -> pd.DataFrame:
    """KnownOrder(as-of, Target) and derived accumulation/momentum/YoY. For day grain pass
    asof_col=CUTOFF_DATE_COL with arrival_col='EntryDate'."""
    panel = panel.copy()
    # The as-of point every query below is anchored on. Read once so the main query, the offsets
    # and the YoY block can't drift onto different anchors - and read through order_book's
    # asof_points, the same definition KnownCanceledOrder uses, so Net = Known - Cancelled is
    # always two numbers taken at one moment.
    asof_values = asof_points(panel, asof_col)

    def _query(request_month_idx, asof):
        """A (group, target-month, as-of) batch - RequestMonthIdx stays a month ordinal on both
        paths, since forecast targets are monthly whatever the as-of grain is."""
        qq = panel[[group_col]].copy()
        qq['RequestMonthIdx'] = request_month_idx
        qq['CutoffMonthIdx'] = asof
        return query_known_order(qq, order_book_code, [group_col], arrival_col=arrival_col)

    target_idx = panel['TargetIdx'].to_numpy()
    panel['KnownOrder'] = np.nan_to_num(_query(target_idx, asof_values), nan=0.0)

    first_entry = panel[group_col].map(first_entry_by_code).to_numpy()

    def _known_offset(months):
        shifted = shift_asof(asof_values, months)
        raw_o = _query(target_idx, shifted)
        return resolve_known_order_gaps(raw_o, asof_month_idx(shifted), first_entry)

    panel['KnownOrder_prevOrigin'] = _known_offset(1)
    panel['Accumulation_1mo'] = panel['KnownOrder'] - panel['KnownOrder_prevOrigin']
    panel['AccumulationPct_1mo'] = safe_ratio(panel['Accumulation_1mo'], panel['KnownOrder_prevOrigin'])

    for k in (3, 6):
        panel[f'KnownOrder_{k}moAgo'] = _known_offset(k)
        panel[f'Momentum_{k}mo'] = safe_ratio(panel['KnownOrder'], panel[f'KnownOrder_{k}moAgo'])
        # raw-difference companion to Momentum_{k}mo - behaves better than a ratio when
        # KnownOrder_{k}moAgo is near zero (common for intermittent demand).
        panel[f'Accumulation_{k}mo'] = panel['KnownOrder'] - panel[f'KnownOrder_{k}moAgo']

    # Seasonal year-on-year: KnownOrder(N,Target) / KnownOrder(N-12,Target-12) - shifts BOTH
    # as-of point and target back exactly 12 months (unlike Momentum_k, which holds Target fixed).
    # Compares the pipeline right now against the analogous point last year for the same
    # calendar month, controlling for seasonality - a distinct signal from Momentum_k or
    # HistVisibility_SameMonth (an averaged ratio-to-eventual, not a raw YoY level comparison).
    # On the day path "the analogous point" is the same day of month a year back, not last year's
    # month-end, which is exactly what shift_asof gives and what makes the ratio comparable.
    yoy_asof = shift_asof(asof_values, 12)
    yoy_raw = _query(target_idx - 12, yoy_asof)
    panel['KnownOrder_YoY_prior'] = resolve_known_order_gaps(
        yoy_raw, asof_month_idx(yoy_asof), first_entry)
    panel['KnownOrder_YoY'] = safe_ratio(panel['KnownOrder'], panel['KnownOrder_YoY_prior'])

    panel['KnownOrder_x_Horizon'] = panel['KnownOrder'] * panel['Horizon']
    return panel


def attach_historical_visibility(panel: pd.DataFrame, order_book: pd.DataFrame,
    first_entry_by_group: pd.Series, group_col: str = 'PlanModelCode',
    prefix: str = '', lookback_months: int = 6,
    same_month_years: int = None) -> pd.DataFrame:
    """Average KnownOrder(M'-h,M') / KnownOrder(N,M') over prior months M' < N: bounded lookback
    and same calendar month."""
    if same_month_years is None:
        span_months = int(panel['OriginIdx'].max() - order_book['EntryMonthIdx'].min())
        same_month_years = span_months // 12 + 1

    unique_og = panel[[group_col, 'OriginIdx']].drop_duplicates().reset_index(drop=True)
    oh = unique_og.merge(pd.DataFrame({'Horizon': range(1, 7)}), how='cross')
    oh['TargetIdx'] = oh['OriginIdx'] + oh['Horizon']

    # (a) All-history, bounded lookback
    all_hist_cand = oh.merge(pd.DataFrame({'m': range(1, lookback_months + 1)}), how='cross')
    all_hist_cand['Mprime'] = all_hist_cand['OriginIdx'] - all_hist_cand['m']
    all_hist_cand = all_hist_cand[~is_covid_month_idx(all_hist_cand['Mprime'])]
    all_hist_feat = _visibility_ratio_and_average(
        all_hist_cand, order_book, group_col, first_entry_by_group,
        out_col=f'{prefix}HistVisibility_AllHistory')

    # (b) Same-calendar-month, matched years back
    same_month_cand = oh.merge(pd.DataFrame({'y': range(1, same_month_years + 1)}), how='cross')
    same_month_cand['Mprime'] = same_month_cand['TargetIdx'] - 12 * same_month_cand['y']
    same_month_cand = same_month_cand[same_month_cand['Mprime'] < same_month_cand['OriginIdx']]
    same_month_cand = same_month_cand[~is_covid_month_idx(same_month_cand['Mprime'])]
    same_month_feat = _visibility_ratio_and_average(
        same_month_cand, order_book, group_col, first_entry_by_group,
        out_col=f'{prefix}HistVisibility_SameMonth')

    panel = panel.merge(all_hist_feat, on=[group_col, 'OriginIdx', 'Horizon'], how='left')
    panel = panel.merge(same_month_feat, on=[group_col, 'OriginIdx', 'Horizon'], how='left')
    return panel


def _visibility_ratio_and_average(cand: pd.DataFrame, order_book: pd.DataFrame, group_col: str,
    first_entry_by_group: pd.Series, out_col: str) -> pd.DataFrame:
    """Visibility ratio per candidate row, averaged per (group_col, OriginIdx, Horizon)."""
    cand = cand[cand['Mprime'] < cand['OriginIdx']].copy()  # strictly prior, both variants

    denom_q = cand.rename(columns={'Mprime': 'RequestMonthIdx', 'OriginIdx': 'CutoffMonthIdx'})
    denom_raw = query_known_order(denom_q[[group_col, 'RequestMonthIdx', 'CutoffMonthIdx']],
        order_book, [group_col])
    fe_denom = denom_q[group_col].map(first_entry_by_group).to_numpy()
    cand['Denom'] = resolve_known_order_gaps(denom_raw, denom_q['RequestMonthIdx'].to_numpy(), fe_denom)

    num_q = cand.copy()
    num_q['RequestMonthIdx'] = num_q['Mprime']
    num_q['CutoffMonthIdx'] = num_q['Mprime'] - num_q['Horizon']
    num_raw = query_known_order(num_q[[group_col, 'RequestMonthIdx', 'CutoffMonthIdx']],
        order_book, [group_col])
    fe_num = num_q[group_col].map(first_entry_by_group).to_numpy()
    num_q['Numer'] = resolve_known_order_gaps(num_raw, num_q['CutoffMonthIdx'].to_numpy(), fe_num)
    num_q['Ratio'] = safe_ratio(num_q['Numer'], num_q['Denom'])

    out = num_q.groupby([group_col, 'OriginIdx', 'Horizon'])['Ratio'].mean().reset_index()
    return out.rename(columns={'Ratio': out_col})


def attach_external_factor_features(panel: pd.DataFrame, event_ranges, end_known_ranges=()) -> pd.DataFrame:
    """Point-in-time HasExternalEvent, plus EventTargetHindsight (target-only, for the sample weight)."""
    panel = panel.copy()
    panel['HasExternalEvent'] = is_known_event_idx(
        panel['OriginIdx'].to_numpy(), panel['TargetIdx'].to_numpy(), event_ranges, end_known_ranges).astype(int)
    panel['EventTargetHindsight'] = is_event_month_idx(panel['TargetIdx'].to_numpy(), event_ranges).astype(int)
    return panel


def attach_lag_event_features(panel: pd.DataFrame, event_ranges, group_col: str = 'PlanModelCode',
        windows: tuple = (6, 12), prefix: str = '') -> pd.DataFrame:
    """LagEventCount{w}: observed lag months in the trailing `w` that fall in an event window."""
    max_w = max(windows)
    lag_cols = [f'{prefix}Lag{k}' for k in range(max_w)]
    unique_oc = panel[[group_col, 'OriginIdx'] + lag_cols].drop_duplicates(subset=[group_col, 'OriginIdx']) \
        .reset_index(drop=True)

    origins = unique_oc['OriginIdx'].to_numpy()
    k = np.arange(max_w)
    month_idx_grid = origins[:, None] - k[None, :]
    is_event_grid = is_event_month_idx(month_idx_grid, event_ranges)
    observed_grid = unique_oc[lag_cols].notna().to_numpy()
    qualifies = is_event_grid & observed_grid

    count_cols = []
    for w in windows:
        col = f'{prefix}LagEventCount{w}'
        unique_oc[col] = qualifies[:, :w].sum(axis=1)
        count_cols.append(col)
    return panel.merge(unique_oc[[group_col, 'OriginIdx'] + count_cols], on=[group_col, 'OriginIdx'], how='left')


def build_market_shock_monthly(orders: pd.DataFrame, request_month_col: str = 'RequestMonthIdx',
        qty_col: str = 'Qty', window: int = 3) -> pd.DataFrame:
    """Market-wide MarketPctChange and MarketZScore_{window}m per RequestMonthIdx."""
    monthly = orders.groupby(request_month_col)[qty_col].sum().sort_index()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        pct_change = monthly.pct_change()
        rolling_mean = monthly.rolling(window, min_periods=1).mean()
        rolling_std = monthly.rolling(window, min_periods=1).std()
        zscore = safe_ratio((monthly - rolling_mean).to_numpy(), rolling_std.to_numpy())
    return pd.DataFrame({
        request_month_col: monthly.index,
        'MarketPctChange': pct_change.to_numpy(),
        f'MarketZScore_{window}m': zscore,
    })


def attach_market_shock_features(panel: pd.DataFrame, market_shock_monthly: pd.DataFrame,
        anchor_col: str = 'OriginIdx') -> pd.DataFrame:
    """Merges the market shock features onto every row by `anchor_col`'s month."""
    monthly = market_shock_monthly.rename(columns={'RequestMonthIdx': anchor_col})
    return panel.merge(monthly, on=anchor_col, how='left')


def attach_adjustment_trailing_features(panel: pd.DataFrame, windows: tuple = (3, 6, 12),
        group_col: str = 'PlanModelCode', history_source: pd.DataFrame = None) -> pd.DataFrame:
    """AdjTrailMean{w}: trailing mean of the entity's settled same-horizon Adjustment. Inference must
    pass `history_source`."""
    panel = panel.copy().reset_index(drop=True)
    history = panel if history_source is None else history_source
    max_w = max(windows)
    result = {f'AdjTrailMean{w}': np.full(len(panel), np.nan) for w in windows}

    for h in panel['Horizon'].unique():
        mask = (panel['Horizon'] == h).to_numpy()
        sub = panel.loc[mask]
        codes = sub[group_col].astype(str).to_numpy()
        origins = sub['OriginIdx'].to_numpy()

        hist_h = history.loc[history['Horizon'] == h]
        lookup = pd.Series(hist_h['Adjustment'].to_numpy(),
            index=pd.MultiIndex.from_arrays([hist_h[group_col].astype(str).to_numpy(), hist_h['OriginIdx'].to_numpy()]))

        vals = np.full((len(sub), max_w), np.nan)
        for j in range(max_w):
            key = pd.MultiIndex.from_arrays([codes, origins - h - j])
            vals[:, j] = lookup.reindex(key).to_numpy()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            for w in windows:
                result[f'AdjTrailMean{w}'][mask] = np.nanmean(vals[:, :w], axis=1)

    for w in windows:
        # a SKU without `w` settled same-horizon observations yet -> 0, same "nothing observed
        # yet" convention resolve_known_order_gaps/moving_average_forecast use elsewhere.
        panel[f'AdjTrailMean{w}'] = np.nan_to_num(result[f'AdjTrailMean{w}'], nan=0.0)
    return panel


def attach_manual_sim_features(panel: pd.DataFrame, windows: tuple = (3, 6, 12),
        group_col: str = 'PlanModelCode', history_source: pd.DataFrame = None) -> pd.DataFrame:
    """ManualSim{w}: reproduces the business's Excel manual forecasting method."""
    panel = panel.copy().reset_index(drop=True)
    history = panel if history_source is None else history_source
    max_w = max(windows)
    codes_all = panel[group_col].astype(str).to_numpy()
    origins_all = panel['OriginIdx'].to_numpy()
    horizons_all = panel['Horizon'].to_numpy()

    h1 = history.loc[history['Horizon'] == 1, [group_col, 'OriginIdx', 'Adjustment']]
    h1_lookup = pd.Series(h1['Adjustment'].to_numpy(),
        index=pd.MultiIndex.from_arrays([h1[group_col].astype(str).to_numpy(), h1['OriginIdx'].to_numpy()]))

    result = {f'ManualSim{w}': np.zeros(len(panel)) for w in windows}

    # base term: window-average of h1 Adjustment, offset 1+j - same for every row regardless
    # of that row's own Horizon.
    vals0 = np.full((len(panel), max_w), np.nan)
    for j in range(max_w):
        key = pd.MultiIndex.from_arrays([codes_all, origins_all - 1 - j])
        vals0[:, j] = h1_lookup.reindex(key).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        for w in windows:
            result[f'ManualSim{w}'] += np.nanmean(vals0[:, :w], axis=1)

    # accumulation terms: row with Horizon=h sums k=1..h-1 - add each k's contribution only to
    # the rows whose own Horizon exceeds k (mirrors the original per-row `for k in range(1, h)`).
    for k in range(1, int(horizons_all.max())):
        acc_k = history.loc[history['Horizon'] == k, [group_col, 'OriginIdx', 'Accumulation_1mo']]
        acc_k_lookup = pd.Series(acc_k['Accumulation_1mo'].to_numpy(),
            index=pd.MultiIndex.from_arrays([acc_k[group_col].astype(str).to_numpy(), acc_k['OriginIdx'].to_numpy()]))

        applies = horizons_all > k
        if not applies.any():
            continue
        codes_sub, origins_sub = codes_all[applies], origins_all[applies]
        vals_k = np.full((applies.sum(), max_w), np.nan)
        for j in range(max_w):
            key = pd.MultiIndex.from_arrays([codes_sub, origins_sub - j])
            vals_k[:, j] = acc_k_lookup.reindex(key).to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            for w in windows:
                result[f'ManualSim{w}'][applies] += np.nanmean(vals_k[:, :w], axis=1)

    for w in windows:
        panel[f'ManualSim{w}'] = np.nan_to_num(result[f'ManualSim{w}'], nan=0.0)
    return panel


def _ols_slope(L: np.ndarray) -> np.ndarray:
    """Vectorized, NaN-aware row-wise OLS slope of L against x=[0..n-1]."""
    n_rows, n_cols = L.shape
    x = np.arange(n_cols, dtype='float64')
    mask = ~np.isnan(L)
    y = np.where(mask, L, 0.0)
    n = mask.sum(axis=1).astype('float64')
    sx = (mask * x).sum(axis=1)
    sxx = (mask * x ** 2).sum(axis=1)
    sy = y.sum(axis=1)
    sxy = (y * x).sum(axis=1)
    denom = n * sxx - sx ** 2
    slope = np.full(n_rows, np.nan)
    valid = (n >= 2) & (denom > 1e-9)
    slope[valid] = (n[valid] * sxy[valid] - sx[valid] * sy[valid]) / denom[valid]
    return slope


def fill_gaps(group: pd.DataFrame, group_col: str = 'PlanModelCode', index_col: str = 'OriginIdx',
        value_col: str = 'Adjustment', fill_value: float = 0.0) -> pd.DataFrame:
    """Densifies one entity's rows to a gapless `index_col` range, filling with `fill_value`."""
    full_idx = pd.RangeIndex(group[index_col].min(), group[index_col].max() + 1, name=index_col)
    out = group.set_index(index_col)[[value_col]].reindex(full_idx, fill_value=fill_value)
    out[group_col] = group.name
    return out.reset_index()


def sku_demand_profile(df: pd.DataFrame, group_col: str = 'PlanModelCode',
        target_idx_col: str = 'TargetIdx', value_col: str = 'ActualOrder',
        part_col: str = 'PartCode') -> pd.DataFrame:
    """Whole-history demand profile per entity: n_periods, pct_zero, ADI, CV2, total_vol, mean_demand."""
    rows = []
    for code, g in df.groupby(group_col, observed=True):
        y = g.sort_values(target_idx_col)[value_col].to_numpy()
        n = len(y)
        nonzero = y[y > 0]
        n_nonzero = len(nonzero)
        adi = n / n_nonzero if n_nonzero > 0 else np.nan
        cv2 = (nonzero.std() / nonzero.mean()) ** 2 if n_nonzero > 1 and nonzero.mean() > 0 else np.nan
        rows.append({
            group_col: code,
            part_col: g[part_col].iloc[0],
            'n_periods': n,
            'pct_zero': 1 - n_nonzero / n if n > 0 else np.nan,
            'ADI': adi,
            'CV2': cv2,
            'total_vol': y.sum(),
            'mean_demand': y.mean(),
        })
    return pd.DataFrame(rows)


def classify_demand(row, adi_thresh: float = 1.32, cv2_thresh: float = 0.49) -> str:
    """Syntetos-Boylan class (Smooth/Intermittent/Erratic/Lumpy) from `row['ADI']` and `row['CV2']`."""
    if pd.isna(row['ADI']) or pd.isna(row['CV2']):
        return 'insufficient_data'
    if row['ADI'] < adi_thresh and row['CV2'] < cv2_thresh:
        return 'Smooth'
    if row['ADI'] >= adi_thresh and row['CV2'] < cv2_thresh:
        return 'Intermittent'
    if row['ADI'] < adi_thresh and row['CV2'] >= cv2_thresh:
        return 'Erratic'
    return 'Lumpy'
