"""Country-mix order-book visibility features: expected fill by horizon from a SKU's buyer-country
mix, since countries order at different lead times.
"""
import logging

import numpy as np
import pandas as pd

from .config import is_covid_month_idx
from .features import safe_ratio, _visibility_ratio_and_average
from .order_book import build_order_book, query_known_order

log = logging.getLogger(__name__)

COUNTRY_COL = 'CustomerCountry'


def build_country_visibility(orders: pd.DataFrame, origins, horizons=range(1, 7),
        lookback_months: int = 6, country_col: str = COUNTRY_COL) -> pd.DataFrame:
    """Per (country, OriginIdx, Horizon) fill ratio, as HistVisibility_AllHistory but grouped by
    country. `origins` should span every OriginIdx in the panel."""
    ob = build_order_book(orders, [country_col])
    first_entry = orders.groupby(country_col)['EntryMonthIdx'].min()

    cand = pd.DataFrame({country_col: first_entry.index.to_numpy()})
    cand = cand.merge(pd.DataFrame({'OriginIdx': list(origins)}), how='cross')
    cand = cand.merge(pd.DataFrame({'Horizon': list(horizons)}), how='cross')
    cand['TargetIdx'] = cand['OriginIdx'] + cand['Horizon']
    cand = cand.merge(pd.DataFrame({'m': range(1, lookback_months + 1)}), how='cross')
    cand['Mprime'] = cand['OriginIdx'] - cand['m']
    # Same COVID exclusion attach_historical_visibility applies - that period's fill behaviour was
    # a one-off, and pooling it into a country's baseline curve would distort it long afterwards.
    cand = cand[~is_covid_month_idx(cand['Mprime'])]

    out = _visibility_ratio_and_average(cand, ob, country_col, first_entry,
        out_col='CountryVisibility')
    log.info("Built country visibility curve: %d (country, origin, horizon) row(s), %d countries.",
        len(out), out[country_col].nunique())
    return out


def build_sku_country_mix(orders: pd.DataFrame, origins, window: int = 12,
        min_share: float = 0.01, country_col: str = COUNTRY_COL,
        group_col: str = 'PlanModelCode') -> pd.DataFrame:
    """Trailing-`window`-month country mix per (SKU, country, OriginIdx); shares below `min_share`
    are dropped and the rest renormalised."""
    monthly = orders.groupby([group_col, country_col, 'RequestMonthIdx'], observed=True)['Qty'] \
        .sum().rename('qty').reset_index()

    frames = []
    for origin in origins:
        lo = origin - window + 1
        window_rows = monthly[(monthly['RequestMonthIdx'] >= lo) & (monthly['RequestMonthIdx'] <= origin)]
        if window_rows.empty:
            continue
        agg = window_rows.groupby([group_col, country_col], observed=True)['qty'].sum().reset_index()
        agg['OriginIdx'] = origin
        frames.append(agg)
    if not frames:
        return pd.DataFrame(columns=[group_col, country_col, 'OriginIdx', 'share'])

    mix = pd.concat(frames, ignore_index=True)
    mix['share'] = mix['qty'] / mix.groupby([group_col, 'OriginIdx'], observed=True)['qty'].transform('sum')

    before = len(mix)
    mix = mix[mix['share'] >= min_share].copy()
    # renormalise so the surviving shares still sum to 1 per (SKU, Origin)
    mix['share'] = mix['share'] / mix.groupby([group_col, 'OriginIdx'], observed=True)['share'].transform('sum')
    log.info("Built SKU country mix: %d row(s) (%d dropped below min_share=%.3f).",
        len(mix), before - len(mix), min_share)
    return mix[[group_col, country_col, 'OriginIdx', 'share']]


def build_focus_country_order_book(orders: pd.DataFrame, focus_country: str,
        country_col: str = COUNTRY_COL, group_col: str = 'PlanModelCode',
        arrival_col: str = 'EntryMonthIdx') -> pd.DataFrame:
    """Entity-grain order book of `focus_country`'s lines; arrival_col='EntryDate' for day grain."""
    focus = orders[orders[country_col] == focus_country]
    if focus.empty:
        log.warning("No order lines for focus_country=%r - focus share features will be all-zero.",
            focus_country)
    return build_order_book(focus, [group_col], arrival_col=arrival_col)


def attach_country_visibility_features(panel: pd.DataFrame, country_visibility: pd.DataFrame,
        sku_country_mix: pd.DataFrame, focus_order_book: pd.DataFrame, focus_country: str = 'CN',
        country_col: str = COUNTRY_COL, group_col: str = 'PlanModelCode',
        min_expected_visibility: float = 0.05, include: bool = True,
        asof_col: str = None, arrival_col: str = 'EntryMonthIdx') -> pd.DataFrame:
    """Attaches the country-visibility features; must run after attach_known_order_features.
    `asof_col`/`arrival_col` affect only the Known{focus}Share query."""
    if not include:
        return panel
    if 'KnownOrder' not in panel.columns:
        raise ValueError("attach_country_visibility_features needs KnownOrder on the panel - run "
                         "attach_known_order_features first (see _attach_all_features ordering).")

    panel = panel.copy()

    # --- (a) mix-weighted expected visibility -------------------------------------------------
    # Drop countries with no curve for that (origin, horizon) and renormalise over the rest, so a
    # missing curve shrinks the evidence base rather than silently biasing the weighted mean down.
    ev = sku_country_mix.merge(country_visibility, on=[country_col, 'OriginIdx'], how='inner')
    ev = ev.dropna(subset=['CountryVisibility'])
    ev['_w'] = ev['share'] * ev['CountryVisibility']
    grouped = ev.groupby([group_col, 'OriginIdx', 'Horizon'], observed=True)
    expected = (grouped['_w'].sum() / grouped['share'].sum()).rename('ExpectedVisibility').reset_index()
    panel = panel.merge(expected, on=[group_col, 'OriginIdx', 'Horizon'], how='left')

    # --- (b) the level that expectation implies -----------------------------------------------
    # KnownOrder / ExpectedVisibility estimates the eventual total, so minus KnownOrder it is a
    # direct, mix-informed estimate of Adjustment itself - the same "baseline as a feature" idea
    # ManualSim{w} uses. Clipped below: a near-zero expectation would otherwise blow the ratio up.
    ev_clipped = panel['ExpectedVisibility'].clip(lower=min_expected_visibility)
    panel['ImpliedAdjustment_Mix'] = panel['KnownOrder'].to_numpy() * (1.0 / ev_clipped.to_numpy() - 1.0)
    panel.loc[panel['ExpectedVisibility'].isna(), 'ImpliedAdjustment_Mix'] = np.nan

    # --- (c) focus-country share of the book visible RIGHT NOW --------------------------------
    q = pd.DataFrame({
        group_col: panel[group_col].to_numpy(),
        'RequestMonthIdx': panel['TargetIdx'].to_numpy(),
        'CutoffMonthIdx': (panel['OriginIdx'] if asof_col is None else panel[asof_col]).to_numpy(),
    })
    # nan -> 0 for the same reason attach_known_order_features does it for KnownOrder: these rows
    # already passed the lifespan filters, so "no order-book entry" means a genuine zero, not a
    # warm-up gap. A SKU that simply never sells to `focus_country` correctly reads 0.
    known_focus = np.nan_to_num(
        query_known_order(q, focus_order_book, [group_col], arrival_col=arrival_col), nan=0.0)
    panel[f'Known{focus_country}Share'] = safe_ratio(known_focus, panel['KnownOrder'].to_numpy())

    # --- (d) ...against what this SKU normally looks like --------------------------------------
    baseline = sku_country_mix[sku_country_mix[country_col] == focus_country][
        [group_col, 'OriginIdx', 'share']].rename(columns={'share': f'Baseline{focus_country}Share'})
    panel = panel.merge(baseline, on=[group_col, 'OriginIdx'], how='left')
    # A SKU absent from the focus country's rows genuinely had no focus-country demand in the
    # window - that is a real 0, not missing data (unlike Known{focus}Share, which safe_ratio
    # leaves NaN when KnownOrder itself is ~0 and the share is undefined).
    panel[f'Baseline{focus_country}Share'] = panel[f'Baseline{focus_country}Share'].fillna(0.0)

    # The signal the whole family exists for: book is under-weighted in the LATE-arriving country
    # relative to this SKU's own norm -> those orders have not landed yet -> expect a bigger
    # positive Adjustment. Negative gap = under-represented so far.
    panel[f'{focus_country}ShareGap'] = (panel[f'Known{focus_country}Share']
        - panel[f'Baseline{focus_country}Share'])
    return panel


def country_visibility_feature_cols(focus_country: str = 'CN') -> list:
    """The family's column names, so _feature_families() doesn't hardcode the focus country."""
    return [
        'ExpectedVisibility', 'ImpliedAdjustment_Mix',
        f'Known{focus_country}Share', f'Baseline{focus_country}Share', f'{focus_country}ShareGap',
    ]
