"""Day-grain-only features: cutoff calendar, arrival velocity and same-day-of-month visibility.
A month-grain panel is returned untouched.
"""
import logging

import numpy as np
import pandas as pd

from .config import CUTOFF_DATE_COL, date_to_month_idx, is_cny_month_idx, is_covid_month_idx
from .cutoff_grain import cutoff_dates_for_origins, cutoff_day_index, month_bounds
from .features import safe_ratio
from .order_book import query_known_order, resolve_known_order_gaps

log = logging.getLogger(__name__)

CUTOFF_CALENDAR_COLS = [
    'CutoffDayIndex', 'CutoffDayOfWeek', 'CutoffWorkingDaysElapsed',
    'CutoffWorkingDaysRemaining', 'CutoffMonthProgress', 'CutoffMonthIsCNY',
]
ARRIVAL_VELOCITY_COLS = [
    'Accumulation_7d', 'Accumulation_14d', 'Accumulation_sinceMonthStart',
    'ArrivalRate_7d', 'ArrivalRate_sinceMonthStart', 'ArrivalAcceleration_7v14',
]
SAME_DAY_VISIBILITY_COLS = [
    'HistVisibility_SameDayOfMonth', 'ImpliedTotal_SameDayOfMonth',
]
INTRAMONTH_FEATURE_COLS = (CUTOFF_CALENDAR_COLS + ARRIVAL_VELOCITY_COLS
                           + SAME_DAY_VISIBILITY_COLS)


def attach_intramonth_features(panel: pd.DataFrame, order_book_code_daily: pd.DataFrame = None,
        order_book_code: pd.DataFrame = None, first_entry_by_code: pd.Series = None,
        group_col: str = 'PlanModelCode', asof_col: str = CUTOFF_DATE_COL,
        lookback_months: int = 6, include_same_day_visibility: bool = True) -> pd.DataFrame:
    """Attaches every intra-month family; returns `panel` unchanged on a month-grain panel."""
    if asof_col not in panel.columns:
        return panel
    if order_book_code_daily is None:
        raise ValueError(
            f"panel carries {asof_col!r} (day grain) but order_book_code_daily is None - build the "
            "context with build_order_book_context(..., cutoff_grain='day').")

    panel = _attach_cutoff_calendar(panel, asof_col)
    panel = _attach_arrival_velocity(panel, order_book_code_daily, first_entry_by_code,
                                     group_col, asof_col)
    if include_same_day_visibility:
        panel = _attach_same_day_visibility(panel, order_book_code_daily, order_book_code,
                                            first_entry_by_code, group_col, asof_col,
                                            lookback_months)
    return panel


def _attach_cutoff_calendar(panel: pd.DataFrame, asof_col: str) -> pd.DataFrame:
    """Where the as-of point sits in its own month. Pure date arithmetic - no order book."""
    panel = panel.copy()
    asof = pd.to_datetime(panel[asof_col])

    panel['CutoffDayIndex'] = cutoff_day_index(panel)
    panel['CutoffDayOfWeek'] = asof.dt.dayofweek.astype('int16')

    month_first = asof.values.astype('datetime64[M]')
    month_after = month_first + np.timedelta64(1, 'M')
    day = asof.values.astype('datetime64[D]')
    # busday_count's end is exclusive, so +1 day makes the cutoff date itself count as elapsed.
    elapsed = np.busday_count(month_first.astype('datetime64[D]'), day + np.timedelta64(1, 'D'))
    total = np.busday_count(month_first.astype('datetime64[D]'), month_after.astype('datetime64[D]'))
    panel['CutoffWorkingDaysElapsed'] = elapsed.astype('int16')
    panel['CutoffWorkingDaysRemaining'] = (total - elapsed).astype('int16')
    # How far through the CUTOFF DATE'S OWN month, normalised for month length and weekday layout
    # - a 20-working-day February and a 23-working-day March are not comparable on a raw day count.
    # Note it resets at the day-0/day-1 boundary and so reads 1.0 at BOTH ends of the refresh
    # cycle; CutoffDayIndex is what orders the cycle. See the module docstring.
    panel['CutoffMonthProgress'] = (elapsed / total).astype('float32')
    panel['CutoffMonthIsCNY'] = is_cny_month_idx(date_to_month_idx(asof)).astype('int8')
    return panel


def _attach_arrival_velocity(panel: pd.DataFrame, book_daily: pd.DataFrame,
        first_entry_by_code: pd.Series, group_col: str, asof_col: str) -> pd.DataFrame:
    """7-day, 14-day and month-to-date accumulation and per-working-day rate from the day book."""
    panel = panel.copy()
    asof = pd.to_datetime(panel[asof_col])
    target_idx = panel['TargetIdx'].to_numpy()
    first_entry = panel[group_col].map(first_entry_by_code).to_numpy()

    def _known_at(asof_values):
        q = panel[[group_col]].copy()
        q['RequestMonthIdx'] = target_idx
        q['CutoffMonthIdx'] = asof_values      # the query's as-of column, day grain here
        raw = query_known_order(q, book_daily, [group_col], arrival_col='EntryDate')
        return resolve_known_order_gaps(raw, date_to_month_idx(pd.Series(asof_values)), first_entry)

    now = np.nan_to_num(_known_at(asof.to_numpy()), nan=0.0)
    for days in (7, 14):
        back = _known_at((asof - pd.Timedelta(days=days)).to_numpy())
        panel[f'Accumulation_{days}d'] = now - back

    # Month to date, measured against the END of the previous month so the window is exactly "this
    # calendar month so far" at every cutoff day - including day 0, where it is the origin month's
    # own accumulation.
    prev_month_end = month_bounds(date_to_month_idx(asof) - 1)[1].to_numpy()
    panel['Accumulation_sinceMonthStart'] = now - _known_at(prev_month_end)

    working = panel['CutoffWorkingDaysElapsed'].to_numpy()
    panel['ArrivalRate_7d'] = panel['Accumulation_7d'] / 5.0       # 7 calendar days = 5 working
    panel['ArrivalRate_sinceMonthStart'] = safe_ratio(
        panel['Accumulation_sinceMonthStart'], np.maximum(working, 1))
    # What share of the last fortnight's arrivals landed in the last week - bounded and readable
    # where a raw rate ratio blows up, and the closest thing to an acceleration signal that
    # survives intermittent demand.
    panel['ArrivalAcceleration_7v14'] = safe_ratio(panel['Accumulation_7d'],
                                                   panel['Accumulation_14d'])
    return panel


def _attach_same_day_visibility(panel: pd.DataFrame, book_daily: pd.DataFrame,
        book_month: pd.DataFrame, first_entry_by_code: pd.Series, group_col: str, asof_col: str,
        lookback_months: int) -> pd.DataFrame:
    """Average share of prior months' book visible at the same day of month, and KnownOrder / it as
    an implied total. Equals HistVisibility_AllHistory at d=0."""
    keys = [group_col, 'OriginIdx', 'Horizon', asof_col]
    uniq = panel[keys].drop_duplicates().reset_index(drop=True)
    uniq['DayIndex'] = cutoff_day_index(
        uniq.assign(**{asof_col: pd.to_datetime(uniq[asof_col])}))
    uniq['TargetIdx'] = uniq['OriginIdx'] + uniq['Horizon']

    cand = uniq.merge(pd.DataFrame({'m': range(1, lookback_months + 1)}), how='cross')
    cand['Mprime'] = cand['OriginIdx'] - cand['m']
    cand = cand[(cand['Mprime'] < cand['OriginIdx']) & ~is_covid_month_idx(cand['Mprime'])]
    cand = cand[cand['Mprime'] - cand['Horizon'] >= 0].reset_index(drop=True)
    log.info("SameDayOfMonth visibility: %d candidate row(s) over %d unique as-of point(s).",
             len(cand), len(uniq))

    fe = cand[group_col].map(first_entry_by_code).to_numpy()

    denom_q = pd.DataFrame({group_col: cand[group_col], 'RequestMonthIdx': cand['Mprime'],
                            'CutoffMonthIdx': cand['OriginIdx']})
    cand['Denom'] = resolve_known_order_gaps(
        query_known_order(denom_q, book_month, [group_col]),
        cand['Mprime'].to_numpy(), fe)

    # The analogous as-of DATE: the same day-index d, but on M''s own origin month (M' - h).
    prior_origin = (cand['Mprime'] - cand['Horizon']).to_numpy()
    prior_asof = np.empty(len(cand), dtype='datetime64[ns]')
    for d, sel in cand.groupby('DayIndex').groups.items():
        pos = cand.index.get_indexer(sel)
        prior_asof[pos] = cutoff_dates_for_origins(prior_origin[pos], int(d))

    num_q = pd.DataFrame({group_col: cand[group_col], 'RequestMonthIdx': cand['Mprime'],
                          'CutoffMonthIdx': prior_asof})
    cand['Numer'] = resolve_known_order_gaps(
        query_known_order(num_q, book_daily, [group_col], arrival_col='EntryDate'),
        date_to_month_idx(pd.Series(prior_asof)), fe)
    cand['Ratio'] = safe_ratio(cand['Numer'], cand['Denom'])

    feat = cand.groupby(keys)['Ratio'].mean().reset_index()
    feat = feat.rename(columns={'Ratio': 'HistVisibility_SameDayOfMonth'})

    panel = panel.merge(feat, on=keys, how='left')
    # KnownOrder / (fraction usually visible by now) - the ratio expressed back in the target's own
    # units, which is what a tree can actually split on against Lag*/TrailMean levels.
    panel['ImpliedTotal_SameDayOfMonth'] = safe_ratio(
        panel['KnownOrder'], panel['HistVisibility_SameDayOfMonth'])
    return panel
