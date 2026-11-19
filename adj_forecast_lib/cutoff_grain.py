"""The as-of schedule: the real date each (OriginIdx, cutoff-day) pair means. Kept separate from
preprocessing so feature modules share one definition.
"""
import numpy as np
import pandas as pd

from .config import CUTOFF_DATE_COL, month_idx_to_yyyymm


def month_bounds(month_idx):
    """(first day, last day) of each month-ordinal, as DatetimeIndexes."""
    yyyymm = month_idx_to_yyyymm(np.asarray(month_idx))
    first = pd.to_datetime([f'{int(v) // 100:04d}-{int(v) % 100:02d}-01' for v in yyyymm])
    return first, first + pd.offsets.MonthEnd(0)


def cutoff_dates_for_origins(origin_idx, day: int) -> np.ndarray:
    """As-of date per OriginIdx: day=0 is the origin month's end, day>=1 that day of the next month
    (clamped to month end)."""
    origin_idx = np.asarray(origin_idx)
    if day == 0:
        return month_bounds(origin_idx)[1].to_numpy()
    first, last = month_bounds(origin_idx + 1)
    offsets = np.minimum(day, last.day.to_numpy()) - 1
    return (first + pd.to_timedelta(offsets, unit='D')).to_numpy()


def cutoff_window(origin_idx) -> pd.DatetimeIndex:
    """As-of dates for one origin: origin month-end through the day before next month's end."""
    _, origin_end = month_bounds([origin_idx])
    _, next_end = month_bounds([origin_idx + 1])
    return pd.date_range(origin_end[0], next_end[0] - pd.Timedelta(days=1), freq='D')


def monthly_anchor_mask(panel: pd.DataFrame) -> np.ndarray:
    """Day-0 rows (as-of = origin month-end), which reproduce the monthly panel."""
    return pd.to_datetime(panel[CUTOFF_DATE_COL]).dt.is_month_end.to_numpy()


def cutoff_day_index(panel: pd.DataFrame) -> np.ndarray:
    """Each row's day index: 0 at origin month-end, else the calendar day. All zeros for a
    month-grain panel."""
    if CUTOFF_DATE_COL not in panel.columns:
        return np.zeros(len(panel), dtype='int16')
    day = pd.to_datetime(panel[CUTOFF_DATE_COL]).dt.day.to_numpy()
    return np.where(monthly_anchor_mask(panel), 0, day).astype('int16')



