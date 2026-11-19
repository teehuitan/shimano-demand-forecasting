"""Order-book primitive: each (group, RequestMonth) cohort's cumulative build-up by arrival, queried
at a historical as-of point without leaking the future.
"""
import numpy as np
import pandas as pd

from .config import CUTOFF_DATE_COL, date_to_month_idx


def _is_day_grain(values) -> bool:
    """Datetime-like (day grain) vs month-ordinal (month grain) - the only two grains in use."""
    return pd.api.types.is_datetime64_any_dtype(values)


def frame_cutoff_grain(df: pd.DataFrame) -> str:
    """'day' if the frame has a datetime CUTOFF_DATE_COL, else 'month'."""
    if CUTOFF_DATE_COL in df.columns and _is_day_grain(df[CUTOFF_DATE_COL]):
        return 'day'
    return 'month'


def build_order_book(orders: pd.DataFrame, group_cols: list,
        arrival_col: str = 'EntryMonthIdx') -> pd.DataFrame:
    """CumQty per (group..., RequestMonthIdx, arrival_col), sorted for merge_asof."""
    keys = group_cols + ['RequestMonthIdx', arrival_col]
    book = orders.groupby(keys, as_index=False)['Qty'].sum()
    book = book.sort_values(group_cols + ['RequestMonthIdx', arrival_col])
    book['CumQty'] = book.groupby(group_cols + ['RequestMonthIdx'])['Qty'].cumsum()
    return book.sort_values(arrival_col).reset_index(drop=True)


def _check_grain(queries: pd.DataFrame, order_book: pd.DataFrame,
        arrival_col: str, asof_col: str) -> None:
    """Raises unless the as-of values and the book's arrival axis are the same grain; a mismatch
    would silently give KnownOrder=0."""
    book_is_day = _is_day_grain(order_book[arrival_col])
    asof_is_day = _is_day_grain(queries[asof_col])
    if book_is_day != asof_is_day:
        name = lambda is_day: 'day grain (datetime)' if is_day else 'month grain (ordinal)'
        raise ValueError(
            f"grain mismatch: order_book[{arrival_col!r}] is {name(book_is_day)} but "
            f"queries[{asof_col!r}] is {name(asof_is_day)}. Pass the arrival_col/asof_col pair "
            "matching the book you built - casting across grains would silently return "
            "KnownOrder=0 everywhere rather than failing.")


def query_known_order(queries: pd.DataFrame, order_book: pd.DataFrame, group_cols: list,
        arrival_col: str = 'EntryMonthIdx', asof_col: str = 'CutoffMonthIdx') -> np.ndarray:
    """Cumulative known quantity at each row's as-of point; NaN (unfilled) where the book has no row
    yet."""
    q = queries.reset_index(drop=True).copy()
    _check_grain(q, order_book, arrival_col, asof_col)
    # Defensive dtype alignment, WITHIN one grain only (see _check_grain for why crossing grains
    # must never reach this): ad hoc query batches default to int64, while order_book's
    # month-index columns are downcast to int32 - merge_asof requires an exact dtype match.
    q['RequestMonthIdx'] = q['RequestMonthIdx'].astype(order_book['RequestMonthIdx'].dtype)
    q[asof_col] = q[asof_col].astype(order_book[arrival_col].dtype)
    q['_qid'] = np.arange(len(q))
    q = q.sort_values(asof_col)
    merged = pd.merge_asof(
        q, order_book,
        left_on=asof_col, right_on=arrival_col,
        by=group_cols + ['RequestMonthIdx'],
        direction='backward',
    )
    merged = merged.sort_values('_qid')
    return merged['CumQty'].to_numpy()


def shift_asof(asof_values, months: int):
    """Moves each as-of point back `months` months, keeping its day of month; a month-end stays a
    month-end."""
    values = np.asarray(asof_values)
    if not _is_day_grain(values):
        return values - months
    asof = pd.Series(values)
    shifted = asof - pd.DateOffset(months=months)
    is_month_end = asof == (asof + pd.offsets.MonthEnd(0))
    return shifted.where(~is_month_end, shifted + pd.offsets.MonthEnd(0)).to_numpy()


def asof_month_idx(asof_values) -> np.ndarray:
    """Month-ordinal of each as-of point, whatever its grain."""
    values = np.asarray(asof_values)
    if _is_day_grain(values):
        return date_to_month_idx(pd.Series(values))
    return values


def asof_points(panel: pd.DataFrame, asof_col: str = None) -> np.ndarray:
    """Each row's as-of point: OriginIdx at month grain, the as-of date column at day grain."""
    return (panel['OriginIdx'] if asof_col is None else panel[asof_col]).to_numpy()


def known_canceled_order(panel: pd.DataFrame, cancel_book: pd.DataFrame,
        group_col: str = 'PlanModelCode', asof_col: str = None,
        arrival_col: str = 'CancelMonthIdx') -> np.ndarray:
    """Quantity cancelled as of the same as-of point as KnownOrder (missing -> 0); raises if it
    exceeds KnownOrder."""
    if cancel_book is None:
        raise ValueError(
            "no cancellation book for this grain - rebuild the context with "
            "build_order_book_context(..., cutoff_grain=<the grain being scored>).")
    q = panel[[group_col]].copy()
    q[group_col] = q[group_col].astype(str)          # book keys are plain str, not category
    q['RequestMonthIdx'] = panel['TargetIdx'].to_numpy()
    q['CutoffMonthIdx'] = asof_points(panel, asof_col)
    cancelled = np.nan_to_num(query_known_order(q, cancel_book, [group_col],
                                                arrival_col=arrival_col), nan=0.0)
    if 'KnownOrder' in panel.columns:
        over = cancelled > panel['KnownOrder'].to_numpy(dtype='float64') + 1e-6
        if over.any():
            bad = panel.loc[over, [group_col, 'OriginIdx', 'TargetIdx']].head(5).to_dict('records')
            raise ValueError(
                f"{int(over.sum())} row(s) show more cancelled than ordered as of the same point "
                f"(e.g. {bad}) - KnownOrder and KnownCanceledOrder were read at different as-of "
                "points or from different books. Refusing to produce a negative net order.")
    return cancelled


def resolve_known_order_gaps(raw_values: np.ndarray, queried_month_idx, group_first_entry_idx) -> np.ndarray:
    """Fills query_known_order's NaN with 0 within the group's lifespan; NaN before its first entry."""
    queried_month_idx = np.asarray(queried_month_idx)
    group_first_entry_idx = np.asarray(group_first_entry_idx)
    in_lifespan = queried_month_idx >= group_first_entry_idx
    return np.where(np.isnan(raw_values) & in_lifespan, 0.0, raw_values)
