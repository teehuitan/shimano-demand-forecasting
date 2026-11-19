"""Fold generators on forecast-origin boundaries; folds cut on TargetIdx <= test_origin to avoid
label leakage."""
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .lifecycle_features import meets_min_launch_age


class FoldSequence(Sequence):
    """Re-iterable list of walk-forward folds stored as boundaries; frames are cut fresh on each access."""

    def __init__(self, panel: pd.DataFrame, specs: list, materialize):
        self._panel = panel
        self._specs = list(specs)
        self._materialize = materialize

    def __len__(self):
        return len(self._specs)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self._materialize(self._panel, s) for s in self._specs[i]]
        return self._materialize(self._panel, self._specs[i])

    def __repr__(self):
        return f'<{type(self).__name__}: {len(self._specs)} fold(s) over {len(self._panel):,} row(s)>'

    @property
    def specs(self) -> list:
        """The per-fold boundary scalars, without materialising anything - for asserting that two
        fold sets line up rather than comparing frames."""
        return list(self._specs)


def _walk_forward_origins(panel_h: pd.DataFrame, horizon: int, initial_train_origins: int,
        max_settled_target_idx, max_folds) -> list:
    """Test origins a per-horizon walk-forward steps through, in order."""
    origins = sorted(panel_h['OriginIdx'].unique())
    max_target = max_settled_target_idx if max_settled_target_idx is not None else panel_h['TargetIdx'].max()
    start_i = initial_train_origins if max_folds is None else max(initial_train_origins, len(origins) - max_folds)

    test_origins = []
    for i in range(start_i, len(origins)):
        test_origin = origins[i]
        if test_origin + horizon > max_target:
            break
        test_origins.append(test_origin)
    return test_origins


def scoreable_rows(df: pd.DataFrame, min_origin_months_since_launch: int = None) -> pd.Series:
    """Mask of rows inference would predict: in the lifecycle and old enough (meets_min_launch_age)."""
    # Panels built before InLifecycle existed only ever held lifecycle rows.
    keep = df['InLifecycle'] if 'InLifecycle' in df.columns else pd.Series(True, index=df.index)
    if min_origin_months_since_launch is not None:
        keep = keep & meets_min_launch_age(df, min_origin_months_since_launch)
    return keep


def fixed_split_horizon(panel_h: pd.DataFrame, horizon: int, test_target_months: int = 15,
        val_months: int = 6, max_settled_target_idx: int = None):
    """Approach 1 fixed train/val/test split for one horizon-specific model."""
    max_target = max_settled_target_idx if max_settled_target_idx is not None else panel_h['TargetIdx'].max()
    eligible_targets = sorted(panel_h.loc[panel_h['TargetIdx'] <= max_target, 'TargetIdx'].unique())
    test_targets = eligible_targets[-test_target_months:]
    test_origin_start = test_targets[0] - horizon  # origin of the first test target
    val_cut = test_origin_start - val_months

    train = panel_h[panel_h['TargetIdx'] <= val_cut]
    val = panel_h[(panel_h['TargetIdx'] > val_cut) & (panel_h['TargetIdx'] <= test_origin_start)]
    test = panel_h[panel_h['TargetIdx'].isin(test_targets) & scoreable_rows(panel_h)]
    return train, val, test


def fixed_split_shared(panel_all: pd.DataFrame, test_target_months: int = 15, val_months: int = 6,
        horizons=range(1, 7), max_settled_target_idx: int = None):
    """Approach 2 fixed split: one boundary, anchored on the largest horizon, for all horizons."""
    max_h = max(horizons)
    max_target = max_settled_target_idx if max_settled_target_idx is not None else panel_all['TargetIdx'].max()
    eligible_targets = sorted(panel_all.loc[panel_all['TargetIdx'] <= max_target, 'TargetIdx'].unique())
    test_targets = eligible_targets[-test_target_months:]
    test_origin_start = test_targets[0] - max_h  # origin of the first test target for the largest horizon
    val_cut = test_origin_start - val_months

    train = panel_all[panel_all['TargetIdx'] <= val_cut]
    val = panel_all[(panel_all['TargetIdx'] > val_cut) & (panel_all['TargetIdx'] <= test_origin_start)]
    test = panel_all[panel_all['TargetIdx'].isin(test_targets) & scoreable_rows(panel_all)]
    return train, val, test


def walk_forward_horizon(panel_h: pd.DataFrame, horizon: int, initial_train_origins: int = 60,
        val_months: int = 3, max_settled_target_idx: int = None, max_folds: int = None,
        min_origin_months_since_launch: int = None):
    """Approach 1 walk-forward (monthly rolling origin, expanding window) as a FoldSequence.
    `min_origin_months_since_launch` restricts test rows only."""
    specs = [(o, o - val_months, min_origin_months_since_launch)
             for o in _walk_forward_origins(panel_h, horizon, initial_train_origins,
                                            max_settled_target_idx, max_folds)]
    return FoldSequence(panel_h, specs, _materialize_horizon_fold)


def _materialize_horizon_fold(panel_h: pd.DataFrame, spec: tuple) -> tuple:
    """One walk_forward_horizon fold as (train, val, test)."""
    test_origin, val_cut, min_origin_months_since_launch = spec
    known = panel_h[panel_h['TargetIdx'] <= test_origin]
    train = known[known['TargetIdx'] <= val_cut]
    val = known[known['TargetIdx'] > val_cut]
    test = panel_h[panel_h['OriginIdx'] == test_origin]
    test = test[scoreable_rows(test, min_origin_months_since_launch)]
    return train, val, test


def fold_step_months(panel_h: pd.DataFrame, horizon: int, initial_train_origins: int = 60,
        max_settled_target_idx: int = None, max_folds: int = None) -> pd.DataFrame:
    """Maps each walk_forward_horizon fold step to its origin/target YYYYMM; pass the same args."""
    from .config import month_idx_to_yyyymm

    origin_idxs = _walk_forward_origins(panel_h, horizon, initial_train_origins,
                                        max_settled_target_idx, max_folds)
    target_idxs = [o + horizon for o in origin_idxs]

    return pd.DataFrame({
        'origin_month': month_idx_to_yyyymm(origin_idxs),
        'target_month': month_idx_to_yyyymm(target_idxs),
    }, index=pd.RangeIndex(len(origin_idxs), name='step'))


def deployment_split_horizon(panel_h: pd.DataFrame, max_settled_target_idx, val_months: int = 6):
    """Approach 1 deployment split: train/val only, val at the end of settled history."""
    max_target = max_settled_target_idx
    eligible = panel_h[panel_h['TargetIdx'] <= max_target]
    eligible_targets = sorted(eligible['TargetIdx'].unique())
    val_cut = eligible_targets[-1] - val_months

    train = eligible[eligible['TargetIdx'] <= val_cut]
    val = eligible[eligible['TargetIdx'] > val_cut]
    return train, val


def deployment_refit_rows(panel_h: pd.DataFrame, max_settled_target_idx) -> pd.DataFrame:
    """All settled rows (train + val) the deployed model is refitted on."""
    return panel_h[panel_h['TargetIdx'] <= max_settled_target_idx]


def deployment_split_shared(panel_all: pd.DataFrame, val_months: int = 6, horizons=range(1, 7),
        max_settled_target_idx: int = None):
    """Approach 2 deployment split: train/val only, pooled across horizons."""
    max_target = max_settled_target_idx if max_settled_target_idx is not None else panel_all['TargetIdx'].max()
    eligible = panel_all[panel_all['TargetIdx'] <= max_target]
    eligible_targets = sorted(eligible['TargetIdx'].unique())
    val_cut = eligible_targets[-1] - val_months

    train = eligible[eligible['TargetIdx'] <= val_cut]
    val = eligible[eligible['TargetIdx'] > val_cut]
    return train, val


def walk_forward_shared(panel_all: pd.DataFrame, initial_train_origins: int = 60, val_months: int = 3,
        horizons=range(1, 7), max_settled_target_idx: int = None, max_folds: int = None,
        min_origin_months_since_launch: int = None):
    """Approach 2 walk-forward: each fold predicts all horizons from one origin. Returns a FoldSequence."""
    origins = sorted(panel_all['OriginIdx'].unique())
    max_target = max_settled_target_idx if max_settled_target_idx is not None else panel_all['TargetIdx'].max()
    start_i = initial_train_origins if max_folds is None else max(initial_train_origins, len(origins) - max_folds)

    specs = []
    for i in range(start_i, len(origins)):
        test_origin = origins[i]
        available_h = [h for h in horizons if test_origin + h <= max_target]
        if not available_h:
            break
        specs.append((test_origin, test_origin - val_months, tuple(available_h),
                      min_origin_months_since_launch))
    return FoldSequence(panel_all, specs, _materialize_shared_fold)


def _materialize_shared_fold(panel_all: pd.DataFrame, spec: tuple) -> dict:
    """One walk_forward_shared fold, as a dict."""
    test_origin, val_cut, available_h, min_origin_months_since_launch = spec
    known = panel_all[panel_all['TargetIdx'] <= test_origin]
    train = known[known['TargetIdx'] <= val_cut]
    val = known[known['TargetIdx'] > val_cut]
    test = panel_all[(panel_all['OriginIdx'] == test_origin) & (panel_all['Horizon'].isin(available_h))]
    test = test[scoreable_rows(test, min_origin_months_since_launch)]
    return {'train': train, 'val': val, 'test': test, 'test_origin': test_origin,
            'available_h': list(available_h)}


def test_rows_for_horizon(h: int, panel: pd.DataFrame, max_settled_target_idx: int,
        initial_train_origins: int = 60, min_origin_months_since_launch: int = None) -> pd.DataFrame:
    """All walk_forward_horizon test rows for this horizon at once, for formula baselines."""
    panel_h = panel[panel['Horizon'] == h]
    origins = sorted(panel_h['OriginIdx'].unique())
    test_origins = [o for o in origins[initial_train_origins:] if o + h <= max_settled_target_idx]
    test = panel_h[panel_h['OriginIdx'].isin(test_origins)]
    return test[scoreable_rows(test, min_origin_months_since_launch)]


def baseline_folds_for_horizon(h: int, panel: pd.DataFrame, max_settled_target_idx: int,
        initial_train_origins: int = 60, min_origin_months_since_launch: int = None):
    """One (known, test) pair per walk_forward_horizon test origin, as a FoldSequence."""
    panel_h = panel[panel['Horizon'] == h]
    origins = sorted(panel_h['OriginIdx'].unique())
    specs = []
    for test_origin in origins[initial_train_origins:]:
        if test_origin + h > max_settled_target_idx:
            break
        specs.append((test_origin, min_origin_months_since_launch))
    return FoldSequence(panel_h, specs, _materialize_baseline_fold)


def _materialize_baseline_fold(panel_h: pd.DataFrame, spec: tuple) -> tuple:
    """One baseline_folds_for_horizon fold as (known, test)."""
    test_origin, min_origin_months_since_launch = spec
    known = panel_h[panel_h['TargetIdx'] <= test_origin]
    test = panel_h[panel_h['OriginIdx'] == test_origin]
    test = test[scoreable_rows(test, min_origin_months_since_launch)]
    return known, test


def all_settled_rows_for_horizon(h: int, panel: pd.DataFrame, max_settled_target_idx: int,
        initial_train_origins: int = 60, min_origin_months_since_launch: int = None) -> pd.DataFrame:
    """Every settled, eligible row for this horizon, each with a 'Split' label."""
    panel_h = panel[panel['Horizon'] == h]
    origins = sorted(panel_h['OriginIdx'].unique())
    settled_origins = [o for o in origins if o + h <= max_settled_target_idx]
    train_origins = set(origins[:initial_train_origins])

    rows = panel_h[panel_h['OriginIdx'].isin(settled_origins)].copy()
    rows = rows[scoreable_rows(rows, min_origin_months_since_launch)]
    rows['Split'] = np.where(rows['OriginIdx'].isin(train_origins), 'train', 'test')
    return rows
