"""Evaluation harness: per-fold metrics, sample-weighting, and the fit/predict wrapper shared by
the main modelling loop and the sweep scripts."""
import logging
import os

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .config import N_ESTIMATORS, EARLY_STOPPING_ROUNDS, is_covid_month_idx

from .preprocessing import _ENTITY_TYPES
from .folds import scoreable_rows

log = logging.getLogger(__name__)

# Only used by the legacy fit_predict() below - current params live in models.DEFAULT_PARAMS.
XGB_PARAMS = dict(
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    eval_metric='rmse',
    tree_method='hist',
    enable_categorical=True,
    random_state=42,
)


def fit_predict(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame = None,
        feature_cols: list = None, target_col: str = 'Adjustment', sample_weight: np.ndarray = None,
        sample_weight_val: np.ndarray = None):
    """Fits the model and predicts `test_df` (y_pred=None when test_df is None); NaNs pass through."""
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]

    model = XGBRegressor(n_estimators=N_ESTIMATORS, early_stopping_rounds=EARLY_STOPPING_ROUNDS, **XGB_PARAMS)
    # val only: early stopping reads the last eval_set entry, and nothing here prints the training
    # curve, so evaluating the training set every round was pure cost (see models.fit_predict_xgboost).
    fit_kwargs = dict(eval_set=[(X_val, y_val)], verbose=False)
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = sample_weight
        fit_kwargs['sample_weight_eval_set'] = [sample_weight_val]
    model.fit(X_train, y_train, **fit_kwargs)
    y_pred = model.predict(test_df[feature_cols]) if test_df is not None else None
    return y_pred, model


def build_covid_sample_weight(df: pd.DataFrame, covid_weight: float) -> np.ndarray:
    """covid_weight for rows whose target month is in COVID_YYYYMM_RANGE, 1.0 otherwise."""
    return np.where(is_covid_month_idx(df['TargetIdx']), covid_weight, 1.0)

def build_young_sku_sample_weight(df: pd.DataFrame, boost_factor: float, months_threshold: int = 6) -> np.ndarray:
    """boost_factor where OriginMonthsSinceLaunch < months_threshold, 1.0 otherwise (NaN is not young)."""
    is_young = df['OriginMonthsSinceLaunch'].notna() & (df['OriginMonthsSinceLaunch'] <= months_threshold)
    return np.where(is_young, boost_factor, 1.0)
 
 
def build_external_event_sample_weight(df: pd.DataFrame, event_weight: float) -> np.ndarray:
    """event_weight for rows whose target month is an event month (hindsight), 1.0 otherwise. COVID is also
    an event, so there the two weights multiply (covid_weight=1.0 weights every event the same)."""
    # Panels before train11 have no EventTargetHindsight; their HasExternalEvent was the hindsight flag.
    col = 'EventTargetHindsight' if 'EventTargetHindsight' in df.columns else 'HasExternalEvent'
    return np.where(df[col] == 1, event_weight, 1.0)


def _raw_accumulators(y_t: np.ndarray, y_p: np.ndarray, mask: np.ndarray) -> dict:
    """Undivided sums/counts behind MAE/RMSE/MAPE, so folds pool exactly."""
    n = int(mask.sum())
    if n == 0:
        return {'sum_err': 0.0, 'sum_abs_err': 0.0, 'sum_sq_err': 0.0, 'sum_abs_pct_err': 0.0,
            'sum_abs_actual': 0.0, 'n': 0, 'n_nonzero': 0, 'n_zero_actual': 0}
    yt, yp = y_t[mask], y_p[mask]
    err = yt - yp
    nonzero = yt != 0
    return {
        'sum_err': float(np.sum(err)),
        'sum_abs_err': float(np.sum(np.abs(err))),
        'sum_sq_err': float(np.sum(err ** 2)),
        'sum_abs_pct_err': float(np.sum(np.abs(err[nonzero] / yt[nonzero]))) if nonzero.any() else 0.0,
        # unlike sum_abs_pct_err, this is summed over ALL rows (not just nonzero-actual) - WAPE's
        # denominator naturally handles zero-actual rows (they contribute 0 to both numerator and
        # denominator) without needing a separate exclusion mask the way MAPE does.
        'sum_abs_actual': float(np.sum(np.abs(yt))),
        'n': n,
        'n_nonzero': int(nonzero.sum()),
        'n_zero_actual': int((~nonzero).sum()),
    }


def evaluate_adjustment_raw(train_df: pd.DataFrame, test_df: pd.DataFrame, y_pred: np.ndarray,
        entity_col: str = 'PlanModelCode', target_col: str = 'Adjustment',
        baseline_col: str = 'KnownOrder', floor_net: bool = True) -> dict:
    """Raw per-fold accumulators for forecast/adjustment views and overall/seen splits.
    `floor_net` applies production's floor - predicted net >= 0, i.e. the forecast never below
    KnownCanceledOrder - and needs that column (False for the manual baseline)."""
    baseline = test_df[baseline_col].to_numpy()
    actual = test_df['ActualOrder'].to_numpy()
    y_true_adjustment = actual - baseline
    if target_col == 'ActualOrder':
        forecast = np.asarray(y_pred)
        y_pred_adjustment = forecast - baseline
    else:
        y_pred_adjustment = np.asarray(y_pred)
        forecast = baseline + y_pred_adjustment
    if floor_net:
        if 'KnownCanceledOrder' not in test_df:
            raise ValueError("floor_net needs a KnownCanceledOrder column - this panel predates it; "
                             "rebuild it with build_full_panel.")
        floor = test_df['KnownCanceledOrder'].to_numpy()
        # only touch floored rows - recomputing (baseline + pred) - baseline everywhere loses float32 exactness
        below = forecast < floor
        forecast = np.where(below, floor, forecast)
        y_pred_adjustment = np.where(below, floor - baseline, y_pred_adjustment)

    seen_codes = set(train_df[entity_col])
    seen_mask = test_df[entity_col].isin(seen_codes).to_numpy()

    out = {}
    for name, (yt, yp) in {
        'adjustment': (y_true_adjustment, y_pred_adjustment),
        'forecast': (actual, forecast),
    }.items():
        overall = _raw_accumulators(yt, np.asarray(yp), np.ones(len(yt), dtype=bool))
        seen = _raw_accumulators(yt, np.asarray(yp), seen_mask)
        out.update({f'{name}_{k}': v for k, v in overall.items()})
        out.update({f'{name}_seen_{k}': v for k, v in seen.items()})
    return out


def evaluate_adjustment_raw_by_day(train_df: pd.DataFrame, test_df: pd.DataFrame,
        y_pred: np.ndarray, day_index, entity_col: str = 'PlanModelCode',
        target_col: str = 'Adjustment', floor_net: bool = True) -> dict:
    """evaluate_adjustment_raw() per cutoff day for one fold, keyed by day; {0: raw} at month grain."""
    day_index = np.asarray(day_index)
    if len(day_index) != len(test_df):
        raise ValueError(f"day_index has {len(day_index)} value(s) but test_df has {len(test_df)} "
                         "row(s) - pass cutoff_day_index(test_df), aligned to this fold's test rows.")
    y_pred = np.asarray(y_pred)
    out = {}
    for day in np.unique(day_index):
        sel = day_index == day
        out[int(day)] = evaluate_adjustment_raw(train_df, test_df[sel], y_pred[sel],
            entity_col=entity_col, target_col=target_col, floor_net=floor_net)
    return out


def pooled_metrics_by_day(per_fold_by_day: list) -> pd.DataFrame:
    """Pools per-fold evaluate_adjustment_raw_by_day() dicts into one metrics row per cutoff day."""
    by_day = {}
    for fold in per_fold_by_day:
        for day, raw in fold.items():
            by_day.setdefault(day, []).append(raw)
    rows = {day: finalize_metrics(combine_raw_accumulators(raws)) for day, raws in by_day.items()}
    out = pd.DataFrame.from_dict(rows, orient='index').sort_index()
    out.index.name = 'day'
    return out


def log_day_curve_to_mlflow(per_fold_by_day: list, key: str = 'forecast_wape') -> pd.DataFrame:
    """Logs the pooled_metrics_by_day() curve as mlflow metric `{key}_by_day` (step = day); returns it."""
    import mlflow

    curve = pooled_metrics_by_day(per_fold_by_day)
    for day, value in curve[key].items():
        if not np.isnan(value):
            mlflow.log_metric(f'{key}_by_day', float(value), step=int(day))
    return curve


def combine_raw_accumulators(raw_list: list) -> dict:
    """Sums evaluate_adjustment_raw() dicts (one per fold) key-by-key - plain addition, never
    averaging, since these are all sums/counts."""
    keys = raw_list[0].keys()
    return {k: sum(r[k] for r in raw_list) for k in keys}


def fold_metrics_list(raw_list: list) -> list:
    """One finalize_metrics() dict per fold (not pooled) - use with fold_spread() to check how
    stable a metric is across folds, which the single pooled number can't show."""
    return [finalize_metrics(raw) for raw in raw_list]


def fold_spread(fold_metrics: list, key: str) -> dict:
    """Unweighted mean/std of one metric across folds, NaN folds dropped."""
    vals = np.array([fm[key] for fm in fold_metrics if not np.isnan(fm[key])])
    if len(vals) == 0:
        return {f'{key}_fold_mean': np.nan, f'{key}_fold_std': np.nan}
    return {
        f'{key}_fold_mean': float(np.mean(vals)),
        f'{key}_fold_std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
    }


def _finalize_one(acc: dict, prefix: str) -> dict:
    n = acc[f'{prefix}_n']
    mae = acc[f'{prefix}_sum_abs_err'] / n if n else np.nan
    rmse = np.sqrt(acc[f'{prefix}_sum_sq_err'] / n) if n else np.nan
    n_nonzero = acc[f'{prefix}_n_nonzero']
    mape = acc[f'{prefix}_sum_abs_pct_err'] / n_nonzero * 100 if n_nonzero else np.nan
    sum_abs_actual = acc[f'{prefix}_sum_abs_actual']
    # volume-weighted MAPE (sum|error| / sum|actual|) - unlike the plain MAPE above, not
    # dominated by low-volume rows' relative error, and needs no separate zero-actual exclusion.
    wape = acc[f'{prefix}_sum_abs_err'] / sum_abs_actual * 100 if sum_abs_actual else np.nan
    # signed counterpart to wape (actual - predicted, not abs) - net over/under-forecast tilt.
    # positive = under-forecasting, negative = over-forecasting; near-zero only means the *net*
    # tilt across everything pooled into this accumulator is small, not that every row is unbiased.
    bias = acc[f'{prefix}_sum_err'] / sum_abs_actual * 100 if sum_abs_actual else np.nan
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'wape': wape, 'bias': bias, 'n': n,
        'n_zero_actual': acc[f'{prefix}_n_zero_actual']}


def finalize_metrics(acc: dict) -> dict:
    """Turns (possibly fold-combined) raw accumulators into final mae/rmse/mape/wape/n/
    n_zero_actual/n_unseen, for both adjustment/forecast views and overall/seen-entity splits."""
    out = {}
    for name in ('adjustment', 'forecast'):
        overall = _finalize_one(acc, name)
        seen = _finalize_one(acc, f'{name}_seen')
        out.update({f'{name}_{k}': v for k, v in overall.items()})
        out.update({f'{name}_seen_{k}': v for k, v in seen.items()})
        out[f'{name}_n_unseen'] = overall['n'] - seen['n']
    return out


def evaluate_adjustment(train_df: pd.DataFrame, test_df: pd.DataFrame, y_pred: np.ndarray,
        entity_col: str = 'PlanModelCode', target_col: str = 'Adjustment') -> dict:
    """MAE/RMSE/MAPE/WAPE for a single fold/split."""
    return finalize_metrics(evaluate_adjustment_raw(train_df, test_df, y_pred, entity_col=entity_col,
        target_col=target_col))


def compute_wape(y_true, y_pred) -> float:
    """WAPE (sum|error| / sum|actual| * 100) for one population; NaN if all actuals are zero."""
    y_true = np.asarray(y_true, dtype='float64')
    y_pred = np.asarray(y_pred, dtype='float64')
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def log_predictions_to_overall(entity_type: str, pred_df: pd.DataFrame, res_path: str, label: str) -> pd.DataFrame:
    """Left-joins one experiment's predictions into the overall results table as `label`'s columns."""
    if entity_type not in _ENTITY_TYPES:
        raise ValueError(f"entity_type must be one of {_ENTITY_TYPES}, got {entity_type!r}")

    key_cols = [entity_type, 'OriginIdx', 'TargetIdx']
    required = key_cols + ['Predicted', 'Split']
    missing = [c for c in required if c not in pred_df.columns]
    if missing:
        raise ValueError(f"pred_df is missing required column(s): {missing}")

    dup_mask = pred_df.duplicated(subset=key_cols, keep=False)
    if dup_mask.any():
        raise ValueError(f"pred_df has {int(dup_mask.sum())} row(s) sharing a duplicate "
            f"{key_cols} key - expected exactly one row per (entity, OriginIdx, TargetIdx).")

    val_col = f'{label}_Val'
    split_col = f'{label}_Split'
    new_cols = pred_df[required].rename(columns={'Predicted': val_col, 'Split': split_col})

    overall = pd.read_parquet(res_path)

    key_check = new_cols[key_cols].merge(overall[key_cols].drop_duplicates(), on=key_cols,
        how='left', indicator=True)
    unmatched = int((key_check['_merge'] == 'left_only').sum())
    if unmatched:
        log.warning("%d row(s) in pred_df for label %r had no matching key in %r and were "
            "dropped by the left join.", unmatched, label, res_path)

    # if this label was logged before, keep its prior values under temp names so the fresh merge
    # can never collide, then coalesce: new (non-null) value wins, else fall back to the prior one
    prev_val_col, prev_split_col = f'__prev_{val_col}', f'__prev_{split_col}'
    had_val = val_col in overall.columns
    had_split = split_col in overall.columns
    overall = overall.rename(columns={
        **({val_col: prev_val_col} if had_val else {}),
        **({split_col: prev_split_col} if had_split else {}),
    })

    merged = overall.merge(new_cols, on=key_cols, how='left')
    if had_val:
        merged[val_col] = merged[val_col].fillna(merged[prev_val_col])
        merged = merged.drop(columns=[prev_val_col])
    if had_split:
        merged[split_col] = merged[split_col].fillna(merged[prev_split_col])
        merged = merged.drop(columns=[prev_split_col])

    merged.to_parquet(res_path, index=False)
    log.info("Merged predictions for label %r into %s", label, res_path)
    return merged


def seed_overall_results(panel_path: str, res_path: str = None,
        min_origin_months_since_launch: int = None) -> pd.DataFrame:
    """Seeds the overall results table with the panel's eligible key rows."""
    filename = os.path.basename(panel_path)
    entity_cols = ['PlanModelCode', 'PartCode']

    if res_path is None:
        parts = filename.split('_')
        if len(parts) < 2:
            raise ValueError(
                f"Can't derive res_path from panel_path filename {filename!r} - expected an "
                "'<entity_type>_..._<...>' naming pattern; pass res_path explicitly instead.")
        res_path = f"results/{parts[0]}_res_{parts[-1]}"

    panel = pd.read_parquet(panel_path)
    key_cols = entity_cols + ['Horizon', 'OriginIdx', 'TargetIdx']
    missing = [c for c in key_cols if c not in panel.columns]
    if missing:
        raise ValueError(f"panel at {panel_path!r} is missing required column(s): {missing}")

    # The same test the folds and production inference apply (folds.scoreable_rows), so the seeded
    # key universe is the population every scored row comes from.
    before = len(panel)
    panel = panel[scoreable_rows(panel, min_origin_months_since_launch)].copy()
    log.info("Dropped %d row(s) outside the lifecycle or below %s month(s) since launch before seeding.",
        before - len(panel), min_origin_months_since_launch)

    seed = panel[key_cols].copy()
    seed['ActualOrder'] = panel['ActualOrder'] if 'ActualOrder' in panel.columns else np.nan
    seed['KnownOrder'] = panel['KnownOrder'] if 'KnownOrder' in panel.columns else np.nan

    seed.to_parquet(res_path, index=False)
    log.info("Wrote %d row(s) to %s", len(seed), res_path)
    return seed

def log_sic_pred_to_overall(sic_path: str, res_path: str, label: str = None) -> pd.DataFrame:
    """Logs a SIC prediction file to the overall table; label=None derives it from the filename."""
    filename = os.path.basename(sic_path)
    parts = filename.split('_')
    if len(parts) < 3:
        raise ValueError(
            f"Can't derive entity_type from panel_path filename {filename!r} - expected an "
            "'<entity_type>_..._<...>' naming pattern; pass entity_type explicitly instead.")
    entity_type = parts[0]
    if entity_type not in _ENTITY_TYPES:
        raise ValueError(f"entity_type from filename {filename!r} must be one of {_ENTITY_TYPES}, got {entity_type!r}")
    if label is None:
        label = parts[1] if parts[1].startswith('sic') else 'sic'
    sic_df = pd.read_parquet(sic_path)
    sic_df['Split'] = None
    return log_predictions_to_overall(entity_type, sic_df, res_path, label)


METRIC_KEYS = [
    'forecast_mae', 'forecast_rmse', 'forecast_mape', 'forecast_wape', 'forecast_bias',
    'forecast_wape_fold_mean', 'forecast_wape_fold_std',
]


def remaining_sweep_runs(experiment_name: str, all_sweep_runs: list, model_names) -> list:
    """Filters `all_sweep_runs` down to combos not yet logged in the `experiment_name` mlflow
    experiment, matching on (model_name, run_name) where run_name = f"{model_name}_h{h} {label}"."""
    import mlflow

    logged = mlflow.search_runs(experiment_names=[experiment_name], max_results=5000)
    # search_runs only materializes a params.* column when at least one returned run logged that
    # param, so a nonempty `logged` doesn't guarantee params.model exists - a run that died before
    # log_param('model') leaves the column out entirely.
    cols = {'params.model', 'tags.mlflow.runName'}
    if len(logged) and cols <= set(logged.columns):
        sweep_logged = logged[logged['params.model'].isin(list(model_names))]
        completed_pairs = set(zip(sweep_logged['params.model'], sweep_logged['tags.mlflow.runName']))
    else:
        completed_pairs = set()

    return [
        (model_name, h, label, overrides) for model_name, h, label, overrides in all_sweep_runs
        if (model_name, f"{model_name}_h{h} {label}") not in completed_pairs
    ]


def build_add_candidates(feature_cols: list, add_map: dict) -> dict:
    """{'baseline': feature_cols} plus one candidate per add_map entry, each adding one family."""
    candidates = {'baseline': list(feature_cols)}
    for label, add_cols in add_map.items():
        candidates[label] = list(feature_cols) + [c for c in add_cols if c not in feature_cols]
    return candidates


def build_drop_candidates(feature_cols: list, drop_map: dict) -> dict:
    """{'baseline': feature_cols} plus one candidate per (label, drop_cols) in drop_map, each with
    those columns removed. Shared by leave-one-family-out and within-family trims."""
    candidates = {'baseline': list(feature_cols)}
    for label, drop_cols in drop_map.items():
        drop_set = set(drop_cols)
        candidates[label] = [c for c in feature_cols if c not in drop_set]
    return candidates


def log_result(approach: str, horizon: int, eval_type: str, fold_data: list, feature_set: str = 'all',
        covid_weight: float = 1.0, entity_col: str = 'PlanModelCode', results_list: list = None,
        target_col: str = 'Adjustment', floor_net: bool = True) -> list:
    """Pools folds (triples or raw dicts) into one row appended to `results_list`; returns per-fold
    metrics."""
    raw = [fd if isinstance(fd, dict)
           else evaluate_adjustment_raw(*fd, entity_col=entity_col, target_col=target_col,
                                        floor_net=floor_net)
           for fd in fold_data]
    metrics = finalize_metrics(combine_raw_accumulators(raw))
    per_fold = fold_metrics_list(raw)
    spread = fold_spread(per_fold, 'forecast_wape')
    results_list.append({'approach': approach, 'horizon': horizon, 'eval_type': eval_type,
        'feature_set': feature_set, 'covid_weight': covid_weight, 'n_folds': len(fold_data),
        **metrics, **spread})
    return per_fold


def log_fold_metrics_to_mlflow(per_fold: list, key: str = 'forecast_wape') -> None:
    """Logs log_result's per-fold metrics as step-indexed mlflow metric `{key}_fold`."""
    import mlflow

    for i, fm in enumerate(per_fold):
        mlflow.log_metric(f'{key}_fold', fm[key], step=i)


def compare_runs_per_fold(experiment_name: str, candidate_run_name: str, current_best_run_name: str,
        key: str = 'forecast_wape', catastrophic_ratio: float = 2.0) -> dict:
    """Win-rate and worst-fold degradation between two runs' `{key}_fold`; raises on fold-count mismatch."""
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    runs = mlflow.search_runs(experiment_names=[experiment_name], max_results=5000)

    def _get_run_id(run_name):
        match = runs[runs['tags.mlflow.runName'] == run_name]
        if match.empty:
            raise ValueError(f"no run named {run_name!r} found in experiment {experiment_name!r}")
        return match.iloc[0]['run_id']

    cand_history = client.get_metric_history(_get_run_id(candidate_run_name), f'{key}_fold')
    best_history = client.get_metric_history(_get_run_id(current_best_run_name), f'{key}_fold')
    if not cand_history or not best_history:
        raise ValueError(f"one or both runs have no logged '{key}_fold' history - were they run "
            f"before log_fold_metrics_to_mlflow was wired in?")

    cand_by_step = {m.step: m.value for m in cand_history}
    best_by_step = {m.step: m.value for m in best_history}
    if set(cand_by_step) != set(best_by_step):
        raise ValueError(f"fold-step mismatch between runs ({len(cand_by_step)} vs "
            f"{len(best_by_step)} steps) - they weren't evaluated on the same fold universe, "
            f"comparing them fold-by-fold would silently pair up the wrong folds")

    steps = sorted(cand_by_step)
    wins = sum(1 for s in steps if cand_by_step[s] < best_by_step[s])
    ratios = [(cand_by_step[s] / best_by_step[s] if best_by_step[s] else np.inf) for s in steps]
    worst_idx = int(np.argmax(ratios))
    return {
        'n_folds': len(steps),
        'win_rate': wins / len(steps),
        'wins': wins,
        'worst_fold_ratio': ratios[worst_idx],
        'worst_fold_index': steps[worst_idx],
        'is_catastrophic': ratios[worst_idx] > catastrophic_ratio,
    }


def log_ma_baseline(label: str, forecast_fn, rows_fn, res_path: str, entity_type: str = 'PlanModelCode') -> None:
    """Scores `forecast_fn` on `rows_fn(h)` for every horizon and logs to the overall table."""
    frames = []
    for h in range(1, 7):
        rows = rows_fn(h)
        predicted = forecast_fn(rows, h)
        frames.append(pd.DataFrame({
            entity_type: rows[entity_type].to_numpy(),
            'OriginIdx': rows['OriginIdx'].to_numpy(),
            'TargetIdx': rows['TargetIdx'].to_numpy(),
            'Predicted': predicted,
            'Split': rows['Split'].to_numpy() if 'Split' in rows.columns else 'test',
        }))
    pred_df = pd.concat(frames, ignore_index=True)
    log_predictions_to_overall(entity_type, pred_df, res_path, label=label)


def log_ma_live(label: str, forecast_fn, live_rows: pd.DataFrame, res_path: str,
        entity_type: str = 'PlanModelCode') -> None:
    """log_ma_baseline for unsettled `live_rows` (Split is None)."""
    frames = []
    for h in range(1, 7):
        rows = live_rows[live_rows['Horizon'] == h]
        predicted = forecast_fn(rows, h)
        frames.append(pd.DataFrame({
            entity_type: rows[entity_type].to_numpy(),
            'OriginIdx': rows['OriginIdx'].to_numpy(),
            'TargetIdx': rows['TargetIdx'].to_numpy(),
            'Predicted': predicted,
            'Split': None,
        }))
    pred_df = pd.concat(frames, ignore_index=True)
    log_predictions_to_overall(entity_type, pred_df, res_path, label=label)


def sample_entities_per_group(df: pd.DataFrame, group_col: str, entity_col: str,
        n_per_group: int = 1, seed: int = 0) -> list:
    """Samples up to `n_per_group` distinct `entity_col` values per `group_col` value."""
    return (
        df[[group_col, entity_col]].drop_duplicates()
            .groupby(group_col, group_keys=False)
            .apply(lambda g: g.sample(min(len(g), n_per_group), random_state=seed), include_groups=False)
            [entity_col].tolist()
    )


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Construct Initial Results Table")
    p.add_argument('--panel-path', help="Path to panel parquet file.")
    p.add_argument('--res-path', default=None,
        help="Path to write the overall results parquet file. If omitted, derived from "
             "--panel-path's filename (entity_type prefix + 'res' + trailing component)."
             "Must be provided if --sic-path is given.")
    p.add_argument('--sic-path', help="Path to SIC parquet file if logging SIC predictions.")
    p.add_argument('--label', help="Label for the predictions. If omitted, derived from --sic-path's filename (middle component).")
    return p.parse_args()


def main():
    args = _parse_args()
    panel_path = args.panel_path
    res_path = args.res_path
    sic_path = args.sic_path
    if panel_path:
        seed_overall_results(args.panel_path, args.res_path)
    if sic_path:
        if not res_path:
            raise ValueError("res_path must be provided when logging SIC predictions.")
        log_sic_pred_to_overall(sic_path, res_path, args.label)

if __name__ == '__main__':
    main()
