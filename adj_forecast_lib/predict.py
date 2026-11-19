"""Score origin month(s) per PlanModelCode with a trained model or the manual "Recent incoming
trend" method, in long format keyed by `Method`."""
import argparse
import logging
import os

import numpy as np
import pandas as pd

from .config import CUTOFF_GRAINS, month_idx_to_yyyymm
from .cutoff_grain import cutoff_day_index
from .logging_setup import LOG_FORMAT
from .model_params import QUANTILE_PARTCODES_BY_PLANT
from .model_registry import (
    check_cutoff_grain, deployed_state, load_model, quantile_label, resolve_deployment_grain,
)
from .models import PREDICT_FUNCS
from .order_book import frame_cutoff_grain
from .preprocessing import preprocess_for_inference, preprocess_for_inference_range

log = logging.getLogger(__name__)

ENTITY_TYPE = 'PlanModelCode'
# CutoffDayIndex says WHICH as-of point produced a row (0 = the origin month's end, the only point
# month grain has; 1..30 = that day of the month after, at day grain), and KnownCanceledOrder is
# read at that same point beside KnownOrder by build_inference_skeleton, so a consumer can net
# orders without re-deriving an as-of point it never saw.
OUTPUT_COLS = ['PartCode', 'SalesModelCode', ENTITY_TYPE, 'TargetYYYYMM', 'OriginYYYYMM',
               'CutoffDayIndex', 'Horizon', 'KnownOrder', 'KnownCanceledOrder', 'Predicted', 'Method']
# Upsert identity - Method is part of the key, since long format lets multiple methods share the
# same (entity, origin, target) as separate rows; CutoffDayIndex too, since a day-grain range
# scores one origin at several as-of points. Floats (KnownOrder/Predicted) excluded on purpose.
UPSERT_KEY = ['PartCode', ENTITY_TYPE, 'TargetYYYYMM', 'OriginYYYYMM', 'CutoffDayIndex', 'Method']


def _load_models(horizons, label, plant):
    log.info("Loading models: plant=%s label=%s horizons=%s", plant, label, list(horizons))
    loaded = {h: load_model(plant=plant, horizon=h, label=label) for h in horizons}
    entries = [entry for _model, _cols, entry in loaded.values()]
    deployment_ids = sorted({str(e['deployment_id']) for e in entries})
    mixed = len(deployment_ids) > 1    # horizons from different deployments - name each one's

    def _describe(e):
        through = e['trained_through']
        through = int(month_idx_to_yyyymm(through)) if through is not None else 'unknown'
        return (f"h{e['horizon']} v{e['version']} (trained through {through}"
                + (f", deployment_id={e['deployment_id']})" if mixed else ")"))

    log.info("Scoring with plant=%s label=%s deployment_id=%s: %s", plant, label,
        'MIXED ' + ', '.join(deployment_ids) if mixed else deployment_ids[0],
        ', '.join(_describe(e) for e in entries))
    return loaded


def _finalize(skeleton: pd.DataFrame, method: str) -> pd.DataFrame:
    """Shared reshape step - `skeleton` must already have PredictedAdjustment attached. `method`
    becomes every row's Method value (long format: rows, not columns, distinguish methods)."""
    out = skeleton.copy()
    out['Predicted'] = out['KnownOrder'] + out['PredictedAdjustment']
    out['OriginYYYYMM'] = month_idx_to_yyyymm(out['OriginIdx'].to_numpy())
    out['TargetYYYYMM'] = month_idx_to_yyyymm(out['TargetIdx'].to_numpy())
    out['CutoffDayIndex'] = cutoff_day_index(out)
    out['Method'] = method
    log.info("Method=%s: %d row(s), %d distinct %s", method, len(out), out[ENTITY_TYPE].nunique(), ENTITY_TYPE)
    return out[OUTPUT_COLS].sort_values(
        ['PartCode', ENTITY_TYPE, 'OriginYYYYMM', 'TargetYYYYMM']).reset_index(drop=True)


def _floor_predicted_net(pred: np.ndarray, known_net: np.ndarray, method: str, horizon: int) -> np.ndarray:
    """Floors Adjustment at -known net so the predicted net order is never below 0."""
    floor = -known_net
    below = pred < floor
    if below.any():
        net = known_net + pred
        log.info("%s h%d: %d of %d predicted net order(s) were below 0 (lowest %.1f) and are "
            "floored to 0, adding %.0f unit(s) in all.", method, horizon, int(below.sum()),
            len(pred), float(net.min()), float(-net[below].sum()))
    return np.maximum(pred, floor)


def _score_with_models(skeleton, horizons, loaded, method) -> pd.DataFrame:
    """`method` is the exact, already-built Method value for every row this call produces (e.g.
    'model_v1' or 'model_v1_quantile') - callers decide what that string is, this just scores."""
    # Checked per horizon: each horizon is its own model version, and a partial retrain could
    # leave them at different grains. The grain is read off the skeleton itself.
    skeleton_grain = frame_cutoff_grain(skeleton)

    frames = []
    for h in horizons:
        model, feature_cols, entry = loaded[h]
        check_cutoff_grain(entry, skeleton_grain)
        # Each horizon can be a different model family (model_params) - model_type picks the
        # matching predict function, since xgboost/lightgbm/catboost aren't call-compatible.
        predict_fn = PREDICT_FUNCS[entry.get('model_type', 'xgboost')]
        sub = skeleton[skeleton['Horizon'] == h].copy()
        # A half with no rows is normal (a plant with none of the quantile PartCodes, or a --codes
        # filter) but must not reach the model: XGBoost answers zero rows with a (0, 0) array that
        # corrupts the frame. Any other shape mismatch is refused rather than broadcast.
        pred = np.asarray(predict_fn(model, sub[feature_cols])) if len(sub) else np.empty(0)
        pred = pred.reshape(-1)
        if len(pred) != len(sub):
            raise ValueError(f"model h{h} returned {pred.shape} predictions for {len(sub)} row(s)")
        known_net = (sub['KnownOrder'] - sub['KnownCanceledOrder']).to_numpy()
        sub['PredictedAdjustment'] = _floor_predicted_net(pred, known_net, method, h)
        frames.append(sub)
    return _finalize(pd.concat(frames, ignore_index=True), method)


def score_manual(skeleton, window) -> pd.DataFrame:
    """The business's ManualSim{window} method, scored off an already-built skeleton."""
    skeleton = skeleton.copy()
    skeleton['PredictedAdjustment'] = skeleton[f'ManualSim{window}']
    return _finalize(skeleton, f'manual_{window}')


def score_models(skeleton, horizons, label, plant, quantile_partcodes=None) -> pd.DataFrame:
    """Scores `skeleton`: quantile_partcodes go to the quantile models, the rest to `label`'s."""
    if quantile_partcodes is None:
        quantile_partcodes = QUANTILE_PARTCODES_BY_PLANT[plant]
    q_label = quantile_label(label)
    loaded_standard = _load_models(horizons, label, plant)
    loaded_quantile = _load_models(horizons, q_label, plant)
    is_quantile = skeleton['PartCode'].isin(quantile_partcodes)

    result_standard = _score_with_models(skeleton[~is_quantile], horizons, loaded_standard, f'model_{label}')
    result_quantile = _score_with_models(skeleton[is_quantile], horizons, loaded_quantile, f'model_{q_label}')
    return pd.concat([result_standard, result_quantile], ignore_index=True) \
        .sort_values(['PartCode', ENTITY_TYPE, 'OriginYYYYMM', 'TargetYYYYMM']).reset_index(drop=True)


def _deployed_grain_unless_given(plant, label, cutoff_grain):
    """An explicit grain is asserted as-is (and still checked against every model); None means
    "the grain `label` is deployed at", via the same resolver the orchestrator and train.py use."""
    if cutoff_grain is not None:
        return cutoff_grain
    return resolve_deployment_grain(deployed_state(plant=plant, label=label))


def predict(plant, origin_yyyymm=None, horizons=range(1, 7), label=None, codes=None,
        min_origin_months_since_launch=3, quantile_partcodes=None, ctx=None, return_ctx=False,
        cutoff_grain=None, cutoff_date=None) -> pd.DataFrame:
    """Scores one origin; None arguments resolve to the latest month, deployed grain and latest file."""
    horizons = list(horizons)
    cutoff_grain = _deployed_grain_unless_given(plant, label, cutoff_grain)
    skeleton, ctx = preprocess_for_inference(plant, origin_yyyymm, horizons, codes=codes,
        min_origin_months_since_launch=min_origin_months_since_launch, ctx=ctx,
        cutoff_grain=cutoff_grain, cutoff_date=cutoff_date)
    result = score_models(skeleton, horizons, label, plant, quantile_partcodes)
    return (result, ctx) if return_ctx else result


def predict_range(plant, start_origin_yyyymm, end_origin_yyyymm, horizons, label, codes=None,
        min_origin_months_since_launch=3, quantile_partcodes=None, cutoff_grain=None,
        cutoff_days=None) -> pd.DataFrame:
    """predict() for every origin month from start_origin_yyyymm to end_origin_yyyymm inclusive."""
    horizons = list(horizons)
    cutoff_grain = _deployed_grain_unless_given(plant, label, cutoff_grain)
    skeleton = preprocess_for_inference_range(plant, start_origin_yyyymm, end_origin_yyyymm,
        horizons, codes=codes, min_origin_months_since_launch=min_origin_months_since_launch,
        cutoff_grain=cutoff_grain, cutoff_days=cutoff_days)
    return score_models(skeleton, horizons, label, plant, quantile_partcodes)


def predict_manual(plant, origin_yyyymm=None, horizons=range(1, 7), window=6, codes=None,
        min_origin_months_since_launch=3, ctx=None, return_ctx=False, *, cutoff_grain,
        cutoff_date=None) -> pd.DataFrame:
    """Scores the business's "Recent incoming trend" method (ManualSim{window}); `cutoff_grain` is required."""
    skeleton, ctx = preprocess_for_inference(plant, origin_yyyymm, list(horizons), codes=codes,
        min_origin_months_since_launch=min_origin_months_since_launch, ctx=ctx,
        cutoff_grain=cutoff_grain, cutoff_date=cutoff_date)
    result = score_manual(skeleton, window)
    return (result, ctx) if return_ctx else result


def predict_manual_range(plant, start_origin_yyyymm, end_origin_yyyymm, horizons, window=6, codes=None,
        min_origin_months_since_launch=3, *, cutoff_grain, cutoff_days=None) -> pd.DataFrame:
    """predict_manual() over a range of origins - see predict_range."""
    skeleton = preprocess_for_inference_range(plant, start_origin_yyyymm, end_origin_yyyymm,
        list(horizons), codes=codes, min_origin_months_since_launch=min_origin_months_since_launch,
        cutoff_grain=cutoff_grain, cutoff_days=cutoff_days)
    return score_manual(skeleton, window)


def merge_prediction_column(result: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """Upserts `result` into the long-format Parquet at `output_path`, keyed on UPSERT_KEY."""
    # Defensive: de-dupe `result` on its own key first (keep last) - shouldn't ever fire given how
    # score_models partitions PartCodes, but guarantees one row per key even if that changes.
    result = result.drop_duplicates(subset=UPSERT_KEY, keep='last')

    # Parent dir (e.g. results/STJ) may not exist yet - to_parquet itself never creates directories.
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    if not os.path.exists(output_path):
        combined = result.sort_values(UPSERT_KEY).reset_index(drop=True)
        log.info("%r did not exist - seeded with %d row(s).", output_path, len(combined))
        combined.to_parquet(output_path, index=False)
        return combined

    existing = pd.read_parquet(output_path)
    if 'CutoffDayIndex' not in existing.columns:
        existing['CutoffDayIndex'] = np.int16(0)
        log.info("%r predates CutoffDayIndex - reading its %d row(s) as day 0 (month grain).",
            output_path, len(existing))
    existing['CutoffDayIndex'] = existing['CutoffDayIndex'].astype(result['CutoffDayIndex'].dtype)
    key_check = existing.merge(result[UPSERT_KEY].drop_duplicates(), on=UPSERT_KEY,
        how='left', indicator=True)
    n_replaced = int((key_check['_merge'] == 'both').sum())
    existing_kept = existing[key_check['_merge'].to_numpy() == 'left_only']
    combined = pd.concat([existing_kept, result], ignore_index=True) \
        .sort_values(UPSERT_KEY).reset_index(drop=True)

    log.info("Upserted into %r: %d row(s) replaced, %d new row(s) added (%d total rows).",
        output_path, n_replaced, len(result) - n_replaced, len(combined))
    combined.to_parquet(output_path, index=False)
    return combined


def _parse_args():
    p = argparse.ArgumentParser(
        description="Score PlanModelCode for one origin month or a range, via a trained model "
                     "or the manual method - one method per run.")
    p.add_argument('--plant', type=int, required=True, help='Plant code to score.')
    p.add_argument('--method', required=True, choices=['model', 'manual'],
        help="'model' loads a trained model (needs --label); 'manual' uses ManualSim{--manual-window}, "
             "no trained model needed.")
    p.add_argument('--label', default=None,
        help="Training run label to load (see train.py --label). Required for --method model.")
    p.add_argument('--manual-window', type=int, default=6, choices=[3, 6, 12],
        help="ManualSim window to use for --method manual (default: 6).")
    p.add_argument('--output', default=None,
        help="Parquet path to upsert this run's rows into (see merge_prediction_column) - "
             "printed to stdout instead if omitted.")
    p.add_argument('--origin', type=int, default=None,
        help="Single origin month as YYYYMM, e.g. 202607. Omit (along with --start-origin/"
             "--end-origin) to default to the latest complete month. Mutually exclusive with "
             "--start-origin/--end-origin.")
    p.add_argument('--start-origin', type=int, default=None,
        help="First origin month (YYYYMM) of a range to score. Requires --end-origin.")
    p.add_argument('--end-origin', type=int, default=None,
        help="Last origin month (YYYYMM, inclusive) of a range to score. Requires --start-origin.")
    p.add_argument('--horizons', default='1,2,3,4,5,6', help="Comma-separated horizons, e.g. 1,2,3.")
    p.add_argument('--codes', default=None,
        help="Comma-separated entity codes to restrict to (default: every entity active at each origin).")
    p.add_argument('--min-origin-months-since-launch', type=int, default=3,
        help="Drop rows younger than this many months since launch (TFGMaterial launch, else first "
             "order) - business doesn't need a forecast until then. Default: 3.")
    p.add_argument('--quantile-partcodes', default=None,
        help="Comma-separated PartCodes always scored with the quantile companion models instead "
             "of --label's standard models. Only used with --method model. Default: the plant's "
             "own (model_params.QUANTILE_PARTCODES_BY_PLANT).")
    p.add_argument('--cutoff-grain', default=None, choices=list(CUTOFF_GRAINS),
        help="As-of grain. --method model: omit to score at the grain --label is deployed at. "
             "--method manual: required (there is no deployment to read it from).")
    p.add_argument('--cutoff-date', default=None,
        help="Day grain, single origin: the as-of date (YYYY-MM-DD). Omit for the day before the "
             "latest source file's date.")
    p.add_argument('--cutoff-days', default=None,
        help="Day grain, --start-origin/--end-origin only: comma-separated cutoff days to score "
             "each origin at, 0 = the origin month's end, 1..30 = that day of the next month "
             "(e.g. 0,15,30). Required there.")
    args = p.parse_args()

    is_range = args.start_origin is not None or args.end_origin is not None
    if args.origin is not None and is_range:
        p.error("--origin cannot be combined with --start-origin/--end-origin.")
    if is_range and (args.start_origin is None or args.end_origin is None):
        p.error("--start-origin and --end-origin must be given together.")
    if args.method == 'model' and args.label is None:
        p.error("--method model requires --label.")
    if args.method == 'manual' and args.cutoff_grain is None:
        p.error("--method manual requires --cutoff-grain month|day.")
    if args.cutoff_date is not None and is_range:
        p.error("--cutoff-date is for a single origin; use --cutoff-days with a range.")
    if args.cutoff_days is not None and not is_range:
        p.error("--cutoff-days is for --start-origin/--end-origin; use --cutoff-date otherwise.")
    return args


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)
    horizons = [int(h) for h in args.horizons.split(',') if h.strip()]
    codes = [c.strip() for c in args.codes.split(',') if c.strip()] if args.codes else None

    # Range only when --start-origin/--end-origin were actually given - omitting all three means
    # "single prediction, latest complete month".
    is_range = args.start_origin is not None or args.end_origin is not None
    cutoff_date = pd.Timestamp(args.cutoff_date) if args.cutoff_date else None
    cutoff_days = ([int(d) for d in args.cutoff_days.split(',') if d.strip()]
                   if args.cutoff_days else None)
    grain = dict(cutoff_grain=args.cutoff_grain)
    common = dict(codes=codes, min_origin_months_since_launch=args.min_origin_months_since_launch)

    if args.method == 'model':
        quantile_partcodes = (tuple(c.strip() for c in args.quantile_partcodes.split(',') if c.strip())
                              if args.quantile_partcodes else None)
        if is_range:
            result = predict_range(args.plant, args.start_origin, args.end_origin, horizons, args.label,
                quantile_partcodes=quantile_partcodes, cutoff_days=cutoff_days, **common, **grain)
        else:
            result = predict(args.plant, args.origin, horizons, args.label,
                quantile_partcodes=quantile_partcodes, cutoff_date=cutoff_date, **common, **grain)
    else:
        if is_range:
            result = predict_manual_range(args.plant, args.start_origin, args.end_origin, horizons,
                window=args.manual_window, cutoff_days=cutoff_days, **common, **grain)
        else:
            result = predict_manual(args.plant, args.origin, horizons, window=args.manual_window,
                cutoff_date=cutoff_date, **common, **grain)

    if args.output:
        merge_prediction_column(result, args.output)
    else:
        print(result.to_string(index=False))


if __name__ == '__main__':
    main()
