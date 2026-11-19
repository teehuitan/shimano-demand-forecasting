"""Train and persist the PlanModelCode deployment models, one per horizon, to MLflow."""
import argparse
import logging

from .config import CUTOFF_GRAINS, N_ESTIMATORS, EARLY_STOPPING_ROUNDS
from .evaluation import (
    build_covid_sample_weight, build_young_sku_sample_weight, build_external_event_sample_weight,
)
from .folds import deployment_refit_rows, deployment_split_horizon
from .logging_setup import LOG_FORMAT
from .model_params import BEST_PARAMS_BY_PLANT, BEST_PARAMS_QUANTILE_BY_PLANT
from .model_registry import deployed_state, quantile_label, resolve_deployment_grain, save_models
from .models import ADAPTERS, REFITTERS
from .order_book import frame_cutoff_grain
from .preprocessing import preprocess_for_training

log = logging.getLogger(__name__)

CATS = ['PlanModelCode', 'SalesModelCode', 'PartCode']
DEFAULT_COVID_WEIGHT = 0.1


def build_training_panel(plant, *, cutoff_grain, max_settled_target=None, ctx=None):
    """Training panel for one plant at `cutoff_grain` -> (panel, feature_cols, max_settled_idx)."""
    if cutoff_grain not in CUTOFF_GRAINS:
        raise ValueError(
            f"Unknown cutoff_grain {cutoff_grain!r} - expected one of {list(CUTOFF_GRAINS)}")
    panel, _feature_families, feature_cols, ctx = preprocess_for_training(
        plant, max_settled_target=max_settled_target, ctx=ctx, cutoff_grain=cutoff_grain)
    # What gets recorded is read off the panel, not the request - and the two must agree.
    built_grain = frame_cutoff_grain(panel)
    if built_grain != cutoff_grain:
        raise ValueError(
            f"requested a {cutoff_grain!r}-grain training panel but got a {built_grain!r}-grain "
            f"one for plant {plant} - refusing to train models under the wrong grain.")
    # Same boundary build_order_book_context resolved the panel with - never recomputed
    # separately, or the split cutoff could silently drift from what the panel itself used.
    return panel, feature_cols, ctx['max_settled_idx']


def fit_horizon_models(panel, feature_cols, max_settled_idx, best_params, *,
        covid_weight=DEFAULT_COVID_WEIGHT, val_months=6, verbose=None) -> list:
    """Per horizon: early-stop on the last `val_months`, then refit on all settled rows for that many
    rounds. Returns [(horizon, spec, model, info, eval_history)]."""
    fitted = []
    for h in range(1, 7):
        panel_h = panel[panel['Horizon'] == h]
        train_df, val_df = deployment_split_horizon(panel_h, max_settled_idx, val_months=val_months)

        spec = best_params[h]
        adapter = ADAPTERS[spec['model']]
        weight_fn = lambda df: (
            build_covid_sample_weight(df, covid_weight)
            * build_young_sku_sample_weight(df, boost_factor=spec['young_sku_boost'])
            * build_external_event_sample_weight(df, spec['external_event_weight'])
        )
        # val_df passed again as `test`: the deployment path has no held-out test region, just
        # train/val for early stopping - wastes one redundant predict() call but avoids a
        # test=None special case in every adapter.
        _, meta = adapter(train_df, val_df, val_df, feature_cols, CATS, spec['params'],
            N_ESTIMATORS, EARLY_STOPPING_ROUNDS, target_col='Adjustment', weight_fn=weight_fn,
            verbose=verbose)
        info = {'n_train': len(train_df), 'n_val': len(val_df), 'n_rounds': int(meta['n_rounds'])}
        eval_history = meta.get('eval_history')
        del train_df, val_df, meta      # the refit rows are both again - don't hold three copies

        refit_df = deployment_refit_rows(panel_h, max_settled_idx)
        model = REFITTERS[spec['model']](refit_df, feature_cols, CATS, spec['params'],
            info['n_rounds'], target_col='Adjustment', weight_fn=weight_fn)
        info['n_refit'] = len(refit_df)
        fitted.append((h, spec, model, info, eval_history))
    return fitted


def _save_specs(fitted, *, label, feature_cols, max_settled_idx, cutoff_grain, covid_weight,
        details) -> list:
    """fit_horizon_models output -> model_registry.save_models items for `label`."""
    return [dict(model=model, horizon=h, feature_cols=feature_cols, params=spec['params'],
                 covid_weight=covid_weight, young_sku_weight=spec['young_sku_boost'],
                 external_event_weight=spec['external_event_weight'], label=label,
                 details=details, trained_through=max_settled_idx, model_type=spec['model'],
                 cutoff_grain=cutoff_grain, training=info, eval_history=eval_history)
            for h, spec, model, info, eval_history in fitted]


def _log_fitted(label, fitted, grain, entries):
    for (h, spec, _m, info, _e), entry in zip(fitted, entries):
        log.info("Trained %r horizon N+%d (%s, %s grain): early stopping on train=%d val=%d chose "
            "%d round(s); deployed model refitted on all %d settled row(s) -> %s v%s", label, h,
            spec['model'], grain, info['n_train'], info['n_val'], info['n_rounds'],
            info['n_refit'], entry['registered_name'], entry['version'])


def train_and_save(plant, label, details=None, covid_weight=DEFAULT_COVID_WEIGHT, val_months=6,
        verbose=None, max_settled_target=None, best_params=None, experiment_name=None,
        ctx=None, *, cutoff_grain):
    """Builds the panel, fits every horizon for one label and saves them as one deployment."""
    if best_params is None:
        best_params = BEST_PARAMS_BY_PLANT[plant]
    panel, feature_cols, max_settled_idx = build_training_panel(
        plant, cutoff_grain=cutoff_grain, max_settled_target=max_settled_target, ctx=ctx)
    fitted = fit_horizon_models(panel, feature_cols, max_settled_idx, best_params,
        covid_weight=covid_weight, val_months=val_months, verbose=verbose)
    entries = save_models(plant=plant, experiment_name=experiment_name, models=_save_specs(
        fitted, label=label, feature_cols=feature_cols, max_settled_idx=max_settled_idx,
        cutoff_grain=cutoff_grain, covid_weight=covid_weight, details=details))
    _log_fitted(label, fitted, cutoff_grain, entries)
    return entries


def train_deployment(plant, label, *, cutoff_grain, details=None,
        covid_weight=DEFAULT_COVID_WEIGHT, val_months=6, verbose=None, max_settled_target=None,
        best_params=None, best_params_quantile=None, experiment_name=None, ctx=None) -> list:
    """Trains `label` and its quantile companion from one panel and saves them as one deployment."""
    if best_params is None:
        best_params = BEST_PARAMS_BY_PLANT[plant]
    if best_params_quantile is None:
        best_params_quantile = BEST_PARAMS_QUANTILE_BY_PLANT[plant]
    panel, feature_cols, max_settled_idx = build_training_panel(
        plant, cutoff_grain=cutoff_grain, max_settled_target=max_settled_target, ctx=ctx)
    common = dict(covid_weight=covid_weight, val_months=val_months, verbose=verbose)
    halves = [(label, fit_horizon_models(panel, feature_cols, max_settled_idx, best_params,
                                         **common)),
              (quantile_label(label), fit_horizon_models(panel, feature_cols, max_settled_idx,
                                                         best_params_quantile, **common))]
    specs = [s for lbl, fitted in halves for s in _save_specs(
        fitted, label=lbl, feature_cols=feature_cols, max_settled_idx=max_settled_idx,
        cutoff_grain=cutoff_grain, covid_weight=covid_weight, details=details)]
    entries = save_models(plant=plant, experiment_name=experiment_name, models=specs)
    for i, (lbl, fitted) in enumerate(halves):
        _log_fitted(lbl, fitted, cutoff_grain, entries[i * 6:(i + 1) * 6])
    return entries


def _parse_args():
    p = argparse.ArgumentParser(
        description="Train and save the PlanModelCode deployment models (one per horizon).")
    p.add_argument('--plant', type=int, required=True, help="Plant code to train.")
    p.add_argument('--label', required=True, help="Label for the training run")
    p.add_argument('--params', choices=['standard', 'quantile', 'both'], default='standard',
        help="Which per-horizon spec to train: the plant's BEST_PARAMS under --label, its "
             "BEST_PARAMS_QUANTILE under the quantile companion, or 'both' - the whole deployment "
             "from one panel in one commit, which is how to change a deployment's grain.")
    p.add_argument('--details', default=None, help="Optional free-text description of this training run")
    p.add_argument('--experiment', default=None,
        help="MLflow experiment name to log into. Omit to use the fixed "
             "'{plant} - Production Models' name main.py's daily runs also use.")
    p.add_argument('--max-settled-target', type=int, default=None,
        help="Max settled target month (YYYYMM) to train on. Omit to derive it from the data "
             "(one month before the latest EntryMonth present).")
    p.add_argument('--covid-weight', type=float, default=DEFAULT_COVID_WEIGHT)
    p.add_argument('--val-months', type=int, default=6)
    p.add_argument('--cutoff-grain', default=None, choices=list(CUTOFF_GRAINS),
        help="As-of grain to train at. Omit to keep the grain --label is already deployed at "
             "(required for a label with nothing deployed yet); pass it to change grain on purpose.")
    p.add_argument('--verbose', type=int, default=None,
        help="Print train/val eval metrics every N boosting rounds during each horizon's fit. "
             "Omit for silent (default).")
    return p.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)
    # Resolved against the BASE label's deployment (which covers its quantile companion too), so
    # every half lands at the deployment's grain - exactly what main.py's retrain does.
    cutoff_grain = resolve_deployment_grain(
        deployed_state(plant=args.plant, label=args.label, allow_mixed=args.cutoff_grain is not None),
        args.cutoff_grain)
    common = dict(plant=args.plant, cutoff_grain=cutoff_grain, details=args.details,
        max_settled_target=args.max_settled_target, covid_weight=args.covid_weight,
        val_months=args.val_months, verbose=args.verbose, experiment_name=args.experiment)
    if args.params == 'both':
        train_deployment(label=args.label, **common)
        return
    # 'standard' keeps `label` bare; 'quantile' saves under model_registry.quantile_label(label),
    # where predict.py's quantile routing and deployed_state both look.
    if args.params == 'quantile':
        best_params, label = BEST_PARAMS_QUANTILE_BY_PLANT[args.plant], quantile_label(args.label)
    else:
        best_params, label = BEST_PARAMS_BY_PLANT[args.plant], args.label
    train_and_save(label=label, best_params=best_params, **common)


if __name__ == '__main__':
    main()
