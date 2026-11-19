"""Model persistence via the MLflow Model Registry: one registered model per (plant, horizon, label),
deployed by moving PRODUCTION_ALIAS once every version is registered.
"""
import json
import logging
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone

import mlflow
from mlflow.entities import Metric
# MlflowException, not its REST-only subclass: a local store raises the base class for the same errors.
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from .config import CUTOFF_GRAINS, MLFLOW_TRACKING_URI, plant_name

log = logging.getLogger(__name__)

# Extension each model family's native save_model()/load format conventionally uses.
_MODEL_EXT = {'xgboost': 'json', 'catboost': 'cbm', 'lightgbm': 'txt'}

_METRICS_PER_BATCH = 900  # mlflow's log_batch cap is 1000

QUANTILE_LABEL_SUFFIX = '_quantile'
PRODUCTION_ALIAS = 'production'
DEFAULT_HORIZONS = tuple(range(1, 7))


def quantile_label(label: str) -> str:
    """The companion label a deployment's quantile models are saved under."""
    return f'{label}{QUANTILE_LABEL_SUFFIX}'


def deployment_labels(label: str) -> tuple:
    """Every label one deployment consists of - see the module docstring."""
    return (label, quantile_label(label))


class MixedGrainDeploymentError(ValueError):
    """A deployment saved at more than one grain; `per_label` maps each label to its grains."""

    def __init__(self, message: str, per_label: dict):
        super().__init__(message)
        self.per_label = per_label


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def _log_eval_history(client: MlflowClient, run_id: str, eval_history: dict) -> None:
    """Logs per-round eval_history as step-indexed mlflow metrics, sanitising metric names."""
    now_ms = int(time.time() * 1000)
    metrics = [
        Metric(key=f"{split}_{re.sub(r'[^\w.-]', '_', metric_name)}", value=float(value),
            timestamp=now_ms, step=step)
        for split, metric_dict in eval_history.items()
        for metric_name, values in metric_dict.items()
        for step, value in enumerate(values)
    ]
    for i in range(0, len(metrics), _METRICS_PER_BATCH):
        client.log_batch(run_id, metrics=metrics[i:i + _METRICS_PER_BATCH])


def _load_native(model_type: str, path: str):
    """Reconstructs each family's model object - same dispatch the old local-file registry used."""
    if model_type == 'xgboost':
        from xgboost import XGBRegressor
        model = XGBRegressor()
        model.load_model(path)
        return model
    if model_type == 'catboost':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(path)
        return model
    if model_type == 'lightgbm':
        import lightgbm as lgb
        return lgb.Booster(model_file=path)
    raise ValueError(f"Unknown model_type {model_type!r} - expected one of {list(_MODEL_EXT)}")


def _registered_name(plant: int, horizon, label: str) -> str:
    return f"{plant_name(plant)}_h{horizon}_{label}"


def _get_or_create_registered_model(client: MlflowClient, name: str) -> None:
    try:
        client.create_registered_model(name)
    except MlflowException as e:
        if e.error_code != 'RESOURCE_ALREADY_EXISTS':
            raise


def save_model(model, *, plant: int, horizon, feature_cols: list, params: dict,
        covid_weight: float, young_sku_weight: float, external_event_weight: float,
        label: str, details=None, trained_through, model_type: str = 'xgboost', cutoff_grain: str,
        experiment_name: str = None, eval_history: dict = None, training: dict = None,
        deployment_id: str = None) -> dict:
    """Registers `model` as a new version for (plant, horizon, label); does not move the alias."""
    ext = _MODEL_EXT.get(model_type)
    if ext is None:
        raise ValueError(f"Unknown model_type {model_type!r} - expected one of {list(_MODEL_EXT)}")
    if cutoff_grain not in CUTOFF_GRAINS:
        raise ValueError(f"cutoff_grain must be one of {list(CUTOFF_GRAINS)}, got {cutoff_grain!r}")

    client = _client()
    mlflow.set_experiment(experiment_name or f"{plant_name(plant)} - Production Models")
    registered_name = _registered_name(plant, horizon, label)

    with mlflow.start_run(run_name=f"h{horizon}_{label}") as run:
        mlflow.log_params({
            'plant': plant, 'horizon': horizon, 'label': label, 'model_type': model_type,
            'covid_weight': covid_weight, 'young_sku_weight': young_sku_weight,
            'external_event_weight': external_event_weight, 'trained_through': trained_through,
            'cutoff_grain': cutoff_grain, 'n_features': len(feature_cols), **params,
            **{f'training_{k}': v for k, v in (training or {}).items()},
        })
        if details:
            mlflow.set_tag('details', details)
        mlflow.log_dict({'feature_cols': list(feature_cols)}, 'feature_cols.json')
        if eval_history:
            _log_eval_history(client, run.info.run_id, eval_history)
        with tempfile.TemporaryDirectory() as tmp:
            model_path = os.path.join(tmp, f'model.{ext}')
            model.save_model(model_path)
            mlflow.log_artifact(model_path, artifact_path='model')
        artifact_uri = run.info.artifact_uri

    _get_or_create_registered_model(client, registered_name)
    tags = {'model_type': model_type, 'cutoff_grain': cutoff_grain,
            'trained_through': str(int(trained_through))}
    if deployment_id:
        tags['deployment_id'] = deployment_id
    mv = client.create_model_version(name=registered_name, source=f'{artifact_uri}/model',
        run_id=run.info.run_id, tags=tags)

    entry = {
        'plant': plant, 'horizon': horizon, 'label': label, 'model_type': model_type,
        'feature_cols': list(feature_cols), 'registered_name': registered_name,
        'version': mv.version, 'run_id': run.info.run_id, 'cutoff_grain': cutoff_grain,
        'trained_through': int(trained_through), 'deployment_id': deployment_id,
    }
    log.info("Logged %s v%s (run %s, %s grain)", registered_name, mv.version, run.info.run_id,
        cutoff_grain)
    return entry


def _check_one_grain(plant, models: list) -> None:
    """Refuses a batch spanning grains, or one leaving a deployment's halves at different grains."""
    grains = {m['cutoff_grain'] for m in models}
    if len(grains) > 1:
        raise MixedGrainDeploymentError(
            f"one save_models batch spans grains {sorted(grains)} - a batch is one training run "
            "at one grain.", {m['label']: [m['cutoff_grain']] for m in models})
    grain, labels = grains.pop(), {m['label'] for m in models}
    client = _client()
    for base in {lbl.removesuffix(QUANTILE_LABEL_SUFFIX) for lbl in labels}:
        for other in set(deployment_labels(base)) - labels:
            kept = {mv.tags.get('cutoff_grain') or 'month' for mv in
                    (_current_version(client, _registered_name(plant, h, other)) for h in DEFAULT_HORIZONS)
                    if mv is not None}
            if kept - {grain}:
                raise MixedGrainDeploymentError(
                    f"saving this batch at {grain!r} would leave deployment {base!r} for plant "
                    f"{plant} with {other!r} at {sorted(kept)}. Change a deployment's grain by "
                    "training both halves together (main.py --cutoff-grain, or train.py --params both).",
                    {other: sorted(kept), **{lbl: [grain] for lbl in labels}})


def save_models(*, plant, models: list, experiment_name: str = None) -> list:
    """Registers all `models` under one deployment_id, then moves PRODUCTION_ALIAS onto them.
    Refuses, before registering anything, a batch that would leave a deployment at mixed grains."""
    _check_one_grain(plant, models)
    deployment_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    entries = [save_model(plant=plant, deployment_id=deployment_id,
                          experiment_name=experiment_name, **spec) for spec in models]
    client = _client()
    previous = {e['registered_name']: _aliased_version(client, e['registered_name']) for e in entries}
    moved = []
    try:
        for e in entries:                               # <- the commit point
            client.set_registered_model_alias(e['registered_name'], PRODUCTION_ALIAS, e['version'])
            moved.append(e['registered_name'])
    except Exception:
        # No MLflow transaction: put back what moved, so the old deployment stays whole.
        log.error("Moving the %r alias failed after %d of %d model(s) - restoring them.",
            PRODUCTION_ALIAS, len(moved), len(entries))
        for name in moved:
            if previous[name] is None:
                client.delete_registered_model_alias(name, PRODUCTION_ALIAS)
            else:
                client.set_registered_model_alias(name, PRODUCTION_ALIAS, previous[name])
        raise
    log.info("Deployed %d model(s) for plant %s as %s (alias %r).", len(entries), plant,
        deployment_id, PRODUCTION_ALIAS)
    return entries


# What MLflow raises when the alias is unset (INVALID_PARAMETER_VALUE) or the registered model doesn't
# exist (RESOURCE_DOES_NOT_EXIST). Anything else - e.g. the server unreachable - must propagate, not
# read as "no alias": that would serve an old version or make a rollback delete the alias.
_NOT_FOUND_CODES = {'INVALID_PARAMETER_VALUE', 'RESOURCE_DOES_NOT_EXIST'}


def _aliased_version(client: MlflowClient, registered_name: str):
    """The version number PRODUCTION_ALIAS points at, or None."""
    try:
        return client.get_model_version_by_alias(registered_name, PRODUCTION_ALIAS).version
    except MlflowException as e:
        if e.error_code not in _NOT_FOUND_CODES:
            raise
        return None


def _current_version(client: MlflowClient, registered_name: str):
    """The production-aliased version, else the highest pre-alias version, else None. A version
    with a deployment_id is never used unaliased - it is from a deployment that died before its
    alias moved, so serving it would deploy part of one."""
    try:
        return client.get_model_version_by_alias(registered_name, PRODUCTION_ALIAS)
    except MlflowException as e:
        if e.error_code not in _NOT_FOUND_CODES:
            raise
    try:
        versions = client.search_model_versions(f"name='{registered_name}'",
            order_by=['version_number DESC'])
    except MlflowException as e:
        if e.error_code not in _NOT_FOUND_CODES:
            raise
        return None
    return next((v for v in versions if 'deployment_id' not in v.tags), None)


def _version_trained_through(client: MlflowClient, mv):
    """trained_through from the version tag, else the run's param (versions saved before tags)."""
    value = mv.tags.get('trained_through')
    if value is None:
        value = client.get_run(mv.run_id).data.params.get('trained_through')
    return int(float(value)) if value is not None else None


def entry_cutoff_grain(entry: dict) -> str:
    """The as-of grain a saved model was trained at - 'month' when absent (see module docstring)."""
    return entry.get('cutoff_grain') or 'month'


def check_cutoff_grain(entry: dict, frame_grain: str) -> None:
    """Raises when a model is about to score rows built at a different as-of grain than it was
    trained at. See the module docstring for why one direction of this is otherwise silent."""
    trained = entry_cutoff_grain(entry)
    if trained != frame_grain:
        raise ValueError(
            f"cutoff-grain mismatch: model label={entry.get('label')!r} h{entry.get('horizon')} was "
            f"trained at {trained!r} grain but is being scored on {frame_grain!r}-grain rows - a "
            f"month-trained model scored on day-grain rows under-forecasts silently. Retrain at "
            f"{frame_grain!r} grain, or score it with a {trained!r}-grain skeleton.")


def load_model(*, plant: int, horizon, label: str):
    """Loads the current (production-aliased, else highest) version for (plant, horizon, label),
    plus its feature_cols and an entry dict (model_type, cutoff_grain, trained_through, ...)."""
    client = _client()
    registered_name = _registered_name(plant, horizon, label)
    mv = _current_version(client, registered_name)
    if mv is None:
        raise FileNotFoundError(
            f"No registered model version for {registered_name!r} "
            f"(plant={plant!r}, horizon={horizon!r}, label={label!r}).")

    model_type = mv.tags.get('model_type', 'xgboost')
    ext = _MODEL_EXT.get(model_type)
    if ext is None:
        raise ValueError(f"Unknown model_type {model_type!r} - expected one of {list(_MODEL_EXT)}")
    model_path = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path=f'model/model.{ext}')
    model = _load_native(model_type, model_path)

    feature_cols_path = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path='feature_cols.json')
    with open(feature_cols_path, 'r', encoding='utf-8') as f:
        feature_cols = json.load(f)['feature_cols']

    entry = {'plant': plant, 'horizon': horizon, 'label': label, 'model_type': model_type,
        'registered_name': registered_name, 'version': mv.version, 'run_id': mv.run_id,
        'cutoff_grain': mv.tags.get('cutoff_grain') or 'month',
        'trained_through': _version_trained_through(client, mv),
        'deployment_id': mv.tags.get('deployment_id')}
    return model, feature_cols, entry


def deployed_state(*, plant, label: str, horizons=DEFAULT_HORIZONS, allow_mixed: bool = False) -> dict:
    """Deployed trained_through (min across models), cutoff_grain, horizons and labels, or None.
    Mixed grains raise unless `allow_mixed=True`."""
    client = _client()
    found = []
    for lbl in deployment_labels(label):
        for h in horizons:
            mv = _current_version(client, _registered_name(plant, h, lbl))
            if mv is not None:
                found.append((lbl, h, mv))
    if not found:
        return None
    per_label = {}
    for lbl, _h, mv in found:
        per_label.setdefault(lbl, set()).add(mv.tags.get('cutoff_grain') or 'month')
    per_label = {lbl: sorted(g) for lbl, g in per_label.items()}
    grains = {g for gs in per_label.values() for g in gs}
    if len(grains) > 1 and not allow_mixed:
        raise MixedGrainDeploymentError(
            f"deployment {label!r} for plant {plant} has models saved at mixed as-of grains "
            f"{per_label} - scoring would hand some of them rows at the wrong grain. Retrain it "
            "at one grain explicitly: main.py --cutoff-grain month|day.", per_label)
    deployments = {mv.tags.get('deployment_id') for _l, _h, mv in found}
    if len(deployments) > 1:
        log.warning("Deployment %r for plant %s mixes model versions from %d deployments %s - a "
            "retrain probably failed part-way through moving the %r alias.", label, plant,
            len(deployments), sorted(d or 'pre-alias' for d in deployments), PRODUCTION_ALIAS)
    through = [t for t in (_version_trained_through(client, mv) for _l, _h, mv in found) if t is not None]
    return {
        'trained_through': min(through) if through else None,
        'cutoff_grain': next(iter(grains)) if len(grains) == 1 else None,
        'horizons': sorted({h for _l, h, _mv in found}),
        'labels': {lbl: gs[0] if len(gs) == 1 else gs for lbl, gs in per_label.items()},
    }


def resolve_deployment_grain(state: dict, requested: str = None) -> str:
    """Returns `requested` if given, else the deployed grain; raises if neither exists."""
    if requested is not None:
        if requested not in CUTOFF_GRAINS:
            raise ValueError(
                f"Unknown cutoff_grain {requested!r} - expected one of {list(CUTOFF_GRAINS)}")
        return requested
    if state is None:
        raise ValueError(
            "Nothing is deployed under this label yet, so there is no grain to keep - choose one "
            "explicitly for the first deployment (main.py / train.py --cutoff-grain month|day).")
    if state['cutoff_grain'] is None:
        raise MixedGrainDeploymentError(
            f"deployment has mixed grains {state.get('labels')} - pass --cutoff-grain to "
            "retrain it at one.", state.get('labels', {}))
    return state['cutoff_grain']
