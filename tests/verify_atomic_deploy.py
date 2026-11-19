"""A deployment is trained from one panel and changes as one unit - or not at all.

Pins what used to be possible:
  * the training panel built more than once per retrain, and scoring loading Fabric again;
  * the two halves seeing different data (a file landing between two builds);
  * a failure between the halves leaving them at different grains, so every later run refused to
    score until someone intervened;
  * a half-written deployment after a crash mid-save.

On MLflow a deployment is the 12 versions the `production` alias points at (a label and its
quantile companion, h1-h6). "Atomic" means: versions are registered first, the alias moves last,
a failed alias move is rolled back, and a version from a deployment that never got its alias is
never served. Real code throughout (run_daily, train_deployment, save_models, deployed_state,
load_model, the panel builders) over the synthetic feed and the throwaway MLflow store from
tests/no_db.py. Runs for every plant in config.PLANT_NAMES, plus an unconfigured plant code.
"""
import logging
import pathlib
import sys
import warnings

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'tests'))
import no_db  # noqa: E402,F401 - every DB engine raises from here on (see no_db.py)

import pandas as pd  # noqa: E402

import main as orch  # noqa: E402
from adj_forecast_lib import db_writer, model_registry as mr, preprocessing as pp  # noqa: E402
from adj_forecast_lib import train as tm  # noqa: E402
from adj_forecast_lib.config import CUTOFF_DATE_COL, PLANT_NAMES, plant_name  # noqa: E402
from adj_forecast_lib.cutoff_grain import cutoff_day_index  # noqa: E402
from synthetic_fabric import FabricStub, RecordingEngine, make_orders  # noqa: E402

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)
FAILURES = []


def check(label, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' - ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(label)


def raises(fn, exc=Exception, needle=''):
    try:
        fn()
    except exc as e:
        return needle in str(e), f'{type(e).__name__}: {str(e)[:160]}'
    except Exception as e:
        return False, f'wrong exception {type(e).__name__}: {str(e)[:160]}'
    return False, 'no exception raised'


PLANTS = sorted(PLANT_NAMES)
ORDERS = make_orders()
FAST = {h: dict(model='xgboost', params=dict(max_depth=2, n_jobs=1, random_state=0,
            enable_categorical=True), young_sku_boost=1.0, external_event_weight=1.0)
        for h in range(1, 7)}
CLIENT = mr._client()


def failing(message, exc=MemoryError):
    """A per-horizon spec whose adapter raises - a fit that dies (OOM, bad data)."""
    tm.ADAPTERS['__fails__'] = lambda *a, **k: (_ for _ in ()).throw(exc(message))
    return {h: dict(s, model='__fails__') for h, s in FAST.items()}


# Scheduled runs train with each plant's own spec - the same dicts train.py reads, made fast here.
for p in PLANTS:
    tm.BEST_PARAMS_BY_PLANT[p] = tm.BEST_PARAMS_QUANTILE_BY_PLANT[p] = FAST
OUTSTANDING = []
orch._outstanding_order_phase = lambda plant, *a, **k: OUTSTANDING.append(plant)
BUILDS = []
_real_build = pp.build_full_panel
pp.build_full_panel = lambda ctx: (BUILDS.append(ctx['cutoff_grain']) or _real_build(ctx))


def run(plant, label, fab=None, **kw):
    """One real run_daily; returns (error, recorded DB statements)."""
    eng, saved = RecordingEngine(), db_writer._engine
    db_writer._engine = eng
    try:
        with (fab or FabricStub(ORDERS)):
            orch.run_daily(plant, label, **kw)
        return None, eng
    except Exception as e:
        return e, eng
    finally:
        db_writer._engine = saved


def aliases(plant, label):
    """{(label, horizon): aliased version} for the whole deployment - what the alias says is live."""
    return {(lbl, h): mr._aliased_version(CLIENT, mr._registered_name(plant, h, lbl))
            for lbl in mr.deployment_labels(label) for h in range(1, 7)}


def n_versions(plant, label):
    return sum(len(CLIENT.search_model_versions(f"name='{mr._registered_name(plant, h, lbl)}'"))
               for lbl in mr.deployment_labels(label) for h in range(1, 7))


def state(plant, label, **kw):
    return mr.deployed_state(plant=plant, label=label, **kw)


def deploy(plant, label, grain='day', **kw):
    with FabricStub(ORDERS):
        return tm.train_deployment(plant, label, cutoff_grain=grain,
                                   **{'best_params': FAST, 'best_params_quantile': FAST, **kw})


for plant in PLANTS:
    tag = f'{plant} ({PLANT_NAMES[plant]})'
    print(f'\n=================== {tag} ===================')

    print('=== 1. one panel build and one Fabric load per run ===')
    BUILDS.clear()
    fab = FabricStub(ORDERS)
    err, _ = run(plant, 'loads', fab, cutoff_grain='day')
    check(f'{tag} retrain + score: ONE panel build, ONE Fabric load (prediction reuses the ctx)',
          err is None and BUILDS == ['day'] and fab.calls['load_io'] == 1,
          f'{err!r} builds={BUILDS} load_io={fab.calls["load_io"]}')
    BUILDS.clear()
    fab = FabricStub(ORDERS)
    err, eng = run(plant, 'loads', fab)
    check(f'{tag} next day (rescore only): no build, ONE Fabric load for model + manual together',
          err is None and BUILDS == [] and fab.calls['load_io'] == 1
          and len(eng.rows_for('PredictionResult')) > 0,
          f'{err!r} builds={BUILDS} load_io={fab.calls["load_io"]}')

    print('=== 2. one panel can safely feed both halves ===')
    with FabricStub(ORDERS):
        panel, feature_cols, settled = tm.build_training_panel(plant, cutoff_grain='day')
    fingerprint = lambda p: (int(pd.util.hash_pandas_object(p, index=True).sum()),
                             tuple(p.columns), tuple(map(str, p.dtypes)))
    before = fingerprint(panel)
    mixed_families = dict(FAST)
    mixed_families[1] = dict(model='catboost', params=dict(depth=2, learning_rate=0.3,
        random_seed=0, allow_writing_files=False, loss_function='MAE'),
        young_sku_boost=1.0, external_event_weight=1.0)
    tm.fit_horizon_models(panel, feature_cols, settled, mixed_families)
    tm.fit_horizon_models(panel, feature_cols, settled, FAST)
    check(f'{tag} fitting both halves (xgboost + catboost) leaves the shared panel bit-identical',
          fingerprint(panel) == before)

    print('=== 3. a file landing mid-retrain cannot split the halves ===')
    moving = FabricStub(ORDERS, file_date=pd.Timestamp('2026-09-30'))
    real_load = moving._load_io

    def load_then_move(p, exclude_partcodes=None):
        out = real_load(p, exclude_partcodes)
        moving.file_date = pd.Timestamp('2026-10-01')      # the Oct 1 file lands right after
        return out
    moving._load_io = load_then_move
    with moving:
        entries = tm.train_deployment(plant, 'race', cutoff_grain='month', best_params=FAST,
                                      best_params_quantile=FAST)
    through = {(e['label'], e['trained_through']) for e in entries}
    check(f'{tag} both halves trained from the one load: same trained_through, one Fabric load',
          len({t for _l, t in through}) == 1 and moving.calls['load_io'] == 1, str(sorted(through)))

    print('=== 4. a deployment is committed atomically ===')
    deploy(plant, 'atomic')
    base, base_n = aliases(plant, 'atomic'), n_versions(plant, 'atomic')
    ok, d = raises(lambda: deploy(plant, 'atomic', best_params_quantile=failing('quantile fit OOM')),
                   MemoryError)
    check(f'{tag} quantile fit fails -> every alias unchanged, nothing registered',
          ok and aliases(plant, 'atomic') == base and n_versions(plant, 'atomic') == base_n, d)

    from xgboost import XGBRegressor
    real_save = XGBRegressor.save_model

    def dies_on_8th(count={'n': 0}):
        def save(self, path):
            count['n'] += 1
            if count['n'] == 8:
                raise OSError('disk full writing model 8 of 12')
            return real_save(self, path)
        return save
    for label in ('atomic', 'fresh'):
        XGBRegressor.save_model = dies_on_8th({'n': 0})
        try:
            ok, d = raises(lambda: deploy(plant, label), OSError, 'disk full')
        finally:
            XGBRegressor.save_model = real_save
        if label == 'atomic':
            check(f'{tag} a save fails on model 8 of 12 -> every alias unchanged',
                  ok and aliases(plant, 'atomic') == base, d)
            loaded = [mr.load_model(plant=plant, horizon=h, label=lbl)
                      for lbl in mr.deployment_labels('atomic') for h in range(1, 7)]
            check(f'{tag} ...the old deployment still loads, all 12 at their aliased version',
                  [e['version'] for _m, _f, e in loaded] == list(base.values()))
        else:
            check(f'{tag} the FIRST deployment of a label dies on model 8 -> the 7 versions it '
                  'registered are never served: nothing is deployed', ok and state(plant, 'fresh') is None,
                  f'{d}; state={state(plant, "fresh")}')

    real_alias = mr.MlflowClient.set_registered_model_alias
    calls = {'n': 0}

    def alias_dies_on_5th(self, name, alias, version):
        calls['n'] += 1
        if calls['n'] == 5:
            raise OSError('lost the connection moving alias 5 of 12')
        return real_alias(self, name, alias, version)
    mr.MlflowClient.set_registered_model_alias = alias_dies_on_5th
    try:
        ok, d = raises(lambda: deploy(plant, 'atomic'), OSError, 'alias 5 of 12')
    finally:
        mr.MlflowClient.set_registered_model_alias = real_alias
    check(f'{tag} the alias move itself fails part-way -> the 4 moved aliases are restored',
          ok and aliases(plant, 'atomic') == base, d)

    new = deploy(plant, 'atomic')
    live = [CLIENT.get_model_version_by_alias(mr._registered_name(plant, h, lbl), mr.PRODUCTION_ALIAS)
            for lbl in mr.deployment_labels('atomic') for h in range(1, 7)]
    check(f'{tag} a successful commit: all 12 aliased versions are this run\'s, one deployment_id',
          [v.version for v in live] == [e['version'] for e in new]
          and len({v.tags.get('deployment_id') for v in live}) == 1)
    # Audit B10: the deployed model is refitted on train + val after early stopping chose its rounds.
    params = [CLIENT.get_run(v.run_id).data.params for v in live]
    refit_ok = lambda p: (int(p['training_n_refit']) == int(p['training_n_train']) + int(p['training_n_val'])
                          and int(p['training_n_val']) > 0 and int(p['training_n_rounds']) >= 1)
    check(f'{tag} every version records its refit: on all n_train + n_val settled rows, with the '
          'round count early stopping chose', all(refit_ok(p) for p in params),
          str([{k: v for k, v in p.items() if k.startswith('training_')} for p in params[:2]]))

    print("=== 5. the halves can't be left at different grains ===")
    model, fcols, entry = mr.load_model(plant=plant, horizon=1, label='atomic')
    one = [dict(model=model, horizon=1, feature_cols=fcols, params={}, covid_weight=.1,
                young_sku_weight=1., external_event_weight=1., label='atomic',
                trained_through=entry['trained_through'], model_type='xgboost', cutoff_grain='month')]
    before, before_n = aliases(plant, 'atomic'), n_versions(plant, 'atomic')
    ok, d = raises(lambda: mr.save_models(plant=plant, models=one), mr.MixedGrainDeploymentError,
                   'would leave deployment')
    check(f'{tag} saving ONE half at another grain is refused, nothing registered',
          ok and aliases(plant, 'atomic') == before and n_versions(plant, 'atomic') == before_n, d)
    two = [dict(one[0], horizon=2, cutoff_grain='day')] + one
    ok, d = raises(lambda: mr.save_models(plant=plant, models=two), mr.MixedGrainDeploymentError,
                   'spans grains')
    check(f'{tag} one batch spanning two grains is refused', ok, d)

    def cli(*argv):
        old = sys.argv
        sys.argv = ['train.py', '--plant', str(plant), '--label', 'atomic', *argv]
        try:
            with FabricStub(ORDERS):
                tm.main()
            return None
        except Exception as e:
            return e
        finally:
            sys.argv = old
    err = cli('--params', 'standard', '--cutoff-grain', 'month')
    check(f'{tag} train.py --params standard at a new grain is refused',
          isinstance(err, mr.MixedGrainDeploymentError) and aliases(plant, 'atomic') == before, repr(err))
    err = cli('--params', 'both', '--cutoff-grain', 'month')
    check(f'{tag} train.py --params both switches the whole deployment',
          err is None and state(plant, 'atomic')['labels'] == {'atomic': 'month',
                                                               'atomic_quantile': 'month'}, repr(err))
    err = cli('--params', 'quantile')
    check(f'{tag} one half retrained at the SAME grain is still allowed',
          err is None and state(plant, 'atomic')['cutoff_grain'] == 'month', repr(err))

    print('=== 6. a failed retrain or switch never leaves the system stuck ===')
    run(plant, 'recover', cutoff_grain='month')
    base = aliases(plant, 'recover')
    tm.BEST_PARAMS_QUANTILE_BY_PLANT[plant] = failing('quantile fit OOM')
    OUTSTANDING.clear()
    err, _ = run(plant, 'recover', cutoff_grain='day')
    tm.BEST_PARAMS_QUANTILE_BY_PLANT[plant] = FAST
    check(f'{tag} switch month -> day dies in the quantile fit: deployment untouched, still month, '
          'OutstandingOrder still refreshed (the incumbent scored)',
          isinstance(err, orch.DailyRunError) and isinstance(err.__cause__, MemoryError)
          and aliases(plant, 'recover') == base and OUTSTANDING == [plant]
          and state(plant, 'recover')['labels'] == {'recover': 'month', 'recover_quantile': 'month'},
          repr(err))
    OUTSTANDING.clear()
    err, _ = run(plant, 'recover')
    check(f'{tag} ...the next scheduled run is NOT stuck: no error, OutstandingOrder refreshed, '
          'month kept', err is None and OUTSTANDING == [plant]
          and state(plant, 'recover')['cutoff_grain'] == 'month', repr(err))
    err, eng = run(plant, 'recover', cutoff_grain='day')
    check(f'{tag} ...rerunning the switch completes it: both halves day, scored at day',
          err is None and state(plant, 'recover')['labels'] == {'recover': 'day', 'recover_quantile': 'day'}
          and (cutoff_day_index(pd.DataFrame({CUTOFF_DATE_COL: pd.to_datetime(
              eng.rows_for('PredictionResult')['OriginDate'])})) > 0).all(), repr(err))
    through = state(plant, 'recover')['trained_through']
    tm.BEST_PARAMS_BY_PLANT[plant] = failing('standard fit OOM')
    OUTSTANDING.clear()
    err, eng = run(plant, 'recover', FabricStub(ORDERS, file_date=pd.Timestamp('2026-10-05')))
    tm.BEST_PARAMS_BY_PLANT[plant] = FAST
    check(f'{tag} month-close retrain dies: old day deployment intact (trained_through unchanged)',
          isinstance(err, orch.DailyRunError) and isinstance(err.__cause__, MemoryError)
          and state(plant, 'recover')['trained_through'] == through
          and state(plant, 'recover')['cutoff_grain'] == 'day', repr(err))
    scored = eng.rows_for('PredictionResult')
    check(f'{tag} ...and that old deployment still scores the new month as of Oct 4, and '
          'OutstandingOrder refreshes',
          len(scored) > 0 and (pd.to_datetime(scored['OriginDate']) == pd.Timestamp('2026-10-04')).all()
          and OUTSTANDING == [plant], f'{len(scored)} rows, outstanding={OUTSTANDING}')
    err, _ = run(plant, 'recover', FabricStub(ORDERS, file_date=pd.Timestamp('2026-10-06')))
    st = state(plant, 'recover')
    check(f'{tag} ...the next file is still stale, retries automatically at day, and succeeds',
          err is None and st['trained_through'] == through + 1 and st['cutoff_grain'] == 'day',
          f'{err!r} {st}')

print('\n=== 7. the registry layer is plant-generic ===')
new_plant = 4400                                   # not in config.PLANT_NAMES
check("an unconfigured plant gets its own name, never another plant's",
      plant_name(new_plant) == '4400' and mr._registered_name(new_plant, 1, 'x').startswith('4400_'))
spec = dict(model=model, horizon=1, feature_cols=fcols, params={}, covid_weight=.1,
            young_sku_weight=1., external_event_weight=1., label='production', trained_through=1,
            model_type='xgboost', cutoff_grain='day')
mr.save_models(plant=new_plant, models=[spec])
check('save -> deployed_state -> load_model round-trips for it',
      state(new_plant, 'production')['cutoff_grain'] == 'day'
      and mr.load_model(plant=new_plant, horizon=1, label='production')[2]['plant'] == new_plant)
lib = {p.name: p.read_text(encoding='utf-8') for p in (_REPO_ROOT / 'adj_forecast_lib').glob('*.py')}
hard = [n for n, s in lib.items() if "'STJ' if" in s or "'SPL' if" in s]
check("no `'STJ' if plant == 3300 else 'SPL'` naming left in the library", not hard, str(hard))

print('\n' + '=' * 78)
print(f'{len(FAILURES)} FAILURE(S): {FAILURES}' if FAILURES else 'ALL CHECKS PASSED')
sys.exit(1 if FAILURES else 0)
