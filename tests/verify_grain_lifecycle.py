"""Grain lifecycle: a deployment keeps its as-of grain across every retrain.

The bug this pins: main._train_phase once called train_and_save without a grain and a 'month'
default filled the gap, so the first scheduled retrain of a DAY deployment saved month models over
the day ones and run_daily silently stopped rescoring daily. Also pinned: the daily flow - one run
per new source file, retrain on a new settled month, the first write of a month clearing the last.

Everything that decides, trains, saves, reads, checks or scores grain is REAL: run_daily,
_train_phase, train_deployment, train_and_save, save_models, deployed_state,
resolve_deployment_grain, predict, load_model, check_cutoff_grain and _prediction_phase, on the
throwaway MLflow store from tests/no_db.py. Stubbed: the Fabric-backed builders, the source-file
date, actual-order collation, the DB writer (a fake that remembers what was written) and the
OutstandingOrder phase. No database, no Azure login.
"""
import inspect
import pathlib
import re
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'tests'))
import no_db  # noqa: E402,F401 - every DB engine raises from here on (see no_db.py)

import main as orch  # noqa: E402
from adj_forecast_lib import model_registry as mr  # noqa: E402
from adj_forecast_lib import predict as pm  # noqa: E402
from adj_forecast_lib import train as tm  # noqa: E402
from adj_forecast_lib.config import CUTOFF_DATE_COL, month_idx_to_yyyymm, to_month_idx  # noqa: E402
from adj_forecast_lib.cutoff_grain import cutoff_dates_for_origins  # noqa: E402

FAILURES = []


def check(label, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' - ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(label)


def raises(fn, exc=Exception, needle=''):
    try:
        fn()
    except exc as e:
        return needle in str(e), f'{type(e).__name__}: {str(e)[:140]}'
    except Exception as e:
        return False, f'wrong exception {type(e).__name__}: {str(e)[:140]}'
    return False, 'no exception raised'


PLANT = 3300
M0 = int(to_month_idx(202607))
SETTLED = {'idx': M0}                      # the settled month the source files currently imply
DAY = {'n': 0}                             # bumped per run: every run sees a NEW source file


def AS_OF():
    """A day in the month after the settled one - the latest file's cutoff (file date - 1)."""
    return pd.Timestamp(cutoff_dates_for_origins([SETTLED['idx']], 10 + DAY['n'] % 15)[0])


FAST = {h: dict(model='xgboost', params=dict(max_depth=2, n_jobs=1, random_state=0,
            enable_categorical=True), young_sku_boost=1.0, external_event_weight=1.0)
        for h in range(1, 7)}
tm.BEST_PARAMS_BY_PLANT[PLANT] = tm.BEST_PARAMS_QUANTILE_BY_PLANT[PLANT] = FAST   # speed only

# ---------------------------------------------------------------------------------------------
# stubs: Fabric builders (real signatures), source file, DB, OutstandingOrder
# ---------------------------------------------------------------------------------------------
BUILT = {'ctx': [], 'train': [], 'infer': []}


def fake_ctx(grain):
    s = SETTLED['idx']
    return {'cutoff_grain': grain, 'max_settled_idx': s, 'effective_ranges': pd.DataFrame({
        'PartCode': ['SL', 'XX'], 'SalesModelCode': ['A1', 'B1'], 'PlanModelCode': ['PNA1', 'PNB1'],
        'PlanModelText': [None, None], 'StartMonthIdx': [s - 40.0] * 2, 'EndMonthIdx': [np.nan] * 2})}


def fake_build_order_book_context(plant, exclude_partcodes=None, max_settled_target=None,
                                  extend_to_orders=None, *, cutoff_grain):
    BUILT['ctx'].append(cutoff_grain)
    return fake_ctx(cutoff_grain)


def fake_preprocess_for_training(plant, exclude_partcodes=None, max_settled_target=None, ctx=None,
                                 *, cutoff_grain):
    BUILT['train'].append(cutoff_grain)
    s = SETTLED['idx']
    df = pd.DataFrame([(h, t) for h in range(1, 7) for t in range(s - 30, s + 1)],
                      columns=['Horizon', 'TargetIdx'])
    df['OriginIdx'] = df['TargetIdx'] - df['Horizon']
    df['a'] = np.random.default_rng(0).normal(size=len(df))
    df['Adjustment'] = 3 * df['a']
    df['OriginMonthsSinceLaunch'] = 24.0
    df['HasExternalEvent'] = 0
    if cutoff_grain == 'day':
        df[CUTOFF_DATE_COL] = AS_OF()
    return df, {}, ['a'], ctx or fake_ctx(cutoff_grain)


def fake_preprocess_for_inference(plant, origin_yyyymm=None, horizons=range(1, 7), codes=None,
        exclude_partcodes=None, min_origin_months_since_launch=3, ctx=None, *, cutoff_grain,
        cutoff_date=None):
    BUILT['infer'].append((cutoff_grain, cutoff_date))
    s = SETTLED['idx']
    sk = pd.DataFrame([(c, m, p, h) for c, m, p in (('PNA1', 'A1', 'SL'), ('PNB1', 'B1', 'XX'))
                       for h in horizons], columns=['PlanModelCode', 'SalesModelCode', 'PartCode', 'Horizon'])
    sk['OriginIdx'] = s
    sk['TargetIdx'] = s + sk['Horizon']
    sk['a'] = 0.5
    sk['KnownOrder'] = 100.0
    sk['ManualSim6'] = 5.0
    sk['KnownCanceledOrder'] = 0.0      # cancellation maths itself is verify_net_orders' job
    if cutoff_grain == 'day':
        sk[CUTOFF_DATE_COL] = pd.Timestamp(cutoff_date)
    return sk, ctx or fake_ctx(cutoff_grain)


class FakeDB:
    """Stands in for db_writer, remembering what a run wrote so the next run's checks see it."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.calls, self.files, self.origins, self.actual = [], set(), set(), set()

    def has_outstanding_for_file(self, plant, source_file):
        return source_file in self.files

    def has_prediction_at(self, plant, origin_date):
        return pd.Timestamp(origin_date) in self.origins

    def has_actual_for_month(self, plant, yyyymm):
        return int(yyyymm) in self.actual

    def month_start_write_clear(self, actual_df, result, plant, origin_date):
        self.calls.append(('month_start_write_clear', origin_date))
        self.actual.add(int(month_idx_to_yyyymm([SETTLED['idx']])[0]))
        self.files, self.origins = set(), {pd.Timestamp(origin_date)}

    def write_prediction_result(self, result, plant, origin_date):
        self.calls.append(('write_prediction_result', origin_date))
        self.origins.add(pd.Timestamp(origin_date))

    def __getattr__(self, name):
        def rec(*a, **k):
            self.calls.append((name, None))
            return 0
        return rec


DB = FakeDB()
LOG = []
tm.preprocess_for_training = fake_preprocess_for_training
pm.preprocess_for_inference = fake_preprocess_for_inference
orch.preprocess_for_inference = fake_preprocess_for_inference
orch.build_order_book_context = fake_build_order_book_context
# the file is dated the day AFTER the cutoff: a file dated D is complete through D-1 only
orch.latest_source_file_date = lambda plant: (AS_OF() + pd.Timedelta(days=1),
    (AS_OF() + pd.Timedelta(days=1)).strftime('%Y%m%d') + '.xlsx')
orch.collate_actual_order = lambda ctx, plant, month_idx=None: pd.DataFrame()
orch.db_writer = DB


def fake_outstanding(plant, file_date, source_file, settled_month_idx):
    LOG.append('outstanding')
    DB.files.add(source_file)


orch._outstanding_order_phase = fake_outstanding
WRITES = ('month_start_write_clear', 'write_prediction_result')


def run(label, new_file=True, **kw):
    """One real run_daily, on a new source file unless `new_file=False`. Returns what it trained,
    how it scored, which DB calls it made, and whether it raised."""
    if new_file:
        DAY['n'] += 1
    for v in BUILT.values():
        v.clear()
    DB.calls.clear()
    LOG.clear()
    try:
        orch.run_daily(PLANT, label, **kw)
        err = None
    except Exception as e:
        err = e
    write = next(((n, d) for n, d in reversed(DB.calls) if n in WRITES), (None, None))
    return {'trained': list(BUILT['train']), 'scored': [g for g, _d in BUILT['infer']],
            'db': [n for n, _d in DB.calls], 'write': write[0], 'origin_date': write[1],
            'as_of': AS_OF(), 'err': err, 'outstanding': list(LOG)}


def state(label, **kw):
    return mr.deployed_state(plant=PLANT, label=label, **kw)


CLIENT = mr._client()


def live_versions(label):
    return {(lbl, h): CLIENT.get_model_version_by_alias(mr._registered_name(PLANT, h, lbl),
                                                        mr.PRODUCTION_ALIAS)
            for lbl in mr.deployment_labels(label) for h in range(1, 7)}


def entry_grains(label):
    return {k: v.tags.get('cutoff_grain') for k, v in live_versions(label).items()}


def lifecycle(label, grain):
    """Bootstrap at `grain`, a same-month day, two month-closes with the SCHEDULED (flag-less)
    run, and the day after. Returns the per-step records."""
    DB.reset()
    SETTLED['idx'] = M0
    steps = {'bootstrap': run(label, cutoff_grain=grain)}
    steps['bootstrap']['state'] = state(label)
    steps['same_month'] = run(label)
    steps['same_month']['state'] = state(label)
    for i, name in ((1, 'close_1'), (2, 'close_2')):
        SETTLED['idx'] = M0 + i
        steps[name] = run(label)
        steps[name]['state'] = state(label)
        steps[name]['entries'] = entry_grains(label)
    steps['after_close_2_next_day'] = run(label)
    steps['same_file_again'] = run(label, new_file=False)
    steps['same_file_forced'] = run(label, new_file=False, force_run=True)
    SETTLED['idx'] = M0
    return steps


print('\n=== 1. resolve_deployment_grain: the one decision ===')
day_state = {'cutoff_grain': 'day', 'labels': {'production': 'day'}}
check('explicit request wins', mr.resolve_deployment_grain(day_state, 'month') == 'month')
check('no request keeps the deployed grain', mr.resolve_deployment_grain(day_state) == 'day')
ok, d = raises(lambda: mr.resolve_deployment_grain(None), ValueError, 'no grain to keep')
check('nothing deployed and nothing requested raises - no month fallback', ok, d)
ok, d = raises(lambda: mr.resolve_deployment_grain(None, 'daily'), ValueError, 'Unknown cutoff_grain')
check('an unknown grain raises', ok, d)
ok, d = raises(lambda: mr.resolve_deployment_grain({'cutoff_grain': None, 'labels': {}}),
               mr.MixedGrainDeploymentError)
check('a mixed deployment cannot be resolved without an explicit grain', ok, d)

print('\n=== 2. a DAY deployment stays day across monthly retrains ===')
day = lifecycle('life_day', 'day')
for step in day.values():
    if step['err']:
        print(f"         raised: {type(step['err']).__name__}: {step['err']}")
check('bootstrap with --cutoff-grain day: ONE panel build at day, both halves deployed at day',
      day['bootstrap']['trained'] == ['day']
      and day['bootstrap']['state']['labels'] == {'life_day': 'day', 'life_day_quantile': 'day'},
      str(day['bootstrap']))
check('bootstrap is the first write of the month: actual + clear + insert in one call',
      day['bootstrap']['write'] == 'month_start_write_clear')
check('same month, next file: rescored at day, no retrain, a plain daily insert',
      day['same_month']['trained'] == [] and day['same_month']['scored'] == ['day']
      and day['same_month']['write'] == 'write_prediction_result', str(day['same_month']))
for name in ('close_1', 'close_2'):
    s = day[name]
    check(f'{name}: scheduled run (no flag) rebuilds the panel ONCE, at DAY',
          s['trained'] == ['day'], str(s['trained']))
    settled_now = M0 + int(name[-1])
    check(f'{name}: every one of the 12 aliased versions is day, trained_through advanced',
          set(s['entries'].values()) == {'day'} and len(s['entries']) == 12
          and s['state']['trained_through'] == settled_now, f"{s['entries']} {s['state']}")
    check(f'{name}: model AND manual methods scored off ONE day skeleton',
          s['scored'] == ['day'], str(s['scored']))
    check(f'{name}: written as the month start (last month cleared), stamped with the day cutoff',
          s['write'] == 'month_start_write_clear' and s['origin_date'] == s['as_of'],
          f"{s['write']} {s['origin_date']} {s['as_of']}")
check('day after the 2nd close: still rescoring daily, no retrain - daily refresh survives',
      day['after_close_2_next_day']['trained'] == []
      and day['after_close_2_next_day']['scored'] == ['day'])
check('deployment metadata: both labels report day',
      day['close_2']['state']['labels'] == {'life_day': 'day', 'life_day_quantile': 'day'})
check('the same source file again: nothing trained, scored or written',
      not day['same_file_again']['trained'] and not day['same_file_again']['scored']
      and not day['same_file_again']['outstanding'] and day['same_file_again']['write'] is None,
      str(day['same_file_again']))
check('...unless forced (DAILY_FORCE_RUN): rescored and OutstandingOrder rewritten, no retrain',
      day['same_file_forced']['trained'] == [] and day['same_file_forced']['scored'] == ['day']
      and day['same_file_forced']['outstanding'] == ['outstanding'], str(day['same_file_forced']))

print('\n=== 3. a MONTH deployment stays month, with its old cadence ===')
mon = lifecycle('life_month', 'month')
check('bootstrap with --cutoff-grain month: one panel build at month',
      mon['bootstrap']['trained'] == ['month'])
check('same month, next file: nothing retrained OR rescored (identical numbers)',
      mon['same_month']['trained'] == [] and mon['same_month']['scored'] == [])
for name in ('close_1', 'close_2'):
    s = mon[name]
    check(f'{name}: retrains at month, all 12 versions month, scores at month, month-start write '
          'stamped at month end', s['trained'] == ['month'] and set(s['entries'].values()) == {'month'}
          and s['scored'] == ['month'] and s['write'] == 'month_start_write_clear'
          and pd.Timestamp(s['origin_date']).is_month_end, f"{s['trained']} {s['scored']} {s['write']}")
check('OutstandingOrder refreshed on every new file, month close or not, with no separate clear',
      all(mon[n]['outstanding'] == ['outstanding'] for n in ('same_month', 'close_1', 'close_2'))
      and not any(c.startswith('clear_') for n in ('same_month', 'close_1', 'close_2')
                  for c in mon[n]['db']))
check('forcing a month-grain rerun rescores even though the month-end forecast exists',
      mon['same_file_forced']['scored'] == ['month'], str(mon['same_file_forced']))

print('\n=== 4. predict() with no grain scores at the DEPLOYED grain ===')
BUILT['infer'].clear()
SETTLED['idx'] = M0 + 2
res = pm.predict(PLANT, label='life_day', cutoff_date=AS_OF())
check('day deployment -> day skeleton requested, scored by both labels',
      BUILT['infer'] == [('day', AS_OF())]
      and sorted(res['Method'].unique()) == ['model_life_day', 'model_life_day_quantile'],
      f"{BUILT['infer']} {sorted(res['Method'].unique())}")
BUILT['infer'].clear()
pm.predict(PLANT, label='life_month')
check('month deployment -> month skeleton requested', BUILT['infer'] == [('month', None)])
ok, d = raises(lambda: pm.predict(PLANT, label='life_day', cutoff_grain='month'), ValueError,
               'cutoff-grain mismatch')
check('asserting the wrong grain still fails loudly at check_cutoff_grain', ok, d)
SETTLED['idx'] = M0

print('\n=== 5. mixed-grain deployments are rejected - and repairable only explicitly ===')
DB.reset()
run('mixed', cutoff_grain='day')
for scenario, keys in {
    'within one label (h1 month, h2-6 day)': [('mixed', 1)],
    'across the label and its quantile companion': [('mixed_quantile', h) for h in range(1, 7)],
}.items():
    live = live_versions('mixed')
    for lbl, h in keys:
        CLIENT.set_model_version_tag(live[(lbl, h)].name, live[(lbl, h)].version, 'cutoff_grain', 'month')
    ok, d = raises(lambda: state('mixed'), mr.MixedGrainDeploymentError, 'mixed as-of grains')
    check(f'{scenario}: deployed_state refuses it', ok, d)
    r = run('mixed')
    check(f'{scenario}: scheduled run refuses too - nothing trained, scored or refreshed',
          isinstance(r['err'], mr.MixedGrainDeploymentError) and not r['trained'] and not r['scored']
          and not r['outstanding'], repr(r['err']))
    r = run('mixed', cutoff_grain='day')
    check(f'{scenario}: --cutoff-grain day repairs it: both halves retrained at day, then scored',
          r['err'] is None and r['trained'] == ['day'] and r['scored'] == ['day']
          and state('mixed')['labels'] == {'mixed': 'day', 'mixed_quantile': 'day'},
          f"{r} {state('mixed', allow_mixed=True)}")
    check(f'{scenario}: the repair run refreshes OutstandingOrder, with no separate clear',
          r['outstanding'] == ['outstanding'] and not any(c.startswith('clear_') for c in r['db']))

print('\n=== 6. switching grain on purpose ===')
DB.reset()
run('switch', cutoff_grain='month')
r = run('switch', cutoff_grain='day')
check('month -> day mid-month retrains immediately at day and scores at day',
      r['trained'] == ['day'] and r['scored'] == ['day'] and state('switch')['cutoff_grain'] == 'day')
r = run('switch')
check('...and the following scheduled run keeps day', r['trained'] == [] and r['scored'] == ['day'])
r = run('switch', cutoff_grain='day')
check('re-passing the SAME grain is a no-op, not a retrain', r['trained'] == [])

print('\n=== 7. first deployment has no default ===')
DB.reset()
r = run('empty')
check('nothing deployed + no --cutoff-grain: raises, trains nothing, no month assumed',
      isinstance(r['err'], ValueError) and 'no grain to keep' in str(r['err']) and not r['trained'],
      repr(r['err']))

print('\n=== 8. train_and_save guards ===')
sig = inspect.signature(tm.train_and_save).parameters['cutoff_grain']
check('cutoff_grain is keyword-only and REQUIRED',
      sig.kind is inspect.Parameter.KEYWORD_ONLY and sig.default is inspect.Parameter.empty)
ok, d = raises(lambda: tm.train_and_save(PLANT, 'x', best_params=FAST), TypeError, 'cutoff_grain')
check('omitting it is a TypeError, not a silent month', ok, d)
ok, d = raises(lambda: tm.train_and_save(PLANT, 'x', best_params=FAST, cutoff_grain='daily'),
               ValueError, 'Unknown cutoff_grain')
check('an unknown grain is refused', ok, d)

DB.reset()
run('guard', cutoff_grain='day')
before = entry_grains('guard')
before_versions = {k: v.version for k, v in live_versions('guard').items()}
tm.preprocess_for_training = lambda plant, exclude_partcodes=None, max_settled_target=None, \
    ctx=None, *, cutoff_grain: fake_preprocess_for_training(plant, cutoff_grain='month')  # drops it
ok, d = raises(lambda: tm.train_and_save(PLANT, 'guard', best_params=FAST, cutoff_grain='day'),
               ValueError, "requested a 'day'-grain training panel but got a 'month'")
tm.preprocess_for_training = fake_preprocess_for_training
check('a panel builder that ignores the grain is caught before anything is saved', ok, d)
check('...and the day deployment is untouched',
      {k: v.version for k, v in live_versions('guard').items()} == before_versions)

real_adapter = tm.ADAPTERS['xgboost']


def dies_at_h4(train, *a, **k):
    if int(train['Horizon'].iloc[0]) == 4:
        raise MemoryError('simulated OOM at h4')
    return real_adapter(train, *a, **k)


SETTLED['idx'] = M0 + 1
tm.ADAPTERS['xgboost'] = dies_at_h4
r = run('guard')
tm.ADAPTERS['xgboost'] = real_adapter
st = state('guard')
check('a retrain that dies mid-horizons leaves the old deployment whole (all day, old month)',
      isinstance(r['err'], orch.DailyRunError) and isinstance(r['err'].__cause__, MemoryError)
      and entry_grains('guard') == before and st['trained_through'] == M0, f'{r["err"]!r} {st}')
check('...and the file is still scored by that old deployment, at day, and OutstandingOrder '
      'refreshed', r['scored'] == ['day'] and r['origin_date'] == r['as_of']
      and r['outstanding'] == ['outstanding'], f"{r['scored']} {r['origin_date']} {r['outstanding']}")
r = run('guard')
check('...so the next file is still stale, retries at day, and succeeds',
      r['err'] is None and r['trained'] == ['day'] and state('guard')['trained_through'] == M0 + 1)
SETTLED['idx'] = M0

print('\n=== 9. a future regression that drops the grain is stopped by run_daily ===')
DB.reset()
run('regression', cutoff_grain='day')
real_td = orch.train_deployment
orch.train_deployment = lambda *a, cutoff_grain, **k: real_td(*a, cutoff_grain='month', **k)
SETTLED['idx'] = M0 + 1
r = run('regression')
orch.train_deployment = real_td
SETTLED['idx'] = M0
check('retrain that saves the wrong grain -> refused, and nothing is scored from it',
      isinstance(r['err'], RuntimeError) and 'refusing to score' in str(r['err']) and not r['scored'],
      repr(r['err']))
check('...and OutstandingOrder is not refreshed either, so it stays aligned with the last forecast',
      r['outstanding'] == [])

print('\n=== 10. train.py CLI resolves grain the same way ===')
seen = []
real_train_and_save = tm.train_and_save
tm.train_and_save = lambda **k: seen.append((k['label'], k['cutoff_grain']))


def cli(*argv):
    seen.clear()
    old = sys.argv
    sys.argv = ['train.py', '--plant', str(PLANT), *argv]
    try:
        tm.main()
        return seen[-1]
    except Exception as e:
        return e
    finally:
        sys.argv = old


try:
    check('no flag on a day deployment -> day', cli('--label', 'life_day') == ('life_day', 'day'))
    check('--params quantile on a day deployment -> the companion, at day',
          cli('--label', 'life_day', '--params', 'quantile') == ('life_day_quantile', 'day'))
    check('no flag on a month deployment -> month',
          cli('--label', 'life_month') == ('life_month', 'month'))
    out = cli('--label', 'cli_empty')
    check('no flag, nothing deployed -> refuses', isinstance(out, ValueError), repr(out))
    check('explicit flag wins',
          cli('--label', 'life_day', '--cutoff-grain', 'month') == ('life_day', 'month'))
finally:
    tm.train_and_save = real_train_and_save

print('\n=== 11. no month default can creep back into the deployment path ===')
for fn in (tm.train_and_save, orch._train_phase, orch._prediction_phase, orch.run_daily, pm.predict):
    p = inspect.signature(fn).parameters['cutoff_grain']
    check(f'{fn.__module__}.{fn.__name__}: cutoff_grain has no month default',
          p.default in (inspect.Parameter.empty, None), f'default={p.default!r}')
srcs = {p.name: p.read_text(encoding='utf-8') for p in
        (_REPO_ROOT / 'main.py', _REPO_ROOT / 'adj_forecast_lib' / 'train.py',
         _REPO_ROOT / 'adj_forecast_lib' / 'predict.py')}
built = [n for n, s in srcs.items() if re.search(r"""f['"][^'"\n]*\}_quantile""", s)]
check('the quantile companion label is built only by model_registry.quantile_label', not built, str(built))
grain_else = [n for n, s in srcs.items() if re.search(r"""else\s+['"]month['"]""", s)]
check("no `... else 'month'` fallback in main/train/predict", not grain_else, str(grain_else))

print('\n' + '=' * 78)
print(f'{len(FAILURES)} FAILURE(S): {FAILURES}' if FAILURES else 'ALL CHECKS PASSED')
sys.exit(1 if FAILURES else 0)
