"""Deployment refit (audit B10): the deployed model learns the latest settled months. Plus the
cost of the early-stopping fit itself (audit B17), at the end.

fit_horizon_models used to deploy the model early stopping produced. That model never saw the last
`val_months` settled targets - they were only the early-stopping signal - so every monthly retrain
threw away the freshest half-year, the months closest to what it then forecasts. Now each horizon
is fitted twice: early stopping on the split chooses the round count, then the deployed model is
refitted on every settled row, the held-out months included, for exactly that many rounds.

In the synthetic panel the driver `a` sat in [0, 1] for years, then moved to [2, 3] in the last six
settled months, and Adjustment kept following it. The model the old code deployed never saw `a`
above 1, so it can only extrapolate flat and under-forecast those months. The refitted model has
seen them and must have learned them. A single code launched in those months is NOT a fair probe:
six rows are too few for XGBoost's gradient-based splits to isolate (quantile gradients are capped
at +/-alpha), whether or not they are in the training data.

Each half of every plant's real deployment is fitted with its real spec (model_params
BEST_PARAMS_BY_PLANT / BEST_PARAMS_QUANTILE_BY_PLANT) through the real adapters, refitters and sample
weights. The adapters and refitters are wrapped only to record what they were given and returned.
fit_horizon_models only reads the panel, so the panel is shared across plants;
verify_atomic_deploy runs the same path end to end for every plant and checks what the registry
records.

B17: XGBoost early-stops on the LAST eval_set entry, which is always val. Putting the training
set in front of it - on every fit, verbose or not - built a second DMatrix of the whole training
set and evaluated it every round, roughly doubling fit memory at day grain, for a curve nothing
read. It is now evaluated only when verbose. The pre-fix call is reproduced, and the model must be
the same. No database.
"""
import os
import pathlib
import sys
import tempfile
import warnings

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'tests'))
import no_db  # noqa: E402,F401 - every DB engine raises from here on (see no_db.py)

# The real catboost spec writes a catboost_info/ folder into the working directory; keep it out
# of the repo. Nothing below reads a cwd-relative file.
os.chdir(tempfile.mkdtemp(prefix='deployment_refit_'))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from adj_forecast_lib import train as tm  # noqa: E402
from adj_forecast_lib.model_params import BEST_PARAMS_BY_PLANT, BEST_PARAMS_QUANTILE_BY_PLANT  # noqa: E402
from adj_forecast_lib.config import to_month_idx  # noqa: E402
from adj_forecast_lib.models import PREDICT_FUNCS  # noqa: E402

warnings.filterwarnings('ignore')
FAILURES = []


def check(label, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' - ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(label)


SETTLED = int(to_month_idx(202608))
VAL_MONTHS = 6
CODES = [f'C{i:02d}' for i in range(12)]
UNSETTLED_LABEL = 1e6              # a target month that hasn't closed: must never be fitted

rng = np.random.default_rng(0)
rows = [(c, t, h) for c in CODES for t in range(SETTLED - 47, SETTLED + 3) for h in range(1, 7)]
panel = pd.DataFrame(rows, columns=['SalesModelCode', 'TargetIdx', 'Horizon'])
panel['OriginIdx'] = panel['TargetIdx'] - panel['Horizon']
panel['PartCode'] = np.where(panel['SalesModelCode'].isin(CODES[:6]), 'SL', 'XX')
RECENT = panel['TargetIdx'] > SETTLED - VAL_MONTHS     # the held-out months (and the unsettled ones)
panel['a'] = rng.uniform(0, 1, len(panel)) + np.where(RECENT, 2.0, 0.0)
panel['Adjustment'] = 10 * panel['a'] + 5
panel.loc[panel['TargetIdx'] > SETTLED, 'Adjustment'] = UNSETTLED_LABEL
panel['OriginMonthsSinceLaunch'] = 24.0
panel['HasExternalEvent'] = 0
for col in ('SalesModelCode', 'PartCode'):
    panel[col] = panel[col].astype('category')
FEATURES = ['SalesModelCode', 'PartCode', 'a']
before = pd.util.hash_pandas_object(panel, index=True).sum()

# Record, don't replace: each wrapper calls the real adapter/refitter and keeps what it was given.
SEEN = {'fit': [], 'refit': []}
real_adapters, real_refitters = dict(tm.ADAPTERS), dict(tm.REFITTERS)


def recording_adapter(name):
    def fit(train, val, test, *a, **k):
        pred, meta = real_adapters[name](train, val, test, *a, **k)
        SEEN['fit'].append({'train': train, 'val': val, 'meta': meta})
        return pred, meta
    return fit


def recording_refitter(name):
    def refit(data, features, cats, params, n_rounds, **k):
        model = real_refitters[name](data, features, cats, params, n_rounds, **k)
        SEEN['refit'].append({'data': data, 'n_rounds': n_rounds, 'model': model})
        return model
    return refit


def n_trees(model, model_type):
    return (model.tree_count_ if model_type == 'catboost'
            else model.get_booster().num_boosted_rounds())


def mae(pred, actual):
    return float(np.mean(np.abs(np.asarray(pred) - actual.to_numpy())))


for name in tm.ADAPTERS:
    tm.ADAPTERS[name], tm.REFITTERS[name] = recording_adapter(name), recording_refitter(name)
HALVES = [(f'{plant} {half}', params)
          for plant in sorted(BEST_PARAMS_BY_PLANT)
          for half, params in (('standard', BEST_PARAMS_BY_PLANT[plant]),
                               ('quantile', BEST_PARAMS_QUANTILE_BY_PLANT[plant]))]
try:
    for half, params in HALVES:
        print(f'\n=== {half} half ===')
        SEEN['fit'].clear()
        SEEN['refit'].clear()
        fitted = tm.fit_horizon_models(panel, FEATURES, SETTLED, params, covid_weight=0.1,
                                       val_months=VAL_MONTHS)
        check(f'{half}: one early-stopping fit and one refit per horizon',
              len(fitted) == len(SEEN['fit']) == len(SEEN['refit']) == 6)
        for (h, spec, model, info, _history), fit, refit in zip(fitted, SEEN['fit'], SEEN['refit']):
            tag = f"{half} h{h} ({spec['model']})"
            panel_h = panel[panel['Horizon'] == h]
            settled_rows = set(panel_h.index[panel_h['TargetIdx'] <= SETTLED])
            check(f'{tag}: early stopping still holds out exactly the last {VAL_MONTHS} settled months',
                  sorted(fit['val']['TargetIdx'].unique())
                  == list(range(SETTLED - VAL_MONTHS + 1, SETTLED + 1))
                  and fit['train']['TargetIdx'].max() == SETTLED - VAL_MONTHS)
            check(f'{tag}: the deployed model is refitted on every settled row - train AND val - '
                  'and on no row whose target has not closed',
                  set(refit['data'].index) == settled_rows
                  == set(fit['train'].index) | set(fit['val'].index)
                  and info['n_refit'] == len(settled_rows) == info['n_train'] + info['n_val'],
                  str(info))
            check(f'{tag}: ...for exactly the {info["n_rounds"]} round(s) early stopping chose, and it '
                  'has that many trees', refit['n_rounds'] == fit['meta']['n_rounds']
                  == info['n_rounds'] == n_trees(model, spec['model']),
                  f"refit={refit['n_rounds']} fit={fit['meta']['n_rounds']} "
                  f"trees={n_trees(model, spec['model'])}")
            check(f'{tag}: the model returned for deployment IS the refit, not the early-stopped fit',
                  model is refit['model'] and model is not fit['meta']['model'])

            recent = panel_h[panel_h['TargetIdx'].between(SETTLED - VAL_MONTHS + 1, SETTLED)]
            predict = PREDICT_FUNCS[spec['model']]
            old_mae = mae(predict(fit['meta']['model'], recent[FEATURES]), recent['Adjustment'])
            new_mae = mae(predict(model, recent[FEATURES]), recent['Adjustment'])
            check(f'{tag}: the model the old code deployed never saw the last {VAL_MONTHS} months, '
                  f'so it under-forecasts them (MAE {old_mae:.1f})', old_mae > 8)
            check(f"{tag}: the deployed model learned them: MAE {new_mae:.1f}, under half the old "
                  "model's", new_mae < 0.5 * old_mae)
            check(f'{tag}: no unsettled label reached it - every prediction is in range',
                  float(np.max(predict(model, panel_h[FEATURES]))) < 1000)
finally:
    tm.ADAPTERS.update(real_adapters)
    tm.REFITTERS.update(real_refitters)

check('fitting never modified the panel it was given',
      pd.util.hash_pandas_object(panel, index=True).sum() == before)

print('\n=== B17. a silent XGBoost fit evaluates only the validation set ===')
from xgboost import XGBRegressor  # noqa: E402
from adj_forecast_lib.config import EARLY_STOPPING_ROUNDS, N_ESTIMATORS  # noqa: E402
from adj_forecast_lib.evaluation import build_covid_sample_weight, fit_predict  # noqa: E402
from adj_forecast_lib.folds import deployment_split_horizon  # noqa: E402

panel_2 = panel[panel['Horizon'] == 2]
train_2, val_2 = deployment_split_horizon(panel_2, SETTLED, val_months=VAL_MONTHS)
spec_2 = next(s for p in sorted(BEST_PARAMS_BY_PLANT) for h, s in BEST_PARAMS_BY_PLANT[p].items()
              if s['model'] == 'xgboost')     # this section is about the XGBoost adapter
weights = lambda df: build_covid_sample_weight(df, 0.1)


def old_xgboost_fit(train, val):
    """The pre-fix call: the training set in eval_set on every fit, verbose or not."""
    w_tr, w_va = weights(train), weights(val)
    model = XGBRegressor(n_estimators=N_ESTIMATORS, early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                         **spec_2['params'])
    model.fit(train[FEATURES], train['Adjustment'],
              eval_set=[(train[FEATURES], train['Adjustment']), (val[FEATURES], val['Adjustment'])],
              verbose=False, sample_weight=w_tr, sample_weight_eval_set=[w_tr, w_va])
    return model


fit_xgb = lambda **k: tm.ADAPTERS['xgboost'](train_2, val_2, val_2, FEATURES, tm.CATS,
    spec_2['params'], N_ESTIMATORS, EARLY_STOPPING_ROUNDS, weight_fn=weights, **k)[1]['model']
old, silent = old_xgboost_fit(train_2, val_2), fit_xgb()
check('a silent fit evaluates only the validation set - no second copy of the training set, no '
      'pass over it every round (the old call evaluated both)',
      list(silent.evals_result()) == ['validation_0'] and len(old.evals_result()) == 2,
      f'silent={list(silent.evals_result())} old={list(old.evals_result())}')
check('...and it is the same model: same stopping round, same predictions',
      silent.best_iteration == old.best_iteration
      and np.allclose(silent.predict(panel_2[FEATURES]), old.predict(panel_2[FEATURES]),
                      rtol=0, atol=1e-6), f'{silent.best_iteration} vs {old.best_iteration}')
loud = fit_xgb(verbose=100000)                  # prints only its first and last round
check('verbose still evaluates both, for the train curve, and stops at the same round',
      len(loud.evals_result()) == 2 and loud.best_iteration == silent.best_iteration)
_, model_e = fit_predict(train_2, val_2, feature_cols=['a'])
check("evaluation.fit_predict (the notebooks' fit) evaluates only the validation set too",
      list(model_e.evals_result()) == ['validation_0'], str(list(model_e.evals_result())))

print('\n' + '=' * 78)
print(f'{len(FAILURES)} FAILURE(S): {FAILURES}' if FAILURES else 'ALL CHECKS PASSED')
sys.exit(1 if FAILURES else 0)
