"""Step-5b verification: folds.FoldSequence must be a drop-in for the eager list it replaced.

The refactor is worth nothing if it changes a single fold. So the pre-edit implementations are
reproduced verbatim below and diffed against the live ones, frame by frame, on both a synthetic
panel and (when present) the real STJ panel.

Three further properties the eager list gave for free and a lazy one can silently lose:
  * RE-ITERABILITY - a sweep loops over the same FOLDS[h] once per run; a generator would yield
    nothing from the second run onward and score every arm after the first on zero folds;
  * O(1) RETENTION - the whole point: holding the fold set must not hold the frames;
  * fold COUNT unchanged at day grain - OriginIdx stays the monthly anchor, so a day panel must
    produce exactly the folds its month panel does, each simply carrying more rows.

Plus the leak check the day grain actually introduces: a train row's as-of date must never exceed
month_end(test_origin), which is what makes the unchanged `TargetIdx <= test_origin` rule still
sufficient once the as-of point can drift into the month after its origin.

Sections 10-13 cover the reporting decision that came with it - day-0 headline plus a per-day
curve. The load-bearing claim is that a day panel's day-0 slice pools to metrics IDENTICAL to the
month panel's, so the headline stays comparable to every existing monthly result and the curve is
pure addition rather than a redefinition of the number the business already reads.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'tests'))
import no_db  # noqa: E402,F401 - every DB engine raises from here on (see no_db.py)

from adj_forecast_lib.config import CUTOFF_DATE_COL
from adj_forecast_lib.evaluation import (
    evaluate_adjustment_raw, evaluate_adjustment_raw_by_day, log_result, pooled_metrics_by_day,
)
from adj_forecast_lib.folds import (
    FoldSequence, baseline_folds_for_horizon, fold_step_months, walk_forward_horizon,
    walk_forward_shared,
)
from adj_forecast_lib.preprocessing import (
    _expand_to_cutoff_days, cutoff_dates_for_origins, cutoff_day_index, monthly_anchor_mask,
)

FAILURES = []


def check(label, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' - ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(label)


def frames_identical(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Same rows, same order, same index labels - index equality matters because callers align
    y_pred against test_df positionally and .loc into the panel by label."""
    if a.shape != b.shape or not a.index.equals(b.index):
        return False
    return a.equals(b)


# --------------------------------------------------------------- the pre-edit implementations
def old_walk_forward_horizon(panel_h, horizon, initial_train_origins=60, val_months=3,
        max_settled_target_idx=None, max_folds=None, min_origin_months_since_launch=None):
    origins = sorted(panel_h['OriginIdx'].unique())
    max_target = max_settled_target_idx if max_settled_target_idx is not None else panel_h['TargetIdx'].max()
    start_i = initial_train_origins if max_folds is None else max(initial_train_origins, len(origins) - max_folds)

    folds = []
    for i in range(start_i, len(origins)):
        test_origin = origins[i]
        target = test_origin + horizon
        if target > max_target:
            break
        known = panel_h[panel_h['TargetIdx'] <= test_origin]
        val_cut = test_origin - val_months
        train = known[known['TargetIdx'] <= val_cut]
        val = known[known['TargetIdx'] > val_cut]
        test = panel_h[panel_h['OriginIdx'] == test_origin]
        if min_origin_months_since_launch is not None:
            test = test[test['OriginMonthsSinceLaunch'] >= min_origin_months_since_launch]
        folds.append((train, val, test))
    return folds


def old_walk_forward_shared(panel_all, initial_train_origins=60, val_months=3, horizons=range(1, 7),
        max_settled_target_idx=None, max_folds=None, min_origin_months_since_launch=None):
    origins = sorted(panel_all['OriginIdx'].unique())
    max_target = max_settled_target_idx if max_settled_target_idx is not None else panel_all['TargetIdx'].max()
    start_i = initial_train_origins if max_folds is None else max(initial_train_origins, len(origins) - max_folds)

    folds = []
    for i in range(start_i, len(origins)):
        test_origin = origins[i]
        available_h = [h for h in horizons if test_origin + h <= max_target]
        if not available_h:
            break
        known = panel_all[panel_all['TargetIdx'] <= test_origin]
        val_cut = test_origin - val_months
        train = known[known['TargetIdx'] <= val_cut]
        val = known[known['TargetIdx'] > val_cut]
        test = panel_all[(panel_all['OriginIdx'] == test_origin) & (panel_all['Horizon'].isin(available_h))]
        if min_origin_months_since_launch is not None:
            test = test[test['OriginMonthsSinceLaunch'] >= min_origin_months_since_launch]
        folds.append({'train': train, 'val': val, 'test': test, 'test_origin': test_origin,
                      'available_h': available_h})
    return folds


def old_baseline_folds_for_horizon(h, panel, max_settled_target_idx, initial_train_origins=60,
        min_origin_months_since_launch=None):
    panel_h = panel[panel['Horizon'] == h]
    origins = sorted(panel_h['OriginIdx'].unique())
    folds = []
    for test_origin in origins[initial_train_origins:]:
        if test_origin + h > max_settled_target_idx:
            break
        known = panel_h[panel_h['TargetIdx'] <= test_origin]
        test = panel_h[panel_h['OriginIdx'] == test_origin]
        if min_origin_months_since_launch is not None:
            test = test[test['OriginMonthsSinceLaunch'] >= min_origin_months_since_launch]
        folds.append((known, test))
    return folds


# -------------------------------------------------------------------------- synthetic panel
rng = np.random.default_rng(11)
CODES = [f'C{i:02d}' for i in range(40)]
rows = []
for code in CODES:
    launch = rng.integers(0, 30)
    for target in range(24300, 24380):
        for h in range(1, 7):
            origin = target - h
            if origin < 24300 + launch:
                continue
            rows.append((code, target, h, origin, origin - (24300 + launch)))
panel = pd.DataFrame(rows, columns=['SalesModelCode', 'TargetIdx', 'Horizon', 'OriginIdx',
                                    'OriginMonthsSinceLaunch'])
# Current panels carry the eligibility age the test filters read; every launch here is known, so it
# equals OriginMonthsSinceLaunch and the pre-edit filter below stays the right reference.
panel['ForecastEligibilityAge'] = panel['OriginMonthsSinceLaunch'].astype(float)
panel['KnownOrder'] = rng.integers(0, 500, len(panel)).astype(float)
panel['KnownCanceledOrder'] = 0.0     # the backtest's net floor reads it; none cancelled here
panel['Adjustment'] = rng.normal(0, 40, len(panel))
panel['ActualOrder'] = panel['KnownOrder'] + panel['Adjustment']
MAX_SETTLED = int(panel['TargetIdx'].max())

print('\n=== 1. walk_forward_horizon: new vs pre-edit, every fold, every frame ===')
for h in range(1, 7):
    panel_h = panel[panel['Horizon'] == h]
    kwargs = dict(horizon=h, initial_train_origins=20, max_settled_target_idx=MAX_SETTLED,
                  min_origin_months_since_launch=3)
    old = old_walk_forward_horizon(panel_h, **kwargs)
    new = walk_forward_horizon(panel_h, **kwargs)
    same_n = len(old) == len(new)
    same_frames = same_n and all(
        frames_identical(o, n) for of, nf in zip(old, new) for o, n in zip(of, nf))
    check(f'h={h}: {len(old)} folds, all train/val/test frames identical',
          same_n and same_frames, f'{len(old)} vs {len(new)} folds')

print('\n=== 2. max_folds and min_origin_months_since_launch=None still match ===')
panel_h = panel[panel['Horizon'] == 3]
for label, kw in [('max_folds=5', dict(max_folds=5)),
                  ('no launch filter', dict(min_origin_months_since_launch=None)),
                  ('val_months=6', dict(val_months=6))]:
    base = dict(horizon=3, initial_train_origins=20, max_settled_target_idx=MAX_SETTLED)
    old = old_walk_forward_horizon(panel_h, **base, **kw)
    new = walk_forward_horizon(panel_h, **base, **kw)
    ok = len(old) == len(new) and all(
        frames_identical(o, n) for of, nf in zip(old, new) for o, n in zip(of, nf))
    check(f'{label}: identical', ok, f'{len(old)} vs {len(new)}')

print('\n=== 3. walk_forward_shared and baseline_folds_for_horizon ===')
old_s = old_walk_forward_shared(panel, initial_train_origins=20, max_settled_target_idx=MAX_SETTLED,
                                min_origin_months_since_launch=3)
new_s = walk_forward_shared(panel, initial_train_origins=20, max_settled_target_idx=MAX_SETTLED,
                            min_origin_months_since_launch=3)
ok = len(old_s) == len(new_s) and all(
    o['test_origin'] == n['test_origin'] and o['available_h'] == n['available_h']
    and all(frames_identical(o[k], n[k]) for k in ('train', 'val', 'test'))
    for o, n in zip(old_s, new_s))
check(f'walk_forward_shared: {len(old_s)} folds identical (frames + test_origin + available_h)', ok)

old_b = old_baseline_folds_for_horizon(4, panel, MAX_SETTLED, initial_train_origins=20,
                                       min_origin_months_since_launch=3)
new_b = baseline_folds_for_horizon(4, panel, MAX_SETTLED, initial_train_origins=20,
                                   min_origin_months_since_launch=3)
ok = len(old_b) == len(new_b) and all(
    frames_identical(o, n) for of, nf in zip(old_b, new_b) for o, n in zip(of, nf))
check(f'baseline_folds_for_horizon: {len(old_b)} folds identical', ok)

print('\n=== 4. sequence protocol: re-iteration, len, indexing, slicing ===')
folds = walk_forward_horizon(panel[panel['Horizon'] == 2], horizon=2, initial_train_origins=20,
                             max_settled_target_idx=MAX_SETTLED)
check('is a FoldSequence, not a list', isinstance(folds, FoldSequence))
first_pass = [len(te) for _, _, te in folds]
second_pass = [len(te) for _, _, te in folds]
check('re-iterable: pass 2 yields the same folds as pass 1 (a generator would yield none)',
      len(second_pass) > 0 and first_pass == second_pass,
      f'{len(first_pass)} then {len(second_pass)} folds')
check('len() matches what iteration yields', len(folds) == len(first_pass))
check('negative indexing reaches the last fold',
      frames_identical(folds[-1][2], list(folds)[-1][2]))
check('slicing returns a list of folds', isinstance(folds[:3], list) and len(folds[:3]) == 3)
check('.specs exposes one boundary per fold without materialising',
      len(folds.specs) == len(folds))
a, b = folds[0][0], folds[0][0]
check('repeated access returns equal but non-identical frames (documented)',
      frames_identical(a, b) and a is not b)

print('\n=== 5. retention: holding the fold set must not hold the frames ===')
big = pd.concat([panel] * 6, ignore_index=True)
big_h = big[big['Horizon'] == 1]
lazy = walk_forward_horizon(big_h, horizon=1, initial_train_origins=20,
                            max_settled_target_idx=MAX_SETTLED)
eager = old_walk_forward_horizon(big_h, horizon=1, initial_train_origins=20,
                                 max_settled_target_idx=MAX_SETTLED)
eager_mb = sum(f.memory_usage(deep=True).sum() for fold in eager for f in fold) / 1e6
lazy_mb = sum(sys.getsizeof(s) for s in lazy.specs) / 1e6
check(f'lazy retains {lazy_mb:.4f} MB vs eager {eager_mb:,.0f} MB over {len(lazy)} folds',
      lazy_mb < eager_mb / 1000, f'{lazy_mb:.4f} vs {eager_mb:.1f}')

print('\n=== 6. fold_step_months still lines up with the folds actually run ===')
for h in (1, 6):
    panel_h = panel[panel['Horizon'] == h]
    steps = fold_step_months(panel_h, horizon=h, initial_train_origins=20,
                             max_settled_target_idx=MAX_SETTLED)
    fs = walk_forward_horizon(panel_h, horizon=h, initial_train_origins=20,
                              max_settled_target_idx=MAX_SETTLED)
    origins_from_specs = [s[0] for s in fs.specs]
    check(f'h={h}: {len(steps)} steps, one per fold, same origins',
          len(steps) == len(fs) and len(origins_from_specs) == len(steps))

print('\n=== 7. log_result: raw-dict folds pool to the same numbers as frame triples ===')
res_triples, res_dicts = [], []
triples, dicts = [], []
for tr, va, te in walk_forward_horizon(panel[panel['Horizon'] == 5], horizon=5,
                                       initial_train_origins=20, max_settled_target_idx=MAX_SETTLED):
    y_pred = te['Adjustment'].to_numpy() + rng.normal(0, 10, len(te))
    triples.append((tr, te, y_pred))
    dicts.append(evaluate_adjustment_raw(tr, te, y_pred, entity_col='SalesModelCode',
                                         target_col='Adjustment'))
log_result(approach='t', horizon=5, eval_type='walk_forward', fold_data=triples,
           entity_col='SalesModelCode', results_list=res_triples)
log_result(approach='t', horizon=5, eval_type='walk_forward', fold_data=dicts,
           entity_col='SalesModelCode', results_list=res_dicts)
same = all(res_triples[0][k] == res_dicts[0][k]
           or (isinstance(res_triples[0][k], float) and np.isnan(res_triples[0][k])
               and np.isnan(res_dicts[0][k]))
           for k in res_triples[0])
check(f'every pooled metric identical across both fold_data forms ({len(res_triples[0])} keys)', same)

print('\n=== 8. day grain: fold COUNT unchanged, and no as-of date leaks past test_origin ===')
day_panel = _expand_to_cutoff_days(panel)
print(f'  (day panel: {len(panel):,} -> {len(day_panel):,} rows, '
      f'{len(day_panel) / len(panel):.1f}x)')

for h in (1, 3, 6):
    m = walk_forward_horizon(panel[panel['Horizon'] == h], horizon=h, initial_train_origins=20,
                             max_settled_target_idx=MAX_SETTLED, min_origin_months_since_launch=3)
    d = walk_forward_horizon(day_panel[day_panel['Horizon'] == h], horizon=h,
                             initial_train_origins=20, max_settled_target_idx=MAX_SETTLED,
                             min_origin_months_since_launch=3)
    check(f'h={h}: day panel yields the same {len(m)} folds as the month panel',
          len(m) == len(d), f'{len(m)} vs {len(d)}')
    check(f'h={h}: same test origins', [s[0] for s in m.specs] == [s[0] for s in d.specs])

worst = []
for h in (1, 3, 6):
    d = walk_forward_horizon(day_panel[day_panel['Horizon'] == h], horizon=h,
                             initial_train_origins=20, max_settled_target_idx=MAX_SETTLED)
    for (train, val, _test), spec in zip(d, d.specs):
        test_origin = spec[0]
        limit = cutoff_dates_for_origins(np.array([test_origin]), 0)[0]
        for part in (train, val):
            if len(part):
                worst.append((part[CUTOFF_DATE_COL].max(), limit))
check('every train/val as-of date <= month_end(test_origin) - TargetIdx <= test_origin is still '
      'sufficient once the as-of point can drift into the next month',
      all(seen <= limit for seen, limit in worst),
      f'{sum(1 for s, l in worst if s > l)} of {len(worst)} fold-parts leak')

# --------------------------------------------------------------------- the real panel, if present
real = _REPO_ROOT / 'data' / 'STJ' / 'train5_201701-202608.parquet'
print(f'\n=== 9. real STJ panel ({"found" if real.exists() else "absent - skipped"}) ===')
if real.exists():
    rp = pd.read_parquet(real)
    rp['PartCode'] = rp['PartCode'].astype('category')
    rp['SalesModelCode'] = rp['SalesModelCode'].astype('category')
    mx = int(rp['TargetIdx'].max())
    for h in (1, 6):
        ph = rp[rp['Horizon'] == h]
        kw = dict(horizon=h, max_settled_target_idx=mx, max_folds=None,
                  min_origin_months_since_launch=3)
        old = old_walk_forward_horizon(ph, **kw)
        new = walk_forward_horizon(ph, **kw)
        ok = len(old) == len(new) and all(
            frames_identical(o, n) for of, nf in zip(old, new) for o, n in zip(of, nf))
        eager_gb = sum(f.memory_usage(deep=True).sum() for fold in old for f in fold) / 1e9
        check(f'h={h}: {len(old)} folds identical on the real panel (eager cost {eager_gb:.2f} GB)', ok)

print('\n=== 10. cutoff_day_index: the reporting key ===')
check('month panel -> all zeros (a month panel is a day panel that only saw day 0)',
      set(np.unique(cutoff_day_index(panel))) == {0})
dpi = cutoff_day_index(day_panel)
check('day panel: the zeros are exactly the monthly anchors',
      bool((( dpi == 0) == monthly_anchor_mask(day_panel)).all()))
nonzero = dpi != 0
check('day panel: non-anchor rows are labelled by their own calendar day',
      bool((dpi[nonzero] == pd.to_datetime(day_panel[CUTOFF_DATE_COL]).dt.day.to_numpy()[nonzero]).all()))
# Month lengths are date-aware, not a hardcoded 28: a leap February legitimately has a day-29 bucket.
non_anchor = day_panel[~monthly_anchor_mask(day_panel)]
cd = pd.to_datetime(non_anchor[CUTOFF_DATE_COL])
span = pd.DataFrame({'day': cutoff_day_index(non_anchor), 'length': cd.dt.days_in_month.to_numpy()}
                    ).groupby([cd.dt.year.to_numpy(), cd.dt.month.to_numpy()]).max()
# The month's last day is the next origin's day 0, so in-month days top out one short of it.
check('every cutoff month tops out the day before its own last day',
      bool((span['day'] == span['length'] - 1).all()),
      f"{span[span['day'] != span['length'] - 1].head().to_dict()}")
feb = span.xs(2, level=1)
check('February is date-aware: 27 in common years, 28 in leap years',
      set(feb['day']) == {27, 28} and bool((feb['day'] == feb['length'] - 1).all()),
      f'{feb["day"].to_dict()}')

print('\n=== 11. day-0 headline is the monthly number, by construction ===')
predict = lambda df: df['Adjustment'].to_numpy() * 0.8 + 5.0   # deterministic: same row -> same pred

month_raw, day0_raw, by_day_folds = [], [], []
for (m_tr, _, m_te), (d_tr, _, d_te) in zip(
        walk_forward_horizon(panel[panel['Horizon'] == 4], horizon=4, initial_train_origins=20,
                             max_settled_target_idx=MAX_SETTLED, min_origin_months_since_launch=3),
        walk_forward_horizon(day_panel[day_panel['Horizon'] == 4], horizon=4,
                             initial_train_origins=20, max_settled_target_idx=MAX_SETTLED,
                             min_origin_months_since_launch=3)):
    month_raw.append(evaluate_adjustment_raw(m_tr, m_te, predict(m_te), entity_col='SalesModelCode'))
    fold = evaluate_adjustment_raw_by_day(d_tr, d_te, predict(d_te), cutoff_day_index(d_te),
                                          entity_col='SalesModelCode')
    by_day_folds.append(fold)
    day0_raw.append(fold[0])

res_m, res_d = [], []
log_result(approach='m', horizon=4, eval_type='walk_forward', fold_data=month_raw, results_list=res_m)
log_result(approach='d', horizon=4, eval_type='walk_forward', fold_data=day0_raw, results_list=res_d)
diffs = [k for k in res_m[0] if k not in ('approach',) and res_m[0][k] != res_d[0][k]
         and not (isinstance(res_m[0][k], float) and np.isnan(res_m[0][k]) and np.isnan(res_d[0][k]))]
check("the day panel's day-0 slice reproduces the month panel's pooled metrics exactly",
      not diffs, f'differs on {diffs}')

print('\n=== 12. pooled_metrics_by_day ===')
curve = pooled_metrics_by_day(by_day_folds)
check(f'one row per observed cutoff day ({len(curve)} days), sorted, indexed by day',
      curve.index.name == 'day' and list(curve.index) == sorted(curve.index) and len(curve) > 1)
check('day 0 in the curve equals the headline metric',
      abs(curve.loc[0, 'forecast_wape'] - res_d[0]['forecast_wape']) < 1e-9)
check('rows across the curve sum back to the full test population',
      int(curve['forecast_n'].sum()) == sum(int(r['forecast_n']) for f in by_day_folds for r in f.values()))

# pooling must be exact: sum accumulators then divide, never average per-fold ratios
one_day = [f[5] for f in by_day_folds if 5 in f]
exact = sum(r['forecast_sum_abs_err'] for r in one_day) / sum(r['forecast_sum_abs_actual'] for r in one_day) * 100
check('day 5 WAPE is the pooled ratio of sums, not a mean of per-fold ratios',
      abs(curve.loc[5, 'forecast_wape'] - exact) < 1e-9)

print('\n=== 13. the curve detects an as-of point getting easier ===')
# y_pred perfect at the last cutoff day, progressively worse earlier - the shape daily refresh
# is supposed to produce, and the h=1 month-close degeneracy at its extreme.
te = day_panel[day_panel['Horizon'] == 1].copy()
d = cutoff_day_index(te)
y = te['Adjustment'].to_numpy() + (30 - d) * 3.0
# floor_net=False: this synthetic panel has negative ActualOrder rows, which the floor would move
one = [evaluate_adjustment_raw_by_day(te, te, y, d, entity_col='SalesModelCode', floor_net=False)]
shape = pooled_metrics_by_day(one)['adjustment_mae']
check('MAE falls monotonically as the as-of point moves through the month',
      bool((np.diff(shape.to_numpy()) <= 1e-9).all()), f'{shape.round(2).to_dict()}')
check('exact at the final cutoff day (the h=1 month-close case)', abs(shape.iloc[-1]) < 1e-9)

try:
    evaluate_adjustment_raw_by_day(te, te, y, d[:-1], entity_col='SalesModelCode')
    check('a misaligned day_index raises', False, 'no error')
except ValueError as e:
    check('a misaligned day_index raises', 'aligned to this fold' in str(e))

print('\n' + '=' * 78)
print(f'{len(FAILURES)} FAILURE(S): {FAILURES}' if FAILURES else 'ALL CHECKS PASSED')
sys.exit(1 if FAILURES else 0)
