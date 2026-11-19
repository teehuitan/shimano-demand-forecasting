"""Net orders are grain-consistent: KnownCanceledOrder is read at the same as-of point as KnownOrder.

The bug this pins: KnownCanceledOrder used to be computed after scoring, from OriginYYYYMM alone,
against a month-grain cancellation book. At day grain that netted only the cancellations
recorded by the end of the ORIGIN month off a KnownOrder counted to the cutoff DATE - so every
cancellation between the two was missed, and KnownNetOrder / PredictedNetOrder came out too high
(reproduced: +80 units on a 90-unit net, +89%). It was never look-ahead - only stale.

Everything here runs the REAL context, book, skeleton, training, scoring and DB-row code over the
synthetic feed (tests/synthetic_fabric.py), and every KnownOrder / KnownCanceledOrder is checked
against ground truth computed straight from the order rows - no order book involved. Every section
runs for every plant in config.PLANT_NAMES. No database: tests/no_db.py makes any real engine raise.

Also covered, since they share the scoring path: day-grain range scoring (it used to be impossible),
the as-of-after-the-data guard, the parquet upsert key, and the empty-scoring-half crash.
"""
import inspect
import logging
import pathlib
import sys
import tempfile
import warnings

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'tests'))
import no_db  # noqa: E402,F401 - every DB engine raises from here on (see no_db.py)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import main as orch  # noqa: E402
from adj_forecast_lib import db_writer, order_book as ob, predict as pm  # noqa: E402
from adj_forecast_lib import preprocessing as pp, train as tm  # noqa: E402
from adj_forecast_lib import model_registry as mr  # noqa: E402
from adj_forecast_lib.config import (  # noqa: E402
    CUTOFF_DATE_COL, PLANT_INCLUDES_MANUAL_METHOD, PLANT_NAMES, to_month_idx,
)
from adj_forecast_lib.cutoff_grain import cutoff_dates_for_origins  # noqa: E402
from synthetic_fabric import (  # noqa: E402
    DATA_THROUGH, FILE_DATE, KNOWN_CODE, KNOWN_TARGET_YYYYMM, FabricStub, RecordingEngine, ground_truth,
    make_orders,
)

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
TMP = pathlib.Path(tempfile.mkdtemp(prefix='net_orders_'))
ORDERS = make_orders()
FAST = {h: dict(model='xgboost', params=dict(max_depth=2, n_jobs=1, random_state=0,
            enable_categorical=True), young_sku_boost=1.0, external_event_weight=1.0)
        for h in range(1, 7)}
DAY_ASOFS = [pd.Timestamp(d) for d in
             ('2026-08-31', '2026-09-05', '2026-09-12', '2026-09-15', '2026-09-20')]
KNOWN_T = int(to_month_idx(KNOWN_TARGET_YYYYMM))


def matches(values, truth, what):
    a, b = np.asarray(values, dtype='float64'), np.asarray(truth, dtype='float64')
    bad = ~np.isclose(a, b)
    return (not bad.any()), f'{int(bad.sum())} of {len(a)} {what} row(s) differ, e.g. {a[bad][:3]} vs {b[bad][:3]}'


def known_row(sk):
    return sk[(sk['PlanModelCode'].astype(str) == KNOWN_CODE) & (sk['TargetIdx'] == KNOWN_T)].iloc[0]


def pre_fix_known_canceled(panel):
    """The removed compute_known_canceled_order, reproduced: month cancel book queried at
    OriginIdx whatever the grain. Month rows must match it exactly (behaviour preserved)."""
    cb = ORDERS[ORDERS['IsCanceled']].drop(columns=['EntryMonthIdx']) \
        .rename(columns={'CancelMonthIdx': 'EntryMonthIdx'})
    cb['EntryMonthIdx'] = cb['EntryMonthIdx'].astype('int32')
    book = ob.build_order_book(cb, ['PlanModelCode'])
    q = pd.DataFrame({'PlanModelCode': panel['PlanModelCode'].astype(str).to_numpy(),
                      'RequestMonthIdx': panel['TargetIdx'].to_numpy(),
                      'CutoffMonthIdx': panel['OriginIdx'].to_numpy()})
    return np.nan_to_num(ob.query_known_order(q, book, ['PlanModelCode']), nan=0.0)


print('\n=== 1. KnownOrder and KnownCanceledOrder vs the order rows, every row, both grains ===')
SK = {}
with FabricStub(ORDERS):
    for plant in PLANTS:
        tag = f'{plant} ({PLANT_NAMES[plant]})'
        for asof in DAY_ASOFS:
            sk, ctx = pp.preprocess_for_inference(plant, horizons=range(1, 7), cutoff_grain='day',
                cutoff_date=asof, min_origin_months_since_launch=None)
            SK[(plant, 'day', asof)] = (sk, ctx)
            t = ground_truth(ORDERS, sk, CUTOFF_DATE_COL)
            ok_g, d_g = matches(sk['KnownOrder'], t['gross'], 'KnownOrder')
            ok_c, d_c = matches(sk['KnownCanceledOrder'], t['cancelled'], 'KnownCanceledOrder')
            check(f'{tag} day, as of {asof.date()}: both match on all {len(sk)} rows', ok_g and ok_c,
                  f'{d_g}; {d_c}')
        sk_m, ctx_m = pp.preprocess_for_inference(plant, horizons=range(1, 7), cutoff_grain='month',
            min_origin_months_since_launch=None)
        SK[(plant, 'month')] = (sk_m, ctx_m)
        t = ground_truth(ORDERS, sk_m)
        ok_g, d_g = matches(sk_m['KnownOrder'], t['gross'], 'KnownOrder')
        ok_c, d_c = matches(sk_m['KnownCanceledOrder'], t['cancelled'], 'KnownCanceledOrder')
        check(f'{tag} month: both match on all {len(sk_m)} rows', ok_g and ok_c, f'{d_g}; {d_c}')
        ok, d = matches(sk_m['KnownCanceledOrder'], pre_fix_known_canceled(sk_m), 'month')
        check(f'{tag} month: identical to the pre-fix calculation (month behaviour preserved)', ok, d)

        sk15 = SK[(plant, 'day', pd.Timestamp('2026-09-15'))][0]
        r15, r20 = known_row(sk15), known_row(SK[(plant, 'day', pd.Timestamp('2026-09-20'))][0])
        rm = known_row(sk_m)
        check(f'{tag} known SKU as of Sep 15: gross 270, cancelled 180, net 90',
              (r15['KnownOrder'], r15['KnownCanceledOrder']) == (270, 180),
              f"{r15['KnownOrder']}, {r15['KnownCanceledOrder']}")
        check(f'{tag} ...the Sep 18 cancellation is NOT known on Sep 15, IS known on Sep 20 (no look-ahead)',
              r20['KnownCanceledOrder'] == 220, f"{r20['KnownCanceledOrder']}")
        check(f'{tag} ...month grain (as of Aug 31): 150 / 100, net 50',
              (rm['KnownOrder'], rm['KnownCanceledOrder']) == (150, 100))
        ok, d = matches(pre_fix_known_canceled(sk15), sk15['KnownCanceledOrder'], 'day-15')
        check(f'{tag} the pre-fix calculation DIFFERS at day 15 (the defect this fixes)', not ok)
        d0 = SK[(plant, 'day', pd.Timestamp('2026-08-31'))][0]
        key = ['PlanModelCode', 'TargetIdx', 'Horizon']
        j = d0.assign(PlanModelCode=d0['PlanModelCode'].astype(str)).merge(
            sk_m.assign(PlanModelCode=sk_m['PlanModelCode'].astype(str))[
                key + ['KnownOrder', 'KnownCanceledOrder']], on=key, suffixes=('_d', '_m'))
        check(f'{tag} day 0 == month grain on both columns, row for row ({len(j)} rows)',
              len(j) == len(sk_m) and np.allclose(j['KnownOrder_d'], j['KnownOrder_m'])
              and np.allclose(j['KnownCanceledOrder_d'], j['KnownCanceledOrder_m']))
        every = [s for k, (s, _c) in SK.items() if k[0] == plant]
        check(f'{tag} KnownCanceledOrder <= KnownOrder on every row, every as-of point',
              all((s['KnownCanceledOrder'] <= s['KnownOrder'] + 1e-6).all() for s in every))

print('\n=== 1b. the TRAINING panel carries the same KnownCanceledOrder - the backtest\'s net floor ===')
with FabricStub(ORDERS):
    for plant in PLANTS:
        tag = f'{plant} ({PLANT_NAMES[plant]})'
        for grain, asof_col in (('month', None), ('day', CUTOFF_DATE_COL)):
            panel, _fams, fcols, _ctx = pp.preprocess_for_training(plant, cutoff_grain=grain)
            t = ground_truth(ORDERS, panel, asof_col)
            ok, d = matches(panel['KnownCanceledOrder'], t['cancelled'], 'training KnownCanceledOrder')
            check(f'{tag} {grain}: training KnownCanceledOrder matches the order rows on all '
                  f'{len(panel)} rows', ok, d)
            check(f'{tag} {grain}: ...and is not a feature', 'KnownCanceledOrder' not in fcols
                  and 'KnownCanceledOrder' not in pp.full_feature_cols())
        # An origin whose targets have settled, so the training panel has its rows too (the latest
        # origin's targets are all still open).
        sk_m = pp.preprocess_for_inference(plant, 202602, range(1, 7), cutoff_grain='month',
                                           min_origin_months_since_launch=None)[0]
        key = ['PlanModelCode', 'TargetIdx', 'Horizon']
        m_panel = pp.preprocess_for_training(plant, cutoff_grain='month')[0]
        j = sk_m.assign(PlanModelCode=sk_m['PlanModelCode'].astype(str))[key + ['KnownCanceledOrder']] \
            .merge(m_panel.assign(PlanModelCode=m_panel['PlanModelCode'].astype(str))[key + ['KnownCanceledOrder']],
                   on=key, suffixes=('_live', '_train'))
        check(f'{tag}: a training row and the live row at the same as-of point agree ({len(j)} rows)',
              len(j) > 0 and np.allclose(j['KnownCanceledOrder_live'], j['KnownCanceledOrder_train']))

print('\n=== 2. a grain or as-of mismatch fails loudly instead of netting wrongly ===')
plant = PLANTS[0]
sk15, ctx15 = SK[(plant, 'day', pd.Timestamp('2026-09-15'))]
sk_m, ctx_m = SK[(plant, 'month')]
ok, d = raises(lambda: ob.known_canceled_order(sk15, ctx15['known_canceled_order_book_code'],
                                              asof_col=CUTOFF_DATE_COL, arrival_col='CancelMonthIdx'),
               ValueError, 'grain mismatch')
check('day as-of dates against the MONTH cancellation book -> grain mismatch', ok, d)
later = sk_m.copy()
later[CUTOFF_DATE_COL] = pd.Timestamp('2026-09-20')
ok, d = raises(lambda: ob.known_canceled_order(later, ctx15['known_canceled_order_book_code_daily'],
                                              asof_col=CUTOFF_DATE_COL, arrival_col='CancelKnownDate'),
               ValueError, 'more cancelled than ordered')
check('cancellations read LATER than the orders they net against -> refused', ok, d)
ok, d = raises(lambda: ob.known_canceled_order(sk_m, None), ValueError, 'no cancellation book')
check('no cancellation book for the grain -> refused', ok, d)

print('\n=== 3. what the orchestrator writes to PredictionResult (real _prediction_phase) ===')
# One MLflow store per process, so each deployment below is its own label, not its own folder.
DAY, MONTH, NOQ = 'net_day', 'net_month', 'net_noq'
for plant in PLANTS:
    tag = f'{plant} ({PLANT_NAMES[plant]})'
    with FabricStub(ORDERS):
        tm.train_deployment(plant, DAY, cutoff_grain='day', best_params=FAST, best_params_quantile=FAST)
        eng, saved = RecordingEngine(), db_writer._engine
        db_writer._engine = eng
        try:
            # a file dated Sep 16 is complete through Sep 15 - the cutoff this run scores at
            orch._prediction_phase(plant, DAY, range(1, 7), 3, None, 6, cutoff_grain='day',
                                   file_date=pd.Timestamp('2026-09-16'))
        finally:
            db_writer._engine = saved
    rows, log_rows = eng.rows_for('PredictionResult'), eng.rows_for('PredictionResultLog')
    frame = pd.DataFrame({'PlanModelCode': rows['PlanModelCode'],
                          'TargetIdx': to_month_idx(rows['TargetYearMonth'].astype(int).to_numpy()),
                          'OriginIdx': int(to_month_idx(202608)),
                          CUTOFF_DATE_COL: pd.Timestamp('2026-09-15')})
    t = ground_truth(ORDERS, frame, CUTOFF_DATE_COL)
    ok_g, d_g = matches(rows['KnownGrossOrder'], t['gross'], 'KnownGross')
    ok_k, d_k = matches(rows['KnownGrossOrder'] - rows['KnownNetOrder'], t['cancelled'], 'KnownNet')
    ok_p, d_p = matches(rows['PredictedGrossOrder'] - rows['PredictedNetOrder'], t['cancelled'],
                        'PredictedNet')
    methods = sorted(rows['Method'].unique())
    check(f'{tag} every row, every method {methods}: KnownGross, KnownNet and PredictedNet all net '
          'off the cancellations known on the SAME day', ok_g and ok_k and ok_p,
          f'{d_g}; {d_k}; {d_p}')
    kr = rows[(rows['PlanModelCode'] == KNOWN_CODE)
              & (rows['TargetYearMonth'] == str(KNOWN_TARGET_YYYYMM))]
    # KNOWN1's PartCode is 'K', so it is scored by the standard models (and the manual method where
    # the plant includes it) - never by the quantile half, which only takes quantile PartCodes
    want = {f'model_{DAY}'} | ({'manual_6'} if PLANT_INCLUDES_MANUAL_METHOD[plant] else set())
    check(f'{tag} known SKU: KnownNetOrder 90 (was 170) for every method scoring it {sorted(want)}, '
          'PredictedNet = Gross - 180',
          set(kr['Method']) == want and len(kr) == len(want)
          and (kr['KnownNetOrder'] == 90).all()
          and np.allclose(kr['PredictedGrossOrder'] - kr['PredictedNetOrder'], 180),
          kr[['Method', 'KnownNetOrder']].to_dict('records').__repr__())
    check(f'{tag} one as-of point on every row: OriginDate 2026-09-15',
          (pd.to_datetime(rows['OriginDate']) == pd.Timestamp('2026-09-15')).all())
    check(f'{tag} PredictionResultLog receives the identical rows',
          len(log_rows) == len(rows) and log_rows.equals(rows))
mixed = pd.DataFrame({'OriginYYYYMM': [202608, 202608], 'CutoffDayIndex': [15, 0]})
ok, d = raises(lambda: orch._check_single_asof(mixed, 202608), ValueError, 'different as-of point')
check('rows from two as-of points reaching one DB write are refused', ok, d)

print('\n=== 4. range scoring, at the grain it is deployed at ===')
with FabricStub(ORDERS):
    for plant in PLANTS:
        tag = f'{plant} ({PLANT_NAMES[plant]})'
        res = pm.predict_range(plant, 202606, 202608, range(1, 7), DAY, cutoff_days=[0, 10])
        pairs = set(zip(res['OriginYYYYMM'], res['CutoffDayIndex']))
        check(f'{tag} day deployment, no grain passed: scored at day, every (origin, day) present',
              pairs == {(o, d) for o in (202606, 202607, 202608) for d in (0, 10)}, str(sorted(pairs)))
        o_idx = to_month_idx(res['OriginYYYYMM'].to_numpy())
        dates = np.array([cutoff_dates_for_origins(np.array([o]), int(d))[0]
                          for o, d in zip(o_idx, res['CutoffDayIndex'])])
        frame = pd.DataFrame({'PlanModelCode': res['PlanModelCode'].astype(str),
                              'TargetIdx': to_month_idx(res['TargetYYYYMM'].to_numpy()),
                              'OriginIdx': o_idx, CUTOFF_DATE_COL: dates})
        t = ground_truth(ORDERS, frame, CUTOFF_DATE_COL)
        ok_g, d_g = matches(res['KnownOrder'], t['gross'], 'range KnownOrder')
        ok_c, d_c = matches(res['KnownCanceledOrder'], t['cancelled'], 'range KnownCanceledOrder')
        check(f'{tag} ...each row known/cancelled as of ITS OWN as-of point ({len(res)} rows)',
              ok_g and ok_c, f'{d_g}; {d_c}')
        # day 30 is skipped for the May origin: Jun 30 is the Jun origin's day 0, not May's day 30
        res_e = pm.predict_range(plant, 202605, 202607, range(1, 7), DAY, cutoff_days=[30])
        pairs_e = set(zip(res_e['OriginYYYYMM'], res_e['CutoffDayIndex']))
        check(f'{tag} a cutoff day outside the origin\'s window is skipped',
              pairs_e == {(202606, 30), (202607, 30)}, str(sorted(pairs_e)))
        for label, fn, needle in [
            ('day range without cutoff_days', lambda: pm.predict_range(
                plant, 202607, 202608, range(1, 7), DAY), 'needs cutoff_days'),
            ('a cutoff day after the latest source file (Aug origin, day 25 = Sep 25 > Sep 20)',
             lambda: pm.predict_range(plant, 202608, 202608, range(1, 7), DAY, cutoff_days=[25]),
             "after the data's own as-of date"),
            ('asserting month grain on the day deployment', lambda: pm.predict_range(
                plant, 202607, 202608, range(1, 7), DAY, cutoff_grain='month'),
             'cutoff-grain mismatch'),
        ]:
            ok, d = raises(fn, ValueError, needle)
            check(f'{tag} refused: {label}', ok, d)

        tm.train_deployment(plant, MONTH, cutoff_grain='month', best_params=FAST,
                            best_params_quantile=FAST)
        res_m = pm.predict_range(plant, 202606, 202608, range(1, 7), MONTH)
        t = ground_truth(ORDERS, pd.DataFrame({
            'PlanModelCode': res_m['PlanModelCode'].astype(str),
            'TargetIdx': to_month_idx(res_m['TargetYYYYMM'].to_numpy()),
            'OriginIdx': to_month_idx(res_m['OriginYYYYMM'].to_numpy())}))
        check(f'{tag} month deployment: month rows (day index 0 only), cancellations as of month end',
              set(res_m['CutoffDayIndex']) == {0} and np.allclose(res_m['KnownCanceledOrder'],
                                                                   t['cancelled']))
        for label, fn, needle in [
            ('cutoff_days on a month deployment', lambda: pm.predict_range(
                plant, 202607, 202608, range(1, 7), MONTH, cutoff_days=[0]), 'day grain only'),
            ('a month origin that has not closed (Sep 2026: month end after Sep 20)',
             lambda: pm.predict_range(plant, 202609, 202609, range(1, 7), MONTH),
             "after the data's own as-of date"),
        ]:
            ok, d = raises(fn, ValueError, needle)
            check(f'{tag} refused: {label}', ok, d)
        man = pm.predict_manual_range(plant, 202607, 202608, range(1, 7), cutoff_grain='day',
                                      cutoff_days=[0, 15])
        check(f'{tag} manual range at day grain scores every (origin, day)',
              set(zip(man['OriginYYYYMM'], man['CutoffDayIndex']))
              == {(o, d) for o in (202607, 202608) for d in (0, 15)})

        out = TMP / f'range_{plant}.parquet'
        pm.merge_prediction_column(res, str(out))
        again = pm.merge_prediction_column(res, str(out))
        check(f'{tag} upsert: re-running the same range replaces rows, and two cutoff days of one '
              'origin are two rows, not a collision', len(again) == len(res))
        old = TMP / f'old_{plant}.parquet'
        res_m.drop(columns=['CutoffDayIndex', 'KnownCanceledOrder']).to_parquet(old, index=False)
        merged = pm.merge_prediction_column(res_m, str(old))
        check(f'{tag} a parquet from before CutoffDayIndex is read as day 0 and upserted in place',
              len(merged) == len(res_m) and set(merged['CutoffDayIndex']) == {0})

print('\n=== 5. a scoring half with no rows no longer crashes the run ===')
no_quantile_parts = make_orders(part_map={'Z': 'Z'})          # PartCodes A/B/C/K: no SL or RD
with FabricStub(no_quantile_parts):
    for plant in PLANTS:
        tag = f'{plant} ({PLANT_NAMES[plant]})'
        tm.train_deployment(plant, NOQ, cutoff_grain='month', best_params=FAST,
                            best_params_quantile=FAST)
        for label, parts, want in [('none of quantile_partcodes present', ('SL', 'RD'),
                                    {f'model_{NOQ}'}),
                                   ('every row routed to the quantile half', ('A', 'B', 'C', 'K'),
                                    {f'model_{NOQ}_quantile'})]:
            try:
                r = pm.predict(plant, label=NOQ, quantile_partcodes=parts)
                check(f'{tag} {label}: scores cleanly', set(r['Method']) == want and len(r) > 0,
                      str(set(r['Method'])))
            except Exception as e:
                check(f'{tag} {label}: scores cleanly', False, f'{type(e).__name__}: {e}')

print('\n=== 6. no month default and no post-hoc netting left on these paths ===')
for fn in (pm.predict_manual, pm.predict_manual_range, pp.preprocess_for_inference,
           pp.preprocess_for_inference_range, pp.preprocess_for_training,
           pp.build_order_book_context, mr.save_model):
    prm = inspect.signature(fn).parameters['cutoff_grain']
    check(f'{fn.__module__}.{fn.__name__}: cutoff_grain keyword-only and required',
          prm.kind is inspect.Parameter.KEYWORD_ONLY and prm.default is inspect.Parameter.empty)
main_src = (_REPO_ROOT / 'main.py').read_text(encoding='utf-8')
check('compute_known_canceled_order (month-only, post-scoring) is gone',
      not hasattr(ob, 'compute_known_canceled_order') and 'compute_known_canceled_order' not in main_src)
check("main.py never assigns KnownCanceledOrder itself - it arrives on the skeleton rows",
      "['KnownCanceledOrder'] =" not in main_src)
check('the file date the stubs used is the data as-of date',
      all(c['data_asof'] == DATA_THROUGH for (_s, c) in SK.values()))

print('\n' + '=' * 78)
print(f'{len(FAILURES)} FAILURE(S): {FAILURES}' if FAILURES else 'ALL CHECKS PASSED')
print(f'(temp dir: {TMP})')
sys.exit(1 if FAILURES else 0)
