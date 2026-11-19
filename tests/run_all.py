"""Runs every verification suite in this directory and reports a single pass/fail.

    python tests/run_all.py            # all suites
    python tests/run_all.py -v         # plus each suite's own per-check output

Exits non-zero if any suite fails, so it works as a pre-commit / pre-refactor gate.
"""
import pathlib
import subprocess
import sys

# Suite output can hold any character; the Windows console code page can't print all of them.
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SUITES = [
    ('verify_order_book', 'order-book grain plumbing: month-end equivalence, grain guard'),
    ('verify_grain_contract', 'registry cutoff_grain: the silent month-model-on-day-rows failure'),
    ('verify_known_order', 'attach_known_order_features: month path vs the pre-edit function'),
    ('verify_ctx_routing', 'ctx dual books: which families move intra-month, and which must not'),
    ('verify_day_skeleton', 'day-grain skeletons: day-0 anchor, de-duplication, inference dates'),
    ('verify_folds', 'lazy folds: identical to the eager list, re-iterable, O(1) retention'),
    ('verify_intramonth', 'intra-month family: absent at month grain, two day-0 identities'),
    ('verify_daily_cadence', 'orchestrator: retrain monthly, rescore daily, PendingOrder always'),
    ('verify_grain_lifecycle', 'grain lifecycle: a day deployment stays day across retrains'),
    ('verify_net_orders', 'net orders: cancellations netted at the same as-of point as orders'),
    ('verify_atomic_deploy', 'deployments: one panel build, atomic commit, no split halves'),
    ('verify_no_lookahead', 'no look-ahead: a training row equals a live run on its as-of date'),
    ('verify_truncation_invariance', 'no leakage: every feature unchanged when later orders are cut off'),
    ('verify_cny_calendar', 'CNY calendar: defined for every year, never negative (audit B1)'),
    ('verify_input_coverage', 'input guards: bad dates, TFG key matching (audit B2-B3)'),
    ('verify_forecast_eligibility', 'eligibility: production and every backtest score the same rows (B4)'),
    ('verify_run_resilience', 'daily run: rerun-safe atomic writes, survives failures (B5-B8, B13, B18-B19)'),
    ('verify_deployment_refit', 'deployed model refitted on the latest settled months (B10, B17)'),
    ('verify_log_retention', 'daily run prunes old log files and own AIAPPLOGS rows, never raises'),
]

HERE = pathlib.Path(__file__).resolve().parent
VERBOSE = '-v' in sys.argv or '--verbose' in sys.argv

# Each suite runs with tests/no_db.py imported FIRST, so every database engine is already a raising
# stub before the suite's own imports - see no_db.py. A suite that doesn't also import it itself
# is refused, so running it directly can't bypass the guard either.
_GUARDED = ("import runpy, sys; sys.path.insert(0, {here!r}); import no_db; "
            "assert no_db.installed(); sys.argv = [{path!r}]; "
            "runpy.run_path({path!r}, run_name='__main__')")

failed = []
for name, blurb in SUITES:
    path = str(HERE / f'{name}.py')
    if 'import no_db' not in (HERE / f'{name}.py').read_text(encoding='utf-8'):
        print(f"FAIL  {name:24s} does not import no_db - refusing to run it unguarded")
        failed.append(name)
        continue
    proc = subprocess.run([sys.executable, '-X', 'utf8', '-c',
                           _GUARDED.format(here=str(HERE), path=path)],
                          capture_output=True, text=True, encoding='utf-8', errors='replace')
    tail = [ln for ln in (proc.stdout or '').splitlines()
            if 'ALL CHECKS PASSED' in ln or 'FAILURE(S)' in ln]
    verdict = tail[-1] if tail else f'NO RESULT (exit {proc.returncode})'
    print(f"{'ok  ' if proc.returncode == 0 else 'FAIL'}  {name:24s} {verdict}")
    print(f"      {blurb}")
    if VERBOSE:
        print('\n'.join('      | ' + ln for ln in (proc.stdout or '').splitlines()))
    if proc.returncode != 0:
        failed.append(name)
        if not VERBOSE:   # a failure always shows its detail, verbose or not
            print('\n'.join('      | ' + ln for ln in (proc.stdout or '').splitlines()[-40:]))
            if proc.stderr:
                print('\n'.join('      ! ' + ln for ln in proc.stderr.splitlines()[-15:]))

print('\n' + '=' * 70)
print(f'{len(failed)} of {len(SUITES)} suite(s) FAILED: {failed}' if failed
      else f'all {len(SUITES)} suites passed')
sys.exit(1 if failed else 0)
