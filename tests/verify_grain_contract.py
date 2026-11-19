"""Edit-2 verification: the cutoff-grain contract (hazard #2).

Closes the SILENT half of the grain mismatch - a month-trained model scored on day-grain rows.
feature_cols already covers the loud half (day-trained model + month rows -> KeyError on the
missing day-only columns), so this checks the half that previously sailed through.

Writes into the throwaway MLflow store tests/no_db.py sets up; never the real tracking server.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / 'tests'))
import no_db  # noqa: E402,F401 - every DB engine raises from here on (see no_db.py)

from adj_forecast_lib.config import CUTOFF_DATE_COL, CUTOFF_GRAINS  # noqa: E402
from adj_forecast_lib.model_registry import (  # noqa: E402
    _client, check_cutoff_grain, entry_cutoff_grain, load_model, save_model,
)
from adj_forecast_lib.order_book import frame_cutoff_grain

FAILURES = []


def check(label, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' - ' + detail) if detail and not cond else ''}")
    if not cond:
        FAILURES.append(label)


def raises(fn, needle):
    try:
        fn()
    except Exception as e:
        return needle in str(e), f'{type(e).__name__}: {str(e)[:120]}'
    return False, 'no exception raised'


print('1. frame_cutoff_grain reads the frame, absence == month')
month_df = pd.DataFrame({'OriginIdx': [24320, 24321], 'Horizon': [1, 2]})
day_df = month_df.assign(**{CUTOFF_DATE_COL: pd.to_datetime(['2026-08-14', '2026-08-14'])})
check('no cutoff column -> month', frame_cutoff_grain(month_df) == 'month')
check('datetime cutoff column -> day', frame_cutoff_grain(day_df) == 'day')
# A non-datetime column of that name is NOT day grain - guards against a stray string column
# silently flipping a whole pipeline to the day path.
bad_df = month_df.assign(**{CUTOFF_DATE_COL: ['2026-08-14', '2026-08-14']})
check('string column of the same name -> still month', frame_cutoff_grain(bad_df) == 'month')
check('empty frame with the column -> day', frame_cutoff_grain(day_df.iloc[:0]) == 'day')

print('\n2. save_model records the grain, and validates it')
from xgboost import XGBRegressor
X = pd.DataFrame({'a': np.arange(20.0), 'b': np.arange(20.0)[::-1]})
y = np.arange(20.0)
model = XGBRegressor(n_estimators=2, max_depth=2).fit(X, y)

common = dict(feature_cols=['a', 'b'], params={'max_depth': 2},
              covid_weight=0.1, young_sku_weight=1.0, external_event_weight=1.0,
              trained_through=24320, model_type='xgboost', plant=3300)

e_month = save_model(model, horizon=1, label='t_month', cutoff_grain='month', **common)
e_day = save_model(model, horizon=1, label='t_day', cutoff_grain='day', **common)
check("stored cutoff_grain='month'", e_month['cutoff_grain'] == 'month')
check("stored cutoff_grain='day'", e_day['cutoff_grain'] == 'day')

ok, detail = raises(lambda: save_model(model, horizon=2, label='t_default', **common),
                    'cutoff_grain')
check('no default grain: omitting it raises instead of silently recording month', ok, detail)

ok, detail = raises(lambda: save_model(model, horizon=3, label='t_bad',
                                       cutoff_grain='daily', **common), 'cutoff_grain must be one of')
check('a typo raises rather than being stored', ok, detail)
check('CUTOFF_GRAINS is the single source of truth', tuple(CUTOFF_GRAINS) == ('month', 'day'))

print('\n3. round-trip through load_model')
_m, _fc, entry_rt = load_model(plant=3300, horizon=1, label='t_day')
check('grain survives save -> MLflow version tag -> load', entry_cutoff_grain(entry_rt) == 'day')

print('\n4. backward compatibility with versions registered before the tag existed')
_client().delete_model_version_tag(e_month['registered_name'], e_month['version'], 'cutoff_grain')
_m, _fc, entry_legacy = load_model(plant=3300, horizon=1, label='t_month')
check('legacy version loads at all, and reads as month', entry_legacy['cutoff_grain'] == 'month')
check("legacy entry reads as 'month'", entry_cutoff_grain(entry_legacy) == 'month')
ok, detail = raises(lambda: check_cutoff_grain(entry_legacy, 'day'), 'cutoff-grain mismatch')
check('legacy entry still guarded against day-grain rows', ok, detail)

print('\n5. check_cutoff_grain - the actual guard')
try:
    check_cutoff_grain(e_month, 'month')
    check_cutoff_grain(e_day, 'day')
    check('matching grains pass', True)
except Exception as e:
    check('matching grains pass', False, str(e))

ok, detail = raises(lambda: check_cutoff_grain(e_month, 'day'), 'cutoff-grain mismatch')
check('THE SILENT ONE: month-trained model vs day rows now raises', ok, detail)
ok, detail = raises(lambda: check_cutoff_grain(e_day, 'month'), 'cutoff-grain mismatch')
check('day-trained model vs month rows raises', ok, detail)

msg = raises(lambda: check_cutoff_grain(e_month, 'day'), '')[1]
check('error names the label and both grains',
      "'t_month'" in msg and "'month'" in msg and "'day'" in msg, msg)

print('\n6. today is a no-op - current panels/skeletons are month grain')
real_panel = next(iter(sorted((_REPO_ROOT / 'data' / 'STJ').glob('train7_*[0-9].parquet'))), None)
if real_panel is not None:
    import pyarrow.parquet as pq
    cols = pq.ParquetFile(real_panel).schema.names
    stub = pd.DataFrame({c: [] for c in cols})
    check('the real STJ panel reads as month grain', frame_cutoff_grain(stub) == 'month')
else:
    print('  SKIP  real panel not found')

print('\n7. module graph - no circular import from the new config dependency')
try:
    import adj_forecast_lib.train      # noqa: F401
    import adj_forecast_lib.predict    # noqa: F401
    check('train + predict import (they build DB engines at import)', True)
except Exception as e:
    check('train + predict import', False, f'{type(e).__name__}: {e}')

print('\n' + '=' * 62)
print(f"{len(FAILURES)} FAILURE(S): {FAILURES}" if FAILURES else 'ALL CHECKS PASSED')
sys.exit(1 if FAILURES else 0)
