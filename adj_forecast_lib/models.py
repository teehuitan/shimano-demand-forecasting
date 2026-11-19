import logging

log = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    'xgboost': dict(max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        objective='reg:squarederror', eval_metric='rmse', tree_method='hist',
        enable_categorical=True, random_state=42),
    'lightgbm': dict(objective='l2', metric='l2', learning_rate=0.05, num_leaves=63,
        min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        lambda_l2=1.0, verbosity=-1, seed=42),
    'catboost': dict(loss_function='RMSE', depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
        random_seed=42),
}


TWEEDIE_POWERS = [1.1, 1.3, 1.5, 1.7, 1.9]


def xgb_objective_specs():
    """(label, param overrides) pairs for sweeping XGBoost's objective/eval_metric."""
    specs = [
        ('obj = RMSE', {'objective': 'reg:squarederror', 'eval_metric': 'rmse'}),
        ('obj = pseudohuber', {'objective': 'reg:pseudohubererror', 'eval_metric': 'mae'}),
        ('obj = MAE', {'objective': 'reg:absoluteerror', 'eval_metric': 'mae'}),
    ]
    for p in TWEEDIE_POWERS:
        specs.append((f'tweedie power = {p}',
            {'objective': 'reg:tweedie', 'tweedie_variance_power': p,
             'eval_metric': f'tweedie-nloglik@{p}'}))
        specs.append((f'tweedie power = {p}, eval=MAE',
            {'objective': 'reg:tweedie', 'tweedie_variance_power': p, 'eval_metric': 'mae'}))
    return specs


def cat_objective_specs():
    """CatBoost counterpart to xgb_objective_specs() - same TWEEDIE_POWERS catalog, expressed as
    loss_function/eval_metric overrides."""
    specs = [
        ('obj = RMSE', {'loss_function': 'RMSE'}),
        ('obj = MAE', {'loss_function': 'MAE'}),
    ]
    for p in TWEEDIE_POWERS:
        specs.append((f'tweedie power = {p}', {'loss_function': f'Tweedie:variance_power={p}'}))
        specs.append((f'tweedie power = {p}, eval=MAE',
            {'loss_function': f'Tweedie:variance_power={p}', 'eval_metric': 'MAE'}))
    return specs


def _split_xy(train, val, test, features, target_col, weight_fn):
    X_tr, y_tr = train[features], train[target_col]
    X_va, y_va = val[features], val[target_col]
    w_tr = weight_fn(train) if weight_fn else None
    w_va = weight_fn(val) if weight_fn else None
    return X_tr, y_tr, X_va, y_va, test[features], w_tr, w_va


def fit_predict_xgboost(train, val, test, features, cats, params, n_estimators,
        early_stopping_rounds, target_col='Adjustment', weight_fn=None, verbose=None):
    from xgboost import XGBRegressor
    # `cats` is unused: XGBoost has no cat_features argument, it reads the pandas 'category'
    # dtype the panel is already cast to. Kept in the signature so every adapter is callable
    # identically via ADAPTERS.
    # verbose: None/False silent, an int N prints every N rounds - normalized to False (not left
    # as None) 
    verbose = verbose or False
    X_tr, y_tr, X_va, y_va, X_te, w_tr, w_va = _split_xy(
        train, val, test, features, target_col, weight_fn)

    model = XGBRegressor(n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds, **params)
    # Early stopping reads only the LAST eval_set entry, which is always val. The training set is
    # put in front of it only when verbose, to print both curves: evaluating it means a second
    # DMatrix of the whole training set and a pass over it every round - roughly double the fit's
    # memory at day grain - for numbers nobody reads when silent. The fitted model is identical.
    on_train = [(X_tr, y_tr)] if verbose else []
    fit_kwargs = dict(eval_set=on_train + [(X_va, y_va)], verbose=verbose)
    if w_tr is not None:
        fit_kwargs.update(sample_weight=w_tr,
                          sample_weight_eval_set=([w_tr] if verbose else []) + [w_va])
    model.fit(X_tr, y_tr, **fit_kwargs)
    # Already computed by fit() above for early stopping - just reading it back. val is always the
    # LAST eval_set entry; the train curve exists only when verbose put train in front of it.
    raw = model.evals_result()
    eval_history = ({'train': raw.get('validation_0', {}), 'val': raw.get('validation_1', {})} if verbose
                    else {'train': {}, 'val': raw.get('validation_0', {})})
    # best_iteration is 0-based; n_rounds is the tree count the early-stopped model predicts with.
    return model.predict(X_te), {'best_iteration': model.best_iteration,
                                 'n_rounds': model.best_iteration + 1, 'model': model,
                                 'eval_history': eval_history}


def fit_predict_lightgbm(train, val, test, features, cats, params, n_estimators,
        early_stopping_rounds, target_col='Adjustment', weight_fn=None, verbose=None):
    import lightgbm as lgb
    verbose = verbose or False  # see fit_predict_xgboost's comment
    X_tr, y_tr, X_va, y_va, X_te, w_tr, w_va = _split_xy(
        train, val, test, features, target_col, weight_fn)

    use_cats = [c for c in cats if c in features]
    ds_tr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr,
        categorical_feature=use_cats, free_raw_data=False)
    ds_va = lgb.Dataset(X_va, label=y_va, weight=w_va, reference=ds_tr,
        categorical_feature=use_cats, free_raw_data=False)
    raw = {}  # populated in place by record_evaluation, same shape as xgboost/catboost's own history
    callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=bool(verbose)), lgb.record_evaluation(raw)]
    if verbose:
        callbacks.append(lgb.log_evaluation(period=verbose))
    model = lgb.train(params, ds_tr, num_boost_round=n_estimators, valid_sets=[ds_tr, ds_va],
        valid_names=['train', 'val'], callbacks=callbacks)
    eval_history = {'train': raw.get('train', {}), 'val': raw.get('val', {})}
    # .predict() defaults to best_iteration when early stopping fired. LightGBM's best_iteration
    # is already a round COUNT (1-based), and 0 if early stopping never ran.
    return model.predict(X_te), {'best_iteration': model.best_iteration,
                                 'n_rounds': model.best_iteration or model.current_iteration(),
                                 'model': model, 'eval_history': eval_history}


def fit_predict_catboost(train, val, test, features, cats, params, n_estimators,
        early_stopping_rounds, target_col='Adjustment', weight_fn=None, verbose=None):
    from catboost import CatBoostRegressor, Pool
    verbose = verbose or False  # see fit_predict_xgboost's comment
    use_cats = [c for c in cats if c in features]
    train, val, test = (_catboost_cats(d, use_cats) for d in (train, val, test))
    X_tr, y_tr, X_va, y_va, X_te, w_tr, w_va = _split_xy(
        train, val, test, features, target_col, weight_fn)

    pool_tr = Pool(X_tr, y_tr, cat_features=use_cats, weight=w_tr)
    pool_va = Pool(X_va, y_va, cat_features=use_cats, weight=w_va)
    model = CatBoostRegressor(iterations=n_estimators,
        early_stopping_rounds=early_stopping_rounds, verbose=verbose, **params)
    # CatBoost's own verbose logging already prints both "learn" (train) and eval_set (val)
    # metrics per round
    model.fit(pool_tr, eval_set=pool_va, use_best_model=True)
    raw = model.get_evals_result()
    eval_history = {'train': raw.get('learn', {}), 'val': raw.get('validation', {})}
    # get_best_iteration is 0-based; use_best_model=True has already cut the model to n_rounds.
    return (model.predict(Pool(X_te, cat_features=use_cats)),
            {'best_iteration': model.get_best_iteration(),
             'n_rounds': model.get_best_iteration() + 1, 'model': model, 'eval_history': eval_history})


def _catboost_cats(d, use_cats):
    """Casts categorical columns to str for CatBoost, with missing as its own level."""
    return d.assign(**{c: d[c].astype('object').fillna('__NA__').astype(str) for c in use_cats})


ADAPTERS = {
    'xgboost': fit_predict_xgboost,
    'lightgbm': fit_predict_lightgbm,
    'catboost': fit_predict_catboost,
}


# Refitters: the second half of a deployment fit (train.fit_horizon_models). An ADAPTERS call
# early-stops on a held-out validation tail, which picks the number of boosting rounds; the
# refitter then trains on EVERY row it is given - train and validation together - for exactly
# that many rounds, with no early stopping and no eval set. Same signature across families, and
# the same weighting and categorical handling as the matching adapter.

def refit_xgboost(data, features, cats, params, n_rounds, target_col='Adjustment',
        weight_fn=None):
    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=n_rounds, **params)
    model.fit(data[features], data[target_col],
              sample_weight=weight_fn(data) if weight_fn else None)
    return model


def refit_lightgbm(data, features, cats, params, n_rounds, target_col='Adjustment',
        weight_fn=None):
    import lightgbm as lgb
    use_cats = [c for c in cats if c in features]
    ds = lgb.Dataset(data[features], label=data[target_col],
        weight=weight_fn(data) if weight_fn else None, categorical_feature=use_cats)
    return lgb.train(params, ds, num_boost_round=n_rounds)


def refit_catboost(data, features, cats, params, n_rounds, target_col='Adjustment',
        weight_fn=None):
    from catboost import CatBoostRegressor, Pool
    use_cats = [c for c in cats if c in features]
    data = _catboost_cats(data, use_cats)
    model = CatBoostRegressor(iterations=n_rounds, verbose=False, **params)
    model.fit(Pool(data[features], data[target_col], cat_features=use_cats,
                   weight=weight_fn(data) if weight_fn else None))
    return model


REFITTERS = {
    'xgboost': refit_xgboost,
    'lightgbm': refit_lightgbm,
    'catboost': refit_catboost,
}


def predict_xgboost(model, X):
    return model.predict(_unseen_categories_to_nan(model, X))


def _unseen_categories_to_nan(model, X):
    """Sets category values the model was not trained on to NaN, so XGBoost scores them as missing
    instead of raising and failing the whole plant's forecast. CatBoost (h1) needs no such step."""
    known = model.get_booster().get_categories(export_to_arrow=True).to_arrow()
    X = X.copy()
    for col, cats in known:
        if cats is None or col not in X.columns:
            continue
        # XGBoost checks the column's category list, not just the row values - remove them from it.
        unknown = X[col].cat.categories.difference(cats.to_pylist())
        if len(unknown) == 0:
            continue
        unseen = X[col].isin(unknown)
        if unseen.any():
            log.warning("%d row(s) have a %s the model was not trained on (first ordered or re-coded "
                "after the last retrain) - predicted with %s treated as unknown: %s",
                int(unseen.sum()), col, col, sorted(X.loc[unseen, col].astype(str).unique()))
        X[col] = X[col].cat.remove_categories(unknown)
    return X


def predict_lightgbm(model, X):
    return model.predict(X)


def predict_catboost(model, X):
    """Predicts with CatBoost using the same categorical encoding as at fit time."""
    from catboost import Pool
    use_cats = [c for c in X.columns if str(X[c].dtype) == 'category']
    X = X.copy()
    for c in use_cats:
        X[c] = X[c].astype('object').fillna('__NA__').astype(str)
    return model.predict(Pool(X, cat_features=use_cats))


PREDICT_FUNCS = {
    'xgboost': predict_xgboost,
    'lightgbm': predict_lightgbm,
    'catboost': predict_catboost,
}
