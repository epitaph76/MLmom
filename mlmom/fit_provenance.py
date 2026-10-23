"""Worker-owned snapshots taken at backend fit entry, before candidate mutation."""
from __future__ import annotations

import contextvars
import copy
import functools
import hashlib
import inspect
import json
import threading
from contextlib import contextmanager

_active = contextvars.ContextVar('mlmom_fit_capture', default=None)
_lock = threading.RLock()
_installed = False
_records = {}
_fallback_scope = None


def records_for(context):
    return _records.setdefault(context.fit_id, {})


def fingerprint(values, *, labels=False):
    if values is None:
        return None
    import numpy as np
    import pandas as pd
    data = np.asarray(values)
    if labels and data.ndim == 1:
        data = pd.factorize(data, sort=False)[0]
    elif np.issubdtype(data.dtype, np.number):
        data = data.astype('float64')
    data = np.ascontiguousarray(data)
    if data.dtype.hasobject or data.dtype.kind in {'U', 'S'}:
        payload = json.dumps(data.tolist(), ensure_ascii=False, default=str).encode()
    else:
        payload = data.tobytes()
    return hashlib.sha256(str(data.shape).encode() + payload).hexdigest()


def serializable_params(model):
    try:
        values = model.get_params(deep=True)
    except (AttributeError, TypeError):
        values = getattr(model, '_init_params', {})
    clean = {}
    for key, value in values.items():
        try:
            clean[key] = json.loads(json.dumps(value, allow_nan=False))
        except (ValueError, TypeError):
            pass
    return clean


def _patch(cls, method='fit'):
    original = cls.__dict__.get(method)
    if original is None or not callable(original) or getattr(original, '_mlmom_capture', False):
        return
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        return

    @functools.wraps(original)
    def captured(self, *args, **kwargs):
        scope = _active.get() or _fallback_scope
        if scope is None:
            return original(self, *args, **kwargs)
        try:
            bound = signature.bind(self, *args, **kwargs).arguments
        except TypeError:
            return original(self, *args, **kwargs)
        y = bound.get('y', bound.get('Y', bound.get('endog',bound.get('y_train'))))
        if y is None and type(self).__module__.startswith('statsmodels'):
            y = getattr(self, 'endog', None)
        X = bound.get('X', bound.get('data',bound.get('X_train',bound.get('train_data'))))
        if y is None and hasattr(X,'get_label'):
            y=X.get_label()
        if y is None and hasattr(X,'columns'):
            label=getattr(self,'label',None) or (bound.get('roles') or {}).get('target')
            if isinstance(label,str) and label in X:
                y=X[label]
        weights = bound.get('sample_weight', bound.get('weight'))
        if weights is None and type(self).__module__.startswith('statsmodels'):
            weights = getattr(self, 'weights', None)
        if weights is None:
            extras=bound.get('fit_kwargs',bound.get('kwargs',{}))
            if isinstance(extras,dict):weights=extras.get('sample_weight')
        if weights is not None and scope['weights'] is None:
            import numpy as np
            if np.all(np.asarray(weights)==1):weights=None
        record = dict(backend=type(self).__module__.split('.')[0],
            component_type=type(self).__module__ + '.' + type(self).__name__,
            object_id=id(self), params=serializable_params(self),
            rows=len(y) if y is not None else (X.shape[0] if hasattr(X, 'shape') else None),
            target_fingerprint=fingerprint(y, labels=scope['classification']),
            weight_fingerprint=fingerprint(weights),
            expected_target_fingerprint=scope['target'], expected_weight_fingerprint=scope['weights'],
            target_columns=scope['target_columns'],
            stage=scope['stage'], code_hash=scope['config'].get('_code_hash'),
            preprocessing_hash=scope['config'].get('_preprocessing_hash'),
            model_spec=scope['spec'], completed=False)
        if scope.get('native_parent') is not None:
            record['native_parent_entry']=scope['native_parent']
        # Keep the first (outermost) entry: super().fit must not overwrite it.
        scope['records'].setdefault(id(self), record)
        previous_parent=scope.get('native_parent')
        is_native=type(self).__module__.split('.')[0] in {'flaml','autogluon','lightautoml'}
        if is_native:
            scope['native_parent']=scope['records'][id(self)]
        try:
            result = original(self, *args, **kwargs)
        finally:
            if is_native:scope['native_parent']=previous_parent
        scope['records'][id(self)]['completed'] = True
        if result is not None and result is not self:
            scope['records'][id(result)] = copy.deepcopy(scope['records'][id(self)])
        return result
    captured._mlmom_capture = True
    setattr(cls, method, captured)


def install():
    """Install before candidate imports; future sklearn subclasses are covered too."""
    global _installed
    with _lock:
        if _installed:
            return
        from sklearn.base import BaseEstimator
        import sklearn.ensemble, sklearn.linear_model, sklearn.tree, sklearn.svm
        import sklearn.neighbors, sklearn.naive_bayes, sklearn.cluster, sklearn.multioutput, sklearn.pipeline
        seen = set()
        def visit(cls):
            if cls in seen:
                return
            seen.add(cls)
            if cls.__module__.startswith('sklearn'):
                _patch(cls)
            for child in cls.__subclasses__():
                visit(child)
        visit(BaseEstimator)
        # Backend imports are bounded and never install/download packages.
        for package, names in [('lightgbm', ['LGBMModel', 'LGBMRegressor', 'LGBMClassifier', 'LGBMRanker']),
                               ('xgboost', ['XGBModel', 'XGBRegressor', 'XGBClassifier', 'XGBRanker']),
                               ('catboost', ['CatBoost', 'CatBoostRegressor', 'CatBoostClassifier', 'CatBoostRanker'])]:
            try:
                module = __import__(package)
                for name in names:
                    _patch(getattr(module, name))
            except ImportError:
                continue
        _installed = True
        # These packages are imported lazily by candidates; intercept only their public fit classes.
        import builtins, sys
        original_import=builtins.__import__
        watched={'statsmodels.regression.linear_model':('OLS','WLS'),
                 'flaml.automl.automl':('AutoML',),
                 'autogluon.tabular.predictor.predictor':('TabularPredictor',),
                 'lightautoml.automl.base':('AutoML',),
                 'lightautoml.automl.presets.tabular_presets':('TabularAutoML',)}
        state=threading.local()
        def patch_loaded():
            for module_name,names in watched.items():
                module=sys.modules.get(module_name)
                if module is not None:
                    for name in names:
                        cls=getattr(module,name,None)
                        if cls is not None:
                            for base in cls.__mro__:
                                if base.__module__.startswith(module_name.split('.')[0]):
                                    _patch(base);_patch(base,'fit_predict');_patch(base,'retrain_from_log')
        def importing(name,*args,**kwargs):
            result=original_import(name,*args,**kwargs)
            if not getattr(state,'busy',False) and name.split('.')[0] in {'statsmodels','flaml','autogluon','lightautoml'}:
                state.busy=True
                try:patch_loaded()
                finally:state.busy=False
            return result
        patch_loaded()
        builtins.__import__=importing


@contextmanager
def capture(context, config, y, weights):
    global _fallback_scope
    from .model_spec import model_spec
    install()
    classification = 'classification' in context.task_type
    import numpy as np
    columns=[fingerprint(np.asarray(y)[:,i],labels=classification) for i in range(np.asarray(y).shape[1])] if y is not None and np.asarray(y).ndim==2 else None
    scope = dict(records=records_for(context), classification=classification,target_columns=columns,
        target=fingerprint(y, labels=classification), weights=fingerprint(weights),
        config=config, spec=model_spec(config), stage=getattr(context, 'stage', 'fit'))
    token = _active.set(scope)
    previous=_fallback_scope
    _fallback_scope=scope  # sklearn's trusted thread-based component dispatch inherits this scope.
    try:
        yield
    finally:
        _active.reset(token)
        _fallback_scope=previous


def record_for(component, traces):
    record = traces.get(id(component))
    if record is None and hasattr(component, 'steps'):
        record = traces.get(id(component.steps[-1][1]))
    return copy.deepcopy(record) if record else None
