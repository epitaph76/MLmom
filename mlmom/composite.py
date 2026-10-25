"""Trusted composition of frozen generated candidates, with raw fold-local inputs."""
from __future__ import annotations

import copy
import time
import uuid
from dataclasses import replace
from pathlib import Path
import numpy as np


class FrozenEnsemble:
    """SDK implementation used by a small generated composition entry point.

    Components retain their own preprocessing and runtime policies. No fitted
    component, prediction cache or validation metadata is borrowed from history.
    """
    def __init__(self, config, component_root):
        self.config = copy.deepcopy(config)
        self.component_root = Path(component_root)
        self.sources = self.config['ensemble_sources']
        self.params = copy.deepcopy(config['model_params'])
        self.models = {}
        self.meta = None
        self.fold_evidence = []
        if not 2 <= len(self.sources) <= 3 or len({s['name'] for s in self.sources}) != len(self.sources):
            raise ValueError('FrozenEnsemble requires two or three distinct named components')

    def _base(self, source):
        from .sdk import load_module
        from .persistence import tree_hash
        directory = self.component_root / source['name']
        if not directory.resolve().is_relative_to(self.component_root.resolve()):
            raise ValueError('Component path escapes bundled source')
        if tree_hash(directory) != source['code_hash']:
            raise ValueError('Bundled component source hash mismatch')
        config = copy.deepcopy(source['config'])
        if config.get('ensemble_components') or config.get('ensemble_sources'):
            raise ValueError('Recursive ensembles are not admitted')
        config['model_params'] = copy.deepcopy(self.params[source['name']])
        config['_code_hash'] = source['code_hash']
        return load_module(directory), config

    def _fit_base(self, source, X, y, context, *, indices=None, tag='full', final=False):
        from .execution import export_params
        from .execution_resources import execution_config
        module, config = self._base(source)
        model = module.create_candidate(execution_config(config, context.cpus))
        take = lambda value: value if indices is None or value is None else np.asarray(value)[indices]
        c = replace(context, groups=take(context.groups), sample_weight=take(context.sample_weight),
            validation_X=None, validation_y=None, validation_groups=None, validation_weight=None,
            runtime_policies_applied=False, fit_rows=len(X),
            fit_id=uuid.uuid4().hex, parent_fit_id=context.fit_id, component_name=source['name'],
            started=time.monotonic(), time_limit_seconds=max(.1, context.remaining_seconds),
            work_dir=str(Path(context.work_dir)/source['name']/tag))
        Path(c.work_dir).mkdir(parents=True, exist_ok=True)
        model.fit(X, y, c)
        from .execution_facts import record_fit
        record_fit(model, config, c)
        if final:
            self.params[source['name']] = export_params(model, config['model_params'])
        return model

    def _prediction(self, model, X):
        from .evaluation import checked_predictions
        value = np.asarray(model.predict(X, {'kind': 'probability' if self.classification else 'value'}))
        if self.classification and not self.multilabel:
            if value.ndim == 1 or value.shape[1] == 1:
                if len(self.classes) != 2:
                    raise ValueError('A multiclass component must return all probability columns')
                value = value.reshape(-1)
                value = np.column_stack([1-value, value])
            return checked_predictions(value, len(X), 'probability', classes=self.classes)
        return checked_predictions(value, len(X), 'probability' if self.classification else 'value')

    def _folds(self, X, y, context):
        from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
        count = int(self.params.get('oof_folds', 3))
        if count < 2 or len(X) < count:
            raise ValueError('Insufficient rows for ensemble OOF folds')
        single = self.classification and not self.multilabel
        groups = context.groups
        if groups is not None:
            if len(np.unique(groups)) < count:
                raise ValueError('Insufficient independent groups for ensemble OOF')
            splitter = StratifiedGroupKFold(count, shuffle=True, random_state=context.seed) if single else GroupKFold(count)
            folds = splitter.split(X, y, groups)
        else:
            splitter = StratifiedKFold(count, shuffle=True, random_state=context.seed) if single else KFold(count, shuffle=True, random_state=context.seed)
            folds = splitter.split(X, y)
        result = []
        for a, b in folds:
            if not len(a) or not len(b) or np.intersect1d(a,b).size:
                raise ValueError('Invalid ensemble OOF split')
            if groups is not None and set(np.asarray(groups)[a]) & set(np.asarray(groups)[b]):
                raise ValueError('Group leakage in ensemble OOF')
            if single and (set(np.asarray(y)[a]) != set(self.classes) or set(np.asarray(y)[b]) != set(self.classes)):
                raise ValueError('Every ensemble OOF fold must contain every class')
            if self.multilabel and any(len(np.unique(np.asarray(y)[a, j])) < 2 for j in range(np.asarray(y).shape[1])):
                raise ValueError('Every multilabel OOF training fold needs both classes')
            result.append((a,b))
        return result

    def _labels(self, probabilities):
        return (probabilities >= .5).astype(int) if self.multilabel else np.asarray(self.classes)[probabilities.argmax(axis=1)]

    def fit(self, X, y, context):
        from .search_groups import SUPERVISED
        from .model_spec import model_spec
        if context.task_type not in SUPERVISED:
            raise ValueError('Automatic ensembles require a supported regression/classification task')
        context.fit_rows=len(X)
        self.classification = 'classification' in context.task_type
        self.multilabel = context.task_type == 'multilabel_classification'
        self.classes = list(context.classes)
        self.method = self.params['method']
        if self.method not in {'mean','median','weighted','stack'} or (self.method=='median' and self.classification):
            raise ValueError('Unsupported ensemble recipe')
        self.weights = np.full(len(self.sources), 1/len(self.sources))
        requested_weights = self.params.get('weights',self.params.get('component_weights'))
        if self.method == 'mean' and requested_weights is not None:
            if isinstance(requested_weights,dict):
                requested_weights=[requested_weights[source['name']] for source in self.sources]
            values=np.asarray(requested_weights,dtype=float)
            if values.shape != self.weights.shape or not np.isfinite(values).all() or (values<0).any() or values.sum()<=0:
                raise ValueError('Fixed component weights must be finite, nonnegative and match the sources')
            self.weights=values/values.sum()
        self.fold_evidence = []
        if self.method in {'weighted','stack'}:
            folds = self._folds(X, y, context)
            blocks = []
            for source in self.sources:
                oof, covered = None, np.zeros(len(X), dtype=int)
                for number, (a,b) in enumerate(folds):
                    model = self._fit_base(source, X.iloc[a], y.iloc[a] if hasattr(y,'iloc') else np.asarray(y)[a],
                                           context, indices=a, tag=f'oof-{number}')
                    prediction = self._prediction(model, X.iloc[b])
                    if oof is None:
                        oof = np.empty((len(X), *prediction.shape[1:]))
                    oof[b] = prediction
                    covered[b] += 1
                    self.fold_evidence.append(dict(component=source['name'], fold=number,
                        train_indices=a.tolist(), validation_indices=b.tolist()))
                if not np.all(covered == 1) or not np.isfinite(oof).all():
                    raise ValueError('Incomplete or overlapping ensemble OOF predictions')
                blocks.append(oof)
            if self.method == 'weighted':
                self._fit_weights(blocks, y, context)
            else:
                from sklearn.linear_model import Ridge, LogisticRegression
                matrix = np.column_stack([p.reshape(len(X), -1) for p in blocks])
                if self.multilabel:
                    self.meta = []
                    for j in range(np.asarray(y).shape[1]):
                        meta = LogisticRegression(max_iter=1000, random_state=context.seed)
                        target = np.asarray(y)[:, j]
                        meta.fit(matrix, target, sample_weight=context.sample_weight)
                        self.meta.append(meta)
                else:
                    self.meta = LogisticRegression(max_iter=1000, random_state=context.seed) if self.classification else Ridge()
                    self.meta.fit(matrix, y, sample_weight=context.sample_weight)
        self.models = {s['name']:self._fit_base(s, X, y, context, tag='refit', final=True) for s in self.sources}
        self.source_specs = {s['name']:(self.models[s['name']].export_model_spec()
            if callable(getattr(self.models[s['name']],'export_model_spec',None)) else
            model_spec(dict(s['config'],_code_hash=s['code_hash']),self.params[s['name']])) for s in self.sources}
        return self

    def _fit_weights(self, blocks, y, context):
        import optuna
        from types import SimpleNamespace
        from .evaluation import score_predictions, prediction_kind
        task_context = context.task_context or SimpleNamespace(task=SimpleNamespace(primary_metric=context.metric,
            task_type=context.task_type), classes=context.classes, metric_options={})
        kind = prediction_kind(context.metric, context.task_type)
        def objective(trial):
            weights = np.array([trial.suggest_float(f'w{i}',0.,1.) for i in range(len(blocks))])
            weights = weights/weights.sum() if weights.sum() else np.full(len(blocks),1/len(blocks))
            prediction = np.tensordot(weights, np.asarray(blocks), axes=1)
            if kind == 'label': prediction = self._labels(prediction)
            return score_predictions(y, prediction, task_context, weights=context.sample_weight)
        study = optuna.create_study(direction=context.direction, sampler=optuna.samplers.TPESampler(seed=context.seed))
        study.enqueue_trial({f'w{i}':1. for i in range(len(blocks))})
        trials = int(self.params.get('weight_trials',15))
        if not 1 <= trials <= 1000:
            raise ValueError('Invalid ensemble weight-fit trial count')
        study.optimize(objective, n_trials=trials,
                       timeout=max(.1,context.remaining_seconds*.2) if context.timeouts_enabled else None)
        weights = np.array([study.best_params[f'w{i}'] for i in range(len(blocks))])
        self.weights = weights/weights.sum() if weights.sum() else np.full(len(blocks),1/len(blocks))
        self.weight_fit = dict(trials=len(study.trials), score=study.best_value, score_kind='ensemble_training_oof')

    def predict(self, X, prediction_spec):
        values = [self._prediction(self.models[s['name']], X) for s in self.sources]
        if self.method == 'stack':
            matrix = np.column_stack([p.reshape(len(X),-1) for p in values])
            if self.multilabel:
                prediction = np.column_stack([m.predict_proba(matrix)[:,list(m.classes_).index(1)] for m in self.meta])
            elif self.classification:
                prediction = self.meta.predict_proba(matrix)[:, [list(self.meta.classes_).index(c) for c in self.classes]]
            else:
                prediction = self.meta.predict(matrix)
        elif self.method == 'median':
            prediction = np.median(values,axis=0)
        else:
            prediction = np.tensordot(self.weights,np.asarray(values),axes=1)
        return self._labels(prediction) if self.classification and prediction_spec['kind']=='label' else prediction


    def export_model_params(self):
        return dict(copy.deepcopy(self.params), __source_specs=copy.deepcopy(self.source_specs))

    def save(self, artifact_dir):
        import cloudpickle
        import shutil
        root = Path(artifact_dir)
        root.mkdir(parents=True, exist_ok=True)
        # Retain source for independent loading/refit after the original run disappears.
        source_root = self.component_root.parent
        for source in list(source_root.rglob('*.py')):
            destination = root/'source'/source.relative_to(source_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        for name, model in self.models.items():
            (root/'models'/name).mkdir(parents=True,exist_ok=True)
            model.save(str(root/'models'/name))
        models = self.models
        self.models = {}
        try:
            (root/'ensemble.pkl').write_bytes(cloudpickle.dumps(self))
        finally:
            self.models = models

    @classmethod
    def load(cls, artifact_dir):
        import cloudpickle
        from .sdk import load_module
        root = Path(artifact_dir)
        result = cloudpickle.loads((root/'ensemble.pkl').read_bytes())
        result.component_root = root/'source'/'components'
        from .persistence import tree_hash
        expected=result.config.get('_code_hash')
        if expected and tree_hash(root/'source')!=expected:
            raise ValueError('Saved ensemble source hash mismatch')
        if any(tree_hash(result.component_root/s['name'])!=s['code_hash'] for s in result.sources):
            raise ValueError('Saved component source hash mismatch')
        result.models = {s['name']:load_module(result.component_root/s['name']).load_candidate(str(root/'models'/s['name']))
                         for s in result.sources}
        return result
