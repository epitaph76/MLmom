"""Blending and stacking fitted only on the current experiment's training data."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import numpy as np
import time
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputClassifier
from .adapters import LibraryCandidate, classification,load_adapter


class EnsembleCandidate:
    def __init__(self, config):
        self.config = config
        self.method = config.get('model_params',{}).get('method','blend')
        self.models = []

    def fit(self, X, y, context):
        if not context.task_type.endswith('regression') and not classification(context.task_type):
            raise ValueError('Ensemble adapter supports regression and classification')
        self.classes = context.classes
        self.kind = 'probability' if classification(context.task_type) else 'value'
        self.multilabel = context.task_type == 'multilabel_classification'
        components = self.config.get('model_params',{}).get('__ensemble_frozen',self.config.get('ensemble_components',[]))
        if len(components) < 2: raise ValueError('At least two ensemble components are required')
        self.meta = None
        if self.method == 'stack':
            if context.groups is not None:
                folds = list(GroupKFold(3).split(X,y,context.groups))
            elif classification(context.task_type) and not self.multilabel:
                folds = list(StratifiedKFold(3,shuffle=True,random_state=context.seed).split(X,y))
            else:
                folds = list(KFold(3,shuffle=True,random_state=context.seed).split(X))
            blocks = []
            for component in components:
                block = None
                for number,(train,valid) in enumerate(folds):
                    model = LibraryCandidate(component)
                    c = replace(context, groups=np.asarray(context.groups)[train] if context.groups is not None else None,
                        sample_weight=np.asarray(context.sample_weight)[train] if context.sample_weight is not None else None,
                        validation_X=X.iloc[valid],validation_y=y.iloc[valid],
                        work_dir=str(Path(context.work_dir)/f'oof-{len(blocks)}-{number}'),
                        time_limit_seconds=max(.1,context.remaining_seconds/(len(components)*4)),started=time.monotonic())
                    model.fit(X.iloc[train],y.iloc[train],c)
                    p = np.asarray(model.predict(X.iloc[valid],{'kind':self.kind})).reshape(len(valid),-1)
                    if block is None: block = np.zeros((len(X),p.shape[1]))
                    block[valid] = p
                blocks.append(block)
            oof = np.column_stack(blocks)
            if classification(context.task_type):
                self.meta = LogisticRegression(max_iter=500)
                if self.multilabel: self.meta = MultiOutputClassifier(self.meta)
            else: self.meta = Ridge()
            self.meta.fit(oof,y,**({'sample_weight':context.sample_weight} if context.sample_weight is not None else {}))
        elif self.method != 'blend':
            raise ValueError('Unknown ensemble method')
        for i,component in enumerate(components):
            model = LibraryCandidate(component)
            model.fit(X,y,replace(context,work_dir=str(Path(context.work_dir)/f'component-{i}'),
                                 time_limit_seconds=max(.1,context.remaining_seconds/(len(components)-i)),started=time.monotonic()))
            self.models.append(model)
        return self

    def export_model_params(self):
        params=dict(self.config.get('model_params',{}))
        params['__ensemble_frozen']=[dict(model.config,model_params=model.export_model_params()) for model in self.models]
        return params

    def predict(self, X, prediction_spec):
        values = [np.asarray(m.predict(X,{'kind':self.kind})) for m in self.models]
        if self.meta:
            stacked = np.column_stack([p.reshape(len(X),-1) for p in values])
            if prediction_spec['kind']=='probability':
                result = self.meta.predict_proba(stacked)
                return np.column_stack([p[:,1] for p in result]) if isinstance(result,list) else result
            return self.meta.predict(stacked)
        blended = np.mean(values,axis=0)
        if self.kind != 'probability' or prediction_spec['kind']=='probability': return blended
        if self.multilabel: return (blended >= .5).astype(int)
        return np.asarray(self.classes)[blended.argmax(axis=1)]

    def save(self, artifact_dir):
        import cloudpickle
        path = Path(artifact_dir); path.mkdir(parents=True,exist_ok=True)
        models=self.models
        for i,model in enumerate(models):model.save(str(path/f'component-{i}'))
        self.model_count=len(models)
        self.models=[]
        try:(path/'ensemble.pkl').write_bytes(cloudpickle.dumps(self))
        finally:
            self.models=models

    def native_history(self):
        return {'components':[model.native_history() for model in self.models]}


def load_ensemble(artifact_dir):
    import cloudpickle
    candidate=cloudpickle.loads((Path(artifact_dir)/'ensemble.pkl').read_bytes())
    if hasattr(candidate,'model_count'):
        candidate.models=[load_adapter(Path(artifact_dir)/f'component-{i}') for i in range(candidate.model_count)]
    return candidate
