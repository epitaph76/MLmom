# Estimator-only lightautoml reference

Runnable LightAutoML 0.4.2 single-target tabular reference with one explicitly selected untuned lgb/cb learner, frozen counts and no native ensemble-weight reselection.
The SDK owns feature preprocessing and target/weight/output policies.

```python
from pathlib import Path
import copy
import cloudpickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

def root_mean_squared_error(y_true,y_pred,sample_weight=None,**kwargs):
    error=(np.asarray(y_true).ravel()-np.asarray(y_pred).ravel())**2
    return float(np.sqrt(np.average(error,weights=sample_weight)))

class Candidate:
    def __init__(self, config):
        self.config=copy.deepcopy(config)
        self.params=dict(config.get('model_params',{}))

    def fit(self,X,y,context):
        if context.sample_weight is not None or context.validation_weight is not None:
            raise ValueError('This reference is unweighted; implement and verify a weighted recipe explicitly')
        if not isinstance(X,pd.DataFrame):
            raise ValueError('LightAutoML reference requires DataFrame output')
        if '__mlmom_target__' in X:
            raise ValueError('Reserved target collides with an input feature')
        self.classes=list(context.classes)
        self.classification=context.task_type in {'binary_classification','multiclass_classification'}
        self.encoder=LabelEncoder().fit(context.classes) if self.classification else None
        labels=self.encoder.transform(np.asarray(y)) if self.encoder is not None else np.asarray(y)
        params=dict(self.params)
        frozen=params.pop('__native_frozen',None)
        learner=params.pop('learner','lgb')
        count=int(params.pop('num_trees',200))
        learner_params=params.pop('learner_params',{})
        if params:
            raise ValueError('Unsupported reference parameters: '+', '.join(sorted(params)))
        if frozen:
            learner,count,learner_params=frozen['learner'],frozen['num_trees'],frozen['learner_params']
        if learner not in {'lgb','cb'}:
            raise ValueError('Use one explicit untuned lgb or cb learner')
        expected={'regression':'rmse','binary_classification':'roc_auc','multiclass_classification':'log_loss'}.get(context.task_type)
        if context.metric!=expected:
            raise ValueError('Implement an exact LightAutoML task/objective or choose a direct fixed estimator')
        identity=(self.config.get('target_transform',{}).get('name','identity')=='identity'
            and self.config.get('target_transform',{}).get('clip_min') is None
            and self.config.get('prediction_policy',{}).get('clip_min') is None
            and self.config.get('prediction_policy',{}).get('nonfinite','error')=='error')
        if not frozen and (not identity or context.task_context is None or context.task_context.metric_options):
            raise ValueError('Native selection needs identity policies and default metric options; use an exact recipe or direct fixed estimator')
        if not frozen and context.validation_X is None:
            raise ValueError('Native selection requires explicit internal holdout')
        task_name={'regression':'reg','binary_classification':'binary','multiclass_classification':'multiclass'}[context.task_type]
        task=Task(task_name,metric=root_mean_squared_error,greater_is_better=False) if task_name=='reg' else Task(task_name)
        allowance=context.remaining_seconds if context.timeouts_enabled else context.time_limit_seconds
        if not np.isfinite(allowance):
            raise ValueError('Native library budget must be finite')
        base=dict(learner_params,num_trees=int(count),early_stopping_rounds=int(count)+1)
        if learner=='lgb':
            base.update(num_threads=context.cpus,seed=context.seed)
        else:
            base.update(thread_count=context.cpus,task_type='CPU',use_best_model=False,random_seed=context.seed)
        options={learner+'_params':{'default_params':base,'freeze_defaults':True}}
        self.model=TabularAutoML(task=task,timeout=max(1,int(allowance)),cpu_limit=context.cpus,
            memory_limit=max(1,int(context.memory_gb)),gpu_ids='',
            general_params={'use_algos':[[learner]]},reader_params={'cv':3,'random_state':context.seed},**options)
        train=X.assign(__mlmom_target__=labels)
        if frozen:
            valid=train
        else:
            vy=self.encoder.transform(np.asarray(context.validation_y)) if self.encoder is not None else np.asarray(context.validation_y)
            valid=context.validation_X.assign(__mlmom_target__=vy)
        self.model.fit_predict(train,roles={'target':'__mlmom_target__'},valid_data=valid,verbose=0)
        self.frozen=dict(learner=learner,num_trees=int(count),learner_params=copy.deepcopy(learner_params))
        self.history=dict(mode='fixed_configuration' if frozen else 'untuned_holdout',new_trials=0)
        self.stopping_info=dict(mode='fixed',reason='One fixed untuned learner; no blend weight search or early-stop count selection',metric=context.metric)
        return self

    def predict(self,X,prediction_spec):
        values=np.asarray(self.model.predict(X).data)
        if not self.classification:
            return values.ravel()
        if values.shape[1]==1:
            values=np.column_stack([1-values[:,0],values[:,0]])
        if prediction_spec['kind']=='probability':
            return values[:,self.encoder.transform(self.classes)]
        return self.encoder.inverse_transform(values.argmax(axis=1))

    def export_model_params(self):
        return dict(self.params,__native_frozen=self.frozen)

    def native_history(self):
        return self.history

    def save(self,artifact_dir):
        (Path(artifact_dir)/'candidate.pkl').write_bytes(cloudpickle.dumps(self))

def create_candidate(config):
    return Candidate(config)

def load_candidate(artifact_dir):
    return cloudpickle.loads((Path(artifact_dir)/'candidate.pkl').read_bytes())
```

Task(reg) defaults to MSE upstream; the explicit RMSE callable matches this primary metric. Binary uses AUC and multiclass cross-entropy. Full-data refit uses train-as-validation only to satisfy the preset API, with the same fixed learner/count and no score used as evidence. Broader objectives/weighting/policies require an explicitly implemented exact recipe. No text, pretrained models, outer Optuna or silently substituted folds.
