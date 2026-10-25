# Estimator-only autogluon reference

Runnable AutoGluon 1.3.1 single-output regression/binary/multiclass holdout path. This reference deliberately covers unweighted tasks with the named exact native objectives.
The SDK owns feature preprocessing and target/weight/output policies.

```python
from pathlib import Path
import copy
import cloudpickle
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

class Candidate:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.params = dict(config.get('model_params', {}))

    def fit(self, X, y, context):
        if context.sample_weight is not None or context.validation_weight is not None:
            raise ValueError('This reference is unweighted; use a verified weighted implementation')
        if not isinstance(X,pd.DataFrame):
            raise ValueError('AutoGluon reference requires preprocessing DataFrame output')
        label = '__mlmom_target__'
        if label in X:
            raise ValueError('Reserved target collides with an input feature')
        self.classes = list(context.classes)
        self.classification = context.task_type in {'binary_classification','multiclass_classification'}
        metric = {'rmse':'root_mean_squared_error','mae':'mean_absolute_error','macro_f1':'f1_macro',
                  'f1_macro':'f1_macro','accuracy':'accuracy','roc_auc':'roc_auc','log_loss':'log_loss'}.get(context.metric)
        if metric is None:
            raise ValueError('Implement an exact objective for this primary metric or use a direct fixed estimator')
        params = dict(self.params)
        frozen = params.pop('__native_frozen',None)
        families = params.pop('hyperparameters',{'GBM':{},'CAT':{},'XGB':{},'RF':{},'XT':{}})
        if params:
            raise ValueError('Unsupported reference parameters: '+', '.join(sorted(params)))
        identity = (self.config.get('target_transform',{}).get('name','identity')=='identity'
            and self.config.get('target_transform',{}).get('clip_min') is None
            and self.config.get('prediction_policy',{}).get('clip_min') is None
            and self.config.get('prediction_policy',{}).get('nonfinite','error')=='error')
        if not frozen and (not identity or context.task_context is None or context.task_context.metric_options):
            raise ValueError('Native search reference needs identity policies and default metric options; use an exact custom recipe or direct fixed model')
        if not frozen and context.validation_X is None:
            raise ValueError('Native search requires the explicit internal holdout')
        if frozen:
            families = {frozen['family']:copy.deepcopy(frozen['params'])}
        if not set(families) <= {'GBM','CAT','XGB','RF','XT'}:
            raise ValueError('Only the declared non-pretrained families are used by this reference')
        train = X.assign(**{label:np.asarray(y)})
        valid = context.validation_X.assign(**{label:np.asarray(context.validation_y)}) if context.validation_X is not None and not frozen else None
        allowance = context.remaining_seconds if context.timeouts_enabled else context.time_limit_seconds
        if not np.isfinite(allowance):
            raise ValueError('Native library budget must be finite')
        self.model = TabularPredictor(label=label,path=str(Path(context.work_dir)/'predictor'),eval_metric=metric,verbosity=0)
        self.model.fit(train,tuning_data=valid,time_limit=max(1.,allowance),
            hyperparameters=families,fit_weighted_ensemble=False,num_cpus=context.cpus,num_gpus=0,
            fit_strategy='sequential',ag_args_fit={'early_stop':None} if frozen else {})
        if frozen:
            self.model.refit_full(model='best',set_best_to_refit_full=True,num_cpus=context.cpus,
                                  num_gpus=0,fit_strategy='sequential')
            self.frozen = frozen
        else:
            info = self.model.model_info(self.model.model_best)
            family = {'LGBModel':'GBM','LightGBMModel':'GBM','CatBoostModel':'CAT',
                      'XGBoostModel':'XGB','RFModel':'RF','XTModel':'XT'}.get(str(info['model_type']))
            if family is None:
                raise ValueError('Selected native family has no frozen reference recipe')
            self.frozen = dict(family=family,params=dict(info['hyperparameters'],**info.get('hyperparameters_fit',{})))
        self.history = dict(mode='fixed_configuration' if frozen else 'native_holdout',new_trials=0 if frozen else len(self.model.model_names()))
        self.stopping_info = dict(mode='fixed' if frozen else 'primary_metric',
            reason='Frozen family/parameters with full-data refit' if frozen else
                   'Exact native primary objective with identity policies',metric=context.metric)
        return self

    def predict(self, X, prediction_spec):
        if prediction_spec['kind']=='probability':
            return self.model.predict_proba(X).loc[:,self.classes].to_numpy()
        return np.asarray(self.model.predict(X))

    def export_model_params(self):
        return dict(self.params,__native_frozen=self.frozen)

    def native_history(self):
        return self.history

    def save(self, artifact_dir):
        self.model.save_space()
        self.model.clone(str(Path(artifact_dir)/'predictor'))
        state={k:v for k,v in vars(self).items() if k!='model'}
        (Path(artifact_dir)/'state.pkl').write_bytes(cloudpickle.dumps(state))

def create_candidate(config):
    return Candidate(config)

def load_candidate(artifact_dir):
    candidate=Candidate({})
    vars(candidate).update(cloudpickle.loads((Path(artifact_dir)/'state.pkl').read_bytes()))
    candidate.model=TabularPredictor.load(str(Path(artifact_dir)/'predictor'),verbosity=0)
    return candidate
```

Unknown protocols are not an automatic admission veto. This concrete recipe searches only explicit non-pretrained families, freezes selected hyperparameters (including trained counts), and refits that family without another family search. Cached fit data are removed before cloning. Native selection with transformed targets, output policies or nondefault metric options requires an exact custom recipe; choose a direct fixed estimator when unavailable.
