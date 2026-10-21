# Direct LightAutoML reference
```python
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
# Exact objective mappings: regression=rmse, binary=roc_auc, multiclass=log_loss.
self.model = TabularAutoML(task=Task('reg'), timeout=max(1.,context.remaining_seconds),
    cpu_limit=context.cpus, memory_limit=max(1,int(context.memory_gb)), gpu_ids='',
    general_params={'use_algos':[['lgb','cb']]}, reader_params={'cv':3,'random_state':context.seed})
train = X.assign(__target__=np.asarray(y))
valid = context.validation_X.assign(__target__=np.asarray(context.validation_y))
self.model.fit_predict(train, roles={'target':'__target__'}, valid_data=valid)
```
Adapt Task('binary') / Task('multiclass') and encode/inverse-map classification labels. Do not silently substitute an objective. predict(X).data yields values; expand binary one-column probabilities. Persist fitted model and transforms. Fixed refit must use frozen learner settings, no tuning algorithms; in-sample reports are not external evaluation. No text/pretrained pipelines or outer Optuna.
