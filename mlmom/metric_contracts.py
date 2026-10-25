"""Executable metric names, task scopes and options shared by every boundary."""
import re

ALIASES = {
    'root_mean_squared_error':'rmse', 'root_mean_square_error':'rmse',
    'mean_squared_error':'mse', 'mean_absolute_error':'mae', 'r2_score':'r2',
    'mean_absolute_percentage_error':'mape',
    'root_mean_squared_logarithmic_error':'rmsle', 'root_mean_squared_log_error':'rmsle',
    'symmetric_mean_absolute_percentage_error':'smape', 'mean_squared_logarithmic_error':'msle',
    'area_under_roc_curve':'roc_auc', 'roc_auc_score':'roc_auc', 'auc':'roc_auc',
    'balanced_accuracy_score':'balanced_accuracy', 'average_precision_score':'average_precision',
    'f1_score':'f1', 'macro_f1':'f1_macro', 'micro_f1':'f1_micro', 'weighted_f1':'f1_weighted',
    'mean_average_precision':'map', 'mean_reciprocal_rank':'mrr',
    'matthews_corrcoef':'mcc', 'matthews_correlation_coefficient':'mcc', 'matthews_correlation':'mcc',
}
REGRESSION = ['regression', 'multioutput_regression', 'time_series_forecasting']
CLASSIFICATION = ['binary_classification', 'multiclass_classification', 'multilabel_classification']


def canonical_metric(value):
    name = re.sub(r'[^a-z0-9]+', '_', (value or '').strip().lower()).strip('_')
    return ALIASES.get(name, name)


def normalise_metric(value):
    """Preserve the historical task-parser spelling of F1 aliases."""
    name = canonical_metric(value)
    return {'f1_macro':'macro_f1','f1_micro':'micro_f1','f1_weighted':'weighted_f1'}.get(name, name)


METRICS = {}


def _add(names, tasks, direction, options=(), *, weighted=True):
    for name in names:
        METRICS[name] = dict(task_types=list(tasks), direction=direction,
                             options=set(options), sample_weight=weighted)


_add(['rmse','mse','mae'], REGRESSION, 'minimize', ['multioutput'])
_add(['r2'], REGRESSION, 'maximize', ['multioutput'])
_add(['rmsle','mape','smape'], REGRESSION, 'minimize')
_add(['accuracy'], CLASSIFICATION+['anomaly_detection'], 'maximize')
_add(['balanced_accuracy'], CLASSIFICATION[:2]+['anomaly_detection'], 'maximize')
_add(['f1','precision','recall'], CLASSIFICATION+['anomaly_detection'], 'maximize', ['average','pos_label'])
_add(['f1_macro','f1_micro','f1_weighted'], CLASSIFICATION+['anomaly_detection'], 'maximize', ['average'])
_add(['roc_auc'], CLASSIFICATION+['anomaly_detection'], 'maximize', ['average','multi_class','max_fpr'])
_add(['average_precision'], CLASSIFICATION+['anomaly_detection'], 'maximize', ['average','pos_label'])
_add(['log_loss'], CLASSIFICATION, 'minimize')
_add(['mcc'], CLASSIFICATION[:2], 'maximize')
_add(['ndcg','map','mrr'], ['ranking'], 'maximize', ['k'], weighted=False)
_add(['silhouette'], ['clustering'], 'maximize', ['sample_size','random_state','metric'], weighted=False)
_add(['calinski_harabasz'], ['clustering'], 'maximize', weighted=False)
_add(['davies_bouldin'], ['clustering'], 'minimize', weighted=False)

# Compatibility view includes aliases, without maintaining a second capability table.
METRIC_OPTIONS = {name:set(spec['options']) for name,spec in METRICS.items()}
METRIC_OPTIONS.update({alias:set(METRICS[name]['options']) for alias,name in ALIASES.items() if name in METRICS})


def metric_capabilities():
    return {metric:sorted(options) for metric,options in METRIC_OPTIONS.items()}


def metric_catalog():
    return [dict(name=name, aliases=sorted(k for k,v in ALIASES.items() if v==name),
                 task_types=spec['task_types'], direction=spec['direction'],
                 options=sorted(spec['options']), sample_weight=spec['sample_weight'])
            for name,spec in sorted(METRICS.items())]


def metric_errors(name, task_type, options):
    name = canonical_metric(name)
    spec = METRICS.get(name)
    if spec is None:
        return [f'Unsupported metric: {name}; choose an executable metric from supported_metrics']
    errors = []
    if task_type not in spec['task_types']:
        errors.append(f'Metric {name} does not support task type {task_type}')
    unknown = set(options)-spec['options']
    if unknown:
        errors.append(f'Unsupported options for {name}: '+', '.join(sorted(unknown)))
    if name in {'f1','precision','recall','f1_macro','f1_micro','f1_weighted'}:
        average = options.get('average',name[3:] if name.startswith('f1_') else 'binary')
        if average not in ('binary','macro','micro','weighted','samples'):
            errors.append(f'Metric {name} requires a scalar averaging mode')
        elif average == 'binary' and task_type in CLASSIFICATION[1:]:
            errors.append(f'Metric {name} requires an explicit non-binary average for {task_type}')
        elif average == 'samples' and task_type != 'multilabel_classification':
            errors.append('average=samples is supported only for multilabel classification')
    if name == 'roc_auc' and task_type == 'multiclass_classification':
        if options.get('max_fpr') not in (None,1,1.):
            errors.append('Multiclass ROC AUC does not support partial AUC (max_fpr must be null or 1)')
        if options.get('multi_class','ovr') not in ('ovr','ovo'):
            errors.append('Multiclass ROC AUC requires multi_class=ovr or ovo')
    return errors
