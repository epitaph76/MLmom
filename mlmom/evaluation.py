"""Trusted metric computation, prediction validation and aggregate diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics as m
from .task_validator import _normalise_metric
from .pipeline_contracts import EvaluationResult, TaskContext


def metric_name(name: str) -> str:
    value = _normalise_metric(name)
    return {'macro_f1': 'f1_macro', 'micro_f1': 'f1_micro', 'weighted_f1': 'f1_weighted',
            'auc': 'roc_auc', 'mean_squared_error': 'mse', 'mean_absolute_error': 'mae',
            'r2_score': 'r2', 'mean_average_precision': 'map'}.get(value, value)


def prediction_kind(metric: str, task_type: str) -> str:
    metric = metric_name(metric)
    if metric in {'roc_auc', 'average_precision', 'log_loss'}:
        return 'anomaly_score' if task_type == 'anomaly_detection' else 'probability'
    if task_type == 'ranking':
        return 'ranking_score'
    if task_type == 'clustering':
        return 'cluster'
    if 'classification' in task_type:
        return 'label'
    return 'value'


def checked_predictions(value, rows: int, kind: str, *, classes=None):
    a = np.asarray(value)
    if a.ndim not in (1, 2) or len(a) != rows:
        raise ValueError(f'Expected {rows} predictions, received shape {a.shape}')
    if pd.isna(a).any():
        raise ValueError('Predictions contain missing values')
    if np.issubdtype(a.dtype,np.number) and not np.isfinite(a).all():raise ValueError('Numeric predictions must be finite')
    if kind not in {'label', 'cluster'}:
        a = a.astype(float)
        if not np.isfinite(a).all():
            raise ValueError('Predictions must be finite')
    if kind == 'label' and classes and a.ndim == 1:
        if not set(a.tolist()) <= set(classes):
            raise ValueError('Predictions contain unknown class labels')
    if kind == 'probability':
        if np.any(a < -1e-8) or np.any(a > 1 + 1e-8):
            raise ValueError('Invalid probabilities')
        if classes and a.ndim==2:
            if a.shape[1]!=len(classes):raise ValueError('Probability columns do not match class order')
            if not np.allclose(a.sum(axis=1),1.,atol=1e-6):raise ValueError('Class probabilities must sum to one')
    return a


def score_predictions(y, prediction, context: TaskContext, *, X=None, groups=None, weights=None) -> float:
    name = metric_name(context.task.primary_metric)
    kind = prediction_kind(name, context.task.task_type)
    n = len(X) if y is None else len(y)
    p = checked_predictions(prediction, n, kind, classes=context.classes)
    opts = dict(context.metric_options)
    if weights is not None:
        if name in {'ndcg','map','mrr','silhouette','calinski_harabasz','davies_bouldin'}:
            raise ValueError('This metric has no agreed sample-weight semantics')
        if name!='smape':opts['sample_weight']=weights
    if name in {'silhouette', 'calinski_harabasz', 'davies_bouldin'}:
        if X is None or not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
            raise ValueError('Clustering metrics require explicit finite numeric metric features')
        numeric = np.asarray(X, dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError('Clustering metric features must be finite')
        fn = {'silhouette': m.silhouette_score, 'calinski_harabasz': m.calinski_harabasz_score,
              'davies_bouldin': m.davies_bouldin_score}[name]
        if name == 'silhouette':
            opts.setdefault('sample_size', min(2000, n))
            opts.setdefault('random_state', 42)
        result = fn(numeric, p, **opts)
    elif y is None:
        raise ValueError('A supervised metric requires evaluation targets')
    elif name in {'ndcg', 'map', 'mrr'}:
        if groups is None:
            raise ValueError('Ranking metrics require query groups')
        labels = np.asarray(y)
        values = []
        for group in pd.unique(groups):
            mask = np.asarray(groups) == group
            truth, scores = labels[mask], np.asarray(p)[mask].ravel()
            if len(truth) < 2:
                continue
            k = opts.get('k', len(truth))
            if name == 'ndcg':
                values.append(m.ndcg_score([truth], [scores], k=k))
            else:
                relevant = truth[np.argsort(-scores, kind='stable')[:k]] > 0
                hits = np.flatnonzero(relevant)
                if name == 'mrr':
                    values.append(1 / (hits[0] + 1) if len(hits) else 0.)
                else:
                    denom = min(int((truth > 0).sum()), k)
                    values.append(float(np.sum(np.arange(1, len(hits)+1)/(hits+1))) / denom if denom else 0.)
        if not values:
            raise ValueError('No evaluable ranking queries')
        result = np.mean(values)
    elif name in {'rmse', 'mse', 'rmsle', 'mae', 'mape', 'smape', 'r2'}:
        truth = np.asarray(y)
        if name in {'rmse', 'mse'}:
            result = m.mean_squared_error(truth, p, **opts)
            if name == 'rmse': result = np.sqrt(result)
        elif name == 'rmsle': result = np.sqrt(m.mean_squared_log_error(truth, p, **opts))
        elif name == 'mae': result = m.mean_absolute_error(truth, p, **opts)
        elif name == 'mape': result = m.mean_absolute_percentage_error(truth, p, **opts)
        elif name == 'r2': result = m.r2_score(truth, p, **opts)
        else:
            denom = np.abs(truth) + np.abs(p)
            error=np.divide(2*np.abs(truth-p), denom, out=np.zeros_like(p,dtype=float),where=denom!=0)
            result=np.average(error,axis=0,weights=weights).mean()
    elif name == 'roc_auc':
        if np.asarray(y).ndim == 1 and len(context.classes) == 2 and p.ndim == 2:
            p = p[:, 1]
        if p.ndim == 2 and np.asarray(y).ndim == 1:
            opts.setdefault('multi_class', 'ovr')
            opts.setdefault('labels', context.classes)
        result = m.roc_auc_score(y, p, **opts)
    elif name == 'average_precision':
        if p.ndim == 2 and p.shape[1] == 2 and np.asarray(y).ndim == 1:
            p = p[:, 1]
        if np.asarray(y).ndim == 1 and len(context.classes) > 2:
            from sklearn.preprocessing import label_binarize
            y = label_binarize(y, classes=context.classes)
        elif np.asarray(y).ndim == 1 and len(context.classes) == 2:
            y = np.asarray(y) == context.classes[1]
        result = m.average_precision_score(y, p, **opts)
    elif name == 'log_loss':
        if context.classes: opts.setdefault('labels', context.classes)
        result = m.log_loss(y, p, **opts)
    elif name.startswith('f1'):
        opts.setdefault('average', name.removeprefix('f1_') if name != 'f1' else 'binary')
        if opts['average']=='binary' and len(context.classes)==2:opts.setdefault('pos_label',context.classes[1])
        opts.setdefault('zero_division', 0)
        result = m.f1_score(y, p, **opts)
    elif name in {'precision', 'recall'}:
        if opts.get('average','binary')=='binary' and len(context.classes)==2:opts.setdefault('pos_label',context.classes[1])
        opts.setdefault('zero_division', 0)
        result = getattr(m, name + '_score')(y, p, **opts)
    elif name in {'accuracy', 'balanced_accuracy'}:
        result = getattr(m, name + '_score')(y, p, **opts)
    else:
        raise ValueError(f'Unsupported metric: {name}')
    result = float(result)
    if not np.isfinite(result):
        raise ValueError('Metric is not finite')
    return result


def diagnostics(y, prediction, context: TaskContext, *, X=None) -> dict:
    if y is None:
        labels, counts = np.unique(prediction, return_counts=True)
        return {'cluster_sizes': dict(zip(map(str, labels), map(int, counts)))}
    if 'classification' in context.task.task_type and np.asarray(y).ndim == 1:
        if np.asarray(prediction).ndim!=1:return {}
        result={'classification_report': m.classification_report(y, prediction, output_dict=True, zero_division=0),
                'confusion_matrix': m.confusion_matrix(y, prediction, labels=context.classes).tolist(),
                'class_order': context.classes}
        row_error=(np.asarray(y)!=np.asarray(prediction)).astype(float)
    elif 'regression' in context.task.task_type:
        error = np.asarray(y) - np.asarray(prediction)
        result={'absolute_error_quantiles': dict(zip(['p50','p90','p99'], np.quantile(np.abs(error), [.5,.9,.99]).tolist())),
                'mean_residual': float(np.mean(error))}
        row_error=np.abs(error) if error.ndim==1 else np.abs(error).mean(axis=1)
    else:return {}
    slices=[]
    if X is not None:
        for column in X.select_dtypes(include='number').columns[:3]:
            try:bins=pd.qcut(X[column],4,labels=False,duplicates='drop').to_numpy()
            except ValueError:continue
            for quantile in range(4):
                mask=bins==quantile
                if mask.sum()>=20:slices.append({'feature':column,'quantile':quantile+1,'rows':int(mask.sum()),'mean_error':float(np.mean(row_error[mask]))})
    result['safe_slices']=slices
    return result


def better(new: float, old: float, direction: str, delta=0.) -> bool:
    gain = new-old if direction == 'maximize' else old-new
    return gain > max(delta, 1e-12 * max(1., abs(new), abs(old)))


def compact_diagnostics(value):
    """Keep numerical evidence in prompts; full matrices remain in artifacts."""
    result={k:v for k,v in value.items() if k in {'artifact_ref','absolute_error_quantiles','mean_residual','cluster_sizes'}}
    report=value.get('classification_report',{})
    classes=[(k,v) for k,v in report.items() if isinstance(v,dict) and k not in {'macro avg','weighted avg','micro avg'}]
    if classes:result['worst_classes']=dict(sorted(classes,key=lambda item:item[1].get('f1-score',0))[:5])
    if 'macro avg' in report:result['macro_average']=report['macro avg']
    result['worst_slices']=sorted(value.get('safe_slices',[]),key=lambda row:-row['mean_error'])[:6]
    inspection=value.get('model_inspection',{})
    for key in ['feature_importance','permutation_importance','contributions']:
        data=inspection.get(key)
        if data:
            result[key]={k:v for k,v in data.items() if k in {'status','reason','baseline'}}
            scores=data.get('values',data.get('mean_absolute',{}))
            result[key]['top']=dict(sorted(scores.items(),key=lambda item:-item[1])[:5])
    return result


def evaluate(experiment_id: str, split_id: str, y, prediction, context: TaskContext, *, X=None, groups=None, weights=None, label_prediction=None) -> EvaluationResult:
    return EvaluationResult(experiment_id=experiment_id, split_id=split_id,
        metric=metric_name(context.task.primary_metric), direction=context.task.metric_direction,
        score=score_predictions(y, prediction, context, X=X, groups=groups,weights=weights),
        diagnostics=diagnostics(y, label_prediction if label_prediction is not None else prediction, context, X=X))
