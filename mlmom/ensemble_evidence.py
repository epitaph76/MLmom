"""Advisory diagnostics of historical inner OOF; never ensemble training data."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from .evaluation import checked_predictions, prediction_kind, score_predictions


def protocol_digest(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def oof_snapshot(protocol, context, indices, predictions, kind, y, weights=None):
    """Self-contained historical training diagnostics, including original-scale targets."""
    indices = np.asarray(indices, dtype=int)
    by_index = {i: (f['number'], source) for f in protocol['folds']
                for i, source in zip(f['validation_indices'], f['source_validation_indices'])}
    return dict(version=2, dataset_hash=context.dataset_hash,
        task_type=str(context.task.task_type), primary_metric=context.task.primary_metric,
        metric_options=context.metric_options, classes=context.classes,
        protocol_digest=protocol_digest(protocol), outer_split_id=protocol['outer_split_id'],
        indices=indices.tolist(), source_indices=[by_index[i][1] for i in indices],
        fold_numbers=[by_index[i][0] for i in indices], prediction_kind=kind,
        predictions=np.asarray(predictions).tolist(), targets=None if y is None else np.asarray(y)[indices].tolist(),
        sample_weight=None if weights is None else np.asarray(weights)[indices].tolist(),
        scope='historical_inner_validation', output_scale='original')


def component_availability(row):
    run = row.get('run', {})
    return dict(oof_status='registered_not_inspected' if run.get('oof_ref') else 'unavailable',
        oof_ref=run.get('oof_ref'), protocol_ref=run.get('validation_protocol_ref'),
        refit_cost=component_refit_cost(row))


def component_refit_cost(row):
    run = row.get('run', {})
    fits = [f for f in run.get('execution_facts', [])
            if f.get('stage') in {'full_refit', 'submission_refit'} and not f.get('component')
            and isinstance(f.get('duration_seconds'), (int, float))
            and np.isfinite(f['duration_seconds']) and f['duration_seconds'] >= 0]
    samples = [dict(seconds=f['duration_seconds'], fit_rows=f.get('fit_rows'), stage=f['stage'],
                    fit_id=f.get('fit_id'), cpus=run.get('resource_cpus'),
                    memory_gb=run.get('resource_memory_gb')) for f in fits]
    return dict(status='measured' if samples else 'unknown', samples=samples,
        source_experiment_seconds=run.get('duration_seconds'),
        source_duration_scope='historical experiment wall time, may include HPO; not selected-refit latency')


def _load(row, context):
    run = row.get('run', {})
    if not run.get('oof_ref') or not run.get('validation_protocol_ref'):
        raise ValueError('OOF artifact or validation protocol is unavailable')
    for reference in (run['oof_ref'], run['validation_protocol_ref']):
        if Path(reference).stat().st_size > 16*1024*1024:
            raise ValueError('Historical diagnostic artifact exceeds the 16 MiB read limit; measurement unavailable')
    data = json.loads(Path(run['oof_ref']).read_text(encoding='utf-8'))
    protocol = json.loads(Path(run['validation_protocol_ref']).read_text(encoding='utf-8'))
    if data.get('version') != 2:
        raise ValueError('Legacy OOF lacks targets, weights or class/metric scope; no error claim is possible')
    if data.get('targets') is None:
        raise ValueError('Error diagnostics require historical validation targets')
    expected = dict(dataset_hash=context.dataset_hash, task_type=str(context.task.task_type),
        primary_metric=context.task.primary_metric, metric_options=context.metric_options,
        classes=context.classes, scope='historical_inner_validation', output_scale='original')
    if any(data.get(k) != v for k, v in expected.items()):
        raise ValueError('OOF dataset, task, metric options, classes or output scale differ')
    if data.get('protocol_digest') != protocol_digest(protocol):
        raise ValueError('OOF validation protocol differs from its saved protocol')
    if protocol.get('dataset_hash') != context.dataset_hash:
        raise ValueError('Validation protocol belongs to a different dataset')
    if data.get('outer_split_id') != run.get('split_id'):
        raise ValueError('OOF belongs to a different outer training scope')
    mapping = {i: (f['number'], source) for f in protocol['folds']
               for i, source in zip(f['validation_indices'], f['source_validation_indices'])}
    indices = data['indices']
    if len(set(indices)) != len(indices) or set(indices) != set(mapping):
        raise ValueError('OOF rows do not cover the saved validation folds exactly once')
    if ([mapping[i][0] for i in indices] != data['fold_numbers'] or
            [mapping[i][1] for i in indices] != data['source_indices'] or
            len(set(data['source_indices'])) != len(indices)):
        raise ValueError('OOF row/fold alignment is inconsistent')
    kind = prediction_kind(context.task.primary_metric, context.task.task_type)
    if data['prediction_kind'] != kind:
        raise ValueError('OOF prediction kind differs from the primary metric')
    order = np.argsort(data['source_indices'])
    p = checked_predictions(data['predictions'], len(indices), kind, classes=context.classes)[order]
    y = np.asarray(data['targets'])[order]
    if len(y) != len(indices):
        raise ValueError('OOF targets do not match the validation rows')
    w = np.ones(len(y)) if data['sample_weight'] is None else np.asarray(data['sample_weight'], dtype=float)[order]
    if w.shape != (len(y),) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('OOF weights must be finite, nonnegative, with positive total')
    return dict(data=data, predictions=p, targets=y, weights=w,
                rows=np.asarray(data['source_indices'])[order],
                folds=np.asarray(data['fold_numbers'])[order])


def _correlation(a, b, w):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a-np.average(a, weights=w), b-np.average(b, weights=w)
    scale = np.sqrt(np.average(a*a, weights=w)*np.average(b*b, weights=w))
    return dict(value=float(np.clip(np.average(a*b, weights=w)/scale, -1, 1)) if scale > 0 else None,
                reason=None if scale > 0 else 'constant error vector; correlation undefined')


def _errors(item, context):
    y, p = item['targets'], item['predictions']
    if context.task.task_type in {'regression', 'multioutput_regression'}:
        residual = (p.reshape(y.shape)-y).reshape(len(y), -1)
        loss = np.mean(np.abs(residual), axis=1)
        return residual, loss, None, 'signed_residual_per_output', 'mean_absolute_error_per_row'
    if 'classification' not in context.task.task_type:
        raise ValueError('Pair error diagnostics currently support regression and classification')
    probability = item['data']['prediction_kind'] == 'probability'
    if context.task.task_type == 'multilabel_classification':
        labels = (p >= .5).astype(int) if probability else p
        error = np.any(labels != y, axis=1)
        loss = np.mean((p-y)**2, axis=1) if probability else np.mean(labels != y, axis=1)
    else:
        if probability:
            if p.ndim == 1 or p.shape[1] == 1:
                p = np.column_stack([1-p.ravel(), p.ravel()])
            labels = np.asarray(context.classes)[p.argmax(axis=1)]
            indicator = (y.reshape(-1, 1) == np.asarray(context.classes)).astype(float)
            loss = np.mean((p-indicator)**2, axis=1)
        else:
            labels = p.ravel()
            loss = (labels != y.ravel()).astype(float)
        error = labels != y.ravel()
    return error.astype(float)[:, None], loss, error, 'misclassification_indicator', (
        'brier_mean_over_classes' if probability else 'misclassification_rate')


def _compatible(items):
    first = items[0]
    for item in items[1:]:
        if item['data']['protocol_digest'] != first['data']['protocol_digest']:
            raise ValueError('Different validation protocols; common row intersection alone is insufficient')
        for key in ('rows', 'folds', 'targets', 'weights'):
            if not np.array_equal(item[key], first[key]):
                raise ValueError('OOF row/fold/target/weight alignment differs: '+key)
        if (item['data']['sample_weight'] is None) != (first['data']['sample_weight'] is None):
            raise ValueError('OOF weighting policy differs')


def _gain(items, context, ids):
    """A fixed equal mean diagnostic: no weights/meta-model are fitted on these labels."""
    first = items[0]
    kind = first['data']['prediction_kind']
    if kind not in {'value', 'probability'}:
        return dict(status='unavailable', reason='Saved labels cannot reconstruct the runtime probability blend')
    blocks = [i['predictions'] for i in items]
    if kind == 'probability' and context.task.task_type != 'multilabel_classification':
        blocks = [np.column_stack([1-p.ravel(), p.ravel()]) if p.ndim == 1 or p.shape[1] == 1 else p for p in blocks]
    else:
        blocks = [p.reshape(first['targets'].shape) for p in blocks]
    def scores(p):
        values = [score_predictions(first['targets'][mask], p[mask], context,
                      weights=first['weights'][mask] if first['data']['sample_weight'] is not None else None)
                  for mask in (first['folds'] == f for f in np.unique(first['folds']))]
        return dict(mean_fold_score=float(np.mean(values)), fold_scores=values,
            pooled_oof_score=score_predictions(first['targets'], p, context,
                weights=first['weights'] if first['data']['sample_weight'] is not None else None))
    single = {key: scores(p) for key, p in zip(ids, blocks)}
    blend = scores(np.mean(blocks, axis=0))
    minimize = context.task.metric_direction == 'minimize'
    best = (min if minimize else max)(s['mean_fold_score'] for s in single.values())
    gain = (best-blend['mean_fold_score']) if minimize else (blend['mean_fold_score']-best)
    marginal = {}
    if len(items) == 3:
        for j, key in enumerate(ids):
            reduced = scores(np.mean([p for i,p in enumerate(blocks) if i != j], axis=0))
            improvement = reduced['mean_fold_score']-blend['mean_fold_score']
            marginal[key] = dict(without_component=reduced,
                improvement_from_adding_component=improvement if minimize else -improvement)
    return dict(status='measured', recipe='fixed_equal_mean', metric=context.task.primary_metric,
        direction=context.task.metric_direction, components=single, blend=blend, improvement_over_best=gain,
        leave_one_component_out=marginal,
        scope='historical_inner_oof_diagnostic', limitation='Component configurations and composition may have been selected on these folds; this is not independent confirmation of generalization or of weighted/stack recipes')


def diversity_evidence(rows, context, *, error_quantile=.8, methods=(), oof_folds=3):
    """Small requested sets only. Missing/incompatible data annotate, never reject a recipe."""
    if not 2 <= len(rows) <= 3 or not 0 < error_quantile < 1:
        raise ValueError('Compare two or three components; error_quantile must be in (0,1)')
    ids = [r['experiment_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Use distinct component experiments')
    result = dict(status='unverified_hypothesis', component_experiment_ids=ids,
        scope='historical_inner_oof_diagnostic', pairs=[], limitations=[], incremental_benefit=None,
        sources={r['experiment_id']:dict(oof_ref=r.get('run',{}).get('oof_ref'),
            protocol_ref=r.get('run',{}).get('validation_protocol_ref')) for r in rows})
    costs = {r['experiment_id']: component_refit_cost(r) for r in rows}
    known = all(c['samples'] for c in costs.values())
    result['full_refit_cost'] = dict(status='estimated' if known else 'unknown', components=costs,
        historical_component_fit_sum_seconds=sum(c['samples'][-1]['seconds'] for c in costs.values()) if known else None,
        recipes={m: dict(full_component_fits=len(rows), oof_component_fits=len(rows)*oof_folds if m in {'weighted','stack'} else 0,
                        meta_or_weight_search_cost='unknown' if m in {'weighted','stack'} else 'not_applicable') for m in methods},
        scope='Historical component fit sum at recorded rows/resources, excluding persistence and prediction. Not measured full ensemble latency; OOF/meta costs and scheduling overlap differ. Advisory only.')
    loaded, errors = {}, {}
    for row in rows:
        try:
            loaded[row['experiment_id']] = _load(row, context)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            errors[row['experiment_id']] = str(error)[:500]
    for a, b in itertools.combinations(ids, 2):
        pair = dict(component_experiment_ids=[a, b], status='unavailable')
        try:
            if a in errors or b in errors:
                raise ValueError('; '.join(errors[k] for k in (a, b) if k in errors))
            x, z = loaded[a], loaded[b]
            _compatible([x, z])
            ea, la, ma, error_kind, loss_kind = _errors(x, context)
            eb, lb, mb, _, _ = _errors(z, context)
            thresholds = None
            if ma is None:
                qa, qb = np.quantile(la, error_quantile), np.quantile(lb, error_quantile)
                ma, mb = (la > 0) & (la >= qa), (lb > 0) & (lb >= qb)
                thresholds = dict(rule='positive row MAE >= own unweighted quantile, ties included',
                                  quantile=error_quantile, values=[float(qa), float(qb)])
            both, union, w = ma & mb, ma | mb, x['weights']
            pair.update(status='measured', support_rows=len(w), weight_sum=float(w.sum()),
                protocol_digest=x['data']['protocol_digest'],
                dataset_hash=x['data']['dataset_hash'], outer_split_id=x['data']['outer_split_id'],
                fold_count=len(np.unique(x['folds'])), metric=context.task.primary_metric,
                metric_options=context.metric_options, class_order=context.classes, output_scale='original',
                weighting='sample_weight' if x['data']['sample_weight'] is not None else 'unweighted',
                error_correlation=dict(kind=error_kind, per_output=[_correlation(ea[:, j], eb[:, j], w) for j in range(ea.shape[1])]),
                loss_correlation=dict(kind=loss_kind, **_correlation(la, lb, w)),
                error_overlap=dict(thresholds=thresholds, errors_a=int(ma.sum()), errors_b=int(mb.sum()),
                    intersection_rows=int(both.sum()), union_rows=int(union.sum()),
                    weighted_intersection_rate=float(np.average(both, weights=w)),
                    weighted_union_rate=float(np.average(union, weights=w)),
                    weighted_jaccard=float(w[both].sum()/w[union].sum()) if w[union].sum() else None))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            pair['reason'] = str(error)[:700]
        result['pairs'].append(pair)
    try:
        if errors:
            raise ValueError('; '.join(f'{k}: {v}' for k, v in errors.items()))
        items = [loaded[k] for k in ids]
        _compatible(items)
        result['incremental_benefit'] = _gain(items, context, ids)
    except (ValueError, KeyError, TypeError, IndexError) as error:
        result['incremental_benefit'] = dict(status='unavailable', reason=str(error)[:700])
    gain = result['incremental_benefit']
    if all(p['status'] == 'measured' for p in result['pairs']):
        result['status'] = 'measured_diversity'
        if gain['status'] == 'measured':
            result['status'] = 'historical_gain_observed' if gain['improvement_over_best'] > 0 else 'historical_gain_not_observed'
    result['limitations'].append('Historical OOF is diagnostic only; never reused as current meta training data or independent confirmation. Unknown measurements do not block an experiment.')
    return result
