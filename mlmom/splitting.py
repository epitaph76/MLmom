"""Reproducible outer and inner splits with group and multilabel constraints."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit, KFold, StratifiedKFold, GroupKFold
from .pipeline_contracts import SplitSpec, TaskContext
from .inputs import targets


def group_keys(frame: pd.DataFrame, columns: list[str]):
    if not columns:
        return None
    if frame[columns].isna().any().any():
        raise ValueError('Group identifiers contain missing values')
    return pd.util.hash_pandas_object(frame[columns], index=False).to_numpy().astype(str)


def split_indices(frame: pd.DataFrame, context: TaskContext, fraction: float, seed: int):
    if context.task.time_column or context.task.task_type == 'time_series_forecasting':
        raise ValueError('deferred_time_series_support')
    groups = group_keys(frame, context.task.group_columns)
    index = np.arange(len(frame))
    y = targets(frame, context)
    if groups is not None:
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(index, groups=groups))
        strategy = 'group_shuffle'
    elif context.task.task_type == 'multilabel_classification':
        labels = np.asarray(y, dtype=int)
        if not np.isin(labels, [0, 1]).all():
            raise ValueError('Multilabel targets must be binary indicators')
        # Iterative rare-label assignment. Singleton positives remain in train.
        rng = np.random.default_rng(seed)
        wanted = max(1, round(len(frame)*fraction))
        desired = labels.sum(axis=0)*fraction
        protected = np.any((labels == 1) & (labels.sum(axis=0) <= 1), axis=1)
        available = ~protected
        chosen = []
        remaining = labels.copy()
        while len(chosen) < wanted and available.any():
            counts = remaining[available].sum(axis=0)
            eligible = np.flatnonzero((counts > 0) & (desired > 0))
            if len(eligible):
                label = eligible[np.argmin(counts[eligible])]
                candidates = np.flatnonzero(available & (labels[:, label] == 1))
                gains = labels[candidates] @ np.maximum(desired, 0)
                candidates = candidates[gains == gains.max()]
            else:
                candidates = np.flatnonzero(available)
            item = int(rng.choice(candidates))
            # Retain at least one positive of every observed label in train.
            if np.any(((labels.sum(axis=0)-labels[chosen].sum(axis=0)-labels[item]) < 1) & (labels.sum(axis=0)>0)):
                available[item] = False
                continue
            chosen.append(item)
            available[item] = False
            desired -= labels[item]
        if len(chosen) < wanted:
            raise ValueError('Insufficient multilabel coverage for the requested split')
        b = np.array(chosen)
        a = np.setdiff1d(index, b)
        strategy = 'iterative_multilabel'
    else:
        stratify = y if context.task.task_type in {'binary_classification','multiclass_classification'} else None
        a, b = train_test_split(index, test_size=fraction, random_state=seed, stratify=stratify)
        strategy = 'stratified_shuffle' if stratify is not None else 'shuffle'
    if y is not None and context.task.task_type in {'binary_classification','multiclass_classification'}:
        if set(np.asarray(y)[a]) != set(np.asarray(y)) or set(np.asarray(y)[b]) != set(np.asarray(y)):
            raise ValueError('Each classification split must contain every class')
    return np.asarray(a), np.asarray(b), strategy


def make_split(frame: pd.DataFrame, context: TaskContext, round_id: str, fraction: float, seed: int):
    a, b, strategy = split_indices(frame, context, fraction, seed)
    digest = hashlib.sha256(a.tobytes()+b.tobytes()+context.dataset_hash.encode()).hexdigest()[:16]
    return SplitSpec(split_id=f'{round_id}-{digest}', round_id=round_id, dataset_hash=context.dataset_hash,
                     seed=seed, strategy=strategy, train_indices=a.tolist(), holdout_indices=b.tolist())


def inner_protocol(frame, context, *, fraction=.2, seed=43, outer_split_id=None, source_indices=None):
    """Indices are positional, independent of pandas labels and user ID columns."""
    strategy = context.task.validation_strategy.value
    count = context.task.validation_folds
    index = np.arange(len(frame))
    source = np.asarray(source_indices if source_indices is not None else index, dtype=int)
    if len(source) != len(frame) or len(np.unique(source)) != len(source):
        raise ValueError('Invalid outer training index mapping')
    groups = group_keys(frame, context.task.group_columns)
    y = targets(frame, context)
    classification = context.task.task_type in {'binary_classification', 'multiclass_classification'}
    if groups is not None and strategy not in {'holdout', 'group_kfold'}:
        raise ValueError('Grouped data requires group_kfold or a group-aware holdout')
    if strategy == 'holdout':
        a, b, implementation = split_indices(frame, context, fraction, seed)
        pairs = [(a, b)]
    else:
        if len(frame) < count:
            raise ValueError('Insufficient rows for validation_folds')
        if strategy == 'group_kfold':
            if groups is None or len(np.unique(groups)) < count:
                raise ValueError('Insufficient groups for validation_folds')
            pairs = GroupKFold(n_splits=count).split(index, y, groups)
        elif strategy == 'stratified_kfold':
            if not classification:
                raise ValueError('StratifiedKFold requires a single classification target')
            if pd.Series(np.asarray(y)).value_counts().min() < count:
                raise ValueError('Insufficient examples of a rare class for validation_folds')
            pairs = StratifiedKFold(n_splits=count, shuffle=True, random_state=seed).split(index, y)
        elif strategy == 'kfold':
            if classification:
                raise ValueError('Classification requires stratified_kfold or holdout')
            pairs = KFold(n_splits=count, shuffle=True, random_state=seed).split(index)
        else:
            raise ValueError(f'Unsupported internal validation strategy: {strategy}')
        pairs = list(pairs)
        implementation = strategy
    folds = []
    all_classes = set(np.asarray(y)) if classification else None
    for number, (a, b) in enumerate(pairs):
        if not len(a) or not len(b) or np.intersect1d(a, b).size:
            raise ValueError('Invalid internal fold')
        if groups is not None and set(groups[a]) & set(groups[b]):
            raise ValueError('Group leakage in internal fold')
        if classification and (set(np.asarray(y)[a]) != all_classes or set(np.asarray(y)[b]) != all_classes):
            raise ValueError(f'Each classification fold must contain every class (fold {number})')
        folds.append(dict(number=number, train_indices=a.tolist(), validation_indices=b.tolist(),
                          source_train_indices=source[a].tolist(), source_validation_indices=source[b].tolist()))
    return dict(version=1, strategy=strategy, implementation=implementation, seed=seed,
                n_splits=len(folds), validation_folds=count if strategy != 'holdout' else None,
                fraction=fraction if strategy == 'holdout' else None, aggregation='mean_fold_score',
                std_ddof=0, dataset_hash=context.dataset_hash, outer_split_id=outer_split_id, folds=folds)
