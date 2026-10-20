"""Reproducible outer and inner splits with group and multilabel constraints."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
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
