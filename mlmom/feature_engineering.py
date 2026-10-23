"""Trusted single-table feature operators. No target, code execution or batch fitting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from .feature_contracts import FeatureSet, feature_hash


def validate_features(definition, allowed_columns):
    definition = FeatureSet.model_validate(definition)
    allowed = set(allowed_columns)
    for item in definition.features:
        used = set(item.columns) | ({item.by} if item.by is not None else set())
        if not used <= allowed:
            raise ValueError('Feature references forbidden or missing columns: ' + ', '.join(sorted(used - allowed)))
    return definition


class FeatureTransformer:
    def __init__(self, definition):
        self.definition = FeatureSet.model_validate(definition)

    @staticmethod
    def numeric(series):
        return pd.to_numeric(series, errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan)

    def values(self, X, item, fitting=False):
        column = X[item.columns[0]] if item.columns else None
        op = item.op
        if op == 'missing':
            return column.isna().astype(float)
        if op == 'frequency':
            key = column.astype(object)
            if fitting:
                self.maps[item.name] = key.value_counts(dropna=False, normalize=True)
            return key.map(self.maps[item.name]).fillna(0).astype(float)
        if op.startswith('group_'):
            key = X[item.by].astype(object)
            if fitting:
                if op == 'group_count':
                    mapping = key.value_counts(dropna=False)
                    fallback = 0.
                else:
                    numeric = self.numeric(column)
                    grouped = numeric.groupby(key, dropna=False, observed=True)
                    stat = op.split('_', 1)[1]
                    mapping = grouped.std(ddof=0) if stat == 'std' else getattr(grouped, stat)()
                    fallback = float(numeric.std(ddof=0) if stat == 'std' else getattr(numeric, stat)())
                self.maps[item.name] = mapping
                self.fallbacks[item.name] = fallback
            return key.map(self.maps[item.name]).astype(float).fillna(self.fallbacks[item.name])
        if op == 'date_part':
            dates = pd.to_datetime(column, format='mixed', errors='coerce', utc=True)
            return getattr(dates.dt, item.part).astype(float)
        left = self.numeric(column)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            if op == 'log':
                return np.log(left.where(left > 0))
            right = self.numeric(X[item.columns[1]])
            if op == 'ratio':
                return left / right.where(right != 0)
            if op == 'difference':
                return left - right
            if op == 'product':
                return left * right
        raise ValueError('Unsupported feature operator')

    def fit_transform(self, X):
        validate_features(self.definition, X.columns)
        self.maps, self.fallbacks, self.fills = {}, {}, {}
        arrays = []
        for item in self.definition.features:
            values = self.numeric(self.values(X, item, fitting=True))
            finite = values.dropna()
            self.fills[item.name] = float(finite.median()) if len(finite) else 0.
            arrays.append(values.fillna(self.fills[item.name]).to_numpy())
        return np.column_stack(arrays) if arrays else np.empty((len(X), 0))

    def transform(self, X):
        validate_features(self.definition, X.columns)
        arrays = [self.numeric(self.values(X, f)).fillna(self.fills[f.name]).to_numpy()
                  for f in self.definition.features]
        return np.column_stack(arrays) if arrays else np.empty((len(X), 0))


class FeatureAugmentingPreprocessor:
    """Keep the original representation and append numeric columns positionally."""
    def __init__(self, base, definition):
        self.base = base
        self.features = FeatureTransformer(definition)

    def append(self, base, extra):
        if len(base.shape) != 2 or base.shape[0] != len(extra):
            raise ValueError('Feature preprocessing must preserve rows')
        self.base_width = base.shape[1]
        self.output_width = self.base_width + extra.shape[1]
        if sparse.issparse(base):
            return sparse.hstack([base, sparse.csr_matrix(extra)], format='csr')
        if isinstance(base, pd.DataFrame):
            names = ['__fe_' + f.name for f in self.features.definition.features]
            if set(names) & set(base.columns):
                raise ValueError('Engineered feature name collides with base output')
            addition = pd.DataFrame(extra, columns=names, index=base.index)
            return pd.concat([base, addition], axis=1)
        return np.concatenate([np.asarray(base), extra], axis=1)

    def fit_transform(self, X, y, context):
        allowed = getattr(context.task_context, 'feature_columns', X.columns)
        validate_features(self.features.definition, allowed)
        extra = self.features.fit_transform(X)
        return self.append(self.base.fit_transform(X, y, context), extra)

    def transform(self, X):
        return self.append(self.base.transform(X), self.features.transform(X))

    def manifest(self):
        return dict(feature_set_hash=feature_hash(self.features.definition),
                    names=[f.name for f in self.features.definition.features],
                    base_width=self.base_width, output_width=self.output_width)


def screen_features(definition, X, folds=None):
    definition = validate_features(definition, X.columns)
    reasons = {f.name: [] for f in definition.features}
    for indices in ([f['train_indices'] for f in folds] if folds else [list(range(len(X)))]):
        train = X.iloc[indices]
        values = FeatureTransformer(definition).fit_transform(train)
        seen = []
        for i, item in enumerate(definition.features):
            column = values[:, i]
            reason = None
            if not np.isfinite(column).all():
                raise ValueError('Nonfinite feature: ' + item.name)
            if len(np.unique(column)) <= 1:
                reason = 'Constant feature: ' + item.name
            elif any(np.array_equal(column, other) for other in seen):
                reason = 'Duplicate feature values: ' + item.name
            elif any(np.array_equal(column, pd.to_numeric(train[c], errors='coerce').to_numpy(dtype=float), equal_nan=True)
                     for c in train.select_dtypes(include=['number', 'bool']).columns):
                reason = 'Feature duplicates an input column: ' + item.name
            reasons[item.name].append(reason)
            seen.append(column)
    for values in reasons.values():
        if values and all(values):
            raise ValueError(values[0])
    return definition
