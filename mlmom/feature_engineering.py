"""Fold-fitted Python feature programs and optional declarative shortcuts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from .feature_contracts import FeatureSet, feature_hash


def validate_features(definition, allowed_columns):
    definition = FeatureSet.model_validate(definition.model_dump(mode='python') if isinstance(definition, FeatureSet) else definition)
    allowed = set(allowed_columns)
    for item in definition.features:
        used = set(item.columns) | ({item.by} if item.by is not None else set())
        if not used <= allowed:
            raise ValueError('Feature references forbidden or missing columns: ' + ', '.join(sorted(used - allowed)))
    return definition


class FeatureTransformer:
    def __init__(self, definition):
        self.definition = FeatureSet.model_validate(definition.model_dump(mode='python') if isinstance(definition, FeatureSet) else definition)

    @staticmethod
    def load_program(program):
        import cloudpickle
        import hashlib
        import sys
        import types
        name = 'mlmom_features_' + hashlib.sha256(program.expression_key().encode()).hexdigest()[:24]
        module = types.ModuleType(name)
        module.__file__ = '<feature_program:' + program.name + '>'
        sys.modules[name] = module
        exec(compile(program.source, module.__file__, 'exec'), module.__dict__)
        cloudpickle.register_pickle_by_value(module)
        return module.create_features(dict(program.config))

    def block(self, value, prefix):
        if sparse.issparse(value):
            block = value.tocsr()
        else:
            block = np.asarray(value)
            if block.ndim == 1:
                block = block.reshape(-1, 1)
        if len(block.shape) != 2:
            raise ValueError('Feature program must return a two-dimensional matrix')
        columns = list(map(str, value.columns)) if isinstance(value, pd.DataFrame) else list(map(str, range(block.shape[1])))
        return block, [prefix + '_' + column for column in columns]

    @staticmethod
    def combine(blocks, rows):
        if any(sparse.issparse(block) for block in blocks):
            return sparse.hstack([sparse.csr_matrix(block) for block in blocks], format='csr')
        return np.column_stack(blocks) if blocks else np.empty((rows, 0))

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

    def fit_transform(self, X, y=None, context=None):
        validate_features(self.definition, X.columns)
        self.maps, self.fallbacks, self.fills = {}, {}, {}
        arrays = []
        for item in self.definition.features:
            values = self.numeric(self.values(X, item, fitting=True))
            finite = values.dropna()
            self.fills[item.name] = float(finite.median()) if len(finite) else 0.
            arrays.append(values.fillna(self.fills[item.name]).to_numpy())
        self.program_instances = []
        self.output_names = [f.name for f in self.definition.features]
        blocks = [np.column_stack(arrays)] if arrays else []
        for program in self.definition.programs:
            try:
                instance = self.load_program(program)
                value = instance.fit_transform(X.copy(), y.copy() if hasattr(y, 'copy') else y, context)
                block, names = self.block(value, program.name)
                if block.shape[0] != len(X):
                    raise ValueError('Feature program must preserve rows')
                self.program_instances.append((program.name, instance, names))
                self.output_names.extend(names)
                blocks.append(block)
            except Exception as error:
                raise ValueError(f'Feature program {program.name}: {type(error).__name__}: {error}') from error
        return self.combine(blocks, len(X))

    def transform(self, X):
        validate_features(self.definition, X.columns)
        arrays = [self.numeric(self.values(X, f)).fillna(self.fills[f.name]).to_numpy()
                  for f in self.definition.features]
        blocks = [np.column_stack(arrays)] if arrays else []
        for name, instance, expected_names in getattr(self, 'program_instances', []):
            try:
                block, names = self.block(instance.transform(X.copy()), name)
                if block.shape[0] != len(X) or names != expected_names:
                    raise ValueError('Feature program changed row count or output columns during transform')
                blocks.append(block)
            except Exception as error:
                raise ValueError(f'Feature program {name}: {type(error).__name__}: {error}') from error
        return self.combine(blocks, len(X))


class FeatureAugmentingPreprocessor:
    """Keep the original representation and append numeric columns positionally."""
    def __init__(self, base, definition):
        self.base = base
        self.features = FeatureTransformer(definition)

    def append(self, base, extra):
        if len(base.shape) != 2 or base.shape[0] != extra.shape[0]:
            raise ValueError('Feature preprocessing must preserve rows')
        self.base_width = base.shape[1]
        self.output_width = self.base_width + extra.shape[1]
        if sparse.issparse(base) or sparse.issparse(extra):
            return sparse.hstack([base, sparse.csr_matrix(extra)], format='csr')
        if isinstance(base, pd.DataFrame):
            names = ['__fe_' + name for name in getattr(self.features, 'output_names', [f.name for f in self.features.definition.features])]
            if set(names) & set(base.columns):
                raise ValueError('Engineered feature name collides with base output')
            addition = pd.DataFrame(extra, columns=names, index=base.index)
            return pd.concat([base, addition], axis=1)
        return np.concatenate([np.asarray(base), extra], axis=1)

    def fit_transform(self, X, y, context):
        allowed = getattr(context.task_context, 'feature_columns', X.columns)
        validate_features(self.features.definition, allowed)
        from dataclasses import replace
        feature_context = replace(context, validation_X=None, validation_y=None, validation_groups=None, validation_weight=None)
        extra = self.features.fit_transform(X, y, feature_context)
        return self.append(self.base.fit_transform(X, y, context), extra)

    def transform(self, X):
        return self.append(self.base.transform(X), self.features.transform(X))

    def manifest(self):
        return dict(feature_set_hash=feature_hash(self.features.definition),
                    names=getattr(self.features, 'output_names', [f.name for f in self.features.definition.features]),
                    base_width=self.base_width, output_width=self.output_width)
