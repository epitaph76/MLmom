"""Resolve organizer inputs without trusting paths embedded in task prose."""
from __future__ import annotations

import json
import re
from pathlib import Path
import pandas as pd
from .pipeline_contracts import InputFiles, TaskContext
from .persistence import file_hash


def inside(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path escapes the permitted workspace')
    return path


def resolve_inputs(task_path: Path) -> InputFiles:
    task_path = task_path.resolve(strict=True)
    root = task_path.parent
    text = task_path.read_text(encoding='utf-8-sig')
    manifest_path = root / 'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig')) if manifest_path.exists() else {}

    def find(role: str, default: str, optional=False) -> Path | None:
        if role in manifest:
            path = inside(root, manifest[role])
            if not path.is_file():
                raise FileNotFoundError(f'Manifest {role}: {path}')
            return path
        # Prefer existing explicit task paths, then a sibling, then a unique basename.
        mentioned = re.findall(r'`([^`\n]+\.(?:csv|tsv))`', text)
        exact = {inside(root, name) for name in mentioned
                 if ('/' in name or '\\' in name) and Path(name).name.lower() == default and inside(root, name).is_file()}
        if len(exact) == 1:
            return exact.pop()
        if len(exact) > 1:
            raise ValueError(f'Ambiguous {role} paths in task')
        sibling = root / default
        if sibling.is_file():
            return sibling
        matches = [p for p in root.rglob(default) if p.is_file()]
        if len(matches) == 1:
            return matches[0].resolve()
        if not matches and optional:
            return None
        raise ValueError(f'Expected a unique {role}, found {len(matches)}')

    train = find('train', 'train.csv')
    test = find('test', 'test.csv')
    sample = find('sample_submission', 'sample_submission.csv', True)
    dictionaries = [inside(root, value) for value in manifest.get('dictionaries', [])]
    if not dictionaries:
        dictionary = find('dictionary', 'data_dictionary.csv', True)
        dictionaries = [dictionary] if dictionary else []
    if any(not p.is_file() for p in dictionaries):
        raise FileNotFoundError('A manifest dictionary is missing')
    return InputFiles(task=str(task_path), root=str(root), train=str(train), test=str(test),
                      dictionaries=[str(p) for p in dictionaries],
                      sample_submission=str(sample) if sample else None,
                      manifest=str(manifest_path) if manifest_path.exists() else None)


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep='\t' if Path(path).suffix.lower() == '.tsv' else ',',
                       encoding='utf-8-sig', **kwargs)


def validate_tables(context: TaskContext) -> None:
    train = read_table(context.files.train, nrows=0)
    test = read_table(context.files.test, nrows=0)
    required = context.feature_columns + context.task.id_columns + context.task.group_columns
    missing = set(required) - set(test.columns)
    if missing:
        raise ValueError(f'Test is missing required columns: {sorted(missing)}')
    if not set(context.task.target_columns) <= set(train.columns):
        raise ValueError('Train is missing targets')
    if file_hash(Path(context.files.train)) != context.dataset_hash:
        raise ValueError('Training data changed after inspection')


def features(frame: pd.DataFrame, context: TaskContext) -> pd.DataFrame:
    return frame.loc[:, context.feature_columns]


def targets(frame: pd.DataFrame, context: TaskContext):
    if not context.task.target_columns:
        return None
    y = frame[context.task.target_columns]
    if y.isna().any().any():
        raise ValueError('Missing target values require an explicit data policy')
    return y.iloc[:, 0] if y.shape[1] == 1 else y
