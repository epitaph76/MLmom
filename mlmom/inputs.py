"""Resolve organizer inputs without trusting paths embedded in task prose."""
from __future__ import annotations

import json
import re
import csv
from pathlib import Path
import pandas as pd
from .pipeline_contracts import InputFiles, TaskContext
from .persistence import file_hash
from .contracts import TableReadOptions


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


def detect_table_format(path: str | Path, *, encoding: str = 'utf-8-sig') -> TableReadOptions:
    """Read a bounded text sample; never load an entire CSV to sniff its header."""
    with Path(path).open(encoding=encoding, newline='') as stream:
        sample = stream.read(16_384)
    if Path(path).suffix.lower() == '.tsv':
        delimiter = '\t'
    else:
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=',;\t|').delimiter
        except csv.Error:
            try:
                delimiter = csv.Sniffer().sniff(sample.splitlines()[0], delimiters=',;\t|').delimiter
            except (csv.Error, IndexError):
                delimiter = ','
    return TableReadOptions(encoding=encoding, delimiter=delimiter)


def read_table(path: str | Path, *, table_format: TableReadOptions | dict | None = None, **kwargs) -> pd.DataFrame:
    """Shared reader for inspection, execution and submission templates.

    Persisted options are authoritative. Older callers/profiles detect the same
    supported encodings and delimiters on demand.
    """
    kwargs.setdefault('low_memory', False)
    if table_format is not None:
        options = TableReadOptions.model_validate(table_format)
        frame = pd.read_csv(path, sep=options.delimiter, encoding=options.encoding, **kwargs)
        frame.attrs['table_format'] = options.model_dump()
        return frame
    last_error = None
    for encoding in ('utf-8-sig', 'cp1251'):
        try:
            options = detect_table_format(path, encoding=encoding)
            return read_table(path, table_format=options, **kwargs)
        except UnicodeDecodeError as error:
            last_error = error
    raise ValueError(f'Could not decode table {path}: {last_error}')


def validate_tables(context: TaskContext) -> None:
    train = read_table(context.files.train, table_format=context.table_formats.get(context.files.train), nrows=0)
    test = read_table(context.files.test, table_format=context.table_formats.get(context.files.test), nrows=0)
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
