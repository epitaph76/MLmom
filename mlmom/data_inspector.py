"""Universal deterministic EDA for tabular AutoML tasks.

The inspector accepts a directory, optional manifest, explicit CLI overrides,
column dictionaries, and free-form task text. It profiles the data without
modifying source files and produces a versioned JSON contract plus a bounded
context for an LLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from .contracts import (
    ColumnProfile,
    ColumnRole,
    DataIssue,
    DatasetChecks,
    DatasetProfile,
    DistributionBucket,
    FileRole,
    InferenceConfidence,
    MetricDirection,
    SemanticType,
    Severity,
    SplitColumnStats,
    TableProfile,
    TaskSpec,
    TaskType,
)


_DATA_SUFFIXES = {".csv", ".tsv"}
_TEXT_SUFFIXES = {".md", ".txt"}
_NAME_ALIASES = {"feature", "column", "column_name", "name", "field", "поле", "столбец"}
_ROLE_ALIASES = {"role", "column_role", "роль"}
_TYPE_ALIASES = {"dtype", "type", "data_type", "тип"}
_DESCRIPTION_ALIASES = {"description", "description_ru", "описание"}
_UNIT_ALIASES = {"unit", "units", "единица", "единицы"}


@dataclass
class InputLayout:
    root: Path
    dataset_name: str
    train_path: Path
    test_path: Path | None = None
    sample_path: Path | None = None
    dictionary_paths: list[Path] = field(default_factory=list)
    task_path: Path | None = None
    auxiliary_paths: list[Path] = field(default_factory=list)
    target_columns: list[str] = field(default_factory=list)
    id_columns: list[str] = field(default_factory=list)
    task_type: TaskType | None = None
    primary_metric: str | None = None
    time_column: str | None = None
    group_columns: list[str] = field(default_factory=list)
    weight_column: str | None = None


def _finite_float(value: Any, digits: int = 6) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _display_value(value: Any, max_length: int = 100) -> str:
    text = "<missing>" if pd.isna(value) else str(value)
    return text if len(text) <= max_length else text[: max_length - 1] + "…"


def _format_number(value: float) -> str:
    if abs(value) >= 1_000_000 or (value and abs(value) < 0.001):
        return f"{value:.3g}"
    if float(value).is_integer():
        return f"{value:.0f}"
    return f"{value:.4g}"


def _normalise_name(name: str) -> str:
    return re.sub(r"[^a-zа-я0-9]+", "_", name.lower()).strip("_")


def _resolve_path(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


class DataInspector:
    """Inspect supervised or unsupervised tabular datasets."""

    def __init__(
        self,
        *,
        top_values: int = 6,
        histogram_bins: int = 6,
        high_missing_threshold: float = 0.40,
        shift_warning_threshold: float = 0.25,
    ) -> None:
        if top_values < 1 or histogram_bins < 3:
            raise ValueError("top_values must be positive and histogram_bins must be at least 3")
        self.top_values = top_values
        self.histogram_bins = histogram_bins
        self.high_missing_threshold = high_missing_threshold
        self.shift_warning_threshold = shift_warning_threshold

    def inspect_directory(
        self,
        data_dir: str | Path,
        dataset_name: str | None = None,
        *,
        manifest_path: str | Path | None = None,
        train_path: str | Path | None = None,
        test_path: str | Path | None = None,
        sample_submission_path: str | Path | None = None,
        dictionary_paths: Iterable[str | Path] = (),
        task_description_path: str | Path | None = None,
        target_columns: Iterable[str] = (),
        id_columns: Iterable[str] = (),
        task_type: TaskType | str | None = None,
        primary_metric: str | None = None,
        time_column: str | None = None,
        group_columns: Iterable[str] = (),
        weight_column: str | None = None,
    ) -> DatasetProfile:
        root = Path(data_dir).resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)
        manifest = self._load_manifest(root, manifest_path)
        layout = self._discover_layout(root, manifest)

        layout.dataset_name = dataset_name or manifest.get("dataset_name") or root.name
        layout.train_path = _resolve_path(root, train_path) or layout.train_path
        layout.test_path = _resolve_path(root, test_path) if test_path else layout.test_path
        layout.sample_path = (
            _resolve_path(root, sample_submission_path)
            if sample_submission_path
            else layout.sample_path
        )
        explicit_dictionaries = [_resolve_path(root, path) for path in dictionary_paths]
        if explicit_dictionaries:
            layout.dictionary_paths = [path for path in explicit_dictionaries if path is not None]
        layout.task_path = (
            _resolve_path(root, task_description_path)
            if task_description_path
            else layout.task_path
        )
        layout.target_columns = list(target_columns) or list(manifest.get("target_columns", []))
        layout.id_columns = list(id_columns) or list(manifest.get("id_columns", []))
        selected_task_type = task_type or manifest.get("task_type")
        if selected_task_type:
            layout.task_type = TaskType(selected_task_type)
        layout.primary_metric = primary_metric or manifest.get("primary_metric")
        layout.time_column = time_column or manifest.get("time_column")
        layout.group_columns = list(group_columns) or list(manifest.get("group_columns", []))
        layout.weight_column = weight_column or manifest.get("weight_column")
        return self._inspect_layout(layout)

    def inspect(
        self,
        *,
        train_path: str | Path,
        test_path: str | Path | None = None,
        dictionary_path: str | Path | None = None,
        sample_submission_path: str | Path | None = None,
        task_description_path: str | Path | None = None,
        dataset_name: str = "dataset",
        target_columns: Iterable[str] = (),
        id_columns: Iterable[str] = (),
        task_type: TaskType | str | None = None,
        primary_metric: str | None = None,
        time_column: str | None = None,
        group_columns: Iterable[str] = (),
        weight_column: str | None = None,
    ) -> DatasetProfile:
        train = Path(train_path).resolve()
        layout = InputLayout(
            root=train.parent,
            dataset_name=dataset_name,
            train_path=train,
            test_path=Path(test_path).resolve() if test_path else None,
            sample_path=Path(sample_submission_path).resolve() if sample_submission_path else None,
            dictionary_paths=[Path(dictionary_path).resolve()] if dictionary_path else [],
            task_path=Path(task_description_path).resolve() if task_description_path else None,
            target_columns=list(target_columns),
            id_columns=list(id_columns),
            task_type=TaskType(task_type) if task_type else None,
            primary_metric=primary_metric,
            time_column=time_column,
            group_columns=list(group_columns),
            weight_column=weight_column,
        )
        return self._inspect_layout(layout)

    @staticmethod
    def _load_manifest(root: Path, manifest_path: str | Path | None) -> dict[str, Any]:
        candidate = _resolve_path(root, manifest_path)
        if candidate is None:
            automatic = root / "dataset_manifest.json"
            candidate = automatic if automatic.is_file() else None
        if candidate is None:
            return {}
        return json.loads(candidate.read_text(encoding="utf-8"))

    def _discover_layout(self, root: Path, manifest: dict[str, Any]) -> InputLayout:
        files = sorted(path for path in root.rglob("*") if path.is_file())
        data_files = [path for path in files if path.suffix.lower() in _DATA_SUFFIXES]
        if not data_files:
            raise FileNotFoundError(f"No CSV or TSV files found under {root}")

        manifest_train = _resolve_path(root, manifest.get("train"))
        manifest_test = _resolve_path(root, manifest.get("test"))
        manifest_sample = _resolve_path(root, manifest.get("sample_submission"))
        manifest_dicts = [
            path for value in manifest.get("dictionaries", [])
            if (path := _resolve_path(root, value)) is not None
        ]
        manifest_task = _resolve_path(root, manifest.get("task_description"))
        manifest_aux = [
            path for value in manifest.get("auxiliary_tables", [])
            if (path := _resolve_path(root, value)) is not None
        ]

        dictionaries = manifest_dicts or [path for path in data_files if self._looks_like_dictionary(path)]
        sample = manifest_sample or self._first_named(data_files, ("sample_submission", "submission_sample", "submission"))
        train = manifest_train or self._first_named(data_files, ("train", "training", "обуч"))
        test = manifest_test or self._first_named(data_files, ("test", "holdout", "scoring", "тест"))

        excluded = set(dictionaries)
        if sample:
            excluded.add(sample)
        candidates = [path for path in data_files if path not in excluded]
        if train is None or test is None:
            inferred_train, inferred_test = self._infer_train_test_pair(candidates)
            train = train or inferred_train
            test = test or inferred_test
        if train is None and len(candidates) == 1:
            train = candidates[0]
        if train is None:
            raise ValueError("Could not identify a training table. Use --train or dataset_manifest.json.")

        task = manifest_task or self._first_task_text(files)
        used = {train, *dictionaries}
        used.update(path for path in (test, sample) if path is not None)
        auxiliary = manifest_aux or [path for path in data_files if path not in used]
        return InputLayout(
            root=root,
            dataset_name=str(manifest.get("dataset_name") or root.name),
            train_path=train,
            test_path=test,
            sample_path=sample,
            dictionary_paths=dictionaries,
            task_path=task,
            auxiliary_paths=auxiliary,
        )

    @staticmethod
    def _first_named(paths: Iterable[Path], tokens: Iterable[str]) -> Path | None:
        token_list = tuple(token.lower() for token in tokens)
        return next((path for path in paths if any(token in path.stem.lower() for token in token_list)), None)

    @staticmethod
    def _first_task_text(paths: Iterable[Path]) -> Path | None:
        tokens = ("task", "problem", "description", "услов", "задач")
        return next(
            (
                path for path in paths
                if path.suffix.lower() in _TEXT_SUFFIXES
                and any(token in path.stem.lower() for token in tokens)
            ),
            None,
        )

    def _looks_like_dictionary(self, path: Path) -> bool:
        if any(token in path.stem.lower() for token in ("dictionary", "schema", "columns", "словар")):
            return True
        try:
            columns = {_normalise_name(column) for column in self._peek_columns(path)}
        except Exception:
            return False
        has_name = bool(columns & _NAME_ALIASES)
        has_metadata = bool(columns & (_ROLE_ALIASES | _TYPE_ALIASES | _DESCRIPTION_ALIASES))
        return has_name and has_metadata

    def _infer_train_test_pair(self, candidates: list[Path]) -> tuple[Path | None, Path | None]:
        if not candidates:
            return None, None
        if len(candidates) == 1:
            return candidates[0], None
        headers = {path: self._peek_columns(path) for path in candidates}
        best: tuple[float, Path, Path] | None = None
        for possible_train in candidates:
            train_columns = headers[possible_train]
            train_set = set(train_columns)
            for possible_test in candidates:
                if possible_train == possible_test:
                    continue
                test_columns = headers[possible_test]
                test_set = set(test_columns)
                extra = train_set - test_set
                if not extra or len(extra) > max(10, len(train_set) // 4):
                    continue
                overlap = len(train_set & test_set) / max(len(train_set | test_set), 1)
                order_bonus = 0.1 if [c for c in train_columns if c in test_set] == test_columns else 0.0
                score = overlap + order_bonus
                if best is None or score > best[0]:
                    best = (score, possible_train, possible_test)
        return (best[1], best[2]) if best else (candidates[0], None)

    def _peek_columns(self, path: Path) -> list[str]:
        return list(self._read_table(path, nrows=0).columns)

    @staticmethod
    def _sniff_delimiter(path: Path, encoding: str) -> str:
        if path.suffix.lower() == ".tsv":
            return "\t"
        sample = path.read_text(encoding=encoding, errors="replace")[:16_384]
        try:
            return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            return ","

    def _read_table(self, path: Path, *, nrows: int | None = None) -> pd.DataFrame:
        if not path.is_file():
            raise FileNotFoundError(path)
        errors: list[str] = []
        for encoding in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                delimiter = self._sniff_delimiter(path, encoding)
                return pd.read_csv(path, sep=delimiter, encoding=encoding, nrows=nrows, low_memory=False)
            except (UnicodeDecodeError, pd.errors.ParserError) as error:
                errors.append(f"{encoding}: {error}")
        raise ValueError(f"Could not read {path}: {'; '.join(errors)}")

    def _read_task_text(self, path: Path | None) -> str | None:
        if path is None or not path.is_file():
            return None
        for encoding in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                return path.read_text(encoding=encoding)[:20_000]
            except UnicodeDecodeError:
                continue
        return None

    def _inspect_layout(self, layout: InputLayout) -> DatasetProfile:
        train = self._read_table(layout.train_path)
        test = self._read_table(layout.test_path) if layout.test_path else None
        sample = self._read_table(layout.sample_path) if layout.sample_path else None
        task_text = self._read_task_text(layout.task_path)
        metadata = self._read_dictionaries(layout.dictionary_paths)

        targets = self._infer_targets(train, test, sample, metadata, layout.target_columns)
        ids = self._infer_ids(train, test, sample, metadata, layout.id_columns, targets)
        time_column = self._infer_time_column(train, metadata, layout.time_column, targets, ids)
        groups = self._infer_role_columns(metadata, ColumnRole.GROUP, train.columns, layout.group_columns)
        weight = layout.weight_column or self._first_role_column(metadata, ColumnRole.WEIGHT, train.columns)
        task = self._infer_task(
            train=train,
            targets=targets,
            ids=ids,
            time_column=time_column,
            groups=groups,
            weight_column=weight,
            metadata=metadata,
            task_text=task_text,
            explicit_type=layout.task_type,
            explicit_metric=layout.primary_metric,
        )

        columns: list[ColumnProfile] = []
        issues: list[DataIssue] = []
        for name in train.columns:
            role = self._column_role(name, task, metadata)
            test_series = test[name] if test is not None and name in test.columns else None
            profile = self._profile_column(
                name=name,
                role=role,
                train_series=train[name],
                test_series=test_series,
                metadata=metadata.get(name, {}),
            )
            columns.append(profile)
            issues.extend(self._column_issues(profile))

        checks = self._build_checks(train, test, sample, metadata, task)
        issues.extend(self._check_issues(checks))
        target_summaries = {
            target: self._target_summary(train[target], task, columns)
            for target in targets
        }
        issues.extend(self._target_issues(train, task, target_summaries))
        issues.extend(self._leakage_issues(train, task, columns))
        issues.extend(self._task_definition_issues(task))
        if task.task_type == TaskType.UNKNOWN or task.inference_confidence == InferenceConfidence.LOW:
            issues.append(DataIssue(
                code="ambiguous_task_definition",
                severity=Severity.ERROR if not task.target_columns else Severity.WARNING,
                message="Task type or target is ambiguous. Supply explicit hints in dataset_manifest.json.",
                columns=task.target_columns,
            ))

        files = self._profile_files(layout, train, test, sample)
        hints = self._preprocessing_hints(columns, task, issues, checks)
        notes = self._relationship_notes(layout, train, files)
        return DatasetProfile(
            dataset_name=layout.dataset_name,
            root_path=str(layout.root),
            task_description=task_text,
            files=files,
            task=task,
            columns=columns,
            target_summaries=target_summaries,
            checks=checks,
            issues=issues,
            preprocessing_hints=hints,
            report_notes=notes,
        )

    def _read_dictionaries(self, paths: Iterable[Path]) -> dict[str, dict[str, str | None]]:
        result: dict[str, dict[str, str | None]] = {}
        for path in paths:
            frame = self._read_table(path)
            normalised = {_normalise_name(column): column for column in frame.columns}
            name_column = next((normalised[name] for name in _NAME_ALIASES if name in normalised), None)
            if name_column is None:
                continue
            role_column = next((normalised[name] for name in _ROLE_ALIASES if name in normalised), None)
            type_column = next((normalised[name] for name in _TYPE_ALIASES if name in normalised), None)
            description_column = next((normalised[name] for name in _DESCRIPTION_ALIASES if name in normalised), None)
            unit_column = next((normalised[name] for name in _UNIT_ALIASES if name in normalised), None)
            for row in frame.to_dict(orient="records"):
                name = str(row[name_column])
                result[name] = {
                    "role": None if role_column is None or pd.isna(row[role_column]) else str(row[role_column]),
                    "dtype": None if type_column is None or pd.isna(row[type_column]) else str(row[type_column]),
                    "description": None if description_column is None or pd.isna(row[description_column]) else str(row[description_column]),
                    "unit": None if unit_column is None or pd.isna(row[unit_column]) else str(row[unit_column]),
                }
        return result

    @staticmethod
    def _normalise_role(value: str | None) -> ColumnRole | None:
        if not value:
            return None
        normalised = _normalise_name(value)
        aliases = {
            "id": ColumnRole.ID,
            "identifier": ColumnRole.ID,
            "идентификатор": ColumnRole.ID,
            "target": ColumnRole.TARGET,
            "label": ColumnRole.TARGET,
            "цель": ColumnRole.TARGET,
            "feature": ColumnRole.FEATURE,
            "признак": ColumnRole.FEATURE,
            "time": ColumnRole.TIME,
            "date": ColumnRole.TIME,
            "datetime": ColumnRole.TIME,
            "время": ColumnRole.TIME,
            "group": ColumnRole.GROUP,
            "group_id": ColumnRole.GROUP,
            "группа": ColumnRole.GROUP,
            "weight": ColumnRole.WEIGHT,
            "sample_weight": ColumnRole.WEIGHT,
            "вес": ColumnRole.WEIGHT,
            "ignore": ColumnRole.IGNORED,
            "ignored": ColumnRole.IGNORED,
        }
        return aliases.get(normalised)

    def _infer_targets(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame | None,
        sample: pd.DataFrame | None,
        metadata: dict[str, dict[str, str | None]],
        explicit: list[str],
    ) -> list[str]:
        if explicit:
            self._validate_columns(explicit, train, "target")
            return explicit
        declared = [name for name, meta in metadata.items() if self._normalise_role(meta.get("role")) == ColumnRole.TARGET]
        declared = [name for name in declared if name in train.columns]
        if declared:
            return declared
        if test is not None:
            train_only = [
                name for name in train.columns
                if name not in test.columns
                and self._normalise_role(metadata.get(name, {}).get("role"))
                not in {ColumnRole.WEIGHT, ColumnRole.IGNORED}
            ]
            if train_only:
                return train_only
        if sample is not None:
            outputs = [name for name in sample.columns if (test is None or name not in test.columns) and name in train.columns]
            if outputs:
                return outputs
        return []

    def _infer_ids(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame | None,
        sample: pd.DataFrame | None,
        metadata: dict[str, dict[str, str | None]],
        explicit: list[str],
        targets: list[str],
    ) -> list[str]:
        if explicit:
            self._validate_columns(explicit, train, "ID")
            return explicit
        declared = [name for name, meta in metadata.items() if self._normalise_role(meta.get("role")) == ColumnRole.ID]
        declared = [name for name in declared if name in train.columns]
        if declared:
            return declared
        common = set(train.columns)
        if test is not None:
            common &= set(test.columns)
        if sample is not None:
            common &= set(sample.columns)
        named = [
            name for name in train.columns
            if name in common and name not in targets
            and (_normalise_name(name) in {"id", "row_id", "record_id", "index"} or _normalise_name(name).endswith("_id"))
        ]
        return named[:3]

    @staticmethod
    def _validate_columns(columns: Iterable[str], frame: pd.DataFrame, label: str) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Unknown {label} columns: {missing}")

    def _infer_time_column(
        self,
        train: pd.DataFrame,
        metadata: dict[str, dict[str, str | None]],
        explicit: str | None,
        targets: list[str],
        ids: list[str],
    ) -> str | None:
        if explicit:
            self._validate_columns([explicit], train, "time")
            return explicit
        declared = [name for name, meta in metadata.items() if self._normalise_role(meta.get("role")) == ColumnRole.TIME]
        declared = [name for name in declared if name in train.columns]
        if declared:
            return declared[0]
        tokens = ("date", "datetime", "timestamp", "event_time", "дата", "время")
        for name in train.columns:
            if name in targets or name in ids:
                continue
            normalised = _normalise_name(name)
            if any(token in normalised for token in tokens) and self._datetime_parse_rate(train[name]) >= 0.80:
                return name
        return None

    def _infer_role_columns(
        self,
        metadata: dict[str, dict[str, str | None]],
        role: ColumnRole,
        available: Iterable[str],
        explicit: Iterable[str],
    ) -> list[str]:
        explicit_list = list(explicit)
        if explicit_list:
            return [name for name in explicit_list if name in available]
        available_set = set(available)
        return [
            name for name, meta in metadata.items()
            if name in available_set and self._normalise_role(meta.get("role")) == role
        ]

    def _first_role_column(
        self,
        metadata: dict[str, dict[str, str | None]],
        role: ColumnRole,
        available: Iterable[str],
    ) -> str | None:
        columns = self._infer_role_columns(metadata, role, available, ())
        return columns[0] if columns else None

    def _infer_task(
        self,
        *,
        train: pd.DataFrame,
        targets: list[str],
        ids: list[str],
        time_column: str | None,
        groups: list[str],
        weight_column: str | None,
        metadata: dict[str, dict[str, str | None]],
        task_text: str | None,
        explicit_type: TaskType | None,
        explicit_metric: str | None,
    ) -> TaskSpec:
        text = (task_text or "").lower()
        evidence: list[str] = []
        confidence = InferenceConfidence.HIGH if explicit_type else InferenceConfidence.MEDIUM

        if explicit_type:
            task_type = explicit_type
            evidence.append("Task type supplied explicitly.")
        elif not targets:
            if any(token in text for token in ("anomaly", "outlier", "аномал")):
                task_type = TaskType.ANOMALY_DETECTION
                evidence.append("Anomaly-detection wording found in task description.")
            elif any(token in text for token in ("cluster", "кластер")):
                task_type = TaskType.CLUSTERING
                evidence.append("Clustering wording found in task description.")
            else:
                task_type = TaskType.UNKNOWN
                confidence = InferenceConfidence.LOW
                evidence.append("No target column or unsupervised-task keyword was identified.")
        elif len(targets) > 1:
            if all(ptypes.is_numeric_dtype(train[target].dtype) for target in targets):
                task_type = TaskType.MULTIOUTPUT_REGRESSION
            else:
                task_type = TaskType.MULTILABEL_CLASSIFICATION
            evidence.append(f"Detected {len(targets)} target columns.")
        else:
            target = targets[0]
            series = train[target]
            unique = int(series.nunique(dropna=True))
            declared = str(metadata.get(target, {}).get("dtype") or "").lower()
            if any(token in text for token in ("time series", "forecast", "временн", "прогноз") ) and time_column:
                task_type = TaskType.TIME_SERIES_FORECASTING
                evidence.append("Forecasting wording and a time column were detected.")
            elif any(token in text for token in ("ranking", "rank ", "ранжир")):
                task_type = TaskType.RANKING
                evidence.append("Ranking wording found in task description.")
            elif any(token in text for token in ("classification", "классификац", "macro f1", "macro_f1", "accuracy", "roc auc", "roc_auc")):
                task_type = TaskType.BINARY_CLASSIFICATION if unique == 2 else TaskType.MULTICLASS_CLASSIFICATION
                evidence.append("Classification metric or wording found in task description.")
            elif any(token in text for token in ("regression", "регрес", "rmse", "mae", "mse", "r2")):
                task_type = TaskType.REGRESSION
                evidence.append("Regression metric or wording found in task description.")
            elif declared in {"category", "categorical", "string", "text", "class"} or ptypes.is_string_dtype(series.dtype) or ptypes.is_bool_dtype(series.dtype):
                task_type = TaskType.BINARY_CLASSIFICATION if unique == 2 else TaskType.MULTICLASS_CLASSIFICATION
                evidence.append("Target values or dictionary type are categorical.")
            elif ptypes.is_integer_dtype(series.dtype) and unique == 2:
                task_type = TaskType.BINARY_CLASSIFICATION
                evidence.append("Numeric target has exactly two integer values.")
            elif ptypes.is_integer_dtype(series.dtype) and unique <= 30 and _rate(unique, len(series)) <= 0.05:
                task_type = TaskType.MULTICLASS_CLASSIFICATION
                confidence = InferenceConfidence.LOW
                evidence.append("Low-cardinality integer target looks categorical; explicit confirmation is recommended.")
            elif ptypes.is_numeric_dtype(series.dtype):
                task_type = TaskType.REGRESSION
                evidence.append("Target is continuous numeric.")
            else:
                task_type = TaskType.UNKNOWN
                confidence = InferenceConfidence.LOW
                evidence.append("Target type is unsupported or ambiguous.")

        metric = explicit_metric or self._metric_from_text(text) or self._default_metric(task_type)
        direction = self._metric_direction(metric)
        classes: list[str] = []
        class_count: int | None = None
        if len(targets) == 1 and task_type in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        }:
            values = train[targets[0]].dropna().astype(str).unique().tolist()
            classes = sorted(values)
            class_count = len(classes)
        return TaskSpec(
            task_type=task_type,
            target_columns=targets,
            id_columns=ids,
            time_column=time_column,
            group_columns=groups,
            weight_column=weight_column,
            primary_metric=metric,
            metric_direction=direction,
            class_count=class_count,
            classes=classes,
            inference_confidence=confidence,
            inference_evidence=evidence,
        )

    @staticmethod
    def _metric_from_text(text: str) -> str | None:
        metrics = (
            ("macro_f1", ("macro f1", "macro-f1", "macro_f1")),
            ("weighted_f1", ("weighted f1", "weighted-f1", "weighted_f1")),
            ("roc_auc", ("roc auc", "roc-auc", "roc_auc", "auc")),
            ("log_loss", ("log loss", "logloss", "cross entropy")),
            ("rmse", ("rmse", "root mean squared")),
            ("mae", ("mae", "mean absolute")),
            ("mse", ("mse", "mean squared")),
            ("mape", ("mape", "mean absolute percentage")),
            ("r2", ("r2", "r^2", "coefficient of determination")),
            ("accuracy", ("accuracy", "точност")),
            ("f1", ("f1", "f-score")),
            ("ndcg", ("ndcg",)),
        )
        return next((name for name, tokens in metrics if any(token in text for token in tokens)), None)

    @staticmethod
    def _default_metric(task_type: TaskType) -> str | None:
        defaults = {
            TaskType.REGRESSION: "rmse",
            TaskType.MULTIOUTPUT_REGRESSION: "rmse",
            TaskType.TIME_SERIES_FORECASTING: "rmse",
            TaskType.BINARY_CLASSIFICATION: "f1",
            TaskType.MULTICLASS_CLASSIFICATION: "macro_f1",
            TaskType.MULTILABEL_CLASSIFICATION: "macro_f1",
            TaskType.RANKING: "ndcg",
        }
        return defaults.get(task_type)

    @staticmethod
    def _metric_direction(metric: str | None) -> MetricDirection:
        if metric is None:
            return MetricDirection.UNKNOWN
        lower = metric.lower()
        if any(token in lower for token in ("loss", "error", "rmse", "mae", "mse", "mape")):
            return MetricDirection.MINIMIZE
        return MetricDirection.MAXIMIZE

    def _column_role(
        self,
        name: str,
        task: TaskSpec,
        metadata: dict[str, dict[str, str | None]],
    ) -> ColumnRole:
        if name in task.target_columns:
            return ColumnRole.TARGET
        if name in task.id_columns:
            return ColumnRole.ID
        if name == task.time_column:
            return ColumnRole.TIME
        if name in task.group_columns:
            return ColumnRole.GROUP
        if name == task.weight_column:
            return ColumnRole.WEIGHT
        declared = self._normalise_role(metadata.get(name, {}).get("role"))
        return declared or ColumnRole.FEATURE

    def _semantic_type(
        self,
        name: str,
        series: pd.Series,
        metadata: dict[str, str | None],
        role: ColumnRole,
    ) -> SemanticType:
        declared = str(metadata.get("dtype") or "").lower()
        if declared in {"category", "categorical", "class"}:
            return SemanticType.CATEGORICAL
        if declared in {"datetime", "date", "timestamp", "time"}:
            return SemanticType.DATETIME
        if declared in {"boolean", "bool"}:
            return SemanticType.BOOLEAN
        if declared in {"text", "string"}:
            return SemanticType.TEXT
        if self._looks_binary_indicator(name, series, metadata):
            return SemanticType.BOOLEAN
        if declared in {"numeric", "number", "integer", "int", "float", "double"}:
            return SemanticType.NUMERIC
        if role == ColumnRole.TIME and self._datetime_parse_rate(series) >= 0.80:
            return SemanticType.DATETIME
        if ptypes.is_bool_dtype(series.dtype):
            return SemanticType.BOOLEAN
        if ptypes.is_numeric_dtype(series.dtype):
            return SemanticType.NUMERIC
        if ptypes.is_datetime64_any_dtype(series.dtype):
            return SemanticType.DATETIME
        if self._looks_datetime_name(name) and self._datetime_parse_rate(series) >= 0.80:
            return SemanticType.DATETIME
        if ptypes.is_string_dtype(series.dtype) or isinstance(series.dtype, pd.CategoricalDtype):
            non_missing = series.dropna().astype(str)
            if non_missing.empty:
                return SemanticType.CATEGORICAL
            unique_ratio = non_missing.nunique() / len(non_missing)
            median_length = float(non_missing.str.len().median())
            return SemanticType.TEXT if unique_ratio > 0.50 and median_length >= 24 else SemanticType.CATEGORICAL
        return SemanticType.UNKNOWN

    @staticmethod
    def _looks_binary_indicator(
        name: str,
        series: pd.Series,
        metadata: dict[str, str | None],
    ) -> bool:
        if not ptypes.is_numeric_dtype(series.dtype):
            return False
        values = set(pd.to_numeric(series.dropna(), errors="coerce").dropna().unique().tolist())
        if not values or not values.issubset({0, 1}):
            return False
        normalised = _normalise_name(name)
        metadata_text = " ".join(str(value).lower() for value in metadata.values() if value)
        return normalised.startswith(("is_", "has_", "flag_")) or "0/1" in metadata_text or "boolean" in metadata_text

    @staticmethod
    def _looks_datetime_name(name: str) -> bool:
        normalised = _normalise_name(name)
        return any(token in normalised for token in ("date", "datetime", "timestamp", "created_at", "updated_at", "дата", "время"))

    @staticmethod
    def _datetime_parse_rate(series: pd.Series) -> float:
        non_missing = series.dropna()
        if non_missing.empty:
            return 0.0
        sample = non_missing.iloc[: min(500, len(non_missing))]
        parsed = pd.to_datetime(sample, errors="coerce")
        return float(parsed.notna().mean())

    def _profile_column(
        self,
        *,
        name: str,
        role: ColumnRole,
        train_series: pd.Series,
        test_series: pd.Series | None,
        metadata: dict[str, str | None],
    ) -> ColumnProfile:
        semantic = self._semantic_type(name, train_series, metadata, role)
        train_stats = self._split_stats(train_series, semantic)
        test_stats = self._split_stats(test_series, semantic) if test_series is not None else None
        if role == ColumnRole.TARGET and semantic in {SemanticType.CATEGORICAL, SemanticType.BOOLEAN}:
            train_stats.distribution = self._categorical_distribution(
                train_series,
                len(train_series),
                limit=50,
            )
        unseen_count, unseen_rate, unseen_examples = self._unseen_values(
            name, train_series, test_series, semantic, metadata
        )
        shift = self._shift_score(train_series, test_series, semantic)
        tags = self._column_tags(name, role, semantic, train_stats, shift)
        return ColumnProfile(
            name=name,
            role=role,
            semantic_type=semantic,
            pandas_dtype=str(train_series.dtype),
            declared_dtype=metadata.get("dtype"),
            description=metadata.get("description"),
            unit=metadata.get("unit"),
            train=train_stats,
            test=test_stats,
            shift_score=shift,
            unseen_test_value_count=unseen_count,
            unseen_test_row_rate=unseen_rate,
            unseen_test_examples=unseen_examples,
            tags=tags,
        )

    def _split_stats(self, series: pd.Series | None, semantic: SemanticType) -> SplitColumnStats:
        if series is None:
            raise ValueError("series must not be None")
        rows = len(series)
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))
        summary: dict[str, Any]
        distribution: list[DistributionBucket]
        infinite = 0
        if semantic == SemanticType.NUMERIC:
            numeric = pd.to_numeric(series, errors="coerce")
            infinite = int(np.isinf(numeric).sum())
            finite = numeric[np.isfinite(numeric)]
            summary = self._numeric_summary(finite, rows, infinite)
            distribution = self._numeric_distribution(finite, rows)
        elif semantic == SemanticType.DATETIME:
            parsed = pd.to_datetime(series, errors="coerce")
            summary, distribution = self._datetime_summary(parsed, rows)
        elif semantic == SemanticType.TEXT:
            summary, distribution = self._text_summary(series, rows)
        else:
            summary = {}
            distribution = self._categorical_distribution(series, rows)
        return SplitColumnStats(
            rows=rows,
            missing_count=missing,
            missing_rate=_rate(missing, rows),
            unique_count=unique,
            unique_rate=_rate(unique, max(rows - missing, 1)),
            infinite_count=infinite,
            summary=summary,
            distribution=distribution,
        )

    @staticmethod
    def _numeric_summary(finite: pd.Series, rows: int, infinite: int) -> dict[str, Any]:
        if finite.empty:
            return {"finite_count": 0, "infinite_count": infinite}
        q = finite.quantile([0.01, 0.25, 0.50, 0.75, 0.99])
        return {
            "finite_count": int(len(finite)),
            "min": _finite_float(finite.min()),
            "q01": _finite_float(q.loc[0.01]),
            "q25": _finite_float(q.loc[0.25]),
            "median": _finite_float(q.loc[0.50]),
            "mean": _finite_float(finite.mean()),
            "q75": _finite_float(q.loc[0.75]),
            "q99": _finite_float(q.loc[0.99]),
            "max": _finite_float(finite.max()),
            "std": _finite_float(finite.std()),
            "skew": _finite_float(finite.skew()),
            "zero_rate": _rate(int((finite == 0).sum()), rows),
            "negative_rate": _rate(int((finite < 0).sum()), rows),
            "infinite_count": infinite,
        }

    def _numeric_distribution(self, finite: pd.Series, denominator: int) -> list[DistributionBucket]:
        if finite.empty:
            return []
        values = finite.astype(float)
        unique = np.sort(values.unique())
        if len(unique) <= self.histogram_bins:
            counts = values.value_counts().sort_index()
            return [
                DistributionBucket(
                    label=_format_number(float(value)),
                    lower=_finite_float(value),
                    upper=_finite_float(value),
                    count=int(count),
                    rate=_rate(int(count), denominator),
                )
                for value, count in counts.items()
            ]

        minimum = float(values.min())
        maximum = float(values.max())
        robust_low = float(values.quantile(0.01))
        robust_high = float(values.quantile(0.99))
        use_outer = robust_low > minimum or robust_high < maximum
        if use_outer and robust_high > robust_low:
            inner_count = self.histogram_bins - 2
            edges = np.linspace(robust_low, robust_high, inner_count + 1)
            buckets: list[DistributionBucket] = []
            lower_count = int((values < robust_low).sum())
            buckets.append(DistributionBucket(
                label=f"< {_format_number(robust_low)}",
                upper=_finite_float(robust_low),
                count=lower_count,
                rate=_rate(lower_count, denominator),
            ))
            for index in range(inner_count):
                lower = float(edges[index])
                upper = float(edges[index + 1])
                if index == inner_count - 1:
                    count = int(((values >= lower) & (values <= upper)).sum())
                else:
                    count = int(((values >= lower) & (values < upper)).sum())
                buckets.append(DistributionBucket(
                    label=f"{_format_number(lower)} – {_format_number(upper)}",
                    lower=_finite_float(lower),
                    upper=_finite_float(upper),
                    count=count,
                    rate=_rate(count, denominator),
                ))
            upper_count = int((values > robust_high).sum())
            buckets.append(DistributionBucket(
                label=f"> {_format_number(robust_high)}",
                lower=_finite_float(robust_high),
                count=upper_count,
                rate=_rate(upper_count, denominator),
            ))
            return buckets

        if maximum == minimum:
            return [DistributionBucket(
                label=_format_number(minimum), lower=minimum, upper=maximum,
                count=len(values), rate=_rate(len(values), denominator),
            )]
        counts, edges = np.histogram(values, bins=self.histogram_bins)
        return [
            DistributionBucket(
                label=f"{_format_number(float(edges[index]))} – {_format_number(float(edges[index + 1]))}",
                lower=_finite_float(edges[index]),
                upper=_finite_float(edges[index + 1]),
                count=int(count),
                rate=_rate(int(count), denominator),
            )
            for index, count in enumerate(counts)
        ]

    def _categorical_distribution(
        self,
        series: pd.Series,
        denominator: int,
        *,
        limit: int | None = None,
    ) -> list[DistributionBucket]:
        counts = series.value_counts(dropna=False)
        selected_limit = limit or self.top_values
        top = counts.head(selected_limit)
        buckets = [
            DistributionBucket(label=_display_value(value), count=int(count), rate=_rate(int(count), denominator))
            for value, count in top.items()
        ]
        remaining = int(counts.iloc[selected_limit:].sum())
        if remaining:
            buckets.append(DistributionBucket(label="<other>", count=remaining, rate=_rate(remaining, denominator)))
        return buckets

    def _text_summary(self, series: pd.Series, rows: int) -> tuple[dict[str, Any], list[DistributionBucket]]:
        text = series.dropna().astype(str)
        if text.empty:
            return {"non_missing_count": 0}, []
        lengths = text.str.len()
        summary = {
            "length_min": int(lengths.min()),
            "length_median": _finite_float(lengths.median()),
            "length_p95": _finite_float(lengths.quantile(0.95)),
            "length_max": int(lengths.max()),
            "empty_string_rate": _rate(int((lengths == 0).sum()), rows),
        }
        return summary, self._numeric_distribution(lengths, rows)

    def _datetime_summary(
        self,
        parsed: pd.Series,
        rows: int,
    ) -> tuple[dict[str, Any], list[DistributionBucket]]:
        valid = parsed.dropna()
        if valid.empty:
            return {"parse_rate": 0.0}, []
        summary = {
            "parse_rate": _rate(len(valid), rows),
            "min": valid.min().isoformat(),
            "max": valid.max().isoformat(),
        }
        years = valid.dt.year.value_counts().sort_index()
        distribution = [
            DistributionBucket(label=str(year), count=int(count), rate=_rate(int(count), rows))
            for year, count in years.head(20).items()
        ]
        if len(years) > 20:
            remaining = int(years.iloc[20:].sum())
            distribution.append(DistributionBucket(label="<other years>", count=remaining, rate=_rate(remaining, rows)))
        return summary, distribution

    def _unseen_values(
        self,
        name: str,
        train: pd.Series,
        test: pd.Series | None,
        semantic: SemanticType,
        metadata: dict[str, str | None],
    ) -> tuple[int, float, list[str]]:
        if test is None:
            return 0, 0.0, []
        metadata_text = " ".join(
            str(value).lower()
            for value in (name, metadata.get("unit"), metadata.get("description"))
            if value
        )
        numeric_code = (
            semantic == SemanticType.NUMERIC
            and any(token in metadata_text for token in ("code", "period", "код", "период"))
            and train.nunique(dropna=True) <= 50
            and _rate(int(train.nunique(dropna=True)), max(int(train.notna().sum()), 1)) <= 0.05
        )
        if semantic not in {SemanticType.CATEGORICAL, SemanticType.BOOLEAN} and not numeric_code:
            return 0, 0.0, []
        train_values = set(train.dropna().astype(str).unique())
        test_values = test.dropna().astype(str)
        mask = ~test_values.isin(train_values)
        unseen = sorted(test_values[mask].unique().tolist())
        return len(unseen), _rate(int(mask.sum()), len(test)), unseen[: self.top_values]

    @staticmethod
    def _shift_score(train: pd.Series, test: pd.Series | None, semantic: SemanticType) -> float | None:
        if test is None:
            return None
        missing_delta = abs(float(train.isna().mean()) - float(test.isna().mean()))
        if semantic == SemanticType.NUMERIC:
            left = pd.to_numeric(train, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            right = pd.to_numeric(test, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if left.empty or right.empty:
                return _finite_float(missing_delta)
            pooled_std = math.sqrt((float(left.var()) + float(right.var())) / 2)
            mean_shift = (
                abs(float(left.mean()) - float(right.mean())) / pooled_std
                if math.isfinite(pooled_std) and pooled_std > 0
                else 0.0
            )
            return _finite_float(max(mean_shift, missing_delta))
        if semantic in {SemanticType.CATEGORICAL, SemanticType.BOOLEAN}:
            left_freq = train.fillna("<missing>").astype(str).value_counts(normalize=True)
            right_freq = test.fillna("<missing>").astype(str).value_counts(normalize=True)
            values = left_freq.index.union(right_freq.index)
            total_variation = 0.5 * (
                left_freq.reindex(values, fill_value=0) - right_freq.reindex(values, fill_value=0)
            ).abs().sum()
            return _finite_float(max(float(total_variation), missing_delta))
        return _finite_float(missing_delta)

    @staticmethod
    def _column_tags(
        name: str,
        role: ColumnRole,
        semantic: SemanticType,
        stats: SplitColumnStats,
        shift: float | None,
    ) -> list[str]:
        tags: list[str] = []
        if stats.missing_rate >= 0.40:
            tags.append("high_missing")
        if stats.unique_count <= 1:
            tags.append("constant")
        if role == ColumnRole.FEATURE and stats.unique_rate >= 0.98:
            tags.append("near_unique")
        if semantic == SemanticType.CATEGORICAL and stats.unique_count > 100:
            tags.append("high_cardinality")
        if role in {ColumnRole.FEATURE, ColumnRole.TIME, ColumnRole.GROUP} and shift is not None and shift >= 0.25:
            tags.append("train_test_shift")
        if any(token in _normalise_name(name) for token in ("date", "time", "year", "month", "период")):
            tags.append("temporal_candidate")
        return tags

    def _column_issues(self, profile: ColumnProfile) -> list[DataIssue]:
        issues: list[DataIssue] = []
        if profile.role == ColumnRole.FEATURE and "high_missing" in profile.tags:
            issues.append(DataIssue(
                code="high_missing_rate",
                severity=Severity.WARNING,
                message=f"{profile.name}: {profile.train.missing_rate:.1%} missing in train.",
                columns=[profile.name],
                details={
                    "train_missing_rate": profile.train.missing_rate,
                    "test_missing_rate": profile.test.missing_rate if profile.test else None,
                },
            ))
        if profile.role == ColumnRole.FEATURE and "constant" in profile.tags:
            issues.append(DataIssue(
                code="constant_feature",
                severity=Severity.WARNING,
                message=f"{profile.name} is constant in train.",
                columns=[profile.name],
            ))
        if profile.role == ColumnRole.FEATURE and "near_unique" in profile.tags:
            issues.append(DataIssue(
                code="near_unique_feature",
                severity=Severity.INFO,
                message=f"{profile.name} is nearly unique and may behave like an identifier.",
                columns=[profile.name],
            ))
        if profile.unseen_test_value_count:
            code = "unseen_test_categories" if profile.semantic_type != SemanticType.NUMERIC else "unseen_test_codes"
            issues.append(DataIssue(
                code=code,
                severity=Severity.WARNING,
                message=f"{profile.name}: test contains values absent from train.",
                columns=[profile.name],
                details={
                    "unseen_values": profile.unseen_test_value_count,
                    "test_row_rate": profile.unseen_test_row_rate,
                    "examples": profile.unseen_test_examples,
                },
            ))
        if profile.role == ColumnRole.FEATURE and "train_test_shift" in profile.tags:
            issues.append(DataIssue(
                code="train_test_shift",
                severity=Severity.WARNING,
                message=f"{profile.name}: train/test shift score is {profile.shift_score:.3f}.",
                columns=[profile.name],
                details={"shift_score": profile.shift_score},
            ))
        if profile.train.infinite_count:
            issues.append(DataIssue(
                code="infinite_values",
                severity=Severity.ERROR,
                message=f"{profile.name} contains infinite values.",
                columns=[profile.name],
                details={"count": profile.train.infinite_count},
            ))
        return issues

    def _build_checks(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame | None,
        sample: pd.DataFrame | None,
        metadata: dict[str, dict[str, str | None]],
        task: TaskSpec,
    ) -> DatasetChecks:
        feature_columns = [
            column for column in train.columns
            if column not in task.target_columns and column not in task.id_columns
            and column != task.weight_column
        ]
        expected_test = [column for column in train.columns if column not in task.target_columns]
        train_hash = pd.util.hash_pandas_object(train[feature_columns], index=False) if feature_columns else pd.Series(dtype="uint64")
        target_frame = train[task.target_columns].astype(str).agg("\x1f".join, axis=1) if task.target_columns else None
        conflicting = None
        if task.target_columns and not train_hash.empty and target_frame is not None:
            grouped = pd.DataFrame({"feature_hash": train_hash, "target": target_frame})
            conflicting = int(grouped.groupby("feature_hash", sort=False)["target"].nunique().gt(1).sum())

        if test is None:
            schema_match = None
            train_only: list[str] = []
            test_only: list[str] = []
            test_hash = pd.Series(dtype="uint64")
            overlap = None
        else:
            schema_match = expected_test == list(test.columns)
            train_only = [column for column in train.columns if column not in test.columns]
            test_only = [column for column in test.columns if column not in train.columns]
            shared_features = [column for column in feature_columns if column in test.columns]
            test_hash = pd.util.hash_pandas_object(test[shared_features], index=False) if shared_features else pd.Series(dtype="uint64")
            comparable_train_hash = pd.util.hash_pandas_object(train[shared_features], index=False) if shared_features else pd.Series(dtype="uint64")
            overlap = int(test_hash.isin(set(comparable_train_hash.tolist())).sum()) if shared_features else None

        duplicate_train_ids, null_train_ids = self._id_checks(train, task.id_columns)
        duplicate_test_ids, null_test_ids = self._id_checks(test, task.id_columns) if test is not None else (None, None)

        expected_sample = [*task.id_columns, *task.target_columns]
        sample_columns_match = list(sample.columns) == expected_sample if sample is not None and expected_sample else None
        sample_rows_match = len(sample) == len(test) if sample is not None and test is not None else None
        sample_order_match: bool | None = None
        if sample is not None and test is not None and task.id_columns and all(
            column in sample.columns and column in test.columns for column in task.id_columns
        ):
            sample_order_match = bool(sample[task.id_columns].equals(test[task.id_columns]))

        return DatasetChecks(
            train_test_schema_match=schema_match,
            train_only_columns=train_only,
            test_only_columns=test_only,
            duplicate_train_rows=int(train.duplicated().sum()),
            duplicate_train_feature_rows=int(train_hash.duplicated().sum()) if not train_hash.empty else None,
            duplicate_test_feature_rows=int(test_hash.duplicated().sum()) if not test_hash.empty else None,
            conflicting_duplicate_target_groups=conflicting,
            train_test_feature_overlap_rows=overlap,
            duplicate_train_ids=duplicate_train_ids,
            duplicate_test_ids=duplicate_test_ids,
            null_train_ids=null_train_ids,
            null_test_ids=null_test_ids,
            sample_columns_match=sample_columns_match,
            sample_row_count_matches_test=sample_rows_match,
            sample_id_order_matches_test=sample_order_match,
            dictionary_covers_all_train_columns=(set(metadata) >= set(train.columns)) if metadata else None,
        )

    @staticmethod
    def _id_checks(frame: pd.DataFrame | None, ids: list[str]) -> tuple[int | None, int | None]:
        if frame is None or not ids or not all(column in frame.columns for column in ids):
            return None, None
        return int(frame.duplicated(subset=ids).sum()), int(frame[ids].isna().any(axis=1).sum())

    @staticmethod
    def _check_issues(checks: DatasetChecks) -> list[DataIssue]:
        issues: list[DataIssue] = []
        failures = {
            "schema_mismatch": checks.train_test_schema_match is False,
            "duplicate_train_ids": bool(checks.duplicate_train_ids),
            "duplicate_test_ids": bool(checks.duplicate_test_ids),
            "null_ids": bool(checks.null_train_ids) or bool(checks.null_test_ids),
            "sample_columns_mismatch": checks.sample_columns_match is False,
            "sample_row_count_mismatch": checks.sample_row_count_matches_test is False,
            "sample_id_order_mismatch": checks.sample_id_order_matches_test is False,
            "dictionary_mismatch": checks.dictionary_covers_all_train_columns is False,
        }
        messages = {
            "schema_mismatch": "Train feature columns and test columns differ or are ordered differently.",
            "duplicate_train_ids": "Train contains duplicate ID combinations.",
            "duplicate_test_ids": "Test contains duplicate ID combinations.",
            "null_ids": "At least one ID column contains missing values.",
            "sample_columns_mismatch": "Sample submission columns do not match IDs and targets.",
            "sample_row_count_mismatch": "Sample submission row count does not match test.",
            "sample_id_order_mismatch": "Sample submission ID order does not match test.",
            "dictionary_mismatch": "Column dictionaries do not cover every train column.",
        }
        for code, failed in failures.items():
            if failed:
                issues.append(DataIssue(code=code, severity=Severity.ERROR, message=messages[code]))
        if checks.duplicate_train_feature_rows:
            issues.append(DataIssue(
                code="duplicate_feature_rows",
                severity=Severity.WARNING,
                message="Train contains identical feature rows; keep duplicate groups in one validation fold.",
                details={
                    "duplicate_train_rows": checks.duplicate_train_feature_rows,
                    "conflicting_target_groups": checks.conflicting_duplicate_target_groups,
                },
            ))
        if checks.train_test_feature_overlap_rows:
            issues.append(DataIssue(
                code="train_test_feature_overlap",
                severity=Severity.WARNING,
                message="Some test rows exactly match train rows after excluding IDs and targets.",
                details={"matching_test_rows": checks.train_test_feature_overlap_rows},
            ))
        return issues

    def _target_summary(
        self,
        target: pd.Series,
        task: TaskSpec,
        columns: list[ColumnProfile],
    ) -> dict[str, Any]:
        profile = next(column for column in columns if column.name == target.name)
        return {
            "semantic_type": profile.semantic_type,
            "missing_count": profile.train.missing_count,
            "missing_rate": profile.train.missing_rate,
            "unique_count": profile.train.unique_count,
            "summary": profile.train.summary,
            "distribution": [bucket.model_dump(mode="json") for bucket in profile.train.distribution],
        }

    @staticmethod
    def _target_issues(
        train: pd.DataFrame,
        task: TaskSpec,
        summaries: dict[str, dict[str, Any]],
    ) -> list[DataIssue]:
        issues: list[DataIssue] = []
        for target, summary in summaries.items():
            if summary["missing_count"]:
                issues.append(DataIssue(
                    code="missing_target",
                    severity=Severity.ERROR,
                    message=f"{target} contains missing target values.",
                    columns=[target],
                    details={"count": summary["missing_count"]},
                ))
        if len(task.target_columns) == 1 and task.task_type in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        }:
            target = task.target_columns[0]
            counts = train[target].value_counts(dropna=True)
            if len(counts) and counts.min() > 0 and counts.max() / counts.min() >= 10:
                issues.append(DataIssue(
                    code="class_imbalance",
                    severity=Severity.WARNING,
                    message="Target classes are strongly imbalanced; use stratified validation and class-aware metrics.",
                    columns=[target],
                    details={"max_to_min_ratio": _finite_float(counts.max() / counts.min())},
                ))
        if len(task.target_columns) == 1 and task.task_type in {
            TaskType.REGRESSION,
            TaskType.TIME_SERIES_FORECASTING,
        }:
            target = task.target_columns[0]
            skew = pd.to_numeric(train[target], errors="coerce").skew()
            if math.isfinite(float(skew)) and abs(float(skew)) >= 2:
                issues.append(DataIssue(
                    code="skewed_regression_target",
                    severity=Severity.WARNING,
                    message="Regression target is strongly skewed; evaluate transforms but score on the required scale.",
                    columns=[target],
                    details={"skew": _finite_float(skew)},
                ))
        return issues

    @staticmethod
    def _task_definition_issues(task: TaskSpec) -> list[DataIssue]:
        supervised = {
            TaskType.REGRESSION,
            TaskType.MULTIOUTPUT_REGRESSION,
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
            TaskType.TIME_SERIES_FORECASTING,
            TaskType.RANKING,
        }
        issues: list[DataIssue] = []
        if task.task_type in supervised and not task.target_columns:
            issues.append(DataIssue(
                code="missing_target_definition",
                severity=Severity.ERROR,
                message="The selected supervised task has no target columns.",
            ))
        if task.task_type == TaskType.TIME_SERIES_FORECASTING and not task.time_column:
            issues.append(DataIssue(
                code="missing_time_column",
                severity=Severity.ERROR,
                message="Time-series forecasting requires an explicit time column.",
            ))
        if task.task_type == TaskType.RANKING and not task.group_columns:
            issues.append(DataIssue(
                code="missing_ranking_group",
                severity=Severity.ERROR,
                message="Ranking requires a query or group column for validation and metrics.",
            ))
        return issues

    @staticmethod
    def _leakage_issues(
        train: pd.DataFrame,
        task: TaskSpec,
        columns: list[ColumnProfile],
    ) -> list[DataIssue]:
        if len(task.target_columns) != 1:
            return []
        target_name = task.target_columns[0]
        target = train[target_name]
        issues: list[DataIssue] = []
        semantic = {column.name: column.semantic_type for column in columns}
        for column in train.columns:
            if column == target_name or column in task.id_columns:
                continue
            candidate = train[column]
            exact = candidate.equals(target)
            correlation = None
            if ptypes.is_numeric_dtype(candidate.dtype) and ptypes.is_numeric_dtype(target.dtype):
                correlation = candidate.corr(target)
            if exact or (correlation is not None and math.isfinite(float(correlation)) and abs(float(correlation)) >= 0.999999):
                issues.append(DataIssue(
                    code="target_leakage_candidate",
                    severity=Severity.ERROR,
                    message=f"{column} reproduces or almost perfectly tracks the target.",
                    columns=[column, target_name],
                    details={"exact_match": exact, "correlation": _finite_float(correlation)},
                ))
            elif semantic.get(column) == SemanticType.CATEGORICAL and candidate.astype(str).equals(target.astype(str)):
                issues.append(DataIssue(
                    code="target_leakage_candidate",
                    severity=Severity.ERROR,
                    message=f"{column} duplicates the target labels.",
                    columns=[column, target_name],
                ))
        return issues

    def _profile_files(
        self,
        layout: InputLayout,
        train: pd.DataFrame,
        test: pd.DataFrame | None,
        sample: pd.DataFrame | None,
    ) -> list[TableProfile]:
        profiles = [self._table_profile(layout.train_path, FileRole.TRAIN, train)]
        if layout.test_path and test is not None:
            profiles.append(self._table_profile(layout.test_path, FileRole.TEST, test))
        if layout.sample_path and sample is not None:
            profiles.append(self._table_profile(layout.sample_path, FileRole.SAMPLE_SUBMISSION, sample))
        for path in layout.dictionary_paths:
            try:
                frame = self._read_table(path)
                profiles.append(self._table_profile(path, FileRole.DATA_DICTIONARY, frame))
            except Exception as error:
                profiles.append(self._failed_table_profile(path, FileRole.DATA_DICTIONARY, error))
        for path in layout.auxiliary_paths:
            try:
                frame = self._read_table(path)
                profiles.append(self._table_profile(path, FileRole.AUXILIARY, frame))
            except Exception as error:
                profiles.append(self._failed_table_profile(path, FileRole.AUXILIARY, error))
        if layout.task_path:
            profiles.append(TableProfile(
                file_name=layout.task_path.name,
                path=str(layout.task_path.resolve()),
                role=FileRole.TASK_DESCRIPTION,
                size_bytes=layout.task_path.stat().st_size,
            ))
        return profiles

    @staticmethod
    def _table_profile(path: Path, role: FileRole, frame: pd.DataFrame) -> TableProfile:
        return TableProfile(
            file_name=path.name,
            path=str(path.resolve()),
            role=role,
            rows=len(frame),
            columns=len(frame.columns),
            column_names=list(frame.columns),
            size_bytes=path.stat().st_size,
            memory_bytes=int(frame.memory_usage(index=True, deep=True).sum()),
            duplicate_rows=int(frame.duplicated().sum()),
        )

    @staticmethod
    def _failed_table_profile(path: Path, role: FileRole, error: Exception) -> TableProfile:
        return TableProfile(
            file_name=path.name,
            path=str(path.resolve()),
            role=role,
            size_bytes=path.stat().st_size,
            read_error=str(error)[:500],
        )

    def _relationship_notes(
        self,
        layout: InputLayout,
        train: pd.DataFrame,
        files: list[TableProfile],
    ) -> list[str]:
        notes: list[str] = []
        train_columns = set(train.columns)
        for file in files:
            if file.role != FileRole.AUXILIARY or not file.column_names:
                continue
            shared = sorted(train_columns & set(file.column_names))
            if shared:
                notes.append(f"Auxiliary table {file.file_name} shares columns with train: {', '.join(shared[:12])}.")
        if layout.task_path is None:
            notes.append("No task description file was identified; metric and task inference rely on data only.")
        if not layout.dictionary_paths:
            notes.append("No column dictionary was identified; semantic types and roles rely on heuristics.")
        return notes

    @staticmethod
    def _preprocessing_hints(
        columns: list[ColumnProfile],
        task: TaskSpec,
        issues: list[DataIssue],
        checks: DatasetChecks,
    ) -> list[str]:
        hints: list[str] = []
        feature_columns = [column for column in columns if column.role == ColumnRole.FEATURE]
        if any(column.semantic_type == SemanticType.CATEGORICAL for column in feature_columns):
            hints.append("Fit categorical handling inside each training fold and support unknown categories.")
        if any(column.semantic_type == SemanticType.TEXT for column in feature_columns):
            hints.append("Treat free text separately; do not feed raw strings into ordinary numeric estimators.")
        if any(column.semantic_type == SemanticType.DATETIME for column in feature_columns) or task.time_column:
            hints.append("Derive time features without using future information and prefer time-aware validation when order matters.")
        if any(column.train.missing_count for column in feature_columns):
            hints.append("Fit imputers inside each training fold; consider missingness indicators when absence may carry signal.")
        if any(issue.code == "duplicate_feature_rows" for issue in issues):
            hints.append("Place identical feature rows in the same validation fold to prevent duplicate leakage.")
        if any(issue.code in {"unseen_test_codes", "train_test_shift"} for issue in issues):
            hints.append("Choose validation splits that reproduce observed train/test group, period, or distribution shifts.")
        if task.task_type in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        }:
            hints.append("Stratify validation where possible and optimize the declared class-aware metric.")
        if task.task_type in {
            TaskType.REGRESSION,
            TaskType.MULTIOUTPUT_REGRESSION,
            TaskType.TIME_SERIES_FORECASTING,
        }:
            hints.append("Evaluate predictions on the metric's required target scale after any inverse transformation.")
        if task.id_columns:
            hints.append("Exclude ID columns from model features unless a controlled experiment demonstrates legitimate signal.")
        if checks.conflicting_duplicate_target_groups:
            hints.append("Identical features have conflicting targets; deterministic lookup cannot resolve every repeated row.")
        return list(dict.fromkeys(hints))


def build_llm_context(profile: DatasetProfile, max_chars: int = 30_000) -> dict[str, Any]:
    """Build a compact JSON-safe context and keep complete details in the full profile."""

    important_columns = {
        column
        for issue in profile.issues
        if issue.severity != Severity.INFO
        for column in issue.columns
    } | set(profile.task.target_columns) | set(profile.task.id_columns)
    ordered = sorted(
        profile.columns,
        key=lambda column: (
            column.name not in important_columns,
            column.role == ColumnRole.FEATURE,
            -(column.shift_score or 0),
            column.name,
        ),
    )
    detail_candidates = sorted(
        (column for column in profile.columns if column.role == ColumnRole.FEATURE),
        key=lambda column: (
            column.name not in important_columns,
            -(column.shift_score or 0),
            -column.train.missing_rate,
        ),
    )
    detailed_names = set(profile.task.target_columns)
    detailed_names.update(
        column.name
        for column in profile.columns
        if column.semantic_type in {SemanticType.CATEGORICAL, SemanticType.TEXT, SemanticType.DATETIME}
    )
    detailed_names.update(column.name for column in detail_candidates[:12])
    compact_columns: list[dict[str, Any]] = []
    for column in ordered:
        compact_summary = {
            key: value
            for key, value in column.train.summary.items()
            if key in {"min", "q01", "median", "mean", "q99", "max", "skew", "length_median", "length_p95", "parse_rate"}
        }
        item: dict[str, Any] = {
            "name": column.name,
            "role": column.role,
            "type": column.semantic_type,
            "missing_train": column.train.missing_rate,
            "missing_test": column.test.missing_rate if column.test else None,
            "unique_train": column.train.unique_count,
            "summary": column.train.summary if column.name in detailed_names else compact_summary,
            "distribution": (
                [
                    {"range_or_value": bucket.label, "rate": bucket.rate, "count": bucket.count}
                    for bucket in column.train.distribution
                ]
                if column.name in detailed_names
                else []
            ),
            "shift": column.shift_score,
            "tags": column.tags,
        }
        if column.description:
            item["description"] = column.description[:220]
        if column.unit:
            item["unit"] = column.unit
        if column.unseen_test_value_count:
            item["unseen_test"] = {
                "value_count": column.unseen_test_value_count,
                "row_rate": column.unseen_test_row_rate,
                "examples": column.unseen_test_examples,
            }
        compact_columns.append(item)

    issue_groups: list[dict[str, Any]] = []
    for code in sorted({issue.code for issue in profile.issues}):
        matches = [issue for issue in profile.issues if issue.code == code]
        columns = list(dict.fromkeys(column for issue in matches for column in issue.columns))
        severity = max(matches, key=lambda issue: {Severity.INFO: 0, Severity.WARNING: 1, Severity.ERROR: 2}[issue.severity]).severity
        issue_groups.append({
            "code": code,
            "severity": severity,
            "count": len(matches),
            "columns": columns[:25],
            "example": matches[0].message,
        })

    payload: dict[str, Any] = {
        "schema_version": profile.schema_version,
        "dataset": profile.dataset_name,
        "task_description": (profile.task_description or "")[:3_000],
        "task": profile.task.model_dump(mode="json"),
        "tables": [
            {
                "file": file.file_name,
                "role": file.role,
                "rows": file.rows,
                "columns": file.columns,
                "duplicate_rows": file.duplicate_rows,
                "read_error": file.read_error,
            }
            for file in profile.files
        ],
        "target_summaries": profile.target_summaries,
        "checks": profile.checks.model_dump(mode="json"),
        "issue_groups": issue_groups,
        "preprocessing_hints": profile.preprocessing_hints,
        "notes": profile.report_notes,
        "columns": compact_columns,
        "omitted_columns": [],
    }

    def size() -> int:
        return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    if size() > max_chars:
        for item in payload["columns"]:
            item.pop("description", None)
    if size() > max_chars:
        for item in payload["columns"]:
            if item["name"] not in profile.task.target_columns:
                item["distribution"] = []
                item["summary"] = {
                    key: value for key, value in item["summary"].items()
                    if key in {"min", "median", "mean", "max", "length_median", "length_p95"}
                }
    if size() > max_chars:
        payload["task_description"] = payload["task_description"][:500]
    while size() > max_chars and payload["columns"]:
        removed = payload["columns"].pop()
        payload["omitted_columns"].append(removed["name"])
    if size() > max_chars:
        raise ValueError("max_chars is too small for the mandatory dataset context")
    return payload


def _write_json(path: Path, value: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if compact
        else json.dumps(value, ensure_ascii=False, indent=2)
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Universal tabular dataset profiler for an AutoML agent.")
    parser.add_argument("data_dir", type=Path, help="Directory containing task files")
    parser.add_argument("--manifest", type=Path, help="Optional dataset_manifest.json")
    parser.add_argument("--name", help="Dataset name")
    parser.add_argument("--train", type=Path, help="Train CSV/TSV; overrides discovery")
    parser.add_argument("--test", type=Path, help="Test CSV/TSV; overrides discovery")
    parser.add_argument("--sample", type=Path, help="Sample submission CSV/TSV")
    parser.add_argument("--dictionary", type=Path, action="append", default=[], help="Column dictionary; repeatable")
    parser.add_argument("--task-description", type=Path, help="Task Markdown or text file")
    parser.add_argument("--target", action="append", default=[], help="Target column; repeatable")
    parser.add_argument("--id", action="append", default=[], help="ID column; repeatable")
    parser.add_argument("--task-type", choices=[value.value for value in TaskType])
    parser.add_argument("--metric")
    parser.add_argument("--time-column")
    parser.add_argument("--group-column", action="append", default=[])
    parser.add_argument("--weight-column")
    parser.add_argument("--output", type=Path, required=True, help="Detailed profile JSON")
    parser.add_argument("--llm-output", type=Path, required=True, help="Compact LLM JSON")
    parser.add_argument("--report-output", type=Path, help="Compact Markdown report for the LLM")
    parser.add_argument("--llm-max-chars", type=int, default=30_000)
    args = parser.parse_args()

    profile = DataInspector().inspect_directory(
        args.data_dir,
        args.name,
        manifest_path=args.manifest,
        train_path=args.train,
        test_path=args.test,
        sample_submission_path=args.sample,
        dictionary_paths=args.dictionary,
        task_description_path=args.task_description,
        target_columns=args.target,
        id_columns=args.id,
        task_type=args.task_type,
        primary_metric=args.metric,
        time_column=args.time_column,
        group_columns=args.group_column,
        weight_column=args.weight_column,
    )
    _write_json(args.output, profile.model_dump(mode="json"))
    context = build_llm_context(profile, max_chars=args.llm_max_chars)
    _write_json(args.llm_output, context, compact=True)
    if args.report_output:
        from .llm_report import build_llm_markdown

        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(
            build_llm_markdown(profile, max_chars=args.llm_max_chars),
            encoding="utf-8",
        )
    print(json.dumps({
        "dataset": profile.dataset_name,
        "task_type": profile.task.task_type,
        "targets": profile.task.target_columns,
        "tables": len(profile.files),
        "issues": len(profile.issues),
        "profile": str(args.output),
        "llm_context": str(args.llm_output),
        "llm_report": str(args.report_output) if args.report_output else None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
