"""Versioned JSON contracts for dataset inspection.

The models contain no pandas or NumPy objects. They can be persisted by the
Experiment Store, consumed by the Evaluation System, and reduced to an LLM
prompt without serialising raw records.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskType(StrEnum):
    REGRESSION = "regression"
    MULTIOUTPUT_REGRESSION = "multioutput_regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    RANKING = "ranking"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"
    UNKNOWN = "unknown"


class MetricDirection(StrEnum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    UNKNOWN = "unknown"


class ValidationStrategy(StrEnum):
    """Cross-validation scheme selected for model evaluation."""

    STRATIFIED_KFOLD = "stratified_kfold"
    KFOLD = "kfold"
    GROUP_KFOLD = "group_kfold"
    TIME_SERIES_SPLIT = "time_series_split"
    HOLDOUT = "holdout"
    UNSPECIFIED = "unspecified"


class InferenceConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SemanticType(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


class ColumnRole(StrEnum):
    ID = "id"
    FEATURE = "feature"
    TARGET = "target"
    TIME = "time"
    GROUP = "group"
    WEIGHT = "weight"
    IGNORED = "ignored"


class FileRole(StrEnum):
    TRAIN = "train"
    DATA_DICTIONARY = "data_dictionary"
    TASK_DESCRIPTION = "task_description"
    UNKNOWN = "unknown"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ResultFormat(StrictModel):
    """Machine-readable output requirements extracted from the task text."""

    file_name: str | None = None
    encoding: str | None = None
    delimiter: str | None = None
    include_index: bool | None = None
    columns: list[str] = Field(default_factory=list)
    id_columns: list[str] = Field(default_factory=list)
    prediction_columns: list[str] = Field(default_factory=list)
    row_order: str | None = None
    one_prediction_per_id: bool | None = None
    finite_predictions_required: bool | None = None
    missing_predictions_allowed: bool | None = None
    duplicate_ids_allowed: bool | None = None


class TaskSpec(StrictModel):
    """Task definition inferred from data, descriptions, and explicit hints."""

    task_type: TaskType
    target_columns: list[str] = Field(default_factory=list)
    id_columns: list[str] = Field(default_factory=list)
    time_column: str | None = None
    group_columns: list[str] = Field(default_factory=list)
    weight_column: str | None = None
    primary_metric: str | None = None
    metric_direction: MetricDirection = MetricDirection.UNKNOWN
    class_count: int | None = None
    classes: list[str] = Field(default_factory=list)
    inference_confidence: InferenceConfidence = InferenceConfidence.LOW
    inference_evidence: list[str] = Field(default_factory=list)
    result_format: ResultFormat = Field(default_factory=ResultFormat)
    requirements: list[str] = Field(default_factory=list)


class TaskDefinition(StrictModel):
    """Canonical task description proposed by the primary LLM node."""

    schema_version: str = "1.0.0"
    objective: str
    task_type: TaskType
    target_columns: list[str] = Field(default_factory=list)
    id_columns: list[str] = Field(default_factory=list)
    time_column: str | None = None
    group_columns: list[str] = Field(default_factory=list)
    weight_column: str | None = None
    excluded_feature_columns: list[str] = Field(default_factory=list)
    primary_metric: str
    metric_direction: MetricDirection
    validation_strategy: ValidationStrategy
    result_format: ResultFormat
    requirements: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class TaskValidationFinding(StrictModel):
    code: str
    severity: Severity
    message: str
    field: str | None = None
    expected: Any | None = None
    actual: Any | None = None


class TaskValidationReport(StrictModel):
    valid: bool
    findings: list[TaskValidationFinding] = Field(default_factory=list)


class TaskAnalysisResult(StrictModel):
    status: Literal["validated", "failed"]
    attempts: int
    model: str
    task: TaskDefinition | None = None
    validation: TaskValidationReport


class DataIssue(StrictModel):
    code: str
    severity: Severity
    message: str
    columns: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class DistributionBucket(StrictModel):
    """One numeric range or categorical value with count and share."""

    label: str
    count: int
    rate: float
    lower: float | None = None
    upper: float | None = None


class SplitColumnStats(StrictModel):
    rows: int
    missing_count: int
    missing_rate: float
    unique_count: int
    unique_rate: float
    infinite_count: int = 0
    summary: dict[str, Any] = Field(default_factory=dict)
    distribution: list[DistributionBucket] = Field(default_factory=list)


class ColumnProfile(StrictModel):
    name: str
    role: ColumnRole
    semantic_type: SemanticType
    pandas_dtype: str
    declared_dtype: str | None = None
    description: str | None = None
    unit: str | None = None
    train: SplitColumnStats
    tags: list[str] = Field(default_factory=list)


class TableProfile(StrictModel):
    file_name: str
    path: str
    role: FileRole
    rows: int | None = None
    columns: int | None = None
    column_names: list[str] = Field(default_factory=list)
    size_bytes: int
    memory_bytes: int | None = None
    duplicate_rows: int | None = None
    sampled_for_statistics: bool = False
    read_error: str | None = None


class DatasetChecks(StrictModel):
    duplicate_train_rows: int | None = None
    duplicate_train_feature_rows: int | None = None
    conflicting_duplicate_target_groups: int | None = None
    duplicate_train_ids: int | None = None
    null_train_ids: int | None = None
    dictionary_covers_all_train_columns: bool | None = None


class DatasetProfile(StrictModel):
    """Full deterministic output of the universal Data Inspector."""

    schema_version: str = "3.0.0"
    dataset_name: str
    root_path: str
    task_description: str | None = None
    files: list[TableProfile]
    task: TaskSpec
    columns: list[ColumnProfile]
    target_summaries: dict[str, dict[str, Any]] = Field(default_factory=dict)
    checks: DatasetChecks
    issues: list[DataIssue] = Field(default_factory=list)
    preprocessing_hints: list[str] = Field(default_factory=list)
    report_notes: list[str] = Field(default_factory=list)
