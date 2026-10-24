"""Executable task inputs and metric name normalization."""

from __future__ import annotations

import re
import json

from .contracts import (
    DatasetProfile,
    MetricDirection,
    Severity,
    TaskDefinition,
    TaskType,
    TaskValidationFinding,
    TaskValidationReport,
    ValidationStrategy,
)


_SUPERVISED_TYPES = {
    TaskType.REGRESSION,
    TaskType.MULTIOUTPUT_REGRESSION,
    TaskType.BINARY_CLASSIFICATION,
    TaskType.MULTICLASS_CLASSIFICATION,
    TaskType.MULTILABEL_CLASSIFICATION,
    TaskType.TIME_SERIES_FORECASTING,
    TaskType.RANKING,
}

def _normalise_metric(value: str | None) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    aliases = {
        "root_mean_squared_error": "rmse",
        "root_mean_square_error": "rmse",
        "mean_absolute_percentage_error": "mape",
        "area_under_roc_curve": "roc_auc",
        "roc_auc_score": "roc_auc",
        "balanced_accuracy_score": "balanced_accuracy",
        "average_precision_score": "average_precision",
        "mean_squared_logarithmic_error": "msle",
        "root_mean_squared_logarithmic_error": "rmsle",
        "root_mean_squared_log_error": "rmsle",
        "symmetric_mean_absolute_percentage_error": "smape",
        "f1_score": "f1",
        "f1_macro": "macro_f1",
        "f1_micro": "micro_f1",
        "f1_weighted": "weighted_f1",
        "mean_absolute_error": "mae",
        "mean_squared_error": "mse",
        "r2_score": "r2",
        "mean_average_precision": "map",
        "mean_reciprocal_rank": "mrr",
    }
    return aliases.get(text, text)


class TaskValidator:
    """Only the columns and basic inputs needed to execute a confirmed task."""

    def validate(self, profile: DatasetProfile, task: TaskDefinition | None,
                 *, generation_error: str | None = None) -> TaskValidationReport:
        findings = []
        def add(code, message, field=None):
            findings.append(TaskValidationFinding(code=code, severity=Severity.ERROR,
                                                  message=message, field=field))
        if task is None:
            add('invalid_llm_output', generation_error or 'A TaskDefinition is required.')
        else:
            available = {column.name for column in profile.columns}
            referenced = set(task.target_columns + task.id_columns + task.group_columns + task.excluded_feature_columns)
            referenced.update(v for v in (task.time_column, task.weight_column) if v)
            if referenced - available:
                add('unknown_columns', 'Missing train columns: '+', '.join(sorted(referenced-available)), 'columns')
            if task.task_type in _SUPERVISED_TYPES and not task.target_columns:
                add('missing_target', 'A supervised task needs target columns.', 'target_columns')
            if not task.primary_metric or task.metric_direction == MetricDirection.UNKNOWN:
                add('missing_metric', 'Metric and optimization direction are required.', 'primary_metric')
            try:
                json.dumps(task.metric_options, allow_nan=False)
            except (TypeError, ValueError):
                add('invalid_metric_option', 'Metric options must be finite JSON values.', 'metric_options')
        return TaskValidationReport(valid=not findings, findings=findings)
