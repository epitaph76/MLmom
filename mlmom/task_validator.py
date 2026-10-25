"""Executable task inputs and metric name normalization."""

from __future__ import annotations

import json
from .metric_contracts import normalise_metric, metric_errors

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
    return normalise_metric(value)


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
            elif task.task_type != TaskType.UNKNOWN:
                for message in metric_errors(task.primary_metric, task.task_type, task.metric_options):
                    add('unsupported_metric', message, 'primary_metric')
                if _normalise_metric(task.primary_metric)=='mcc' and task.metric_direction!=MetricDirection.MAXIMIZE:
                    add('invalid_metric_direction', 'MCC must be maximized.', 'metric_direction')
            try:
                json.dumps(task.metric_options, allow_nan=False)
            except (TypeError, ValueError):
                add('invalid_metric_option', 'Metric options must be finite JSON values.', 'metric_options')
        return TaskValidationReport(valid=not findings, findings=findings)
