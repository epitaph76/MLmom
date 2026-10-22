"""Deterministic validation for LLM-generated task definitions."""

from __future__ import annotations

import re
import json
from typing import Any

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

_REGRESSION_METRICS = {
    "mae", "mape", "mean_absolute_error", "mean_squared_error", "mse",
    "r2", "r2_score", "rmse", "rmsle", "smape",
}
_CLASSIFICATION_METRICS = {
    "accuracy", "average_precision", "auc", "balanced_accuracy", "f1",
    "f1_macro", "f1_micro", "f1_weighted", "log_loss", "macro_f1",
    "micro_f1", "precision", "recall", "roc_auc", "weighted_f1",
}
_RANKING_METRICS = {"map", "mean_average_precision", "mrr", "ndcg"}
_CLUSTERING_METRICS = {"calinski_harabasz", "davies_bouldin", "silhouette"}
_MINIMIZED_METRICS = {
    "log_loss", "mae", "mape", "mean_absolute_error", "mean_squared_error",
    "mse", "rmse", "rmsle", "smape", "davies_bouldin",
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
    """Check claims against the original deterministic DatasetProfile."""

    def validate(
        self,
        profile: DatasetProfile,
        task: TaskDefinition | None,
        *,
        generation_error: str | None = None,
    ) -> TaskValidationReport:
        findings: list[TaskValidationFinding] = []

        def add(
            code: str,
            message: str,
            *,
            field: str | None = None,
            expected: Any | None = None,
            actual: Any | None = None,
            severity: Severity = Severity.ERROR,
        ) -> None:
            # New profiles distinguish evidence from suggestions. Legacy
            # profiles without provenance retain their previous strict checks.
            if (field and code.endswith('mismatch') and
                    profile.task.field_sources.get(field, 'explicit') != 'explicit'):
                severity = Severity.WARNING
            findings.append(TaskValidationFinding(
                code=code,
                severity=severity,
                message=message,
                field=field,
                expected=expected,
                actual=actual,
            ))

        if task is None:
            add(
                "invalid_llm_output",
                generation_error or "The LLM did not produce a valid TaskDefinition JSON object.",
            )
            return TaskValidationReport(valid=False, findings=findings)

        available = {column.name for column in profile.columns}
        referenced = {
            *task.target_columns,
            *task.id_columns,
            *task.group_columns,
            *task.excluded_feature_columns,
        }
        if task.time_column:
            referenced.add(task.time_column)
        if task.weight_column:
            referenced.add(task.weight_column)
        unknown = sorted(referenced - available)
        if unknown:
            add(
                "unknown_columns",
                "The task definition references columns absent from train.",
                field="columns",
                expected=sorted(available),
                actual=unknown,
            )

        inspected = profile.task
        if inspected.task_type != TaskType.UNKNOWN and task.task_type != inspected.task_type:
            add(
                "task_type_mismatch",
                "The LLM task type conflicts with the deterministic inspection.",
                field="task_type",
                expected=inspected.task_type,
                actual=task.task_type,
            )
        if task.task_type in _SUPERVISED_TYPES and not task.target_columns:
            add("missing_target", "A supervised task must define at least one target.", field="target_columns")
        if inspected.target_columns and task.target_columns != inspected.target_columns:
            add(
                "target_mismatch",
                "Target columns must match the inspected task and data dictionary.",
                field="target_columns",
                expected=inspected.target_columns,
                actual=task.target_columns,
            )
        if inspected.id_columns and task.id_columns != inspected.id_columns:
            add(
                "id_mismatch",
                "ID columns must match the inspected task contract.",
                field="id_columns",
                expected=inspected.id_columns,
                actual=task.id_columns,
            )
        for field_name in ("time_column", "weight_column"):
            expected = getattr(inspected, field_name)
            actual = getattr(task, field_name)
            if expected is not None and actual != expected:
                add(
                    f"{field_name}_mismatch",
                    f"{field_name} conflicts with the inspected task contract.",
                    field=field_name,
                    expected=expected,
                    actual=actual,
                )
        if inspected.group_columns and task.group_columns != inspected.group_columns:
            add(
                "group_columns_mismatch",
                "Group columns must match the inspected task contract.",
                field="group_columns",
                expected=inspected.group_columns,
                actual=task.group_columns,
            )

        required_exclusions = set(task.target_columns) | set(task.id_columns)
        missing_exclusions = sorted(required_exclusions - set(task.excluded_feature_columns))
        if missing_exclusions:
            add(
                "leakage_columns_not_excluded",
                "Targets and identifiers must be excluded from model features.",
                field="excluded_feature_columns",
                expected=sorted(required_exclusions),
                actual=task.excluded_feature_columns,
            )

        self._validate_metric(profile, task, add)
        self._validate_metric_options(task, add)
        self._validate_strategy(task, add)
        self._validate_result_format(profile, task, add)

        errors = [finding for finding in findings if finding.severity == Severity.ERROR]
        return TaskValidationReport(valid=not errors, findings=findings)

    def _validate_metric_options(self, task: TaskDefinition, add: Any) -> None:
        metric=_normalise_metric(task.primary_metric)
        allowed={
            'ndcg':{'k'},'map':{'k'},'mean_average_precision':{'k'},'mrr':{'k'},
            'roc_auc':{'average','multi_class','max_fpr'},'auc':{'average','multi_class','max_fpr'},
            'log_loss':set(),'silhouette':{'sample_size','random_state','metric'},
            'average_precision':{'average','pos_label'},'precision':{'average','pos_label'},'recall':{'average','pos_label'},
            'f1':{'average','pos_label'},'f1_macro':{'average'},'macro_f1':{'average'},'f1_micro':{'average'},
            'micro_f1':{'average'},'f1_weighted':{'average'},'weighted_f1':{'average'},
            'rmse':{'multioutput'},'mse':{'multioutput'},'mae':{'multioutput'},'r2':{'multioutput'},
        }.get(metric,set())
        if set(task.metric_options)-allowed:add('unsupported_metric_options','Unsupported metric arguments.',field='metric_options')
        try:json.dumps(task.metric_options,allow_nan=False)
        except (TypeError,ValueError):add('invalid_metric_option','Metric arguments must be finite JSON values.',field='metric_options')
        for key in ['k','sample_size']:
            value=task.metric_options.get(key)
            if value is not None and (type(value) is not int or value<1):add('invalid_metric_option',f'{key} must be a positive integer.',field='metric_options')
        if metric in {'macro_f1','f1_macro','micro_f1','f1_micro','weighted_f1','f1_weighted'}:
            expected='macro' if 'macro' in metric else 'micro' if 'micro' in metric else 'weighted'
            if task.metric_options.get('average',expected)!=expected:add('metric_option_mismatch','Averaging contradicts the named metric.',field='metric_options')
        average=task.metric_options.get('average')
        if 'average' in task.metric_options and (not isinstance(average,str) or average not in {'binary','macro','micro','weighted','samples'}):
            add('invalid_metric_option','A scalar averaging strategy is required.',field='metric_options')
        if average=='samples' and task.task_type!=TaskType.MULTILABEL_CLASSIFICATION:
            add('invalid_metric_option','samples averaging requires multilabel classification.',field='metric_options')
        if average=='binary' and task.task_type not in {TaskType.BINARY_CLASSIFICATION,TaskType.ANOMALY_DETECTION}:
            add('invalid_metric_option','binary averaging requires a binary task.',field='metric_options')
        if task.task_type==TaskType.MULTICLASS_CLASSIFICATION and metric in {'f1','precision','recall'} and average is None:
            add('ambiguous_metric_averaging','Multiclass precision/recall/F1 requires explicit averaging.',field='metric_options')
        if 'max_fpr' in task.metric_options:
            value=task.metric_options['max_fpr']
            if not isinstance(value,(int,float)) or not 0<value<=1 or (task.task_type==TaskType.MULTICLASS_CLASSIFICATION and value!=1):
                add('invalid_metric_option','max_fpr is incompatible with this task.',field='metric_options')

    def _validate_metric(self, profile: DatasetProfile, task: TaskDefinition, add: Any) -> None:
        metric = _normalise_metric(task.primary_metric)
        expected_metric = _normalise_metric(profile.task.primary_metric)
        if expected_metric and metric != expected_metric:
            add(
                "metric_mismatch",
                "The selected metric conflicts with the explicit task metric.",
                field="primary_metric",
                expected=profile.task.primary_metric,
                actual=task.primary_metric,
            )

        if task.task_type in {
            TaskType.REGRESSION,
            TaskType.MULTIOUTPUT_REGRESSION,
            TaskType.TIME_SERIES_FORECASTING,
        }:
            allowed = _REGRESSION_METRICS
        elif task.task_type in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
            TaskType.ANOMALY_DETECTION,
        }:
            allowed = _CLASSIFICATION_METRICS
        elif task.task_type == TaskType.RANKING:
            allowed = _RANKING_METRICS
        elif task.task_type == TaskType.CLUSTERING:
            allowed = _CLUSTERING_METRICS
        else:
            allowed = set()

        if allowed and metric not in allowed:
            add(
                "metric_not_applicable",
                "The metric is not known to be applicable to this prediction type.",
                field="primary_metric",
                expected=sorted(allowed),
                actual=task.primary_metric,
            )
        expected_direction = (
            MetricDirection.MINIMIZE if metric in _MINIMIZED_METRICS else MetricDirection.MAXIMIZE
        )
        if metric in allowed and task.metric_direction != expected_direction:
            add(
                "metric_direction_mismatch",
                "Metric optimization direction is incorrect.",
                field="metric_direction",
                expected=expected_direction,
                actual=task.metric_direction,
            )

    def _validate_strategy(self, task: TaskDefinition, add: Any) -> None:
        if task.task_type == TaskType.MULTILABEL_CLASSIFICATION and task.validation_strategy == ValidationStrategy.STRATIFIED_KFOLD:
            add('unsupported_multilabel_cv', 'Multilabel stratified CV is not implemented; use holdout or group_kfold.', field='validation_strategy')
        if task.task_type == TaskType.TIME_SERIES_FORECASTING or task.time_column:
            if task.validation_strategy != ValidationStrategy.TIME_SERIES_SPLIT:
                add(
                    "time_leakage_validation",
                    "Temporal data requires an ordered time-series split.",
                    field="validation_strategy",
                    expected=ValidationStrategy.TIME_SERIES_SPLIT,
                    actual=task.validation_strategy,
                )
        elif task.group_columns and task.validation_strategy not in {ValidationStrategy.GROUP_KFOLD, ValidationStrategy.HOLDOUT}:
            add(
                "group_leakage_validation",
                "Grouped data requires group-aware validation.",
                field="validation_strategy",
                expected=ValidationStrategy.GROUP_KFOLD,
                actual=task.validation_strategy,
            )
        elif task.task_type in {
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        } and task.validation_strategy == ValidationStrategy.KFOLD:
            add(
                "unstratified_classification_validation",
                "Classification should preserve class proportions across folds.",
                field="validation_strategy",
                expected=ValidationStrategy.STRATIFIED_KFOLD,
                actual=task.validation_strategy,
            )

    def _validate_result_format(self, profile: DatasetProfile, task: TaskDefinition, add: Any) -> None:
        expected = profile.task.result_format
        actual = task.result_format
        scalar_fields = (
            "file_name", "encoding", "delimiter", "include_index", "row_order",
            "one_prediction_per_id", "finite_predictions_required",
            "missing_predictions_allowed", "duplicate_ids_allowed",
            "prediction_kind",
        )
        for field_name in scalar_fields:
            expected_value = getattr(expected, field_name)
            actual_value = getattr(actual, field_name)
            if expected_value is not None and actual_value != expected_value:
                add(
                    "result_format_mismatch",
                    f"Result field {field_name} conflicts with task.md.",
                    field=f"result_format.{field_name}",
                    expected=expected_value,
                    actual=actual_value,
                )
        for field_name in ("columns", "id_columns", "prediction_columns"):
            expected_value = getattr(expected, field_name)
            actual_value = getattr(actual, field_name)
            if expected_value and actual_value != expected_value:
                add(
                    "result_columns_mismatch",
                    f"Result field {field_name} conflicts with task.md.",
                    field=f"result_format.{field_name}",
                    expected=expected_value,
                    actual=actual_value,
                )
        prediction=actual.prediction_kind
        kinds=({'label','probability'} if 'classification' in task.task_type else {'cluster'} if task.task_type==TaskType.CLUSTERING
               else {'anomaly_score','label'} if task.task_type==TaskType.ANOMALY_DETECTION else {'ranking_score','value'} if task.task_type==TaskType.RANKING else {'value'})
        if prediction is not None and prediction not in kinds:add('prediction_kind_mismatch','Output kind is incompatible with the task.',field='result_format.prediction_kind')
        if (task.target_columns and actual.prediction_columns and
                not expected.prediction_columns and actual.prediction_columns != task.target_columns):
            add(
                "prediction_target_mismatch",
                "Prediction columns must be the task target columns.",
                field="result_format.prediction_columns",
                expected=task.target_columns,
                actual=actual.prediction_columns,
            )
