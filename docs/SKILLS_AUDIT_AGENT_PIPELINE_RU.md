# Аудит runtime skills: агентский пайплайн

Проверены все 33 карточки, manifest.json, связанные REFERENCE.md и автоматические наборы ролей. Исправлены устаревшие требования аудита/экспорта, обязательный boosting baseline, владелец target transformations, ансамблевые рецепты, ранний Feature Engineer и бюджеты. Примеры SQL проверяются на Polars 1.26.0; справочные примеры кодеров адаптируются к разделению preprocessing/estimator/SDK.

| Skill | Owner | Result |
|---|---|---|
| autogluon | coding_agent | Updated |
| catboost | coding_agent | Reviewed; compatible |
| ensembles | coding_agent | Updated |
| flaml | coding_agent | Updated |
| imbalance | coding_agent | Reviewed; compatible |
| lightautoml | coding_agent | Updated |
| lightgbm | coding_agent | Updated |
| sklearn | coding_agent | Reviewed; compatible |
| statsmodels | coding_agent | Reviewed; compatible |
| torch | coding_agent | Updated |
| xgboost | coding_agent | Reviewed; compatible |
| context_agent | context_agent | Reviewed; compatible |
| ensemble_agent | ensemble_agent | Updated |
| feature_agent | feature_agent | Updated |
| hypothesis_generator | hypothesis_generator | Updated |
| validation | plan_validator | Updated |
| planner | planner | Updated |
| categorical_encoding | preprocessing_agent | Reviewed; compatible |
| feature_engineering | preprocessing_agent | Reviewed; compatible |
| feature_selection | preprocessing_agent | Reviewed; compatible |
| missing_values | preprocessing_agent | Reviewed; compatible |
| numeric_transforms | preprocessing_agent | Updated |
| preprocessing | preprocessing_agent | Updated |
| preprocessing_core | preprocessing_agent | Reviewed; compatible |
| schema_quality | preprocessing_agent | Reviewed; compatible |
| temporal_group_structure | preprocessing_agent | Reviewed; compatible |
| text_features | preprocessing_agent | Reviewed; compatible |
| polars_sql | reconnaissance_agent | Updated |
| reconnaissance_agent | reconnaissance_agent | Reviewed; compatible |
| result_validator | result_validator | Reviewed; compatible |
| candidate_sdk | shared | Updated |
| metrics | shared | Reviewed; compatible |
| hpo | training_agent | Updated |

Общий дедлайн отключён по умолчанию. Skills и manifests обновляются вместе с изменением контракта; новая версия runtime не возобновляет старые run-каталоги. Сведения о непроверенном backend не являются автоматическим запретом.
