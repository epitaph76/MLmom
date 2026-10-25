# Факты исполнения и наследование — runtime v8 (основа v7)

Дополнения v8: [конфигурация checkpoint, scorer и инструменты](SKILL_CONTRACTS_V8_RU.md).

SDK/worker записывает `FitExecutionFacts(version=1)` сразу после успешного candidate fit,
до предсказания. Каждая запись имеет fit_id, stage, trial/fold, split, доступный estimator,
параметры и их источник, manifest программ с config/source_hash и число колонок.
SDK-ансамбли сохраняют записи компонентов с parent_fit_id. Произвольные внутренние
вызовы библиотек внутри Python-программы не инструментируются и не проверяются аудитом.

Атомарные `fit-execution/<fit_id>.json` остаются после ошибок и перезапуска. Worker results,
ExperimentRun, fold_results и metadata опубликованной модели связывают эти сведения.
Отсутствующие сведения — unknown/null. Пустой список программ и ноль derived_columns
обозначают наблюдаемое отсутствие SDK-признаков. Никаких дополнительных fit или проверок
агентом для получения manifest нет. Необязательный сломанный manifest не блокирует обучение.

Общий execution_context передаёт current_configuration, execution_contract,
observed_execution, measurements, configuration_diff и historical_baseline. Снимок
champion строится до Context и входит в кеш brief. Related submission fits передаются
со своей стадией submission_refit/split all-train; их ширина не приписывается внутренним folds.
Сведения доступны ролям планирования, review, ансамбля, признаков, обучения и кодирования.

## Полный план с наследованием

Если hypothesis.based_on равен текущему champion.candidate_id, контроллер берёт его
замороженные plan/run с лучшими параметрами до Plan Validator и кодирования. Пропуски,
null и пустые model_params/preprocessing сохраняют основу; model_params накладываются
по ключам, непустой preprocessing задаёт полную обновлённую спецификацию. Policies
наследуются с явно указанными изменениями. Ресурсы/бюджет нового эксперимента задаёт новый план.

`feature_set_id=null` сохраняет признаки. Ненулевой зарегистрированный ID заменяет набор.
`remove_feature_names=["gbm_oof_representation"]` удаляет ровно названную программу;
`reset_fields=["feature_set"]` очищает весь набор. reset_fields также поддерживает
model_params, preprocessing, target_transform, weight_policy, prediction_policy.
Разный backend начинает с новых model_params; смена estimator внутри backend использует
явный reset model_params. based_on=null задаёт независимый эксперимент.

Например, для добавления carat×grade interactions Planner возвращает прежний preprocessing
с дополнительным пунктом и feature_set_id=null. Все четыре программы сохраняются.
Если он явно убирает четыре программы, configuration_diff отражает удаления вместе с
добавлением interactions. Это наблюдаемая разница, а не новый запрет на эксперимент.
После fit разница дополняется исполненными параметрами и хешами исходников.

## Evidence и review

Один словарь материалов формирует evidence_refs и обработчик read_evidence. Ссылки
candidate, evaluation, execution, plan и source открываются с пагинацией. Исходники
зарегистрированных программ также доступны как source:feature_programs/<name>.py.

Review сохраняет runtime execution_context отдельно от explanation/recommendations.
Утверждение reviewer не меняет записанную модель. parent_control, cycle_incumbent и
baseline передаются раздельно: улучшение полного набора против baseline не доказывает
пользу alpha-тюнинга. Нет сопоставимого parent — нет измеренной поддержки HPO-эффекта;
новое обучение ради отсутствующей ссылки не запускается.

## Воспроизведение

`tests/fixtures/diamonds_r1_boundary.json` извлечён только из завершённого r1. Он содержит
реальные программы и сохранённое наблюдение 28+282=310 для submission_refit/all-train.
Отсутствующие manifest внутренних fits не восстанавливаются предположением.

Офлайн и живой replay запускаются в новых каталогах:

```powershell
.\.venv\Scripts\python.exe scripts/replay_diamonds_context.py --output artifacts/diamonds-v7-offline
.\.venv\Scripts\python.exe scripts/replay_diamonds_context.py --live --output artifacts/diamonds-v7-live
```

Replay вызывает только роли и чтение evidence; sandbox/обучение моделей не создаётся.
Входы, ответы, чтения ссылок и ошибки API сохраняются. Недоступный API означает incomplete.
Новые исполнения используют session runtime_contract_version=8. Старые каталоги не
перезаписываются, inference artifacts сохраняются; ModelSpec v2 и FeatureSet v1 не меняются.
Это тесты разработки, без новых обязательных проверок перед каждым экспериментом.
