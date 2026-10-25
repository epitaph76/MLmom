# Согласование скиллов и runtime v8

Реализация закрывает замечания R01–R11 из аудита 25 октября 2026 года и актуализирует
фикстуры rotating holdout. Старый аудит, runs и их доказательства сохраняются.

## Интерфейсы

TrainingCheckpoint.sampled_params содержит выбор Optuna; effective_configuration — полную
конфигурацию выбранного trial. best_params — полный экспорт для refit. HPO передаёт номер
лучшего trial; worker читает его selection.json и ModelSpec, сохраняя фиксированные параметры,
политики и выбранные iteration counts. Контекст не требует дополнительного fit. Без успешного
trial наблюдаемые факты остаются неизвестными.

FitContext.validation_scorer(predictions, kind) — опциональный сервис текущего внутреннего
holdout. kind/metric/direction описывают primary metric. Передаются сырые предсказания estimator;
SDK применяет target inverse и output policy ровно один раз и оценивает на исходной шкале
с исходными validation weights/options. Эти замыкания и FitContext не сохраняются в модели.
Preprocessing, feature programs и base OOF fits получают validation_scorer=None.

Python remaining_seconds остаётся infinity при отключённых таймаутах; JSON бюджеты используют
null. Native-бюджет библиотеки конечен. PreprocessingSpec.categorical_features — имена выходных
столбцов DataFrame либо индексы матрицы; формат native categories зависит от библиотеки.

## Метрики и обучение

metric_contracts хранит поддерживаемые имена/алиасы, типы задач и опции. MCC доступна для
binary/multiclass, использует labels и sample weights, максимизируется. Неизвестная метрика
не подтверждается и не заменяется proxy. FLAML передаёт train/validation/refit weights и
использует точный original-scale scorer. LightGBM/XGBoost/MLP references используют его
для stopping. Если точный callback конкретного рецепта недоступен, выполняется fixed fit;
опциональный stopping_info сохраняет объяснение в FitExecutionFacts.stopping.

validation_strategies содержит проверенные reference-протоколы. Holdout проверяется для
FLAML/AutoGluon/LightAutoML. Пустой список означает непроверенность, а не машинный запрет.
AutoGluon/LightAutoML references явно описывают границы невзвешенных native objectives;
другие задачи могут реализовывать собственный точный рецепт. Политики можно искать через
policy_params.<policy>.<field>; это не estimator constructor kwargs.

## Инструменты

Feature Agent получает list_tables/preview_csv/query_csv/correlate_csv для зарегистрированных
разрешённых train columns текущего цикла. Source references и query cache сохраняются;
read_evidence читает сохранённые ответы по страницам; read_query_result листает полный
сохранённый Parquet по строкам и столбцам, включая данные за пределами первого preview.
Ссылка разрешается контроллером по source_ref этой сессии; произвольные пути не принимаются.
Holdout/test не регистрируются.
Профиль сообщает total_columns, columns_truncated и способ получения остальных столбцов.
Исследовательские статистики помогают выбрать гипотезу; обучаемые карты вычисляются в folds.

Preprocessing Agent получает list_skills/read_skills, inspect_api, чтение собственного исходника
и optional check_preprocessing(source). Проверка выполняет только синтетические transforms и
serialization без estimator. Дополнительно прочитанные карточки входят в skill_versions.
Обязательный smoke и дополнительные estimator/provenance gates не вводятся.

## Проверка и доказательства

tests/test_skill_contract_v8.py проверяет воспроизведения, полную конфигурацию, метрики,
scorer/label scope, границы CSV tools и provenance динамически прочитанных карточек.
scripts/verify_skill_contracts_v8.py исполняет исходники из REFERENCE.md, проверяет точные
callbacks, weights, relocation, class order и fixed refit. Полные записи лежат в новых
artifacts/skill-contracts-v8/ каталогах. Реальные LLM-проверки используют существующий
DeepSeek dev-профиль и общий предел 20 HTTP attempts, включая retries.

| Замечание | Изменение | Основная проверка | Результат |
|---|---|---|---|
| R01 | Материализация selected configuration из sampled/selection/ModelSpec | CheckpointConfigurationTests; import/failure/resume tests | Закрыто; фиксированные параметры, policies и export counts сохранены, лишних fit нет |
| R02 | Единый каталог и MCC | MetricContractV8Tests; weighted MCC в трёх библиотеках | Закрыто; string labels, multiclass, отрицательные/константные scores и неизвестные сочетания проверены |
| R03 | FLAML train/validation/refit weights | NativeWeightContractTests; реальные native cases | Закрыто; проверяются разные веса текущих строк при search, validation и refit |
| R04 | SDK primary scorer, callbacks, fixed fallback и причины | ValidationScorerTests; log-scale library cases | Закрыто; контрпример с обратным порядком моделей в log-шкале воспроизведён |
| R05 | Проверенные holdout paths; unknown не veto | Native reference cases и planning tests | Закрыто; FLAML/AutoGluon/LightAutoML holdout подтверждены с указанными границами |
| R06 | Backend-specific categories и metadata | Native category/OOV cases трёх boosters | Закрыто; преобразования и прогнозы совпадают после переноса/save/load |
| R07 | Train-scoped tools, refs, полная пагинация | RoleToolTests и live Feature Agent | Закрыто; запрет чужих refs/таблиц, восстановление refs и колонка f100 реально проверены |
| R08 | Preprocessor API/cards/optional transform probe | RoleToolTests и live preprocessing | Закрыто; чужие карточки недоступны, прочитанные версии входят в candidate |
| R09 | Явное различие infinity/null | Optional timeout tests и SDK docs | Закрыто; публичный Python API сохраняется |
| R10 | 11 актуальных references; factories, persistence/refit | Library matrix; export edge cases; packaging | Закрыто в области конкретных рецептов, без обещания всех режимов upstream |
| R11 | Поддержка policy search routes в HPO инструкциях | Checkpoint/policy tests и live Training Agent | Закрыто; пространство поиска из checkpoint не переписывается |

Для v8 создаётся новый run directory; resume v7 не переносится автоматически. Версии изменённых
карточек повышены вместе с исходниками и документацией: проверены 33 карточки, изменены
19 пакетов инструкций, обновлены все 11 REFERENCE.md. Максимум регрессионных instructions
среди ролей — 22 458 символа при лимите 24 000; references имеют отдельный лимит 48 000.

## Приёмка и регрессии

Полный discovery выполнил **399 тестов**: 0 runtime errors, 1 failure, 1 штатный skip
(Linux boot-time regression на Windows). Единственный failure — устаревшее ожидание
`tools is None` для preprocessing Agent в short_cycles. Оно заменено проверкой наличия
нужных инструментов; требование direct-first без инструментов для model coder и прежние
проверки лимитов сохранены. После этого и окончательных изменений повторены **59 затронутых
тестов — все успешно**. Итоговая совокупность discovery и целевых повторов: 399 tests,
0 failures, 0 errors, 1 штатный skip. Полный discovery повторно не запускался: после
исправлений повторялись только затронутые проверки, как предусмотрено планом.

Шесть исходных errors из аудита устранены отдельно: два места создания EvaluationResult
в test_rotating_holdout теперь содержат обязательные evaluation-поля. Assertions результатов
этих тестов сохранены. Исправление short_cycles относится к новому интерфейсу инструментов,
а не к этим шести исходным ошибкам.

Подтверждён **41 основной библиотечный сценарий** и ещё **3 проверки export counts**.
Окончательные проверки выявили и закрыли экспорт запрошенного вместо фактического числа
шагов в CatBoost/LightGBM/PyTorch: при остановке после одного дерева/эпохи теперь экспортируется
1, а не 8. Затронутые библиотечные сценарии повторены. Предыдущие ошибки callback XGBoost
и pickle PyTorch также сохранены в первых журналах и имеют успешные исправленные повторы.

Основные доказательства находятся в `artifacts/skill-contracts-v8/`: `full-suite-1/tests.json`,
`full-suite-1/tests.log`, `affected-final.log`, `skills-inventory.json`, итоговая
`acceptance.json` и library `summary.json` в отдельных новых каталогах. В acceptance.json
зафиксированы исходный результат полного discovery, исправленный повтор, hashes архивов
и ссылки на каждый выбранный успешный библиотечный сценарий; первоначальные failures
не удаляются и не переписываются.

## Подтверждённая область библиотечных рецептов

Результат reference-проверки относится к конкретному примеру и параметрам, а не ко всем
возможностям установленной библиотеки. Проверяются fit, predict, JSON export параметров,
save/load после переноса каталога и удаления зависимости от прежнего рабочего каталога,
порядок строк/классов и новый fixed refit. Модели во всех случаях малы; эти проверки
подтверждают контракты исполнения, а не качество или скорость на больших данных.

| Backend | Подтверждённые сценарии |
|---|---|
| sklearn 1.5.2 | Regression, binary, multiclass; estimator-only ExtraTrees |
| LightGBM 4.6.0 | Три single-target задачи, weighted MCC binary/multiclass, log1p, native categories/OOV |
| CatBoost 1.2.8 | Три single-target задачи, веса, log1p с fixed fallback, native string categories/OOV |
| XGBoost 2.1.4 | Три single-target задачи, weighted MCC binary/multiclass, log1p, native categories/OOV |
| FLAML 2.3.6 | Weighted holdout search и frozen weighted refit для трёх задач; MCC и log1p primary objective |
| statsmodels 0.14.0 | Unweighted OLS, перенос inference state, fixed refit |
| AutoGluon 1.3.1 | Docker: regression/binary/multiclass, explicit GBM, holdout, frozen refit_full |
| LightAutoML 0.4.2 | Docker: regression/binary/multiclass, один untuned lgb, holdout, fixed refit |
| PyTorch 2.5.1+cpu | Docker: regression, multioutput, binary, multiclass, multilabel, веса, log1p, state_dict persistence |

Для AutoGluon/LightAutoML подтверждён именно невзвешенный рецепт с identity policies и
default metric options. Это явно описано в карточках. Для иной primary metric, весов или
преобразований требуется собственная точная реализация либо прямой fixed estimator.
Native CV, групповые native search, все семейства AutoGluon и все пресеты LightAutoML
этим прогоном не сертифицируются. Отсутствие такой проверки остаётся неопределённостью,
которую оценивает агент; новый программный запрет не добавляется.

## Живые роли и расход API

DeepSeek `deepseek/deepseek-v4.1-flash-1017`: 13 HTTP attempts из общего предела 20,
включая любые повторы. Получено 13 ответов; usage: 88 589 input tokens, 5 365 output tokens,
93 954 total tokens. Ключи очищены существующим TraceSink; архив и журналы дополнительно
проверены на присутствие настроенного ключа. Новые обращения после завершения диагностики
не выполняются. Лимит фиксируется перед отправкой каждого HTTP attempt в общем
`artifacts/skill-contracts-v8/llm-ledger.json`, который сохраняется между запусками.

| Роль | Запросы | Наблюдение |
|---|---:|---|
| Feature Agent | 7 | Вызвал list_tables, preview_csv, query_csv, correlate_csv; написал Python-программу, прошедшую настоящий fit и отдельную загрузку/predict |
| Preprocessing Agent | 4 | Вызвал read_skills, inspect_api, optional check_preprocessing; код прошёл fit и отдельную загрузку/predict; прочитанные версии записаны в candidate |
| Training Agent | 1 | Верно назвал n_estimators=120, max_depth=8, min_samples_leaf=2 и tail_aware; выбрал refit при исчерпанном trial limit |
| Plan validator | 1 | Принял исполнимый fixed план binary MCC; замечания о конкретизации estimator/типов столбцов остались предупреждениями |

Две ошибки первой диагностической обвязки сохранены в исходных журналах: standalone
preprocessing source не имел маркера, который обычно добавляет coding graph, а тестовый
TrainingCheckpoint содержал лишние metric/direction. Первый пример исправлен и сохранённая
программа Feature Agent проверена повторно без нового LLM-вызова. Training Agent вызван
после исправления fixture. Эти сбои не скрыты и не представлены как отказы runtime.

Доказательства: `llm-1/trace.jsonl`, `llm-1/trace_payloads/`,
`llm-feature-replay-2/summary.json`, `llm-training-2/summary.json` внутри нового каталога
`artifacts/skill-contracts-v8/`. Первый `llm-1/summary.json` сохраняет первоначальные ошибки;
итог следует проверять вместе с исправленными повторными сценариями.
