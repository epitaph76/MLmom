# Скилы в экспериментальной ветке

[Текущие правила](MINIMAL_CHECKS_RU.md): карточки сохранены по ролям; совместимость
оценивают агенты, а ошибки библиотек возвращаются при исполнении.

# Runtime skills по агентам

Ensemble Agent 1.1.0 получает compare_oof_components и сохраняемый Diversity Evidence:
парные ошибки, measured blend gain и стоимость refit с явными unknown. Инструмент
доступен в согласованном runtime; одного обновления карточки недостаточно.
Подробности: [ENSEMBLE_DIVERSITY_RU.md](ENSEMBLE_DIVERSITY_RU.md).

Реестр рекурсивно читает skills/<роль>/<skill_id>/manifest.json. ID существующих скилов сохранены. Список, инструменты чтения и начальный контекст ограничены ролью; чтение чужого скила отклоняется.

<!-- skill-registry:start -->
Всего карточек: 33; ready: 33; placeholder: 0.

| Владелец | Карточки | Автоматический набор |
|---|---|---|
| coding_agent | autogluon, catboost, ensembles, flaml, imbalance, lightautoml, lightgbm, sklearn, statsmodels, torch, xgboost | candidate_sdk, metrics |
| context_agent | context_agent | context_agent |
| ensemble_agent | ensemble_agent | ensemble_agent |
| feature_agent | feature_agent | feature_agent |
| hypothesis_generator | hypothesis_generator | hypothesis_generator |
| plan_validator | validation | validation, metrics |
| planner | planner | planner |
| preprocessing_agent | categorical_encoding, feature_engineering, feature_selection, missing_values, numeric_transforms, preprocessing, preprocessing_core, schema_quality, temporal_group_structure, text_features | candidate_sdk, preprocessing, preprocessing_core |
| reconnaissance_agent | polars_sql, reconnaissance_agent | reconnaissance_agent, polars_sql |
| result_validator | result_validator | result_validator |
| shared | candidate_sdk, metrics | - |
| training_agent | hpo | metrics, hpo |
<!-- skill-registry:end -->

COMMON.md входит в candidate_sdk. Дополнительные скилы выбираются среди принадлежащих роли; REFERENCE.md подключается кодерам по необходимости. Feature Agent получает свой исследовательский REFERENCE.md в обоих proposal-входах. Общие пакеты доступны ролям, но не загружаются всем автоматически. Карточка ensembles доступна coding_agent: blending и OOF stacking разрешены. Компоненты и meta-model реализуются собственным кодом кандидата с сохранением изоляции фолдов.

Разведчик автоматически получает `polars_sql` при начальном анализе и повторной
разведке. Карточка описывает SQLContext закреплённой версии Polars 1.26.0:
схемы, NULL, агрегаты, группировку вычисленных бинов через подзапрос,
многоколоночный DISTINCT, дубликаты, пересечение ключей и сравнение train/test.
Все SQL-блоки из самой карточки выполняются через CSV worker в
`tests/test_polars_sql_skill.py`; отдельная проверка подтверждает передачу обеих
карточек в LLM-контекст и их включение в плоский ZIP. Новые инструкции применяются
к новым запускам; уже работающий Docker использует сохранённую копию исходников.

SkillRegistry.load(..., role=...) и list(role=...) применяются на LLM-границах. Внутренняя проверка совместимости и старые интеграции могут использовать административную загрузку без роли; общий каталог в LLM не передаётся. Лимит основных инструкций — 24 000 символов на роль, с примерами для кодирования — 48 000. COMMON включается в candidate_sdk и SHA256 один раз. Версии и хеши сохраняются в кандидатах.

Загрузка скилов проверяет существование ID и границы роли. Тип задачи и зависимости
в метаданных помогают агенту выбрать технику, но не блокируют план. Полные тексты загружаются при передаче задания конкретному
агенту; его лимит контекста сохраняется. Большой суммарный список техник ансамбля
сам по себе не означает переполнение контекста отдельного исполнителя.

## Проверка реестра

`python scripts/check_skill_docs.py` проверяет количество, статусы, владельцев и автоматические наборы. `--write` обновляет генерируемую таблицу. Генератор гипотез, планер и валидатор результата получают специализированные карточки через существующие автоматические наборы.

Проверки: tests/test_skills.py, tests/test_agent_roles.py, tests/test_packaging.py. Упаковщик рекурсивно включает skills. Graphify, его окружение и графы не входят в submission.

## Каталог техник для планирования

technique_skills передаёт генератору гипотез, Planner и валидатору только метаданные
совместимых готовых скилов предобработки, imbalance и ensembles: ID, название, роль и типы задач.
Планировщик описывает нужные операции и включает соответствующие ID в plan.skill_ids.
Python передаёт полные инструкции выбранных карточек их владельцам: преобразования X —
preprocessing_agent, sampling, target transforms и thresholds — coding_agent.
Каталог не расширяет право Planner читать инструкции чужой роли. preprocessing_core
загружается агенту предобработки автоматически; остальные новые техники — по выбору плана.

Приняты восемь новых карточек и обновлён imbalance из ветки preprocessing-skills
(исходный коммит 9b500446ebb06837729eec36546ae66c269106e7).
[Результат ревью и адаптации](PREPROCESSING_SKILLS_REVIEW.md).

Три decision-role карточки из коммита тиммейта 7235080 приняты с адаптацией:
версии 1.2.0, сохранены правила полного CV/refit и независимого сравнения.
В planner добавлен явный формат search_space; ансамбли разрешены и доступны
в каталоге техник. Роль context_agent остаётся подключённой; её альтернативный
прямой путь пока исследован только локальным сравнением входов.

## Согласование v8

Полные checkpoint configurations, MCC, original-scale scorer, новые инструменты ролей и
проверенные estimator-only примеры описаны в [SKILL_CONTRACTS_V8_RU.md](SKILL_CONTRACTS_V8_RU.md).
ready означает готовый пакет инструкций; поддержку конкретного исполнения подтверждают отдельные тесты.

## Иерархическое исследование сущностей

feature_agent 1.2.0 получает последовательность «структура → инварианты → составные ключи и
linkage → отношения/время → инференс → ablation по сегментам». Справочник содержит порядок
следующих диагностических запросов, очередь гипотез и критерии покрытия, неоднозначности
и стоимости. Precision/recall linkage оцениваются по независимым labels; без них точность
остаётся неизвестной. Скаут 2.4.0 передаёт структурные наблюдения и гипотезы для повторной
проверки Feature Agent на разрешённой части train.

В каталоге остаётся 33 карточки; REFERENCE.md теперь 14. Featuretools/Splink описаны как
опциональные реализации методов; они не добавлены в обязательные зависимости. Детали,
исполняемые SQL-примеры и проверки: [ENTITY_DISCOVERY_PROCEDURE_RU.md](ENTITY_DISCOVERY_PROCEDURE_RU.md).

## EDA, конструирование и вклад признаков

Разведчик получает обязательную EDA-матрицу из десяти тем в eda_research_plan; статусы и
незавершённые проверки сохраняются. Непроверенные темы означают partial для исследования,
но не отменяют подтверждённую задачу. record_eda_status сохраняет промежуточное покрытие,
save_observation сохраняет поддержку, знаменатели и alternative_checks в agent_notes.

feature_engineering 1.1.0 описывает шесть семейств семантических преобразований с отдельными
baseline comparisons; feature_selection 1.1.0 — repeated/joint permutation importance,
drop-column ablation и fold stability. Их REFERENCE.md передаются агенту предобработки
через существующую загрузку выбранных карточек; недоступные сравнения явно остаются
unmeasured и передаются владельцу эксперимента. Подробности: [RESEARCH_METHODOLOGY_RU.md](RESEARCH_METHODOLOGY_RU.md).
