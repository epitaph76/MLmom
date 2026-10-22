# Runtime skills по агентам

Реестр рекурсивно читает skills/<роль>/<skill_id>/manifest.json. Всего 27 готовых пакетов и три заглушки. ID существующих скилов сохранены. Список, инструменты чтения и начальный контекст ограничены ролью; чтение чужого скила отклоняется.

| Папка | Скилы |
|---|---|
| preprocessing_agent | preprocessing, preprocessing_core, feature_engineering, categorical_encoding, missing_values, numeric_transforms, feature_selection, schema_quality, temporal_group_structure, text_features |
| coding_agent | sklearn, catboost, lightgbm, xgboost, statsmodels, torch, flaml, autogluon, lightautoml, imbalance, ensembles |
| context_agent | context_agent: смысл задачи и ранжирование для Planner |
| reconnaissance_agent | reconnaissance_agent: требования и исследование CSV |
| training_agent | hpo: контроль обучения на checkpoint |
| plan_validator | validation |
| shared | COMMON.md, candidate_sdk, metrics |

Автоматические наборы: preprocessing_agent — candidate_sdk, preprocessing и preprocessing_core; coding_agent — candidate_sdk и metrics; training_agent — metrics и hpo; plan_validator — validation и metrics. Разведчик получает готовые инструкции подтверждения требований и CSV-исследования; агент контекста — инструкции подготовки briefing; остальные роли — свою заглушку. Дополнительные скилы выбираются среди принадлежащих роли; REFERENCE.md подключается кодерам по необходимости. Общие пакеты доступны ролям, но не загружаются всем автоматически.

SkillRegistry.load(..., role=...) и list(role=...) применяются на LLM-границах. Внутренняя проверка совместимости и старые интеграции могут использовать административную загрузку без роли; общий каталог в LLM не передаётся. Лимит основных инструкций — 24 000 символов на роль, с примерами для кодирования — 48 000. COMMON включается в candidate_sdk и SHA256 один раз. Версии и хеши сохраняются в кандидатах.

## Где нет специализированных скилов

Файлы со status=placeholder: агенты работают по runtime-инструкциям и схемам, дополнительных предметных инструкций пока нет.

- skills/result_validator/result_validator/SKILL.md
- skills/hypothesis_generator/hypothesis_generator/SKILL.md
- skills/planner/planner/SKILL.md

Проверки: tests/test_skills.py, tests/test_agent_roles.py, tests/test_packaging.py. Упаковщик рекурсивно включает skills. Graphify, его окружение и графы не входят в submission.

## Каталог техник для планирования

technique_skills передаёт генератору гипотез, Planner и валидатору только метаданные
совместимых готовых скилов предобработки и imbalance: ID, название, роль и типы задач.
Планировщик описывает нужные операции и включает соответствующие ID в plan.skill_ids.
Python передаёт полные инструкции выбранных карточек их владельцам: преобразования X —
preprocessing_agent, sampling, target transforms и thresholds — coding_agent.
Каталог не расширяет право Planner читать инструкции чужой роли. preprocessing_core
загружается агенту предобработки автоматически; остальные новые техники — по выбору плана.

Приняты восемь новых карточек и обновлён imbalance из ветки preprocessing-skills
(исходный коммит 9b500446ebb06837729eec36546ae66c269106e7).
[Результат ревью и адаптации](PREPROCESSING_SKILLS_REVIEW.md).
