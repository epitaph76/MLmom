# Агенты и контроль обучения

Текущий контракт экспериментальной ветки: [минимальные проверки](MINIMAL_CHECKS_RU.md).

| Роль | Ответственность |
|---|---|
| reconnaissance_agent | Читает условие, подтверждает исполнимую постановку, исследует зарегистрированные CSV и сохраняет наблюдения |
| context_agent | Сохраняет смысл условия, выбирает находки и их приоритеты для Planner |
| hypothesis_generator | Предлагает гипотезы с учётом истории и отказов |
| planner | Оформляет модель, предобработку, параметры, ресурсы и пространство HPO |
| plan_validator | Самостоятельно принимает или отклоняет план; замечания возвращаются Planner |
| preprocessing_agent | Пишет preprocessing.py с fit_transform/transform и сериализуемым состоянием |
| coding_agent | Пишет candidate.py с fit/predict/save и create_candidate/load_candidate |
| training_agent | Получает checkpoints после initial fit, пакета HPO или ошибки; выбирает continue/stop/refit |
| feature_agent | Исследует разрешённый train через CSV tools, пишет Python-программы признаков или DSL и получает внутренние результаты |
| ensemble_agent | Выбирает сохранённые компоненты для ансамблей |
| result_validator | Даёт рекомендательную оценку реализации и измеренного эффекта |

Python владеет split, Scheduler, ресурсами, вычислением метрик, хранением и CSV.
SDK компонует предобработку и модель, выполняет target/weight/output policies.
Validation и refit создают свежие экземпляры. Внешние метки и LLM-ключи workers не передаются.

Plan validator получает один план и сведения о соседних вариантах; его решение не проходит
повторную машинную проверку цитат/обоснования. Result validator не меняет выбор Champion.
Наблюдения и brief сохраняются без проверки точного совпадения чисел, цитат или приоритетов.

TrainingDecision задаёт continue, stop или refit и additional_trials для продления неизменённого
эксперимента. Конкретные числа выбирает Optuna; next_params больше не исполняется.
При отсутствии дедлайна remaining_seconds=null. Явный пользовательский trial_limit сохраняется;
при сбое роли используется программная политика. Поиск заканчивается refit выбранных параметров.

Первый набор моделей и priority выбирают агенты для задачи. Feature agent предлагает признаки
сразу после Context agent, затем runtime измеряет их на замороженном контроле до длительного HPO.
Независимая подготовка выполняется параллельно; общий scheduler учитывает все CPU/RAM leases.
Первый успешный baseline выпускает full-train результат до окончания остальных экспериментов.
Полный refit для CSV имеет роль submission_refit и не входит в сравнение моделей.
Повторные попытки ограничены; resume сохраняет выполненную работу и историю ошибок.
Ошибки исходника/обучения передаются кодеру с source line и traceback для исправления.
Ансамбли исполняют только явно выбранные агентом рецепты. Валидатор получает execution_contract
с актуальным DSL и порядком композиции. Остановка обозначается interrupted/external_stop отдельно
от наличия результата; опубликованный full-train CSV сохраняется.

Детали инструментов/скилов: [SKILLS.md](SKILLS.md). Процессы и ресурсы:
[WORKER_SUPERVISION_RU.md](WORKER_SUPERVISION_RU.md), [WORKER_RUNTIME.md](WORKER_RUNTIME.md).

## Контракт групп исполнения v4

Planner возвращает PlannerPlan без search_group_id/search_group_kind/parent_hypothesis.
Контроллер создаёт SearchGroupPlan, связывает варианты до plan review и сохраняет
HypothesisPlan с производными полями для scheduler/store. Историческое происхождение
эксперимента остаётся в based_on/based_on_hypothesis_id. Ошибки отдельного рецепта
ансамбля сохраняются в ensemble_recipe_failed; остальные выбранные рецепты продолжаются.
Версия групп введена в v4; текущая session runtime_contract_version=8.
Детали: [CONTROLLER_GROUPS_RU.md](CONTROLLER_GROUPS_RU.md).

## Длительности и замены признаков v5

Исторические длительности являются справочной информацией, а не программным
условием допуска. Реальные дедлайны и CPU/RAM исполняет scheduler. Feature agent
может вернуть replace с удалениями и добавлениями: один итоговый набор измеряется
относительно исходного без промежуточного требования улучшения. Обе стороны
сохраняются в evidence. Подробности: [ADVISORY_COSTS_AND_REPLACEMENT_RU.md](ADVISORY_COSTS_AND_REPLACEMENT_RU.md).

## Python-признаки v6

Feature Agent пишет programs с create_features(config), fit_transform(X,y,context),
transform(X). DSL необязателен, лимита 16 колонок нет. Разрешены supervised-признаки
на fold train с корректным исключением собственной метки/cross-fitting. SDK исполняет
и сохраняет модули вместе с моделью. Подробности: [PYTHON_FEATURE_PROGRAMS_RU.md](PYTHON_FEATURE_PROGRAMS_RU.md).

## Факты исполнения v7

Все роли получают актуальную конфигурацию и наблюдаемые сведения SDK отдельно от
исторического baseline и мнений reviewer. Полные планы наследуют champion до review;
удаление признаков задаётся явно. Ссылки evidence формируются из читаемых материалов.
Подробности и replay diamonds: [EXECUTION_FACTS_RU.md](EXECUTION_FACTS_RU.md).

## Согласованные скиллы и контракты v8

TrainingCheckpoint разделяет sampled_params и полную effective_configuration выбранного
trial; best_params включает сохранённые параметры refit. MCC поддерживается для single-label
классификации. Опциональный validation_scorer считает primary metric на исходной шкале,
с опциями и весами; preprocessing, feature programs и внутренние base OOF не получают его.
Preprocessing Agent читает свои карточки, inspect_api и может проверить собственный модуль
без estimator. Feature Agent исследует только зарегистрированный train своего цикла;
усечённый профиль дополнен пагинацией и читаемыми source_ref. read_query_result читает
строки/столбцы полных сохранённых результатов, read_evidence — страницы metadata. Подробности:
[SKILL_CONTRACTS_V8_RU.md](SKILL_CONTRACTS_V8_RU.md). Для v8 нужен новый run directory.

Feature Agent 1.2.0 дополнен процедурой исследования сущностей и отношений с очередью
гипотез, отдельной оценкой FP/FN linkage, as-of доступностью и сегментными ablation.
Его REFERENCE.md передаётся автоматически и раннему proposal, и следующему поисковому
вызову, с версией/хешем пакета. Разведчик передаёт только структурные наблюдения, не fitted
maps. Схемы API v8 и агентское методологическое одобрение сохраняются; новый admission gate
не вводится. Подробности: [ENTITY_DISCOVERY_PROCEDURE_RU.md](ENTITY_DISCOVERY_PROCEDURE_RU.md).

Разведчик 2.4.0 получает eda_research_plan и сохраняет покрытие десяти тем через
record_eda_status/agent_notes.eda_matrix. Неполное покрытие отражается в partial и
unfinished_checks, не блокируя обучение по подтверждённой задаче. Наблюдения сохраняют
размер подгруппы, знаменатель и выполненные проверки альтернатив в agent_notes.
Агент предобработки получает расширенные feature_engineering/feature_selection 1.1.0,
реализует назначенную логику и сообщает необходимые сравнения владельцу эксперимента;
он не получает outer labels и не обучает predictive estimator ради диагностики.
Подробности: [RESEARCH_METHODOLOGY_RU.md](RESEARCH_METHODOLOGY_RU.md).

Ensemble Agent 1.1.0 получает compare_oof_components: структурированный Diversity Evidence
по сохранённым inner OOF (корреляция/пересечение ошибок, прирост фиксированного среднего
по primary metric, стоимость refit). Отчёт сохраняется и передаётся Planner/plan validator.
Недоступные/несовместимые данные означают непроверенную гипотезу; историческая стоимость
не становится порогом допуска. Подробности: [ENSEMBLE_DIVERSITY_RU.md](ENSEMBLE_DIVERSITY_RU.md).
