# Архитектура MLmom: агентская ветка

Текущий контракт: [агенты и технический минимум](docs/MINIMAL_CHECKS_RU.md).
Запуск идёт через `solution.py`; LangGraph связывает роли и сохраняет checkpoints в SQLite.

## Разделение ответственности

| Компонент | Ответственность |
|---|---|
| Inspector и CSV tools | Профиль и измерения на зарегистрированных read-only таблицах |
| Разведчик и Context agent | Исполнимая задача, наблюдения и смысловой brief |
| Генератор, Planner, plan_validator | Гипотезы, задания и агентский допуск |
| preprocessing_agent и coding_agent | Раздельные модули преобразований и модели |
| SDK | Композиция, target/weight/output policies, сохранение и загрузка |
| Scheduler и worker | CPU/RAM, процессы, validation/HPO, refit и prediction |
| training_agent | Checkpoints, continue/stop/refit, предложения trials |
| feature_agent и ensemble_agent | Поиск признаков и композиции сохранённых моделей |
| Evaluation и Submission | Измеренные метрики, Champion и CSV каждого цикла |
| result_validator | Рекомендации о реализации и измеренном эффекте |

## Путь эксперимента

Профиль → задача → контекст → гипотеза → план → агентский review → исходники →
синтаксис/интерфейс → внутреннее обучение/validation → refit → внешний score →
Champion → полный submission refit и CSV → рекомендации следующему циклу.

Python не проверяет обоснование решения plan_validator. Кодер может запросить smoke;
обязательного синтетического обучения нет. Ошибки worker содержат исходную строку и traceback.
Для каждого fold/refit создаются свежие модель и предобработка. Внешние метки не входят в
FitContext. Submission refit хранится отдельно и не используется как сравнительная модель.
Ансамбли обучают новые экземпляры компонентов; stacking использует OOF.

## Интерфейсы и хранение

`candidate.py` предоставляет `create_candidate(config)` и `load_candidate(artifact_dir)`.
Объект реализует `fit`, `predict`, `save`. `preprocessing.py` предоставляет
`create_preprocessor(config)` с `fit_transform` и `transform`.

Необязательный `export_model_params()` возвращает JSON-параметры для refit; иначе используются
параметры trial. Полнота экспорта и соответствие estimator плану не проверяются.
`ModelSpec` описывает параметры/политики, `fitted_components()` не требуется.
Старые `execution_report` читаются как optional-поля; новые runs не создают отчёты аудита.
Дополнительные пояснения сохраняются в `agent_notes`.

ExperimentStore хранит планы, версии, метрики, решения и состояние циклов. Хеши защищают
snapshot/resume от подмены. Атомарная публикация сохраняет последний готовый CSV при ошибках.
CPU/RAM, процессы, бюджеты, пути/данные и формат выхода остаются техническими границами.
AST-проверки исходника не являются песочницей.

Навигация: `mlmom/pipeline.py` — оркестрация, `mlmom/worker.py` — исполнение,
`mlmom/sdk.py` — интерфейс и композиция. Граф зависимостей: [GRAPHIFY.md](docs/GRAPHIFY.md).
