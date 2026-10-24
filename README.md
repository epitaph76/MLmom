# MLmom

Агентский pipeline для табличных ML-задач: условие и CSV → обучение и сравнение моделей →
`submission.csv`. Ветка `codex/agent-first-minimal-checks` сохраняет агентскую систему
и технический минимум проверок. Исходное состояние сохранено коммитом `78f5409`.

## Запуск

Нужен Python 3.12 и зависимости проекта. Подключение задаётся через
`OPENAI_API_KEY`/`OPENAI_BASE_URL`; локальный dev-профиль также читает игнорируемый `.env`.

```bash
python -m pip install -e .
python solution.py /path/to/task.md --profile dev --run-dir artifacts/experiment-01
```

На платформе: `python solution.py task.md`. Конфигурация: `--config config.json`.
Для восстановления той же версии/конфигурации используйте прежний `--run-dir` и `--resume`.
Эксперименты этой ветки запускайте в новых каталогах.

## Работа системы

Разведчик подтверждает исполнимую постановку, Context agent готовит brief, генератор и Planner
предлагают эксперименты. Агент — валидатор плана самостоятельно разрешает исполнение или
возвращает замечания. Два кодера пишут предобработку и модель. Workers выполняют validation/HPO
и refit; внешний score выбирает Champion. Feature agent и ensemble agent сохраняют свои роли,
result_validator даёт рекомендации. Реальные ошибки возвращаются кодеру с traceback.

Удалены AST-политики импортов, сверка цитат/чисел, fitted-estimator audit, перехват fit,
обязательные smoke и export/recreate. Smoke доступен как инструмент агента.
`export_model_params` необязателен, `fitted_components` не требуется.

Сохраняются границы данных/каталогов, CPU/RAM, процессы, заданные бюджеты и ограниченные повторы,
корректность прогнозов/метрик и CSV. Новые проверки добавляем по воспроизводимым сбоям.
Подробности: [технический минимум](docs/MINIMAL_CHECKS_RU.md), [архитектура](ARCHITECTURE.md),
[роли](docs/AGENT_ROLES.md), [скилы](docs/SKILLS.md).

## Циклы и артефакты

По умолчанию работа непрерывна: `timeouts_enabled=false`, без лимита циклов/plateau.
CPU/RAM и trial-лимиты действуют всегда. Ограниченный запуск задаётся через `max_rounds`
или включённые таймауты. Repair/resume не обновляют исходные окна.

Первый цикл сравнивает бустинги; затем работают HPO, поиск признаков и следующие гипотезы.
После каждого цикла победитель обучается на всех размеченных строках для отдельного CSV.
Эта модель имеет роль `submission_refit` и не участвует в сравнении или пуле ансамблей.
Ответ лежит рядом с `task.md`, версии циклов — в `<run-dir>/cycles/rN/submission.csv`.
Сохраняются исходники, параметры, метрики, решения и traceback. Ошибка не затирает готовый CSV.

## Проверки и упаковка

Тесты используют подставной LLM и настоящее локальное обучение. Платные API-прогоны выполняются
отдельно. Плоский архив содержит `solution.py`, `requirements.txt` и `mlmom_bundle.py`.

```bash
python -m unittest discover -v
python scripts/package_solution.py --output dist/mlmom-agent-first-minimal-checks.zip
```

[Проверка ветки](VALIDATION.md). Старые архивы в `submission/` относятся к прежней строгой
реализации. [Трассировка](docs/TRACING.md), [контроль workers](docs/WORKER_SUPERVISION_RU.md),
[локальный граф Graphify](docs/GRAPHIFY.md).
