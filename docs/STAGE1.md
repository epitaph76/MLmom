# Анализ данных и постановка задачи

## Data Inspector
`mlmom.data_inspector` — встроенный узел общего пайплайна, который читает train/словари/условие и сохраняет профиль, компактный JSON и Markdown без сырых строк. Ниже приведены команды отдельного диагностического запуска из корня репозитория.

```bash
python -m mlmom.data_inspector /path/to/dataset --output artifacts/profile.json --llm-output artifacts/context.json --report-output artifacts/report.md
```

## Manifest и overrides
`dataset_manifest.json` задаёт train, dictionaries, task_description, targets, IDs, task type, metric, time/group/weight columns; пример находится в [dataset_manifest.example.json](../dataset_manifest.example.json). Явные CLI overrides приоритетнее manifest, а manifest приоритетнее эвристик.

```bash
python -m mlmom.data_inspector /path/to/dataset --manifest dataset_manifest.json --output artifacts/profile.json --llm-output artifacts/context.json
```

## Разведчик и подтверждение требований
Отдельный агент постановки удалён. После Data Inspector разведчик читает условие и профиль,
вызывает confirm_task с TaskDefinition и исправляет ошибки Python TaskValidator.
Контракт сохраняется до исследования данных. Затем доступны выборки CSV, SQL Polars и корреляции.
Начальная разведка ограничена 300 секундами, последующие — 120; при таймауте используются
сохранённые требования и находки. Без корректных требований обучение не начинается.
Запуск полного процесса: `python solution.py path/to/task.md`.

## Артефакты
`profile.json` содержит таблицы, колонки, распределения, роли и риски; `context.json` и `report.md` ограничены размером и предназначены для LLM. `task.json` хранит тип задачи, метрику, служебные колонки, выходной контракт и результат проверки.

## Ограничения анализа
Анализ распознаёт forecasting, а исполнитель возвращает для такой постановки статус отложенной поддержки. Неоднозначная постановка отмечается в профиле и считается подтверждённой только после проверки `TaskDefinition`.
