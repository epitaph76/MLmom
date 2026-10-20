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

## LLM-постановка
`mlmom.task_graph` преобразует профиль в `TaskDefinition` и проверяет его детерминированным валидатором, при ошибке возвращая feedback в LLM. Подключение читается из окружения или локального `.env`, количество попыток ограничено.

```bash
python -m mlmom.task_graph --profile artifacts/profile.json --output artifacts/task.json --max-attempts 3
```

## Артефакты
`profile.json` содержит таблицы, колонки, распределения, роли и риски; `context.json` и `report.md` ограничены размером и предназначены для LLM. `task.json` хранит тип задачи, метрику, служебные колонки, выходной контракт и результат проверки.

## Ограничения анализа
Анализ распознаёт forecasting, а исполнитель возвращает для такой постановки статус отложенной поддержки. Неоднозначная постановка отмечается в профиле и считается подтверждённой только после проверки `TaskDefinition`.
