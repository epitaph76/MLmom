# MLMom: Data Inspector and Task Analysis

Data Inspector — первый узел AutoML-агента. Он принимает ровно три вида входа:
обучающую таблицу, словарь колонок и текст задачи. Затем он создаёт компактный
`TaskSpec` и профиль train для LLM. Исходные файлы не изменяются.

`test.csv` и шаблон submission относятся к ML Execution Runtime. Data Inspector
их не читает, не анализирует и не использует для определения таргета.


## Поддерживаемые сценарии

- regression и multi-output regression;
- binary, multiclass и multilabel classification;
- time-series forecasting при наличии временной колонки;
- ranking;
- anomaly detection и clustering без таргета;
- составные ID, несколько таргетов, group/time/weight columns;
- несколько словарей колонок;
- произвольные имена файлов через `dataset_manifest.json` или CLI overrides.

Если задача неоднозначна, инспектор не маскирует догадку под факт. Он ставит
низкую confidence и создаёт предупреждение `ambiguous_task_definition`.

## Результаты

За один запуск создаются три артефакта:

1. `profile.json` — полный контракт `DatasetProfile` для runtime, evaluator и
   experiment store.
2. `llm_context.json` — ограниченный по размеру JSON без сырых строк.
3. `llm_report.md` — читаемый LLM-отчёт с таблицами диапазонов, долей,
   пропусков, кардинальностей и рисков.

Пример таблицы распределения в отчёте:

| Range or class | 1–10 | 10–20 | 20–30 |
|---|---:|---:|---:|
| Share | 3% | 30% | 15% |
| Rows | 30 | 300 | 150 |

Для числовых столбцов диапазоны формируются автоматически. Крайние 1% и 99%
отделяются от основного диапазона, чтобы единичные выбросы не делали интервалы
бесполезными. Для категорий показываются классы или top values с долями.

## Что анализируется

- состав train, словаря и текстового условия;
- таргеты, ID, time/group/weight columns, тип задачи и метрика;
- формат результата: имя файла, колонки, кодировка, разделитель, индекс и порядок строк;
- требования к пропускам, бесконечностям, уникальности ID и числу прогнозов;
- тип, описание, единицы, пропуски и кардинальность каждой колонки;
- числовые квантили и диапазоны с количеством и долей строк;
- распределения категорий, boolean, datetime и длины текста;
- повторяющиеся признаки и конфликтующие таргеты;
- потенциальная утечка таргета;
- дисбаланс классов и перекос регрессионного таргета;
- рекомендации по препроцессингу и валидации.

Data Inspector ничего не удаляет, не импутирует и не кодирует. Трансформации
должны обучаться внутри train-fold на следующем этапе.

## Автоопределение и manifest

При стандартных названиях достаточно каталога: инспектор найдёт `train`,
словарь и текст задачи. Файлы с именами test/scoring/submission намеренно
игнорируются. Для произвольной структуры
используйте [dataset_manifest.example.json](dataset_manifest.example.json).

Важные поля manifest:

- `train`;
- `dictionaries`;
- `task_description`;
- `target_columns`, `id_columns`;
- `task_type`, `primary_metric`;
- `time_column`, `group_columns`, `weight_column`.

Явные значения manifest имеют приоритет над эвристиками. CLI overrides имеют
приоритет над manifest.

## Запуск

Из корня репозитория в окружении Python 3.12:

```powershell
python -m mlmom.data_inspector `
  "C:\path\to\dataset" `
  --output artifacts\dataset.profile.json `
  --llm-output artifacts\dataset.llm.json `
  --report-output artifacts\dataset.llm.md
```

С manifest:

```powershell
python -m mlmom.data_inspector `
  "C:\path\to\dataset" `
  --manifest dataset_manifest.json `
  --output artifacts\dataset.profile.json `
  --llm-output artifacts\dataset.llm.json `
  --report-output artifacts\dataset.llm.md
```

Пример явных overrides:

```powershell
python -m mlmom.data_inspector `
  "C:\path\to\dataset" `
  --train learning.csv `
  --target label `
  --id record_id `
  --task-type multiclass_classification `
  --metric macro_f1 `
  --output artifacts\dataset.profile.json `
  --llm-output artifacts\dataset.llm.json `
  --report-output artifacts\dataset.llm.md
```

## Тесты

```powershell
python -m unittest discover -v
```

Тесты проверяют произвольные имена файлов, manifest, полное игнорирование
test/submission, автоопределение regression/classification, извлечение формата
результата, datetime, text, категории, числовые диапазоны и лимит контекста.

## Первичный LLM-модуль и валидация задачи

После Data Inspector запускается LangGraph-процесс:

```text
profile.json -> LLM analysis -> deterministic validation
                      ^                  |
                      |---- feedback ----|
```

LLM получает полный текст задания и компактный результат Data Inspector, но не
читает сырые строки train. Она возвращает строгий `TaskDefinition`: тип задачи,
таргеты, метрику и направление оптимизации, схему валидации, служебные колонки,
формат результата, требования, допущения и риски.

Python-валидатор независимо проверяет, что:

- все упомянутые колонки существуют в train;
- target и ID совпадают с профилем и словарём данных;
- target и ID исключены из признаков;
- метрика применима к типу предсказания и имеет верное направление;
- временная и групповая структура не нарушается при валидации;
- формат результата соответствует требованиям из `task.md`.

При ошибке валидатор возвращает структурированный feedback в LLM. После трёх
неудачных попыток процесс завершается со статусом `failed` и кодом выхода 2,
поэтому некорректный контракт не попадёт в ML Execution Runtime.

### Настройка RouterAI

Скопируйте `.env.example` в локальный `.env` и добавьте ключ. `.env` уже
исключён из Git.

```powershell
Copy-Item .env.example .env
```

```dotenv
ROUTERAI_API_KEY=your-real-key
ROUTERAI_BASE_URL=https://routerai.ru/api/v1
ROUTERAI_MODEL=deepseek/deepseek-v4.1-flash-1017
```

Не записывайте настоящий ключ в Python-файл, README, commit или сообщение чата.

После создания `profile.json` запустите:

```powershell
python -m mlmom.task_graph `
  --profile artifacts\dataset.profile.json `
  --output artifacts\dataset.task.json `
  --max-attempts 3
```


