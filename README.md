# MLMom: Universal Data Inspector

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
