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

## Полный исследовательский пайплайн

Реализованы этапы 2-4 и финализация: отдельные роли генератора гипотез,
Planner, LLM Validator и Coding Agent; Python-проверки планов и кода;
процессные workers; Optuna/native AutoML; новый split каждого раунда;
переобучение incumbent; журнал, checkpoints и атомарный submission.
Подробности и ограничения описаны в [ARCHITECTURE.md](ARCHITECTURE.md).

На платформе с предустановленным `requirements.txt`:

```bash
python solution.py /path/to/task.md
```

Платформа подставляет `OPENAI_API_KEY` и `OPENAI_BASE_URL` через окружение.
`OPENAI_MODEL` необязателен: по умолчанию `qwen/qwen3.8-27b`.
Платформенный профиль использует только переменные окружения, без `.env` и
RouterAI fallback. Ключ не включается в ZIP. MCP не требуется.

Для локальной проверки с ключом организаторов добавьте в игнорируемый `.env`:

```dotenv
OPENAI_API_KEY=your-team-key
OPENAI_BASE_URL=https://changellenge.up.railway.app/v1
OPENAI_MODEL=qwen/qwen3.8-27b
```

Общий лимит команды — 600 рублей в день, сброс в 03:00 МСК.
Программного денежного лимита нет; короткие проверки тоже расходуют этот бюджет.

Локальная установка минимального runtime и запуск:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
# Дополнительные ML-библиотеки устанавливаются заранее из requirements.txt.
.\.venv\Scripts\python.exe solution.py path\to\task.md --config profiles\dev.json
```

Dev-профиль принимает также старые `ROUTERAI_*` переменные из локального
`.env`. Стандартные `OPENAI_*` имеют приоритет. Настройки модели общие для
всех ролей; переход на выданную Qwen не требует менять код.

Параметры времени, retries, HPO, ресурсов, seed, metric options и формата
предсказаний задаются JSON-конфигурацией (`RunConfig`). По умолчанию:

- 180 минут на весь запуск, 40 минут резерва на финальное обучение;
- цель первого файла - 10 минут, до трёх параллельных экспериментов;
- до 20 Optuna trials, patience 5, до трёх попыток плана/кода;
- до двух дополнительных исправлений runtime-ошибок;
- остановка после пяти раундов без победы над incumbent на текущем split.

Во время исследования test предсказывается split-моделью. Финальный refit
использует весь train и сохраняется отдельно. При ошибке новый файл не
заменяет последний проверенный submission. Если LLM недоступна до получения
рабочего кода, встроенный baseline не запускается.

### Скиллы и инструменты

`skills/` содержит 15 версионированных пакетов библиотек/техник. Все
`SKILL.md` намеренно оставлены заглушками. Реестр, загрузка manifest,
совместимость, ограничение контекста и хеширование реализованы. Кодер
может писать рабочий код с незаполненными инструкциями.

Типизированные tools предоставляют кодеру чтение, поиск, запись, patch и
ограниченные проверки. Генерация выполняется по единому SDK из
`mlmom/sdk.py`; готовые CPU-адаптеры находятся в `mlmom/adapters.py`.
Доступность библиотеки подтверждается probe, а не только её названием.

### Артефакты и восстановление

Каждый запуск создаёт `artifacts/<run>/` с session, profile/task, исходными
планами, неизменяемыми версиями кандидатов, split-индексами, результатами,
моделями и логами. Метаданные/события находятся в `experiments.sqlite3`,
состояние графа - в `checkpoints.sqlite3`, итог - в `summary.json`.

```bash
python solution.py /path/to/task.md --config profiles/platform.json --run-dir artifacts/my-run
python solution.py /path/to/task.md --config profiles/platform.json --run-dir artifacts/my-run --resume
```

Восстановление требует тех же входных файлов и конфигурации. Общий срок
не продлевается. Завершённые эксперименты не обучаются повторно.

### Проверка и упаковка

```bash
python -m unittest discover -v
python scripts/package_solution.py --output dist/mlmom-solution.zip
```

Тесты используют mock LLM и не расходуют API. Отдельный тест полных архивов:

```bash
python scripts/smoke_organizer.py regression.zip multiclass.zip --output artifacts/organizer-smoke
```

Проверка на Linux без сети (Dockerfile устанавливает зависимости только при сборке):

```bash
docker build -t mlmom-core:test .
docker run --rm --network none --cpus 8 --memory 32g --entrypoint python mlmom-core:test -m unittest discover -v
```

`Dockerfile.full` дополнительно собирает проверочное окружение всех библиотек
организаторов с CPU-версией torch. На самой платформе установка не выполняется.

Готовый `dist/mlmom-solution.zip` загружается в Boosters. В корне ровно три файла:
`solution.py`, неизменённый `requirements.txt` организаторов и `mlmom_bundle.py`.
Последний содержит исходный код собственного пакета, manifests, заглушки
инструкций и профили в виде Python-строк. При старте он восстанавливает их в
`.mlmom-runtime-<hash>/` рядом с точкой входа. Это необходимо, потому что
Boosters переносит только корневые `.py` и `requirements.txt`, игнорируя папки.
Сторонних пакетов, установки зависимостей или загрузок при запуске нет.
Датасеты, `.env`, тестовые mock-агенты и обученные модели в архив не входят.

### Границы v1

Процессная изоляция защищает от ошибок и превышения ресурсов, но не является
защитой от намеренно враждебного Python-кода. Отдельного финального audit
holdout нет: метрики раундов являются исследовательскими. Временные задачи
распознаются, но откладываются. Clustering-метрики требуют конечного числового
пространства признаков; anomaly-оценка требует явных evaluation labels.
Неоднозначные метрики и выходные контракты отклоняются.


