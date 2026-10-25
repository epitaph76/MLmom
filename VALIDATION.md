# Diversity Evidence ансамблей — 26.10.2026

ensemble_agent 1.1.0: compare_oof_components, корреляция/пересечение ошибок на совместимых
исторических OOF, primary-metric gain фиксированного среднего и leave-one-component-out
для тройки, advisory refit cost и явные статусы неизвестных данных. Отчёт сохранён и
доставляется Planner и plan validator. Новые worker artifacts имеют version=2; старые
OOF и runs не переписываются. Схема runtime v8 сохраняется с необязательным fit duration.

Проверки: 76 регрессионных тестов, 40 проверок остальных задач/контрактов/исследовательских
карточек и финальные 9 тестов Diversity Evidence прошли. После уточнения связанных
карточек повторно прошли 19 тестов карточек/доставки/упаковки. Это 119 разных тестов.
Полный suite повторно не запускался. Реальные worker
fit/save/load, original-scale weights/targets, class order, разные probability shapes,
metric options/direction, multioutput, несовместимые/legacy OOF, восстановление и отсутствие
новых cost gates проверены. Новых платных LLM-запросов нет.

Промежуточный final-card-check выявил превышение старого combined context на 197 символов.
Повторяющаяся формулировка validation сокращена; лимиты не увеличивались, итоговый повтор
успешен. Первоначальный лог ошибки сохранён вместе с card-check-accepted.log.

В каталоге по-прежнему 33 карточки и 14 справочников; ensemble instructions — 6 122 символа
из 24 000. Проверки документации, формата карточки, упаковки и diff пройдены; граф обновлён.
Отчёт: [ENSEMBLE_DIVERSITY_RU.md](docs/ENSEMBLE_DIVERSITY_RU.md). Результаты и SHA256:
`artifacts/ensemble-diversity-2026-10-26/summary.json`.

Согласованные архивы включают также предыдущие дополнения исследовательской методики:

- `dist/mlmom-all-skills-v8-ensemble-diversity-2026-10-26.zip`.
- `dist/mlmom-runtime-v8-ensemble-diversity-2026-10-26.zip`.

Предыдущие архивы и результаты сохранены.

# Уточнение исследовательской методики — 26.10.2026

Feature Agent 1.2.0, reconnaissance_agent 2.4.0, feature_engineering/feature_selection 1.1.0:
иерархия исследования сущностей, обязательная EDA-матрица и диагностика вклада признаков.
Проверены 93 затронутых теста; после окончательных уточнений отчётности успешно повторены
40 тестов и затем 8 тестов формата инструмента. Полный suite повторно не запускался.
SQL, независимый pair oracle, repeated/joint permutation, восстановление EDA-прогресса,
сохранение подтверждённой задачи и доставка references/хешей проверены непосредственно.

В каталоге 33 карточки и 14 REFERENCE.md. Лимиты соблюдены: Feature Agent с reference —
23 259 символов из 24 000; все preprocessing cards с references — 43 836 из 48 000.
Новых платных LLM-вызовов нет. Граф обновлён локально. API исполнителя v8 сохраняется;
неполное исследование имеет partial, а подтверждённая задача может продолжать обучение.

Отчёт: [RESEARCH_METHODOLOGY_RU.md](docs/RESEARCH_METHODOLOGY_RU.md).
Доказательства: `artifacts/entity-discovery-procedure-2026-10-25/research-methodology-summary.json`.
Согласованные архивы:

- `dist/mlmom-all-skills-v8-research-methodology-2026-10-26.zip` — 33 карточки и 14 справочников.
- `dist/mlmom-runtime-v8-research-methodology-2026-10-26.zip` — runtime с новыми инструментами/карточками.

Старые отчёты, runs и архивы сохранены; результаты первоначальной приёмки v8 приведены ниже.

# Приёмка скиллов и runtime v8 — 25.10.2026

Ветка: `codex/skill-contracts-v8`; исходный commit:
`7ad10668c87de56f0dfb0684d2de3b90a959c6be`.
Подробный отчёт R01–R11, область рецептов и ограничения:
[SKILL_CONTRACTS_V8_RU.md](docs/SKILL_CONTRACTS_V8_RU.md).

| Проверка | Итог |
|---|---|
| Регрессионное покрытие | 399 tests: 398 passed, 1 штатный Linux-only skip |
| Метод | Полный discovery + 59 успешных затронутых тестов после исправлений |
| Первичный discovery | 0 errors, 1 failure: старое ожидание отсутствия preprocessor tools; исправлено |
| Исходные fixture errors | Все 6 устранены; assertions rotating holdout сохранены |
| Карточки / references | Проверены 33 / 11; minor versions повышены у 19 пакетов |
| Реальные библиотеки | 41 основной сценарий + 3 export-count edge cases; все подтверждены |
| Docker | Существующий mlmom-ml:test, актуальные исходники, network=none, CPU-only |
| Живые роли | Feature, preprocessing, Training, plan validator; 13 HTTP attempts из 20 |
| API usage | 88 589 prompt + 5 365 completion = 93 954 total tokens |
| Документация / ZIP / секреты | Проверено; плоский ZIP содержит ровно 3 корневых файла |
| Graphify | Обновлён локально в code-only режиме |

Машиночитаемая сводка: `artifacts/skill-contracts-v8/acceptance.json`.
Первоначальные ошибки, отдельные повторы и полный журнал discovery сохранены рядом.
Результат не выдаёт первый discovery за полностью успешный: итог учитывает последующий
успешный повтор его единственного failure и остальных затронутых модулей.

Финальный архив: `dist/mlmom-skill-contracts-v8-accepted.zip`, 231 186 байта.
SHA-256: `8a501badb80ff66137724285448419bc4dea824f3bb62094a0e66e8172196575`.
Промежуточный архив, исходный аудит и старые runs сохранены. Новые исполнения требуют
нового каталога для runtime contract 8. Методологическое одобрение остаётся агентским;
estimator-аудиты, provenance gates и обязательный smoke не добавлялись.

# Историческая проверка ветки с минимальными проверками

Ветка: `codex/agent-first-minimal-checks`. Исходное состояние, включая прежние локальные
изменения: `78f5409`. Правила ветки: [MINIMAL_CHECKS_RU.md](docs/MINIMAL_CHECKS_RU.md).

## Результаты

| Проверка | Результат |
|---|---|
| Полный unittest discover | 331 тестов; ошибок 0, падений 0, пропущено 1 |
| Среда | Windows, Python 3.12.10 |
| Исходники и плоский ZIP | Совпадают |
| Сканирование ZIP на настроенные секреты | passed |
| Проверка каталога скилов | passed |
| git diff --check | passed |
| Граф Graphify | Обновлён локально, без semantic backend |

Стандартный повтор: `python -m unittest discover -v`.
В выполненном прогоне все 47 найденных модулей запущены изолированно
в 4 процессах. Число выполненных тестов сверено с unittest discovery:
331 == 331. Исполняемые модули и карточки агентов
не менялись во время проверки.
6 модулей повторены последовательно после первого запуска:
короткие Windows-таймауты оказались чувствительны к параллельной нагрузке; также уточнены
fixtures агентского отказа и финального refit. Функциональные многоцикловые тесты используют
стандартный режим без таймаутов; отдельные проверки включённых таймаутов сохранены.
Первичные результаты находятся в `tests.initial.json`, прежние попытки — в `previous_attempt`;
итог включает последние фактические результаты всех модулей.
Сводный JSON: `artifacts/minimal-checks/full-discovery/tests.json`; журналы каждого модуля —
в подкаталогах `artifacts/minimal-checks/full-discovery/` (локальные, игнорируются Git).

Проверены реальные локальные fit/predict/save/load/refit с подставным LLM, восстановление,
циклы CSV, поиск признаков и ансамбли без execution provenance. Отдельный сценарий использует
кандидат без export_model_params/fitted_components: ошибка fit → агентский patch по traceback →
успешное обучение и CSV → resume. Обязательного smoke в нём нет.
Проверки ресурсов/процессов, границ данных и формата прогнозов/CSV сохранены.

## Архив

`dist/mlmom-agent-first-minimal-checks.zip` — 199292 байт.
SHA-256: `b04839377d93aee6044050b20020bde7ca209314bf062ebd48ad6ca074a5b804`.
Содержит только `solution.py`, `requirements.txt`, `mlmom_bundle.py`.
Старые архивы и отчёты в `submission/` относятся к строгой реализации.

Платные LLM/API и инфраструктура организаторов не запускались. Проверки подтверждают
работоспособность локального цикла, а не качество решений реальной модели.
