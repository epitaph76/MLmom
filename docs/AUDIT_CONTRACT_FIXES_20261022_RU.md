# Контракты и достоверность экспериментов — 23.10.2026

Основание: исторические QWEN_AUDIT_RU.md и CRITICAL_AUDIT_RU.md. Их отчёты,
SQLite и модели не изменяются. Новые доказательства сохраняются в
`artifacts/audit-contract-fixes-20261022/`.

Итог реального аудита 23.10: regression прошла 3 цикла, final full-data refit
и изолированный inference. Multiclass засчитала 2 цикла; fit и анализ третьего
завершились, но post-recon остановлен явным budget_exceeded (HTTP 422).
Финального refit multiclass нет; real_api = failed. Платные запросы прекращены.
Runtime ZIP и .env не менялись. Дополнительный аудит выявил остаточные дефекты,
поэтому формулировки ниже описывают реализованные проверки и их намерение,
а не полное доказательство всех требований исходного плана.
[Итог и ограничения](../artifacts/qwen-old-gateway-lifecycle-20261023/ANALYSIS_RU.md),
[машинный список дефектов](../artifacts/qwen-old-gateway-lifecycle-20261023/contract-findings.json).

## Исправления

- Наблюдения сохраняются независимо. `rows` — необязательная измеренная поддержка;
  `result_rows` — размер SQL-выдачи. Измерения берутся из ячеек одного источника,
  служебные числа не принимаются. Ошибка указывает поле/значение/hint. Две одинаковые
  неуспешные записи фиксируют unfinished check. Неверный proposal не стирает остальные;
  локальные IDs разрешаются только однозначно подтверждённым наблюдениям.
- Контракт `export_model_params()` требует полный воспроизводимый JSON, включая
  компоненты, веса, target/output параметры и HPO overrides. Пропуск запрошенного
  поля отклоняется; аудит выявил пробел для незадекларированных target transforms.
  Фабрики получают копии конфигурации. Smoke проверяет export/recreate, фактическую
  конфигурацию и прогнозы одинаковых обычных fit.
- Backend API проверяется после каждого fit. Execution report разделяет запрошенные,
  выполненные и сохранённые итерации, выбранную итерацию и причину остановки.
  Gate стоит до внешней оценки, Champion и submission. Его отказ означает
  implementation=mismatch, effect=inconclusive; LLM не может отменить отказ.
- Регрессионные срезы используют mae/rmse и mean_residual = mean(y_true-y_pred).
  Положительный residual означает недопредсказание. Классификация использует error_rate,
  multilabel — hamming_loss. Диагностика явно unweighted; взвешивание основной метрики
  указано отдельно. score_kind различает CV mean, pooled OOF и outer holdout. Python
  считает relative gain только для одинаковых dataset/split/metric/options/weighting.
- Champion передаётся фактическим snapshot с hash конфигурации, кодом, preprocessing
  и scoped score. original_hypothesis — исторический текст. single_change_ablation
  наследует текущие параметры без HPO.
- duplicate_policy=group_features действует внутри и снаружи: дубли исходных признаков
  без ID/target и явные группы объединяются через связанные компоненты. Пересечение
  проверяется; недостаток независимых групп вызывает ошибку.
- Единый реестр metric_options, проверенные Polars SQL-шаблоны и короткие ошибки
  заменяют скрытые ограничения. Future imports разрешены. CatBoost callback возвращает
  True для продолжения и подключается только при включённых таймаутах.
- Ошибки worker содержат тип, конкретный файл/строку, hint и ссылку на полный
  traceback.log. Ремонт получает существующий код и инструмент apply_patch;
  traceback больше не заполняет основной feedback.
- Глобальный billing/API-budget отказ сохраняет результаты и завершает все роли;
  runner не начинает следующий кейс с тем же отказом.

## Интерфейсы

fitted_components() раскрывает каждый base/meta компонент; одиночный кандидат может
использовать model. FitContext содержит timeouts_enabled; Torch использует
track_optimizer(optimizer, batch_size) для фактических steps/epochs. В плане добавлены
stopping_policy, component_stopping_policies, parameter_routes, base_config_hash,
change_kind, validation_protocol. CandidateVersion хранит smoke report, ExperimentRun —
отчёт refit. Старые артефакты без provenance не получают статус проверенных.

## Историческое воспроизведение

scripts/replay_audit_contracts.py открывает SQLite read-only и сверяет SHA256 до/после.
Новые observation содержат численные ячейки сохранённых результатов, без новой
научной интерпретации и без LLM-запросов:

| Запуск | CSV-результаты | Observation при replay | Историческая начальная разведка |
|---|---:|---:|---|
| Qwen | 25 | 22 | 0 observation, 0 proposal |
| DeepSeek | 54 | 51 | 2 observation, 0 proposal |

Остальные результаты — схемы либо не имеют численных ячеек. Исторический blend загружен
без обучения: CatBoost имеет одно дерево. Новый gate отклоняет его за неполный экспорт
и необъяснённое завершение вместо 800 итераций. Доказательства: accepted/historical-replay/report.json и
accepted/historical-replay/historical-blend-execution.json. Сквозной тест проверяет передачу observation
через context_agent до hypothesis_generator.

## Проверка

Инспектор учитывает sparse metric history: сохранённые backend rounds дают
независимую нижнюю границу выполненных итераций. Проверены 8 деревьев CatBoost
при 4 записях истории (`metric_period=3`). Фиксированный план требует точного
числа итераций: превышение лимита также отклоняется. Для Torch число эпох
вычисляется без округления из фактических optimizer steps; лишняя часть эпохи
не скрывается. Регрессии проверяют обе политики остановки.

Замороженный ZIP: `submission/mlmom-solution.zip`, SHA256
`cf471a24ce2ca6a6adb60992faa56bbcda9030029ed9a6f2df145e395bc1fe64`.
Доказательства этой сборки: `artifacts/audit-contract-fixes-20261022/accepted/`.
Промежуточные `source-initial/`, `final/`, `verified/` и `release/` не подтверждают
этот hash. Предыдущие ZIP сохранены; предыдущие машинные отчёты находятся в
`verified/previous-*`, `release/previous-canonical-validation.json` и
`accepted/previous-canonical-validation.json`.

| Проверка этой сборки | Результат |
|---|---|
| Полный набор исходников, Linux / Python 3.12.15 | 228 тестов, 0 ошибок/падений/пропусков, 627.54 с |
| Полный набор runtime из ZIP, Linux / Python 3.12.15 | 228 тестов, 0 ошибок/падений/пропусков, 673.19 с |
| Предоставленные зависимости | Все версии совпадают, Torch 2.5.1+cpu |
| Offline regression / RMSE | 3 цикла, 3 внешних split, 2 incumbent recheck, 22 449 строк, 561.45 с |
| Offline multiclass / Macro F1 | 3 цикла, 3 внешних split, 2 incumbent recheck, 41 166 строк, 223.52 с |
| Финальный refit и перенос модели | Обе задачи прошли; inference в новом процессе совпал на 128 строках |
| Изолированный inference | Обе offline-модели: 128 строк совпали; исходные каталоги недоступны, сеть отключена |
| Исторический replay | 22 / 51 сохранённое наблюдение; SHA256 SQLite не изменился; старый blend отклонён |
| Реальная Qwen | Regression passed 3/3; multiclass 2/3 и global_api_failure по бюджету; final refit multiclass отсутствует; real_api = failed |

Офлайн-стенд заменяет только LLM внешней тестовой fixture. Данные, обучение, HPO,
CV, внешняя оценка, refit и inference настоящие. Эта проверка не подтверждает
качество работы реального Qwen. Registry/skills, precision/rows/IDs/resume,
полный export/recreate и ансамбли с плоским HPO, фактические backend-параметры,
CatBoost callback/ранняя остановка, остатки и области метрик, Champion/ablation,
утечка дублей, future imports и глобальный billing stop покрыты полным набором.

Промежуточный Qwen regression на hash 805d… остановлен перед обучением после
обнаружения sparse-history дефекта. Его 10 наблюдений и 4 предложения сохранены;
все 10 IDs дошли до context_agent и hypothesis_generator.
Следующий промежуточный прогон ba7d… остановлен для исправления допуска превышения
итераций: сохранены 21 наблюдение и 6 предложений; все 21 ID также найдены в
запросах обеих следующих ролей. Доказательства: соответствующие
`verified/qwen-real/` и `release/qwen-real/` — `interrupted.json` и
`regression/attempt-1/initial-observation-handoff.json`.
Эти прогоны не подтверждают текущую сборку.

После успешной офлайн-приёмки текущего ZIP запущена последовательная приёмка
Qwen regression и multiclass: целевые три завершённых цикла на задачу, без общего
лимита API-запросов, timeouts_enabled=false. Подключение и .env не изменены.
На генерации модели regression шлюз трижды вернул HTTP 502 `upstream error`.
По явному запросу пользователя контроллер и контейнер остановлены; multiclass
не запускался. Наблюдения, состояние, исходники и логи сохранены, временный
api.env удалён. Реальных завершённых циклов и fit нет. Манифест паузы:
`accepted/qwen-real/paused.json`. Полный реальный refit и inference не подтверждены.

На текущем hash Qwen сохранила 9 наблюдений и 3 предложения. Все 9 канонических
IDs найдены в фактических запросах context_agent и hypothesis_generator.
Доказательство: `accepted/qwen-real/regression/attempt-1/initial-observation-handoff.json`.
Это подтверждает передачу наблюдений; полный цикл обучения проверяется отдельно.

Диагностика старого шлюза `https://changellenge.up.railway.app/v1` с тем же ключом
и Qwen: короткий запрос прошёл за 1.33 с; точный сохранённый запрос coding_agent,
который трижды получал 502 на новом шлюзе, прошёл за 17.45 с. Получены корректный
JSON CodeBundle и синтаксически корректный candidate.py. Код не исполнялся и не
применялся к приостановленному эксперименту. .env и ZIP не изменены; эксперимент
остаётся на паузе. Доказательство: `old-gateway-probe-20261023/coding-replay-20261022T232012Z/report.json`.

По новому запросу пользователя запущены с нуля обе задачи из присланных ZIP,
последовательно через старый шлюз, на том же runtime hash. Полный жизненный цикл,
включая ошибки и версии кода, сохранён в
[новом отчёте](../artifacts/qwen-old-gateway-lifecycle-20261023/LIFECYCLE_RU.md).
Ранее приостановленный прогон и его артефакты сохранены отдельно.
Новый запуск завершился 23.10 в 14:28 МСК явным отказом по бюджету:
gateway Current cost=602.0488979117156, Max budget=600.0. После terminal failure
новых HTTP attempts нет. На обоих кейсах сохранены все успешные save IDs;
точные метрики независимо пересчитаны, фактические duplicate groups не пересекают
outer/inner splits. Остаточные проблемы: custom class_weight policy конфликтует
с native параметром, mutable post-fit атрибут ошибочно служит provenance,
полнота export не охватывает скрытые константы, protocol dict слишком строго
сравнивается, HPO-вмешательство и внутренние CV metrics неполно передаются reviewer.
Они перечислены с доказательствами в новом отчёте; runtime во время аудита
не ремонтировался. Итог [acceptance-summary.json](../artifacts/qwen-old-gateway-lifecycle-20261023/acceptance-summary.json).

[Исходники](../artifacts/audit-contract-fixes-20261022/accepted/source-linux/tests.json),
[ZIP](../artifacts/audit-contract-fixes-20261022/accepted/zip-linux/tests.json),
[offline](../artifacts/audit-contract-fixes-20261022/accepted/offline/reports.json),
[replay](../artifacts/audit-contract-fixes-20261022/accepted/historical-replay/report.json),
[машинная приёмка](../submission/validation.json).
Платформенный score и скрытые test labels этими проверками не устанавливаются.
