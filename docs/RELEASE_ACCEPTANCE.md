# Приёмка A → B → C

Каждый отчёт относится к SHA256 конкретного submission/mlmom-solution.zip.
Стенд Linux/Docker использует Python 3.12 и фиксированные зависимости; CPU-only
torch с суффиксом +cpu допустим. Версии окружения сохраняются в отчётах тестов.
Это приближение инфраструктуры организаторов, не проверка их платформы.

Порядок:

1. `python scripts/run_tests.py --output artifacts/release-abc/source`
2. `python scripts/package_solution.py --output submission/mlmom-solution.zip`
3. `python scripts/validate_packaged.py submission/mlmom-solution.zip --output artifacts/release-abc/packaged`
4. Повторить validate_packaged в Linux-образе с предоставленными зависимостями.
5. `python scripts/release_cases.py --archive submission/mlmom-solution.zip --output artifacts/release-abc/offline --mode offline --image mlmom-ml:test --seconds 900`
6. Только после автоматической приёмки: та же команда с `--mode real`, новым output
   и `--enable-timeouts --seconds 1800`. Она читает существующее подключение Qwen, не меняет модель
   или provider. Лимит каждой задачи включает все попытки, простой и проверку.
7. `python scripts/check_environment.py --output artifacts/release-abc/environment.json`
   выполнить в том же Linux-образе; проверяются все предоставленные версии.
8. `python scripts/report_release.py --source-tests artifacts/release-abc/source/tests.json --tests artifacts/release-abc/packaged/tests.json --linux-tests artifacts/release-abc/linux-packaged/tests.json --offline-reports artifacts/release-abc/offline/reports.json --real-reports artifacts/release-abc/real/reports.json --environment artifacts/release-abc/environment.json`

release_cases копирует только объявленные входы и три корневых файла ZIP в
отдельный каталог. В контейнере запускается точная команда
`python3 solution.py task.md`. Для ограничения приёмочного бюджета используются
`MLMOM_CONFIG` (тот же JSON, что CLI --config) и `MLMOM_RUN_DIR`; CLI имеет приоритет.
Без переменных используется стандартная конфигурация с выключенными таймаутами.

Offline использует внешний sitecustomize и fixtures/tests.fakes для подмены
только LLM-клиента. Они не входят в ZIP; модель, предобработка, поиск и worker
реальны. Runtime загружается из ZIP до подключения тестовых fixtures. Real
запускается без этой подмены; ключи передаются через временный env-file и затем
удаляются. В отчёты ключи не записываются.
Для явно согласованного выбора Qwen при другом локальном default используйте
`--model qwen/qwen3.8-27b`; подключение и ключ можно передать через OPENAI_*
только процессу приёмки, не редактируя .env. Параметр --supplemental у report_release
добавляет отдельные проверки того же SHA256, выполненные после основного набора.

Для отдельно согласованной диагностики другого API используйте `--mode real
--diagnostic-api --model <явно выбранная модель>` и отдельный output. Подключение
передаётся только процессу через OPENAI_*. Такие отчёты содержат
`diagnostic_api=true` и не допускаются к Qwen-приёмке даже при успешных трёх
циклах. Модель не подменяется автоматически; проверки артефактов те же.
Для явно согласованного расширенного диагностического прогона доступны
`--seconds` до 10800 на задачу и `--fit-time-multiplier 3`: последнее увеличивает
окна первого эксперимента, последующих экспериментов и поиска втрое, без
изменения общего дедлайна, количества trials и резерва на финальное обучение,
если временные лимиты включены. В ограниченной Qwen-приёмке используется
явный --enable-timeouts --seconds 1800.

Обе задачи должны завершить три цикла с разными внешними split и повторным
обучением incumbent. Проверяются RMSE/site_eui и Macro F1/attack_cat, точные
колонки, число строк, порядок id и допустимые значения. Отдельный процесс копирует
исходники кандидата и модель, выполняет inference в новом worker и сравнивает
предсказания с submission.

report_release не объявляет успех по одному комплекту тестов. Неисполненные
проверки — not_verified; неверный хеш, незавершённый прогон или провал проверки —
failed. Проверка фактической платформы имеет отдельный статус not_verified.
После изменения ZIP соответствующие испытания выполняются заново.

## Таймауты и диагностические прогоны

В обычном runtime timeouts_enabled=false по умолчанию: HTTP, роли, CSV, coding,
worker, folds, HPO и ожидание training-agent не завершаются по времени. Ограничения
памяти/CPU, число раундов/attempts/trials, plateau, схемы и refit остаются. Для
явного возврата временных лимитов используйте timeouts_enabled=true и нужные
*_seconds; скрытых верхних пределов 30/300 секунд больше нет. Resume требует
сохранения выбранной политики, а ограниченные исследования сохраняют исходные
дедлайны.

release_cases с --diagnostic-api по умолчанию также не имеет внешнего дедлайна.
Для ограниченной диагностики добавьте --enable-timeouts и --seconds. Для ограниченных offline-регрессий и Qwen-приёмки таймауты теперь также
включаются только явным --enable-timeouts. Без этого флага внешнего дедлайна
стенда нет. Прогоны других моделей остаются диагностическими.
