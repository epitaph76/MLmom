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
   и `--seconds 1800`. Она читает существующее подключение Qwen, не меняет модель
   или provider. Лимит каждой задачи включает все попытки, простой и проверку.
7. `python scripts/report_release.py --source-tests artifacts/release-abc/source/tests.json --tests artifacts/release-abc/packaged/tests.json --linux-tests artifacts/release-abc/linux-packaged/tests.json --offline-reports artifacts/release-abc/offline/reports.json --real-reports artifacts/release-abc/real/reports.json`

release_cases копирует только объявленные входы и три корневых файла ZIP в
отдельный каталог. В контейнере запускается точная команда
`python3 solution.py task.md`. Для ограничения приёмочного бюджета используются
`MLMOM_CONFIG` (тот же JSON, что CLI --config) и `MLMOM_RUN_DIR`; CLI имеет приоритет.
Без переменных стандартный запуск и бюджет не меняются.

Offline использует внешний sitecustomize и fixtures/tests.fakes для подмены
только LLM-клиента. Они не входят в ZIP; модель, предобработка, поиск и worker
реальны. Runtime загружается из ZIP до подключения тестовых fixtures. Real
запускается без этой подмены; ключи передаются через временный env-file и затем
удаляются. В отчёты ключи не записываются.

Обе задачи должны завершить три цикла с разными внешними split и повторным
обучением incumbent. Проверяются RMSE/site_eui и Macro F1/attack_cat, точные
колонки, число строк, порядок id и допустимые значения. Отдельный процесс копирует
исходники кандидата и модель, выполняет inference в новом worker и сравнивает
предсказания с submission.

report_release не объявляет успех по одному комплекту тестов. Неисполненные
проверки — not_verified; неверный хеш, незавершённый прогон или провал проверки —
failed. Проверка фактической платформы имеет отдельный статус not_verified.
После изменения ZIP соответствующие испытания выполняются заново.
