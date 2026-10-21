# Архив для загрузки

`mlmom-solution.zip` содержит три корневых файла: `solution.py`, `requirements.txt`, `mlmom_bundle.py`. Платформа запускает решение отдельно для каждой задачи; подключение к API передаётся через окружение.

В текущем ZIP снят клиентский лимит выходных токенов; на извлечённом коде прошли **16 проверок** LLM-клиентов, трассировки и упаковки с mock transport. До этой правки полная проверка включала 81 тест в Windows/Python 3.12.10 и обе организаторские задачи с mock LLM и реальным CatBoost; результаты содержат 22 449 и 41 166 строк. Это функциональная проверка, а не подтверждение платформенного score или качества реальной LLM. Реальный API после снятия лимита не запускался.

[validation.json](validation.json) содержит SHA256, хеш встроенных исходников и сведения о проверках. [Полный отчёт](../VALIDATION.md) описывает результаты и ограничения; [трассировка](../docs/TRACING.md) — расположение API-запросов, ответов, скиллов и решений Scheduler.

```bash
python scripts/package_solution.py --output submission/mlmom-solution.zip
python scripts/validate_packaged.py submission/mlmom-solution.zip --output artifacts/reliable-start-release
```
