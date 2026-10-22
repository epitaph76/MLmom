# Исправления по ревью 22.10.2026

Исправлены восемь найденных стыков runtime:

- Multilabel log-loss учитывает положительные и отрицательные метки, усредняется по меткам и поддерживает веса строк.
- Worker передаёт прогнозы в Parquet. Строковые классы сохраняют тип; проверка допустимых классов применяется и к массивам `(n, 1)`. Исторические CSV по-прежнему читаются.
- Для stacking введён `fit_stack(X, y, context, create_preprocessor)`: модель получает исходные признаки и создаёт свежие `PreparedCandidate` для базовых OOF-фолдов и refit. Внешний wrapper не обучает общий предобработчик перед stacking. Неадаптированный старый stack-код отклоняется на smoke.
- Native AutoML-компоненты отклоняются в planner и worker до проверки их frozen-refit протокола. Slow-компоненты отклоняются в начальном lightweight-плане. Обычные ансамбли сохраняются.
- Неподдерживаемая multilabel stratified CV отклоняется при подтверждении постановки. Поддерживаемые альтернативы — holdout и group_kfold.
- Session без timeout-флага мигрирует в прежний timed-режим. При повторном resume политика берётся из session, если пользователь её не указал; явное изменение отклоняется.
- Ограниченные diagnostic и replay entrypoints явно включают таймауты.
- Модуль предобработки загружается в том же изолированном контексте sibling-imports, что и модель. Локальные helper-модули доступны и регистрируются для переносимого сохранения.

Дополнительно исправлен конструктор RouterAIChatClient с timeout_seconds=None. Обновлены инструкции кодера, версия ensemble skill и локальный Graphify.

Новые regression tests: `tests/test_review_regressions.py`. Проверены BCE/weights, типы меток, ограничения компонентов, stacking OOF isolation, перенос и загрузка generated stack, локальные импорты и миграция session.

Сборка: `submission/mlmom-solution.zip`. Текущий машинный статус приёмки — `submission/validation.json`. Старые отчёты и архивы с другими именами относятся к прежним hash.

На этапе проверки этих исправлений платные LLM-запуски и проверка платформы не выполнялись. Поддержка нового stack-контракта проверена generated-кодом в smoke, а не ответами реальной модели. Последующий архитектурный разбор реальных DeepSeek и Qwen прогонов приведён в [ARCHITECTURE_AUDIT_20261022_RU.md](ARCHITECTURE_AUDIT_20261022_RU.md); обнаруженные там дополнительные дефекты не следует считать исправленными этим изменением.

Проверка завершена: 199 source tests без failures/errors/skips, 514.19 секунды, Windows/Python 3.12.10. Дополнительно все 8 новых regression tests прошли на извлечённом runtime ZIP. Hash сборки: `d00c03291f75bfe0c75c002480c60da5379a169fbab0da91bed895ec815e1f2c`.
Полные отчёты: artifacts/review-fixes-20261022/tests.json и packaged-regressions/tests.json.
