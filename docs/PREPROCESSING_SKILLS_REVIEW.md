# Ревью preprocessing-skills — 22 октября 2026

Источник: [коммит тиммейта 9b50044](https://github.com/epitaph76/MLmom/commit/9b500446ebb06837729eec36546ae66c269106e7).
Приняты все девять карточек: восемь новых инструкций и объединённый imbalance.
Исходная ветка основана на более ранней версии проекта; её изменения объединены
с текущей архитектурой ролей без отката остальных компонентов.

| Карточка | Роль | Результат ревью |
|---|---|---|
| categorical_encoding | preprocessing_agent | Train-only словарь, unseen fallback, sparse OHE; target encoding только OOF с учётом групп |
| missing_values | preprocessing_agent | Простая fold-local импутация, all-missing колонки, признаки пропусков и сохранение состояния |
| feature_selection | preprocessing_agent | Отбор и PCA внутри train-fold, без отбора на полной размеченной таблице |
| numeric_transforms | preprocessing_agent | Обоснованные scaling/clipping, sparse без центрирования; преобразование y делегировано кодеру модели |
| preprocessing_core | preprocessing_agent | Автоматический базовый контракт; новый fit учит свежую предобработку, load/predict используют сохранённое состояние |
| schema_quality | preprocessing_agent | Дубли и шумные метки — диагностические находки; preprocessor не удаляет строки и не исправляет y |
| temporal_group_structure | preprocessing_agent | Split принадлежит оркестратору; target aggregates требуют group-safe OOF, frequency maps учатся на train |
| text_features | preprocessing_agent | Ограниченный sparse TF-IDF/hash, train-only vocabulary/IDF, без новых библиотек и pretrained моделей |
| imbalance | coding_agent | Объединена с существующей карточкой; weights/sampling/thresholds, экспорт порогов для refit, без предположения о поддержке multilabel у SMOTE |

Удалён дублирующий ID imbalance, папка Preprocessing заменена владельцами runtime.
Карточки не устанавливают зависимости и не запускают преобразования автоматически.
Планировщик получает совместимый каталог метаданных и выбирает технику по данным;
инструкции получает только исполняющая роль.

preprocessing.py сохраняет число и порядок строк при fit_transform и transform.
Любое согласованное sampling обучающей выборки выполняется на стороне estimator
с одинаковым индексированием X, y, groups и sample_weight. Validation/test не меняются.
Явные временные задачи по-прежнему требуют поддержанного runtime-контракта.

Проверки: уникальность и владение скилов, совместимость задач, бюджеты контекста,
обнаружение техник планировщиком, передача выбранных инструкций нужной роли,
двухраундовый pipeline и resume, наличие карточек в автономном ZIP.
Проверки с подставным LLM подтверждают подключение; они не измеряют качество
будущего сгенерированного препроцессинга или прирост метрики.
