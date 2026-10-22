# Локальная карта Graphify

Исходники скачаны в .tools/graphify-src, отдельное окружение — .tools/graphify-env. Commit и версия записаны в graphify.lock.json. ML requirements и окружение обучения не изменяются.

Обновление из корня проекта:

```powershell
.\.venv\Scripts\python.exe scripts/update_graph.py
```

Скрипт выполняет локальный AST-разбор с --code-only и кластеризацию с --no-label. Документы не отправляются в LLM, API-ключ не нужен. .graphifyignore исключает данные, модели, результаты, окружения, submission и исходники Graphify.

Результаты: graphify-out/graph.html (интерактивный граф), graph.json (связи с источниками), GRAPH_REPORT.md (обзор). Граф показывает связи кода; он не доказывает корректность ML-методологии. Описание ролей находится в AGENT_ROLES.md.

```powershell
.\.tools\graphify-env\Scripts\graphify.exe explain PreparedCandidate --graph graphify-out/graph.json
.\.tools\graphify-env\Scripts\graphify.exe query "training_checkpoints" --graph graphify-out/graph.json
.\.tools\graphify-env\Scripts\graphify.exe path run_search worker_checkpoint --graph graphify-out/graph.json
```

На другой машине: клонировать официальный https://github.com/Graphify-Labs/graphify в служебную папку, checkout commit из lock-файла, создать отдельный venv и установить checkout через его python -m pip install .tools/graphify-src. На Linux исполняемые файлы находятся в bin вместо Scripts. Скрипт проверяет commit checkout. Обновление графа запускается вручную после изменения кода.
