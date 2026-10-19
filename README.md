# MLmom

Основа ML Research Agent: Pi Web управляет сессиями Pi, TypeScript extension вызывает Python backend и асинхронный Research API. Сейчас готовы инфраструктура и тестовый поиск; обучение и оценка моделей ещё не реализованы.

## Запуск

Нужны Node.js ≥22.19, Python ≥3.12 и Git. Проверено на Windows с Node 24 и Python 3.14; те же npm-команды предназначены для Linux.

```sh
npm ci
npm run setup
```

В созданном `.env` задайте `ROUTERAI_API_KEY`, затем:

```sh
npm start
```

Откройте [Pi Web](http://127.0.0.1:30141), выберите папку этого репозитория как проект и разрешите загрузку его расширения, если Pi запросит доверие. В новой сессии выберите RouterAI и попросите: «Вызови get_capabilities, затем research для проверки интеграции; дождись результата через get_research_status».

По умолчанию backend работает на `8010`, Research mock — на `8011`. `Ctrl+C` останавливает процессы запуска; ключ не нужен для локальных API и offline-тестов, но нужен для ответа LLM в интерфейсе.

В PowerShell при блокировке `npm.ps1` используйте `npm.cmd` вместо `npm`. Для Python с другим именем задайте переменную `PYTHON` перед `npm run setup`.

## Проверка

```sh
npm run check
```

Проверяются экспорт OpenAPI, TypeScript, Python API, HTTP-клиент и реальный цикл Pi со сценарным offline-провайдером. Вызовов внешних моделей в этих тестах нет.

При работающем `npm start` доступны дополнительные проверки:

```sh
npm run smoke:pi
npm run smoke:router
```

`smoke:pi` использует offline LLM и Research mock. `smoke:router` явно обращается к платному RouterAI и проверяет ответ, streaming и вызов инструмента backend через Pi.

## Интеграция

- [Архитектура и настройки](docs/architecture.md)
- [Контракт поиска для тиммейта](docs/research-api.md)
- [Research OpenAPI 3.1](contracts/research.openapi.json)
- [Backend OpenAPI 3.1](contracts/backend.openapi.json)
- [JSON-примеры](contracts/examples)
- [Фактически выполненные проверки](docs/verification.md)

После изменения Python-моделей или маршрутов выполните `npm run contracts` и включите оба OpenAPI-файла и сгенерированные TypeScript-типы в изменение. MLflow, Research KB, обучение и бюджет экспериментов пока не подключены; контейнеров нет.
