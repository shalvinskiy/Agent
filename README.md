# Agent  — агент end-to-end суммаризации статей

Агент на **LangChain + LangGraph + MCP + Gemini 3.5 Flash Lite**, который сам:

1. читает JSON со статьями;
2. для каждой статьи вызывает инструмент суммаризации;
3. сохраняет результат в файл.

В Python **нет цикла** «пройтись по статьям и вызвать LLM». Оркестрацию делает LLM-агент через tool calls.

Интерпретатор: `/usr/bin/python3` (Python 3.10.12).

---

## Архитектура

```
main.py
   │
   ▼
agent/graph.py  ── LangGraph ReAct-агент (Gemini orchestrator)
   │
   │  MultiServerMCPClient (stdio)
   ▼
mcp_server/server.py  ── MCP FastMCP
   ├── read_articles_file
   ├── summarize_article   ── отдельный вызов Gemini (суммаризатор)
   └── save_summaries
```

| Компонент | Роль |
|-----------|------|
| `ChatGoogleGenerativeAI` (`gemini-3.5-flash-lite`) | Оркестратор: решает, какой tool вызвать |
| MCP-сервер (`FastMCP`, transport=`stdio`) | Отдаёт инструменты агенту по протоколу MCP |
| `langchain-mcp-adapters` | Превращает MCP tools в LangChain tools |
| `create_react_agent` (LangGraph) | ReAct-цикл: мысль → tool → наблюдение → … |
| `google.genai` внутри `summarize_article` | Отдельная LLM-суммаризация одной статьи |

---

## Структура проекта

```
Rag_lokis/
├── data/
│   └── articles.json          # входные статьи
├── output/
│   └── summaries.json         # результат (создаётся агентом)
├── mcp_server/
│   ├── __init__.py
│   └── server.py              # MCP-сервер и 3 инструмента
├── agent/
│   ├── __init__.py
│   └── graph.py               # сборка LangGraph-агента + MCP-клиент
├── main.py                    # точка входа
├── .env                       # GOOGLE_API_KEY и имена моделей
├── requirements.txt
└── README.md
```

---

## Установка

Используется системный Python `/usr/bin/python3`. Пакеты ставятся в систему **только если их ещё нет**.

Уже были в окружении: `langchain`, `langchain-core`, `python-dotenv`, `google-genai`.

Доустановлены отсутствующие:

```bash
pip3 install langgraph langchain-google-genai langchain-mcp-adapters 'mcp>=1.29.0,<2.0.0'
```

Или из файла:

```bash
/usr/bin/python3 -m pip install -r requirements.txt
```

> Важно: `mcp` нужен **1.29.x** (не 2.0) — иначе ломается `langchain-mcp-adapters`.

### Конфигурация `.env`

```env
GOOGLE_API_KEY=ваш_ключ
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_SUMMARIZER_MODEL=gemini-3.5-flash-lite
```

- `GEMINI_MODEL` — модель оркестратора (агент).
- `GEMINI_SUMMARIZER_MODEL` — модель внутри tool `summarize_article`.

---

## Запуск агента

Из корня проекта:

```bash
cd /home/romanrussia/Code_git/Rag_lokis
/usr/bin/python3 main.py
```

С путями явно:

```bash
/usr/bin/python3 main.py \
  --input data/articles.json \
  --output output/summaries.json
```

Подробный лог сообщений:

```bash
/usr/bin/python3 main.py --verbose
```

Ожидаемый результат: файл `output/summaries.json` вида:

```json
[
  {
    "id": "1",
    "title": "...",
    "summary": "..."
  }
]
```

---

## Как работает end-to-end (без Python-цикла по статьям)

1. `main.py` собирает агента и передаёт одну задачу текстом:
   > «Просуммируй все статьи из `data/articles.json` и сохрани в `output/summaries.json`.»
2. LangGraph ReAct-агент (Gemini) сам вызывает:
   - `read_articles_file` → получает список статей;
   - `summarize_article` **отдельно для каждой** статьи;
   - `save_summaries` → пишет итоговый JSON.
3. В коде нет `for article in articles: llm.invoke(...)`. Цикл — это последовательность tool calls, которую планирует сама модель.

---

## Разбор модулей и функций

### `main.py`

| Функция / участок | Назначение |
|-------------------|------------|
| `parse_args()` | CLI: `--input`, `--output`, `--verbose` |
| `_print_message(msg, verbose)` | Печатает шаги агента: AI tool_calls и ответы TOOL |
| `run(input_path, output_path, verbose)` | `await build_agent()`, стримит `agent.astream(...)`, проверяет наличие выходного файла |
| `main()` | Парсит аргументы и запускает `asyncio.run(run(...))` |

Поток выполнения:

```text
parse_args → build_agent → astream(HumanMessage) → лог tool calls → проверка output/
```

---

### `agent/graph.py`

| Функция | Назначение |
|---------|------------|
| `build_llm()` | Создаёт `ChatGoogleGenerativeAI` с `GEMINI_MODEL` (оркестратор, `temperature=0.1`) |
| `build_mcp_client()` | `MultiServerMCPClient`: поднимает MCP-сервер через stdio: `/usr/bin/python3 -m mcp_server.server` |
| `build_agent()` | `await client.get_tools()` → `create_react_agent(model, tools, prompt=SYSTEM_PROMPT)` |
| `default_user_message(input, output)` | Текст задачи пользователю/агенту |

`SYSTEM_PROMPT` жёстко требует:

- читать файл только через tool;
- суммаризировать **каждую** статью отдельным `summarize_article`;
- не писать саммари «из головы»;
- сохранить через `save_summaries`.

Параметры MCP-клиента:

```python
MultiServerMCPClient({
  "articles": {
    "command": "/usr/bin/python3",
    "args": ["-m", "mcp_server.server"],
    "transport": "stdio",
    "cwd": PROJECT_ROOT,
    "env": {..., "GOOGLE_API_KEY": ..., "GEMINI_SUMMARIZER_MODEL": ...},
  }
})
```

---

### `mcp_server/server.py`

MCP-сервер на `FastMCP` (`mcp.server.fastmcp`).

| Функция / tool | Назначение |
|----------------|------------|
| `_resolve_path(path)` | Относительные пути → абсолютные от корня проекта |
| `_get_genai_client()` | Ленивый `google.genai.Client` (не грузим SDK до первого summarize) |
| `read_articles_file(path)` | Читает JSON, валидирует массив, возвращает `{path, count, articles:[{id,title,text}]}` |
| `summarize_article(article_id, title, text)` | Отдельный вызов Gemini → JSON `{id, title, summary}` |
| `save_summaries(path, summaries_json)` | Парсит JSON-массив саммари, создаёт каталог, пишет файл |
| `main()` | `mcp.run(transport="stdio")` |

#### `read_articles_file`

- Вход: путь (`data/articles.json`).
- Выход: JSON-строка с статьями или `{"error": "..."}`.
- Ошибки: файл не найден, битый JSON, не массив.

#### `summarize_article`

- Вход: `article_id`, `title`, `text`.
- Внутри: prompt на русском «3–6 предложений, сохранить факты/цифры/сроки».
- Модель: `GEMINI_SUMMARIZER_MODEL` (= `gemini-3.5-flash-lite`).
- Выход: `{"id","title","summary"}` или `{"error": ...}`.

#### `save_summaries`

- Вход: путь выхода + `summaries_json` (строка с JSON-массивом).
- Создаёт `output/` при необходимости.
- Выход: `{"status":"ok","path":"...","saved_count":N}`.

Ручной запуск MCP-сервера (обычно не нужен — клиент поднимает сам):

```bash
cd /home/romanrussia/Code_git/Rag_lokis
PYTHONPATH=. /usr/bin/python3 -m mcp_server.server
```

---

## Данные

Вход `data/articles.json` — массив объектов:

```json
[
  {
    "id": "1",
    "title": "Заголовок",
    "text": "Полный текст статьи..."
  }
]
```

В репозитории 10 корпоративных статей (регламенты, инциденты, SLA, отчёты и т.д.).

---

## Почему MCP + LangGraph

- **MCP** — единый протокол инструментов; сервер можно переиспользовать из других клиентов.
- **langchain-mcp-adapters** — без ручного дублирования схем tools в LangChain.
- **LangGraph `create_react_agent`** — стандартный агентный цикл с tool calling; агент сам решает порядок и число вызовов.

---

## Типичный лог успешного прогона

```text
[AI] tool_calls → read_articles_file(['path'])
[TOOL:read_articles_file] {"path":"...","count":10,"articles":[...]}...
[AI] tool_calls → summarize_article(['article_id', 'title', 'text'])
[TOOL:summarize_article] {"id":"1","title":"...","summary":"..."}...
... (ещё 9 раз summarize_article) ...
[AI] tool_calls → save_summaries(['path', 'summaries_json'])
[TOOL:save_summaries] {"status":"ok","path":".../output/summaries.json","saved_count":10}...
[AI] Обработано 10 статей, результат: output/summaries.json
```

---

## Устранение проблем

| Симптом | Что проверить |
|---------|----------------|
| `Не задан GOOGLE_API_KEY` | Файл `.env` в корне, переменная не пустая |
| `429 RESOURCE_EXHAUSTED` | Квота Gemini; смените модель в `.env` или подождите |
| `ImportError: RequestContext` / MCP | Версия `mcp` должна быть `<2.0` (рекомендуется `1.29.0`) |
| MCP tools пустой список | `PYTHONPATH` / `cwd` = корень проекта; модуль `mcp_server.server` импортируется |
| Нет `output/summaries.json` | Запустите с `--verbose`, смотрите, вызывался ли `save_summaries` |
| Агент сам пишет саммари без tool | Усилен `SYSTEM_PROMPT`; перезапустите; temperature уже низкая |

Проверка, что MCP отдаёт tools:

```bash
cd /home/romanrussia/Code_git/Rag_lokis
/usr/bin/python3 - <<'PY'
import asyncio
from agent.graph import build_mcp_client

async def main():
    client = build_mcp_client()
    tools = await client.get_tools()
    print([t.name for t in tools])

asyncio.run(main())
PY
```

Ожидается:

```text
['read_articles_file', 'summarize_article', 'save_summaries']
```

---

## Зависимости (кратко)

| Пакет | Зачем |
|-------|--------|
| `langchain` / `langchain-core` | Сообщения, tools, LLM-абстракции |
| `langgraph` | `create_react_agent`, граф агента |
| `langchain-google-genai` | Оркестратор Gemini в LangChain |
| `langchain-mcp-adapters` | MCP → LangChain tools |
| `mcp` (1.29) | FastMCP-сервер |
| `google-genai` | Вызов Gemini внутри `summarize_article` |
| `python-dotenv` | Загрузка `.env` |
