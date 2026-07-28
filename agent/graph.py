"""
Сборка LangGraph ReAct-агента с инструментами из MCP-сервера.

Агент сам решает, когда читать файл, вызывать суммаризатор и сохранять результат.
Цикла «для каждой статьи вызвать LLM» в Python нет — оркестрация на стороне LLM.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

SYSTEM_PROMPT = """Ты — агент суммаризации корпоративных статей.

Обязательный pipeline (нельзя останавливаться раньше пункта 3):
1) Вызови read_articles_file и получи список статей.
2) Для КАЖДОЙ статьи вызови summarize_article(article_id, title, text).
   Саммари пиши только через этот инструмент, не сочиняй текст сам.
3) ОБЯЗАТЕЛЬНО вызови save_summaries:
   - path = путь выходного файла из задачи пользователя;
   - summaries_json = JSON-массив (строка) вида
     [{"id":"...","title":"...","summary":"..."}, ...]
     со ВСЕМИ саммари, которые вернул summarize_article.
4) Только после успешного save_summaries дай короткий текстовый итог:
   число статей и путь к файлу.

Правила:
- Не завершай работу без вызова save_summaries.
- Не выдумывай содержимое статей и саммари.
- Если саммари уже получены, следующим шагом сразу save_summaries.
"""


def build_llm() -> ChatGoogleGenerativeAI:
    """Создаёт LLM-оркестратор (Gemini Flash Lite) для ReAct-агента."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Не задан GOOGLE_API_KEY в .env")

    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    # temperature не задаём: gemini-3.5-flash-lite использует фиксированные sampling defaults
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        max_output_tokens=8192,
    )


def build_mcp_client() -> MultiServerMCPClient:
    """
    Подключает MCP-сервер инструментов через stdio.

    MultiServerMCPClient сам запускает процесс:
      /usr/bin/python3 -m mcp_server.server
    """
    return MultiServerMCPClient(
        {
            "articles": {
                "command": "/usr/bin/python3",
                "args": ["-m", "mcp_server.server"],
                "transport": "stdio",
                "cwd": str(PROJECT_ROOT),
                "env": {
                    **os.environ,
                    "PYTHONPATH": str(PROJECT_ROOT),
                    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
                    "GEMINI_SUMMARIZER_MODEL": os.getenv(
                        "GEMINI_SUMMARIZER_MODEL", "gemini-3.5-flash-lite"
                    ),
                },
            }
        }
    )


async def build_agent() -> tuple[Any, MultiServerMCPClient]:
    """
    Загружает MCP-инструменты и собирает LangGraph create_react_agent.

    Returns:
        (compiled_graph, mcp_client) — клиент нужно держать живым на время работы.
    """
    client = build_mcp_client()
    tools = await client.get_tools()
    if not tools:
        raise RuntimeError("MCP-сервер не вернул ни одного инструмента")

    llm = build_llm()
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        name="articles_summarizer_agent",
    )
    return agent, client


def default_user_message(
    input_path: str = "data/articles.json",
    output_path: str = "output/summaries.json",
) -> str:
    """Формирует пользовательскую задачу для агента."""
    return (
        f"Просуммируй все статьи из файла {input_path} "
        f"и сохрани результат в {output_path}. "
        f"После всех summarize_article обязательно вызови save_summaries "
        f"с path={output_path}."
    )
