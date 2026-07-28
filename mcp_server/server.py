#!/usr/bin/env python3
"""
MCP-сервер инструментов для end-to-end суммаризации статей.

Инструменты:
  - read_articles_file  — прочитать JSON с статьями
  - summarize_article   — суммаризировать одну статью через Gemini
  - save_summaries      — сохранить результаты в JSON

Запуск (обычно делает MultiServerMCPClient автоматически):
  /usr/bin/python3 -m mcp_server.server
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Корень проекта: .../Rag_lokis
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

mcp = FastMCP(
    name="articles-summarizer-tools",
    instructions=(
        "Инструменты для чтения JSON со статьями, суммаризации каждой статьи "
        "и сохранения итогового файла. Вызывай summarize_article отдельно "
        "для каждой статьи — не суммируй весь файл одним вызовом."
    ),
)


def _resolve_path(path: str) -> Path:
    """Превращает относительный путь в абсолютный относительно корня проекта."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _get_genai_client():
    """Ленивая инициализация клиента Gemini (экономия памяти до первого вызова)."""
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Не задан GOOGLE_API_KEY в окружении или .env")
    return genai.Client(api_key=api_key)


@mcp.tool()
def read_articles_file(path: str) -> str:
    """
    Читает JSON-файл со статьями и возвращает его содержимое как текст.

    Ожидаемый формат: список объектов с полями id, title, text.

    Args:
        path: путь к JSON-файлу (абсолютный или относительно корня проекта),
              например data/articles.json
    """
    file_path = _resolve_path(path)
    if not file_path.exists():
        return json.dumps(
            {"error": f"Файл не найден: {file_path}"},
            ensure_ascii=False,
        )
    if not file_path.is_file():
        return json.dumps(
            {"error": f"Путь не является файлом: {file_path}"},
            ensure_ascii=False,
        )

    try:
        raw = file_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return json.dumps(
            {"error": f"Некорректный JSON: {exc}"},
            ensure_ascii=False,
        )
    except OSError as exc:
        return json.dumps(
            {"error": f"Ошибка чтения файла: {exc}"},
            ensure_ascii=False,
        )

    if not isinstance(data, list):
        return json.dumps(
            {"error": "Ожидался JSON-массив статей"},
            ensure_ascii=False,
        )

    # Возвращаем компактный JSON: id/title/text для каждой статьи
    articles: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        articles.append(
            {
                "id": str(item.get("id", "")),
                "title": str(item.get("title", "")),
                "text": str(item.get("text", "")),
            }
        )

    return json.dumps(
        {
            "path": str(file_path),
            "count": len(articles),
            "articles": articles,
        },
        ensure_ascii=False,
    )


@mcp.tool()
def summarize_article(article_id: str, title: str, text: str) -> str:
    """
    Суммаризирует одну статью через отдельный вызов Gemini Flash.

    Вызывай этот инструмент отдельно для каждой статьи из файла.

    Args:
        article_id: идентификатор статьи (поле id)
        title: заголовок статьи
        text: полный текст статьи
    """
    model = os.getenv("GEMINI_SUMMARIZER_MODEL", "gemini-3.5-flash-lite")
    prompt = (
        "Ты — ассистент по суммаризации корпоративных документов.\n"
        "Сделай краткое, информативное резюме на русском языке (3–6 предложений).\n"
        "Сохрани ключевые факты, цифры, сроки и решения. Без вступлений и метакомментариев.\n\n"
        f"Заголовок: {title}\n\n"
        f"Текст:\n{text}"
    )

    try:
        client = _get_genai_client()
        response = client.models.generate_content(model=model, contents=prompt)
        summary = (getattr(response, "text", None) or "").strip()
        if not summary:
            summary = str(response).strip()
    except Exception as exc:  # noqa: BLE001 — отдаём ошибку агенту как JSON
        return json.dumps(
            {
                "id": article_id,
                "title": title,
                "error": f"Ошибка суммаризации: {exc}",
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "id": article_id,
            "title": title,
            "summary": summary,
        },
        ensure_ascii=False,
    )


@mcp.tool()
def save_summaries(path: str, summaries_json: str) -> str:
    """
    Сохраняет результаты суммаризации в JSON-файл.

    Args:
        path: путь к выходному файлу (например output/summaries.json)
        summaries_json: JSON-строка — массив объектов
            [{"id": "...", "title": "...", "summary": "..."}, ...]
    """
    out_path = _resolve_path(path)

    try:
        data = json.loads(summaries_json)
    except json.JSONDecodeError as exc:
        return json.dumps(
            {"error": f"summaries_json не является валидным JSON: {exc}"},
            ensure_ascii=False,
        )

    if not isinstance(data, list):
        return json.dumps(
            {"error": "summaries_json должен быть JSON-массивом"},
            ensure_ascii=False,
        )

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        return json.dumps(
            {"error": f"Не удалось сохранить файл: {exc}"},
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "status": "ok",
            "path": str(out_path),
            "saved_count": len(data),
        },
        ensure_ascii=False,
    )


def main() -> None:
    """Точка входа MCP-сервера (stdio)."""
    # stdio — транспорт, который поднимает MultiServerMCPClient
    mcp.run(transport="stdio")


if __name__ == "__main__":
    # Гарантируем, что корень проекта есть в PYTHONPATH при прямом запуске
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    main()
