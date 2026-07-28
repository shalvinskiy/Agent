#!/usr/bin/env python3
"""
Точка входа: запуск end-to-end агента суммаризации статей.

Пример:
  /usr/bin/python3 main.py
  /usr/bin/python3 main.py --input data/articles.json --output output/summaries.json
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

# Корень проекта в PYTHONPATH при запуске как скрипт
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import build_agent, default_user_message  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LangGraph + MCP агент суммаризации статей (Gemini Flash Lite)",
    )
    parser.add_argument(
        "--input",
        default="data/articles.json",
        help="Путь к JSON со статьями (относительно корня проекта)",
    )
    parser.add_argument(
        "--output",
        default="output/summaries.json",
        help="Путь для сохранения саммари",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Печатать промежуточные сообщения агента (tool calls / ответы)",
    )
    return parser.parse_args()


def _message_text(content: Any) -> str:
    """Достаёт текст из content (str | list частей Gemini)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


def _print_message(msg: object, verbose: bool) -> None:
    """Краткий лог шагов агента."""
    msg_type = getattr(msg, "type", type(msg).__name__)
    name = getattr(msg, "name", None)

    if msg_type == "ai":
        tool_calls = getattr(msg, "tool_calls", None) or []
        content = _message_text(getattr(msg, "content", ""))
        if tool_calls:
            calls = ", ".join(
                f"{c.get('name')}({list((c.get('args') or {}).keys())})"
                for c in tool_calls
            )
            print(f"[AI] tool_calls → {calls}")
        if content and (verbose or not tool_calls):
            print(f"[AI] {content[:500]}")
    elif msg_type == "tool":
        preview = str(getattr(msg, "content", ""))[:240].replace("\n", " ")
        print(f"[TOOL:{name}] {preview}...")
    elif verbose:
        print(f"[{msg_type}] {msg}")


def _tool_was_called(messages: list[Any], tool_name: str) -> bool:
    for msg in messages:
        if getattr(msg, "type", None) != "ai":
            continue
        for call in getattr(msg, "tool_calls", None) or []:
            if call.get("name") == tool_name:
                return True
    return False


def _last_ai_text(messages: list[Any]) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", None) != "ai":
            continue
        if getattr(msg, "tool_calls", None):
            continue
        text = _message_text(getattr(msg, "content", ""))
        if text.strip():
            return text
    return ""


async def _run_turn(
    agent: Any,
    messages: list[Any],
    verbose: bool,
) -> list[Any]:
    """Один проход агента; возвращает полную историю сообщений."""
    seen = len(messages)
    final_messages = messages
    async for event in agent.astream(
        {"messages": messages},
        stream_mode="values",
        config={"recursion_limit": 60},
    ):
        event_messages = event.get("messages") or []
        if not event_messages:
            continue
        final_messages = event_messages
        # печатаем только новые сообщения
        for msg in event_messages[seen:]:
            _print_message(msg, verbose=verbose)
        seen = len(event_messages)
    return final_messages


async def run(input_path: str, output_path: str, verbose: bool) -> int:
    agent, _client = await build_agent()
    user_text = default_user_message(input_path=input_path, output_path=output_path)
    out = PROJECT_ROOT / output_path

    print("=" * 60)
    print("Задача агенту:")
    print(user_text)
    print("=" * 60)

    messages: list[Any] = [HumanMessage(content=user_text)]

    # До 3 попыток: Gemini иногда останавливается после parallel summarize без save
    for attempt in range(1, 4):
        messages = await _run_turn(agent, messages, verbose=verbose)

        if out.exists():
            break

        if _tool_was_called(messages, "save_summaries"):
            # tool вызван, но файл не появился — нет смысла долбить дальше
            break

        if attempt < 3:
            nudge = (
                f"Саммари уже получены через summarize_article, но файл ещё не сохранён. "
                f"Сейчас обязательно вызови инструмент save_summaries: "
                f"path={output_path}, а summaries_json — JSON-массив всех "
                f'{{"id","title","summary"}} из ответов summarize_article. '
                f"Не пиши саммари текстом — только вызов инструмента."
            )
            print(f"[FOLLOW-UP #{attempt}] просим агента вызвать save_summaries")
            messages = list(messages) + [HumanMessage(content=nudge)]

    final_text = _last_ai_text(messages)

    print("=" * 60)
    print("Итог агента:")
    print(final_text or "(пустой финальный ответ)")
    print("=" * 60)

    if out.exists():
        print(f"Файл результата: {out}")
        return 0

    print(f"Внимание: файл {out} не найден — проверьте логи tool calls.")
    return 1


def main() -> None:
    args = parse_args()
    raise SystemExit(
        asyncio.run(
            run(
                input_path=args.input,
                output_path=args.output,
                verbose=args.verbose,
            )
        )
    )


if __name__ == "__main__":
    main()
