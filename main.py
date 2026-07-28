#!/usr/bin/env python3
"""Точка входа: агент суммаризации статей."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import build_agent, default_user_message  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangGraph + MCP agent")
    parser.add_argument("--input", default="data/articles.json")
    parser.add_argument("--output", default="output/summaries.json")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(getattr(item, "text", "") or ""))
        return "".join(parts)
    return str(content or "")


def _print_message(msg: object, verbose: bool) -> None:
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


async def run(input_path: str, output_path: str, verbose: bool) -> int:
    agent, _client = await build_agent()
    user_text = default_user_message(input_path=input_path, output_path=output_path)

    print("=" * 60)
    print("Задача агенту:")
    print(user_text)
    print("=" * 60)

    messages: list[Any] = [HumanMessage(content=user_text)]
    seen = 0
    async for event in agent.astream(
        {"messages": messages},
        stream_mode="values",
        config={"recursion_limit": 60},
    ):
        event_messages = event.get("messages") or []
        for msg in event_messages[seen:]:
            _print_message(msg, verbose=verbose)
        seen = len(event_messages)
        messages = event_messages

    final_text = ""
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai" and not getattr(msg, "tool_calls", None):
            final_text = _message_text(getattr(msg, "content", ""))
            if final_text.strip():
                break

    print("=" * 60)
    print("Итог агента:")
    print(final_text or "(пустой финальный ответ)")
    print("=" * 60)

    out = PROJECT_ROOT / output_path
    if out.exists():
        print(f"Файл результата: {out}")
        return 0
    print(f"Внимание: файл {out} не найден.")
    return 1


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(run(args.input, args.output, args.verbose)))


if __name__ == "__main__":
    main()
