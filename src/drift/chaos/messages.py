"""Chat-message helpers for the chaos engine.

LangGraph agents almost always carry their conversation in a `messages`
channel: a list of langchain `BaseMessage` objects, `{"role": ..., "content":
...}` dicts, `{"type": ..., "content": ...}` dicts, or `(role, content)`
tuples — all of which `add_messages` coerces. The generic list patterns
(`clear_list`, `reverse_list`, ...) treat that list as opaque, which is
both too weak (they can't say "drop the user's instruction") and sometimes
wrong (reversing a conversation produces an API-invalid sequence — a
ToolMessage before its tool call — so the "crash" is drift's artefact, not
the graph's bug).

This module recognises a messages channel and lets the message-aware
patterns in `drift.chaos.patterns` read and construct messages in whatever
dialect the user's state already uses, so the perturbed state is still
something the graph would accept.
"""
from __future__ import annotations

from typing import Any

# Canonical role names (langchain `type` values). OpenAI-style roles map onto them.
_ROLE_ALIASES: dict[str, str] = {
    "user": "human",
    "human": "human",
    "assistant": "ai",
    "ai": "ai",
    "tool": "tool",
    "function": "tool",
    "system": "system",
    "developer": "system",
}


def _is_langchain_message(value: Any) -> bool:
    if isinstance(value, (dict, list, tuple, set, str, bytes, int, float, bool)) or value is None:
        return False
    return (
        hasattr(value, "type")
        and hasattr(value, "content")
        and callable(getattr(value, "model_copy", None))
    )


def is_message_like(value: Any) -> bool:
    """One chat message in any accepted dialect."""
    if _is_langchain_message(value):
        return True
    if isinstance(value, dict):
        role = value.get("role", value.get("type"))
        return isinstance(role, str) and role.lower() in _ROLE_ALIASES and "content" in value
    if isinstance(value, tuple) and len(value) == 2:
        return isinstance(value[0], str) and value[0].lower() in _ROLE_ALIASES and isinstance(value[1], str)
    return False


def is_messages_list(value: Any) -> bool:
    """A non-empty list where EVERY element is message-like."""
    return isinstance(value, list) and len(value) > 0 and all(is_message_like(m) for m in value)


def message_role(msg: Any) -> str:
    """Canonical role: 'human' | 'ai' | 'tool' | 'system' | ''."""
    if _is_langchain_message(msg):
        return _ROLE_ALIASES.get(str(getattr(msg, "type", "")).lower(), str(getattr(msg, "type", "")))
    if isinstance(msg, dict):
        role = msg.get("role", msg.get("type", ""))
        return _ROLE_ALIASES.get(str(role).lower(), str(role))
    if isinstance(msg, tuple) and len(msg) == 2:
        return _ROLE_ALIASES.get(str(msg[0]).lower(), str(msg[0]))
    return ""


def message_text(msg: Any) -> str:
    """Text content of a message. Content-block lists (Anthropic style) are
    joined on their `text` fields."""
    if _is_langchain_message(msg):
        content = getattr(msg, "content", "")
    elif isinstance(msg, dict):
        content = msg.get("content", "")
    elif isinstance(msg, tuple) and len(msg) == 2:
        content = msg[1]
    else:
        content = ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


def message_tool_calls(msg: Any) -> list[dict]:
    """`[{"name", "args", "id"}, ...]` for an AI message, else []."""
    if _is_langchain_message(msg):
        calls = getattr(msg, "tool_calls", None) or []
    elif isinstance(msg, dict):
        calls = msg.get("tool_calls") or []
    else:
        calls = []
    return [c for c in calls if isinstance(c, dict)]


def last_index_with_role(msgs: list[Any], role: str) -> int:
    for i in range(len(msgs) - 1, -1, -1):
        if message_role(msgs[i]) == role:
            return i
    return -1


def last_tool_name(msgs: list[Any]) -> str | None:
    """Name of the most recently called tool in the conversation, if any."""
    for m in reversed(msgs):
        calls = message_tool_calls(m)
        if calls and isinstance(calls[-1].get("name"), str):
            return calls[-1]["name"]
    return None


def dialect_of(msgs: list[Any]) -> str:
    """'langchain' | 'role_dict' | 'type_dict' | 'tuple' — the shape new
    messages should take so the perturbed list stays homogeneous. Decided
    from the LAST element (the most recent turn is the best template)."""
    if not msgs:
        return "role_dict"
    last = msgs[-1]
    if _is_langchain_message(last):
        return "langchain"
    if isinstance(last, dict):
        return "role_dict" if "role" in last else "type_dict"
    if isinstance(last, tuple):
        return "tuple"
    return "role_dict"


def with_text(msg: Any, new_text: str) -> Any:
    """Copy of `msg` with its text content replaced, same dialect."""
    if _is_langchain_message(msg):
        return msg.model_copy(update={"content": new_text})
    if isinstance(msg, dict):
        out = dict(msg)
        out["content"] = new_text
        return out
    if isinstance(msg, tuple) and len(msg) == 2:
        return (msg[0], new_text)
    return msg


def make_message(
    dialect: str,
    role: str,
    content: str,
    *,
    tool_calls: list[dict] | None = None,
    tool_call_id: str | None = None,
    status: str | None = None,
) -> Any:
    """Build one message in `dialect`. `role` is canonical ('human'|'ai'|'tool').

    Tool and AI-with-tool-calls messages can't be tuples; those fall back to
    role dicts, which `add_messages` coerces fine alongside tuples.
    """
    if dialect == "langchain":
        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
        except Exception:  # noqa: BLE001 — langchain_core not importable; degrade to dicts
            dialect = "type_dict"
        else:
            if role == "ai":
                return AIMessage(content=content, tool_calls=list(tool_calls or []))
            if role == "tool":
                return ToolMessage(content=content, tool_call_id=tool_call_id or "", status=status or "success")
            if role == "system":
                return SystemMessage(content=content)
            return HumanMessage(content=content)

    if dialect == "tuple" and role in ("human", "system") and not tool_calls:
        return ("user" if role == "human" else "system", content)

    key = "type" if dialect == "type_dict" else "role"
    if key == "role":
        role_out = {"human": "user", "ai": "assistant", "tool": "tool", "system": "system"}.get(role, role)
    else:
        role_out = role
    out: dict[str, Any] = {key: role_out, "content": content}
    if tool_calls:
        out["tool_calls"] = list(tool_calls)
    if role == "tool":
        out["tool_call_id"] = tool_call_id or ""
        if status:
            out["status"] = status
    return out


__all__ = [
    "dialect_of",
    "is_message_like",
    "is_messages_list",
    "last_index_with_role",
    "last_tool_name",
    "make_message",
    "message_role",
    "message_text",
    "message_tool_calls",
    "with_text",
]
