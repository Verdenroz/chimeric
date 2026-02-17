"""Shared utility functions for message/tool normalisation and response creation.

These helpers are provider-agnostic; adapters and HttpClient both use them.
"""

from __future__ import annotations

from typing import Any

from .types import CompletionResponse, Input, Message, StreamChunk, Tool, ToolCall, Tools, Usage

__all__ = [
    "create_completion_response",
    "create_stream_chunk",
    "normalize_messages",
    "normalize_tools",
]


def normalize_messages(messages: Input) -> list[Message]:
    """Convert any supported input format to a list of Message objects."""
    if isinstance(messages, Message):
        return [messages]
    if isinstance(messages, str):
        return [Message(role="user", content=messages)]
    if isinstance(messages, dict):
        return [Message(**messages)]

    # Must be a list at this point
    normalized: list[Message] = []
    for msg in messages:
        if isinstance(msg, Message):
            normalized.append(msg)
        elif isinstance(msg, str):
            normalized.append(Message(role="user", content=msg))
        elif isinstance(msg, dict):
            normalized.append(Message(**msg))
        else:
            normalized.append(Message(role="user", content=str(msg)))
    return normalized


def normalize_tools(tools: Tools) -> list[Tool]:
    """Convert any supported tool representation to a list of Tool objects."""
    if not tools:
        return []
    normalized: list[Tool] = []
    for tool in tools:
        if isinstance(tool, Tool):
            normalized.append(tool)
        elif isinstance(tool, dict):
            normalized.append(Tool(**tool))
        else:
            normalized.append(
                Tool(
                    name=getattr(tool, "name", "unknown"),
                    description=getattr(tool, "description", ""),
                    parameters=getattr(tool, "parameters", None),
                    function=getattr(tool, "function", None),
                )
            )
    return normalized


def create_completion_response(
    content: str | list[Any],
    usage: Usage | None = None,
    model: str | None = None,
    tool_calls: list[ToolCall] | None = None,
    metadata: dict[str, Any] | None = None,
) -> CompletionResponse:
    """Build a CompletionResponse, embedding tool_calls in metadata if present."""
    response_metadata: dict[str, Any] = metadata or {}
    if tool_calls:
        response_metadata = {
            **response_metadata,
            "tool_calls": [tc.model_dump() for tc in tool_calls],
        }
    return CompletionResponse(
        content=content,
        usage=usage or Usage(),
        model=model,
        metadata=response_metadata if response_metadata else None,
    )


def create_stream_chunk(
    content: str,
    delta: str | None = None,
    finish_reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> StreamChunk:
    """Build a StreamChunk with the given fields."""
    return StreamChunk(
        content=content,
        delta=delta,
        finish_reason=finish_reason,
        metadata=metadata,
    )
