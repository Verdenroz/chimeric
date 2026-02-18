"""OpenAI Chat Completions adapter.

Shared by OpenAI, Groq, Cerebras, xAI/Grok, and OpenRouter — they all
implement the same /v1/chat/completions wire format.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ..types import CompletionResponse, Message, ModelSummary, StreamChunk, Tool, ToolCall, Usage

if TYPE_CHECKING:
    from .base import StreamState

__all__ = ["OpenAIAdapter"]


class OpenAIAdapter:
    """Adapter for the OpenAI Chat Completions API format."""

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def build_request_body(
        self,
        messages: list[Message],
        model: str,
        stream: bool,
        tools: list[Tool] | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a /v1/chat/completions POST body."""
        # OpenAI uses max_tokens; normalize the Anthropic/Google alias
        if "max_output_tokens" in kwargs:
            kwargs.setdefault("max_tokens", kwargs.pop("max_output_tokens"))
        body: dict[str, Any] = {
            "model": model,
            "messages": [_serialize_message(m) for m in messages],
            "stream": stream,
            **kwargs,
        }
        if tools:
            body["tools"] = [_serialize_tool(t) for t in tools]
        return body

    # ------------------------------------------------------------------
    # Non-streaming response parsing
    # ------------------------------------------------------------------

    def parse_response(self, data: dict[str, Any], model: str) -> CompletionResponse:
        """Parse a completed /v1/chat/completions response."""
        choice = data["choices"][0]
        message = choice["message"]
        content: str = message.get("content") or ""

        raw_usage = data.get("usage", {})
        usage = Usage(
            prompt_tokens=raw_usage.get("prompt_tokens", 0),
            completion_tokens=raw_usage.get("completion_tokens", 0),
            total_tokens=raw_usage.get("total_tokens", 0),
        )

        metadata: dict[str, Any] = {"finish_reason": choice.get("finish_reason")}
        if data.get("id"):
            metadata["id"] = data["id"]

        return CompletionResponse(
            content=content,
            usage=usage,
            model=data.get("model", model),
            metadata=metadata,
        )

    def parse_tool_calls(self, data: dict[str, Any]) -> list[ToolCall] | None:
        """Return ToolCall objects if the model issued function calls."""
        raw_calls = data["choices"][0]["message"].get("tool_calls")
        if not raw_calls:
            return None
        return [
            ToolCall(
                call_id=tc["id"],
                name=tc["function"]["name"],
                arguments=tc["function"].get("arguments", "{}"),
            )
            for tc in raw_calls
        ]

    def build_tool_result_messages(
        self,
        messages: list[Message],
        tool_calls: list[ToolCall],
        results: list[str],
    ) -> list[Message]:
        """Append assistant tool-call message and tool result messages (immutable)."""
        # Reconstruct the assistant turn that issued the calls
        assistant_msg = Message(
            role="assistant",
            content="",
            tool_calls=tool_calls,
        )
        # One tool-result message per call
        result_msgs = [
            Message(role="tool", content=result, tool_call_id=tc.call_id)
            for tc, result in zip(tool_calls, results, strict=False)
        ]
        return [*messages, assistant_msg, *result_msgs]

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def parse_sse_event(self, data: str, state: StreamState) -> StreamChunk | None:
        """Parse one SSE data line from the streaming response."""
        if data == "[DONE]":
            return None

        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            return None

        choices = event.get("choices", [])
        if not choices:
            return None

        delta = choices[0].get("delta", {})
        finish_reason: str | None = choices[0].get("finish_reason")

        # Accumulate tool call argument deltas
        for tc_delta in delta.get("tool_calls", []):
            idx = str(tc_delta.get("index", 0))
            if idx not in state.tool_calls:
                state.tool_calls[idx] = {
                    "id": tc_delta.get("id", ""),
                    "name": tc_delta.get("function", {}).get("name", ""),
                    "arguments": "",
                }
            state.tool_calls[idx]["arguments"] += tc_delta.get("function", {}).get("arguments", "")
            # Update id/name if this chunk carries them
            if tc_delta.get("id"):
                state.tool_calls[idx]["id"] = tc_delta["id"]
            if tc_delta.get("function", {}).get("name"):
                state.tool_calls[idx]["name"] = tc_delta["function"]["name"]

        content_delta: str = delta.get("content") or ""
        state.accumulated_content += content_delta

        return StreamChunk(
            content=state.accumulated_content,
            delta=content_delta if content_delta else None,
            finish_reason=finish_reason,
        )

    def finalize_stream(self, state: StreamState) -> list[ToolCall] | None:
        """Convert accumulated streaming tool calls into ToolCall objects."""
        if not state.tool_calls:
            return None
        calls = [
            ToolCall(
                call_id=tc["id"],
                name=tc["name"],
                arguments=tc.get("arguments", "{}"),
            )
            for tc in state.tool_calls.values()
        ]
        return calls if calls else None

    # ------------------------------------------------------------------
    # Model listing
    # ------------------------------------------------------------------

    def parse_models_response(self, data: dict[str, Any]) -> list[ModelSummary]:
        """Parse GET /models response."""
        return [
            ModelSummary(
                id=m["id"],
                name=m["id"],
                owned_by=m.get("owned_by"),
                created_at=m.get("created"),
            )
            for m in data.get("data", [])
        ]


# ---------------------------------------------------------------------------
# Module-level serialization helpers
# ---------------------------------------------------------------------------


def _serialize_message(msg: Message) -> dict[str, Any]:
    """Convert a Message to the OpenAI wire format."""
    out: dict[str, Any] = {"role": msg.role, "content": msg.content or ""}
    if msg.name:
        out["name"] = msg.name
    if msg.tool_call_id:
        out["tool_call_id"] = msg.tool_call_id
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.call_id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in msg.tool_calls
        ]
    return out


def _serialize_tool(tool: Tool) -> dict[str, Any]:
    """Convert a Tool to the OpenAI function-calling wire format."""
    schema: dict[str, Any] = {}
    if tool.parameters:
        schema = tool.parameters.model_dump(exclude_none=True)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": schema,
        },
    }
