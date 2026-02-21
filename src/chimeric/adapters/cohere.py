"""Cohere v2 Chat API adapter.

Cohere v2 is closer to OpenAI format than v1 was, but with key differences:
- Endpoint is /chat (not /chat/completions)
- Response body: message.content[0].text
- SSE: NDJSON-style with a "type" discriminator field
- Tool results use role "tool" with a document content type
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ..types import (
    CompletionResponse,
    EmbeddingResponse,
    EmbeddingUsage,
    Message,
    ModelSummary,
    StreamChunk,
    Tool,
    ToolCall,
    Usage,
)

if TYPE_CHECKING:
    from .base import StreamState

__all__ = ["CohereAdapter"]


class CohereAdapter:
    """Adapter for the Cohere v2 Chat API."""

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
        """Build a Cohere /v2/chat POST body."""
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
        """Parse a completed Cohere v2 chat response."""
        message = data.get("message", {})
        content_blocks: list[dict[str, Any]] = message.get("content", [])

        text_parts = [b["text"] for b in content_blocks if b.get("type") == "text"]
        content = "".join(text_parts)

        raw_usage = data.get("usage", {})
        tokens = raw_usage.get("tokens", {})
        usage = Usage(
            prompt_tokens=tokens.get("input_tokens", 0),
            completion_tokens=tokens.get("output_tokens", 0),
            total_tokens=tokens.get("input_tokens", 0) + tokens.get("output_tokens", 0),
        )

        return CompletionResponse(
            content=content,
            usage=usage,
            model=data.get("model", model),
            metadata={"finish_reason": data.get("finish_reason")},
        )

    def parse_tool_calls(self, data: dict[str, Any]) -> list[ToolCall] | None:
        """Extract tool calls from a Cohere v2 response."""
        message = data.get("message", {})
        tool_calls = message.get("tool_calls", []) or message.get("tool_plan_calls", [])

        if not tool_calls:
            # Check content blocks for tool_call type
            tool_blocks = [b for b in message.get("content", []) if b.get("type") == "tool_call"]
            if not tool_blocks:
                return None
            return [
                ToolCall(
                    call_id=b["id"],
                    name=b["function"]["name"],
                    arguments=_ensure_arguments_string(b["function"].get("arguments", {})),
                )
                for b in tool_blocks
            ]

        return [
            ToolCall(
                call_id=tc["id"],
                name=tc["function"]["name"],
                arguments=_ensure_arguments_string(tc["function"].get("arguments", {})),
            )
            for tc in tool_calls
        ]

    def build_tool_result_messages(
        self,
        messages: list[Message],
        tool_calls: list[ToolCall],
        results: list[str],
    ) -> list[Message]:
        """Append assistant tool-call and tool result messages (immutable)."""
        assistant_msg = Message(
            role="assistant",
            content="",
            tool_calls=tool_calls,
        )
        result_msgs = [
            Message(role="tool", content=result, tool_call_id=tc.call_id)
            for tc, result in zip(tool_calls, results, strict=False)
        ]
        return [*messages, assistant_msg, *result_msgs]

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def parse_sse_event(self, data: str, state: StreamState) -> StreamChunk | None:
        """Parse one Cohere NDJSON SSE event."""
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            return None

        event_type = event.get("type")

        if event_type == "content-delta":
            delta = event.get("delta", {}).get("message", {}).get("content", {})
            text = delta.get("text", "")
            state.accumulated_content += text
            return StreamChunk(
                content=state.accumulated_content,
                delta=text if text else None,
            )

        if event_type == "tool-call-start":
            tc = event.get("delta", {}).get("message", {}).get("tool_calls", {})
            idx = str(event.get("index", 0))
            state.tool_calls[idx] = {
                "id": tc.get("id", ""),
                "name": tc.get("function", {}).get("name", ""),
                "arguments": "",
            }
            return None

        if event_type == "tool-call-delta":
            delta = event.get("delta", {}).get("message", {}).get("tool_calls", {})
            idx = str(event.get("index", 0))
            if idx in state.tool_calls:
                state.tool_calls[idx]["arguments"] += delta.get("function", {}).get("arguments", "")
            return None

        if event_type == "message-end":
            finish_reason = event.get("delta", {}).get("finish_reason") or event.get(
                "finish_reason"
            )
            return StreamChunk(
                content=state.accumulated_content,
                finish_reason=finish_reason,
            )

        return None

    def finalize_stream(self, state: StreamState) -> list[ToolCall] | None:
        """Convert accumulated streaming tool calls."""
        if not state.tool_calls:
            return None
        calls = [
            ToolCall(
                call_id=tc["id"],
                name=tc["name"],
                arguments=tc.get("arguments", "{}") or "{}",
            )
            for tc in state.tool_calls.values()
        ]
        return calls if calls else None

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def build_embedding_request(
        self,
        input: str | list[str],
        model: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build a Cohere v2 /embed POST body."""
        texts = [input] if isinstance(input, str) else list(input)
        return {
            "model": model,
            "texts": texts,
            "input_type": kwargs.get("input_type", "search_document"),
            "embedding_types": ["float"],
        }

    def parse_embedding_response(
        self,
        data: dict[str, Any],
        model: str,
        *,
        batch: bool,
    ) -> EmbeddingResponse:
        """Parse a Cohere v2 /embed response."""
        vectors: list[list[float]] = data.get("embeddings", {}).get("float", [])
        meta = data.get("meta", {})
        tokens: int = meta.get("billed_units", {}).get("input_tokens", 0)
        usage = EmbeddingUsage(prompt_tokens=tokens, total_tokens=tokens)
        if batch:
            return EmbeddingResponse(embeddings=vectors, model=model, usage=usage)
        return EmbeddingResponse(
            embedding=vectors[0] if vectors else [],
            model=model,
            usage=usage,
        )

    # ------------------------------------------------------------------
    # Model listing
    # ------------------------------------------------------------------

    def parse_models_response(self, data: dict[str, Any]) -> list[ModelSummary]:
        """Parse GET /v2/models response."""
        return [
            ModelSummary(
                id=m.get("name", m.get("id", "")),
                name=m.get("name", m.get("id", "")),
            )
            for m in data.get("models", [])
        ]


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _ensure_arguments_string(arguments: Any) -> str:
    """Return arguments as a JSON string, handling both str and dict inputs."""
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments)


def _serialize_message(msg: Message) -> dict[str, Any]:
    """Convert a Message to Cohere v2 wire format."""
    if msg.role == "tool":
        # Tool results in Cohere v2 format
        return {
            "role": "tool",
            "tool_call_id": msg.tool_call_id or "",
            "content": str(msg.content),
        }

    if isinstance(msg.content, list):
        return {"role": msg.role, "content": msg.content}

    out: dict[str, Any] = {"role": msg.role, "content": str(msg.content or "")}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.call_id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments or "{}"},
            }
            for tc in msg.tool_calls
        ]
    return out


def _serialize_tool(tool: Tool) -> dict[str, Any]:
    """Convert a Tool to the Cohere v2 function schema."""
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
