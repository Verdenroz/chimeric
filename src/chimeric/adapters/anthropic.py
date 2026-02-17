"""Anthropic Messages API adapter.

Notable differences from the OpenAI format:
- System prompt is extracted from the message list into a top-level "system" key
- Tool schemas use "input_schema" instead of "parameters".
- Tool results use role "user" with content type "tool_result".
- SSE event types differ: "content_block_delta" carries text deltas.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ..types import CompletionResponse, Message, ModelSummary, StreamChunk, Tool, ToolCall, Usage

if TYPE_CHECKING:
    from .base import StreamState

__all__ = ["AnthropicAdapter"]


class AnthropicAdapter:
    """Adapter for the Anthropic Messages API."""

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
        """Build an Anthropic /v1/messages POST body.

        System messages are extracted from the message list and placed in
        the top-level "system" field as the API requires.
        """
        system_parts: list[str] = []
        non_system: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(str(msg.content))
            else:
                non_system.append(_serialize_message(msg))

        body: dict[str, Any] = {
            "model": model,
            "messages": non_system,
            "stream": stream,
            **kwargs,
        }
        if system_parts:
            body["system"] = "\n".join(system_parts)
        if tools:
            body["tools"] = [_serialize_tool(t) for t in tools]
        return body

    # ------------------------------------------------------------------
    # Non-streaming response parsing
    # ------------------------------------------------------------------

    def parse_response(self, data: dict[str, Any], model: str) -> CompletionResponse:
        """Parse a completed Anthropic Messages response."""
        content_blocks: list[dict[str, Any]] = data.get("content", [])

        # Collect text from all text blocks
        text_parts = [b["text"] for b in content_blocks if b.get("type") == "text"]
        content = "".join(text_parts)

        raw_usage = data.get("usage", {})
        usage = Usage(
            prompt_tokens=raw_usage.get("input_tokens", 0),
            completion_tokens=raw_usage.get("output_tokens", 0),
            total_tokens=raw_usage.get("input_tokens", 0) + raw_usage.get("output_tokens", 0),
        )

        metadata: dict[str, Any] = {"stop_reason": data.get("stop_reason")}
        if data.get("id"):
            metadata["id"] = data["id"]

        return CompletionResponse(
            content=content,
            usage=usage,
            model=data.get("model", model),
            metadata=metadata,
        )

    def parse_tool_calls(self, data: dict[str, Any]) -> list[ToolCall] | None:
        """Extract tool_use blocks from the response content."""
        tool_blocks = [b for b in data.get("content", []) if b.get("type") == "tool_use"]
        if not tool_blocks:
            return None
        return [
            ToolCall(
                call_id=b["id"],
                name=b["name"],
                arguments=json.dumps(b.get("input", {})),
            )
            for b in tool_blocks
        ]

    def build_tool_result_messages(
        self,
        messages: list[Message],
        tool_calls: list[ToolCall],
        results: list[str],
    ) -> list[Message]:
        """Append assistant turn and tool results in Anthropic format (immutable)."""
        # Reconstruct the assistant turn that produced the calls
        assistant_content: list[dict[str, Any]] = [
            {
                "type": "tool_use",
                "id": tc.call_id,
                "name": tc.name,
                "input": json.loads(tc.arguments),
            }
            for tc in tool_calls
        ]
        assistant_msg = Message(role="assistant", content=assistant_content)

        # Tool results go in a single user message with multiple content blocks
        result_blocks: list[dict[str, Any]] = [
            {
                "type": "tool_result",
                "tool_use_id": tc.call_id,
                "content": result,
            }
            for tc, result in zip(tool_calls, results, strict=False)
        ]
        result_msg = Message(role="user", content=result_blocks)

        return [*messages, assistant_msg, result_msg]

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def parse_sse_event(self, data: str, state: StreamState) -> StreamChunk | None:
        """Parse one Anthropic SSE data payload."""
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            return None

        event_type = event.get("type")

        if event_type == "content_block_start":
            block = event.get("content_block", {})
            if block.get("type") == "tool_use":
                # New tool call block starting
                idx = str(event.get("index", 0))
                state.tool_calls[idx] = {
                    "id": block["id"],
                    "name": block["name"],
                    "arguments": "",
                }
            return None

        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            delta_type = delta.get("type")

            if delta_type == "text_delta":
                text = delta.get("text", "")
                state.accumulated_content += text
                return StreamChunk(
                    content=state.accumulated_content,
                    delta=text if text else None,
                )

            if delta_type == "input_json_delta":
                # Accumulate tool argument JSON fragments
                idx = str(event.get("index", 0))
                if idx in state.tool_calls:
                    state.tool_calls[idx]["arguments"] += delta.get("partial_json", "")
            return None

        if event_type == "message_delta":
            stop_reason = event.get("delta", {}).get("stop_reason")
            raw_usage = event.get("usage", {})
            usage = Usage(
                prompt_tokens=0,  # Already counted in message_start
                completion_tokens=raw_usage.get("output_tokens", 0),
                total_tokens=raw_usage.get("output_tokens", 0),
            )
            return StreamChunk(
                content=state.accumulated_content,
                finish_reason=stop_reason,
                metadata={"usage": usage.model_dump()},
            )

        return None

    def finalize_stream(self, state: StreamState) -> list[ToolCall] | None:
        """Build ToolCall objects from accumulated streaming state."""
        if not state.tool_calls:
            return None
        calls = [
            ToolCall(
                call_id=tc["id"],
                name=tc["name"],
                arguments=tc.get("arguments") or "{}",
            )
            for tc in state.tool_calls.values()
        ]
        return calls if calls else None

    # ------------------------------------------------------------------
    # Model listing
    # ------------------------------------------------------------------

    def parse_models_response(self, data: dict[str, Any]) -> list[ModelSummary]:
        """Parse GET /models response from the Anthropic API."""
        return [
            ModelSummary(
                id=m["id"],
                name=m.get("display_name", m["id"]),
                created_at=None,
            )
            for m in data.get("data", [])
        ]


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_message(msg: Message) -> dict[str, Any]:
    """Convert a Message to the Anthropic wire format.

    Tool result messages use the special Anthropic content-block structure.
    """
    # Tool result: content is already a list of blocks
    if isinstance(msg.content, list):
        return {"role": msg.role, "content": msg.content}

    return {"role": msg.role, "content": msg.content or ""}


def _serialize_tool(tool: Tool) -> dict[str, Any]:
    """Convert a Tool to the Anthropic tool-use schema format."""
    schema: dict[str, Any] = {}
    if tool.parameters:
        schema = tool.parameters.model_dump(exclude_none=True)
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": schema,
    }
