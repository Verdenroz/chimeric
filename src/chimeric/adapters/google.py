"""Google Gemini REST API adapter.

Key differences from OpenAI format:
- URL contains the model name: /models/{model}:generateContent
- Auth via ?key= query param (handled by ProviderConfig + HttpClient)
- Roles: "model" instead of "assistant"
- Parts-based content structure: {"parts": [{"text": "..."}]}
- Tool calling via functionCall/functionResponse parts
- Streaming uses ?alt=sse and returns SSE with JSON payloads
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ..types import CompletionResponse, Message, ModelSummary, StreamChunk, Tool, ToolCall, Usage

if TYPE_CHECKING:
    from .base import StreamState

__all__ = ["GoogleAdapter"]

# Gemini uses "model" where OpenAI uses "assistant"
_ROLE_MAP = {"assistant": "model", "user": "user", "system": "user"}


class GoogleAdapter:
    """Adapter for the Google Gemini REST API."""

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
        """Build a Gemini generateContent POST body.

        System messages are converted to a "system_instruction" field.
        """
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system_parts.append(str(msg.content))
            else:
                contents.append(_serialize_message(msg))

        body: dict[str, Any] = {"contents": contents, **kwargs}
        if system_parts:
            body["system_instruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
        if tools:
            body["tools"] = [{"functionDeclarations": [_serialize_tool(t) for t in tools]}]
        return body

    def get_completion_path(self, model: str, stream: bool) -> str:
        """Return the model-specific endpoint path."""
        action = "streamGenerateContent" if stream else "generateContent"
        return f"/models/{model}:{action}"

    # ------------------------------------------------------------------
    # Non-streaming response parsing
    # ------------------------------------------------------------------

    def parse_response(self, data: dict[str, Any], model: str) -> CompletionResponse:
        """Parse a completed generateContent response."""
        candidate = data.get("candidates", [{}])[0]
        parts = candidate.get("content", {}).get("parts", [])

        text_parts = [p["text"] for p in parts if "text" in p]
        content = "".join(text_parts)

        raw_usage = data.get("usageMetadata", {})
        usage = Usage(
            prompt_tokens=raw_usage.get("promptTokenCount", 0),
            completion_tokens=raw_usage.get("candidatesTokenCount", 0),
            total_tokens=raw_usage.get("totalTokenCount", 0),
        )

        return CompletionResponse(
            content=content,
            usage=usage,
            model=model,
            metadata={"finish_reason": candidate.get("finishReason")},
        )

    def parse_tool_calls(self, data: dict[str, Any]) -> list[ToolCall] | None:
        """Extract functionCall parts from the Gemini response."""
        candidate = data.get("candidates", [{}])[0]
        parts = candidate.get("content", {}).get("parts", [])

        calls = [
            ToolCall(
                call_id=f"call_{i}_{p['functionCall']['name']}",
                name=p["functionCall"]["name"],
                arguments=json.dumps(p["functionCall"].get("args", {})),
            )
            for i, p in enumerate(parts)
            if "functionCall" in p
        ]
        return calls if calls else None

    def build_tool_result_messages(
        self,
        messages: list[Message],
        tool_calls: list[ToolCall],
        results: list[str],
    ) -> list[Message]:
        """Append model function-call turn and user function-response turn (immutable)."""
        # Reconstruct the model turn that issued the calls
        model_parts = [
            {"functionCall": {"name": tc.name, "args": json.loads(tc.arguments)}}
            for tc in tool_calls
        ]
        model_msg = Message(role="assistant", content=model_parts)

        # Function responses go in a single "user" message
        response_parts = [
            {
                "functionResponse": {
                    "name": tc.name,
                    "response": {"result": result},
                }
            }
            for tc, result in zip(tool_calls, results, strict=False)
        ]
        result_msg = Message(role="user", content=response_parts)

        return [*messages, model_msg, result_msg]

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def parse_sse_event(self, data: str, state: StreamState) -> StreamChunk | None:
        """Parse one Gemini SSE data payload."""
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            return None

        candidate = event.get("candidates", [{}])[0]
        parts = candidate.get("content", {}).get("parts", [])
        finish_reason: str | None = candidate.get("finishReason")

        # Handle function calls in streaming (collect for finalize_stream)
        for i, part in enumerate(parts):
            if "functionCall" in part:
                idx = str(i)
                fc = part["functionCall"]
                state.tool_calls[idx] = {
                    "id": f"call_{idx}_{fc['name']}",
                    "name": fc["name"],
                    "arguments": json.dumps(fc.get("args", {})),
                }

        # Collect text deltas
        text_parts = [p["text"] for p in parts if "text" in p]
        text = "".join(text_parts)
        state.accumulated_content += text

        usage_meta = event.get("usageMetadata", {})
        metadata: dict[str, Any] | None = None
        if usage_meta:
            usage = Usage(
                prompt_tokens=usage_meta.get("promptTokenCount", 0),
                completion_tokens=usage_meta.get("candidatesTokenCount", 0),
                total_tokens=usage_meta.get("totalTokenCount", 0),
            )
            metadata = {"usage": usage.model_dump()}

        return StreamChunk(
            content=state.accumulated_content,
            delta=text if text else None,
            finish_reason=finish_reason,
            metadata=metadata,
        )

    def finalize_stream(self, state: StreamState) -> list[ToolCall] | None:
        """Extract tool calls collected during streaming."""
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
        """Parse GET /models response from Gemini API."""
        return [
            ModelSummary(
                # Strip the "models/" prefix from Gemini model names
                id=m["name"].removeprefix("models/"),
                name=m.get("displayName", m["name"].removeprefix("models/")),
                description=m.get("description"),
            )
            for m in data.get("models", [])
        ]


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_message(msg: Message) -> dict[str, Any]:
    """Convert a Message to the Gemini contents format."""
    role = _ROLE_MAP.get(msg.role, msg.role)

    # Content is already a list of parts (tool calls/results)
    if isinstance(msg.content, list):
        return {"role": role, "parts": msg.content}

    return {"role": role, "parts": [{"text": msg.content or ""}]}


def _serialize_tool(tool: Tool) -> dict[str, Any]:
    """Convert a Tool to a Gemini functionDeclaration."""
    schema: dict[str, Any] = {}
    if tool.parameters:
        schema = tool.parameters.model_dump(exclude_none=True)
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": schema,
    }
