"""Adapter protocol and shared streaming state.

Each adapter is a stateless object implementing this protocol.  All methods
are pure (no I/O); the HTTP client in http.py handles transport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..types import (
        CompletionResponse,
        EmbeddingResponse,
        Message,
        ModelSummary,
        StreamChunk,
        Tool,
        ToolCall,
    )

__all__ = ["Adapter", "EmbeddingAdapter", "StreamState"]


@dataclass
class StreamState:
    """Mutable accumulator for one streaming response.

    Adapters receive this object across successive SSE events so they can
    build up tool-call argument strings without allocating new objects.

    Attributes:
        accumulated_content: Full text content received so far.
        tool_calls: In-progress tool calls keyed by their index or ID.
                    Each value is a partial dict; finalized by finalize_stream().
    """

    accumulated_content: str = ""
    tool_calls: dict[str, dict[str, Any]] = field(default_factory=dict)


@runtime_checkable
class Adapter(Protocol):
    """Contract every adapter must satisfy.

    Implementations live in adapter-specific modules.
    Methods are called by HttpClient in order:

        build_request_body  → POST body
        parse_response      → CompletionResponse  (non-streaming)
        parse_tool_calls    → tool calls to execute (non-streaming)
        parse_sse_event     → StreamChunk per SSE line (streaming)
        finalize_stream     → tool calls after stream ends (streaming)
        build_tool_result_messages → append results before next turn
        parse_models_response → list of ModelSummary objects
    """

    def build_request_body(
        self,
        messages: list[Message],
        model: str,
        stream: bool,
        tools: list[Tool] | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Serialize messages + parameters into the provider's POST body."""
        ...

    def parse_response(self, data: dict[str, Any], model: str) -> CompletionResponse:
        """Build a CompletionResponse from a completed (non-streaming) API response."""
        ...

    def parse_tool_calls(self, data: dict[str, Any]) -> list[ToolCall] | None:
        """Extract tool calls from a non-streaming API response, or None."""
        ...

    def build_tool_result_messages(
        self,
        messages: list[Message],
        tool_calls: list[ToolCall],
        results: list[str],
    ) -> list[Message]:
        """Return a new message list with tool results appended.

        Must never mutate the input list.
        """
        ...

    def parse_sse_event(self, data: str, state: StreamState) -> StreamChunk | None:
        """Parse one SSE data payload, update state, and return a StreamChunk.

        Returns None for events that carry no displayable content (e.g. role
        markers, ping events).
        """
        ...

    def finalize_stream(self, state: StreamState) -> list[ToolCall] | None:
        """Extract completed tool calls from streaming state after the stream ends."""
        ...

    def parse_models_response(self, data: dict[str, Any]) -> list[ModelSummary]:
        """Parse the provider's model-listing response into ModelSummary objects."""
        ...


@runtime_checkable
class EmbeddingAdapter(Protocol):
    """Contract for adapters that support embedding requests.

    Separate from :class:`Adapter` so that providers without embedding
    support (Anthropic, Google, Cohere) need no stub methods.
    """

    def build_embedding_request(
        self,
        input: str | list[str],
        model: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Serialize embedding input into the provider's POST body."""
        ...

    def parse_embedding_response(
        self,
        data: dict[str, Any],
        model: str,
        *,
        batch: bool,
    ) -> EmbeddingResponse:
        """Parse the provider's embedding API response.

        Args:
            data: Raw JSON response from the provider.
            model: Model identifier used in the request.
            batch: True when the caller passed a list of strings, False for a
                   single string.  Controls whether ``embedding`` or
                   ``embeddings`` is populated on the result.
        """
        ...
