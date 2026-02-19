"""Core data models for the Chimeric library.

All provider interactions are expressed in terms of these types, keeping
application code independent of which provider is being used.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from enum import Enum
from typing import Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CompletionResponse",
    "Input",
    "Message",
    "Metadata",
    "ModelSummary",
    "Provider",
    "StreamChunk",
    "Tool",
    "ToolCall",
    "ToolCallChunk",
    "ToolExecutionResult",
    "ToolParameters",
    "Tools",
    "Usage",
]

# Flexible input types accepted by generate()/agenerate()
Tools: TypeAlias = Iterable[Any] | None
Metadata: TypeAlias = dict[str, Any]


###################
# MESSAGE TYPES
###################


class Message(BaseModel):
    """Standardized message format for cross-provider compatibility.

    Attributes:
        role: Sender role — "system", "user", "assistant", or "tool".
        content: Text string or list of content parts (multimodal / tool results).
        name: Optional sender identifier.
        tool_calls: Tool calls issued by the assistant in this message.
        tool_call_id: Links a "tool" role message to its originating call.
    """

    role: str
    content: str | list[Any]
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


# Accepts plain strings, dicts, Message objects, or lists thereof
Input: TypeAlias = str | dict[str, Any] | list[Any] | Message | list[Message]


###################
# PROVIDER TYPES
###################


class Provider(Enum):
    """Supported LLM providers.

    Enum values match the string keys used in PROVIDER_REGISTRY.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    CEREBRAS = "cerebras"
    COHERE = "cohere"
    GROK = "grok"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    CUSTOM = "custom"


###################
# MODEL TYPES
###################


class ModelSummary(BaseModel):
    """Lightweight description of a model returned by list_models().

    Attributes:
        id: Canonical model identifier (e.g. "gpt-4o", "claude-3-5-sonnet-20241022").
        name: Human-readable display name.
        description: Optional capability summary.
        owned_by: Entity that publishes this model.
        created_at: Unix timestamp of when the model entry was created.
        metadata: Provider-specific extras.
        provider: Name of the provider that offers this model.
    """

    id: str
    name: str
    description: str | None = None
    owned_by: str | None = None
    created_at: int | None = None
    metadata: Metadata | None = None
    provider: str | None = None

    def __str__(self) -> str:
        """Return a brief string representation."""
        return f"{self.name} ({self.provider})"


###################
# RESPONSE TYPES
###################


class Usage(BaseModel):
    """Token-usage statistics for a completion request."""

    model_config = ConfigDict(extra="allow")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResponse(BaseModel):
    """Unified response from a chat completion call.

    Attributes:
        content: Generated text (or list of content parts).
        usage: Token usage statistics.
        model: Model that produced the response.
        metadata: Provider-specific extras (finish_reason, request ID, etc.).
        parsed: Parsed Pydantic model instance when response_model is used (excluded from serialization).
    """

    content: str | list[Any] = Field(default_factory=list)
    usage: Usage | None = None
    model: str | None = None
    metadata: Metadata | None = None
    parsed: Any = Field(default=None, exclude=True)

    def __str__(self) -> str:
        """Return response text."""
        return (
            self.content
            if isinstance(self.content, str)
            else " ".join(str(c) for c in self.content if c)
        )


class ToolCall(BaseModel):
    """A single tool call issued by the model.

    Attributes:
        call_id: Provider-specific call identifier (used to match results).
        name: Name of the tool/function being called.
        arguments: JSON-encoded arguments string.
        metadata: Provider-specific extras.
    """

    call_id: str
    name: str
    arguments: str
    metadata: Metadata | None = None


class ToolCallChunk(BaseModel):
    """Streaming tool call information, built up across SSE chunks.

    Attributes:
        id: Unique identifier for this tool call.
        call_id: Provider-specific call identifier.
        name: Name of the function being called.
        arguments: Accumulated arguments JSON string.
        arguments_delta: Fragment added in the latest chunk.
        status: Lifecycle state ("started" | "arguments_streaming" | "completed").
    """

    id: str
    call_id: str | None = None
    name: str
    arguments: str = ""
    arguments_delta: str | None = None
    status: str | None = None


class StreamChunk(BaseModel):
    """A single chunk in a streaming chat response.

    Attributes:
        content: Accumulated text content up to this chunk.
        delta: Incremental text added in this chunk (None for metadata-only chunks).
        finish_reason: Why the model stopped (present only on the final chunk).
        metadata: Provider-specific extras (token counts, request IDs, etc.).
        parsed: Parsed Pydantic model instance set on the final chunk when response_model is used (excluded from serialization).
    """

    content: str | list[Any] = Field(default_factory=list)
    delta: str | None = None
    finish_reason: str | None = None
    metadata: Metadata | None = None
    parsed: Any = Field(default=None, exclude=True)

    def __str__(self) -> str:
        """Return the incremental delta, or empty string."""
        return self.delta if self.delta is not None else ""


###################
# TOOL TYPES
###################


class ToolParameters(BaseModel):
    """JSON Schema definition for tool parameters."""

    model_config = ConfigDict(extra="allow")

    type: str = "object"
    strict: bool = True
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] | None = None
    additionalProperties: bool = False

    def model_dump(self, exclude_none: bool = True, **kwargs: Any) -> dict[str, Any]:
        """Return dictionary representation, excluding None values by default."""
        return super().model_dump(exclude_none=exclude_none, **kwargs)


class Tool(BaseModel):
    """A callable function exposed to the model via function-calling.

    Attributes:
        name: Unique identifier used by the model to invoke this tool.
        description: What the tool does and when to use it.
        parameters: JSON Schema for the arguments it accepts.
        function: Python callable implementing the tool (excluded from serialization).
    """

    name: str
    description: str
    parameters: ToolParameters | None = None
    function: Callable[..., Any] | None = Field(default=None, exclude=True)

    def __str__(self) -> str:
        """Return a brief string representation."""
        return f"Tool({self.name}: {self.description[:50]}...)"


class ToolExecutionResult(BaseModel):
    """Outcome of executing a single tool call.

    Attributes:
        call_id: ID of the originating tool call.
        name: Tool that was called.
        arguments: Arguments that were passed.
        result: Return value, if successful.
        error: Error message, if execution failed.
        is_error: True when execution raised an exception.
    """

    call_id: str
    name: str
    arguments: str

    result: str | None = None
    error: str | None = None
    is_error: bool = False
