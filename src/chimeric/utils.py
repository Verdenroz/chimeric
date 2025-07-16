from typing import Any

from pydantic import BaseModel, Field

from .types import (
    ChimericCompletionResponse,
    ChimericStreamChunk,
    CompletionResponse,
    Input,
    Message,
    StreamChunk,
    Tool,
    ToolCallChunk,
    Tools,
    Usage,
)


class StreamState(BaseModel):
    """Maintains state during streaming.

    Attributes:
        accumulated_content: The accumulated text content so far.
        tool_calls: Dictionary of tool calls being streamed, keyed by call ID.
        metadata: Additional metadata accumulated during streaming.
    """

    accumulated_content: str = ""
    tool_calls: dict[str, ToolCallChunk] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StreamProcessor:
    """Standardized stream processing logic for all providers."""

    def __init__(self):
        self.state = StreamState()

    def process_content_delta(self, delta: str) -> StreamChunk:
        """Processes a text content delta."""
        self.state.accumulated_content += delta
        return StreamChunk(
            content=self.state.accumulated_content,
            delta=delta,
        )

    def process_tool_call_start(self, call_id: str, name: str, provider_call_id: str | None = None) -> None:
        """Processes the start of a tool call.
        
        Args:
            call_id: Primary identifier for tracking this tool call (usually provider's id field)
            name: Name of the function being called
            provider_call_id: Provider-specific call ID for outputs (optional, defaults to call_id)
        """
        self.state.tool_calls[call_id] = ToolCallChunk(
            id=call_id,
            call_id=provider_call_id or call_id,
            name=name,
            arguments="",
            status="started",
        )

    def process_tool_call_delta(self, call_id: str, arguments_delta: str) -> None:
        """Processes a tool call arguments delta."""
        if call_id in self.state.tool_calls:
            self.state.tool_calls[call_id].arguments += arguments_delta
            self.state.tool_calls[call_id].arguments_delta = arguments_delta
            self.state.tool_calls[call_id].status = "arguments_streaming"

    def process_tool_call_complete(self, call_id: str) -> None:
        """Marks a tool call as complete."""
        if call_id in self.state.tool_calls:
            self.state.tool_calls[call_id].status = "completed"
            self.state.tool_calls[call_id].arguments_delta = None

    def get_completed_tool_calls(self) -> list[ToolCallChunk]:
        """Returns all completed tool calls."""
        return [tc for tc in self.state.tool_calls.values() if tc.status == "completed"]


def normalize_messages(messages: Input) -> list[Message]:
    """Converts various input formats to standardized Message objects."""
    if isinstance(messages, str):
        return [Message(role="user", content=messages)]

    if isinstance(messages, Message):
        return [messages]

    if isinstance(messages, dict):
        return [Message(**messages)]

    if isinstance(messages, list):
        normalized = []
        for msg in messages:
            if isinstance(msg, str):
                normalized.append(Message(role="user", content=msg))
            elif isinstance(msg, Message):
                normalized.append(msg)
            elif isinstance(msg, dict):
                normalized.append(Message(**msg))
            else:
                normalized.append(Message(role="user", content=str(msg)))
        return normalized

    return [Message(role="user", content=str(messages))]


def normalize_tools(tools: Tools) -> list[Tool]:
    """Converts various tool formats to standardized Tool objects."""
    if not tools:
        return []

    normalized = []
    for tool in tools:
        if isinstance(tool, Tool):
            normalized.append(tool)
        elif isinstance(tool, dict):
            # Convert dict to a Tool object
            normalized.append(Tool(**tool))
        else:
            # Try to extract from object attributes
            normalized.append(
                Tool(
                    name=getattr(tool, "name", "unknown"),
                    description=getattr(tool, "description", ""),
                    parameters=getattr(tool, "parameters", {}),
                    function=getattr(tool, "function", None),
                )
            )
    return normalized


def create_completion_response(
    native_response: Any,
    content: str | list[Any],
    usage: Usage | None = None,
    model: str | None = None,
    tool_calls: list[Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Creates a standardized completion response."""
    response_metadata = metadata or {}
    if tool_calls:
        response_metadata["tool_calls"] = [tc.model_dump() for tc in tool_calls]

    return ChimericCompletionResponse(
        native=native_response,
        common=CompletionResponse(
            content=content, usage=usage or Usage(), model=model, metadata=response_metadata
        ),
    )


def create_stream_chunk(
    native_event: Any,
    processor: StreamProcessor,
    content_delta: str | None = None,
    finish_reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Creates a standardized stream chunk."""
    if content_delta is not None:
        chunk = processor.process_content_delta(content_delta)
    else:
        chunk = StreamChunk(
            content=processor.state.accumulated_content,
            finish_reason=finish_reason,
            metadata=metadata or processor.state.metadata,
        )

    return ChimericStreamChunk(native=native_event, common=chunk)


