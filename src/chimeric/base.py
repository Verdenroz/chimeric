from abc import ABC, abstractmethod
import asyncio
from collections.abc import AsyncGenerator, Generator
import concurrent.futures
import contextlib
from datetime import datetime
import inspect
import json
import time
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from .exceptions import (
    ChimericError,
    ProviderError,
    ToolRegistrationError,
)
from .tools import ToolManager
from .types import (
    Capability,
    ChimericCompletionResponse,
    ChimericFileUploadResponse,
    ChimericStreamChunk,
    CompletionResponse,
    Input,
    Message,
    ModelSummary,
    StreamChunk,
    Tool,
    ToolCall,
    ToolCallChunk,
    ToolExecutionResult,
    Tools,
    Usage,
)

__all__ = [
    "ChimericAsyncClient",
    "ChimericClient",
    "CompletionResponseType",
    "FileUploadResponseType",
    "StreamProcessor",
    "StreamType",
]


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

    def process_tool_call_start(self, call_id: str, name: str) -> None:
        """Processes the start of a tool call."""
        self.state.tool_calls[call_id] = ToolCallChunk(
            id=call_id,
            call_id=call_id,
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


# Type variables for provider-specific types
ClientType = TypeVar("ClientType")
CompletionResponseType = TypeVar("CompletionResponseType")
StreamType = TypeVar("StreamType")
FileUploadResponseType = TypeVar("FileUploadResponseType")


class ChimericClient(
    ABC,
    Generic[
        ClientType,
        CompletionResponseType,
        StreamType,
        FileUploadResponseType,
    ],
):
    """Abstract base class for synchronous LLM provider clients.

    This class provides a unified interface and common functionality for all
    provider implementations, including message normalization, tool execution,
    and response standardization.
    """

    def __init__(
        self,
        api_key: str,
        tool_manager: ToolManager,
        **kwargs: Any,
    ) -> None:
        """Initializes the base client with common settings."""
        self.api_key = api_key
        self.tool_manager = tool_manager
        self.created_at = datetime.now()
        self._request_count = 0
        self._last_request_time: float | None = None
        self._error_count = 0

        # Get client type and filter kwargs
        client_type = self._get_client_type()
        sync_kwargs = self._filter_client_kwargs(client_type, kwargs)

        # Initialize client
        self._client: ClientType = self._init_client(client_type, **sync_kwargs)
        self._capabilities = self._get_capabilities()

    # ====================================================================
    # Abstract methods - Required for all providers
    # ====================================================================

    @abstractmethod
    def _get_client_type(self) -> type:
        """Returns the actual client type used by the provider.

        Example:
            return openai.Client
        """
        pass

    @abstractmethod
    def _init_client(self, client_type: type, **kwargs: Any) -> ClientType:
        """Initializes the provider's synchronous client."""
        pass

    @abstractmethod
    def _get_capabilities(self) -> Capability:
        """Returns the capabilities supported by this provider."""
        pass

    @abstractmethod
    def _list_models_impl(self) -> list[ModelSummary]:
        """Lists available models from the provider's API."""
        pass

    @abstractmethod
    def _messages_to_provider_format(self, messages: list[Message]) -> Any:
        """Converts standardized messages to provider-specific format."""
        pass

    @abstractmethod
    def _tools_to_provider_format(self, tools: list[Tool]) -> Any:
        """Converts standardized tools to provider-specific format."""
        pass

    @abstractmethod
    def _make_provider_request(
        self,
        messages: Any,
        model: str,
        stream: bool,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Makes the actual API request to the provider."""
        pass

    @abstractmethod
    def _process_provider_stream_event(
        self, event: Any, processor: StreamProcessor
    ) -> ChimericStreamChunk | None:
        """Processes a provider-specific stream event.

        Providers should use _create_stream_chunk() to create standardized chunks.

        Example:
            # For content delta
            if hasattr(event, 'delta'):
                return self._create_stream_chunk(
                    native_event=event,
                    processor=processor,
                    content_delta=event.delta
                )

            # For finish event
            if event.finish_reason:
                return self._create_stream_chunk(
                    native_event=event,
                    processor=processor,
                    finish_reason=event.finish_reason
                )
        """
        pass

    @abstractmethod
    def _extract_usage_from_response(self, response: Any) -> Usage:
        """Extracts usage information from provider response."""
        pass

    @abstractmethod
    def _extract_content_from_response(self, response: Any) -> str | list[Any]:
        """Extracts content from provider response."""
        pass

    @abstractmethod
    def _extract_tool_calls_from_response(self, response: Any) -> list[ToolCall] | None:
        """Extracts tool calls from provider response."""
        pass

    def _process_provider_stream(
        self, stream: Any, processor: StreamProcessor
    ) -> Generator[ChimericStreamChunk[StreamType], None, None]:
        """Processes a provider stream using the processor."""
        for event in stream:
            chunk = self._process_provider_stream_event(event, processor)
            if chunk:
                yield chunk

        # Execute any accumulated tool calls
        if processor.state.tool_calls:
            tool_results = self._execute_accumulated_tool_calls(processor.state.tool_calls)
            if tool_results:
                # Yield final chunk with tool results
                final_chunk = StreamChunk(
                    content=processor.state.accumulated_content,
                    finish_reason="tool_calls",
                    metadata={"tool_results": [tr.model_dump() for tr in tool_results]},
                )
                yield ChimericStreamChunk(native=None, common=final_chunk)

    # ====================================================================
    # Optional methods - Override based on capabilities
    # ====================================================================

    @staticmethod
    def _get_model_aliases() -> list[str]:
        """Return model aliases to include in model listings."""
        return []

    def _upload_file(self, **kwargs: Any) -> ChimericFileUploadResponse[FileUploadResponseType]:
        """Provider-specific file upload implementation.

        Only implement if supports_files() returns True.
        """
        raise NotImplementedError(
            f"Provider {self.__class__.__name__} does not support file uploads"
        )

    # ====================================================================
    # Message and tool normalization
    # ====================================================================

    @staticmethod
    def _normalize_messages(messages: Input) -> list[Message]:
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

    @staticmethod
    def _normalize_tools(tools: Tools) -> list[Tool]:
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

    # ====================================================================
    # Tool execution
    # ====================================================================

    def _execute_tool_call(self, call: ToolCall) -> ToolExecutionResult:
        """Executes a single tool call with standardized error handling."""
        result = ToolExecutionResult(call_id=call.call_id, name=call.name, arguments=call.arguments)

        try:
            tool = self.tool_manager.get_tool(call.name)
            if not tool or not callable(tool.function):
                raise ToolRegistrationError(f"Tool '{call.name}' is not callable")

            args = json.loads(call.arguments) if call.arguments else {}
            execution_result = tool.function(**args)
            result.result = str(execution_result)

        except json.JSONDecodeError as e:
            result.error = f"Invalid JSON arguments: {e}"
            result.is_error = True
        except ToolRegistrationError as e:
            result.error = str(e)
            result.is_error = True
        except Exception as e:
            result.error = f"Tool execution failed: {e}"
            result.is_error = True

        return result

    def _execute_tool_calls(self, calls: list[ToolCall]) -> list[ToolExecutionResult]:
        """Executes multiple tool calls in parallel."""
        if not calls:
            return []

        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_call = {
                executor.submit(self._execute_tool_call, call): call for call in calls
            }
            for future in concurrent.futures.as_completed(future_to_call):
                call = future_to_call[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(
                        ToolExecutionResult(
                            call_id=call.call_id,
                            name=call.name,
                            arguments=call.arguments,
                            error=str(e),
                            is_error=True,
                        )
                    )
        return results

    def _execute_accumulated_tool_calls(
        self, tool_calls: dict[str, ToolCallChunk]
    ) -> list[ToolExecutionResult]:
        """Executes accumulated tool calls from streaming.

        Converts ToolCallChunk objects to ToolCall and executes them.
        """
        calls = []
        for tool_call in tool_calls.values():
            if tool_call.status == "completed":
                calls.append(
                    ToolCall(
                        call_id=tool_call.call_id or tool_call.id,
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                    )
                )

        return self._execute_tool_calls(calls)

    # ====================================================================
    # Response creation helpers
    # ====================================================================

    @staticmethod
    def _create_completion_response(
        native_response: CompletionResponseType,
        content: str | list[Any],
        usage: Usage | None = None,
        model: str | None = None,
        tool_calls: list[ToolExecutionResult] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChimericCompletionResponse[CompletionResponseType]:
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

    @staticmethod
    def _create_stream_chunk(
        native_event: StreamType,
        processor: StreamProcessor,
        content_delta: str | None = None,
        finish_reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChimericStreamChunk[StreamType]:
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

    # ====================================================================
    # Public API
    # ====================================================================

    def chat_completion(
        self,
        messages: Input,
        model: str,
        stream: bool = False,
        tools: Tools = None,
        auto_tool: bool = True,
        **kwargs: Any,
    ) -> (
        ChimericCompletionResponse[CompletionResponseType]
        | Generator[ChimericStreamChunk[StreamType], None, None]
    ):
        """Generates a synchronous chat completion.

        Args:
            messages: Input messages (string, list, or Message objects)
            model: Model identifier to use
            stream: Whether to stream the response
            tools: Optional list of tools to make available
            auto_tool: If True, automatically includes all registered tools
            **kwargs: Provider-specific parameters

        Returns:
            ChimericCompletionResponse or Generator of ChimericStreamChunk

        Raises:
            ChimericError: If requested capability is not supported
            ProviderError: If the provider API returns an error
        """
        if stream and not self.supports_streaming():
            raise ChimericError("This provider does not support streaming responses")

        try:
            self._request_count += 1
            self._last_request_time = time.time()

            # Process tools
            final_tools = tools
            if not final_tools and auto_tool and self.supports_tools():
                final_tools = self.tool_manager.get_all_tools()

            if final_tools and not self.supports_tools():
                raise ChimericError("This provider does not support tool calling")

            # Normalize inputs
            normalized_messages = self._normalize_messages(messages)
            normalized_tools = self._normalize_tools(final_tools) if final_tools else None

            # Convert to provider format
            provider_messages = self._messages_to_provider_format(normalized_messages)
            provider_tools = (
                self._tools_to_provider_format(normalized_tools) if normalized_tools else None
            )

            # Make API call
            response = self._make_provider_request(
                provider_messages, model, stream, provider_tools, **kwargs
            )

            if stream:
                # Create StreamProcessor for streaming responses
                stream_processor = StreamProcessor()
                return self._process_provider_stream(response, stream_processor)
            # Extract standard components
            content = self._extract_content_from_response(response)
            usage = self._extract_usage_from_response(response)
            tool_calls = self._extract_tool_calls_from_response(response) if final_tools else None

            # Execute tool calls if any
            tool_results = None
            if tool_calls:
                tool_results = self._execute_tool_calls(tool_calls)

            return self._create_completion_response(
                native_response=response,
                content=content,
                usage=usage,
                model=model,
                tool_calls=tool_results,
            )
        except Exception:
            self._error_count += 1
            raise

    def upload_file(self, **kwargs: Any) -> ChimericFileUploadResponse[FileUploadResponseType]:
        """Uploads a file to the provider.

        Args:
            **kwargs: Provider-specific file upload parameters

        Returns:
            ChimericFileUploadResponse with native and common formats

        Raises:
            NotImplementedError: If provider doesn't support files
            ProviderError: If upload fails
        """
        if not self.supports_files():
            raise NotImplementedError("Provider does not support file uploads")

        try:
            self._request_count += 1
            self._last_request_time = time.time()
            return self._upload_file(**kwargs)
        except Exception as e:
            self._error_count += 1
            if isinstance(e, (ProviderError, ValueError, NotImplementedError)):
                raise

            provider_name = getattr(self, "_provider_name", self.__class__.__name__)
            raise ProviderError(
                provider=provider_name,
                response_text=f"Failed to upload file: {e!s}",
                endpoint="file_upload",
                status_code=getattr(e, "status_code", None),
            ) from e

    def list_models(self) -> list[ModelSummary]:
        """Lists all available models including aliases."""
        api_models = self._list_models_impl()

        # Add aliases
        aliases = self._get_model_aliases()
        alias_models = [ModelSummary(id=alias, name=alias) for alias in aliases]

        return api_models + alias_models

    def get_model_info(self, model_id: str) -> ModelSummary:
        """Gets information about a specific model."""
        models = self.list_models()
        for model in models:
            if model.id == model_id or model.name == model_id:
                return model
        raise ValueError(f"Model {model_id} not found")

    # ====================================================================
    # Capability checks
    # ====================================================================

    def supports_multimodal(self) -> bool:
        """Whether the provider supports multimodal inputs."""
        return self._capabilities.multimodal

    def supports_tools(self) -> bool:
        """Whether the provider supports tool/function calling."""
        return self._capabilities.tools

    def supports_agents(self) -> bool:
        """Whether the provider supports agent workflows."""
        return self._capabilities.agents

    def supports_files(self) -> bool:
        """Whether the provider supports file uploads."""
        return self._capabilities.files

    def supports_streaming(self) -> bool:
        """Whether the provider supports streaming responses."""
        return self._capabilities.streaming

    # ====================================================================
    # Properties
    # ====================================================================

    @property
    def capabilities(self) -> Capability:
        """All capabilities of this provider."""
        return self._capabilities

    @property
    def client(self) -> ClientType:
        """The underlying synchronous client."""
        return self._client

    @property
    def request_count(self) -> int:
        """Total number of API requests made."""
        return self._request_count

    @property
    def error_count(self) -> int:
        """Total number of errors encountered."""
        return self._error_count

    @property
    def last_request_time(self) -> float | None:
        """Unix timestamp of the last request."""
        return self._last_request_time

    @property
    def client_age(self) -> float:
        """Age of this client in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

    # ====================================================================
    # Utility methods
    # ====================================================================

    @staticmethod
    def _filter_client_kwargs(client_type: type, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Filters kwargs to only include valid parameters for a client type."""
        if client_type is object:
            return kwargs.copy()

        try:
            init_method = getattr(client_type, "__init__", None)
            if init_method is None:
                return kwargs.copy()

            sig = inspect.signature(init_method)
            param_names = {
                name
                for name, param in sig.parameters.items()
                if name != "self" and param.kind != inspect.Parameter.POSITIONAL_ONLY
            }

            has_var_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
            )

            if has_var_kwargs:
                return kwargs.copy()

            return {k: v for k, v in kwargs.items() if k in param_names}

        except (ValueError, TypeError, AttributeError):
            return kwargs.copy()

    @staticmethod
    def _filter_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Filters kwargs to only include valid parameters for a function."""
        try:
            sig = inspect.signature(func)
            param_names = {
                name
                for name, param in sig.parameters.items()
                if param.kind != inspect.Parameter.POSITIONAL_ONLY
            }

            has_var_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
            )

            if has_var_kwargs:
                return kwargs.copy()

            return {k: v for k, v in kwargs.items() if k in param_names}
        except (ValueError, TypeError, AttributeError):
            return {}

    # ====================================================================
    # Context managers
    # ====================================================================

    def __enter__(self):
        """Enters the context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exits the context manager, closing the client."""
        if hasattr(self._client, "close"):
            self._client.close()

    # ====================================================================
    # String representations
    # ====================================================================

    def __repr__(self) -> str:
        """Returns a detailed string representation."""
        return (
            f"{self.__class__.__name__}("
            f"capabilities={self._capabilities}, "
            f"requests={self._request_count}, "
            f"errors={self._error_count})"
        )

    def __str__(self) -> str:
        """Returns a human-readable string representation."""
        return (
            f"{self.__class__.__name__} Client\n"
            f"- Created: {self.created_at}\n"
            f"- Requests: {self._request_count}\n"
            f"- Errors: {self._error_count}\n"
            f"- Capabilities: {self._capabilities}"
        )


class ChimericAsyncClient(
    ABC,
    Generic[
        ClientType,
        CompletionResponseType,
        StreamType,
        FileUploadResponseType,
    ],
):
    """Abstract base class for asynchronous LLM provider clients.

    This class provides a unified interface and common functionality for all
    provider implementations, including message normalization, tool execution,
    and response standardization.
    """

    def __init__(
        self,
        api_key: str,
        tool_manager: ToolManager,
        **kwargs: Any,
    ) -> None:
        """Initializes the base async client with common settings."""
        self.api_key = api_key
        self.tool_manager = tool_manager
        self.created_at = datetime.now()
        self._request_count = 0
        self._last_request_time: float | None = None
        self._error_count = 0

        # Get client type and filter kwargs
        client_type = self._get_async_client_type()
        async_kwargs = self._filter_client_kwargs(client_type, kwargs)

        # Initialize client
        self._async_client: ClientType = self._init_async_client(client_type, **async_kwargs)
        self._capabilities = self._get_capabilities()

    # ====================================================================
    # Abstract methods - Required for all providers
    # ====================================================================

    @abstractmethod
    def _get_async_client_type(self) -> type:
        """Returns the actual async client type used by the provider.

        Example:
            return openai.AsyncClient
        """
        pass

    @abstractmethod
    def _init_async_client(self, async_client_type: type, **kwargs: Any) -> ClientType:
        """Initializes the provider's asynchronous client."""
        pass

    @abstractmethod
    def _get_capabilities(self) -> Capability:
        """Returns the capabilities supported by this provider."""
        pass

    @abstractmethod
    def _list_models_impl(self) -> list[ModelSummary]:
        """Lists available models from the provider's API."""
        pass

    @abstractmethod
    def _messages_to_provider_format(self, messages: list[Message]) -> Any:
        """Converts standardized messages to provider-specific format."""
        pass

    @abstractmethod
    def _tools_to_provider_format(self, tools: list[Tool]) -> Any:
        """Converts standardized tools to provider-specific format."""
        pass

    @abstractmethod
    async def _make_async_provider_request(
        self,
        messages: Any,
        model: str,
        stream: bool,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Makes the actual async API request to the provider."""
        pass

    @abstractmethod
    def _process_provider_stream_event(
        self, event: Any, processor: StreamProcessor
    ) -> ChimericStreamChunk | None:
        """Processes a provider-specific stream event.

        Providers should use _create_stream_chunk() to create standardized chunks.

        Example:
            # For content delta
            if hasattr(event, 'delta'):
                return self._create_stream_chunk(
                    native_event=event,
                    processor=processor,
                    content_delta=event.delta
                )

            # For finish event
            if event.finish_reason:
                return self._create_stream_chunk(
                    native_event=event,
                    processor=processor,
                    finish_reason=event.finish_reason
                )
        """
        pass

    @abstractmethod
    def _extract_usage_from_response(self, response: Any) -> Usage:
        """Extracts usage information from provider response."""
        pass

    @abstractmethod
    def _extract_content_from_response(self, response: Any) -> str | list[Any]:
        """Extracts content from provider response."""
        pass

    @abstractmethod
    def _extract_tool_calls_from_response(self, response: Any) -> list[ToolCall] | None:
        """Extracts tool calls from provider response."""
        pass

    async def _process_async_provider_stream(
        self, stream: Any, processor: StreamProcessor
    ) -> AsyncGenerator[ChimericStreamChunk[StreamType], None]:
        """Processes an async provider stream using the processor."""
        async for event in stream:
            chunk = self._process_provider_stream_event(event, processor)
            if chunk:
                yield chunk

        # Execute any accumulated tool calls
        if processor.state.tool_calls:
            tool_results = await self._execute_accumulated_tool_calls(processor.state.tool_calls)
            if tool_results:
                # Yield final chunk with tool results
                final_chunk = StreamChunk(
                    content=processor.state.accumulated_content,
                    finish_reason="tool_calls",
                    metadata={"tool_results": [tr.model_dump() for tr in tool_results]},
                )
                yield ChimericStreamChunk(native=None, common=final_chunk)

    # ====================================================================
    # Optional methods - Override based on capabilities
    # ====================================================================

    @staticmethod
    def _get_model_aliases() -> list[str]:
        """Return model aliases to include in model listings."""
        return []

    def _upload_file(self, **kwargs: Any) -> ChimericFileUploadResponse[FileUploadResponseType]:
        """Provider-specific file upload implementation.

        Only implement if supports_files() returns True.
        """
        raise NotImplementedError(
            f"Provider {self.__class__.__name__} does not support file uploads"
        )

    # ====================================================================
    # Message and tool normalization
    # ====================================================================

    @staticmethod
    def _normalize_messages(messages: Input) -> list[Message]:
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

    @staticmethod
    def _normalize_tools(tools: Tools) -> list[Tool]:
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

    # ====================================================================
    # Tool execution
    # ====================================================================

    async def _execute_tool_call(self, call: ToolCall) -> ToolExecutionResult:
        """Executes a single tool call with standardized error handling."""
        result = ToolExecutionResult(call_id=call.call_id, name=call.name, arguments=call.arguments)

        try:
            tool = self.tool_manager.get_tool(call.name)
            if not tool or not callable(tool.function):
                raise ToolRegistrationError(f"Tool '{call.name}' is not callable")

            args = json.loads(call.arguments) if call.arguments else {}

            # Check if the tool function is async
            if inspect.iscoroutinefunction(tool.function):
                execution_result = await tool.function(**args)
            else:
                execution_result = tool.function(**args)

            result.result = str(execution_result)

        except json.JSONDecodeError as e:
            result.error = f"Invalid JSON arguments: {e}"
            result.is_error = True
        except ToolRegistrationError as e:
            result.error = str(e)
            result.is_error = True
        except Exception as e:
            result.error = f"Tool execution failed: {e}"
            result.is_error = True

        return result

    async def _execute_tool_calls(self, calls: list[ToolCall]) -> list[ToolExecutionResult]:
        """Executes multiple tool calls in parallel using asyncio."""
        if not calls:
            return []

        # Create tasks for all tool calls
        tasks = [self._execute_tool_call(call) for call in calls]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(
                    ToolExecutionResult(
                        call_id=calls[i].call_id,
                        name=calls[i].name,
                        arguments=calls[i].arguments,
                        error=str(result),
                        is_error=True,
                    )
                )
            else:
                final_results.append(result)

        return final_results

    async def _execute_accumulated_tool_calls(
        self, tool_calls: dict[str, ToolCallChunk]
    ) -> list[ToolExecutionResult]:
        """Executes accumulated tool calls from streaming.

        Converts ToolCallChunk objects to ToolCall and executes them.
        """
        calls = []
        for tool_call in tool_calls.values():
            if tool_call.status == "completed":
                calls.append(
                    ToolCall(
                        call_id=tool_call.call_id or tool_call.id,
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                    )
                )

        return await self._execute_tool_calls(calls)

    # ====================================================================
    # Response creation helpers
    # ====================================================================

    @staticmethod
    def _create_completion_response(
        native_response: CompletionResponseType,
        content: str | list[Any],
        usage: Usage | None = None,
        model: str | None = None,
        tool_calls: list[ToolExecutionResult] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChimericCompletionResponse[CompletionResponseType]:
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

    @staticmethod
    def _create_stream_chunk(
        native_event: StreamType,
        processor: StreamProcessor,
        content_delta: str | None = None,
        finish_reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChimericStreamChunk[StreamType]:
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

    # ====================================================================
    # Public API
    # ====================================================================

    async def chat_completion(
        self,
        messages: Input,
        model: str,
        stream: bool = False,
        tools: Tools = None,
        auto_tool: bool = True,
        **kwargs: Any,
    ) -> (
        ChimericCompletionResponse[CompletionResponseType]
        | AsyncGenerator[ChimericStreamChunk[StreamType], None]
    ):
        """Generates an asynchronous chat completion.

        Args:
            messages: Input messages (string, list, or Message objects)
            model: Model identifier to use
            stream: Whether to stream the response
            tools: Optional list of tools to make available
            auto_tool: If True, automatically includes all registered tools
            **kwargs: Provider-specific parameters

        Returns:
            ChimericCompletionResponse or AsyncGenerator of ChimericStreamChunk

        Raises:
            ChimericError: If requested capability is not supported
            ProviderError: If the provider API returns an error
        """
        if stream and not self.supports_streaming():
            raise ChimericError("This provider does not support streaming responses")

        try:
            self._request_count += 1
            self._last_request_time = time.time()

            # Process tools
            final_tools = tools
            if not final_tools and auto_tool and self.supports_tools():
                final_tools = self.tool_manager.get_all_tools()

            if final_tools and not self.supports_tools():
                raise ChimericError("This provider does not support tool calling")

            # Normalize inputs
            normalized_messages = self._normalize_messages(messages)
            normalized_tools = self._normalize_tools(final_tools) if final_tools else None

            # Convert to provider format
            provider_messages = self._messages_to_provider_format(normalized_messages)
            provider_tools = (
                self._tools_to_provider_format(normalized_tools) if normalized_tools else None
            )

            # Make API call
            response = await self._make_async_provider_request(
                provider_messages, model, stream, provider_tools, **kwargs
            )

            if stream:
                # Create StreamProcessor for streaming responses
                stream_processor = StreamProcessor()
                return self._process_async_provider_stream(response, stream_processor)
            # Extract standard components
            content = self._extract_content_from_response(response)
            usage = self._extract_usage_from_response(response)
            tool_calls = self._extract_tool_calls_from_response(response) if final_tools else None

            # Execute tool calls if any
            tool_results = None
            if tool_calls:
                tool_results = self._execute_tool_calls(tool_calls)

            return self._create_completion_response(
                native_response=response,
                content=content,
                usage=usage,
                model=model,
                tool_calls=tool_results,
            )
        except Exception:
            self._error_count += 1
            raise

    async def upload_file(
        self, **kwargs: Any
    ) -> ChimericFileUploadResponse[FileUploadResponseType]:
        """Uploads a file to the provider.

        Args:
            **kwargs: Provider-specific file upload parameters

        Returns:
            ChimericFileUploadResponse with native and common formats

        Raises:
            NotImplementedError: If provider doesn't support files
            ProviderError: If upload fails
        """
        if not self.supports_files():
            raise NotImplementedError("Provider does not support file uploads")

        try:
            self._request_count += 1
            self._last_request_time = time.time()
            return self._upload_file(**kwargs)
        except Exception as e:
            self._error_count += 1
            if isinstance(e, (ProviderError, ValueError, NotImplementedError)):
                raise

            provider_name = getattr(self, "_provider_name", self.__class__.__name__)
            raise ProviderError(
                provider=provider_name,
                response_text=f"Failed to upload file: {e!s}",
                endpoint="file_upload",
                status_code=getattr(e, "status_code", None),
            ) from e

    def list_models(self) -> list[ModelSummary]:
        """Lists all available models including aliases."""
        api_models = self._list_models_impl()

        # Add aliases
        aliases = self._get_model_aliases()
        alias_models = [ModelSummary(id=alias, name=alias) for alias in aliases]

        return api_models + alias_models

    def get_model_info(self, model_id: str) -> ModelSummary:
        """Gets information about a specific model."""
        models = self.list_models()
        for model in models:
            if model.id == model_id or model.name == model_id:
                return model
        raise ValueError(f"Model {model_id} not found")

    # ====================================================================
    # Capability checks
    # ====================================================================

    def supports_multimodal(self) -> bool:
        """Whether the provider supports multimodal inputs."""
        return self._capabilities.multimodal

    def supports_tools(self) -> bool:
        """Whether the provider supports tool/function calling."""
        return self._capabilities.tools

    def supports_agents(self) -> bool:
        """Whether the provider supports agent workflows."""
        return self._capabilities.agents

    def supports_files(self) -> bool:
        """Whether the provider supports file uploads."""
        return self._capabilities.files

    def supports_streaming(self) -> bool:
        """Whether the provider supports streaming responses."""
        return self._capabilities.streaming

    # ====================================================================
    # Properties
    # ====================================================================

    @property
    def capabilities(self) -> Capability:
        """All capabilities of this provider."""
        return self._capabilities

    @property
    def async_client(self) -> ClientType:
        """The underlying asynchronous client."""
        return self._async_client

    @property
    def request_count(self) -> int:
        """Total number of API requests made."""
        return self._request_count

    @property
    def error_count(self) -> int:
        """Total number of errors encountered."""
        return self._error_count

    @property
    def last_request_time(self) -> float | None:
        """Unix timestamp of the last request."""
        return self._last_request_time

    @property
    def client_age(self) -> float:
        """Age of this client in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

    # ====================================================================
    # Utility methods
    # ====================================================================

    @staticmethod
    def _filter_client_kwargs(client_type: type, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Filters kwargs to only include valid parameters for a client type."""
        if client_type is object:
            return kwargs.copy()

        try:
            init_method = getattr(client_type, "__init__", None)
            if init_method is None:
                return kwargs.copy()

            sig = inspect.signature(init_method)
            param_names = {
                name
                for name, param in sig.parameters.items()
                if name != "self" and param.kind != inspect.Parameter.POSITIONAL_ONLY
            }

            has_var_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
            )

            if has_var_kwargs:
                return kwargs.copy()

            return {k: v for k, v in kwargs.items() if k in param_names}

        except (ValueError, TypeError, AttributeError):
            return kwargs.copy()

    @staticmethod
    def _filter_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Filters kwargs to only include valid parameters for a function."""
        try:
            sig = inspect.signature(func)
            param_names = {
                name
                for name, param in sig.parameters.items()
                if param.kind != inspect.Parameter.POSITIONAL_ONLY
            }

            has_var_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
            )

            if has_var_kwargs:
                return kwargs.copy()

            return {k: v for k, v in kwargs.items() if k in param_names}
        except (ValueError, TypeError, AttributeError):
            return {}

    # ====================================================================
    # Context managers
    # ====================================================================

    async def __aenter__(self):
        """Enters the async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exits the async context manager, closing the async client."""
        if self._async_client is not None:
            # Try async close first
            aclose_method = getattr(self._async_client, "aclose", None)
            if callable(aclose_method):
                with contextlib.suppress(Exception):
                    result = aclose_method()
                    if inspect.iscoroutine(result):
                        await result

            # Fallback to sync close
            close_method = getattr(self._async_client, "close", None)
            if callable(close_method):
                with contextlib.suppress(Exception):
                    close_method()

    # ====================================================================
    # String representations
    # ====================================================================

    def __repr__(self) -> str:
        """Returns a detailed string representation."""
        return (
            f"{self.__class__.__name__}("
            f"capabilities={self._capabilities}, "
            f"requests={self._request_count}, "
            f"errors={self._error_count})"
        )

    def __str__(self) -> str:
        """Returns a human-readable string representation."""
        return (
            f"{self.__class__.__name__} Client\n"
            f"- Created: {self.created_at}\n"
            f"- Requests: {self._request_count}\n"
            f"- Errors: {self._error_count}\n"
            f"- Capabilities: {self._capabilities}"
        )
