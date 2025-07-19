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

from .exceptions import (
    ChimericError,
    ToolRegistrationError,
)
from .tools import ToolManager
from .types import (
    Capability,
    ChimericCompletionResponse,
    ChimericStreamChunk,
    Input,
    Message,
    ModelSummary,
    Tool,
    ToolCall,
    ToolExecutionResult,
    Tools,
    Usage,
)
from .utils import (
    StreamProcessor,
    create_completion_response,
    normalize_messages,
    normalize_tools,
)

__all__ = [
    "ChimericAsyncClient",
    "ChimericClient",
    "ChunkType",
    "CompletionResponseType",
]

# Type variables for provider-specific types
ClientType = TypeVar("ClientType")
CompletionResponseType = TypeVar("CompletionResponseType")
ChunkType = TypeVar("ChunkType")


class ChimericClient(
    ABC,
    Generic[
        ClientType,
        CompletionResponseType,
        ChunkType,
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
        """Initializes the base client with common settings.

        Args:
            api_key: The API key for the provider.
            tool_manager: The tool manager instance.
            **kwargs: Additional provider-specific keyword arguments.
        """
        self.api_key = api_key
        self.tool_manager = tool_manager
        self.created_at = datetime.now()
        self._request_count = 0
        self._last_request_time: float | None = None
        self._error_count = 0

        # Get client type and initialize client
        client_type = self._get_client_type()

        # Initialize client
        self._client: ClientType = self._init_client(client_type, **kwargs)
        self._capabilities = self._get_capabilities()

    # ====================================================================
    # Abstract methods - Required for all providers
    # ====================================================================

    @abstractmethod
    def _get_client_type(self) -> type:
        """Returns the actual client type used by the provider.

        This abstract method must be implemented by subclasses to specify the
        synchronous client class from the provider's library.

        Example:
            return openai.Client

        Returns:
            The provider's client class.
        """
        pass

    @abstractmethod
    def _init_client(self, client_type: type, **kwargs: Any) -> ClientType:
        """Initializes the provider's synchronous client.

        Args:
            client_type: The client class to initialize.
            **kwargs: Provider-specific arguments for client initialization.

        Returns:
            An instance of the provider's synchronous client.
        """
        pass

    @abstractmethod
    def _get_capabilities(self) -> Capability:
        """Returns the capabilities supported by this provider.

        Returns:
            A Capability object detailing supported features.
        """
        pass

    @abstractmethod
    def _list_models_impl(self) -> list[ModelSummary]:
        """Lists available models from the provider's API.

        Returns:
            A list of ModelSummary objects.
        """
        pass

    @abstractmethod
    def _messages_to_provider_format(self, messages: list[Message]) -> Any:
        """Converts standardized messages to provider-specific format.

        Args:
            messages: A list of standardized Message objects.

        Returns:
            Messages in the format required by the provider's API.
        """
        pass

    @abstractmethod
    def _tools_to_provider_format(self, tools: list[Tool]) -> Any:
        """Converts standardized tools to provider-specific format.

        Args:
            tools: A list of standardized Tool objects.

        Returns:
            Tools in the format required by the provider's API.
        """
        pass

    @abstractmethod
    def _make_provider_request(
        self,
        messages: Any,
        model: str,
        stream: bool,
        tools: Any = None,
        **kwargs: Any,
    ) -> CompletionResponseType | Any:
        """Makes the actual API request to the provider.

        Args:
            messages: Provider-formatted messages.
            model: The model to use for the request.
            stream: Whether to stream the response.
            tools: Provider-formatted tools, if any.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The raw response from the provider's API.
        """
        pass

    @abstractmethod
    def _process_provider_stream_event(
        self, event: Any, processor: StreamProcessor
    ) -> ChimericStreamChunk[ChunkType] | None:
        """Processes a provider-specific stream event.

        Providers should use create_stream_chunk() to create standardized chunks.

        Args:
            event: The native stream event from the provider.
            processor: The stream processor to manage stream state.

        Returns:
            A standardized ChimericStreamChunk or None if the event is to be
            skipped.

        Example:
            # For content delta
            if hasattr(event, 'delta'):
                return create_stream_chunk(
                    native_event=event,
                    processor=processor,
                    content_delta=event.delta
                )

            # For finish event
            if event.finish_reason:
                return create_stream_chunk(
                    native_event=event,
                    processor=processor,
                    finish_reason=event.finish_reason
                )
        """
        pass

    @abstractmethod
    def _extract_usage_from_response(self, response: CompletionResponseType) -> Usage:
        """Extracts usage information from provider response.

        Args:
            response: The native response from the provider.

        Returns:
            A standardized Usage object.
        """
        pass

    @abstractmethod
    def _extract_content_from_response(self, response: CompletionResponseType) -> str | list[Any]:
        """Extracts content from provider response.

        Args:
            response: The native response from the provider.

        Returns:
            The string content or list of content parts from the response.
        """
        pass

    @abstractmethod
    def _extract_tool_calls_from_response(
        self, response: CompletionResponseType
    ) -> list[ToolCall] | None:
        """Extracts tool calls from provider response.

        Args:
            response: The native response from the provider.

        Returns:
            A list of ToolCall objects or None if no tool calls are present.
        """
        pass

    # ====================================================================
    # Optional methods - Override based on capabilities
    # ====================================================================

    def _get_model_aliases(self) -> list[str]:
        """Return model aliases to include in model listings.

        Returns:
            A list of string aliases for models.
        """
        return []

    # ====================================================================
    # Tool execution
    # ====================================================================

    def _execute_tool_call(self, call: ToolCall) -> ToolExecutionResult:
        """Executes a single tool call with standardized error handling.

        Args:
            call: The ToolCall object to execute.

        Returns:
            A ToolExecutionResult object with the outcome.

        Raises:
            ToolRegistrationError: If the tool is not found or not callable.
        """
        result = ToolExecutionResult(call_id=call.call_id, name=call.name, arguments=call.arguments)

        tool = self.tool_manager.get_tool(call.name)
        if not tool or not callable(tool.function):
            raise ToolRegistrationError(f"Tool '{call.name}' is not callable")

        try:
            args = json.loads(call.arguments) if call.arguments else {}

            # Check if the tool function is async
            if inspect.iscoroutinefunction(tool.function):
                execution_result = asyncio.run(tool.function(**args))
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

    def _execute_tool_calls(self, calls: list[ToolCall]) -> list[ToolExecutionResult]:
        """Executes multiple tool calls in parallel.

        Args:
            calls: A list of ToolCall objects to execute.

        Returns:
            A list of ToolExecutionResult objects.
        """
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

    def _handle_tool_calling_completion(
        self, messages: Any, model: str, tools: Any, **kwargs: Any
    ) -> ChimericCompletionResponse[CompletionResponseType]:
        """Handles tool calling with iterative approach until completion.

        This method follows the elegant pattern of making requests, executing tools,
        and continuing until no more tool calls are needed.

        Args:
            messages: The initial provider-formatted messages.
            model: The model to use for the requests.
            tools: The provider-formatted tools.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The final ChimericCompletionResponse after all tool calls are resolved.
        """
        current_messages = list(messages) if isinstance(messages, list) else [messages]
        all_tool_results = []
        total_usage = Usage()

        # Continue until no more tool calls
        while True:
            # Make API request
            response: CompletionResponseType = self._make_provider_request(
                current_messages, model, False, tools, **kwargs
            )

            # Accumulate usage
            response_usage = self._extract_usage_from_response(response)
            if response_usage:
                total_usage.prompt_tokens += response_usage.prompt_tokens or 0
                total_usage.completion_tokens += response_usage.completion_tokens or 0
                total_usage.total_tokens += response_usage.total_tokens or 0

            # Check for tool calls
            tool_calls = self._extract_tool_calls_from_response(response)

            if not tool_calls:
                # No more tool calls, this is our final response
                final_response: CompletionResponseType = response
                break

            # Execute tool calls
            tool_results = self._execute_tool_calls(tool_calls)
            all_tool_results.extend(tool_results)

            # Update messages for next iteration
            current_messages = self._update_messages_with_tool_calls(
                current_messages, response, tool_calls, tool_results
            )

        # Extract final content
        content = self._extract_content_from_response(final_response)

        return create_completion_response(
            native_response=final_response,
            content=content,
            usage=total_usage,
            model=model,
            tool_calls=all_tool_results if all_tool_results else None,
        )

    def _update_messages_with_tool_calls(
        self,
        messages: list[Any],
        assistant_response: CompletionResponseType | ChunkType | Any,
        tool_calls: list[ToolCall],
        tool_results: list[ToolExecutionResult],
    ) -> list[Any]:
        """Updates message history with assistant response and tool results.

        This method should be overridden by providers to handle their specific
        message format for tool calls.

        Args:
            messages: The current list of provider-formatted messages.
            assistant_response: The native response from the assistant that
                contained the tool calls.
            tool_calls: The list of tool calls made by the assistant.
            tool_results: The results of executing the tool calls.

        Returns:
            An updated list of provider-formatted messages.

        Raises:
            NotImplementedError: If the provider supports tools but has not
                implemented this method.
        """
        # Default implementation - providers should override this if they support tool calling
        raise NotImplementedError(
            f"Provider {self.__class__.__name__} must implement _update_messages_with_tool_calls"
        )

    # ====================================================================
    # Streaming with tool support
    # ====================================================================

    def _process_provider_stream(
        self, stream: Any, processor: StreamProcessor
    ) -> Generator[ChimericStreamChunk[ChunkType], None, None]:
        """Processes a provider stream using the processor.

        Args:
            stream: The native stream object from the provider.
            processor: The stream processor to manage stream state.

        Yields:
            Standardized ChimericStreamChunk objects.
        """
        for event in stream:
            chunk = self._process_provider_stream_event(event, processor)
            if chunk:
                yield chunk

    def _handle_streaming_tool_calls(
        self,
        stream: Any,
        processor: StreamProcessor,
        messages: Any,
        model: str,
        tools: Any,
        **kwargs: Any,
    ) -> Generator[ChimericStreamChunk[ChunkType], None, None]:
        """Handles streaming with tool call support.

        Args:
            stream: The initial native stream from the provider.
            processor: The stream processor for the initial stream.
            messages: The initial provider-formatted messages.
            model: The model to use for requests.
            tools: The provider-formatted tools.
            **kwargs: Additional provider-specific parameters.

        Yields:
            ChimericStreamChunk objects from all sequential API calls.
        """
        # First, yield all chunks from the initial stream
        final_event: ChunkType | None = None
        for event in stream:
            final_event = event
            chunk = self._process_provider_stream_event(event, processor)
            if chunk:
                yield chunk

        # Check if we accumulated any tool calls
        completed_tool_calls = processor.get_completed_tool_calls()
        if completed_tool_calls:
            # Convert to ToolCall objects
            tool_calls = [
                ToolCall(
                    call_id=tc.call_id or tc.id,
                    name=tc.name,
                    arguments=tc.arguments,
                    metadata={"original_id": tc.id} if tc.id != (tc.call_id or tc.id) else None,
                )
                for tc in completed_tool_calls
            ]
            # Execute tools
            tool_results = self._execute_tool_calls(tool_calls)
            # Update messages with tool results
            current_messages = self._update_messages_with_tool_calls(
                messages, final_event, tool_calls, tool_results
            )

            # Make another request and stream it
            continuation_response = self._make_provider_request(
                current_messages, model, True, tools, **kwargs
            )

            # Create new processor for continuation
            continuation_processor = StreamProcessor()
            yield from self._handle_streaming_tool_calls(
                continuation_response,
                continuation_processor,
                current_messages,
                model,
                tools,
                **kwargs,
            )

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
        | Generator[ChimericStreamChunk[ChunkType], None, None]
    ):
        """Generates a synchronous chat completion.

        Args:
            messages: Input messages, which can be a string, a list of strings,
                or a list of Message objects.
            model: The identifier of the model to use for the completion.
            stream: If True, the response will be streamed as a generator of
                ChimericStreamChunk objects. Defaults to False.
            tools: A list of tools to make available to the model.
            auto_tool: If True and no tools are provided, all tools registered
                with the ToolManager will be used. Defaults to True.
            **kwargs: Additional provider-specific parameters to pass to the API.

        Returns:
            If stream is False, a ChimericCompletionResponse object.
            If stream is True, a generator of ChimericStreamChunk objects.

        Raises:
            ChimericError: If a requested capability (e.g., streaming or tools)
                is not supported by the provider.
            ProviderError: If the provider's API returns an error.
        """
        if stream and not self.supports_streaming():
            raise ChimericError("This provider does not support streaming responses")

        if tools and not self.supports_tools():
            raise ChimericError("This provider does not support tool calling")

        try:
            self._request_count += 1
            self._last_request_time = time.time()

            # Process tools
            final_tools = tools
            if not final_tools and auto_tool and self.supports_tools():
                final_tools = self.tool_manager.get_all_tools()

            # Normalize inputs
            normalized_messages = normalize_messages(messages)
            normalized_tools = normalize_tools(final_tools) if final_tools else None

            # Convert to provider format
            provider_messages = self._messages_to_provider_format(normalized_messages)
            provider_tools = (
                self._tools_to_provider_format(normalized_tools) if normalized_tools else None
            )

            if stream:
                # Streaming with tool support
                response = self._make_provider_request(
                    provider_messages, model, True, provider_tools, **kwargs
                )
                processor = StreamProcessor()

                if provider_tools:
                    return self._handle_streaming_tool_calls(
                        response, processor, provider_messages, model, provider_tools, **kwargs
                    )
                return self._process_provider_stream(response, processor)

            # Non-streaming
            if provider_tools:
                return self._handle_tool_calling_completion(
                    provider_messages, model, provider_tools, **kwargs
                )
            # Simple completion without tools
            response: CompletionResponseType = self._make_provider_request(
                provider_messages, model, False, None, **kwargs
            )
            content = self._extract_content_from_response(response)
            usage = self._extract_usage_from_response(response)

            return create_completion_response(
                native_response=response,
                content=content,
                usage=usage,
                model=model,
            )
        except Exception:
            self._error_count += 1
            raise

    def list_models(self) -> list[ModelSummary]:
        """Lists all available models including aliases.

        Returns:
            A list of ModelSummary objects for all available models.
        """
        api_models = self._list_models_impl()

        # Add aliases
        aliases = self._get_model_aliases()
        alias_models = [ModelSummary(id=alias, name=alias) for alias in aliases]

        return api_models + alias_models

    def get_model_info(self, model_id: str) -> ModelSummary:
        """Gets information about a specific model.

        Args:
            model_id: The ID or name of the model to look up.

        Returns:
            A ModelSummary object for the specified model.

        Raises:
            ValueError: If the model is not found.
        """
        models = self.list_models()
        for model in models:
            if model.id == model_id or model.name == model_id:
                return model
        raise ValueError(f"Model {model_id} not found")

    # ====================================================================
    # Capability checks
    # ====================================================================

    def supports_tools(self) -> bool:
        """Checks if the provider supports tool/function calling.

        Returns:
            True if tool calling is supported, False otherwise.
        """
        return self._capabilities.tools

    def supports_streaming(self) -> bool:
        """Checks if the provider supports streaming responses.

        Returns:
            True if streaming is supported, False otherwise.
        """
        return self._capabilities.streaming

    # ====================================================================
    # Properties
    # ====================================================================

    @property
    def capabilities(self) -> Capability:
        """All capabilities of this provider.

        Returns:
            A Capability object.
        """
        return self._capabilities

    @property
    def client(self) -> ClientType:
        """The underlying synchronous client instance.

        Returns:
            The provider-specific synchronous client.
        """
        return self._client

    @property
    def request_count(self) -> int:
        """Total number of API requests made.

        Returns:
            The total number of requests.
        """
        return self._request_count

    @property
    def error_count(self) -> int:
        """Total number of errors encountered.

        Returns:
            The total number of errors.
        """
        return self._error_count

    @property
    def last_request_time(self) -> float | None:
        """Unix timestamp of the last request.

        Returns:
            The timestamp of the last request, or None if no requests have been made.
        """
        return self._last_request_time

    @property
    def client_age(self) -> float:
        """Age of this client instance in seconds.

        Returns:
            The age of the client in seconds.
        """
        return (datetime.now() - self.created_at).total_seconds()

    # ====================================================================
    # Context managers
    # ====================================================================

    def __enter__(self):
        """Enters the context manager.

        Returns:
            The client instance.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exits the context manager, closing the client if possible.

        Args:
            exc_type: The exception type.
            exc_val: The exception value.
            exc_tb: The exception traceback.
        """
        if hasattr(self._client, "close"):
            self._client.close()

    # ====================================================================
    # String representations
    # ====================================================================

    def __repr__(self) -> str:
        """Returns a detailed string representation of the client.

        Returns:
            A string representation.
        """
        return (
            f"{self.__class__.__name__}("
            f"capabilities={self._capabilities}, "
            f"requests={self._request_count}, "
            f"errors={self._error_count})"
        )

    def __str__(self) -> str:
        """Returns a human-readable string representation of the client.

        Returns:
            A human-readable string.
        """
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
        ChunkType,
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
        """Initializes the base async client with common settings.

        Args:
            api_key: The API key for the provider.
            tool_manager: The tool manager instance.
            **kwargs: Additional provider-specific keyword arguments.
        """
        self.api_key = api_key
        self.tool_manager = tool_manager
        self.created_at = datetime.now()
        self._request_count = 0
        self._last_request_time: float | None = None
        self._error_count = 0

        # Get client type and initialize client
        client_type = self._get_async_client_type()

        # Initialize client
        self._async_client: ClientType = self._init_async_client(client_type, **kwargs)
        self._capabilities = self._get_capabilities()

    # ====================================================================
    # Abstract methods - Required for all providers
    # ====================================================================

    @abstractmethod
    def _get_async_client_type(self) -> type:
        """Returns the actual async client type used by the provider.

        This abstract method must be implemented by subclasses to specify the
        asynchronous client class from the provider's library.

        Example:
            return openai.AsyncClient

        Returns:
            The provider's async client class.
        """
        pass

    @abstractmethod
    def _init_async_client(self, async_client_type: type, **kwargs: Any) -> ClientType:
        """Initializes the provider's asynchronous client.

        Args:
            async_client_type: The async client class to initialize.
            **kwargs: Provider-specific arguments for client initialization.

        Returns:
            An instance of the provider's asynchronous client.
        """
        pass

    @abstractmethod
    def _get_capabilities(self) -> Capability:
        """Returns the capabilities supported by this provider.

        Returns:
            A Capability object detailing supported features.
        """
        pass

    @abstractmethod
    async def _list_models_impl(self) -> list[ModelSummary]:
        """Lists available models from the provider's API asynchronously.

        Returns:
            A list of ModelSummary objects.
        """
        pass

    @abstractmethod
    def _messages_to_provider_format(self, messages: list[Message]) -> Any:
        """Converts standardized messages to provider-specific format.

        Args:
            messages: A list of standardized Message objects.

        Returns:
            Messages in the format required by the provider's API.
        """
        pass

    @abstractmethod
    def _tools_to_provider_format(self, tools: list[Tool]) -> Any:
        """Converts standardized tools to provider-specific format.

        Args:
            tools: A list of standardized Tool objects.

        Returns:
            Tools in the format required by the provider's API.
        """
        pass

    @abstractmethod
    async def _make_async_provider_request(
        self,
        messages: Any,
        model: str,
        stream: bool,
        tools: Any = None,
        **kwargs: Any,
    ) -> CompletionResponseType | Any:
        """Makes the actual async API request to the provider.

        Args:
            messages: Provider-formatted messages.
            model: The model to use for the request.
            stream: Whether to stream the response.
            tools: Provider-formatted tools, if any.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The raw response from the provider's API.
        """
        pass

    @abstractmethod
    def _process_provider_stream_event(
        self, event: Any, processor: StreamProcessor
    ) -> ChimericStreamChunk[ChunkType] | None:
        """Processes a provider-specific stream event.

        Providers should use create_stream_chunk() to create standardized chunks.

        Args:
            event: The native stream event from the provider.
            processor: The stream processor to manage stream state.

        Returns:
            A standardized ChimericStreamChunk or None if the event is to be
            skipped.

        Example:
            # For content delta
            if hasattr(event, 'delta'):
                return create_stream_chunk(
                    native_event=event,
                    processor=processor,
                    content_delta=event.delta
                )

            # For finish event
            if event.finish_reason:
                return create_stream_chunk(
                    native_event=event,
                    processor=processor,
                    finish_reason=event.finish_reason
                )
        """
        pass

    @abstractmethod
    def _extract_usage_from_response(self, response: CompletionResponseType) -> Usage:
        """Extracts usage information from provider response.

        Args:
            response: The native response from the provider.

        Returns:
            A standardized Usage object.
        """
        pass

    @abstractmethod
    def _extract_content_from_response(self, response: CompletionResponseType) -> str | list[Any]:
        """Extracts content from provider response.

        Args:
            response: The native response from the provider.

        Returns:
            The string content or list of content parts from the response.
        """
        pass

    @abstractmethod
    def _extract_tool_calls_from_response(
        self, response: CompletionResponseType
    ) -> list[ToolCall] | None:
        """Extracts tool calls from provider response.

        Args:
            response: The native response from the provider.

        Returns:
            A list of ToolCall objects or None if no tool calls are present.
        """
        pass

    # ====================================================================
    # Optional methods - Override based on capabilities
    # ====================================================================

    def _get_model_aliases(self) -> list[str]:
        """Return model aliases to include in model listings.

        Returns:
            A list of string aliases for models.
        """
        return []

    # ====================================================================
    # Tool execution
    # ====================================================================

    async def _execute_tool_call(self, call: ToolCall) -> ToolExecutionResult:
        """Executes a single tool call asynchronously.

        Handles both sync and async tool functions.

        Args:
            call: The ToolCall object to execute.

        Returns:
            A ToolExecutionResult object with the outcome.

        Raises:
            ToolRegistrationError: If the tool is not found or not callable.
        """
        result = ToolExecutionResult(call_id=call.call_id, name=call.name, arguments=call.arguments)

        tool = self.tool_manager.get_tool(call.name)
        if not tool or not callable(tool.function):
            raise ToolRegistrationError(f"Tool '{call.name}' is not callable")

        try:
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
        """Executes multiple tool calls in parallel using asyncio.

        Args:
            calls: A list of ToolCall objects to execute.

        Returns:
            A list of ToolExecutionResult objects.
        """
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

    async def _handle_tool_calling_completion(
        self, messages: Any, model: str, tools: Any, **kwargs: Any
    ) -> ChimericCompletionResponse[CompletionResponseType]:
        """Handles tool calling with iterative approach until completion (async).

        Args:
            messages: The initial provider-formatted messages.
            model: The model to use for the requests.
            tools: The provider-formatted tools.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The final ChimericCompletionResponse after all tool calls are resolved.
        """
        current_messages = list(messages) if isinstance(messages, list) else [messages]
        all_tool_results = []
        total_usage = Usage()
        final_response: CompletionResponseType | None = None

        # Continue until no more tool calls
        while True:
            # Make API request
            response: CompletionResponseType = await self._make_async_provider_request(
                current_messages, model, False, tools, **kwargs
            )

            # Accumulate usage
            response_usage = self._extract_usage_from_response(response)
            if response_usage:
                total_usage.prompt_tokens += response_usage.prompt_tokens or 0
                total_usage.completion_tokens += response_usage.completion_tokens or 0
                total_usage.total_tokens += response_usage.total_tokens or 0

            # Check for tool calls
            tool_calls = self._extract_tool_calls_from_response(response)

            if not tool_calls:
                # No more tool calls, this is our final response
                final_response = response
                break

            # Execute tool calls
            tool_results = await self._execute_tool_calls(tool_calls)
            all_tool_results.extend(tool_results)

            # Update messages for next iteration
            current_messages = self._update_messages_with_tool_calls(
                current_messages, response, tool_calls, tool_results
            )

        # Extract final content
        content = self._extract_content_from_response(final_response)

        return create_completion_response(
            native_response=final_response,
            content=content,
            usage=total_usage,
            model=model,
            tool_calls=all_tool_results if all_tool_results else None,
        )

    def _update_messages_with_tool_calls(
        self,
        messages: list[Any],
        assistant_response: CompletionResponseType | ChunkType | Any,
        tool_calls: list[ToolCall],
        tool_results: list[ToolExecutionResult],
    ) -> list[Any]:
        """Updates message history with assistant response and tool results.

        This method should be overridden by providers to handle their specific
        message format for tool calls.

        Args:
            messages: The current list of provider-formatted messages.
            assistant_response: The native response from the assistant that
                contained the tool calls.
            tool_calls: The list of tool calls made by the assistant.
            tool_results: The results of executing the tool calls.

        Returns:
            An updated list of provider-formatted messages.

        Raises:
            NotImplementedError: If the provider supports tools but has not
                implemented this method.
        """
        # Default implementation - providers should override this if they support tool calling
        raise NotImplementedError(
            f"Provider {self.__class__.__name__} must implement _update_messages_with_tool_calls"
        )

    # ====================================================================
    # Streaming with tool support (Async)
    # ====================================================================

    async def _process_async_provider_stream(
        self, stream: Any, processor: StreamProcessor
    ) -> AsyncGenerator[ChimericStreamChunk[ChunkType], None]:
        """Processes an async provider stream using the processor.

        Args:
            stream: The native async stream object from the provider.
            processor: The stream processor to manage stream state.

        Yields:
            Standardized ChimericStreamChunk objects.
        """
        async for event in stream:
            chunk = self._process_provider_stream_event(event, processor)
            if chunk:
                yield chunk

    async def _handle_streaming_tool_calls(
        self,
        stream: Any,
        processor: StreamProcessor,
        messages: Any,
        model: str,
        tools: Any,
        **kwargs: Any,
    ) -> AsyncGenerator[ChimericStreamChunk[ChunkType], None]:
        """Handles streaming with tool call support (async).

        Args:
            stream: The initial native async stream from the provider.
            processor: The stream processor for the initial stream.
            messages: The initial provider-formatted messages.
            model: The model to use for requests.
            tools: The provider-formatted tools.
            **kwargs: Additional provider-specific parameters.

        Yields:
            ChimericStreamChunk objects from all sequential API calls.
        """
        # First, yield all chunks from the initial stream
        final_event: ChunkType | None = None
        async for event in stream:
            final_event = event
            chunk = self._process_provider_stream_event(event, processor)
            if chunk:
                yield chunk

        # Check if we accumulated any tool calls
        completed_tool_calls = processor.get_completed_tool_calls()
        if completed_tool_calls:
            # Convert to ToolCall objects
            tool_calls = [
                ToolCall(
                    call_id=tc.call_id or tc.id,
                    name=tc.name,
                    arguments=tc.arguments,
                    metadata={"original_id": tc.id} if tc.id != (tc.call_id or tc.id) else None,
                )
                for tc in completed_tool_calls
            ]

            # Execute tools
            tool_results = await self._execute_tool_calls(tool_calls)

            # Update messages with tool results
            current_messages = self._update_messages_with_tool_calls(
                messages, final_event, tool_calls, tool_results
            )

            # Make another request and stream it
            continuation_response = await self._make_async_provider_request(
                current_messages, model, True, tools, **kwargs
            )

            # Create new processor for continuation
            continuation_processor = StreamProcessor()
            async for chunk in self._handle_streaming_tool_calls(
                continuation_response,
                continuation_processor,
                current_messages,
                model,
                tools,
                **kwargs,
            ):
                yield chunk

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
        | AsyncGenerator[ChimericStreamChunk[ChunkType], None]
    ):
        """Generates an asynchronous chat completion.

        Args:
            messages: Input messages, which can be a string, a list of strings,
                or a list of Message objects.
            model: The identifier of the model to use for the completion.
            stream: If True, the response will be streamed as an async generator
                of ChimericStreamChunk objects. Defaults to False.
            tools: A list of tools to make available to the model.
            auto_tool: If True and no tools are provided, all tools registered
                with the ToolManager will be used. Defaults to True.
            **kwargs: Additional provider-specific parameters to pass to the API.

        Returns:
            If stream is False, a ChimericCompletionResponse object.
            If stream is True, an async generator of ChimericStreamChunk objects.

        Raises:
            ChimericError: If a requested capability (e.g., streaming or tools)
                is not supported by the provider.
            ProviderError: If the provider's API returns an error.
        """
        if stream and not self.supports_streaming():
            raise ChimericError("This provider does not support streaming responses")

        if tools and not self.supports_tools():
            raise ChimericError("This provider does not support tool calling")

        try:
            self._request_count += 1
            self._last_request_time = time.time()

            # Process tools
            final_tools = tools
            if not final_tools and auto_tool and self.supports_tools():
                final_tools = self.tool_manager.get_all_tools()

            # Normalize inputs
            normalized_messages = normalize_messages(messages)
            normalized_tools = normalize_tools(final_tools) if final_tools else None

            # Convert to provider format
            provider_messages = self._messages_to_provider_format(normalized_messages)
            provider_tools = (
                self._tools_to_provider_format(normalized_tools) if normalized_tools else None
            )

            if stream:
                # Streaming with tool support
                response = await self._make_async_provider_request(
                    provider_messages, model, True, provider_tools, **kwargs
                )
                processor = StreamProcessor()

                if provider_tools:
                    return self._handle_streaming_tool_calls(
                        response, processor, provider_messages, model, provider_tools, **kwargs
                    )
                return self._process_async_provider_stream(response, processor)

            # Non-streaming
            if provider_tools:
                return await self._handle_tool_calling_completion(
                    provider_messages, model, provider_tools, **kwargs
                )
            # Simple completion without tools
            response: CompletionResponseType = await self._make_async_provider_request(
                provider_messages, model, False, None, **kwargs
            )
            content = self._extract_content_from_response(response)
            usage = self._extract_usage_from_response(response)

            return create_completion_response(
                native_response=response,
                content=content,
                usage=usage,
                model=model,
            )
        except Exception:
            self._error_count += 1
            raise

    async def list_models(self) -> list[ModelSummary]:
        """Lists all available models including aliases asynchronously.

        Returns:
            A list of ModelSummary objects for all available models.
        """
        api_models = await self._list_models_impl()

        # Add aliases
        aliases = self._get_model_aliases()
        alias_models = [ModelSummary(id=alias, name=alias) for alias in aliases]

        return api_models + alias_models

    async def get_model_info(self, model_id: str) -> ModelSummary:
        """Gets information about a specific model asynchronously.

        Args:
            model_id: The ID or name of the model to look up.

        Returns:
            A ModelSummary object for the specified model.

        Raises:
            ValueError: If the model is not found.
        """
        models = await self.list_models()
        for model in models:
            if model.id == model_id or model.name == model_id:
                return model
        raise ValueError(f"Model {model_id} not found")

    # ====================================================================
    # Capability checks
    # ====================================================================

    def supports_tools(self) -> bool:
        """Checks if the provider supports tool/function calling.

        Returns:
            True if tool calling is supported, False otherwise.
        """
        return self._capabilities.tools

    def supports_streaming(self) -> bool:
        """Checks if the provider supports streaming responses.

        Returns:
            True if streaming is supported, False otherwise.
        """
        return self._capabilities.streaming

    # ====================================================================
    # Properties
    # ====================================================================

    @property
    def capabilities(self) -> Capability:
        """All capabilities of this provider.

        Returns:
            A Capability object.
        """
        return self._capabilities

    @property
    def async_client(self) -> ClientType:
        """The underlying asynchronous client instance.

        Returns:
            The provider-specific asynchronous client.
        """
        return self._async_client

    @property
    def request_count(self) -> int:
        """Total number of API requests made.

        Returns:
            The total number of requests.
        """
        return self._request_count

    @property
    def error_count(self) -> int:
        """Total number of errors encountered.

        Returns:
            The total number of errors.
        """
        return self._error_count

    @property
    def last_request_time(self) -> float | None:
        """Unix timestamp of the last request.

        Returns:
            The timestamp of the last request, or None if no requests have been made.
        """
        return self._last_request_time

    @property
    def client_age(self) -> float:
        """Age of this client instance in seconds.

        Returns:
            The age of the client in seconds.
        """
        return (datetime.now() - self.created_at).total_seconds()

    # ====================================================================
    # Context managers
    # ====================================================================

    async def __aenter__(self):
        """Enters the async context manager.

        Returns:
            The async client instance.
        """
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exits the async context manager, closing the async client.

        Args:
            exc_type: The exception type.
            exc_val: The exception value.
            exc_tb: The exception traceback.
        """
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
        """Returns a detailed string representation of the async client.

        Returns:
            A string representation.
        """
        return (
            f"{self.__class__.__name__}("
            f"capabilities={self._capabilities}, "
            f"requests={self._request_count}, "
            f"errors={self._error_count})"
        )

    def __str__(self) -> str:
        """Returns a human-readable string representation of the async client.

        Returns:
            A human-readable string.
        """
        return (
            f"{self.__class__.__name__} Client\n"
            f"- Created: {self.created_at}\n"
            f"- Requests: {self._request_count}\n"
            f"- Errors: {self._error_count}\n"
            f"- Capabilities: {self._capabilities}"
        )
