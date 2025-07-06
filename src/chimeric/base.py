from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator
import contextlib
from datetime import datetime
import inspect
import time
from typing import Any, Generic, TypeVar

from .exceptions import (
    ChimericError,
    ProviderError,
)
from .tools import ToolManager
from .types import (
    Capability,
    ChimericCompletionResponse,
    ChimericFileUploadResponse,
    ChimericStreamChunk,
    Input,
    ModelSummary,
    Tool,
    ToolCallChunk,
    Tools,
)

__all__ = [
    "AsyncClientType",
    "BaseClient",
    "ClientType",
    "CompletionResponseType",
    "FileUploadResponseType",
    "StreamType",
]

# Type variables for provider-specific responses.
ClientType = TypeVar("ClientType")
AsyncClientType = TypeVar("AsyncClientType")
CompletionResponseType = TypeVar("CompletionResponseType")
StreamType = TypeVar("StreamType")
FileUploadResponseType = TypeVar("FileUploadResponseType")


class BaseClient(
    ABC,
    Generic[
        ClientType,
        AsyncClientType,
        CompletionResponseType,
        StreamType,
        FileUploadResponseType,
    ],
):
    """Abstract base class for all LLM provider clients.

    This class defines the unified interface that provider-specific clients must implement to ensure consistent behavior across different LLM backends.
    It provides common functionality like request tracking, error handling, and capability management.

    The class uses generics to maintain type safety while allowing provider-specific response types. Subclasses should specify their concrete types
    when inheriting from this class.

    Type Parameters:
    ClientType: The provider's native client type
    CompletionResponseType: The provider's completion response type
    StreamType: The provider's streaming chunk type
    AgentResponseType: The provider's agent response type
    FileUploadResponseType: The provider's file upload response type

    Attributes:
        api_key: Optional API key for authentication.
        created_at: Timestamp when the client was instantiated.
    """

    def __init__(
        self,
        api_key: str,
        tool_manager: ToolManager,
        **kwargs: Any,
    ) -> None:
        """Initializes the base client with common settings.

        Sets up request tracking, error counting, and initializes both sync
        and async clients with filtered kwargs specific to each client type.

        Args:
            api_key: Optional API key for authentication. Some providers may
                not require this if using other auth methods.
            tool_manager: ToolManager instance for managing tools available
            **kwargs: Additional provider-specific options. Invalid options
                for the specific provider will be automatically filtered out
                based on the client's __init__ signature.
        """
        self.api_key = api_key
        self.tool_manager = tool_manager
        self.created_at = datetime.now()
        self._request_count = 0
        self._last_request_time: float | None = None
        self._error_count = 0

        # Get the actual client types from the generic parameters.
        client_types = self._get_generic_types()

        # Filter kwargs for each client type to avoid invalid parameter errors.
        sync_kwargs = self._filter_client_kwargs(client_types["sync"], kwargs)
        async_kwargs = self._filter_client_kwargs(client_types["async"], kwargs)

        # Initialize clients with filtered kwargs.
        self._client: ClientType = self._init_client(client_types["sync"], **sync_kwargs)
        self._async_client: AsyncClientType = self._init_async_client(
            client_types["async"], **async_kwargs
        )
        self._capabilities = self._get_capabilities()

    @abstractmethod
    def _get_generic_types(self) -> dict[str, type]:
        """Returns the actual types from the Generic parameters.

        Since Python's runtime type information from Generics is limited,
        subclasses must override this method to provide the actual client
        types they use. This enables proper kwargs filtering.

        Returns:
            A dictionary with 'sync' and 'async' keys containing the actual
            client types used by the provider.

        Example:
            return {
                "sync": openai.Client,
                "async": openai.AsyncClient,
            }
        """
        # Fallback implementation that won't filter anything.
        return {
            "sync": object,
            "async": object,
        }

    @staticmethod
    def _filter_client_kwargs(client_type: type, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Filters kwargs to only include those valid for the client type.

        Inspects the client's __init__ method signature and filters out any
        kwargs that aren't accepted by that method. This prevents errors
        when initializing provider-specific clients with unsupported parameters.

        Args:
            client_type: The client class to filter kwargs for.
            kwargs: Original keyword arguments to filter.

        Returns:
            A new dictionary containing only the kwargs that are valid for
            the client type's __init__ method.
        """
        if client_type is object:
            # Fallback case when actual type info isn't available.
            return kwargs.copy()

        try:
            init_method = getattr(client_type, "__init__", None)
            if init_method is None:
                return kwargs.copy()

            sig = inspect.signature(init_method)

            # Get parameter names (excluding 'self' and positional-only).
            param_names = {
                name
                for name, param in sig.parameters.items()
                if name != "self" and param.kind != inspect.Parameter.POSITIONAL_ONLY
            }

            # Check if the method accepts **kwargs.
            has_var_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
            )

            if has_var_kwargs:
                # Method accepts arbitrary kwargs, return all.
                return kwargs.copy()

            # Filter kwargs to only include those in the method signature.
            return {k: v for k, v in kwargs.items() if k in param_names}

        except (ValueError, TypeError, AttributeError):
            # If the signature inspection fails, return all kwargs as fallback.
            return kwargs.copy()

    # ====================================================================
    # Core abstract methods (must be implemented by provider clients)
    # ====================================================================

    @abstractmethod
    def _init_client(self, client_type: type, **kwargs: Any) -> ClientType:
        """Initializes the provider-specific synchronous client.

        Args:
            client_type: The actual client type for reference.
            **kwargs: Provider-specific initialization options that have
                been pre-filtered for this client type.

        Returns:
            An instance of the provider's synchronous client.

        Example:
            return openai.Client(api_key=self.api_key, **kwargs)
        """

    @abstractmethod
    def _init_async_client(self, async_client_type: type, **kwargs: Any) -> AsyncClientType | None:
        """Initializes the provider-specific asynchronous client.

        Args:
            async_client_type: The actual async client type for reference.
            **kwargs: Provider-specific initialization options that have
                been pre-filtered for this client type.

        Returns:
            An instance of the provider's async client, or None if the
            provider doesn't support async operations.

        Example:
            return openai.AsyncClient(api_key=self.api_key, **kwargs)
        """
        return None  # Default implementation for providers without async support.

    @abstractmethod
    def _get_capabilities(self) -> Capability:
        """Returns the features supported by this provider.

        Returns:
            A Capability object indicating which features (multimodal input,
            tool calling, file uploads, etc.) are supported by this provider.

        Example:
            return Capability(
                multimodal=True,
                tools=True,
                streaming=True,
                files=False,
                agents=False,
            )
        """

    def _get_model_aliases(self) -> list[str]:
        """Returns a list of model alias names to include in model listings.

        Providers can override this method to provide additional model aliases
        that will be included in the list_models() output.

        Returns:
            A list of alias model names. By default, returns an empty list (no aliases).
        """
        return []

    @abstractmethod
    def _list_models_impl(self) -> list[ModelSummary]:
        """Provider-specific implementation of model listing from API.

        Returns:
            A list of ModelSummary objects containing basic information
            about each available model from the provider's API.

        Raises:
            ProviderError: If the provider's API returns an error.
        """

    def list_models(self) -> list[ModelSummary]:
        """Lists available models from this provider, including aliases.

        Returns:
            A list of ModelSummary objects containing basic information
            about each available model, including any aliases defined by _get_model_aliases().

        Raises:
            ProviderError: If the provider's API returns an error.
        """
        # Get models from provider API
        api_models = self._list_models_impl()

        # Get aliases and add them as simple ModelSummary objects
        aliases = self._get_model_aliases()
        alias_models = [
            ModelSummary(
                id=alias,
                name=alias,
            )
            for alias in aliases
        ]

        return api_models + alias_models

    # ====================================================================
    # Chat completion methods
    # ====================================================================

    def _encode_tools(self, tools: Tools = None) -> Tools:
        """Encodes tools into a format suitable for the provider's API."""
        if not tools:
            return None

        return [tool.model_dump() if isinstance(tool, Tool) else tool for tool in tools]

    def _accumulate_tool_call_arguments(
        self, tool_calls: dict[str, ToolCallChunk], tool_call_id: str, arguments_delta: str
    ) -> None:
        """Accumulates arguments for a streaming tool call.

        Args:
            tool_calls: Dictionary mapping tool call IDs to ToolCallChunk objects.
            tool_call_id: The ID of the tool call to update.
            arguments_delta: The new arguments fragment to append.
        """
        if tool_call_id in tool_calls:
            tool_calls[tool_call_id].arguments += arguments_delta
            tool_calls[tool_call_id].arguments_delta = arguments_delta

    def _execute_accumulated_tool_calls(
        self, tool_calls: dict[str, ToolCallChunk]
    ) -> list[dict[str, Any]]:
        """Executes all accumulated tool calls and returns their results.

        Args:
            tool_calls: Dictionary of accumulated tool calls.

        Returns:
            List of tool call results with metadata.
        """
        results = []
        for tool_call in tool_calls.values():
            if tool_call.status == "completed":
                try:
                    tool = self.tool_manager.get_tool(tool_call.name)
                    if tool and tool.function:
                        import json

                        args = json.loads(tool_call.arguments)
                        result = tool.function(**args)
                        results.append(
                            {
                                "call_id": tool_call.call_id or tool_call.id,
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                                "result": str(result),
                            }
                        )
                except Exception as e:
                    results.append(
                        {
                            "call_id": tool_call.call_id or tool_call.id,
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                            "error": str(e),
                        }
                    )
        return results

    def _process_tools(self, auto_tool: bool, tools: Tools = None) -> Tools:
        """Processes tools for chat completion requests."""
        if tools and not self.supports_tools():
            raise ChimericError("This provider does not support tool calling")

        final_tools = tools
        if not final_tools and auto_tool:
            # Automatically include all tools bound to the client if no tools specified
            final_tools = self.tool_manager.get_all_tools()

        return self._encode_tools(final_tools)

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

        Handles request tracking and error counting, then delegates to the
        provider-specific implementation.

        Args:
            messages: Input for the completion. Can be a string prompt,
                list of messages, or dictionary of provider-specific parameters.
            model: Model identifier to use for the completion.
            tools: Optional list of tools to make available to the model.
            stream: Whether to return a streaming response.
            auto_tool: If True, automatically includes all tools bound to the client
            **kwargs: Provider-specific parameters passed directly to the
                underlying SDK after filtering.

        Returns:
            Either a ChimericCompletionResponse containing both native and
            standardized response formats, or a Generator of streaming chunks
            if streaming is enabled.

        Raises:
            ProviderError: If the provider's API returns an error.
            ValueError: If required parameters are missing or invalid.
        """
        if stream and not self.supports_streaming():
            raise ChimericError("This provider does not support streaming responses")
        try:
            self._request_count += 1
            self._last_request_time = time.time()
            encoded_tools = self._process_tools(auto_tool, tools)

            return self._chat_completion_impl(messages, model, stream, encoded_tools, **kwargs)
        except Exception:
            self._error_count += 1
            raise

    async def achat_completion(
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

        Async version of chat_completion with the same functionality but
        non-blocking operation.

        Args:
            messages: Input for the completion. Can be a string prompt,
                list of messages, or dictionary of provider-specific parameters.
            model: Model identifier to use for the completion.
            tools: Optional list of tools to make available to the model.
            stream: Whether to return a streaming response.
            auto_tool: If True, automatically includes all tools bound to the client
            **kwargs: Provider-specific parameters passed directly to the
                underlying SDK after filtering.

        Returns:
            Either a ChimericCompletionResponse containing both native and
            standardized response formats, or an AsyncGenerator of streaming
            chunks if streaming is enabled.

        Raises:
            ProviderError: If the provider's API returns an error.
            ValueError: If required parameters are missing or invalid.
            NotImplementedError: If the provider doesn't support async operations.
        """
        if stream and not self.supports_streaming():
            raise ChimericError("This provider does not support streaming responses")
        try:
            self._request_count += 1
            self._last_request_time = time.time()
            encoded_tools = self._process_tools(auto_tool, tools)
            return await self._achat_completion_impl(
                messages, model, stream, encoded_tools, **kwargs
            )
        except Exception:
            self._error_count += 1
            raise

    @abstractmethod
    def _chat_completion_impl(
        self,
        messages: Input,
        model: str,
        stream: bool = False,
        tools: Tools = None,
        **kwargs: Any,
    ) -> (
        ChimericCompletionResponse[CompletionResponseType]
        | Generator[ChimericStreamChunk[StreamType], None, None]
    ):
        """Provider-specific implementation of synchronous chat completion.

        This method contains the actual logic for making API calls to the
        provider's chat completion endpoint. It should handle both streaming
        and non-streaming responses.

        Args:
            messages: Input for the completion.
            model: Model identifier to use.
            stream: Whether to return a streaming response.
            tools: Optional list of tools to make available to the model.
            **kwargs: Provider-specific parameters that have been filtered
                for this method's signature.

        Returns:
            Either a ChimericCompletionResponse wrapping the provider's native
            response, or a Generator yielding ChimericStreamChunk objects for
            streaming responses.
        """

    @abstractmethod
    async def _achat_completion_impl(
        self,
        messages: Input,
        model: str,
        stream: bool = False,
        tools: Tools = None,
        **kwargs: Any,
    ) -> (
        ChimericCompletionResponse[CompletionResponseType]
        | AsyncGenerator[ChimericStreamChunk[StreamType], None]
    ):
        """Provider-specific implementation of asynchronous chat completion.

        Async version of _chat_completion_impl with the same responsibilities
        but using async/await patterns.

        Args:
            messages: Input for the completion.
            model: Model identifier to use.
            tools: Optional list of tools to make available to the model.
            stream: Whether to return a streaming response.
            **kwargs: Provider-specific parameters that have been filtered
                for this method's signature.

        Returns:
            Either a ChimericCompletionResponse wrapping the provider's native
            response, or an AsyncGenerator yielding ChimericStreamChunk objects
            for streaming responses.
        """

    # ====================================================================
    # Model information methods
    # ====================================================================

    def get_model_info(self, model_id: str) -> ModelSummary:
        """Gets detailed information about a specific model.

        Args:
            model_id: ID or name of the model to get information for.

        Returns:
            A ModelSummary object containing detailed information about the model.
            If the model_id is an alias, returns the alias model info.

        Raises:
            ValueError: If the model is not found or not available from this provider.
        """
        models = self.list_models()
        for model in models:
            if model.id == model_id or model.name == model_id:
                return model
        raise ValueError(f"Model {model_id} not found")

    def upload_file(self, **kwargs: Any) -> ChimericFileUploadResponse[FileUploadResponseType]:
        """Uploads a file to the provider for use in conversations.

        Args:
            **kwargs: Provider-specific arguments for file upload (e.g.,
                file path, file object, purpose, etc.).

        Returns:
            A ChimericFileUploadResponse containing both the provider's native
            response and standardized file information.

        Raises:
            NotImplementedError: If the provider does not support file uploads.
            ProviderError: If there is an error uploading the file.
            ValueError: If required parameters are missing or invalid.
        """
        if not self.supports_files():
            raise NotImplementedError("Provider does not support file uploads")

        try:
            self._request_count += 1
            self._last_request_time = time.time()
            return self._upload_file(**kwargs)
        except ProviderError:
            # Re-raise ProviderError as-is to preserve status_code and other details
            self._error_count += 1
            raise
        except ValueError:
            # Re-raise ValueError as-is for invalid parameters
            self._error_count += 1
            raise
        except Exception as e:
            self._error_count += 1
            provider_name = getattr(self, "_provider_name", self.__class__.__name__)
            raise ProviderError(
                provider=provider_name,
                response_text=f"Failed to upload file: {e!s}",
                endpoint="file_upload",
                status_code=e.status_code if hasattr(e, "status_code") else None,
            ) from e

    @abstractmethod
    def _upload_file(self, **kwargs: Any) -> ChimericFileUploadResponse[FileUploadResponseType]:
        """Provider-specific implementation of file upload.

        Subclasses that support file uploads should override this method.

        Args:
            **kwargs: Provider-specific file upload parameters.

        Returns:
            A ChimericFileUploadResponse wrapping the provider's native response.

        Raises:
            NotImplementedError: By default, since this must be implemented
                by subclasses that support file uploads.
        """
        raise NotImplementedError("File upload not implemented")

    # ====================================================================
    # Capability check methods
    # ====================================================================

    def supports_multimodal(self) -> bool:
        """Checks if the provider supports multimodal inputs.

        Returns:
            True if the provider can handle image, audio, or video inputs
            in addition to text.
        """
        return self._capabilities.multimodal

    def supports_tools(self) -> bool:
        """Checks if the provider supports tool/function calling.

        Returns:
            True if the provider can call external tools or functions
            during completion generation.
        """
        return self._capabilities.tools

    def supports_agents(self) -> bool:
        """Checks if the provider supports agent creation and management.

        Returns:
            True if the provider supports creating persistent agents
            with custom instructions and capabilities.
        """
        return self._capabilities.agents

    def supports_files(self) -> bool:
        """Checks if the provider supports file uploads.

        Returns:
            True if files can be uploaded to the provider for use
            in conversations or as context.
        """
        return self._capabilities.files

    def supports_streaming(self) -> bool:
        """Checks if the provider supports streaming responses.

        Returns:
            True if the provider can return responses as a stream
            of chunks rather than waiting for the complete response.
        """
        return self._capabilities.streaming

    # ====================================================================
    # Utility methods and properties
    # ====================================================================

    @staticmethod
    def _filter_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Filters kwargs to only include those valid for a specific function.

        This helper ensures only relevant arguments are passed to provider
        SDK methods by inspecting the function signature and filtering out
        unsupported parameters.

        Args:
            func: The function whose signature will be used for filtering.
            kwargs: Original keyword arguments to filter.

        Returns:
            A filtered dictionary containing only the kwargs that are valid
            for the specified function's signature.
        """
        try:
            sig = inspect.signature(func)
            # Get parameter names that aren't positional-only.
            param_names = {
                name
                for name, param in sig.parameters.items()
                if param.kind != inspect.Parameter.POSITIONAL_ONLY
            }

            # Check if function accepts **kwargs.
            has_var_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
            )

            if has_var_kwargs:
                # Function accepts arbitrary kwargs, return all.
                return kwargs.copy()

            # Filter kwargs to only include those in the function signature.
            return {k: v for k, v in kwargs.items() if k in param_names}
        except (ValueError, TypeError, AttributeError):
            # If the signature inspection fails, return empty dict to be safe.
            return {}

    @property
    def capabilities(self) -> Capability:
        """The supported capabilities and features of this provider."""
        return self._capabilities

    @property
    def client(self) -> ClientType:
        """The underlying provider client instance for synchronous operations."""
        return self._client

    @property
    def async_client(self) -> AsyncClientType | None:
        """The underlying provider async client instance, or None if not available."""
        return self._async_client

    @property
    def request_count(self) -> int:
        """The total number of API requests made through this client instance."""
        return self._request_count

    @property
    def error_count(self) -> int:
        """The total number of errors encountered during API requests."""
        return self._error_count

    @property
    def last_request_time(self) -> float | None:
        """Unix timestamp of the last request, or None if no requests made."""
        return self._last_request_time

    @property
    def client_age(self) -> float:
        """The age of this client instance in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

    # ====================================================================
    # Context manager methods
    # ====================================================================

    def __enter__(
        self,
    ) -> "BaseClient[ClientType, AsyncClientType, CompletionResponseType, StreamType, FileUploadResponseType]":
        """Context manager entry.

        Returns:
            Self for use in with statements.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exits the synchronous context manager.

        Ensures proper cleanup of synchronous client resources.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        if hasattr(self._client, "close"):
            self._client.close()

    async def __aenter__(
        self,
    ) -> "BaseClient[ClientType, AsyncClientType, CompletionResponseType, StreamType, FileUploadResponseType]":
        """Async context manager entry.

        Returns:
            Self for use in async with statements.
        """
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exits the asynchronous context manager.

        Ensures proper cleanup of async client resources, trying both
        async and sync close methods as fallbacks.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        if self._async_client is not None:
            # Try async close first if available.
            aclose_method = getattr(self._async_client, "aclose", None)
            if callable(aclose_method):
                with contextlib.suppress(Exception):
                    result = aclose_method()
                    # Await if the result is a coroutine.
                    if inspect.iscoroutine(result):
                        await result

            # Fallback to synchronous close.
            close_method = getattr(self._async_client, "close", None)
            if callable(close_method):
                with contextlib.suppress(Exception):
                    close_method()

    # ====================================================================
    # String representation
    # ====================================================================

    def __repr__(self) -> str:
        """Returns a detailed string representation of the client."""
        return (
            f"{self.__class__.__name__}("
            f"capabilities={self._capabilities}, "
            f"requests={self._request_count}, "
            f"errors={self._error_count})"
        )

    def __str__(self) -> str:
        """Returns a human-readable string representation of the client."""
        return (
            f"{self.__class__.__name__} Client\n"
            f"- Created: {self.created_at}\n"
            f"- Requests: {self._request_count}\n"
            f"- Errors: {self._error_count}\n"
            f"- Capabilities: {self._capabilities}"
        )
