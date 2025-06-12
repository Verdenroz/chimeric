from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator
import contextlib
from datetime import datetime
import inspect
import time
from typing import Any, Generic, TypeVar

from .exceptions import (
    ProviderError,
    ToolRegistrationError,
)
from .types import (
    AgentResponse,
    Capability,
    CompletionResponse,
    FileUploadResponse,
    Input,
    MediaProcessingResult,
    MediaType,
    ModelInfo,
    ModelSummary,
    StreamChunk,
    Tool,
    ToolType,
    Usage,
)

__all__ = [
    "AgentResponse",
    "BaseClient",
    "Capability",
    "CompletionResponse",
    "FileUploadResponse",
    "Input",
    "MediaProcessingResult",
    "MediaType",
    "ModelInfo",
    "ModelSummary",
    "StreamChunk",
    "Tool",
    "ToolType",
    "Usage",
]

# Generic type for the client instance
ClientType = TypeVar("ClientType")


class BaseClient(ABC, Generic[ClientType]):
    """Abstract base class for all LLM provider clients.

    This defines the unified interface that provider-specific clients must
    implement to ensure consistent behavior across different backends.

    Attributes:
        api_key: API key for authentication.
        provider_kwargs: Additional provider-specific options.
        created_at: Timestamp when the client was instantiated.
    """

    def __init__(
        self,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize common client settings.

        Args:
            api_key: API key for authentication.
            **kwargs: Additional provider-specific options. Invalid options
                     for this provider will be automatically filtered out.
        """
        self.api_key = api_key
        # Filter kwargs before storing to avoid issues
        self.provider_kwargs = self._filter_init_kwargs(kwargs)
        self.created_at = datetime.now()
        self._request_count = 0
        self._last_request_time = None
        self._error_count = 0

        # Initialize with filtered kwargs
        self._client: ClientType = self._init_client(**self.provider_kwargs)
        self._async_client: ClientType | None = None
        self._capabilities = self._get_capabilities()

    def _filter_init_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Filter kwargs for the _init_client method.

        This method inspects the _init_client signature and only passes
        kwargs that the method can accept.

        Args:
            kwargs: Original kwargs to filter

        Returns:
            Filtered kwargs suitable for _init_client
        """
        return self._filter_kwargs(self._init_client, kwargs)

    # ====================================================================
    # Core abstract methods (must be implemented by provider-specific clients)
    # ====================================================================

    @abstractmethod
    def _init_client(self, **kwargs: Any) -> ClientType:
        """Initialize the provider-specific SDK or connection.

        Args:
            **kwargs: Provider-specific initialization options.

        Returns:
            An instance of the provider's client.
        """
        pass

    @abstractmethod
    def _get_capabilities(self) -> Capability:
        """Describe which features this provider supports.

        Returns:
            Capability object indicating supported features.
        """
        pass

    @abstractmethod
    def list_models(self) -> list[ModelSummary]:
        """List available models.

        Retrieves a list of models available through this provider.

        Returns:
            List of ModelSummary objects containing model information.
        """
        pass

    # ====================================================================
    # Chat completion methods
    # ====================================================================

    def chat_completion(
        self,
        messages: Input,
        model: str,
        **kwargs: Any,
    ) -> CompletionResponse | Generator[StreamChunk, None, None]:
        """Synchronously generate a chat completion.

        Args:
            messages: Input for the completion - can be a string prompt, list of messages,
                  or dictionary of provider-specific parameters.
            model: Model identifier to use.
            **kwargs: Provider-specific parameters that will be passed directly to the underlying SDK.

        Returns:
            Either a full CompletionResponse or a stream of StreamChunks.
        """
        try:
            self._request_count += 1
            self._last_request_time = time.time()
            filtered_kwargs = self._filter_kwargs(self._chat_completion_impl, kwargs)
            return self._chat_completion_impl(messages, model, **filtered_kwargs)
        except Exception:
            self._error_count += 1
            raise

    async def achat_completion(
        self,
        messages: Input,
        model: str,
        **kwargs: Any,
    ) -> CompletionResponse | AsyncGenerator[StreamChunk, None]:
        """Asynchronously generate a chat completion.

        Args:
            messages: Input for the completion - can be a string prompt, list of messages,
                  or dictionary of provider-specific parameters.
            model: Model identifier to use.
            **kwargs: Provider-specific parameters that will be passed directly to the underlying SDK.

        Returns:
            Either a full CompletionResponse or a stream of StreamChunks.
        """
        try:
            self._request_count += 1
            self._last_request_time = time.time()
            filtered_kwargs = self._filter_kwargs(self._achat_completion_impl, kwargs)
            return await self._achat_completion_impl(messages, model, **filtered_kwargs)
        except Exception:
            self._error_count += 1
            raise

    @abstractmethod
    def _chat_completion_impl(
        self,
        messages: Input,
        model: str,
        **kwargs: Any,
    ) -> CompletionResponse | Generator[StreamChunk, None, None]:
        """Provider-specific implementation of synchronous chat completion.

        This method should be implemented by provider-specific subclasses.

        Args:
            messages: Input for the completion.
            model: Model identifier to use.
            **kwargs: Provider-specific parameters.

        Returns:
            Either a full CompletionResponse or a stream of StreamChunks.
        """
        pass

    @abstractmethod
    async def _achat_completion_impl(
        self,
        messages: Input,
        model: str,
        **kwargs: Any,
    ) -> CompletionResponse | AsyncGenerator[StreamChunk, None]:
        """Provider-specific implementation of asynchronous chat completion.

        This method should be implemented by provider-specific subclasses.

        Args:
            messages: Input for the completion.
            model: Model identifier to use.
            **kwargs: Provider-specific parameters.

        Returns:
            Either a full CompletionResponse or a stream of StreamChunks.
        """
        pass

    # ====================================================================
    # Model information methods
    # ====================================================================

    def get_model_info(self, model_id: str) -> ModelInfo:
        """Get information about a specific model.

        Retrieves detailed information about a specific model.

        Args:
            model_id: ID or name of the model to get information for.

        Returns:
            ModelInfo object containing detailed model information.

        Raises:
            ValueError: If model is not found.
        """
        models = self.list_models()
        for model in models:
            if model.id == model_id or model.name == model_id:
                return ModelInfo(
                    id=model.id,
                    name=model.name,
                    description=model.description,
                )
        raise ValueError(f"Model {model_id} not found")

    # ====================================================================
    # Multimodal methods
    # ====================================================================

    def process_media(
        self, media_type: MediaType, content: bytes | str, **kwargs: Any
    ) -> MediaProcessingResult:
        """Convert raw media into provider-specific format.

        Args:
            media_type: Type of media being provided.
            content: Raw bytes or a URI string.
            **kwargs: Provider-specific media options.

        Returns:
            Processed media descriptor.

        Raises:
            NotImplementedError: If multimodal is not supported.
        """
        if not self.supports_multimodal():
            raise NotImplementedError("Multimodal not supported.")

        try:
            self._request_count += 1
            self._last_request_time = time.time()
            return self._process_media(media_type, content, **kwargs)
        except Exception:
            self._error_count += 1
            raise

    def _process_media(
        self, media_type: MediaType, content: bytes | str, **kwargs: Any
    ) -> MediaProcessingResult:
        """Override in subclasses that support multimodal.

        Provider-specific implementation of media processing.

        Args:
            media_type: Type of media being processed.
            content: Media content as bytes or string.
            **kwargs: Additional provider-specific arguments.

        Returns:
            MediaProcessingResult with provider-specific media representation.

        Raises:
            NotImplementedError: By default, as this needs to be implemented by subclasses.
        """
        raise NotImplementedError("Multimodal processing not implemented")

    # ====================================================================
    # Tool and function calling methods
    # ====================================================================

    def register_tool(self, tool: Tool) -> None:
        """Register a tool for use in conversations.

        Makes a tool available for use in later conversations.

        Args:
            tool: Tool definition to register.

        Raises:
            NotImplementedError: If provider does not support tools.
            ToolRegistrationError: If the tool cannot be registered.
        """
        if not self.supports_tools():
            raise NotImplementedError("Provider does not support tools")

        try:
            self._request_count += 1
            self._last_request_time = time.time()
            self._register_tool(tool)
        except Exception as e:
            self._error_count += 1
            raise ToolRegistrationError(tool.name, f"Provider error: {e!s}") from e

    def _register_tool(self, tool: Tool) -> None:
        """Override in subclasses that support tools.

        Provider-specific implementation of tool registration.

        Args:
            tool: Tool definition to register.

        Raises:
            NotImplementedError: By default, as this needs to be implemented by subclasses.
        """
        raise NotImplementedError("Tool registration not implemented")

    # ====================================================================
    # Agent methods
    # ====================================================================

    def create_agent(
        self, name: str, instructions: str, tools: list[Tool] | None = None, **kwargs: Any
    ) -> AgentResponse:
        """Create an agent.

        Creates a persistent agent with specific instructions and capabilities.

        Args:
            name: Name of the agent.
            instructions: Instructions for the agent.
            tools: Optional list of tools available to the agent.
            **kwargs: Additional provider-specific arguments.

        Returns:
            AgentResponse with provider-specific agent representation.

        Raises:
            NotImplementedError: If provider does not support agents.
            ProviderError: If there is an error creating the agent.
        """
        if not self.supports_agents():
            raise NotImplementedError("Provider does not support agents")

        try:
            self._request_count += 1
            self._last_request_time = time.time()
            return self._create_agent(name, instructions, tools, **kwargs)
        except Exception as e:
            self._error_count += 1
            if hasattr(self, "_provider_name"):
                provider_name = self._provider_name
            else:
                provider_name = self.__class__.__name__
            raise ProviderError(
                provider=provider_name, response_text=f"Failed to create agent: {e!s}"
            ) from e

    def _create_agent(
        self, name: str, instructions: str, tools: list[Tool] | None = None, **kwargs: Any
    ) -> AgentResponse:
        """Override in subclasses that support agents.

        Provider-specific implementation of agent creation.

        Args:
            name: Name of the agent.
            instructions: Instructions for the agent.
            tools: Optional list of tools available to the agent.
            **kwargs: Additional provider-specific arguments.

        Returns:
            AgentResponse with provider-specific agent representation.

        Raises:
            NotImplementedError: By default, as this needs to be implemented by subclasses.
        """
        raise NotImplementedError("Agent creation not implemented")

    # ====================================================================
    # File/Document processing methods
    # ====================================================================

    def upload_file(self, file_path: str, purpose: str = "assistants") -> FileUploadResponse:
        """Upload a file to the provider.

        Uploads a file to the provider's storage for use in conversations.

        Args:
            file_path: Path to the file to upload.
            purpose: Purpose of the file upload (e.g., "assistants", "fine-tuning").

        Returns:
            FileUploadResponse with provider-specific file information.

        Raises:
            NotImplementedError: If provider does not support file uploads.
            ProviderError: If there is an error uploading the file.
        """
        if not self.supports_files():
            raise NotImplementedError("Provider does not support file uploads")

        try:
            self._request_count += 1
            self._last_request_time = time.time()
            return self._upload_file(file_path, purpose)
        except Exception as e:
            self._error_count += 1
            if hasattr(self, "_provider_name"):
                provider_name = self._provider_name
            else:
                provider_name = self.__class__.__name__
            raise ProviderError(
                provider=provider_name,
                response_text=f"Failed to upload file '{file_path}': {e!s}",
                endpoint="file_upload",
            ) from e

    def _upload_file(self, file_path: str, purpose: str) -> FileUploadResponse:
        """Override in subclasses that support file uploads.

        Provider-specific implementation of file upload.

        Args:
            file_path: Path to the file to upload.
            purpose: Purpose of the file upload.

        Returns:
            FileUploadResponse with provider-specific file information.

        Raises:
            NotImplementedError: By default, as this needs to be implemented by subclasses.
        """
        raise NotImplementedError("File upload not implemented")

    # ====================================================================
    # Capability check methods
    # ====================================================================

    def supports_multimodal(self) -> bool:
        """Check if the provider supports image/audio/video inputs.

        Returns:
            True if multimodal inputs are supported, False otherwise.
        """
        return self._capabilities.multimodal

    def supports_tools(self) -> bool:
        """Check if the provider supports tool/function calling.

        Returns:
            True if tool/function calling is supported, False otherwise.
        """
        return self._capabilities.tools

    def supports_agents(self) -> bool:
        """Check if the provider supports agent creation.

        Returns:
            True if agent creation is supported, False otherwise.
        """
        return self._capabilities.agents

    def supports_files(self) -> bool:
        """Check if the provider supports file uploads.

        Returns:
            True if file uploads are supported, False otherwise.
        """
        return self._capabilities.files

    def supports_streaming(self) -> bool:
        """Check if the provider supports streaming responses.

        Returns:
            True if streaming responses are supported, False otherwise.
        """
        return self._capabilities.streaming

    # ====================================================================
    # Utility methods and properties
    # ====================================================================

    @staticmethod
    def _filter_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Filter keyword arguments to only include those valid for a specific function.

        This helper ensures only relevant arguments are passed to the provider SDK methods.
        It inspects the function signature and only passes kwargs that the function can accept.

        Args:
            func: The function whose signature will be used to filter kwargs.
            kwargs: Original keyword arguments to filter.

        Returns:
            A filtered dictionary containing only the kwargs that are valid for the specified function.
        """
        try:
            sig = inspect.signature(func)
            # Create a set of parameter names that aren't positional-only
            param_names = {
                name
                for name, param in sig.parameters.items()
                if param.kind not in (inspect.Parameter.POSITIONAL_ONLY,)
            }

            # Include **kwargs-style parameters by checking for VAR_KEYWORD
            has_var_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
            )

            if has_var_kwargs:
                # Function accepts arbitrary kwargs, so return all kwargs
                return kwargs.copy()

            # Filter kwargs to only include those in the function signature
            return {k: v for k, v in kwargs.items() if k in param_names}
        except (ValueError, TypeError, AttributeError):
            # If we can't inspect the signature, return empty dict
            return {}

    @property
    def capabilities(self) -> Capability:
        """Get supported capabilities and features of the provider.

        Returns:
            Capability object with supported features.
        """
        return self._capabilities

    @property
    def client(self) -> ClientType:
        """Get the underlying provider client instance.

        This is the actual client used to make API requests.

        Returns:
            The provider-specific client instance.
        """
        return self._client

    @property
    def request_count(self) -> int:
        """Get the number of API requests made by this client instance.

        Returns:
            The total number of API requests made through this client.
        """
        return self._request_count

    @property
    def error_count(self) -> int:
        """Get the number of errors encountered by this client instance.

        Returns:
            The total number of errors encountered during API requests.
        """
        return self._error_count

    @property
    def last_request_time(self) -> float | None:
        """Get the timestamp of the last API request made.

        Returns:
            Unix timestamp of the last request, or None if no requests made.
        """
        return self._last_request_time

    @property
    def client_age(self) -> float:
        """Get the age of this client in seconds.

        Returns:
            Number of seconds since this client was instantiated.
        """
        return (datetime.now() - self.created_at).total_seconds()

    # ====================================================================
    # Context manager methods
    # ====================================================================

    def __enter__(self) -> "BaseClient[ClientType]":
        """Context manager entry.

        Returns:
            Self for use in context manager.
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit.

        Ensures proper cleanup of resources when exiting a context manager.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        if hasattr(self._client, "close"):
            self._client.close()

    async def __aenter__(self) -> "BaseClient[ClientType]":
        """Async context manager entry.

        Returns:
            Self for use in async context manager.
        """
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit.

        Ensures proper cleanup of async resources when exiting an async context manager.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.
        """
        if self._async_client is not None:
            # Try async close if available
            aclose_method = getattr(self._async_client, "aclose", None)
            if callable(aclose_method):
                with contextlib.suppress(Exception):
                    result = aclose_method()
                    # Check if the result is a coroutine object that can be awaited
                    if inspect.iscoroutine(result):
                        await result

            # Fallback to synchronous close if async close is not available
            close_method = getattr(self._async_client, "close", None)
            if callable(close_method):
                with contextlib.suppress(Exception):
                    close_method()

    # ====================================================================
    # String representation
    # ====================================================================

    def __repr__(self) -> str:
        """Return a string representation of the client.

        Returns:
            A string with the class name and key properties.
        """
        return (
            f"{self.__class__.__name__}("
            f"capabilities={self._capabilities}, "
            f"requests={self._request_count}, "
            f"errors={self._error_count})"
        )

    def __str__(self) -> str:
        """Return a human-readable string representation of the client.

        Returns:
            A formatted string with client details.
        """
        return (
            f"{self.__class__.__name__} Client\n"
            f"- Created: {self.created_at}\n"
            f"- Requests: {self._request_count}\n"
            f"- Errors: {self._error_count}\n"
            f"- Capabilities: {self._capabilities}"
        )
