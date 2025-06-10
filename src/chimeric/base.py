from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Awaitable, Generator
import contextlib
from typing import Any, cast

from .types import (
    AgentResponse,
    Capability,
    CompletionResponse,
    FileUploadResponse,
    MediaProcessingResult,
    MediaType,
    Message,
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
    "MediaProcessingResult",
    "MediaType",
    "Message",
    "ModelInfo",
    "ModelSummary",
    "StreamChunk",
    "Tool",
    "ToolType",
    "Usage",
]


class BaseClient(ABC):
    """Abstract base class for all LLM provider clients.

    This defines the unified interface that provider-specific clients must
    implement to ensure consistent behavior across different backends.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize common client settings.

        Args:
            api_key: API key for authentication.
            base_url: Base endpoint for requests.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry count for failed requests.
            **kwargs: Additional provider-specific options.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.provider_kwargs = kwargs

        self._client = self._init_client(**kwargs)
        self._async_client: Any = None
        self._capabilities = self._get_capabilities()

    @abstractmethod
    def _init_client(self, **kwargs: Any) -> Any:
        """Initialize the provider-specific SDK or connection.

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
    def chat_completion(
        self,
        messages: list[Message],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse | Generator[StreamChunk, None, None]:
        """Synchronously generate a chat completion.

        Args:
            messages: Conversation history as Message objects.
            model: Model identifier to use.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.
            stream: Whether to emit partial results.
            tools: Tools available for function-calling.
            **kwargs: Provider-specific overrides.

        Returns:
            Either a full CompletionResponse or a stream of StreamChunks.
        """
        pass

    @abstractmethod
    async def achat_completion(
        self,
        messages: list[Message],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse | AsyncGenerator[StreamChunk, None]:
        """Asynchronously generate a chat completion.

        Args:
            messages: Conversation history as Message objects.
            model: Model identifier to use.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.
            stream: Whether to emit partial results.
            tools: Tools available for function-calling.
            **kwargs: Provider-specific overrides.

        Returns:
            Either a full CompletionResponse or a stream of StreamChunks.
        """
        pass

    def supports_multimodal(self) -> bool:
        """Return True if image/audio/video inputs are supported."""
        return self._capabilities.multimodal

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
        return self._process_media(media_type, content, **kwargs)

    def _process_media(
        self, media_type: MediaType, content: bytes | str, **kwargs: Any
    ) -> MediaProcessingResult:
        """Override in subclasses that support multimodal.

        Provider-specific implementation of media processing.

        Args:
            media_type: Type of media being processed
            content: Media content as bytes or string
            **kwargs: Additional provider-specific arguments

        Returns:
            MediaProcessingResult with provider-specific media representation

        Raises:
            NotImplementedError: By default, as this needs to be implemented by subclasses
        """
        raise NotImplementedError("Multimodal processing not implemented")

    # Tool/Function calling methods
    def supports_tools(self) -> bool:
        """Check if the provider supports tool/function calling.

        Returns:
            True if tool/function calling is supported, False otherwise
        """
        return self._capabilities.tools

    def register_tool(self, tool: Tool) -> None:
        """Register a tool for use in conversations.

        Makes a tool available for use in later conversations.

        Args:
            tool: Tool definition to register

        Raises:
            NotImplementedError: If provider does not support tools
        """
        if not self.supports_tools():
            raise NotImplementedError("Provider does not support tools")
        self._register_tool(tool)

    def _register_tool(self, tool: Tool) -> None:
        """Override in subclasses that support tools.

        Provider-specific implementation of tool registration.

        Args:
            tool: Tool definition to register

        Raises:
            NotImplementedError: By default, as this needs to be implemented by subclasses
        """
        raise NotImplementedError("Tool registration not implemented")

    # Agent creation methods
    def supports_agents(self) -> bool:
        """Check if the provider supports agent creation.

        Returns:
            True if agent creation is supported, False otherwise
        """
        return self._capabilities.agents

    def create_agent(
        self, name: str, instructions: str, tools: list[Tool] | None = None, **kwargs: Any
    ) -> AgentResponse:
        """Create an agent.

        Creates a persistent agent with specific instructions and capabilities.

        Args:
            name: Name of the agent
            instructions: Instructions for the agent
            tools: Optional list of tools available to the agent
            **kwargs: Additional provider-specific arguments

        Returns:
            AgentResponse with provider-specific agent representation

        Raises:
            NotImplementedError: If provider does not support agents
        """
        if not self.supports_agents():
            raise NotImplementedError("Provider does not support agents")
        return self._create_agent(name, instructions, tools, **kwargs)

    def _create_agent(
        self, name: str, instructions: str, tools: list[Tool] | None = None, **kwargs: Any
    ) -> AgentResponse:
        """Override in subclasses that support agents.

        Provider-specific implementation of agent creation.

        Args:
            name: Name of the agent
            instructions: Instructions for the agent
            tools: Optional list of tools available to the agent
            **kwargs: Additional provider-specific arguments

        Returns:
            AgentResponse with provider-specific agent representation

        Raises:
            NotImplementedError: By default, as this needs to be implemented by subclasses
        """
        raise NotImplementedError("Agent creation not implemented")

    # File/Document processing
    def supports_files(self) -> bool:
        """Check if the provider supports file uploads.

        Returns:
            True if file uploads are supported, False otherwise
        """
        return self._capabilities.files

    def upload_file(self, file_path: str, purpose: str = "assistants") -> FileUploadResponse:
        """Upload a file to the provider.

        Uploads a file to the provider's storage for use in conversations.

        Args:
            file_path: Path to the file to upload
            purpose: Purpose of the file upload (e.g., "assistants", "fine-tuning")

        Returns:
            FileUploadResponse with provider-specific file information

        Raises:
            NotImplementedError: If provider does not support file uploads
        """
        if not self.supports_files():
            raise NotImplementedError("Provider does not support file uploads")
        return self._upload_file(file_path, purpose)

    def _upload_file(self, file_path: str, purpose: str) -> FileUploadResponse:
        """Override in subclasses that support file uploads.

        Provider-specific implementation of file upload.

        Args:
            file_path: Path to the file to upload
            purpose: Purpose of the file upload

        Returns:
            FileUploadResponse with provider-specific file information

        Raises:
            NotImplementedError: By default, as this needs to be implemented by subclasses
        """
        raise NotImplementedError("File upload not implemented")

    # Model listing and info
    @abstractmethod
    def list_models(self) -> list[ModelSummary]:
        """List available models.

        Retrieves a list of models available through this provider.

        Returns:
            List of ModelSummary objects containing model information
        """
        pass

    def get_model_info(self, model_id: str) -> ModelInfo:
        """Get information about a specific model.

        Retrieves detailed information about a specific model.

        Args:
            model_id: ID or name of the model to get information for

        Returns:
            ModelInfo object containing detailed model information

        Raises:
            ValueError: If model is not found
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

    # Utility methods
    def get_capabilities(self) -> Capability:
        """Return capabilities supported by this provider.

        Returns:
            Capability object with supported features
        """
        return self._capabilities

    def __enter__(self) -> "BaseClient":
        """Context manager entry.

        Returns:
            Self for use in context manager
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit.

        Ensures proper cleanup of resources when exiting a context manager.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        if hasattr(self._client, "close"):
            self._client.close()

    async def __aenter__(self) -> "BaseClient":
        """Async context manager entry.

        Returns:
            Self for use in async context manager
        """
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit.

        Ensures proper cleanup of async resources when exiting an async context manager.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        if self._async_client is None:
            return

        # Try synchronous close first if available
        if hasattr(self._async_client, "close"):
            with contextlib.suppress(Exception):
                self._async_client.close()
            return

        # Try async close if available
        if not hasattr(self._async_client, "aclose"):
            return

        aclose_method = self._async_client.aclose
        if not callable(aclose_method):
            return

        with contextlib.suppress(Exception):
            result = aclose_method()
            if result is not None:
                awaitable = cast("Awaitable[None]", result)
                await awaitable

    def __repr__(self) -> str:
        """Return a string representation of the client."""
        return f"{self.__class__.__name__}(capabilities={self._capabilities})"
