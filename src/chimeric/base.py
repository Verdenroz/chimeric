from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from enum import Enum
from typing import Any


class MediaType(Enum):
    """Supported media types for multimodal inputs."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"


class ToolType(Enum):
    """Supported tool types."""

    FUNCTION = "function"
    CODE_INTERPRETER = "code_interpreter"
    FILE_SEARCH = "file_search"
    WEB_SEARCH = "web_search"


@dataclass
class Message:
    """Unified message format across all providers."""

    role: str
    content: str | list[dict[str, Any]]
    metadata: dict[str, Any] | None = None


@dataclass
class Tool:
    """Unified tool definition."""

    type: ToolType
    name: str
    description: str
    parameters: dict[str, Any] | None = None
    function: callable | None = None


@dataclass
class CompletionResponse:
    """Unified response format."""

    content: str
    usage: dict[str, int] | None = None
    model: str | None = None
    finish_reason: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class StreamChunk:
    """Unified streaming chunk format."""

    content: str
    delta: str | None = None
    finish_reason: str | None = None
    metadata: dict[str, Any] | None = None


class BaseClient(ABC):
    """Abstract base class for all LLM provider clients.

    This class defines the unified interface that all provider-specific
    clients must implement, ensuring consistency across different LLM providers.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        **kwargs,
    ):
        """Initialize the base client.

        Args:
            api_key: API key for authentication
            base_url: Base URL for API requests
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            **kwargs: Provider-specific arguments passed to underlying client
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.provider_kwargs = kwargs

        # Initialize provider-specific client
        self._client = self._init_client(**kwargs)
        self._async_client = None

        # Initialize capabilities
        self._capabilities = self._get_capabilities()

    @abstractmethod
    def _init_client(self, **kwargs) -> Any:
        """Initialize the provider-specific client."""
        pass

    @abstractmethod
    def _get_capabilities(self) -> dict[str, bool]:
        """Return dict of supported capabilities for this provider."""
        pass

    # Core chat completion methods
    @abstractmethod
    def chat_completion(
        self,
        messages: list[Message],
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> CompletionResponse | Generator[StreamChunk, None, None]:
        """Generate chat completion."""
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
        **kwargs,
    ) -> CompletionResponse | AsyncGenerator[StreamChunk, None]:
        """Generate async chat completion."""
        pass

    # Multimodal methods (optional implementation)
    def supports_multimodal(self) -> bool:
        """Check if provider supports multimodal inputs."""
        return self._capabilities.get("multimodal", False)

    def process_media(
        self, media_type: MediaType, content: bytes | str, **kwargs
    ) -> dict[str, Any]:
        """Process media content for provider-specific format."""
        if not self.supports_multimodal():
            raise NotImplementedError("Provider does not support multimodal inputs")
        return self._process_media(media_type, content, **kwargs)

    def _process_media(
        self, media_type: MediaType, content: bytes | str, **kwargs
    ) -> dict[str, Any]:
        """Override in subclasses that support multimodal."""
        raise NotImplementedError("Multimodal processing not implemented")

    # Tool/Function calling methods
    def supports_tools(self) -> bool:
        """Check if provider supports tool/function calling."""
        return self._capabilities.get("tools", False)

    def register_tool(self, tool: Tool) -> None:
        """Register a tool for use in conversations."""
        if not self.supports_tools():
            raise NotImplementedError("Provider does not support tools")
        self._register_tool(tool)

    def _register_tool(self, tool: Tool) -> None:
        """Override in subclasses that support tools."""
        raise NotImplementedError("Tool registration not implemented")

    # Agent creation methods
    def supports_agents(self) -> bool:
        """Check if provider supports agent creation."""
        return self._capabilities.get("agents", False)

    def create_agent(
        self, name: str, instructions: str, tools: list[Tool] | None = None, **kwargs
    ) -> Any:
        """Create an agent."""
        if not self.supports_agents():
            raise NotImplementedError("Provider does not support agents")
        return self._create_agent(name, instructions, tools, **kwargs)

    def _create_agent(
        self, name: str, instructions: str, tools: list[Tool] | None = None, **kwargs
    ) -> Any:
        """Override in subclasses that support agents."""
        raise NotImplementedError("Agent creation not implemented")

    # File/Document processing
    def supports_files(self) -> bool:
        """Check if provider supports file uploads."""
        return self._capabilities.get("files", False)

    def upload_file(self, file_path: str, purpose: str = "assistants") -> dict[str, Any]:
        """Upload a file to the provider."""
        if not self.supports_files():
            raise NotImplementedError("Provider does not support file uploads")
        return self._upload_file(file_path, purpose)

    def _upload_file(self, file_path: str, purpose: str) -> dict[str, Any]:
        """Override in subclasses that support file uploads."""
        raise NotImplementedError("File upload not implemented")

    # Model listing and info
    @abstractmethod
    def list_models(self) -> list[dict[str, Any]]:
        """List available models."""
        pass

    def get_model_info(self, model_id: str) -> dict[str, Any]:
        """Get information about a specific model."""
        models = self.list_models()
        for model in models:
            if model.get("id") == model_id or model.get("name") == model_id:
                return model
        raise ValueError(f"Model {model_id} not found")

    # Utility methods
    def get_capabilities(self) -> dict[str, bool]:
        """Return dict of all capabilities supported by this provider."""
        return self._capabilities.copy()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if hasattr(self._client, "close"):
            self._client.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._async_client and hasattr(self._async_client, "aclose"):
            await self._async_client.aclose()
