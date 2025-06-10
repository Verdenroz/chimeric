from .base import (
    BaseClient,
    Capability,
    CompletionResponse,
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
from .client import Chimeric, Provider
from .exceptions import (
    AuthenticationError,
    ChimericError,
    ConfigurationError,
    ModelNotSupportedError,
    ProviderError,
    ProviderNotFoundError,
    RateLimitError,
    ToolRegistrationError,
    ValidationError,
)
from .tools import ToolManager, ToolParameterMetadata, tool_parameter

__all__ = [
    # Base models
    "AuthenticationError",
    "BaseClient",
    "Capability",
    "Chimeric",
    "ChimericError",
    "CompletionResponse",
    "ConfigurationError",
    "MediaProcessingResult",
    "MediaType",
    "Message",
    "ModelInfo",
    "ModelNotSupportedError",
    "ModelSummary",
    "Provider",
    "ProviderError",
    "ProviderNotFoundError",
    "RateLimitError",
    "StreamChunk",
    "Tool",
    "ToolManager",
    "ToolParameterMetadata",
    "ToolRegistrationError",
    "ToolType",
    "Usage",
    "ValidationError",
    "tool_parameter",
]

__version__ = "0.1.0"
