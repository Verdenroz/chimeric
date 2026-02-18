"""Chimeric — unified LLM interface for multiple providers."""

from .chimeric import Chimeric
from .exceptions import (
    AuthenticationError,
    ChimericError,
    ModelNotSupportedError,
    ProviderError,
    ProviderNotFoundError,
    RateLimitError,
    StructuredOutputError,
    ToolRegistrationError,
)
from .types import (
    CompletionResponse,
    Input,
    Message,
    Metadata,
    ModelSummary,
    Provider,
    StreamChunk,
    Tool,
    ToolCall,
    Tools,
    Usage,
)

__all__ = [
    # Primary interface
    "Chimeric",
    # Exceptions
    "AuthenticationError",
    "ChimericError",
    "ModelNotSupportedError",
    "ProviderError",
    "ProviderNotFoundError",
    "RateLimitError",
    "StructuredOutputError",
    "ToolRegistrationError",
    # Response types
    "CompletionResponse",
    "StreamChunk",
    "Usage",
    # Input / routing types
    "Provider",
    "Input",
    "Message",
    "Metadata",
    "Tool",
    "ToolCall",
    "Tools",
    # Model discovery
    "ModelSummary",
]

__version__ = "0.2.0"
