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
    EmbeddingResponse,
    EmbeddingUsage,
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
    "AuthenticationError",
    "Chimeric",
    "ChimericError",
    "CompletionResponse",
    "EmbeddingResponse",
    "EmbeddingUsage",
    "Input",
    "Message",
    "Metadata",
    "ModelNotSupportedError",
    "ModelSummary",
    "Provider",
    "ProviderError",
    "ProviderNotFoundError",
    "RateLimitError",
    "StreamChunk",
    "StructuredOutputError",
    "Tool",
    "ToolCall",
    "ToolRegistrationError",
    "Tools",
    "Usage",
]

__version__ = "0.3.0"
