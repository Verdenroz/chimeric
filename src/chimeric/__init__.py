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
from .tools import ToolManager
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
    ToolCallChunk,
    ToolExecutionResult,
    ToolParameters,
    Tools,
    Usage,
)

__all__ = [
    "AuthenticationError",
    "Chimeric",
    "ChimericError",
    "CompletionResponse",
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
    "ToolCallChunk",
    "ToolExecutionResult",
    "ToolManager",
    "ToolParameters",
    "ToolRegistrationError",
    "Tools",
    "Usage",
]

__version__ = "0.2.0"
