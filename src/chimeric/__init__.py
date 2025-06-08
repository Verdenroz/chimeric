from .base import BaseClient, CompletionResponse, MediaType, Message, StreamChunk, Tool, ToolType
from .client import Chimeric
from .exceptions import ChimericError, ProviderNotFoundError

# Main exports
__all__ = [
    # Main client
    "Chimeric",
    # Base classes
    "BaseClient",
    # Data types
    "Message",
    "Tool",
    "CompletionResponse",
    "StreamChunk",
    # Enums
    "MediaType",
    "ToolType",
    # Exceptions
    "ChimericError",
    "ProviderNotFoundError",
]

__version__ = "0.1.0"
