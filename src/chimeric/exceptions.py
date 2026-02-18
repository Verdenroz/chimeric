"""Chimeric exception hierarchy."""

from __future__ import annotations

from typing import Any

__all__ = [
    "AuthenticationError",
    "ChimericError",
    "ModelNotSupportedError",
    "ProviderError",
    "ProviderNotFoundError",
    "RateLimitError",
    "StructuredOutputError",
    "ToolRegistrationError",
]


class ChimericError(Exception):
    """Base class for all Chimeric errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return the error message."""
        return self.message

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        if self.details:
            return f"{self.__class__.__name__}(message='{self.message}', details={self.details})"
        return f"{self.__class__.__name__}(message='{self.message}')"


class ProviderNotFoundError(ChimericError):
    """Raised when a requested provider is not configured or recognised."""

    def __init__(self, provider: str, available_providers: list[str] | None = None) -> None:
        message = f"Provider '{provider}' not found or configured"
        if available_providers:
            message += f". Available providers: {', '.join(available_providers)}"
        details: dict[str, Any] = {
            "requested_provider": provider,
            "available_providers": available_providers or [],
        }
        super().__init__(message, details)
        self.provider = provider
        self.available_providers = available_providers or []


class ModelNotSupportedError(ChimericError):
    """Raised when a model is not available from any configured provider."""

    def __init__(
        self,
        model: str,
        provider: str | None = None,
        supported_models: list[str] | None = None,
    ) -> None:
        message = f"Model '{model}' is not supported"
        if provider:
            message += f" by provider '{provider}'"
        if supported_models:
            message += f". Supported models: {', '.join(supported_models)}"
        details: dict[str, Any] = {
            "requested_model": model,
            "provider": provider,
            "supported_models": supported_models or [],
        }
        super().__init__(message, details)
        self.model = model
        self.provider = provider
        self.supported_models = supported_models or []


class ProviderError(ChimericError):
    """Raised when a provider API call fails (4xx/5xx other than 401/429)."""

    def __init__(
        self,
        error: Exception | None,
        provider: str,
        message: str | None = None,
        status_code: int | None = None,
        **kwargs: Any,
    ) -> None:
        if message is None:
            msg = f"Provider '{provider}' failed: {error!s}"
        else:
            msg = f"Provider '{provider}': {message}"

        details: dict[str, Any] = {
            "provider": provider,
            "error": str(error) if error else None,
            "error_type": type(error).__name__ if error else None,
            "status_code": status_code,
            **kwargs,
        }
        super().__init__(msg, details)
        self.provider = provider
        self.error = error
        self.status_code = status_code
        self.extra_details = kwargs


class AuthenticationError(ChimericError):
    """Raised when the provider returns HTTP 401 (invalid or missing API key)."""

    def __init__(self, provider: str, message: str | None = None) -> None:
        msg = f"Authentication failed for provider '{provider}'"
        if message:
            msg += f": {message}"
        details: dict[str, Any] = {"provider": provider, "status_code": 401}
        super().__init__(msg, details)
        self.provider = provider


class RateLimitError(ChimericError):
    """Raised when the provider returns HTTP 429 (rate limit exceeded)."""

    def __init__(self, provider: str, message: str | None = None) -> None:
        msg = f"Rate limit exceeded for provider '{provider}'"
        if message:
            msg += f": {message}"
        details: dict[str, Any] = {"provider": provider, "status_code": 429}
        super().__init__(msg, details)
        self.provider = provider


class StructuredOutputError(ChimericError):
    """Raised when a model response cannot be parsed into the requested structured output.

    Attributes:
        model_name: Name of the Pydantic model that parsing was attempted against.
        reason: Description of why parsing failed.
        raw_content: The raw response content that failed to parse.
    """

    def __init__(
        self,
        model_name: str,
        reason: str,
        raw_content: str | None = None,
    ) -> None:
        message = f"Structured output parsing failed for '{model_name}': {reason}"
        details: dict[str, Any] = {
            "model_name": model_name,
            "reason": reason,
            "raw_content": raw_content,
        }
        super().__init__(message, details)
        self.model_name = model_name
        self.reason = reason
        self.raw_content = raw_content


class ToolRegistrationError(ChimericError):
    """Raised when tool registration fails (bad signature, name conflict, etc.)."""

    def __init__(
        self,
        tool_name: str,
        reason: str | None = None,
        existing_tool: bool = False,
    ) -> None:
        message = f"Failed to register tool '{tool_name}'"
        if existing_tool:
            message += ": tool with this name already exists"
        elif reason:
            message += f": {reason}"
        details: dict[str, Any] = {
            "tool_name": tool_name,
            "reason": reason,
            "existing_tool": existing_tool,
        }
        super().__init__(message, details)
        self.tool_name = tool_name
        self.reason = reason
        self.existing_tool = existing_tool
