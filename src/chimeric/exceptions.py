from typing import Any

__all__ = [
    "ChimericError",
    "ModelNotSupportedError",
    "ProviderError",
    "ProviderNotFoundError",
    "ToolRegistrationError",
]


class ChimericError(Exception):
    """Base exception for all Chimeric-related errors.

    All custom exceptions in the Chimeric library are inherited from this base class
    to allow for easier exception handling and identification.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message
            details: Optional additional details about the error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return the error message."""
        return self.message

    def __repr__(self) -> str:
        """Return a detailed representation of the error."""
        if self.details:
            return f"{self.__class__.__name__}(message='{self.message}', details={self.details})"
        return f"{self.__class__.__name__}(message='{self.message}')"


class ProviderNotFoundError(ChimericError):
    """Raised when a requested provider is not available or configured.

    This error is raised when trying to use a provider that hasn't been
    configured or is not supported by the current installation.
    """

    def __init__(self, provider: str, available_providers: list[str] | None = None) -> None:
        """Initialize the error.

        Args:
            provider: Name of the provider that was not found
            available_providers: List of available providers
        """
        message = f"Provider '{provider}' not found or configured"
        if available_providers:
            message += f". Available providers: {', '.join(available_providers)}"

        details = {
            "requested_provider": provider,
            "available_providers": available_providers or [],
        }

        super().__init__(message, details)
        self.provider = provider
        self.available_providers = available_providers or []


class ModelNotSupportedError(ChimericError):
    """Raised when a model is not supported by any configured provider.

    This error is raised when trying to use a model that is not available
    through any of the configured providers.
    """

    def __init__(
        self, model: str, provider: str | None = None, supported_models: list[str] | None = None
    ) -> None:
        """Initialize the error.

        Args:
            model: Name of the unsupported model
            provider: Provider where the model was expected
            supported_models: List of supported models
        """
        message = f"Model '{model}' is not supported"
        if provider:
            message += f" by provider '{provider}'"

        details = {
            "requested_model": model,
            "provider": provider,
            "supported_models": supported_models or [],
        }

        super().__init__(message, details)
        self.model = model
        self.provider = provider
        self.supported_models = supported_models or []


class ProviderError(ChimericError):
    """Raised when a provider operation fails.

    This is a flexible error class that can wrap any provider-specific error
    while maintaining a consistent interface across all providers.
    """

    def __init__(
        self,
        provider: str,
        message: str | None = None,
        error: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the error.

        Args:
            provider: Name of the provider where the operation failed
            message: Custom error message. If not provided, will be derived from error
            error: The original provider-specific exception that caused this error
            **kwargs: Additional provider-specific error details
        """
        # Generate message if not provided
        if message is None:
            if error:
                message = f"Provider '{provider}' failed: {error!s}"
            else:
                message = f"Provider '{provider}' failed"
        else:
            message = f"Provider '{provider}': {message}"

        # Build details dictionary with all provided information
        details = {
            "provider": provider,
            "error": str(error) if error else None,
            "error_type": type(error).__name__ if error else None,
            **kwargs,
        }

        super().__init__(message, details)
        self.provider = provider
        self.error = error
        self.extra_details = kwargs


class ToolRegistrationError(ChimericError):
    """Raised when there's an error registering a tool.

    This error is raised when a tool cannot be registered due to
    invalid parameters, naming conflicts, or other registration issues.
    """

    def __init__(
        self, tool_name: str, reason: str | None = None, existing_tool: bool = False
    ) -> None:
        """Initialize the error.

        Args:
            tool_name: Name of the tool that failed to register
            reason: Reason for the registration failure
            existing_tool: Whether the error is due to a tool with the same name existing
        """
        message = f"Failed to register tool '{tool_name}'"
        if existing_tool:
            message += ": tool with this name already exists"
        elif reason:
            message += f": {reason}"

        details = {
            "tool_name": tool_name,
            "reason": reason,
            "existing_tool": existing_tool,
        }

        super().__init__(message, details)
        self.tool_name = tool_name
        self.reason = reason
        self.existing_tool = existing_tool
