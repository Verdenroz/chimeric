from typing import Any

__all__ = [
    "AuthenticationError",
    "ChimericError",
    "ConfigurationError",
    "ModelNotSupportedError",
    "ProviderError",
    "ProviderNotFoundError",
    "RateLimitError",
    "ToolRegistrationError",
    "ValidationError",
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


class AuthenticationError(ChimericError):
    """Raised when authentication fails with a provider.

    This error is raised when API credentials are invalid, expired,
    or insufficient for the requested operation.
    """

    def __init__(self, provider: str, reason: str | None = None) -> None:
        """Initialize the error.

        Args:
            provider: Name of the provider where authentication failed
            reason: Optional reason for the authentication failure
        """
        message = f"Authentication failed for provider '{provider}'"
        if reason:
            message += f": {reason}"

        details = {
            "provider": provider,
            "reason": reason,
        }

        super().__init__(message, details)
        self.provider = provider
        self.reason = reason


class RateLimitError(ChimericError):
    """Raised when rate limits are exceeded.

    This error is raised when the provider's API rate limits have been
    reached and requests are being throttled or rejected.
    """

    def __init__(
        self, provider: str, retry_after: int | None = None, limit_type: str | None = None
    ) -> None:
        """Initialize the error.

        Args:
            provider: Name of the provider where rate limit was exceeded
            retry_after: Seconds to wait before retrying
            limit_type: Type of rate limit that was exceeded
        """
        message = f"Rate limit exceeded for provider '{provider}'"
        if limit_type:
            message += f" ({limit_type})"
        if retry_after:
            message += f". Retry after {retry_after} seconds"

        details = {
            "provider": provider,
            "retry_after": retry_after,
            "limit_type": limit_type,
        }

        super().__init__(message, details)
        self.provider = provider
        self.retry_after = retry_after
        self.limit_type = limit_type


class ProviderError(ChimericError):
    """Raised when an API call fails.

    This is a general error for API-related failures that don't fall
    into more specific categories like authentication or rate limiting.
    """

    def __init__(
        self,
        provider: str,
        status_code: int | None = None,
        response_text: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        """Initialize the error.

        Args:
            provider: Name of the provider where the API call failed
            status_code: HTTP status code returned by the API
            response_text: Response text from the failed API call
            endpoint: API endpoint that failed
        """
        message = f"API call failed for provider '{provider}'"
        if status_code:
            message += f" (status: {status_code})"
        if endpoint:
            message += f" at endpoint '{endpoint}'"

        details = {
            "provider": provider,
            "status_code": status_code,
            "response_text": response_text,
            "endpoint": endpoint,
        }

        super().__init__(message, details)
        self.provider = provider
        self.status_code = status_code
        self.response_text = response_text
        self.endpoint = endpoint


class ConfigurationError(ChimericError):
    """Raised when there's an issue with client configuration.

    This error is raised when the client is misconfigured or when
    required configuration parameters are missing or invalid.
    """

    def __init__(
        self, parameter: str | None = None, value: Any = None, expected: str | None = None
    ) -> None:
        """Initialize the error.

        Args:
            parameter: Name of the configuration parameter that's invalid
            value: The invalid value that was provided
            expected: Description of what was expected
        """
        if parameter:
            message = f"Invalid configuration for parameter '{parameter}'"
            if expected:
                message += f". Expected: {expected}"
            if value is not None:
                message += f". Got: {value}"
        else:
            message = "Configuration error"

        details = {
            "parameter": parameter,
            "value": value,
            "expected": expected,
        }

        super().__init__(message, details)
        self.parameter = parameter
        self.value = value
        self.expected = expected


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


class ValidationError(ChimericError):
    """Raised when input validation fails.

    This error is raised when function parameters, model inputs,
    or other data fail validation checks.
    """

    def __init__(
        self, field: str | None = None, value: Any = None, constraint: str | None = None
    ) -> None:
        """Initialize the error.

        Args:
            field: Name of the field that failed validation
            value: The invalid value
            constraint: Description of the validation constraint that was violated
        """
        if field:
            message = f"Validation failed for field '{field}'"
            if constraint:
                message += f": {constraint}"
            if value is not None:
                message += f" (value: {value})"
        else:
            message = "Validation error"
            if constraint:
                message += f": {constraint}"

        details = {
            "field": field,
            "value": value,
            "constraint": constraint,
        }

        super().__init__(message, details)
        self.field = field
        self.value = value
        self.constraint = constraint
