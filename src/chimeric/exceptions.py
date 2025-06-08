class ChimericError(Exception):
    """Base exception for all Chimeric-related errors."""

    pass


class ProviderNotFoundError(ChimericError):
    """Raised when a requested provider is not available or configured."""

    pass


class ModelNotSupportedError(ChimericError):
    """Raised when a model is not supported by any configured provider."""

    pass


class AuthenticationError(ChimericError):
    """Raised when authentication fails with a provider."""

    pass


class RateLimitError(ChimericError):
    """Raised when rate limits are exceeded."""

    pass


class APIError(ChimericError):
    """Raised when an API call fails."""

    pass


class ConfigurationError(ChimericError):
    """Raised when there's an issue with client configuration."""

    pass
