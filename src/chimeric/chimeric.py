from collections.abc import AsyncGenerator, Callable, Generator
import os
from typing import Any

from .base import BaseClient
from .exceptions import ChimericError, ProviderError, ProviderNotFoundError
from .providers.cerebras.client import CerebrasClient
from .providers.cohere.client import CohereClient
from .providers.google.client import GoogleClient
from .providers.groq.client import GroqClient
from .providers.huggingface.client import HuggingFaceClient
from .providers.openai.client import OpenAIClient
from .providers.replicate.client import ReplicateClient
from .tools import ToolManager
from .types import (
    Capability,
    CompletionResponse,
    Input,
    ModelSummary,
    Provider,
    StreamChunk,
    Tool,
)

__all__ = [
    "PROVIDER_CLIENTS",
    "Chimeric",
    "Provider",
]

# Mapping of provider enums to their corresponding client classes.
PROVIDER_CLIENTS: dict[Provider, type[BaseClient[Any]]] = {
    Provider.OPENAI: OpenAIClient,
    Provider.GOOGLE: GoogleClient,
    Provider.CEREBRAS: CerebrasClient,
    Provider.COHERE: CohereClient,
    Provider.GROQ: GroqClient,
    Provider.HUGGINGFACE: HuggingFaceClient,
    Provider.REPLICATE: ReplicateClient,
}


class Chimeric:
    """Main Chimeric client with unified interface across all LLM providers.

    This class provides a single interface that can work with any supported
    LLM provider, with automatic provider detection and seamless switching.

    Examples:
        # Auto-detect from environment variables
        client = Chimeric()

        # Explicit API key configuration
        chimeric = Chimeric(
            openai_api_key="sk-...",
            anthropic_api_key="sk-ant-...",
            google_api_key="...",
            cerebras_api_key="...",
            cohere_api_key="...",
            groq_api_key="...",
            huggingface_api_key="...",
            replicate_api_token="...",
            aws_access_key_id="...",
            aws_secret_access_key="...",
            aws_region="us-east-1"
        )

        # Generate completions with automatic provider routing
        response = chimeric.generate(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
        )

        # Use specific provider
        response = chimeric.generate(
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": "Hello"}]
        )
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        google_api_key: str | None = None,
        cerebras_api_key: str | None = None,
        cohere_api_key: str | None = None,
        groq_api_key: str | None = None,
        huggingface_token: str | None = None,
        replicate_api_token: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the Chimeric client.

        Args:
            openai_api_key: OpenAI API key for authentication.
            anthropic_api_key: Anthropic API key for authentication.
            google_api_key: Google API key for authentication.
            cerebras_api_key: Cerebras API key for authentication.
            cohere_api_key: Cohere API key for authentication.
            groq_api_key: Groq API key for authentication.
            huggingface_token: HuggingFace token for authentication.
            replicate_api_token: Replicate API token for authentication.
            **kwargs: Additional provider-specific configuration options.
        """
        self.providers: dict[Provider, BaseClient[Any]] = {}
        self.primary_provider: Provider | None = None

        # Initialize providers from explicit API keys.
        self._initialize_providers_from_config(
            openai_api_key,
            anthropic_api_key,
            google_api_key,
            cerebras_api_key,
            cohere_api_key,
            groq_api_key,
            huggingface_token,
            replicate_api_token,
        )

        # Auto-detect providers from environment variables.
        self._detect_providers_from_environment(kwargs)

        # Initialize the tool management system.
        self._tool_manager = ToolManager()

    def _initialize_providers_from_config(
        self,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        google_api_key: str | None = None,
        cerebras_api_key: str | None = None,
        cohere_api_key: str | None = None,
        groq_api_key: str | None = None,
        huggingface_token: str | None = None,
        replicate_api_token: str | None = None,
    ) -> None:
        """Initializes providers from explicitly provided API keys.

        Args:
            openai_api_key: OpenAI API key.
            anthropic_api_key: Anthropic API key.
            google_api_key: Google API key.
            cerebras_api_key: Cerebras API key.
            cohere_api_key: Cohere API key.
            groq_api_key: Groq API key.
            huggingface_token: HuggingFace token.
            replicate_api_token: Replicate API token.
        """
        provider_configs: list[tuple[Provider, str | None]] = [
            (Provider.OPENAI, openai_api_key),
            (Provider.ANTHROPIC, anthropic_api_key),
            (Provider.GOOGLE, google_api_key),
            (Provider.CEREBRAS, cerebras_api_key),
            (Provider.COHERE, cohere_api_key),
            (Provider.GROQ, groq_api_key),
            (Provider.HUGGINGFACE, huggingface_token),
            (Provider.REPLICATE, replicate_api_token),
        ]

        # Initialize providers that have API keys provided.
        for provider, api_key in provider_configs:
            if api_key is not None:
                self._add_provider(provider, api_key=api_key)

    def _detect_providers_from_environment(self, kwargs: dict[str, Any]) -> None:
        """Auto-detects available providers from environment variables.

        Args:
            kwargs: Additional configuration options to pass to providers.
        """
        # Map providers to their possible environment variable names.
        env_variable_map: dict[Provider, list[str]] = {
            Provider.OPENAI: ["OPENAI_API_KEY"],
            Provider.GOOGLE: ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            Provider.ANTHROPIC: ["ANTHROPIC_API_KEY"],
            Provider.CEREBRAS: ["CEREBRAS_API_KEY"],
            Provider.COHERE: ["COHERE_API_KEY", "CO_API_KEY"],
            Provider.GROQ: ["GROQ_API_KEY"],
            Provider.HUGGINGFACE: ["HUGGINGFACE_API_KEY", "HF_TOKEN"],
            Provider.REPLICATE: ["REPLICATE_API_TOKEN"],
        }

        # Check environment variables for each provider.
        for provider, env_vars in env_variable_map.items():
            if provider in self.providers:
                continue  # Skip if already configured from explicit parameters.

            for env_var in env_vars:
                env_value = os.environ.get(env_var)
                if env_value:
                    # Create clean kwargs without conflicting api_key parameter.
                    clean_kwargs = kwargs.copy()
                    clean_kwargs.pop("api_key", None)

                    self._add_provider(provider, api_key=env_value, **clean_kwargs)
                    break

    def _add_provider(self, provider: Provider, **kwargs: Any) -> None:
        """Adds a provider client to the available providers.

        Args:
            provider: The provider enum to add.
            **kwargs: Configuration options for the provider client.

        Raises:
            ProviderNotFoundError: If the provider is not supported.
            ChimericError: If provider initialization fails.
        """
        if provider not in PROVIDER_CLIENTS:
            raise ProviderNotFoundError(f"Provider {provider.value} not supported")

        try:
            client_class = PROVIDER_CLIENTS[provider]
            client = client_class(**kwargs)

            self.providers[provider] = client

            # Set the first successfully initialized provider as primary.
            if self.primary_provider is None:
                self.primary_provider = provider

        except (ImportError, ModuleNotFoundError, ValueError) as e:
            raise ChimericError(f"Failed to initialize provider {provider.value}: {e}") from e

    def generate(
        self,
        model: str,
        messages: Input,
        **kwargs: Any,
    ) -> CompletionResponse | Generator[StreamChunk, None, None]:
        """Generates chat completion using the appropriate provider for the model.

        Args:
            model: Model name to use (determines provider automatically).
            messages: List of messages in provider-compatible format.
            **kwargs: Additional provider-specific arguments (temperature, tools, etc.).

        Returns:
            CompletionResponse object or a generator yielding StreamChunk objects
            if streaming is enabled.

        Raises:
            ChimericError: If no suitable provider is found for the model.
        """
        target_provider = self._select_provider_by_model(model)
        client = self.providers[target_provider]

        return client.chat_completion(
            messages=messages,
            model=model,
            **kwargs,
        )

    async def agenerate(
        self,
        model: str,
        messages: Input,
        **kwargs: Any,
    ) -> CompletionResponse | AsyncGenerator[StreamChunk, None]:
        """Asynchronously generates chat completion.

        Args:
            model: Model name to use (determines provider automatically).
            messages: List of messages in provider-compatible format.
            **kwargs: Additional provider-specific arguments (temperature, tools, etc.).

        Returns:
            CompletionResponse object or an async generator yielding StreamChunk
            objects if streaming is enabled.

        Raises:
            ChimericError: If no suitable provider is found for the model.
        """
        target_provider = self._select_provider_by_model(model)
        client = self.providers[target_provider]

        return await client.achat_completion(
            messages=messages,
            model=model,
            **kwargs,
        )

    def _select_provider_by_model(self, model: str) -> Provider:
        """Selects the appropriate provider based on model name patterns.

        Args:
            model: The name of the model to use.

        Returns:
            The provider enum to use for this model.

        Raises:
            ChimericError: If no suitable provider is found.
        """
        model_lower = model.lower()

        # Define model patterns for each provider.
        provider_patterns = [
            (Provider.OPENAI, ["gpt", "chatgpt", "davinci", "curie", "babbage", "ada", "o1"]),
            (Provider.GOOGLE, ["google", "palm", "bison", "gemini"]),
            (Provider.CEREBRAS, ["cerebras", "llama3.1"]),
            (Provider.COHERE, ["command", "cohere"]),
            (Provider.GROQ, ["mixtral", "llama2", "gemma"]),
        ]

        # Check each provider's patterns.
        for provider, patterns in provider_patterns:
            if provider in self.providers and any(pattern in model_lower for pattern in patterns):
                return provider

        # HuggingFace models typically have "org/model" format.
        if "/" in model_lower and Provider.HUGGINGFACE in self.providers:
            return Provider.HUGGINGFACE

        # Fallback to primary provider if available.
        if self.primary_provider and self.primary_provider in self.providers:
            return self.primary_provider

        # Use the first available provider as last resort.
        if self.providers:
            return next(iter(self.providers.keys()))

        raise ChimericError(f"No suitable provider found for model: {model}")

    def list_models(self, provider: str | None = None) -> list[ModelSummary]:
        """Lists available models from specified provider or all providers.

        Args:
            provider: Provider name to list models for, or None for all providers.

        Returns:
            List of ModelSummary objects containing model information.

        Raises:
            ProviderNotFoundError: If the specified provider is not configured.
        """
        if provider:
            provider_enum = Provider(provider.lower())
            if provider_enum not in self.providers:
                raise ProviderNotFoundError(f"Provider {provider} not configured")

            models = self.providers[provider_enum].list_models()
            # Ensure provider information is set on each model.
            for model in models:
                if model.provider is None:
                    model.provider = provider_enum.value
            return models

        # Collect models from all configured providers.
        all_models: list[ModelSummary] = []
        for provider_enum, client in self.providers.items():
            try:
                models = client.list_models()
                # Ensure provider information is set on each model.
                for model in models:
                    model.provider = provider_enum.value
                all_models.extend(models)
            except (
                ImportError,
                ModuleNotFoundError,
                ValueError,
                ProviderError,
                ConnectionError,
                TimeoutError,
            ):
                # Skip providers that fail due to connection or API issues.
                continue
        return all_models

    def get_capabilities(self, provider: str | None = None) -> Capability:
        """Gets capabilities for a specific provider or merged capabilities.

        Args:
            provider: Optional provider name to get capabilities for.

        Returns:
            Capability object with supported features.

        Raises:
            ProviderNotFoundError: If the specified provider is not configured.
        """
        if provider:
            provider_enum = Provider(provider.lower())
            if provider_enum not in self.providers:
                raise ProviderNotFoundError(f"Provider {provider} not configured")
            return self.providers[provider_enum].capabilities

        # Merge capabilities from all providers (union of all features).
        merged_capabilities = Capability()
        for client in self.providers.values():
            capabilities = client.capabilities
            for field_name in merged_capabilities.model_dump():
                if getattr(capabilities, field_name):
                    setattr(merged_capabilities, field_name, True)
        return merged_capabilities

    def get_provider_client(self, provider: str) -> BaseClient[Any]:
        """Gets direct access to a provider's client instance.

        Args:
            provider: Provider name to get the client for.

        Returns:
            The provider's client instance.

        Raises:
            ProviderNotFoundError: If the provider is not configured.
        """
        provider_enum = Provider(provider.lower())
        if provider_enum not in self.providers:
            raise ProviderNotFoundError(f"Provider {provider} not configured")
        return self.providers[provider_enum]

    def tool(
        self, name: str | None = None, description: str | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a function as a Chimeric tool.

        This decorator allows easy registration of functions as tools for LLM
        providers that support tool/function calling. Function parameters and
        return types are automatically inferred from type annotations.

        Args:
            name: Optional name for the tool. Defaults to function name.
            description: Optional description. Defaults to function's docstring.

        Returns:
            A decorator function that registers the decorated function as a tool.

        Example:
            @chimeric.tool()
            def search_web(query: str) -> list[str]:
                '''Searches the web for information.'''
                return ["result1", "result2"]
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return self._tool_manager.register(func=func, name=name, description=description)

        return decorator

    def get_tools(self) -> list[Tool]:
        """Gets the list of all registered tools.

        Returns:
            List of all registered Tool instances.
        """
        return self._tool_manager.get_all_tools()

    @property
    def available_providers(self) -> list[str]:
        """Gets the list of configured provider names.

        Returns:
            List of provider name strings.
        """
        return [provider.value for provider in self.providers]

    def __repr__(self) -> str:
        """Returns a string representation of the Chimeric client.

        Returns:
            String representation showing configured providers and primary provider.
        """
        configured_providers = [provider.value for provider in self.providers]
        primary_provider_name = self.primary_provider.value if self.primary_provider else None

        return f"Chimeric(providers={configured_providers}, primary={primary_provider_name})"
