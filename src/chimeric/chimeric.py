from collections.abc import AsyncGenerator, Callable, Generator
import os
from typing import Any

from .base import ChimericAsyncClient, ChimericClient
from .exceptions import ChimericError, ModelNotSupportedError, ProviderError, ProviderNotFoundError
from .tools import ToolManager
from .types import (
    Capability,
    ChimericStreamChunk,
    CompletionResponse,
    Input,
    ModelSummary,
    Provider,
    StreamChunk,
    Tool,
    Tools,
)

__all__ = [
    "ASYNC_PROVIDER_CLIENTS",
    "PROVIDER_CLIENTS",
    "Chimeric",
]


# Build provider mappings conditionally based on available dependencies
def _build_provider_mappings() -> tuple[dict[Provider, type], dict[Provider, type]]:
    """Build provider client mappings, including only providers with available dependencies."""
    sync_clients = {}
    async_clients = {}

    # Define all possible providers with their import paths
    provider_imports = {
        Provider.OPENAI: ("chimeric.providers.openai.client", "OpenAIClient", "OpenAIAsyncClient"),
        Provider.ANTHROPIC: (
            "chimeric.providers.anthropic.client",
            "AnthropicClient",
            "AnthropicAsyncClient",
        ),
        Provider.GOOGLE: ("chimeric.providers.google.client", "GoogleClient", "GoogleAsyncClient"),
        Provider.CEREBRAS: (
            "chimeric.providers.cerebras.client",
            "CerebrasClient",
            "CerebrasAsyncClient",
        ),
        Provider.COHERE: ("chimeric.providers.cohere.client", "CohereClient", "CohereAsyncClient"),
        Provider.GROK: ("chimeric.providers.grok.client", "GrokClient", "GrokAsyncClient"),
        Provider.GROQ: ("chimeric.providers.groq.client", "GroqClient", "GroqAsyncClient"),
    }

    # Try to import each provider and add to mappings if successful
    for provider, (module_path, sync_class, async_class) in provider_imports.items():
        try:
            module = __import__(module_path, fromlist=[sync_class, async_class])
            sync_clients[provider] = getattr(module, sync_class)
            async_clients[provider] = getattr(module, async_class)
        except (ImportError, ModuleNotFoundError):
            # Skip providers with missing dependencies
            continue

    return sync_clients, async_clients


# Build the provider mappings with only available providers
PROVIDER_CLIENTS, ASYNC_PROVIDER_CLIENTS = _build_provider_mappings()


class Chimeric:
    """Unified interface for multiple LLM providers with automatic provider detection.

    Supports OpenAI, Anthropic, Google AI, Cerebras, Cohere, xAI Grok, and Groq
    with automatic model-to-provider routing and tool management.

    Examples:
        Basic usage:

        >>> client = Chimeric()  # Auto-detects API keys from environment
        >>> response = client.generate(model="gpt-4o", messages="Hello!")

        Streaming:

        >>> for chunk in client.generate(model="gpt-4o", messages="Tell a story", stream=True):
        ...     print(chunk.content, end="")

        Tool registration:

        >>> @client.tool()
        ... def get_weather(city: str) -> str:
        ...     return f"Weather in {city}: Sunny"
        >>> response = client.generate(model="gpt-4o", messages="What's the weather in NYC?")
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        google_api_key: str | None = None,
        cerebras_api_key: str | None = None,
        cohere_api_key: str | None = None,
        grok_api_key: str | None = None,
        groq_api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Chimeric client with provider configuration.

        API keys can be provided explicitly or via environment variables.

        Environment variables:
        - OPENAI_API_KEY
        - ANTHROPIC_API_KEY
        - GOOGLE_API_KEY or GEMINI_API_KEY
        - CEREBRAS_API_KEY
        - COHERE_API_KEY or CO_API_KEY
        - GROK_API_KEY or XAI_API_KEY
        - GROQ_API_KEY

        Args:
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            anthropic_api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            google_api_key: Google AI API key (defaults to GOOGLE_API_KEY or GEMINI_API_KEY env var)
            cerebras_api_key: Cerebras API key (defaults to CEREBRAS_API_KEY env var)
            cohere_api_key: Cohere API key (defaults to COHERE_API_KEY or CO_API_KEY env var)
            grok_api_key: xAI Grok API key (defaults to GROK_API_KEY or XAI_API_KEY env var)
            groq_api_key: Groq API key (defaults to GROQ_API_KEY env var)
            **kwargs: Provider-specific options (timeout, base_url, max_retries, etc.)

        Raises:
            ChimericError: If no providers can be initialized
        """
        self.providers: dict[Provider, ChimericClient[Any, Any, Any]] = {}
        self.async_providers: dict[Provider, ChimericAsyncClient[Any, Any, Any]] = {}
        self.primary_provider: Provider | None = None

        # Initialize the tool management system.
        self._tool_manager = ToolManager()

        # Mapping of canonical model names to their providers
        self._model_provider_mapping: dict[str, Provider] = {}

        # Initialize providers from explicit API keys.
        self._initialize_providers_from_config(
            openai_api_key,
            anthropic_api_key,
            google_api_key,
            cerebras_api_key,
            cohere_api_key,
            grok_api_key,
            groq_api_key,
            **kwargs,
        )

        # Auto-detect providers from environment variables.
        self._detect_providers_from_environment(kwargs)

    def _initialize_providers_from_config(
        self,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        google_api_key: str | None = None,
        cerebras_api_key: str | None = None,
        cohere_api_key: str | None = None,
        grok_api_key: str | None = None,
        groq_api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes providers from explicitly provided API keys.

        Args:
            openai_api_key: OpenAI API key.
            anthropic_api_key: Anthropic API key.
            google_api_key: Google API key.
            cerebras_api_key: Cerebras API key.
            cohere_api_key: Cohere API key.
            grok_api_key: Grok API key.
            groq_api_key: Groq API key.
            **kwargs: Additional provider-specific configuration parameters.
        """
        provider_configs: list[tuple[Provider, str | None]] = [
            (Provider.OPENAI, openai_api_key),
            (Provider.ANTHROPIC, anthropic_api_key),
            (Provider.GOOGLE, google_api_key),
            (Provider.CEREBRAS, cerebras_api_key),
            (Provider.COHERE, cohere_api_key),
            (Provider.GROK, grok_api_key),
            (Provider.GROQ, groq_api_key),
        ]

        # Initialize providers that have API keys provided.
        for provider, api_key in provider_configs:
            if api_key is not None:
                self._add_provider(
                    provider, api_key=api_key, tool_manager=self._tool_manager, **kwargs
                )
                self._add_async_provider(
                    provider, api_key=api_key, tool_manager=self._tool_manager, **kwargs
                )

    def _detect_providers_from_environment(self, kwargs: dict[str, Any]) -> None:
        """Auto-detects available providers from environment variables.

        Args:
            kwargs: Additional configuration options to pass to providers.
        """
        # Map providers to their possible environment variable names.
        env_variable_map: dict[Provider, list[str]] = {
            Provider.OPENAI: ["OPENAI_API_KEY"],
            Provider.ANTHROPIC: ["ANTHROPIC_API_KEY"],
            Provider.GOOGLE: ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            Provider.CEREBRAS: ["CEREBRAS_API_KEY"],
            Provider.COHERE: ["COHERE_API_KEY", "CO_API_KEY"],
            Provider.GROK: ["GROK_API_KEY", "XAI_API_KEY"],
            Provider.GROQ: ["GROQ_API_KEY"],
        }

        # Check environment variables for each provider.
        for provider, env_vars in env_variable_map.items():
            if provider in self.providers:
                continue  # Skip if already configured from explicit parameters.

            # Skip if provider dependencies are not available
            if provider not in PROVIDER_CLIENTS:
                continue

            for env_var in env_vars:
                env_value = os.environ.get(env_var)
                if env_value:
                    # Create clean kwargs without a conflicting api_key parameter.
                    clean_kwargs = kwargs.copy()
                    clean_kwargs.pop("api_key", None)

                    self._add_provider(
                        provider, api_key=env_value, tool_manager=self._tool_manager, **clean_kwargs
                    )
                    self._add_async_provider(
                        provider, api_key=env_value, tool_manager=self._tool_manager, **clean_kwargs
                    )
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
            available = [p.value for p in PROVIDER_CLIENTS]
            raise ProviderNotFoundError(provider.value, available)

        try:
            client_class = PROVIDER_CLIENTS[provider]
            client = client_class(**kwargs)

            self.providers[provider] = client

            # Set the first successfully initialized provider as primary.
            if self.primary_provider is None:
                self.primary_provider = provider

            # Populate model mapping for this provider
            self._populate_models_for_provider(provider, client)

        except (ImportError, ModuleNotFoundError) as e:
            raise ChimericError(
                f"Failed to initialize provider {provider.value}. Are you sure the provider is in the environment?: {e}"
            ) from e

    def _add_async_provider(self, provider: Provider, **kwargs: Any) -> None:
        """Adds an async provider client to the available async providers.

        Args:
            provider: The provider enum to add.
            **kwargs: Configuration options for the async provider client.

        Raises:
            ProviderNotFoundError: If the provider is not supported.
            ChimericError: If provider initialization fails.
        """
        if provider not in ASYNC_PROVIDER_CLIENTS:
            available = [p.value for p in ASYNC_PROVIDER_CLIENTS]
            raise ProviderNotFoundError(f"Async provider {provider.value}", available)

        try:
            async_client_class = ASYNC_PROVIDER_CLIENTS[provider]
            async_client = async_client_class(**kwargs)

            self.async_providers[provider] = async_client

        except (ImportError, ModuleNotFoundError, ValueError):
            # Skip providers with missing dependencies instead of crashing
            pass

    def _populate_models_for_provider(
        self, provider: Provider, client: ChimericClient[Any, Any, Any]
    ) -> None:
        """Populates the mapping with models from a specific provider.

        Args:
            provider: The provider enum.
            client: The provider's client instance.

        Raises:
            ProviderError: If the provider fails to list models.
        """
        try:
            models = client.list_models()

            # Add both model IDs and names to mapping (using canonical form)
            for model in models:
                canon_id = "".join(ch for ch in model.id.lower() if ch.isalnum())
                canon_name = "".join(ch for ch in model.name.lower() if ch.isalnum())

                # Store both canonical ID and name pointing to this provider
                self._model_provider_mapping[canon_id] = provider
                if canon_name != canon_id:  # Avoid duplicate entries
                    self._model_provider_mapping[canon_name] = provider

        except Exception as e:
            raise ProviderError(
                provider=provider.value,
                message=None,
                error=e,
            ) from e

    @staticmethod
    def _transform_stream(
        stream: Generator[ChimericStreamChunk[Any], None, None], native: bool = False
    ) -> Generator[StreamChunk, None, None]:
        """Transform a ChimericStreamChunk generator to return the native or common format."""
        for chunk in stream:
            yield chunk.native if native else chunk.common

    @staticmethod
    async def _atransform_stream(
        stream: AsyncGenerator[ChimericStreamChunk[Any]], native: bool = False
    ) -> AsyncGenerator[StreamChunk, None]:
        """Transform an async ChimericStreamChunk generator to return the native or common format."""
        async for chunk in stream:
            yield chunk.native if native else chunk.common

    def generate(
        self,
        model: str,
        messages: Input,
        stream: bool = False,
        tools: Tools = None,
        auto_tool: bool = True,
        native: bool = False,
        provider: str | None = None,
        **kwargs: Any,
    ) -> CompletionResponse | Generator[StreamChunk, None, None]:
        """Generate chat completion using the appropriate provider for the model.

        Args:
            model: Model name (e.g., "gpt-4o", "gemini-2.5-flash")
            messages: Messages as string, dict, or list of dicts with 'role'/'content' keys
            stream: If True, returns generator for streaming responses
            tools: List of functions/Tool objects for model to call, or None
            auto_tool: If True, includes all registered tools when tools=None
            native: If True, returns provider's native response format
            provider: Force specific provider ('openai', 'anthropic', etc.)
            **kwargs: Provider options (temperature, max_tokens, top_p, etc.)

        Returns:
            CompletionResponse or Generator[StreamChunk] for streaming

        Raises:
            ModelNotSupportedError: Model not available from any configured provider
            ProviderNotFoundError: Specified provider not configured
            ChimericError: Provider or authentication errors

        Examples:
            Basic text generation:

            >>> response = client.generate(
            ...     model="gpt-4o",
            ...     messages=[{"role": "user", "content": "Hello, how are you?"}]
            ... )
            >>> print(response.content)

            Streaming response:

            >>> for chunk in client.generate(
            ...     model="claude-3-5-haiku-latest",
            ...     messages=[{"role": "user", "content": "Write a story"}],
            ...     stream=True
            ... ):
            ...     print(chunk.content, end="", flush=True)

            With tools/function calling:

            >>> def get_weather(city: str) -> str:
            ...     return f"Weather in {city}: Sunny, 72°F"
            >>>
            >>> response = client.generate(
            ...     model="gpt-4o",
            ...     messages=[{"role": "user", "content": "What's the weather in NYC?"}],
            ...     tools=[get_weather]
            ... )

            Force specific provider:

            >>> response = client.generate(
            ...     model="gpt-4o",
            ...     messages=[{"role": "user", "content": "Hello"}],
            ...     provider="openai"  # Force OpenAI even if other providers support gpt-4o
            ... )

            Advanced parameters:

            >>> response = client.generate(
            ...     model="gpt-4o",
            ...     messages=[{"role": "user", "content": "Generate JSON"}],
            ...     temperature=0.7,
            ...     max_tokens=1000,
            ...     response_format={"type": "json_object"}
            ... )

        Note:
            - Model names are matched using a canonical form (alphanumeric characters only)
            - Provider selection uses a cached mapping for performance
            - Providers with connection issues are silently skipped during model lookup
            - Tool execution happens automatically when the model calls functions
        """
        target_provider = self._select_provider(model, provider)
        client = self.providers[target_provider]

        chimeric_completion = client.chat_completion(
            messages=messages,
            model=model,
            stream=stream,
            tools=tools,
            auto_tool=auto_tool,
            **kwargs,
        )
        if isinstance(chimeric_completion, Generator):
            # If the response is a generator, it means streaming is enabled.
            return self._transform_stream(chimeric_completion, native=native)

        return chimeric_completion.native if native else chimeric_completion.common

    async def agenerate(
        self,
        model: str,
        messages: Input,
        stream: bool = False,
        tools: Tools = None,
        auto_tool: bool = True,
        native: bool = False,
        provider: str | None = None,
        **kwargs: Any,
    ) -> CompletionResponse | AsyncGenerator[StreamChunk, None]:
        """Async version of generate() for non-blocking chat completion.

        Args:
            model: Model name (e.g., "gpt-4o", "claude-3-5-haiku-latest")
            messages: Messages as string, dict, or list of dicts with 'role'/'content' keys
            stream: If True, returns async generator for streaming responses
            tools: List of functions/Tool objects for model to call, or None
            auto_tool: If True, includes all registered tools when tools=None
            native: If True, returns provider's native response format
            provider: Force specific provider ('openai', 'anthropic', etc.)
            **kwargs: Provider options (temperature, max_tokens, top_p, etc.)

        Returns:
            CompletionResponse or AsyncGenerator[StreamChunk] for streaming

        Raises:
            ModelNotSupportedError: Model not available from any configured provider
            ProviderNotFoundError: Specified provider not configured
            ChimericError: Provider or authentication errors

        Examples:
            Basic async text generation:

            >>> import asyncio
            >>> async def main():
            ...     response = await client.agenerate(
            ...         model="gpt-4o",
            ...         messages=[{"role": "user", "content": "Hello, how are you?"}]
            ...     )
            ...     print(response.content)
            ...     return response
            >>> asyncio.run(main())

            Async streaming response:

            >>> async def stream_example():
            ...     async for chunk in client.agenerate(
            ...         model="claude-3-5-haiku-latest",
            ...         messages=[{"role": "user", "content": "Write a story"}],
            ...         stream=True
            ...     ):
            ...         print(chunk.content, end="", flush=True)
            >>> asyncio.run(stream_example())

            Async with tools/function calling:

            >>> async def tool_example():
            ...     def get_weather(city: str) -> str:
            ...         return f"Weather in {city}: Sunny, 72°F"
            ...
            ...     response = await client.agenerate(
            ...         model="gpt-4o",
            ...         messages=[{"role": "user", "content": "What's the weather in NYC?"}],
            ...         tools=[get_weather]
            ...     )
            ...     return response
            >>> asyncio.run(tool_example())

            Multiple concurrent requests:

            >>> async def concurrent_example():
            ...     tasks = [
            ...         client.agenerate(model="gpt-4o", messages=[{"role": "user", "content": f"Tell me about {topic}"}])
            ...         for topic in ["Python", "JavaScript", "Rust"]
            ...     ]
            ...     responses = await asyncio.gather(*tasks)
            ...     return responses
            >>> asyncio.run(concurrent_example())

        Note:
            - All error handling and provider selection logic is identical to generate()
            - Async generators must be consumed with `async for` loops
            - Multiple concurrent requests can be made using asyncio.gather()
            - Tool execution happens automatically and asynchronously when the model calls functions
        """
        target_provider = self._select_async_provider(model, provider)
        async_client = self.async_providers[target_provider]

        chimeric_completion = await async_client.chat_completion(
            model=model,
            messages=messages,
            stream=stream,
            tools=tools,
            auto_tool=auto_tool,
            **kwargs,
        )
        if isinstance(chimeric_completion, AsyncGenerator):
            # If the response is an async generator, it means streaming is enabled.
            return self._atransform_stream(chimeric_completion, native=native)

        return chimeric_completion.native if native else chimeric_completion.common

    def _select_provider(self, model: str, provider: str | None = None) -> Provider:
        """Selects the appropriate provider based on explicit provider or model availability.

        Args:
            model: The name of the model to use.
            provider: Optional provider name to force using a specific provider.

        Returns:
            The provider enum to use for this model.

        Raises:
            ProviderNotFoundError: If the specified provider is not configured or
                                 if no provider supports the requested model.
        """
        if provider:
            # Use explicitly specified provider
            try:
                provider_enum = Provider(provider.lower())
            except ValueError as e:
                raise ProviderNotFoundError(f"Unknown provider: {provider}") from e

            if provider_enum not in self.providers:
                raise ProviderNotFoundError(f"Provider {provider} not configured")

            return provider_enum

        # Auto-detect provider by model
        return self._select_provider_by_model(model)

    def _select_async_provider(self, model: str, provider: str | None = None) -> Provider:
        """Selects the appropriate async provider based on explicit provider or model availability.

        Args:
            model: The name of the model to use.
            provider: Optional provider name to force using a specific provider.

        Returns:
            The provider enum to use for this model.

        Raises:
            ProviderNotFoundError: If the specified provider is not configured or
                                 if no provider supports the requested model.
        """
        if provider:
            # Use explicitly specified provider
            try:
                provider_enum = Provider(provider.lower())
            except ValueError as e:
                raise ProviderNotFoundError(f"Unknown provider: {provider}") from e

            if provider_enum not in self.async_providers:
                raise ProviderNotFoundError(f"Async provider {provider} not configured")

            return provider_enum

        # Auto-detect provider by model (use sync providers for model detection, same cache)
        return self._select_provider_by_model(model)

    def _select_provider_by_model(self, model: str) -> Provider:
        """Selects the appropriate provider based on model availability.

        This method uses preloaded model mapping for fast lookups and falls back
        to individual provider queries if the model is not in the mapping.

        Args:
            model: The name of the model to use.

        Returns:
            The provider enum to use for this model.

        Raises:
            ModelNotSupportedError: If no provider supports the requested model.
        """
        canon_model = "".join(ch for ch in model.lower() if ch.isalnum())

        # Check preloaded mapping first
        provider = self._model_provider_mapping.get(canon_model)
        if provider and provider in self.providers:
            return provider

        # If not in preloaded mapping, try dynamic lookup (fallback for new models)
        for provider, client in self.providers.items():
            try:
                models = client.list_models()

                # Build canonical sets for IDs and display names
                canon_ids = {"".join(ch for ch in m.id.lower() if ch.isalnum()) for m in models}
                canon_names = {"".join(ch for ch in m.name.lower() if ch.isalnum()) for m in models}

                if canon_model in canon_ids or canon_model in canon_names:
                    # Add to mapping for future lookups
                    self._model_provider_mapping[canon_model] = provider
                    return provider

            except (
                ImportError,
                ModuleNotFoundError,
                ValueError,
                ProviderError,
                ConnectionError,
                TimeoutError,
            ):
                # Skip providers that fail due to connection or API issues
                continue

        # If no provider found, build list of available models and raise exception
        available_models = []
        for _provider, client in self.providers.items():
            try:
                models = client.list_models()
                for m in models:
                    available_models.append(f"{m.id}")
            except (
                ImportError,
                ModuleNotFoundError,
                ValueError,
                ProviderError,
                ConnectionError,
                TimeoutError,
            ):
                # Skip providers that fail due to connection or API issues
                continue

        # If only one provider, include it in the error for more specific messaging
        provider_name = (
            next(iter(self.providers.keys())).value if len(self.providers) == 1 else None
        )
        raise ModelNotSupportedError(
            model=model, provider=provider_name, supported_models=available_models
        )

    def list_models(self, provider: str | None = None) -> list[ModelSummary]:
        """List available models from specified provider or all providers.

        Args:
            provider: Provider name ('openai', 'anthropic', etc.) or None for all

        Returns:
            List of ModelSummary objects with id, name, and provider fields

        Raises:
            ProviderNotFoundError: Specified provider not configured
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

    @property
    def capabilities(self) -> Capability:
        """Merged capabilities from all configured providers."""
        # Merge capabilities from all providers (union of all features).
        merged_values = {
            "streaming": False,
            "tools": False,
        }

        # Collect capabilities from all providers
        for client in self.providers.values():
            capabilities = client.capabilities
            for field_name in merged_values:
                if getattr(capabilities, field_name):
                    merged_values[field_name] = True

        # Create a new instance with the merged values
        return Capability(**merged_values)

    def get_capabilities(self, provider: str | None = None) -> Capability:
        """Get capabilities for specific provider or merged from all providers.

        Args:
            provider: Provider name or None for merged capabilities

        Returns:
            Capability object with streaming and tools boolean fields

        Raises:
            ProviderNotFoundError: Specified provider not configured
        """
        if provider:
            provider_enum = Provider(provider.lower())
            if provider_enum not in self.providers:
                raise ProviderNotFoundError(f"Provider {provider} not configured")
            return self.providers[provider_enum].capabilities

        # Use the property for merged capabilities
        return self.capabilities

    def _get_provider_client(self, provider: str) -> ChimericClient[Any, Any, Any]:
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

    def _clear_model_cache(self) -> None:
        """Clears the model-to-provider mapping.

        This can be useful if providers add or remove models dynamically.
        The mapping will be repopulated with fresh data from all providers.
        Providers that fail to list models will be skipped.
        """
        self._model_provider_mapping.clear()
        # Repopulate the mapping from all configured providers
        for provider, client in self.providers.items():
            try:
                self._populate_models_for_provider(provider, client)
            except ProviderError:
                # Skip providers that fail to list models
                continue

    def tool(
        self,
        name: str | None = None,
        description: str | None = None,
        strict: bool = True,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a function as a tool for LLM function calling.

        Args:
            name: Custom name for tool (defaults to function name)
            description: Custom description (defaults to function docstring)
            strict: Enforce strict type checking (default True)

        Returns:
            Decorator that registers function and returns it unchanged

        Example:
            >>> @client.tool()
            ... def get_weather(city: str) -> str:
            ...     '''Get weather for a city.'''
            ...     return f"Sunny in {city}"
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return self._tool_manager.register(
                func=func, name=name, description=description, strict=strict
            )

        return decorator

    @property
    def tools(self) -> list[Tool]:
        """List of all registered tools for function calling."""
        return self._tool_manager.get_all_tools()

    @property
    def available_providers(self) -> list[str]:
        """List of successfully configured provider names."""
        return [provider.value for provider in self.providers]

    def __repr__(self) -> str:
        """Returns a string representation of the Chimeric client.

        Returns:
            String representation showing configured providers and primary provider.
        """
        configured_providers = [provider.value for provider in self.providers]
        primary_provider_name = self.primary_provider.value if self.primary_provider else None

        return f"Chimeric(providers={configured_providers}, primary={primary_provider_name})"
