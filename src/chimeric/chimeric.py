"""Chimeric — unified interface for multiple LLM providers.

The Chimeric class is the single entry point for applications.  It:
  - discovers configured providers from API keys (explicit or env vars),
  - routes model names to the correct provider via a canonical-name cache,
  - delegates all HTTP and serialisation work to HttpClient + adapters.

No provider SDK is imported here.  Adding a new provider means adding one
entry to PROVIDER_REGISTRY in config.py.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from .config import PROVIDER_REGISTRY
from .exceptions import ModelNotSupportedError, ProviderError, ProviderNotFoundError
from .http import HttpClient
from .tools import ToolManager
from .types import Provider
from .utils import normalize_messages, normalize_tools

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Generator

    from .config import ProviderConfig
    from .types import CompletionResponse, Input, ModelSummary, StreamChunk, Tool, Tools

__all__ = ["Chimeric"]


class Chimeric:
    """Unified interface for multiple LLM providers.

    Supports OpenAI, Anthropic, Google Gemini, Cerebras, Cohere, xAI Grok,
    Groq, and OpenRouter with automatic model-to-provider routing and
    integrated tool management.

    API keys are resolved from explicit parameters first, then environment
    variables.  Providers without a discoverable API key are silently skipped.

    Examples:
        Basic usage::

            client = Chimeric()
            response = client.generate("gpt-4o", "Hello!")

        Streaming::

            for chunk in client.generate("gpt-4o", "Tell a story", stream=True):
                print(chunk.delta or "", end="", flush=True)

        Tool registration::

            @client.tool()
            def get_weather(city: str) -> str:
                '''Return weather for a city.'''
                return f"Sunny in {city}"

            response = client.generate("gpt-4o", "Weather in NYC?")
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
        openrouter_api_key: str | None = None,
        timeout: float = 60.0,
        **_kwargs: Any,
    ) -> None:
        """Initialise Chimeric with provider configuration.

        Args:
            openai_api_key: OpenAI API key (env: OPENAI_API_KEY).
            anthropic_api_key: Anthropic API key (env: ANTHROPIC_API_KEY).
            google_api_key: Google AI key (env: GOOGLE_API_KEY or GEMINI_API_KEY).
            cerebras_api_key: Cerebras API key (env: CEREBRAS_API_KEY).
            cohere_api_key: Cohere API key (env: COHERE_API_KEY or CO_API_KEY).
            grok_api_key: xAI Grok key (env: GROK_API_KEY or XAI_API_KEY).
            groq_api_key: Groq API key (env: GROQ_API_KEY).
            openrouter_api_key: OpenRouter key (env: OPENROUTER_API_KEY).
            timeout: HTTP request timeout in seconds.
            **_kwargs: Accepted but ignored for forward-compatibility.
        """
        self._http = HttpClient(timeout=timeout)
        self._tool_manager = ToolManager()

        # provider_name -> (ProviderConfig, api_key)
        self._providers: dict[str, tuple[ProviderConfig, str]] = {}
        # canonical model name -> provider name
        self._model_cache: dict[str, str] = {}

        # Explicit keys take precedence over env vars
        explicit: dict[str, str | None] = {
            "openai": openai_api_key,
            "anthropic": anthropic_api_key,
            "google": google_api_key,
            "cerebras": cerebras_api_key,
            "cohere": cohere_api_key,
            "grok": grok_api_key,
            "groq": groq_api_key,
            "openrouter": openrouter_api_key,
        }
        for name, key in explicit.items():
            if key:
                self._register_provider(name, key)

        # Auto-detect remaining providers from environment
        for name, config in PROVIDER_REGISTRY.items():
            if name in self._providers:
                continue  # already registered
            for env_var in config.api_key_env_vars:
                key = os.environ.get(env_var)
                if key:
                    self._register_provider(name, key)
                    break

    # ------------------------------------------------------------------
    # Provider management
    # ------------------------------------------------------------------

    def _register_provider(self, name: str, api_key: str) -> None:
        """Register a provider and populate the model cache."""
        config = PROVIDER_REGISTRY.get(name)
        if config is None:
            return
        self._providers[name] = (config, api_key)
        self._populate_model_cache(name, config, api_key)

    def _populate_model_cache(self, name: str, config: ProviderConfig, api_key: str) -> None:
        """Fetch available models and index them by canonical name."""
        try:
            models = self._http.list_models(config, api_key)
        except Exception:
            # Network errors at init time are non-fatal; routing falls back to
            # live queries in _find_provider_for_model().
            return

        for model in models:
            for raw in (model.id, model.name):
                canon = _canonical(raw)
                if canon:
                    self._model_cache[canon] = name

    # ------------------------------------------------------------------
    # Sync public API
    # ------------------------------------------------------------------

    def generate(
        self,
        model: str,
        messages: Input,
        stream: bool = False,
        tools: Tools = None,
        auto_tool: bool = True,
        provider: str | Provider | None = None,
        **kwargs: Any,
    ) -> CompletionResponse | Generator[StreamChunk, None, None]:
        """Generate a chat completion.

        Args:
            model: Model identifier (e.g. "gpt-4o", "claude-3-5-sonnet-20241022").
            messages: User input — string, dict, Message, or list thereof.
            stream: Yield StreamChunks instead of returning a CompletionResponse.
            tools: Tools to make available for function calling.
            auto_tool: Include all registered tools when tools is None.
            provider: Force a specific provider instead of auto-routing.
            **kwargs: Provider pass-through (temperature, max_tokens, etc.).

        Returns:
            CompletionResponse for non-streaming; Generator[StreamChunk] for streaming.

        Raises:
            ModelNotSupportedError: Model not found in any configured provider.
            ProviderNotFoundError: Explicit provider not configured.
            ProviderError: Provider API call failed.
        """
        config, api_key = self._resolve_provider(model, provider)
        resolved_tools = self._resolve_tools(tools, auto_tool)
        normalized = normalize_messages(messages)

        return self._http.complete(
            config=config,
            api_key=api_key,
            messages=normalized,
            model=model,
            stream=stream,
            tools=resolved_tools or None,
            **kwargs,
        )

    def list_models(self, provider: str | Provider | None = None) -> list[ModelSummary]:
        """List available models from one or all configured providers.

        Args:
            provider: Provider name to query, or None for all.

        Returns:
            List of ModelSummary objects tagged with their provider name.

        Raises:
            ProviderNotFoundError: Named provider is not configured.
        """
        if provider:
            config, api_key = self._get_provider(provider)
            models = self._http.list_models(config, api_key)
            for m in models:
                m.provider = provider
            return models

        all_models: list[ModelSummary] = []
        for name, (config, api_key) in self._providers.items():
            try:
                models = self._http.list_models(config, api_key)
                for m in models:
                    m.provider = name
                all_models.extend(models)
            except (ProviderError, ConnectionError, TimeoutError):
                continue  # skip unavailable providers
        return all_models

    # ------------------------------------------------------------------
    # Async public API
    # ------------------------------------------------------------------

    async def agenerate(
        self,
        model: str,
        messages: Input,
        stream: bool = False,
        tools: Tools = None,
        auto_tool: bool = True,
        provider: str | Provider | None = None,
        **kwargs: Any,
    ) -> CompletionResponse | AsyncGenerator[StreamChunk, None]:
        """Async version of generate().

        Args:
            model: Model identifier.
            messages: User input.
            stream: Return async generator instead of awaitable response.
            tools: Tools to make available for function calling.
            auto_tool: Include all registered tools when tools is None.
            provider: Force a specific provider.
            **kwargs: Provider pass-through options.

        Returns:
            Awaitable CompletionResponse or AsyncGenerator[StreamChunk].
        """
        config, api_key = self._resolve_provider(model, provider)
        resolved_tools = self._resolve_tools(tools, auto_tool)
        normalized = normalize_messages(messages)

        return await self._http.acomplete(
            config=config,
            api_key=api_key,
            messages=normalized,
            model=model,
            stream=stream,
            tools=resolved_tools or None,
            **kwargs,
        )

    async def alist_models(self, provider: str | Provider | None = None) -> list[ModelSummary]:
        """Async version of list_models()."""
        if provider:
            config, api_key = self._get_provider(provider)
            models = await self._http.alist_models(config, api_key)
            for m in models:
                m.provider = provider
            return models

        all_models: list[ModelSummary] = []
        for name, (config, api_key) in self._providers.items():
            try:
                models = await self._http.alist_models(config, api_key)
                for m in models:
                    m.provider = name
                all_models.extend(models)
            except (ProviderError, ConnectionError, TimeoutError):
                continue
        return all_models

    # ------------------------------------------------------------------
    # Tool management
    # ------------------------------------------------------------------

    def tool(
        self,
        name: str | None = None,
        description: str | None = None,
        strict: bool = True,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a callable as an LLM tool.

        Args:
            name: Override the function name.
            description: Override the docstring description.
            strict: Enforce strict parameter validation.

        Example::

            @client.tool()
            def get_weather(city: str) -> str:
                '''Get weather for a city.'''
                return f"Sunny in {city}"
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return self._tool_manager.register(
                func=func, name=name, description=description, strict=strict
            )

        return decorator

    @property
    def tools(self) -> list[Tool]:
        """All currently registered tools."""
        return self._tool_manager.get_all_tools()

    @property
    def available_providers(self) -> list[str]:
        """Names of all configured providers."""
        return list(self._providers.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_provider(self, name: str | Provider) -> tuple[ProviderConfig, str]:
        """Retrieve a configured provider by name.

        Raises:
            ProviderNotFoundError: Provider is not in the registry or not configured.
        """
        key = name.value if isinstance(name, Provider) else name
        if key not in PROVIDER_REGISTRY:
            raise ProviderNotFoundError(key, list(self._providers.keys()))
        if key not in self._providers:
            raise ProviderNotFoundError(key, list(self._providers.keys()))
        return self._providers[key]

    def _resolve_provider(self, model: str, provider: str | Provider | None) -> tuple[ProviderConfig, str]:
        """Return (config, api_key) for the target provider.

        If provider is explicit, validate and return it.  Otherwise route
        by model name via the cache or a live search.
        """
        if provider:
            return self._get_provider(provider)
        return self._find_provider_for_model(model)

    def _find_provider_for_model(self, model: str) -> tuple[ProviderConfig, str]:
        """Locate the provider that offers this model.

        Checks the in-memory cache first.  Falls back to live model-listing
        queries for models added after initialisation.

        Raises:
            ModelNotSupportedError: No configured provider recognises the model.
        """
        canon = _canonical(model)

        # Fast path: check pre-populated cache
        provider_name = self._model_cache.get(canon)
        if provider_name and provider_name in self._providers:
            return self._providers[provider_name]

        # Slow path: query each provider live
        for name, (config, api_key) in self._providers.items():
            try:
                models = self._http.list_models(config, api_key)
            except (ProviderError, ConnectionError, TimeoutError):
                continue

            for m in models:
                self._model_cache[_canonical(m.id)] = name
                self._model_cache[_canonical(m.name)] = name

            if canon in self._model_cache:
                return self._providers[self._model_cache[canon]]

        # Build a best-effort list of known models for the error message
        available: list[str] = list(self._model_cache)
        provider_name = next(iter(self._providers), None)
        raise ModelNotSupportedError(
            model=model,
            provider=provider_name,
            supported_models=available or None,
        )

    def _resolve_tools(self, tools: Tools, auto_tool: bool) -> list[Tool]:
        """Return the effective tool list for this request.

        If tools is None and auto_tool is True, include all registered tools.
        Otherwise normalise the caller-supplied tools.
        """
        if tools is None:
            return self._tool_manager.get_all_tools() if auto_tool else []
        return normalize_tools(tools)

    def _clear_model_cache(self) -> None:
        """Invalidate the model cache (useful when providers add new models)."""
        self._model_cache.clear()
        for name, (config, api_key) in self._providers.items():
            self._populate_model_cache(name, config, api_key)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"Chimeric(providers={self.available_providers})"


def _canonical(name: str) -> str:
    """Reduce a model name to alphanumeric lowercase for fuzzy matching."""
    return "".join(ch for ch in name.lower() if ch.isalnum())
