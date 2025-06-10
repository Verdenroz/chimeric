from collections.abc import AsyncGenerator, Callable, Generator
import os
from typing import Any

from .base import BaseClient
from .exceptions import ChimericError, ProviderNotFoundError
from .providers.aws.client import AWSClient
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
    Message,
    MessageDict,
    ModelSummary,
    Provider,
    ProviderConfig,
    StreamChunk,
    Tool,
)

__all__ = [
    "PROVIDER_CLIENTS",
    "Chimeric",
    "Provider",
]

# Provider client mapping
PROVIDER_CLIENTS: dict[Provider, type[BaseClient]] = {
    Provider.OPENAI: OpenAIClient,
    Provider.GOOGLE: GoogleClient,
    Provider.CEREBRAS: CerebrasClient,
    Provider.COHERE: CohereClient,
    Provider.GROQ: GroqClient,
    Provider.HUGGINGFACE: HuggingFaceClient,
    Provider.REPLICATE: ReplicateClient,
    Provider.AWS: AWSClient,
}


class Chimeric:
    """Main Chimeric client with unified interface across all LLM providers.

    This class provides a single interface that can work with any supported
    LLM provider, with automatic provider detection and seamless switching.

    Examples:
        # Auto-detect from environment variables
        client = Chimeric()

        # Explicit API key configuration
        client = Chimeric(
            openai_api_key="sk-...",
            anthropic_api_key="sk-ant-..."
        )

        # Generate completions with automatic provider routing
        response = client.generate(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}]
        )

        # Use specific provider
        response = client.generate(
            model="claude-3-opus-20240229",
            messages=[{"role": "user", "content": "Hello"}]
        )
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize Chimeric client.

        Args:
            **kwargs: Provider-specific API keys and configuration options.
                     Common keys include:
                     - openai_api_key: OpenAI API key
                     - anthropic_api_key: Anthropic API key
                     - google_api_key: Google/Gemini API key
                     - cohere_api_key: Cohere API key
                     - groq_api_key: Groq API key
                     - huggingface_api_key: HuggingFace API key
                     - replicate_api_token: Replicate API token
                     - aws_access_key_id: AWS access key ID
                     - aws_secret_access_key: AWS secret access key
                     - aws_region: AWS region
        """
        self.providers: dict[Provider, BaseClient] = {}
        self.primary_provider: Provider | None = None

        # Store provider configurations
        self._provider_configs = self._parse_provider_configs(kwargs)

        # Auto-detect and initialize available providers
        self._auto_detect_providers()

        # Initialize tool registry
        self._tool_manager = ToolManager()

    @staticmethod
    def _parse_provider_configs(kwargs: dict[str, Any]) -> dict[Provider, ProviderConfig]:
        """Parse provider-specific configurations from kwargs.

        Args:
            kwargs: Dictionary of configuration options

        Returns:
            Dictionary mapping providers to their configurations
        """
        configs: dict[Provider, ProviderConfig] = {}

        # OpenAI configuration
        if kwargs.get("openai_api_key"):
            openai_config: ProviderConfig = {"api_key": kwargs["openai_api_key"]}

            if kwargs.get("openai_base_url"):
                openai_config["base_url"] = kwargs["openai_base_url"]

            if kwargs.get("openai_organization"):
                openai_config["organization"] = kwargs["openai_organization"]

            configs[Provider.OPENAI] = openai_config

        # Anthropic configuration (mapped to AWS for now)
        # This will set Provider.AWS. If direct AWS keys are also provided, they will overwrite this.
        if kwargs.get("anthropic_api_key"):
            anthropic_config: ProviderConfig = {"api_key": kwargs["anthropic_api_key"]}

            if kwargs.get("anthropic_base_url"):
                anthropic_config["base_url"] = kwargs["anthropic_base_url"]

            configs[Provider.AWS] = anthropic_config

        # Google/Gemini configuration
        if kwargs.get("google_api_key"):
            configs[Provider.GOOGLE] = {"api_key": kwargs["google_api_key"]}

        # Cohere configuration
        if kwargs.get("cohere_api_key"):
            configs[Provider.COHERE] = {"api_key": kwargs["cohere_api_key"]}

        # Groq configuration
        if kwargs.get("groq_api_key"):
            configs[Provider.GROQ] = {"api_key": kwargs["groq_api_key"]}

        # HuggingFace configuration
        if kwargs.get("huggingface_api_key"):
            configs[Provider.HUGGINGFACE] = {"api_key": kwargs["huggingface_api_key"]}

        # Replicate configuration
        if kwargs.get("replicate_api_token"):
            configs[Provider.REPLICATE] = {"api_key": kwargs["replicate_api_token"]}

        # AWS configuration (direct)
        # This might overwrite the Provider.AWS config if it was set by Anthropic keys
        if kwargs.get("aws_access_key_id") or kwargs.get("aws_region"):
            aws_direct_config: ProviderConfig = {}

            if kwargs.get("aws_access_key_id"):
                aws_direct_config["aws_access_key_id"] = kwargs["aws_access_key_id"]

            if kwargs.get("aws_secret_access_key"):
                aws_direct_config["aws_secret_access_key"] = kwargs["aws_secret_access_key"]

            aws_direct_config["region"] = kwargs.get("aws_region", "us-east-1")

            # If Provider.AWS was already set (e.g., by Anthropic keys),
            # this direct configuration will overwrite it.
            configs[Provider.AWS] = aws_direct_config

        return configs

    def _auto_detect_providers(self) -> None:
        """Auto-detect available providers from environment variables and explicit configs."""
        # Environment variable detection map
        detection_map: dict[Provider, list[str]] = {
            Provider.OPENAI: ["OPENAI_API_KEY"],
            Provider.GOOGLE: ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            Provider.CEREBRAS: ["CEREBRAS_API_KEY"],
            Provider.COHERE: ["COHERE_API_KEY", "CO_API_KEY"],
            Provider.GROQ: ["GROQ_API_KEY"],
            Provider.HUGGINGFACE: ["HUGGINGFACE_API_KEY", "HF_TOKEN"],
            Provider.REPLICATE: ["REPLICATE_API_TOKEN"],
            Provider.AWS: ["AWS_ACCESS_KEY_ID", "AWS_PROFILE", "ANTHROPIC_API_KEY"],
        }

        # First, initialize providers from explicit configurations
        for provider, config in self._provider_configs.items():
            try:
                self._add_provider(provider, **config)
            except (ImportError, ModuleNotFoundError, ValueError):
                # Skip providers that fail to initialize due to missing dependencies or invalid config
                continue

        # Then, detect from environment variables
        for provider, env_vars in detection_map.items():
            if provider in self.providers:
                continue  # Skip if already configured

            for env_var in env_vars:
                if os.environ.get(env_var):
                    try:
                        self._add_provider(provider, api_key=os.environ[env_var])
                        break
                    except (ImportError, ModuleNotFoundError, ValueError):
                        # Skip providers that fail to initialize
                        continue

    def _add_provider(self, provider: Provider, **kwargs: Any) -> None:
        """Add a provider to the client.

        Args:
            provider: The provider enum to add
            **kwargs: Configuration options for the provider

        Raises:
            ProviderNotFoundError: If the provider is not supported
        """
        if provider not in PROVIDER_CLIENTS:
            raise ProviderNotFoundError(f"Provider {provider.value} not supported")

        client_class = PROVIDER_CLIENTS[provider]
        client = client_class(**kwargs)

        self.providers[provider] = client

        if self.primary_provider is None:
            self.primary_provider = provider

    def generate(
        self,
        model: str,
        messages: list[MessageDict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse | Generator[StreamChunk, None, None]:
        """Generate chat completion using the best available provider.

        Args:
            model: Model name to use (determines provider automatically)
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            tools: Tools/functions available to the model
            **kwargs: Additional provider-specific arguments

        Returns:
            CompletionResponse object or a generator yielding StreamChunk objects if streaming
        """
        # Convert dict messages to Message objects
        message_objects = [Message(role=msg["role"], content=msg["content"]) for msg in messages]

        # Determine which provider to use based on the model
        target_provider = self._select_provider_by_model(model)
        client = self.providers[target_provider]

        return client.chat_completion(
            messages=message_objects,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            tools=tools,
            **kwargs,
        )

    async def agenerate(
        self,
        model: str,
        messages: list[MessageDict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> CompletionResponse | AsyncGenerator[StreamChunk, None]:
        """Async version of generate.

        Args:
            model: Model name to use (determines provider automatically)
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            tools: Tools/functions available to the model
            **kwargs: Additional provider-specific arguments

        Returns:
            CompletionResponse object or an async generator yielding StreamChunk objects if streaming
        """
        # Convert dict messages to Message objects
        message_objects = [Message(role=msg["role"], content=msg["content"]) for msg in messages]

        # Determine which provider to use based on the model
        target_provider = self._select_provider_by_model(model)
        client = self.providers[target_provider]

        return await client.achat_completion(
            messages=message_objects,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            tools=tools,
            **kwargs,
        )

    def _select_provider_by_model(self, model: str) -> Provider:
        """Select the appropriate provider based on model name.

        Args:
            model: The name of the model to use

        Returns:
            The provider enum to use for this model

        Raises:
            ChimericError: If no suitable provider is found
        """
        model_lower = model.lower()

        # OpenAI models
        if (
            any(
                x in model_lower
                for x in ["gpt", "chatgpt", "davinci", "curie", "babbage", "ada", "o1"]
            )
            and Provider.OPENAI in self.providers
        ):
            return Provider.OPENAI

        # Claude/Anthropic models (handled via AWS Bedrock)
        if (
            any(x in model_lower for x in ["claude", "anthropic"])
            and Provider.AWS in self.providers
        ):
            return Provider.AWS

        # Gemini models
        if (
            any(x in model_lower for x in ["google", "palm", "bison", "gemini"])
            and Provider.GOOGLE in self.providers
        ):
            return Provider.GOOGLE

        # Cerebras models
        if (
            any(x in model_lower for x in ["cerebras", "llama3.1"])
            and Provider.CEREBRAS in self.providers
        ):
            return Provider.CEREBRAS

        # Cohere models
        if (
            any(x in model_lower for x in ["command", "cohere"])
            and Provider.COHERE in self.providers
        ):
            return Provider.COHERE

        # Groq models
        if (
            any(x in model_lower for x in ["mixtral", "llama2", "gemma"])
            and Provider.GROQ in self.providers
        ):
            return Provider.GROQ

        # HuggingFace models
        if (
            "/" in model_lower and Provider.HUGGINGFACE in self.providers
        ):  # HuggingFace models typically have format "org/model"
            return Provider.HUGGINGFACE

        # Use primary provider as fallback
        if self.primary_provider and self.primary_provider in self.providers:
            return self.primary_provider

        # Use the first available provider
        if self.providers:
            return next(iter(self.providers.keys()))

        raise ChimericError(f"No suitable provider found for model: {model}")

    def list_models(self, provider: str | None = None) -> list[ModelSummary]:
        """List models from specific provider or all providers.

        Args:
            provider: Provider name to list models for, or None to list models from all providers

        Returns:
            List of ModelSummary objects containing model information

        Raises:
            ProviderNotFoundError: If the specified provider is not configured
        """
        if provider:
            provider_enum = Provider(provider.lower())
            if provider_enum not in self.providers:
                raise ProviderNotFoundError(f"Provider {provider} not configured")
            # Ensure the provider client's list_models returns list[ModelSummary]
            # and add provider information if not already present
            models = self.providers[provider_enum].list_models()
            for model in models:
                if model.provider is None:
                    model.provider = provider_enum.value
            return models

        # Return models from all providers
        all_models: list[ModelSummary] = []
        for prov_name, client in self.providers.items():
            try:
                models = client.list_models()
                # Add provider info to each model
                for model in models:
                    model.provider = prov_name.value  # Ensure provider is set
                all_models.extend(models)
            except (ImportError, ValueError, ConnectionError, TimeoutError):
                # Skip providers that fail to list models due to connection or API issues
                continue
        return all_models

    def get_capabilities(self, provider: str | None = None) -> Capability:
        """Get capabilities for a specific provider or all providers.

        Args:
            provider: Optional provider name to get capabilities for

        Returns:
            Capability object with supported features

        Raises:
            ProviderNotFoundError: If the specified provider is not configured
        """
        if provider:
            provider_enum = Provider(provider.lower())
            if provider_enum not in self.providers:
                raise ProviderNotFoundError(f"Provider {provider} not configured")
            # This now directly returns the Capability object from the provider's client
            return self.providers[provider_enum].get_capabilities()

        # Return merged capabilities from all providers
        merged_capabilities = Capability()  # Initialize with default false values
        for _, client in self.providers.items():
            caps = client.get_capabilities()
            for field_name in merged_capabilities.model_dump():
                if getattr(caps, field_name):
                    setattr(merged_capabilities, field_name, True)
        return merged_capabilities

    def get_provider_client(self, provider: str) -> BaseClient:
        """Get direct access to a provider's client.

        Args:
            provider: Provider name to get the client for

        Returns:
            The provider's client instance

        Raises:
            ProviderNotFoundError: If the provider is not configured
        """
        provider_enum = Provider(provider.lower())
        if provider_enum not in self.providers:
            raise ProviderNotFoundError(f"Provider {provider} not configured")
        return self.providers[provider_enum]

    @property
    def available_providers(self) -> list[str]:
        """Get the list of configured provider names.

        Returns:
            List of provider name strings
        """
        return [p.value for p in self.providers]

    def __repr__(self) -> str:
        """Return a string representation of the Chimeric client."""
        configured = list(self.providers.keys())
        primary_provider_value = self.primary_provider.value if self.primary_provider else None
        return (
            f"Chimeric(providers={[p.value for p in configured]}, primary={primary_provider_value})"
        )

    def tool(
        self, name: str | None = None, description: str | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a function as a Chimeric tool.

        This decorator allows you to easily register any function as a tool that can be used
        with LLM providers that support tool/function calling. The function parameters and
        return types are automatically inferred from type annotations.

        Args:
            name: Optional name for the tool. If None, the function name is used.
            description: Optional description for the tool. If None, the function's docstring is used.

        Returns:
            A decorator function that registers the decorated function as a tool.

        Example:
            ```python
            client = Chimeric()

            @client.tool()
            def search_web(query: str) -> List[str]:
                '''Search the web for information.'''
                # Implementation
                return ["result1", "result2"]
            ```
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            # Use the ToolManager to register the tool with automatic parameter inference
            return self._tool_manager.register(func=func, name=name, description=description)

        return decorator

    def get_tools(self) -> list[Tool]:
        """Returns the list of registered tools.

        Returns:
            List of all registered Tool instances
        """
        return self._tool_manager.get_all_tools()
