from enum import Enum
import os
from typing import Any

from .base import BaseClient, CompletionResponse, Message, Tool
from .exceptions import ChimericError, ProviderNotFoundError

# Import provider clients
from .providers.aws.client import AWSClient
from .providers.cerebras.client import CerebrasClient
from .providers.cohere.client import CohereClient
from .providers.gemini.client import GeminiClient
from .providers.groq.client import GroqClient
from .providers.huggingface.client import HuggingFaceClient
from .providers.openai.client import OpenAIClient
from .providers.replicate.client import ReplicateClient


class Provider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    CEREBRAS = "cerebras"
    COHERE = "cohere"
    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    REPLICATE = "replicate"
    AWS = "aws"


# Provider client mapping
PROVIDER_CLIENTS: dict[Provider, type[BaseClient]] = {
    Provider.OPENAI: OpenAIClient,
    Provider.GEMINI: GeminiClient,
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

    def __init__(self, **kwargs):
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

    def _parse_provider_configs(self, kwargs: dict) -> dict[Provider, dict]:
        """Parse provider-specific configurations from kwargs."""
        configs = {}

        # OpenAI configuration
        if kwargs.get("openai_api_key"):
            configs[Provider.OPENAI] = {
                "api_key": kwargs["openai_api_key"],
                "base_url": kwargs.get("openai_base_url"),
                "organization": kwargs.get("openai_organization"),
            }

        # Anthropic configuration (mapped to AWS for now)
        if kwargs.get("anthropic_api_key"):
            configs[Provider.AWS] = {
                "api_key": kwargs["anthropic_api_key"],
                "base_url": kwargs.get("anthropic_base_url"),
            }

        # Google/Gemini configuration
        if kwargs.get("google_api_key"):
            configs[Provider.GEMINI] = {
                "api_key": kwargs["google_api_key"],
            }

        # Cohere configuration
        if kwargs.get("cohere_api_key"):
            configs[Provider.COHERE] = {
                "api_key": kwargs["cohere_api_key"],
            }

        # Groq configuration
        if kwargs.get("groq_api_key"):
            configs[Provider.GROQ] = {
                "api_key": kwargs["groq_api_key"],
            }

        # HuggingFace configuration
        if kwargs.get("huggingface_api_key"):
            configs[Provider.HUGGINGFACE] = {
                "api_key": kwargs["huggingface_api_key"],
            }

        # Replicate configuration
        if kwargs.get("replicate_api_token"):
            configs[Provider.REPLICATE] = {
                "api_key": kwargs["replicate_api_token"],
            }

        # AWS configuration
        if kwargs.get("aws_access_key_id") or kwargs.get("aws_region"):
            configs[Provider.AWS] = {
                "aws_access_key_id": kwargs.get("aws_access_key_id"),
                "aws_secret_access_key": kwargs.get("aws_secret_access_key"),
                "region": kwargs.get("aws_region", "us-east-1"),
            }

        return configs

    def _auto_detect_providers(self) -> None:
        """Auto-detect available providers from environment variables and explicit configs."""
        # Environment variable detection map
        detection_map = {
            Provider.OPENAI: ["OPENAI_API_KEY"],
            Provider.GEMINI: ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
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
            except Exception:
                # Skip providers that fail to initialize
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
                    except Exception:
                        # Skip providers that fail to initialize
                        continue

    def _add_provider(self, provider: Provider, **kwargs) -> None:
        """Add a provider to the client."""
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
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> CompletionResponse:
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
            CompletionResponse object with the generated content
        """
        # Convert dict messages to Message objects
        message_objects = [Message(role=msg["role"], content=msg["content"]) for msg in messages]

        # Determine which provider to use based on model
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
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> CompletionResponse:
        """Async version of generate."""
        # Convert dict messages to Message objects
        message_objects = [Message(role=msg["role"], content=msg["content"]) for msg in messages]

        # Determine which provider to use based on model
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
        """Select the appropriate provider based on model name."""
        model_lower = model.lower()

        # OpenAI models
        if any(
            x in model_lower for x in ["gpt", "chatgpt", "davinci", "curie", "babbage", "ada", "o1"]
        ):
            if Provider.OPENAI in self.providers:
                return Provider.OPENAI

        # Claude/Anthropic models (handled via AWS Bedrock)
        if any(x in model_lower for x in ["claude", "anthropic"]):
            if Provider.AWS in self.providers:
                return Provider.AWS

        # Gemini models
        if any(x in model_lower for x in ["gemini", "palm", "bison"]):
            if Provider.GEMINI in self.providers:
                return Provider.GEMINI

        # Cerebras models
        if any(x in model_lower for x in ["cerebras", "llama3.1"]):
            if Provider.CEREBRAS in self.providers:
                return Provider.CEREBRAS

        # Cohere models
        if any(x in model_lower for x in ["command", "cohere"]):
            if Provider.COHERE in self.providers:
                return Provider.COHERE

        # Groq models
        if any(x in model_lower for x in ["mixtral", "llama2", "gemma"]):
            if Provider.GROQ in self.providers:
                return Provider.GROQ

        # HuggingFace models
        if "/" in model_lower:  # HuggingFace models typically have format "org/model"
            if Provider.HUGGINGFACE in self.providers:
                return Provider.HUGGINGFACE

        # Use primary provider as fallback
        if self.primary_provider and self.primary_provider in self.providers:
            return self.primary_provider

        # Use first available provider
        if self.providers:
            return next(iter(self.providers.keys()))

        raise ChimericError(f"No suitable provider found for model: {model}")

    def list_models(self, provider: str | None = None) -> list[dict[str, Any]]:
        """List models from specific provider or all providers."""
        if provider:
            provider_enum = Provider(provider.lower())
            if provider_enum not in self.providers:
                raise ProviderNotFoundError(f"Provider {provider} not configured")
            return self.providers[provider_enum].list_models()

        # Return models from all providers
        all_models = []
        for prov, client in self.providers.items():
            try:
                models = client.list_models()
                # Add provider info to each model
                for model in models:
                    model["provider"] = prov.value
                all_models.extend(models)
            except Exception:
                # Skip providers that fail to list models
                continue
        return all_models

    def get_capabilities(self, provider: str | None = None) -> dict[str, bool]:
        """Get capabilities for a specific provider or all providers."""
        if provider:
            provider_enum = Provider(provider.lower())
            if provider_enum not in self.providers:
                raise ProviderNotFoundError(f"Provider {provider} not configured")
            return self.providers[provider_enum].get_capabilities()

        # Return merged capabilities from all providers
        all_capabilities = {}
        for prov, client in self.providers.items():
            caps = client.get_capabilities()
            for capability, supported in caps.items():
                all_capabilities[capability] = all_capabilities.get(capability, False) or supported
        return all_capabilities

    def get_provider_client(self, provider: str) -> BaseClient:
        """Get direct access to a provider's client."""
        provider_enum = Provider(provider.lower())
        if provider_enum not in self.providers:
            raise ProviderNotFoundError(f"Provider {provider} not configured")
        return self.providers[provider_enum]

    @property
    def available_providers(self) -> list[str]:
        """Get list of configured provider names."""
        return [p.value for p in self.providers.keys()]

    def __repr__(self) -> str:
        configured = list(self.providers.keys())
        return f"Chimeric(providers={[p.value for p in configured]}, primary={self.primary_provider.value if self.primary_provider else None})"
