"""Provider registry: connection configs for all supported LLM providers.

Adding a new provider means adding one entry to PROVIDER_REGISTRY.
No classes to instantiate, no SDK imports — pure config data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["PROVIDER_REGISTRY", "ProviderConfig"]


@dataclass(frozen=True)
class ProviderConfig:
    """Immutable connection spec for one LLM provider.

    Attributes:
        name: Canonical provider name (used as dict key in PROVIDER_REGISTRY).
        base_url: API base URL without trailing slash.
        adapter: Which adapter module to use ("openai" | "anthropic" | "google" | "cohere").
        api_key_env_vars: Env-var names checked in order; first non-empty value wins.
        auth_header: Header name for bearer-token auth.
        auth_prefix: Prefix prepended to the API key in the auth header.
        extra_headers: Static headers sent with every request (e.g. version pins).
        completion_path: URL path for chat completions (appended to base_url).
        stream_path: Override path for streaming; None means reuse completion_path.
        models_path: URL path to list available models; None disables model listing.
        auth_style: "header" sends the key in a header; "query_param" appends it to the URL.
        auth_query_param: Query-param name when auth_style == "query_param".
        default_kwargs: Extra body fields injected into every request (e.g. max_tokens).
        default_kwargs: Extra body fields injected into every request (e.g. max_tokens).
    """

    name: str
    base_url: str
    adapter: str
    api_key_env_vars: tuple[str, ...]
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer "
    extra_headers: dict[str, str] = field(default_factory=dict)
    completion_path: str = "/chat/completions"
    stream_path: str | None = None
    models_path: str | None = "/models"
    auth_style: str = "header"  # "header" | "query_param"
    auth_query_param: str = "key"
    default_kwargs: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Provider registry — one entry per provider, ordered for readability.
# OpenAI-compatible providers all share the "openai" adapter; only the
# base_url and env-var names differ.
# ---------------------------------------------------------------------------

PROVIDER_REGISTRY: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        name="openai",
        base_url="https://api.openai.com/v1",
        adapter="openai",
        api_key_env_vars=("OPENAI_API_KEY",),
    ),
    "anthropic": ProviderConfig(
        name="anthropic",
        base_url="https://api.anthropic.com/v1",
        adapter="anthropic",
        api_key_env_vars=("ANTHROPIC_API_KEY",),
        auth_header="x-api-key",
        auth_prefix="",  # Anthropic uses bare key, no "Bearer " prefix
        extra_headers={"anthropic-version": "2023-06-01"},
        completion_path="/messages",
        stream_path="/messages",
        models_path="/models",
        default_kwargs={"max_tokens": 4096},
    ),
    "google": ProviderConfig(
        name="google",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        adapter="google",
        api_key_env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        # Google uses ?key=... query param instead of an Authorization header
        auth_style="query_param",
        auth_query_param="key",
        # Path is dynamic (includes model name); the Google adapter overrides it
        completion_path="/models/{model}:generateContent",
        stream_path="/models/{model}:streamGenerateContent",
        models_path="/models",
    ),
    "cohere": ProviderConfig(
        name="cohere",
        base_url="https://api.cohere.com/v2",
        adapter="cohere",
        api_key_env_vars=("COHERE_API_KEY", "CO_API_KEY"),
        completion_path="/chat",
        stream_path="/chat",
    ),
    "groq": ProviderConfig(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        adapter="openai",
        api_key_env_vars=("GROQ_API_KEY",),
    ),
    "cerebras": ProviderConfig(
        name="cerebras",
        base_url="https://api.cerebras.ai/v1",
        adapter="openai",
        api_key_env_vars=("CEREBRAS_API_KEY",),
    ),
    "grok": ProviderConfig(
        name="grok",
        base_url="https://api.x.ai/v1",
        adapter="openai",
        api_key_env_vars=("GROK_API_KEY", "XAI_API_KEY"),
    ),
    "openrouter": ProviderConfig(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        adapter="openai",
        api_key_env_vars=("OPENROUTER_API_KEY",),
    ),
}
