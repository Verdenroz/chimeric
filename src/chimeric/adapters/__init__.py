"""Adapter registry mapping adapter names to instances.

HttpClient uses ADAPTER_REGISTRY to look up the adapter for a given provider.
"""

from __future__ import annotations

from .anthropic import AnthropicAdapter
from .base import Adapter, EmbeddingAdapter, StreamState
from .cohere import CohereAdapter
from .google import GoogleAdapter
from .openai import OpenAIAdapter

__all__ = [
    "ADAPTER_REGISTRY",
    "Adapter",
    "AnthropicAdapter",
    "CohereAdapter",
    "EmbeddingAdapter",
    "GoogleAdapter",
    "OpenAIAdapter",
    "StreamState",
]

# Keyed by ProviderConfig.adapter — one instance per adapter type (stateless).
ADAPTER_REGISTRY: dict[str, Adapter] = {
    "openai": OpenAIAdapter(),
    "anthropic": AnthropicAdapter(),
    "google": GoogleAdapter(),
    "cohere": CohereAdapter(),
}
