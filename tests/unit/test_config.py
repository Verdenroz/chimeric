"""Tests for the provider configuration registry."""

import pytest

from chimeric.config import PROVIDER_REGISTRY, ProviderConfig


class TestProviderRegistry:
    def test_all_expected_providers_present(self):
        expected = {
            "openai",
            "anthropic",
            "google",
            "cohere",
            "groq",
            "cerebras",
            "grok",
            "openrouter",
        }
        assert expected == set(PROVIDER_REGISTRY.keys())

    def test_each_entry_is_provider_config(self):
        for name, config in PROVIDER_REGISTRY.items():
            assert isinstance(config, ProviderConfig), f"{name} is not a ProviderConfig"

    def test_adapter_names_valid(self):
        valid_adapters = {"openai", "anthropic", "google", "cohere"}
        for name, config in PROVIDER_REGISTRY.items():
            assert config.adapter in valid_adapters, f"{name} has unknown adapter: {config.adapter}"

    def test_openai_compatible_providers_use_openai_adapter(self):
        openai_compat = {"openai", "groq", "cerebras", "grok", "openrouter"}
        for name in openai_compat:
            assert PROVIDER_REGISTRY[name].adapter == "openai", f"{name} should use openai adapter"

    def test_anthropic_config(self):
        config = PROVIDER_REGISTRY["anthropic"]
        assert config.auth_style == "header"
        assert config.auth_prefix == ""  # No "Bearer " prefix
        assert config.auth_header == "x-api-key"
        assert "anthropic-version" in config.extra_headers
        assert "max_tokens" in config.default_kwargs
        assert config.default_kwargs["max_tokens"] == 4096

    def test_google_config(self):
        config = PROVIDER_REGISTRY["google"]
        assert config.auth_style == "query_param"
        assert config.auth_query_param == "key"
        assert config.adapter == "google"

    def test_api_key_env_vars_non_empty(self):
        for name, config in PROVIDER_REGISTRY.items():
            assert len(config.api_key_env_vars) > 0, f"{name} has no env vars"

    def test_base_urls_start_with_https(self):
        for name, config in PROVIDER_REGISTRY.items():
            assert config.base_url.startswith("https://"), f"{name} base_url not HTTPS"

    def test_configs_are_frozen(self):
        config = PROVIDER_REGISTRY["openai"]
        with pytest.raises((AttributeError, TypeError)):
            config.name = "changed"


class TestProviderConfig:
    def test_defaults(self):
        config = ProviderConfig(
            name="test",
            base_url="https://example.com",
            adapter="openai",
            api_key_env_vars=("TEST_API_KEY",),
        )
        assert config.auth_header == "Authorization"
        assert config.auth_prefix == "Bearer "
        assert config.completion_path == "/chat/completions"
        assert config.auth_style == "header"
        assert config.models_path == "/models"
        assert config.default_kwargs == {}
