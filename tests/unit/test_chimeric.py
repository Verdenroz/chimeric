"""Tests for the Chimeric facade (provider routing, tool management, etc.)."""

from collections.abc import AsyncGenerator, Generator
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from chimeric import Chimeric
from chimeric.exceptions import ModelNotSupportedError, ProviderNotFoundError
from chimeric.types import CompletionResponse, ModelSummary, StreamChunk, Usage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_completion(content: str = "Hello!") -> CompletionResponse:
    return CompletionResponse(content=content, usage=Usage(), model="gpt-4o")


def _make_models(*ids: str) -> list[ModelSummary]:
    return [ModelSummary(id=m, name=m) for m in ids]


@pytest.fixture
def mock_http():
    """Return a mock HttpClient pre-configured with sensible defaults."""
    mock = MagicMock()
    mock.list_models.return_value = _make_models("gpt-4o", "gpt-4o-mini")
    mock.complete.return_value = _make_completion()
    mock.acomplete = MagicMock(return_value=_make_completion())
    mock.alist_models = MagicMock(return_value=_make_models("gpt-4o"))
    return mock


@pytest.fixture
def chimeric(mock_http):
    """Chimeric instance with an injected mock HttpClient and one provider."""
    with patch("chimeric.chimeric.HttpClient", return_value=mock_http):
        client = Chimeric(openai_api_key="sk-test")
    return client, mock_http


# ---------------------------------------------------------------------------
# Initialisation tests
# ---------------------------------------------------------------------------


class TestInitialisation:
    def test_explicit_api_key_registers_provider(self, mock_http):
        with patch("chimeric.chimeric.HttpClient", return_value=mock_http):
            client = Chimeric(openai_api_key="sk-test")
        assert "openai" in client.available_providers

    def test_no_providers_when_no_keys(self, mock_http):
        """If no API keys are set, available_providers should be empty."""
        env_vars = {
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "GEMINI_API_KEY": "",
            "CEREBRAS_API_KEY": "",
            "COHERE_API_KEY": "",
            "CO_API_KEY": "",
            "GROK_API_KEY": "",
            "XAI_API_KEY": "",
            "GROQ_API_KEY": "",
            "OPENROUTER_API_KEY": "",
        }
        with (
            patch.dict(os.environ, env_vars, clear=False),
            patch("chimeric.chimeric.HttpClient", return_value=mock_http),
        ):
            client = Chimeric()
        # All env vars are empty strings which are falsy
        assert "openai" not in client.available_providers

    def test_env_var_auto_detects_provider(self, mock_http):
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-test"}),
            patch("chimeric.chimeric.HttpClient", return_value=mock_http),
        ):
            client = Chimeric()
        assert "openai" in client.available_providers

    def test_multiple_providers_registered(self, mock_http):
        mock_http.list_models.return_value = []
        with patch("chimeric.chimeric.HttpClient", return_value=mock_http):
            client = Chimeric(openai_api_key="sk-1", anthropic_api_key="sk-2")
        assert "openai" in client.available_providers
        assert "anthropic" in client.available_providers

    def test_repr(self, chimeric):
        client, _ = chimeric
        assert "openai" in repr(client)


# ---------------------------------------------------------------------------
# Provider selection / routing
# ---------------------------------------------------------------------------


class TestProviderRouting:
    def test_routes_by_model_name_in_cache(self, chimeric):
        client, mock = chimeric
        # "gpt4o" is the canonical form of "gpt-4o", pre-loaded at init
        client.generate("gpt-4o", "Hello")
        mock.complete.assert_called_once()

    def test_explicit_provider_bypasses_routing(self, chimeric):
        client, mock = chimeric
        client.generate("any-model", "Hello", provider="openai")
        mock.complete.assert_called_once()

    def test_unknown_explicit_provider_raises(self, chimeric):
        client, _ = chimeric
        with pytest.raises(ProviderNotFoundError):
            client.generate("gpt-4o", "Hello", provider="nonexistent")

    def test_unconfigured_explicit_provider_raises(self, chimeric):
        client, _ = chimeric
        with pytest.raises(ProviderNotFoundError):
            client.generate("gpt-4o", "Hello", provider="anthropic")

    def test_unknown_model_raises_model_not_supported(self, mock_http):
        mock_http.list_models.return_value = _make_models("gpt-4o")
        with patch("chimeric.chimeric.HttpClient", return_value=mock_http):
            client = Chimeric(openai_api_key="sk-test")
        with pytest.raises(ModelNotSupportedError):
            client.generate("totally-unknown-model-xyz", "Hello")


# ---------------------------------------------------------------------------
# generate() / agenerate()
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_returns_completion_response(self, chimeric):
        client, _ = chimeric
        response = client.generate("gpt-4o", "Hello")
        assert isinstance(response, CompletionResponse)
        assert response.content == "Hello!"

    def test_string_message_normalised(self, chimeric):
        client, mock = chimeric
        client.generate("gpt-4o", "Just a string")
        call_kwargs = mock.complete.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0].role == "user"
        assert messages[0].content == "Just a string"

    def test_streaming_returns_generator(self, chimeric):
        client, mock = chimeric

        def _gen():
            yield StreamChunk(content="Hello", delta="Hello")

        mock.complete.return_value = _gen()
        result = client.generate("gpt-4o", "Hi", stream=True)
        assert isinstance(result, Generator)

    def test_kwargs_forwarded_to_http(self, chimeric):
        client, mock = chimeric
        client.generate("gpt-4o", "Hello", temperature=0.7, max_tokens=100)
        call_kwargs = mock.complete.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 100

    async def test_agenerate_returns_completion(self, chimeric):
        client, mock = chimeric
        mock.acomplete = MagicMock(return_value=MagicMock())
        mock.acomplete.return_value = _make_completion("Async response!")

        # Patch acomplete to be a coroutine

        async def _coro(*args: Any, **kwargs: Any) -> CompletionResponse:
            return _make_completion("Async response!")

        mock.acomplete.side_effect = _coro
        response = await client.agenerate("gpt-4o", "Hello")
        assert isinstance(response, CompletionResponse)

    async def test_agenerate_streaming_returns_async_generator(self, chimeric):
        client, mock = chimeric

        async def _agen() -> AsyncGenerator[StreamChunk, None]:
            yield StreamChunk(content="Hi", delta="Hi")

        async def _coro(*args: Any, **kwargs: Any) -> AsyncGenerator[StreamChunk, None]:
            return _agen()

        mock.acomplete.side_effect = _coro
        result = await client.agenerate("gpt-4o", "Hi", stream=True)
        assert isinstance(result, AsyncGenerator)


# ---------------------------------------------------------------------------
# Tool management
# ---------------------------------------------------------------------------


class TestToolManagement:
    def test_register_tool_via_decorator(self, chimeric):
        client, _ = chimeric

        @client.tool()
        def my_tool(x: int) -> str:
            """A test tool."""
            return str(x)

        assert any(t.name == "my_tool" for t in client.tools)

    def test_auto_tool_includes_registered_tools(self, chimeric):
        client, mock = chimeric

        @client.tool()
        def my_tool(x: int) -> str:
            """A test tool."""
            return str(x)

        client.generate("gpt-4o", "Hello", auto_tool=True)
        call_kwargs = mock.complete.call_args.kwargs
        assert call_kwargs["tools"] is not None
        assert any(t.name == "my_tool" for t in call_kwargs["tools"])

    def test_auto_tool_false_sends_no_tools(self, chimeric):
        client, mock = chimeric

        @client.tool()
        def my_tool(x: int) -> str:
            """A test tool."""
            return str(x)

        client.generate("gpt-4o", "Hello", auto_tool=False)
        call_kwargs = mock.complete.call_args.kwargs
        # tools=None or empty list should be passed
        assert not call_kwargs.get("tools")

    def test_explicit_tools_override_auto(self, chimeric):
        from chimeric.types import Tool

        client, mock = chimeric
        explicit_tool = Tool(name="explicit", description="Explicit tool")
        client.generate("gpt-4o", "Hello", tools=[explicit_tool])
        call_kwargs = mock.complete.call_args.kwargs
        tool_names = [t.name for t in (call_kwargs.get("tools") or [])]
        assert "explicit" in tool_names


# ---------------------------------------------------------------------------
# list_models()
# ---------------------------------------------------------------------------


class TestListModels:
    def test_list_all_providers(self, chimeric):
        client, mock = chimeric
        mock.list_models.return_value = _make_models("gpt-4o")
        models = client.list_models()
        assert len(models) > 0

    def test_list_specific_provider(self, chimeric):
        client, mock = chimeric
        mock.list_models.return_value = _make_models("gpt-4o")
        models = client.list_models(provider="openai")
        assert all(m.provider == "openai" for m in models)

    def test_list_unconfigured_provider_raises(self, chimeric):
        client, _ = chimeric
        with pytest.raises(ProviderNotFoundError):
            client.list_models(provider="anthropic")

    def test_provider_tagged_on_models(self, chimeric):
        client, mock = chimeric
        mock.list_models.return_value = _make_models("gpt-4o", "gpt-4o-mini")
        models = client.list_models(provider="openai")
        assert all(m.provider == "openai" for m in models)
