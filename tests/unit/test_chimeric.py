"""Tests for the Chimeric facade (provider routing, tool management, etc.)."""

from collections.abc import AsyncGenerator, Generator
import os
from typing import Any
from unittest.mock import MagicMock, patch

from pydantic import BaseModel
import pytest

from chimeric import Chimeric
from chimeric.exceptions import ModelNotSupportedError, ProviderNotFoundError, StructuredOutputError
from chimeric.types import (
    CompletionResponse,
    EmbeddingResponse,
    EmbeddingUsage,
    ModelSummary,
    StreamChunk,
    Usage,
)

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

    def test_base_url_registers_custom_provider(self, mock_http):
        """base_url registers an OpenAI-compatible 'custom' provider."""
        mock_http.list_models.return_value = _make_models("qwen2.5:3b", "llama-3")
        with patch("chimeric.chimeric.HttpClient", return_value=mock_http):
            client = Chimeric(base_url="http://127.0.0.1:12434/v1")
        assert "custom" in client.available_providers

    def test_base_url_uses_openai_adapter(self, mock_http):
        """Custom provider inherits the openai adapter and completion path."""
        mock_http.list_models.return_value = _make_models("local-model")
        with patch("chimeric.chimeric.HttpClient", return_value=mock_http):
            client = Chimeric(base_url="http://localhost:8080/v1")
        config, _ = client._providers["custom"]
        assert config.adapter == "openai"
        assert config.base_url == "http://localhost:8080/v1"

    def test_base_url_trailing_slash_stripped(self, mock_http):
        """Trailing slash on base_url is removed."""
        mock_http.list_models.return_value = []
        with patch("chimeric.chimeric.HttpClient", return_value=mock_http):
            client = Chimeric(base_url="http://localhost:8080/v1/")
        config, _ = client._providers["custom"]
        assert not config.base_url.endswith("/")

    def test_base_url_default_api_key_is_local(self, mock_http):
        """Default api_key for custom provider is 'local'."""
        mock_http.list_models.return_value = []
        with patch("chimeric.chimeric.HttpClient", return_value=mock_http):
            client = Chimeric(base_url="http://localhost:11434/v1")
        _, key = client._providers["custom"]
        assert key == "local"

    def test_base_url_custom_api_key(self, mock_http):
        """api_key parameter is used for the custom provider."""
        mock_http.list_models.return_value = []
        with patch("chimeric.chimeric.HttpClient", return_value=mock_http):
            client = Chimeric(base_url="http://localhost:11434/v1", api_key="secret")
        _, key = client._providers["custom"]
        assert key == "secret"

    def test_base_url_models_populate_cache(self, mock_http):
        """Models from the custom endpoint are routable without provider=."""
        mock_http.list_models.return_value = _make_models("qwen2.5:3b")
        mock_http.complete.return_value = _make_completion("hi")
        with patch("chimeric.chimeric.HttpClient", return_value=mock_http):
            client = Chimeric(base_url="http://127.0.0.1:12434/v1")
        # Should route to 'custom' without needing provider= arg
        resp = client.generate("qwen2.5:3b", "Hello")
        assert resp.content == "hi"

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


class TestStructuredOutput:
    class Contact(BaseModel):
        name: str
        email: str

    def test_response_model_sets_parsed(self, chimeric):
        client, mock = chimeric
        mock.complete.return_value = CompletionResponse(
            content='{"name": "Alice", "email": "alice@example.com"}',
            model="gpt-4o",
        )
        response = client.generate("gpt-4o", "Extract contact", response_model=self.Contact)
        assert isinstance(response, CompletionResponse)
        assert isinstance(response.parsed, self.Contact)
        assert response.parsed.name == "Alice"
        assert response.parsed.email == "alice@example.com"

    def test_response_model_and_tools_raises_value_error(self, chimeric):
        from chimeric.types import Tool

        client, _ = chimeric
        tool = Tool(name="my_tool", description="A tool")
        with pytest.raises(ValueError, match="mutually exclusive"):
            client.generate("gpt-4o", "Hello", response_model=self.Contact, tools=[tool])

    def test_response_model_invalid_json_raises_structured_output_error(self, chimeric):
        client, mock = chimeric
        mock.complete.return_value = CompletionResponse(
            content="not valid json at all",
            model="gpt-4o",
        )
        with pytest.raises(StructuredOutputError) as exc_info:
            client.generate("gpt-4o", "Hello", response_model=self.Contact)
        assert exc_info.value.model_name == "Contact"
        assert exc_info.value.raw_content == "not valid json at all"

    def test_response_model_none_backward_compatible(self, chimeric):
        client, mock = chimeric
        mock.complete.return_value = CompletionResponse(content="Hello!", model="gpt-4o")
        response = client.generate("gpt-4o", "Hello")
        assert isinstance(response, CompletionResponse)
        assert response.content == "Hello!"
        assert response.parsed is None

    def test_response_model_injects_format_kwargs(self, chimeric):
        client, mock = chimeric
        mock.complete.return_value = CompletionResponse(
            content='{"name": "Bob", "email": "bob@x.com"}',
            model="gpt-4o",
        )
        client.generate("gpt-4o", "Hello", response_model=self.Contact)
        call_kwargs = mock.complete.call_args.kwargs
        # OpenAI adapter expects response_format
        assert "response_format" in call_kwargs
        assert call_kwargs["response_format"]["type"] == "json_schema"

    def test_response_model_disables_auto_tool(self, chimeric):
        client, mock = chimeric

        @client.tool()
        def my_tool(x: int) -> str:
            """A test tool."""
            return str(x)

        mock.complete.return_value = CompletionResponse(
            content='{"name": "Bob", "email": "bob@x.com"}',
            model="gpt-4o",
        )
        client.generate("gpt-4o", "Hello", response_model=self.Contact)
        call_kwargs = mock.complete.call_args.kwargs
        assert not call_kwargs.get("tools")

    def test_streaming_parses_final_chunk(self, chimeric):
        client, mock = chimeric

        def _stream():
            yield StreamChunk(content='{"name": "Car', delta='{"name": "Car')
            yield StreamChunk(
                content='{"name": "Carol", "email": "carol@x.com"}',
                delta='ol", "email": "carol@x.com"}',
                finish_reason="stop",
            )

        mock.complete.return_value = _stream()
        chunks = list(client.generate("gpt-4o", "Hello", stream=True, response_model=self.Contact))
        final = chunks[-1]
        assert isinstance(final.parsed, self.Contact)
        assert final.parsed.name == "Carol"
        # Non-final chunk should not have parsed set
        assert chunks[0].parsed is None

    def test_streaming_invalid_json_raises_structured_output_error(self, chimeric):
        client, mock = chimeric

        def _stream():
            yield StreamChunk(content="bad", delta="bad", finish_reason="stop")

        mock.complete.return_value = _stream()
        with pytest.raises(StructuredOutputError):
            list(client.generate("gpt-4o", "Hello", stream=True, response_model=self.Contact))

    async def test_agenerate_response_model_sets_parsed(self, chimeric):
        client, mock = chimeric

        async def _coro(*args: Any, **kwargs: Any) -> CompletionResponse:
            return CompletionResponse(
                content='{"name": "Dave", "email": "dave@x.com"}',
                model="gpt-4o",
            )

        mock.acomplete.side_effect = _coro
        response = await client.agenerate("gpt-4o", "Hello", response_model=self.Contact)
        assert isinstance(response, CompletionResponse)
        assert isinstance(response.parsed, self.Contact)
        assert response.parsed.name == "Dave"

    async def test_agenerate_response_model_and_tools_raises(self, chimeric):
        from chimeric.types import Tool

        client, _ = chimeric
        tool = Tool(name="t", description="desc")
        with pytest.raises(ValueError, match="mutually exclusive"):
            await client.agenerate("gpt-4o", "Hello", response_model=self.Contact, tools=[tool])


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


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------


def _make_embedding_response(embedding: list[float] | None = None) -> EmbeddingResponse:
    return EmbeddingResponse(
        embedding=embedding or [0.1, 0.2, 0.3],
        model="text-embedding-3-small",
        usage=EmbeddingUsage(prompt_tokens=5, total_tokens=5),
    )


class TestEmbed:
    def test_embed_routes_to_http(self, chimeric):
        client, mock = chimeric
        mock.embed.return_value = _make_embedding_response()
        result = client.embed("gpt-4o", "hello")
        assert isinstance(result, EmbeddingResponse)
        assert result.embedding == [0.1, 0.2, 0.3]
        mock.embed.assert_called_once()

    def test_embed_forwards_kwargs(self, chimeric):
        client, mock = chimeric
        mock.embed.return_value = _make_embedding_response()
        client.embed("gpt-4o", "hello", dimensions=512)
        call_kwargs = mock.embed.call_args
        assert call_kwargs.kwargs.get("dimensions") == 512

    def test_embed_explicit_provider(self, chimeric):
        client, mock = chimeric
        mock.embed.return_value = _make_embedding_response()
        client.embed("text-embedding-3-small", "hello", provider="openai")
        mock.embed.assert_called_once()

    async def test_aembed_routes_to_http(self, chimeric):
        client, mock = chimeric

        async def _coro(*args: Any, **kwargs: Any) -> EmbeddingResponse:
            return _make_embedding_response()

        mock.aembed.side_effect = _coro
        result = await client.aembed("gpt-4o", "hello")
        assert isinstance(result, EmbeddingResponse)
        assert result.embedding == [0.1, 0.2, 0.3]
