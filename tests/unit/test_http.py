"""Tests for the HttpClient using httpx mock transport."""

from collections.abc import Mapping
import json

import httpx
import pytest

from chimeric.config import PROVIDER_REGISTRY, ProviderConfig
from chimeric.exceptions import AuthenticationError, ProviderError, RateLimitError
from chimeric.http import HttpClient, _build_headers, _build_url, _raise_for_status
from chimeric.types import EmbeddingResponse, Message, Tool

# ---------------------------------------------------------------------------
# Helpers to build mock responses
# ---------------------------------------------------------------------------


def _json_response(body: Mapping[str, object], status: int = 200) -> httpx.Response:
    """Build a minimal httpx.Response with a JSON body."""
    return httpx.Response(
        status_code=status,
        headers={"content-type": "application/json"},
        text=json.dumps(body),
    )


def _sse_response(lines: list[str], status: int = 200) -> httpx.Response:
    """Build an SSE streaming response from a list of data lines."""
    body = "\n".join(f"data: {line}" for line in lines) + "\n"
    return httpx.Response(
        status_code=status,
        headers={"content-type": "text/event-stream"},
        text=body,
    )


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


class TestBuildHeaders:
    def test_bearer_auth(self):
        config = PROVIDER_REGISTRY["openai"]
        headers = _build_headers(config, "sk-test")
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["Content-Type"] == "application/json"

    def test_anthropic_bare_key(self):
        config = PROVIDER_REGISTRY["anthropic"]
        headers = _build_headers(config, "sk-ant-test")
        assert headers["x-api-key"] == "sk-ant-test"
        assert "anthropic-version" in headers

    def test_google_header_skipped(self):
        """Google uses query params; no auth header should be set."""
        config = PROVIDER_REGISTRY["google"]
        headers = _build_headers(config, "my-key")
        assert "Authorization" not in headers


class TestBuildUrl:
    def test_openai_header_auth(self):
        config = PROVIDER_REGISTRY["openai"]
        url = _build_url(config, "/chat/completions", "sk-test")
        assert url == "https://api.openai.com/v1/chat/completions"
        assert "sk-test" not in url

    def test_google_query_param_auth(self):
        config = PROVIDER_REGISTRY["google"]
        url = _build_url(config, "/models/gemini-2.0-flash:generateContent", "my-key")
        assert "key=my-key" in url

    def test_base_url_trailing_slash_stripped(self):
        from chimeric.config import ProviderConfig

        config = ProviderConfig(
            name="test",
            base_url="https://example.com/v1/",
            adapter="openai",
            api_key_env_vars=("TEST_KEY",),
        )
        url = _build_url(config, "/chat/completions", "key")
        assert url == "https://example.com/v1/chat/completions"


class TestRaiseForStatus:
    def test_success_does_nothing(self):
        response = _json_response({"ok": True}, 200)
        _raise_for_status(response, "openai")  # Should not raise

    def test_401_raises_authentication_error(self):
        response = _json_response({"error": {"message": "Invalid key"}}, 401)
        with pytest.raises(AuthenticationError) as exc_info:
            _raise_for_status(response, "openai")
        assert "openai" in str(exc_info.value)

    def test_429_raises_rate_limit_error(self):
        response = _json_response({"error": {"message": "Rate limited"}}, 429)
        with pytest.raises(RateLimitError) as exc_info:
            _raise_for_status(response, "openai")
        assert "openai" in str(exc_info.value)

    def test_500_raises_provider_error(self):
        response = _json_response({"error": {"message": "Server error"}}, 500)
        with pytest.raises(ProviderError) as exc_info:
            _raise_for_status(response, "openai")
        assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# HttpClient integration tests with mock transport
# ---------------------------------------------------------------------------


def _make_transport(*responses: httpx.Response) -> httpx.MockTransport:
    """Return a mock transport that serves responses in sequence."""
    response_iter = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        return next(response_iter)

    return httpx.MockTransport(handler)


class TestHttpClientComplete:
    def test_basic_completion(self):
        completion_response = {
            "choices": [
                {"message": {"content": "Hello!", "role": "assistant"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            "model": "gpt-4o",
        }
        transport = _make_transport(_json_response(completion_response))
        client = HttpClient()
        client._sync_client = httpx.Client(transport=transport)

        config = PROVIDER_REGISTRY["openai"]
        messages = [Message(role="user", content="Hi")]
        response = client.complete(config, "sk-test", messages, "gpt-4o", False, None)

        from chimeric.types import CompletionResponse

        assert isinstance(response, CompletionResponse)
        assert response.content == "Hello!"
        assert response.usage.total_tokens == 8

    def test_tool_calling_loop(self):
        """Verifies that the client loops back when the model returns tool calls."""
        tool_call_response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }
        final_response = {
            "choices": [
                {"message": {"content": "Sunny!", "role": "assistant"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            "model": "gpt-4o",
        }
        transport = _make_transport(
            _json_response(tool_call_response),
            _json_response(final_response),
        )
        client = HttpClient()
        client._sync_client = httpx.Client(transport=transport)

        def get_weather(city: str) -> str:
            return f"Sunny in {city}"

        tool = Tool(name="get_weather", description="Get weather", function=get_weather)
        config = PROVIDER_REGISTRY["openai"]
        messages = [Message(role="user", content="Weather in NYC?")]
        response = client.complete(config, "sk-test", messages, "gpt-4o", False, [tool])

        from chimeric.types import CompletionResponse

        assert isinstance(response, CompletionResponse)
        assert response.content == "Sunny!"

    def test_list_models(self):
        models_response = {
            "data": [
                {"id": "gpt-4o", "owned_by": "openai"},
                {"id": "gpt-4o-mini", "owned_by": "openai"},
            ]
        }
        transport = _make_transport(_json_response(models_response))
        client = HttpClient()
        client._sync_client = httpx.Client(transport=transport)

        config = PROVIDER_REGISTRY["openai"]
        models = client.list_models(config, "sk-test")
        assert len(models) == 2
        assert models[0].id == "gpt-4o"

    def test_list_models_skipped_when_no_path(self):
        from chimeric.config import ProviderConfig

        config = ProviderConfig(
            name="test",
            base_url="https://example.com",
            adapter="openai",
            api_key_env_vars=("KEY",),
            models_path=None,
        )
        client = HttpClient()
        models = client.list_models(config, "key")
        assert models == []


class TestHttpClientStreaming:
    def test_streaming_returns_generator(self):
        lines = [
            json.dumps({"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]}),
            json.dumps({"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]}),
            "[DONE]",
        ]
        transport = _make_transport(_sse_response(lines))
        client = HttpClient()
        client._sync_client = httpx.Client(transport=transport)

        config = PROVIDER_REGISTRY["openai"]
        messages = [Message(role="user", content="Hi")]
        result = client.complete(config, "sk-test", messages, "gpt-4o", True, None)

        from collections.abc import Generator

        assert isinstance(result, Generator)
        chunks = list(result)
        text_chunks = [c for c in chunks if c.delta]
        assert any("Hello" in (c.delta or "") for c in text_chunks)


class TestHttpClientAsync:
    async def test_basic_async_completion(self):
        completion_response = {
            "choices": [
                {"message": {"content": "Hi!", "role": "assistant"}, "finish_reason": "stop"}
            ],
            "usage": {},
            "model": "gpt-4o",
        }
        transport = _make_transport(_json_response(completion_response))
        client = HttpClient()
        client._async_client = httpx.AsyncClient(transport=transport)

        config = PROVIDER_REGISTRY["openai"]
        messages = [Message(role="user", content="Hello")]
        response = await client.acomplete(config, "sk-test", messages, "gpt-4o", False, None)

        from chimeric.types import CompletionResponse

        assert isinstance(response, CompletionResponse)
        assert response.content == "Hi!"

    async def test_async_list_models(self):
        models_response = {"data": [{"id": "gpt-4o", "owned_by": "openai"}]}
        transport = _make_transport(_json_response(models_response))
        client = HttpClient()
        client._async_client = httpx.AsyncClient(transport=transport)

        config = PROVIDER_REGISTRY["openai"]
        models = await client.alist_models(config, "sk-test")
        assert len(models) == 1


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------

_EMBEDDING_RESPONSE = {
    "object": "list",
    "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 5, "total_tokens": 5},
}

_BATCH_EMBEDDING_RESPONSE = {
    "object": "list",
    "data": [
        {"object": "embedding", "embedding": [0.1, 0.2], "index": 0},
        {"object": "embedding", "embedding": [0.3, 0.4], "index": 1},
    ],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 10, "total_tokens": 10},
}


class TestHttpClientEmbed:
    def test_embed_single(self):
        transport = _make_transport(_json_response(_EMBEDDING_RESPONSE))
        client = HttpClient()
        client._sync_client = httpx.Client(transport=transport)

        config = PROVIDER_REGISTRY["openai"]
        result = client.embed(config, "sk-test", "hello", "text-embedding-3-small")

        assert isinstance(result, EmbeddingResponse)
        assert result.embedding == [0.1, 0.2, 0.3]
        assert result.embeddings == []
        assert result.model == "text-embedding-3-small"

    def test_embed_batch(self):
        transport = _make_transport(_json_response(_BATCH_EMBEDDING_RESPONSE))
        client = HttpClient()
        client._sync_client = httpx.Client(transport=transport)

        config = PROVIDER_REGISTRY["openai"]
        result = client.embed(config, "sk-test", ["text1", "text2"], "text-embedding-3-small")

        assert result.embedding is None
        assert len(result.embeddings) == 2

    def test_embed_unsupported_provider(self):
        """Providers whose adapter doesn't implement EmbeddingAdapter raise ProviderError."""
        client = HttpClient()
        config = PROVIDER_REGISTRY["anthropic"]
        with pytest.raises(ProviderError, match="does not support embeddings"):
            client.embed(config, "sk-test", "hello", "model")

    def test_embed_no_embedding_path(self):
        """Providers with embedding_path=None raise ProviderError."""
        config = ProviderConfig(
            name="test",
            base_url="https://example.com/v1",
            adapter="openai",
            api_key_env_vars=("KEY",),
            embedding_path=None,
        )
        client = HttpClient()
        with pytest.raises(ProviderError, match="no embedding endpoint"):
            client.embed(config, "key", "hello", "model")

    def test_embed_http_401(self):
        transport = _make_transport(_json_response({"error": {"message": "Bad key"}}, 401))
        client = HttpClient()
        client._sync_client = httpx.Client(transport=transport)

        config = PROVIDER_REGISTRY["openai"]
        with pytest.raises(AuthenticationError):
            client.embed(config, "bad-key", "hello", "model")


class TestHttpClientAembed:
    async def test_aembed_single(self):
        transport = _make_transport(_json_response(_EMBEDDING_RESPONSE))
        client = HttpClient()
        client._async_client = httpx.AsyncClient(transport=transport)

        config = PROVIDER_REGISTRY["openai"]
        result = await client.aembed(config, "sk-test", "hello", "text-embedding-3-small")

        assert isinstance(result, EmbeddingResponse)
        assert result.embedding == [0.1, 0.2, 0.3]

    async def test_aembed_batch(self):
        transport = _make_transport(_json_response(_BATCH_EMBEDDING_RESPONSE))
        client = HttpClient()
        client._async_client = httpx.AsyncClient(transport=transport)

        config = PROVIDER_REGISTRY["openai"]
        result = await client.aembed(config, "sk-test", ["a", "b"], "text-embedding-3-small")

        assert result.embedding is None
        assert len(result.embeddings) == 2
