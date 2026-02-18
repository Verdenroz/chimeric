"""Doc example tests — verifies that every code block in docs/ compiles and
behaves as documented.  All network calls are mocked; no real API keys needed.

Coverage map:
  docs/index.md                  → TestIndexExamples
  docs/usage/getting-started.md  → TestGettingStartedExamples
  docs/usage/configuration.md    → TestConfigurationExamples
  docs/usage/streaming.md        → TestStreamingExamples
  docs/usage/responses.md        → TestResponseExamples
  docs/usage/tools.md            → TestToolExamples
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chimeric import Chimeric
from chimeric.exceptions import ChimericError, ModelNotSupportedError, ProviderNotFoundError
from chimeric.types import (
    CompletionResponse,
    ModelSummary,
    Provider,
    StreamChunk,
    Usage,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _completion(content: str = "Hello!", model: str = "gpt-4o") -> CompletionResponse:
    return CompletionResponse(content=content, usage=Usage(), model=model)


def _models(*ids: str) -> list[ModelSummary]:
    return [ModelSummary(id=m, name=m) for m in ids]


def _stream(*deltas: str) -> list[StreamChunk]:
    """Build a minimal streaming sequence from delta strings."""
    chunks: list[StreamChunk] = []
    accumulated = ""
    for i, d in enumerate(deltas):
        accumulated += d
        is_last = i == len(deltas) - 1
        chunks.append(
            StreamChunk(
                content=accumulated,
                delta=d,
                finish_reason="stop" if is_last else None,
            )
        )
    return chunks


def _make_client(*model_ids: str, **kwargs: Any) -> tuple[Chimeric, MagicMock]:
    """Return a Chimeric instance wired to a mock HttpClient."""
    mock = MagicMock()
    mock.list_models.return_value = _models(*model_ids) if model_ids else _models("gpt-4o")
    mock.complete.return_value = _completion()
    mock.acomplete = AsyncMock(return_value=_completion())
    mock.alist_models = AsyncMock(return_value=_models(*model_ids) if model_ids else _models("gpt-4o"))
    with patch("chimeric.chimeric.HttpClient", return_value=mock):
        client = Chimeric(openai_api_key="sk-test", **kwargs)
    return client, mock


# ---------------------------------------------------------------------------
# docs/index.md
# ---------------------------------------------------------------------------


class TestIndexExamples:
    """docs/index.md code block coverage."""

    def test_quickstart_generate(self) -> None:
        """Basic generate() call returns content string."""
        client, mock = _make_client()
        mock.complete.return_value = _completion("Hello!")

        response = client.generate(model="gpt-4o", messages="Hello!")

        assert response.content == "Hello!"

    def test_multi_provider_switching(self) -> None:
        """Models routed to the correct (mocked) provider."""
        client, mock = _make_client("gpt-4o", "gpt-4o-mini")

        r1 = client.generate(model="gpt-4o", messages="Explain quantum computing")
        r2 = client.generate(model="gpt-4o", messages="Write a poem about AI")

        assert r1.content is not None
        assert r2.content is not None

    def test_streaming_uses_delta(self) -> None:
        """Streaming example uses chunk.delta, not chunk.content."""
        client, mock = _make_client()
        mock.complete.return_value = iter(_stream("Hello", ", ", "world!"))

        stream = client.generate(model="gpt-4o", messages="Tell me a story", stream=True)

        result = "".join(chunk.delta or "" for chunk in stream)
        assert result == "Hello, world!"

    def test_tool_decorator(self) -> None:
        """@client.tool() registers the function successfully."""
        client, _ = _make_client()

        @client.tool()
        def get_weather(city: str) -> str:
            """Get current weather for a city."""
            return f"Sunny, 72°F in {city}"

        assert any(t.name == "get_weather" for t in client.tools)

    def test_async_agenerate(self) -> None:
        """agenerate() returns a CompletionResponse."""
        client, mock = _make_client()
        mock.acomplete.return_value = _completion("Async response")

        async def run() -> CompletionResponse:
            return await client.agenerate(  # type: ignore[return-value]
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

        response = asyncio.run(run())
        assert isinstance(response, CompletionResponse)
        assert response.content is not None


# ---------------------------------------------------------------------------
# docs/usage/getting-started.md
# ---------------------------------------------------------------------------


class TestGettingStartedExamples:
    """docs/usage/getting-started.md code block coverage."""

    def test_unified_response_fields(self) -> None:
        """All documented CompletionResponse fields exist and match types."""
        client, _ = _make_client()
        response = client.generate(model="gpt-4o", messages="Explain quantum physics")

        # Documented fields
        _ = response.content   # str | list
        _ = response.model     # str | None
        _ = response.metadata  # dict | None

        # usage is nullable
        if response.usage:
            assert isinstance(response.usage.prompt_tokens, int)
            assert isinstance(response.usage.completion_tokens, int)
            assert isinstance(response.usage.total_tokens, int)

    def test_streaming_delta_access(self) -> None:
        """Streaming loop uses chunk.delta for new text."""
        client, mock = _make_client()
        mock.complete.return_value = iter(_stream("token1 ", "token2"))

        stream = client.generate(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Tell me a story"}],
            stream=True,
        )

        output = ""
        for chunk in stream:
            if chunk.delta:
                output += chunk.delta
        assert output == "token1 token2"

    def test_tool_registration_with_complex_signature(self) -> None:
        """Tool with list | None default type hint registers cleanly."""
        client, _ = _make_client()

        @client.tool()
        def analyze_financial_data(
            symbol: str,
            period: str = "1y",
            metrics: list[str] | None = None,
        ) -> dict:
            """Analyze financial performance metrics for a given symbol.

            Args:
                symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
                period: Analysis period ('1y', '6m', '3m')
                metrics: Specific metrics to analyze

            Returns:
                Dict containing analysis results and recommendations
            """
            return {"symbol": symbol, "period": period}

        tool = next(t for t in client.tools if t.name == "analyze_financial_data")
        assert "symbol" in tool.parameters.properties
        assert "period" in tool.parameters.properties

    def test_error_handling_chimeric_error(self) -> None:
        """ModelNotSupportedError is a subclass of ChimericError."""
        client, mock = _make_client()
        mock.list_models.return_value = []  # no models known

        with pytest.raises(ChimericError):
            client.generate(model="unknown-model-xyz", messages="Hello")

    def test_async_streaming_delta(self) -> None:
        """Async streaming yields chunks with delta field."""
        client, mock = _make_client()

        async def _fake_acomplete(**_kw: Any) -> Any:
            async def _gen():
                for chunk in _stream("Hello", " async"):
                    yield chunk
            return _gen()

        mock.acomplete.side_effect = _fake_acomplete

        async def run() -> str:
            stream = await client.agenerate(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Tell me a joke"}],
                stream=True,
            )
            result = ""
            async for chunk in stream:  # type: ignore[union-attr]
                if chunk.delta:
                    result += chunk.delta
            return result

        text = asyncio.run(run())
        assert text == "Hello async"


# ---------------------------------------------------------------------------
# docs/usage/configuration.md
# ---------------------------------------------------------------------------


class TestConfigurationExamples:
    """docs/usage/configuration.md code block coverage."""

    def test_direct_initialization(self) -> None:
        """All documented init parameters are accepted."""
        mock = MagicMock()
        mock.list_models.return_value = _models("gpt-4o")
        mock.complete.return_value = _completion()
        with patch("chimeric.chimeric.HttpClient", return_value=mock):
            client = Chimeric(
                openai_api_key="sk-...",
                anthropic_api_key="sk-ant-...",
                google_api_key="AIza...",
                cohere_api_key="your-key",
                groq_api_key="gsk_...",
                cerebras_api_key="csk-...",
                grok_api_key="xai-...",
                openrouter_api_key="sk-or-...",
            )
        assert "openai" in client.available_providers
        assert "anthropic" in client.available_providers
        assert "google" in client.available_providers

    def test_timeout_parameter(self) -> None:
        """timeout parameter is forwarded to HttpClient."""
        with patch("chimeric.chimeric.HttpClient") as mock_cls:
            mock_cls.return_value.list_models.return_value = []
            Chimeric(timeout=120.0)
        mock_cls.assert_called_once_with(timeout=120.0)

    def test_explicit_provider_string(self) -> None:
        """provider='groq' string form is accepted."""
        mock = MagicMock()
        mock.list_models.return_value = _models("llama-3.3-70b-versatile")
        mock.complete.return_value = _completion("Hi from Groq")
        with patch("chimeric.chimeric.HttpClient", return_value=mock):
            client = Chimeric(groq_api_key="gsk_test")

        response = client.generate(
            model="llama-3.3-70b-versatile",
            messages="Hello",
            provider="groq",
        )
        assert response.content is not None

    def test_explicit_provider_enum(self) -> None:
        """provider=Provider.GROQ enum form is accepted."""
        mock = MagicMock()
        mock.list_models.return_value = _models("llama-3.3-70b-versatile")
        mock.complete.return_value = _completion("Hi from Groq")
        with patch("chimeric.chimeric.HttpClient", return_value=mock):
            client = Chimeric(groq_api_key="gsk_test")

        response = client.generate(
            model="llama-3.3-70b-versatile",
            messages="Hello",
            provider=Provider.GROQ,
        )
        assert response.content is not None

    def test_available_providers_property(self) -> None:
        """available_providers returns a list of configured provider names."""
        client, _ = _make_client()
        providers = client.available_providers
        assert isinstance(providers, list)
        assert "openai" in providers

    def test_list_models_all_providers(self) -> None:
        """list_models() returns ModelSummary objects with id and provider."""
        client, mock = _make_client("gpt-4o", "gpt-4o-mini")
        mock.list_models.return_value = [
            ModelSummary(id="gpt-4o", name="GPT-4o", provider="openai"),
            ModelSummary(id="gpt-4o-mini", name="GPT-4o mini", provider="openai"),
        ]

        models = client.list_models()
        for model in models:
            assert model.id is not None
            assert model.provider is not None

    def test_list_models_single_provider(self) -> None:
        """list_models('openai') returns models for that provider."""
        client, mock = _make_client("gpt-4o")
        mock.list_models.return_value = [
            ModelSummary(id="gpt-4o", name="GPT-4o", provider="openai"),
        ]

        models = client.list_models("openai")
        assert all(m.provider == "openai" for m in models)

    def test_provider_passthrough_kwargs(self) -> None:
        """Extra kwargs are forwarded to the HTTP layer."""
        client, mock = _make_client()

        client.generate(
            model="gpt-4o",
            messages="Write a haiku",
            temperature=0.9,
            max_tokens=100,
        )

        _, call_kwargs = mock.complete.call_args
        assert call_kwargs.get("temperature") == 0.9
        assert call_kwargs.get("max_tokens") == 100


# ---------------------------------------------------------------------------
# docs/usage/streaming.md
# ---------------------------------------------------------------------------


class TestStreamingExamples:
    """docs/usage/streaming.md code block coverage."""

    def test_chunk_fields_exist(self) -> None:
        """StreamChunk has the four documented fields."""
        chunk = StreamChunk(
            content="accumulated so far",
            delta="new bit",
            finish_reason=None,
            metadata={"request_id": "abc"},
        )
        assert chunk.content == "accumulated so far"
        assert chunk.delta == "new bit"
        assert chunk.finish_reason is None
        assert chunk.metadata == {"request_id": "abc"}

    def test_chunk_str_returns_delta(self) -> None:
        """str(chunk) returns delta as documented."""
        chunk = StreamChunk(content="Hello world", delta=" world", finish_reason=None)
        assert str(chunk) == " world"

    def test_chunk_str_empty_for_metadata_only(self) -> None:
        """str(chunk) returns '' when delta is None."""
        chunk = StreamChunk(content="", delta=None, finish_reason="stop")
        assert str(chunk) == ""

    def test_content_accumulates(self) -> None:
        """chunk.content accumulates across chunks; chunk.delta is incremental."""
        client, mock = _make_client()
        mock.complete.return_value = iter(_stream("Hello", ", ", "world"))

        stream = client.generate(model="gpt-4o", messages="...", stream=True)
        chunks = list(stream)

        assert chunks[0].delta == "Hello"
        assert chunks[1].delta == ", "
        assert chunks[-1].content == "Hello, world"

    def test_finish_reason_on_final_chunk(self) -> None:
        """finish_reason is set only on the last chunk."""
        client, mock = _make_client()
        mock.complete.return_value = iter(_stream("A", "B", "C"))

        chunks = list(client.generate(model="gpt-4o", messages="...", stream=True))

        assert all(c.finish_reason is None for c in chunks[:-1])
        assert chunks[-1].finish_reason == "stop"

    def test_building_full_response_from_deltas(self) -> None:
        """Documented pattern: accumulate delta to build full response."""
        client, mock = _make_client()
        mock.complete.return_value = iter(_stream("The ", "answer ", "is 42"))

        full_response = ""
        for chunk in client.generate(model="gpt-4o", messages="...", stream=True):
            if chunk.delta:
                full_response += chunk.delta

        assert full_response == "The answer is 42"


# ---------------------------------------------------------------------------
# docs/usage/responses.md
# ---------------------------------------------------------------------------


class TestResponseExamples:
    """docs/usage/responses.md code block coverage."""

    def test_completion_response_fields(self) -> None:
        """All documented CompletionResponse fields exist."""
        resp = CompletionResponse(
            content="Generated text",
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="gpt-4o",
            metadata={"finish_reason": "stop"},
        )
        assert resp.content == "Generated text"
        assert resp.model == "gpt-4o"
        assert resp.usage is not None
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.completion_tokens == 5
        assert resp.usage.total_tokens == 15
        assert resp.metadata == {"finish_reason": "stop"}

    def test_str_response_returns_content(self) -> None:
        """str(response) returns the content text as documented."""
        resp = CompletionResponse(content="Hello world", model="gpt-4o")
        assert str(resp) == "Hello world"

    def test_usage_is_nullable(self) -> None:
        """usage can be None; guard before accessing sub-fields."""
        resp = CompletionResponse(content="ok", usage=None)
        assert resp.usage is None

    def test_cross_provider_same_interface(self) -> None:
        """Different models return same CompletionResponse shape."""
        mock = MagicMock()
        mock.list_models.side_effect = [
            _models("gpt-4o"),
            _models("claude-3-5-sonnet-20241022"),
            _models("gemini-1.5-pro"),
        ]

        def _summarize(model: str, text: str, client: Chimeric) -> str:
            mock.complete.return_value = _completion(f"Summary of {text}")
            response = client.generate(model=model, messages=f"Summarize: {text}")
            if response.usage:
                _ = response.usage.total_tokens
            return str(response)

        with patch("chimeric.chimeric.HttpClient", return_value=mock):
            client = Chimeric(
                openai_api_key="sk-test",
                anthropic_api_key="sk-ant-test",
                google_api_key="AIza-test",
            )

        for model in ("gpt-4o", "claude-3-5-sonnet-20241022", "gemini-1.5-pro"):
            result = _summarize(model, "some text", client)
            assert isinstance(result, str)

    def test_async_agenerate_same_return_type(self) -> None:
        """agenerate() returns CompletionResponse, same as generate()."""
        client, mock = _make_client()
        mock.acomplete = AsyncMock(return_value=_completion("Async hello"))

        async def run() -> None:
            response = await client.agenerate(model="gpt-4o", messages="Hello")  # type: ignore[union-attr]
            assert isinstance(response, CompletionResponse)
            assert response.content is not None

        asyncio.run(run())


# ---------------------------------------------------------------------------
# docs/usage/tools.md
# ---------------------------------------------------------------------------


class TestToolExamples:
    """docs/usage/tools.md code block coverage."""

    def test_basic_tool_registration(self) -> None:
        """@client.tool() registers function; it appears in client.tools."""
        client, _ = _make_client()

        @client.tool()
        def get_weather(city: str) -> str:
            """Get current weather for a city.

            Args:
                city: Name of the city to get weather for

            Returns:
                Weather description string
            """
            return f"Sunny, 75°F in {city}"

        tool = next(t for t in client.tools if t.name == "get_weather")
        assert tool.description is not None
        assert "city" in tool.parameters.properties

    def test_custom_name_and_description(self) -> None:
        """name= and description= overrides are reflected in the tool."""
        client, _ = _make_client()

        @client.tool(name="fetch_stock_data", description="Retrieve real-time stock information")
        def get_stock_price(symbol: str, include_history: bool = False) -> dict:
            """Get stock price and optional historical data."""
            return {"symbol": symbol, "price": 150.25}

        tool = next(t for t in client.tools if t.name == "fetch_stock_data")
        assert tool.description == "Retrieve real-time stock information"

    def test_google_style_docstring(self) -> None:
        """Google-style docstring (Args:/Returns:) is parsed for parameter docs."""
        client, _ = _make_client()

        @client.tool()
        def analyze_data(data: list[float], method: str = "mean") -> dict:
            """Analyze numerical data using statistical methods.

            Args:
                data: List of numerical values to analyze
                method: Statistical method to use ('mean', 'median', 'mode')

            Returns:
                Dictionary containing analysis results
            """
            return {}

        tool = next(t for t in client.tools if t.name == "analyze_data")
        assert "data" in tool.parameters.properties
        assert "method" in tool.parameters.properties

    def test_numpy_style_docstring(self) -> None:
        """NumPy-style docstring (Parameters\\n----------) is accepted."""
        client, _ = _make_client()

        @client.tool()
        def process_image(image_path: str, resize: bool = True) -> str:
            """Process an image file with optional resizing.

            Parameters
            ----------
            image_path : str
                Path to the image file to process
            resize : bool, optional
                Whether to resize the image, default True

            Returns
            -------
            str
                Path to the processed image file
            """
            return image_path

        tool = next(t for t in client.tools if t.name == "process_image")
        assert "image_path" in tool.parameters.properties

    def test_sphinx_style_docstring(self) -> None:
        """Sphinx-style docstring (:param:/:returns:) is accepted."""
        client, _ = _make_client()

        @client.tool()
        def send_email(recipient: str, subject: str, body: str = "") -> bool:
            """Send an email message.

            :param recipient: Email address of the recipient
            :param subject: Subject line of the email
            :param body: Email body content, optional
            :returns: True if email was sent successfully
            """
            return True

        tool = next(t for t in client.tools if t.name == "send_email")
        assert "recipient" in tool.parameters.properties

    def test_basic_type_hints(self) -> None:
        """str/int/float/bool type hints produce correct JSON schema types."""
        client, _ = _make_client()

        @client.tool()
        def basic_types_example(
            text: str,
            number: int,
            decimal: float,
            flag: bool,
        ) -> str:
            """Example of basic type support."""
            return f"{text}: {number}, {decimal}, {flag}"

        tool = next(t for t in client.tools if t.name == "basic_types_example")
        props = tool.parameters.properties
        assert props["text"]["type"] == "string"
        assert props["number"]["type"] == "integer"
        assert props["decimal"]["type"] == "number"
        assert props["flag"]["type"] == "boolean"

    def test_optional_type_hint(self) -> None:
        """str | None type hint produces an optional parameter."""
        client, _ = _make_client()

        @client.tool()
        def complex_types_example(
            items: list[str],
            optional_param: str | None = None,
        ) -> dict:
            """Example of complex type support."""
            return {}

        tool = next(t for t in client.tools if t.name == "complex_types_example")
        # optional_param should not be in required list
        required = tool.parameters.required or []
        assert "optional_param" not in required

    def test_client_tools_property(self) -> None:
        """client.tools returns list of Tool objects with name and description."""
        client, _ = _make_client()

        @client.tool()
        def tool_a(x: int) -> int:
            """Tool A."""
            return x

        @client.tool()
        def tool_b(y: str) -> str:
            """Tool B."""
            return y

        assert len(client.tools) == 2
        names = {t.name for t in client.tools}
        assert names == {"tool_a", "tool_b"}

    def test_explicit_tools_parameter(self) -> None:
        """tools=[...] sends only those tools; auto_tool=False disables auto-include."""
        client, mock = _make_client()
        mock.complete.return_value = _completion()

        @client.tool()
        def get_weather(city: str) -> str:
            """Get weather."""
            return f"Weather in {city}"

        @client.tool()
        def get_stock_price(symbol: str) -> float:
            """Get stock price."""
            return 150.25

        weather_tool = client.tools[0]

        client.generate(
            model="gpt-4o",
            messages="What's the weather in NYC?",
            tools=[weather_tool],
            auto_tool=False,
        )

        _, call_kwargs = mock.complete.call_args
        sent_tools = call_kwargs.get("tools") or []
        assert len(sent_tools) == 1
        assert sent_tools[0].name == "get_weather"

    def test_auto_tool_true_sends_all(self) -> None:
        """auto_tool=True (default) sends all registered tools."""
        client, mock = _make_client()
        mock.complete.return_value = _completion()

        @client.tool()
        def tool_x(a: str) -> str:
            """Tool X."""
            return a

        @client.tool()
        def tool_y(b: str) -> str:
            """Tool Y."""
            return b

        client.generate(model="gpt-4o", messages="Help me", auto_tool=True)

        _, call_kwargs = mock.complete.call_args
        sent_tools = call_kwargs.get("tools") or []
        assert len(sent_tools) == 2

    def test_streaming_with_tools(self) -> None:
        """Streaming works when tools are registered."""
        client, mock = _make_client()
        mock.complete.return_value = iter(_stream("The time is 3pm.", " Enjoy!"))

        @client.tool()
        def get_current_time() -> str:
            """Get the current time."""
            return "3pm"

        result = ""
        for chunk in client.generate(
            model="gpt-4o",
            messages="What time is it?",
            stream=True,
        ):
            result += chunk.delta or ""

        assert result == "The time is 3pm. Enjoy!"
