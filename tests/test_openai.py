from collections.abc import AsyncGenerator, Generator
import os
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from chimeric import Chimeric
from chimeric.exceptions import ProviderError
import chimeric.providers.openai.client as client_module
from chimeric.providers.openai.client import OpenAIClient
from chimeric.types import (
    ChimericCompletionResponse,
    ChimericStreamChunk,
    CompletionResponse,
    FileUploadResponse,
    ModelInfo,
    ModelSummary,
    StreamChunk,
    Tool,
    ToolType,
)


@pytest.fixture(scope="module")
def chimeric_openai():
    """Create a Chimeric instance configured for OpenAI."""
    return Chimeric(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://test.openai.com/v1",
        timeout=120,
        max_retries=2,
        organization="test_org",
        project="chimeric_openai_test",
    )


@pytest.fixture(scope="module")
def chimeric_openai_client(chimeric_openai) -> OpenAIClient:
    """Get the OpenAIClient from the Chimeric wrapper."""
    return cast("OpenAIClient", chimeric_openai.get_provider_client("openai"))


@pytest.fixture(autouse=True)
def patch_openai_imports(monkeypatch):
    """Stub out actual OpenAI classes to prevent network calls.

    Also set up placeholder Stream types for streaming tests.
    """

    # Create mock implementations for sync and async SDK entrypoints
    def create_openai_mock(api_key: str, **kw: Any) -> SimpleNamespace:
        return SimpleNamespace(api_key=api_key, **kw)

    def create_async_openai_mock(api_key: str, **kw: Any) -> SimpleNamespace:
        return SimpleNamespace(api_key=api_key, **kw)

    monkeypatch.setattr(client_module, "OpenAI", create_openai_mock)
    monkeypatch.setattr(client_module, "AsyncOpenAI", create_async_openai_mock)

    # Create stub Stream types for isinstance checks
    class MockStreamType:
        pass

    class MockAsyncStreamType:
        pass

    monkeypatch.setattr(client_module, "Stream", MockStreamType)
    monkeypatch.setattr(client_module, "AsyncStream", MockAsyncStreamType)
    return


class MockResponse:
    """Mock implementation of a non-streaming OpenAI response."""

    def __init__(self) -> None:
        self.output_text = "hello"
        self.usage = SimpleNamespace(input_tokens=1, output_tokens=2, total_tokens=3)
        self.model = "model-x"

    @staticmethod
    def model_dump() -> dict[str, Any]:
        return {"dumped": True}


class MockFile:
    """Mock implementation of openai.File.create() return value."""

    def __init__(self) -> None:
        self.id = "file-id"
        self.filename = "file.txt"
        self.bytes = 123
        self.purpose = "test"
        self.created_at = 456
        self.status = "ready"

    @staticmethod
    def model_dump() -> dict[str, Any]:
        return {"md": "v"}


# noinspection PyUnusedLocal
class TestOpenAIClient:
    """Tests for the OpenAIClient class."""

    def test_capabilities_and_supports(self, chimeric_openai_client):
        """Test that client reports correct capabilities and support methods."""
        client = chimeric_openai_client
        caps = client.capabilities

        # Verify all expected capabilities are enabled
        assert caps.multimodal
        assert caps.streaming
        assert caps.tools
        assert caps.agents
        assert caps.files

        # Verify the supports_* API methods return correct values
        assert client.supports_multimodal()
        assert client.supports_streaming()
        assert client.supports_tools()
        assert client.supports_agents()
        assert client.supports_files()

    def test_list_models_maps_to_summary(self, chimeric_openai_client, monkeypatch):
        """Test that list_models correctly maps raw model data to ModelSummary objects."""
        client = chimeric_openai_client
        mock = SimpleNamespace(id="m1", owned_by="owner1", created=161803398)

        def mock_list():
            return [mock]

        monkeypatch.setattr(client._client.models, "list", mock_list)
        models = client.list_models()

        # Verify single model is returned and properly mapped
        assert len(models) == 1
        ms = models[0]
        assert isinstance(ms, ModelSummary)
        assert ms.id == "m1"
        assert ms.owned_by == "owner1"
        assert ms.created_at == 161803398

    @pytest.mark.parametrize(
        ("initial", "ev", "expect_acc", "expect_chunk"),
        [
            # Test delta event processing
            (
                "",
                Mock(
                    type="response.output_text.delta",
                    delta="A",
                    model_dump=Mock(return_value={"m": 1}),
                ),
                "A",
                ("A", "A", {"m": 1}),
            ),
            # Test completion event with final text
            (
                "PRE",
                Mock(
                    type="response.completed",
                    response=Mock(
                        output=[Mock(content=[Mock(text="FIN")])],
                        status="DONE",
                    ),
                    model_dump=Mock(return_value={"x": 2}),
                ),
                "PRE",
                ("FIN", None, {"x": 2}, "DONE"),
            ),
            # Test completion with empty output using accumulated text
            (
                "XYZ",
                Mock(
                    type="response.completed",
                    response=Mock(output=[], status="R"),
                    model_dump=Mock(return_value={"y": 3}),
                ),
                "XYZ",
                ("XYZ", None, {"y": 3}, "R"),
            ),
            # Test unknown event type handling
            (
                "ACC",
                Mock(type="something.else", model_dump=Mock(return_value={})),
                "ACC",
                None,
            ),
        ],
    )
    def test_process_event_branches(self, initial, ev, expect_acc, expect_chunk):
        """Test _process_event handles different event types correctly."""
        acc, chunk = OpenAIClient._process_event(ev, initial)
        assert acc == expect_acc

        if expect_chunk is None:
            assert chunk is None
        else:
            if ev.type == "response.output_text.delta":
                # Verify delta event creates proper StreamChunk
                assert isinstance(chunk.common, StreamChunk)
                assert chunk.common.content == expect_chunk[0]
                assert chunk.common.delta == expect_chunk[1]
                assert chunk.common.metadata == expect_chunk[2]
            else:
                # Verify completion event creates proper StreamChunk
                content, _, metadata, finish = expect_chunk
                assert chunk.common.content == content
                assert chunk.common.finish_reason == finish
                assert chunk.common.metadata == metadata

    def test_stream_filters_none_and_orders(self, chimeric_openai_client):
        """Test that _stream filters out None chunks and maintains order."""
        client = chimeric_openai_client

        # Create test events: valid delta, skip event, valid completion
        e1 = Mock(
            type="response.output_text.delta", delta="x", model_dump=Mock(return_value={"a": 1})
        )
        e2 = Mock(type="skip", model_dump=Mock(return_value={}))
        e3 = Mock(
            type="response.completed",
            response=Mock(output=[Mock(content=[Mock(text="Y")])], status="OK"),
            model_dump=Mock(return_value={"b": 2}),
        )

        # Process events and verify only valid ones are returned in order
        # noinspection PyTypeChecker
        out = list(client._stream([e1, e2, e3]))
        assert [c.common.content for c in out] == ["x", "Y"]

    async def test_astream_filters_none_chunks(self, chimeric_openai_client):
        """Test that _astream filters out None chunks from async generator."""
        client = chimeric_openai_client

        async def agen():
            # Yield valid delta, skip event, and valid completion
            yield Mock(
                type="response.output_text.delta", delta="d", model_dump=Mock(return_value={"m": 1})
            )
            yield Mock(type="skip", model_dump=Mock(return_value={}))
            yield Mock(
                type="response.completed",
                response=Mock(output=[Mock(content=[Mock(text="Z")])], status="FIN"),
                model_dump=Mock(return_value={"m": 2}),
            )

        # Collect chunks and verify only valid ones are processed
        out = []
        # noinspection PyTypeChecker
        async for chunk in client._astream(agen()):
            out.append(chunk)

        assert [c.common.content for c in out] == ["d", "Z"]

    def test_format_response_sync(self, chimeric_openai_client):
        """Test synchronous response formatting."""
        client = chimeric_openai_client
        resp = MockResponse()

        # Format response and verify structure
        # noinspection PyTypeChecker
        formatted = client._format_response(resp)
        assert formatted.native is resp
        cr = formatted.common
        assert isinstance(cr, CompletionResponse)
        assert cr.content == "hello"
        assert cr.metadata == {"dumped": True}

    def test_format_response_sync_stream(self, monkeypatch, chimeric_openai_client):
        """Test synchronous stream response formatting."""
        client = chimeric_openai_client

        def mock_stream(s: Any) -> str:
            return "STREAM_OK"

        monkeypatch.setattr(client, "_stream", mock_stream)
        # Create a mock Stream object and verify it's handled correctly
        mock = Mock(spec=client_module.Stream)
        assert client._format_response(mock) == "STREAM_OK"

    async def test_aformat_response_async(self, chimeric_openai_client):
        """Test asynchronous response formatting."""
        client = chimeric_openai_client
        resp = MockResponse()

        # Format async response and verify structure
        # noinspection PyTypeChecker
        formatted = await client._aformat_response(resp)
        assert formatted.native is resp
        assert isinstance(formatted.common, CompletionResponse)

    async def test_aformat_response_async_stream(self, monkeypatch, chimeric_openai_client):
        """Test asynchronous stream response formatting."""
        client = chimeric_openai_client

        def mock_astream(s: Any) -> str:
            return "ASTREAM_OK"

        monkeypatch.setattr(client, "_astream", mock_astream)
        # Create a mock AsyncStream object and verify it's handled correctly
        mock = AsyncMock(spec=client_module.AsyncStream)
        assert await client._aformat_response(mock) == "ASTREAM_OK"

    def test_chat_completion_counts_and_errors(self, chimeric_openai_client, monkeypatch):
        """Test that chat_completion properly tracks request and error counts."""
        client = chimeric_openai_client

        # Test successful request path
        def mock_impl(messages: Any, model: Any, tools: Any = None, **kw: Any) -> str:
            return "DONE"

        monkeypatch.setattr(client, "_chat_completion_impl", mock_impl)
        before_req = client.request_count
        before_err = client.error_count
        out = client.chat_completion("hi", "m1")

        # Verify success increments request count but not error count
        assert out == "DONE"
        assert client.request_count == before_req + 1
        assert client.error_count == 0

        # Test error handling path
        def bad_impl(*args: Any, **kw: Any):
            raise ValueError("fail")

        monkeypatch.setattr(client, "_chat_completion_impl", bad_impl)
        with pytest.raises(ValueError):
            client.chat_completion([], "m2")

        # Verify error increments both request and error counts
        assert client.request_count == before_req + 2
        assert client.error_count == before_err + 1

    async def test_achat_completion_counts_and_errors(self, chimeric_openai_client, monkeypatch):
        """Test that achat_completion properly tracks request and error counts."""
        client = chimeric_openai_client

        # Test successful async request
        async def ok_impl(*args: Any, **kw: Any) -> str:
            return "OK"

        monkeypatch.setattr(client, "_achat_completion_impl", ok_impl)
        before_req = client.request_count
        before_err = client.error_count

        # Verify successful async call increments request count
        assert await client.achat_completion("x", "m") == "OK"
        assert client.request_count == before_req + 1

        # Test async error handling
        async def err_impl(*args: Any, **kw: Any):
            raise RuntimeError("err")

        monkeypatch.setattr(client, "_achat_completion_impl", err_impl)
        with pytest.raises(RuntimeError):
            await client.achat_completion([], "m")

        # Verify error increments error count
        assert client.error_count == before_err + 1

    def test_get_model_info_found_and_notfound(self, chimeric_openai_client, monkeypatch):
        """Test get_model_info for both found and not found models."""
        client = chimeric_openai_client
        ms = ModelSummary(id="MID", name="N", created_at=0, owned_by="")

        def mock_list_models():
            return [ms]

        monkeypatch.setattr(client, "list_models", mock_list_models)

        # Test finding existing model
        info = client.get_model_info("MID")
        assert isinstance(info, ModelInfo)

        # Test error when model not found
        with pytest.raises(ValueError):
            client.get_model_info("NOPE")

    def test_upload_file_success(self, chimeric_openai_client, monkeypatch):
        """Test successful file upload."""
        client = chimeric_openai_client
        mock = MockFile()

        def mock_create(**kw: Any) -> MockFile:
            return mock

        monkeypatch.setattr(client._client.files, "create", mock_create)
        before = client.request_count

        # Perform upload and verify response structure
        resp = client.upload_file(file="x")
        assert resp.native is mock
        cf = resp.common
        assert isinstance(cf, FileUploadResponse)
        assert cf.file_id == "file-id"

        # Verify request count incremented
        assert client.request_count == before + 1

    def test_upload_file_error(self, chimeric_openai_client, monkeypatch):
        """Test file upload error handling."""
        client = chimeric_openai_client

        def boom(**kw: Any):
            raise RuntimeError("broken")

        monkeypatch.setattr(client._client.files, "create", boom)

        # Verify error is wrapped in ProviderError
        with pytest.raises(ProviderError) as ei:
            client.upload_file(x=1)

        # Check that the original error message is preserved
        response_text = ei.value.response_text
        assert response_text is not None
        assert "broken" in response_text

    def test_sync_context_manager_closes(self, monkeypatch, chimeric_openai_client):
        """Test that sync context manager properly closes the client."""

        class C:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        # Replace client with mock and verify close is called
        inst = C()
        chimeric_openai_client._client = inst
        with chimeric_openai_client:
            pass
        assert inst.closed

    async def test_async_context_manager_closes(self, monkeypatch, chimeric_openai_client):
        """Test that async context manager properly closes the client."""

        class AC:
            def __init__(self):
                self.acl = False
                self.cl = False

            def aclose(self):
                async def _():
                    self.acl = True

                return _()

            def close(self):
                self.cl = True

        # Replace async client with mock and verify both close methods called
        inst = AC()
        chimeric_openai_client._async_client = inst
        async with chimeric_openai_client:
            pass
        assert inst.acl
        assert inst.cl

    def test_repr_and_str_contains_counts(self, chimeric_openai_client):
        """Test that repr and str methods include request/error counts."""
        # Verify repr contains expected components
        r = repr(chimeric_openai_client)
        assert "OpenAIClient" in r
        assert "requests=" in r
        assert "errors=" in r

        # Verify str contains expected components
        s = str(chimeric_openai_client)
        assert "OpenAIClient Client" in s
        assert "- Requests:" in s
        assert "- Errors:" in s

    def test_chat_completion_impl_non_streaming(self, chimeric_openai_client, monkeypatch):
        """Test _chat_completion_impl with non-streaming response."""
        client = chimeric_openai_client

        # Create a mock response object
        mock_response = Mock()
        mock_response.output_text = "Hello, world!"
        mock_response.usage = SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15)
        mock_response.model = "gpt-4"
        mock_response.model_dump.return_value = {"response_data": "test"}

        # Mock the client's responses.create method
        mock_create = Mock(return_value=mock_response)
        client._client.responses = SimpleNamespace(create=mock_create)

        # Mock _filter_kwargs to properly filter out invalid parameters
        def mock_filter_kwargs(func, kwargs):
            # Only allow known valid parameters
            allowed = {"temperature", "max_tokens", "stream"}
            return {k: v for k, v in kwargs.items() if k in allowed}

        monkeypatch.setattr(client, "_filter_kwargs", mock_filter_kwargs)

        # Test the implementation
        result = client._chat_completion_impl(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
            invalid_param="should_be_filtered",
        )

        # Verify the result
        assert isinstance(result, ChimericCompletionResponse)
        assert result.native is mock_response
        assert isinstance(result.common, CompletionResponse)
        assert result.common.content == "Hello, world!"
        assert result.common.model == "gpt-4"

        # Verify the client was called correctly
        mock_create.assert_called_once_with(
            model="gpt-4",
            input=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
        )

    def test_chat_completion_impl_streaming(self, chimeric_openai_client, monkeypatch):
        """Test _chat_completion_impl with streaming response."""
        client = chimeric_openai_client

        # Create mock stream response
        mock_stream = Mock(spec=client_module.Stream)

        # Mock the client's responses.create method to return stream
        mock_create = Mock(return_value=mock_stream)
        client._client.responses = SimpleNamespace(create=mock_create)

        # Mock _stream method to return a generator
        def mock_stream_method(stream):
            yield ChimericStreamChunk(
                native=Mock(), common=StreamChunk(content="Hello", delta="Hello", metadata={})
            )
            yield ChimericStreamChunk(
                native=Mock(),
                common=StreamChunk(content="Hello world", delta=" world", metadata={}),
            )

        monkeypatch.setattr(client, "_stream", mock_stream_method)

        # Test the implementation
        result = client._chat_completion_impl(
            messages=[{"role": "user", "content": "Hello"}], model="gpt-4", stream=True
        )

        # Verify the result is a generator
        assert isinstance(result, Generator)

        # Collect all chunks
        chunks = list(result)
        assert len(chunks) == 2
        assert chunks[0].common.content == "Hello"
        assert chunks[1].common.content == "Hello world"

        # Verify the client was called correctly
        mock_create.assert_called_once_with(
            model="gpt-4", input=[{"role": "user", "content": "Hello"}], stream=True
        )

    def test_chat_completion_impl_with_tools(self, chimeric_openai_client, monkeypatch):
        """Test _chat_completion_impl with tool parameter."""
        client = chimeric_openai_client

        # Create mock response
        mock_response = Mock()
        mock_response.output_text = "Function called"
        mock_response.usage = SimpleNamespace(input_tokens=20, output_tokens=10, total_tokens=30)
        mock_response.model = "gpt-4"
        mock_response.model_dump.return_value = {"tools_used": True}

        mock_create = Mock(return_value=mock_response)
        client._client.responses = SimpleNamespace(create=mock_create)

        # Test with tools
        tools = [Tool(type=ToolType.FUNCTION, name="get_weather", description="Get weather info")]

        client._chat_completion_impl(
            messages=[{"role": "user", "content": "What's the weather?"}],
            model="gpt-4",
            tools=tools,
        )

        # Verify tools parameter is handled (note: tools might be transformed or ignored)
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert call_args[1]["model"] == "gpt-4"
        assert call_args[1]["input"] == [{"role": "user", "content": "What's the weather?"}]

    async def test_achat_completion_impl_non_streaming(self, chimeric_openai_client, monkeypatch):
        """Test _achat_completion_impl with non-streaming response."""
        client = chimeric_openai_client

        # Create mock async response
        mock_response = Mock()
        mock_response.output_text = "Async hello!"
        mock_response.usage = SimpleNamespace(input_tokens=8, output_tokens=3, total_tokens=11)
        mock_response.model = "gpt-3.5-turbo"
        mock_response.model_dump.return_value = {"async": True}

        # Mock the async client's responses.create method
        mock_create = AsyncMock(return_value=mock_response)
        client._async_client.responses = SimpleNamespace(create=mock_create)

        # Test the async implementation
        result = await client._achat_completion_impl(
            messages=[{"role": "user", "content": "Async hello"}],
            model="gpt-3.5-turbo",
            temperature=0.5,
        )

        # Verify the result
        assert isinstance(result, ChimericCompletionResponse)
        assert result.native is mock_response
        assert result.common.content == "Async hello!"
        assert result.common.model == "gpt-3.5-turbo"

        # Verify the async client was called correctly
        mock_create.assert_called_once_with(
            model="gpt-3.5-turbo",
            input=[{"role": "user", "content": "Async hello"}],
            temperature=0.5,
        )

    async def test_achat_completion_impl_streaming(self, chimeric_openai_client, monkeypatch):
        """Test _achat_completion_impl with streaming response."""
        client = chimeric_openai_client

        # Create mock async stream
        mock_stream = Mock(spec=client_module.AsyncStream)

        mock_create = AsyncMock(return_value=mock_stream)
        client._async_client.responses = SimpleNamespace(create=mock_create)

        # Mock _astream method to return an async generator
        async def mock_astream_method(stream):
            yield ChimericStreamChunk(
                native=Mock(), common=StreamChunk(content="Async", delta="Async", metadata={})
            )
            yield ChimericStreamChunk(
                native=Mock(),
                common=StreamChunk(content="Async streaming", delta=" streaming", metadata={}),
            )

        monkeypatch.setattr(client, "_astream", mock_astream_method)

        # Test the async implementation
        result = await client._achat_completion_impl(
            messages=[{"role": "user", "content": "Stream test"}], model="gpt-4", stream=True
        )

        # Verify the result is an async generator
        assert isinstance(result, AsyncGenerator)

        # Collect all chunks
        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].common.content == "Async"
        assert chunks[1].common.content == "Async streaming"

        # Verify the async client was called correctly
        mock_create.assert_called_once_with(
            model="gpt-4", input=[{"role": "user", "content": "Stream test"}], stream=True
        )

    def test_chat_completion_impl_kwargs_filtering(self, chimeric_openai_client, monkeypatch):
        """Test that _chat_completion_impl properly filters kwargs."""
        client = chimeric_openai_client

        mock_response = Mock()
        mock_response.output_text = "Filtered"
        mock_response.usage = SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2)
        mock_response.model = "gpt-4"
        mock_response.model_dump.return_value = {}

        mock_create = Mock(return_value=mock_response)
        client._client.responses = SimpleNamespace(create=mock_create)

        # Mock _filter_kwargs to only allow specific parameters
        def mock_filter_kwargs(func, kwargs):
            allowed = {"temperature", "max_tokens", "stream"}
            return {k: v for k, v in kwargs.items() if k in allowed}

        monkeypatch.setattr(client, "_filter_kwargs", mock_filter_kwargs)

        # Test with both valid and invalid kwargs
        client._chat_completion_impl(
            messages=[{"role": "user", "content": "Test"}],
            model="gpt-4",
            temperature=0.8,
            max_tokens=50,
            invalid_param="should_be_removed",
            another_invalid="also_removed",
        )

        # Verify only valid kwargs were passed
        mock_create.assert_called_once_with(
            model="gpt-4",
            input=[{"role": "user", "content": "Test"}],
            temperature=0.8,
            max_tokens=50,
        )

    async def test_achat_completion_impl_kwargs_filtering(
        self, chimeric_openai_client, monkeypatch
    ):
        """Test that _achat_completion_impl properly filters kwargs."""
        client = chimeric_openai_client

        mock_response = Mock()
        mock_response.output_text = "Async filtered"
        mock_response.usage = SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2)
        mock_response.model = "gpt-4"
        mock_response.model_dump.return_value = {}

        mock_create = AsyncMock(return_value=mock_response)
        client._async_client.responses = SimpleNamespace(create=mock_create)

        # Mock _filter_kwargs to only allow specific parameters
        def mock_filter_kwargs(func, kwargs):
            allowed = {"temperature", "top_p"}
            return {k: v for k, v in kwargs.items() if k in allowed}

        monkeypatch.setattr(client, "_filter_kwargs", mock_filter_kwargs)

        # Test with both valid and invalid kwargs
        await client._achat_completion_impl(
            messages=[{"role": "user", "content": "Async test"}],
            model="gpt-4",
            temperature=0.9,
            top_p=0.95,
            presence_penalty=0.1,  # Should be filtered out
            frequency_penalty=0.1,  # Should be filtered out
        )

        # Verify only valid kwargs were passed
        mock_create.assert_called_once_with(
            model="gpt-4",
            input=[{"role": "user", "content": "Async test"}],
            temperature=0.9,
            top_p=0.95,
        )

    def test_chat_completion_impl_empty_messages(self, chimeric_openai_client, monkeypatch):
        """Test _chat_completion_impl with an empty messages list."""
        client = chimeric_openai_client

        mock_response = Mock()
        mock_response.output_text = "Empty response"
        mock_response.usage = SimpleNamespace(input_tokens=0, output_tokens=2, total_tokens=2)
        mock_response.model = "gpt-4"
        mock_response.model_dump.return_value = {}

        mock_create = Mock(return_value=mock_response)
        client._client.responses = SimpleNamespace(create=mock_create)

        # Test with empty messages
        result = client._chat_completion_impl(messages=[], model="gpt-4")

        # Verify the call was made with empty input
        mock_create.assert_called_once_with(model="gpt-4", input=[])
        assert isinstance(result, ChimericCompletionResponse)

    async def test_achat_completion_impl_with_complex_messages(
        self, chimeric_openai_client, monkeypatch
    ):
        """Test _achat_completion_impl with complex message structures."""
        client = chimeric_openai_client

        mock_response = Mock()
        mock_response.output_text = "Complex response"
        mock_response.usage = SimpleNamespace(input_tokens=50, output_tokens=20, total_tokens=70)
        mock_response.model = "gpt-4"
        mock_response.model_dump.return_value = {"complex": True}

        mock_create = AsyncMock(return_value=mock_response)
        client._async_client.responses = SimpleNamespace(create=mock_create)

        # Test with complex messages including the system, user, and assistant
        complex_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        result = await client._achat_completion_impl(
            messages=complex_messages, model="gpt-4", temperature=0.7
        )

        # Verify the complex messages were passed correctly
        mock_create.assert_called_once_with(model="gpt-4", input=complex_messages, temperature=0.7)
        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Complex response"
