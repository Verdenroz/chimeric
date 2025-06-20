from collections.abc import AsyncGenerator, Generator
import json
import os
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

from openai import NOT_GIVEN
import pytest

from chimeric import Chimeric
from chimeric.exceptions import ProviderError, ToolRegistrationError
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
    ToolParameters,
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

    def test_chat_completion_counts_and_errors(self, chimeric_openai_client, monkeypatch):
        """Test that chat_completion properly tracks request and error counts."""
        client = chimeric_openai_client

        # Test successful request path
        def mock_impl(messages: Any, model: Any, stream: Any, tools: Any = None, **kw: Any) -> str:
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

    def test_encode_tools(self, chimeric_openai_client):
        """Test that _encode_tools correctly formats tools for the OpenAI API."""
        client = chimeric_openai_client

        # Test with None tools
        assert client._encode_tools(None) is None

        # Create a tool with parameters
        tool_params = ToolParameters(
            properties={"location": {"type": "string", "description": "City name"}},
            required=["location"],
        )
        tool_with_params = Tool(
            name="get_weather",
            description="Get weather information for a location",
            parameters=tool_params,
        )

        # Create a tool without parameters
        tool_without_params = Tool(name="get_time", description="Get current server time")

        # Test with a list of Tool instances
        encoded_tools = client._encode_tools([tool_with_params, tool_without_params])
        assert isinstance(encoded_tools, list)
        assert len(encoded_tools) == 2

        # Verify first tool (with parameters)
        assert encoded_tools[0]["type"] == "function"
        assert encoded_tools[0]["name"] == "get_weather"
        assert encoded_tools[0]["description"] == "Get weather information for a location"
        assert encoded_tools[0]["parameters"] == tool_params.model_dump()

        # Verify the second tool (without parameters)
        assert encoded_tools[1]["type"] == "function"
        assert encoded_tools[1]["name"] == "get_time"
        assert encoded_tools[1]["description"] == "Get current server time"
        assert encoded_tools[1]["parameters"] is None

        # Test with pre-formatted tool dictionary
        pre_formatted_tool = {
            "type": "function",
            "name": "custom_tool",
            "description": "A pre-formatted tool",
            "parameters": {"type": "object"},
        }

        # Test with mix of Tool instances and pre-formatted tools
        mixed_tools = client._encode_tools([tool_with_params, pre_formatted_tool])
        assert isinstance(mixed_tools, list)
        assert len(mixed_tools) == 2
        assert mixed_tools[0]["name"] == "get_weather"
        assert mixed_tools[1] == pre_formatted_tool  # Should be passed through unchanged

    def test_chat_completion_impl_non_streaming(self, chimeric_openai_client, monkeypatch):
        """Test _chat_completion_impl with non-streaming response."""
        client = chimeric_openai_client

        # Create a mock response object
        mock_response = Mock()
        mock_response.output_text = "Hello, world!"
        mock_response.usage = SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15)
        mock_response.model = "gpt-4"
        mock_response.model_dump.return_value = {"response_data": "test"}
        mock_response.output = ["Hello, world!"]

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
            stream=False,
            tools=NOT_GIVEN,
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
            model="gpt-4",
            input=[{"role": "user", "content": "Hello"}],
            tools=NOT_GIVEN,
            stream=True,
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
        mock_response.output = ["Function called"]

        mock_create = Mock(return_value=mock_response)
        client._client.responses = SimpleNamespace(create=mock_create)

        messages = [{"role": "user", "content": "What's the weather?"}]
        model = "gpt-4"
        # Test with tools
        tools = [Tool(name="get_weather", description="Get weather info")]

        client._chat_completion_impl(
            messages=messages,
            model=model,
            tools=tools,
        )

        # Verify tools parameter is handled (note: tools might be transformed or ignored)
        mock_create.assert_called_once_with(
            model=model,
            input=messages,
            stream=False,
            tools=tools,
        )

    async def test_achat_completion_impl_non_streaming(self, chimeric_openai_client, monkeypatch):
        """Test _achat_completion_impl with non-streaming response."""
        client = chimeric_openai_client

        # Create mock async response
        mock_response = Mock()
        mock_response.output_text = "Async hello!"
        mock_response.usage = SimpleNamespace(input_tokens=8, output_tokens=3, total_tokens=11)
        mock_response.model = "gpt-3.5-turbo"
        mock_response.model_dump.return_value = {"async": True}
        mock_response.output = ["Async hello!"]

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
            stream=False,
            tools=NOT_GIVEN,
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
            model="gpt-4",
            input=[{"role": "user", "content": "Stream test"}],
            tools=NOT_GIVEN,
            stream=True,
        )

    def test_chat_completion_impl_kwargs_filtering(self, chimeric_openai_client, monkeypatch):
        """Test that _chat_completion_impl properly filters kwargs."""
        client = chimeric_openai_client

        mock_response = Mock()
        mock_response.output_text = "Filtered"
        mock_response.usage = SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2)
        mock_response.model = "gpt-4"
        mock_response.model_dump.return_value = {}
        mock_response.output = ["Filtered"]

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
            stream=False,
            tools=NOT_GIVEN,
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
        mock_response.output = ["Async filtered"]

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
            stream=False,
            tools=NOT_GIVEN,
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
        mock_response.output = ["Empty response"]

        mock_create = Mock(return_value=mock_response)
        client._client.responses = SimpleNamespace(create=mock_create)

        # Test with empty messages
        result = client._chat_completion_impl(messages=[], model="gpt-4")

        # Verify the call was made with empty input
        mock_create.assert_called_once_with(model="gpt-4", input=[], stream=False, tools=NOT_GIVEN)
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
        mock_response.output = ["Complex response"]

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
        mock_create.assert_called_once_with(
            model="gpt-4", input=complex_messages, stream=False, tools=NOT_GIVEN, temperature=0.7
        )
        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Complex response"

    def test_process_function_call_success(self, chimeric_openai_client, monkeypatch):
        """Test successful function call processing."""
        client = chimeric_openai_client

        # Create a mock tool function
        def mock_weather_function(location: str, unit: str = "celsius") -> str:
            return f"Weather in {location} is 22°{unit[0].upper()}"

        # Create and register a tool
        tool = Tool(
            name="get_weather",
            description="Get weather for a location",
            function=mock_weather_function,
        )
        client.tool_manager.register(
            func=tool.function,
            name=tool.name,
            description=tool.description,
        )

        # Create a mock function call
        mock_call = Mock()
        mock_call.name = "get_weather"
        mock_call.call_id = "call_123"
        mock_call.arguments = json.dumps({"location": "New York", "unit": "fahrenheit"})

        # Test the function call processing
        result = client._process_function_call(mock_call)

        # Verify the result structure
        assert result["call_id"] == "call_123"
        assert result["name"] == "get_weather"
        assert result["arguments"] == mock_call.arguments
        assert result["result"] == "Weather in New York is 22°F"

    def test_process_function_call_tool_not_registered(self, chimeric_openai_client):
        """Test function call processing when tool is not registered."""
        client = chimeric_openai_client

        # Create a mock function call for unregistered tool
        mock_call = Mock()
        mock_call.name = "unregistered_tool"
        mock_call.call_id = "call_456"
        mock_call.arguments = json.dumps({"param": "value"})

        # Test that ToolRegistrationError is raised
        with pytest.raises(ToolRegistrationError) as exc_info:
            client._process_function_call(mock_call)

        assert "unregistered_tool" in str(exc_info.value)
        assert "No tool registered" in str(exc_info.value)

    def test_process_function_call_tool_not_callable(self, chimeric_openai_client, monkeypatch):
        """Test function call processing when tool has a non-callable function."""
        client = chimeric_openai_client

        mock_tool = Mock(spec=Tool)
        mock_tool.name = "broken_tool"
        mock_tool.description = "A tool with a non-callable function"
        mock_tool.function = "this is a string, not a callable"
        mock_tool.parameters = {
            "type": "object",
            "properties": {"param": {"type": "string"}},
            "required": ["param"],
        }

        # Mock the get_tool method to return our non-callable tool
        def mock_get_tool(name):
            if name == "broken_tool":
                return mock_tool
            raise ToolRegistrationError(f"No tool registered with name '{name}'")

        monkeypatch.setattr(client.tool_manager, "get_tool", mock_get_tool)

        # Create a mock function call
        mock_call = Mock()
        mock_call.name = "broken_tool"
        mock_call.call_id = "call_789"
        mock_call.arguments = json.dumps({"param": "value"})

        # Test that ToolRegistrationError is raised with the correct message
        with pytest.raises(ToolRegistrationError) as exc_info:
            client._process_function_call(mock_call)

        assert "Tool 'broken_tool' is not callable" in str(exc_info.value)

    def test_handle_function_tool_calls_no_calls(self, chimeric_openai_client):
        """Test _handle_function_tool_calls when response has no tool calls."""
        client = chimeric_openai_client

        # Create a response with no tool calls
        mock_response = Mock()
        mock_response.output = ["regular text response"]  # No ResponseFunctionToolCall objects

        messages = [{"role": "user", "content": "Hello"}]

        # Test handling when no tool calls are present
        tool_calls, updated_messages = client._handle_function_tool_calls(mock_response, messages)

        assert tool_calls == []
        assert updated_messages == messages  # Should be unchanged

    def test_handle_function_tool_calls_with_calls(self, chimeric_openai_client, monkeypatch):
        """Test _handle_function_tool_calls with actual tool calls."""
        from openai.types.responses import ResponseFunctionToolCall

        client = chimeric_openai_client

        # Create a mock tool function
        def mock_calculator(operation: str, a: int, b: int) -> int:
            if operation == "add":
                return a + b
            if operation == "multiply":
                return a * b
            return 0

        # Register the tool
        tool = Tool(
            name="calculator",
            description="Perform calculations",
            function=mock_calculator,
        )
        client.tool_manager.register(
            func=tool.function,
            name=tool.name,
            description=tool.description,
        )

        # Create mock tool calls
        mock_call1 = Mock(spec=ResponseFunctionToolCall)
        mock_call1.name = "calculator"
        mock_call1.call_id = "call_add"
        mock_call1.arguments = json.dumps({"operation": "add", "a": 5, "b": 3})

        mock_call2 = Mock(spec=ResponseFunctionToolCall)
        mock_call2.name = "calculator"
        mock_call2.call_id = "call_mult"
        mock_call2.arguments = json.dumps({"operation": "multiply", "a": 4, "b": 7})

        # Create a response with tool calls
        mock_response = Mock()
        mock_response.output = [mock_call1, mock_call2, "some other output"]

        messages = [{"role": "user", "content": "Calculate 5+3 and 4*7"}]

        # Test handling tool calls
        tool_calls, updated_messages = client._handle_function_tool_calls(mock_response, messages)

        # Verify tool calls metadata
        assert len(tool_calls) == 2
        assert tool_calls[0]["call_id"] == "call_add"
        assert tool_calls[0]["result"] == "8"
        assert tool_calls[1]["call_id"] == "call_mult"
        assert tool_calls[1]["result"] == "28"

        # Verify messages were updated with function calls and results
        assert len(updated_messages) == 5  # original + 4 new messages
        assert updated_messages[0] == {"role": "user", "content": "Calculate 5+3 and 4*7"}

        # Check function call messages
        assert updated_messages[1]["type"] == "function_call"
        assert updated_messages[1]["call_id"] == "call_add"
        assert updated_messages[2]["type"] == "function_call_output"
        assert updated_messages[2]["output"] == "8"

    def test_chat_completion_impl_with_tool_calls_two_pass(
        self, chimeric_openai_client, monkeypatch
    ):
        """Test _chat_completion_impl with tool calls requiring second API call."""
        from openai.types.responses import ResponseFunctionToolCall

        client = chimeric_openai_client

        # Register a tool
        def mock_search(query: str) -> str:
            return f"Search results for: {query}"

        tool = Tool(name="search", description="Search the web", function=mock_search)
        client.tool_manager.register(
            func=tool.function,
            name=tool.name,
            description=tool.description,
        )

        # Create mock responses
        # First response with tool call
        mock_call = Mock(spec=ResponseFunctionToolCall)
        mock_call.name = "search"
        mock_call.call_id = "call_search"
        mock_call.arguments = json.dumps({"query": "Python testing"})

        first_response = Mock()
        first_response.output = [mock_call]
        first_response.output_text = ""
        first_response.usage = SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15)
        first_response.model = "gpt-4"
        first_response.model_dump.return_value = {"first_call": True}

        # Second response with final answer
        second_response = Mock()
        second_response.output = ["Based on the search results, here's the answer..."]
        second_response.output_text = "Based on the search results, here's the answer..."
        second_response.usage = SimpleNamespace(input_tokens=20, output_tokens=15, total_tokens=35)
        second_response.model = "gpt-4"
        second_response.model_dump.return_value = {
            "second_call": True,
            "tool_calls": [
                {
                    "call_id": "call_search",
                    "name": "search",
                    "arguments": '{"query": "Python testing"}',
                    "result": "Search results for: Python testing",
                }
            ],
        }

        # Mock the client create method to return different responses
        responses = [first_response, second_response]
        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            result = responses[call_count]
            call_count += 1
            return result

        mock_responses = SimpleNamespace(create=mock_create)
        client._client.responses = mock_responses

        # Test the implementation
        messages = [{"role": "user", "content": "Search for Python testing"}]
        result = client._chat_completion_impl(messages=messages, model="gpt-4", tools=[tool])

        # Verify the result contains tool call metadata
        assert hasattr(result, "common")
        assert result.common.metadata.get("tool_calls") is not None
        assert len(result.common.metadata["tool_calls"]) == 1
        assert (
            result.common.metadata["tool_calls"][0]["result"]
            == "Search results for: Python testing"
        )

        # Verify both API calls were made
        assert call_count == 2

    async def test_achat_completion_impl_with_tool_calls_two_pass(
        self, chimeric_openai_client, monkeypatch
    ):
        """Test _achat_completion_impl with tool calls requiring second API call."""
        from openai.types.responses import ResponseFunctionToolCall

        client = chimeric_openai_client

        # Register a tool
        def mock_translate(text: str, target_lang: str) -> str:
            return f"Translated '{text}' to {target_lang}"

        tool = Tool(name="translate", description="Translate text", function=mock_translate)
        client.tool_manager.register(
            func=tool.function,
            name=tool.name,
            description=tool.description,
        )

        # Create mock tool call
        mock_call = Mock(spec=ResponseFunctionToolCall)
        mock_call.name = "translate"
        mock_call.call_id = "call_translate"
        mock_call.arguments = json.dumps({"text": "Hello", "target_lang": "Spanish"})

        # First response with tool call
        first_response = Mock()
        first_response.output = [mock_call]
        first_response.output_text = ""
        first_response.usage = SimpleNamespace(input_tokens=8, output_tokens=3, total_tokens=11)
        first_response.model = "gpt-3.5-turbo"
        first_response.model_dump.return_value = {"async_first": True}

        # Second response with final answer
        second_response = Mock()
        second_response.output = ["The translation is complete."]
        second_response.output_text = "The translation is complete."
        second_response.usage = SimpleNamespace(input_tokens=15, output_tokens=8, total_tokens=23)
        second_response.model = "gpt-3.5-turbo"
        second_response.model_dump.return_value = {
            "async_second": True,
            "tool_calls": [
                {
                    "call_id": "call_translate",
                    "name": "translate",
                    "arguments": '{"text": "Hello", "target_lang": "Spanish"}',
                    "result": "Translated 'Hello' to Spanish",
                }
            ],
        }

        # Mock the async client create method
        responses = [first_response, second_response]
        call_count = 0

        async def mock_async_create(**kwargs):
            nonlocal call_count
            result = responses[call_count]
            call_count += 1
            return result

        mock_responses = SimpleNamespace(create=mock_async_create)
        client._async_client.responses = mock_responses

        # Test the async implementation
        messages = [{"role": "user", "content": "Translate 'Hello' to Spanish"}]
        result = await client._achat_completion_impl(
            messages=messages, model="gpt-3.5-turbo", tools=[tool]
        )

        # Verify the result contains tool call metadata
        assert hasattr(result, "common")
        assert result.common.metadata.get("tool_calls") is not None
        assert len(result.common.metadata["tool_calls"]) == 1
        assert result.common.metadata["tool_calls"][0]["result"] == "Translated 'Hello' to Spanish"

        # Verify both async API calls were made
        assert call_count == 2

    def test_chat_completion_impl_streaming_with_tools(self, chimeric_openai_client, monkeypatch):
        """Test _chat_completion_impl with streaming and tools (single pass)."""
        client = chimeric_openai_client

        # Create mock stream response
        mock_stream = Mock(spec=client_module.Stream)

        # Mock the client's responses.create method to return stream
        mock_create = Mock(return_value=mock_stream)
        client._client.responses = SimpleNamespace(create=mock_create)

        # Mock _stream method to return a generator
        def mock_stream_method(stream):
            yield ChimericStreamChunk(
                native=Mock(),
                common=StreamChunk(content="Streaming", delta="Streaming", metadata={}),
            )

        monkeypatch.setattr(client, "_stream", mock_stream_method)

        # Test with tools and streaming
        tool = Tool(name="example_tool", description="Example tool")
        result = client._chat_completion_impl(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4",
            tools=[tool],
            stream=True,
        )

        # Verify the result is a generator (streaming response)
        assert hasattr(result, "__iter__")
        chunks = list(result)
        assert len(chunks) == 1
        assert isinstance(chunks[0], ChimericStreamChunk)
        assert chunks[0].common.content == "Streaming"

        # Verify tools were passed to the API
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert call_args[1]["tools"] == [tool]

    async def test_achat_completion_impl_streaming_with_tools(
        self, chimeric_openai_client, monkeypatch
    ):
        """Test _achat_completion_impl with streaming and tools (single pass)."""
        client = chimeric_openai_client

        # Create mock async stream
        mock_stream = Mock(spec=client_module.AsyncStream)

        mock_create = AsyncMock(return_value=mock_stream)
        client._async_client.responses = SimpleNamespace(create=mock_create)

        # Mock _astream method to return an async generator
        async def mock_astream_method(stream):
            yield ChimericStreamChunk(
                native=Mock(),
                common=StreamChunk(content="Async streaming", delta="Async streaming", metadata={}),
            )

        monkeypatch.setattr(client, "_astream", mock_astream_method)

        # Test with tools and streaming
        tool = Tool(name="async_tool", description="Async tool")
        result = await client._achat_completion_impl(
            messages=[{"role": "user", "content": "Hello async"}],
            model="gpt-4",
            tools=[tool],
            stream=True,
        )

        # Verify the result is an async generator
        assert hasattr(result, "__aiter__")
        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].common.content == "Async streaming"

        # Verify tools were passed to the async API
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert call_args[1]["tools"] == [tool]

    def test_create_chimeric_response_with_tool_calls(self, chimeric_openai_client):
        """Test _create_chimeric_response includes tool call metadata."""
        client = chimeric_openai_client

        # Create a mock response
        mock_response = Mock()
        mock_response.output_text = "Response with tool calls"
        mock_response.output = "Response with tool calls"
        mock_response.usage = SimpleNamespace(input_tokens=25, output_tokens=12, total_tokens=37)
        mock_response.model = "gpt-4"
        mock_response.model_dump.return_value = {"base_metadata": True}

        # Create tool calls metadata
        tool_calls = [
            {
                "call_id": "call_1",
                "name": "test_tool",
                "arguments": '{"param": "value"}',
                "result": "tool result",
            }
        ]

        # Test response creation with tool calls
        result = client._create_chimeric_response(mock_response, tool_calls)

        # Verify tool calls are included in metadata
        assert result.common.metadata["tool_calls"] == tool_calls
        assert result.common.metadata["base_metadata"] is True
        assert result.common.content == "Response with tool calls"

    def test_create_chimeric_response_without_tool_calls(self, chimeric_openai_client):
        """Test _create_chimeric_response without tool calls."""
        client = chimeric_openai_client

        # Create a mock response
        mock_response = Mock()
        mock_response.output_text = "Simple response"
        mock_response.output = "Simple response"
        mock_response.usage = SimpleNamespace(input_tokens=5, output_tokens=2, total_tokens=7)
        mock_response.model = "gpt-3.5-turbo"
        mock_response.model_dump.return_value = {"simple": True}

        # Test response creation without tool calls
        result = client._create_chimeric_response(mock_response, [])

        # Verify no tool calls in metadata
        assert "tool_calls" not in result.common.metadata
        assert result.common.metadata["simple"] is True
        assert result.common.content == "Simple response"

    def test_create_chimeric_response_with_response_text_fallback(self, chimeric_openai_client):
        """Test _create_chimeric_response falls back to response.text when output_text is None."""
        client = chimeric_openai_client

        # Create a mock response with None output_text but text in response attribute
        mock_response = Mock()
        mock_response.output_text = None
        mock_response.output = None
        mock_response.response = Mock()
        mock_response.response.text = "Fallback text content"
        mock_response.usage = SimpleNamespace(input_tokens=3, output_tokens=4, total_tokens=7)
        mock_response.model = "gpt-4"
        mock_response.model_dump.return_value = {"fallback": True}

        # Test response creation with fallback
        result = client._create_chimeric_response(mock_response, [])

        # Verify fallback text is used
        assert result.common.content == "Fallback text content"

    def test_process_function_call_with_complex_args(self, chimeric_openai_client):
        """Test _process_function_call with complex nested arguments."""
        client = chimeric_openai_client

        # Create a tool that accepts complex arguments
        def complex_tool(config: dict[Any, Any], items: list[Any], count: int = 1) -> str:
            return f"Processed {len(items)} items with config {config['mode']} (count: {count})"

        tool = Tool(
            name="complex_processor",
            description="Process complex data",
            function=complex_tool,
        )
        client.tool_manager.register(
            func=tool.function,
            name=tool.name,
            description=tool.description,
        )

        # Create a mock function call with complex arguments
        mock_call = Mock()
        mock_call.name = "complex_processor"
        mock_call.call_id = "call_complex"
        mock_call.arguments = json.dumps(
            {
                "config": {"mode": "advanced", "debug": True},
                "items": ["item1", "item2", "item3"],
                "count": 5,
            }
        )

        # Test the function call processing
        result = client._process_function_call(mock_call)

        # Verify complex arguments were handled correctly
        assert result["name"] == "complex_processor"
        assert result["result"] == "Processed 3 items with config advanced (count: 5)"
        assert json.loads(result["arguments"])["config"]["debug"] is True
