from datetime import datetime
import os
from typing import Any, cast
from unittest.mock import ANY, AsyncMock, Mock

from google.genai.types import GenerateContentResponse, GenerateContentResponseUsageMetadata
import pytest

from chimeric import Chimeric
from chimeric.exceptions import ProviderError
import chimeric.providers.google.client as client_module
from chimeric.providers.google.client import GoogleClient
from chimeric.types import (
    ChimericCompletionResponse,
    ChimericFileUploadResponse,
    ChimericStreamChunk,
    CompletionResponse,
    FileUploadResponse,
    ModelSummary,
    StreamChunk,
    Tool,
    Usage,
)


@pytest.fixture(scope="module")
def chimeric_google(google_env):
    """Create a Chimeric instance configured for Google."""
    return Chimeric(
        api_key=os.environ.get("GOOGLE_API_KEY", "test_key"),
    )


@pytest.fixture(scope="module")
def chimeric_google_client(chimeric_google) -> GoogleClient:
    """Get the GoogleClient from the Chimeric wrapper."""
    return cast("GoogleClient", chimeric_google.get_provider_client("google"))


@pytest.fixture(autouse=True)
def patch_google_imports(monkeypatch: pytest.MonkeyPatch):
    """Replace ``google.genai`` symbols with light-weight stubs.

    The real SDK classes are heavy and make network calls.  A minimal stub is
    more than enough for unit-testing the *client* integration logic.
    """

    class _StubModels:
        """Holds "models" RPC methods that we monkey-patch per-test."""

        def list(self):  # pragma: no cover - replaced on demand
            raise NotImplementedError

    class _StubFiles:
        """Holds the ``files.upload`` RPC that we monkey-patch per-test."""

        pass

    class _ClientStub:
        """Minimal sync ``google.genai.Client`` replacement."""

        def __init__(self, api_key: str, **_: Any) -> None:
            self.api_key = api_key
            self.models = _StubModels()
            self.files = _StubFiles()
            # Async surface mimics ``client.aio`` on the real SDK.
            self.aio = Mock(models=_StubModels())

    class _AsyncClientStub(_ClientStub):
        """Placeholder for *type* checks - never directly instantiated."""

        pass

    # Patch the names used inside :pymod:`client_module`.
    monkeypatch.setattr(client_module, "Client", _ClientStub, False)
    monkeypatch.setattr(client_module, "AsyncClient", _AsyncClientStub, False)
    # ``GenerateContentConfig`` is only used for storing attrs, so a Mock works fine.
    monkeypatch.setattr(client_module, "GenerateContentConfig", Mock, False)


class MockGoogleResponse:
    """Mock implementation of a Google GenerateContentResponse."""

    def __init__(self) -> None:
        self.text = "Hello from Gemini"
        self.usage_metadata = Mock(spec=GenerateContentResponseUsageMetadata)
        self.usage_metadata.prompt_token_count = 10
        self.usage_metadata.candidates_token_count = 5
        self.usage_metadata.total_token_count = 15
        self.usage_metadata.cache_tokens_details = "cache_info"
        self.model = "gemini-pro"

    def model_dump(self) -> dict[str, Any]:
        return {"model": self.model, "dumped": True}


class MockGoogleFile:
    """Mock implementation of a Google File object."""

    def __init__(self) -> None:
        self.name = "file-123456"
        self.display_name = "test_file.txt"
        self.size_bytes = 1024
        self.create_time = datetime.now()

    @staticmethod
    def model_dump() -> dict[str, Any]:
        return {"file_metadata": True}


# noinspection PyUnusedLocal
class TestGoogleClient:
    """Tests for the GoogleClient class."""

    def test_capabilities_and_supports(self, chimeric_google_client):
        """Test that client reports correct capabilities and support methods."""
        client = chimeric_google_client
        caps = client.capabilities

        # Verify expected capabilities - Google should support multimodal, streaming, tools, and files, but not agents
        assert caps.multimodal
        assert caps.streaming
        assert caps.tools
        assert not caps.agents
        assert caps.files

        # Verify the supports_* API methods return correct values
        assert client.supports_multimodal()
        assert client.supports_streaming()
        assert client.supports_tools()
        assert not client.supports_agents()
        assert client.supports_files()

    def test_list_models_maps_to_summary(self, chimeric_google_client, monkeypatch):
        """Test that list_models correctly maps raw model data to ModelSummary objects."""
        client = chimeric_google_client

        # Create mock models
        model1 = Mock()
        model1.name = "gemini-pro"
        model1.display_name = "Gemini Pro"
        model1.description = "Text model"

        model2 = Mock()
        model2.name = "gemini-vision"
        model2.display_name = "Gemini Vision"
        model2.description = "Vision model"

        model3 = Mock()
        model3.name = None
        model3.display_name = None
        model3.description = "Missing fields"

        # Mock the list method to return our test models
        def mock_list():
            return [model1, model2, model3]

        monkeypatch.setattr(client._client.models, "list", mock_list)
        models = client.list_models()

        # Verify models are returned and properly mapped
        assert len(models) == 3

        # Check first model has correct fields
        assert isinstance(models[0], ModelSummary)
        assert models[0].id == "gemini-pro"
        assert models[0].name == "Gemini Pro"
        assert models[0].description == "Text model"

        # Check the second model
        assert models[1].id == "gemini-vision"
        assert models[1].name == "Gemini Vision"

        # Check third model uses defaults for missing fields
        assert models[2].id == "unknown"
        assert models[2].name == "Unknown Model"

    def test_process_stream_event(self, chimeric_google_client):
        """Test _process_stream_event correctly processes stream events."""
        # Test with event that has text
        event_with_text = Mock()
        event_with_text.text = "hello"
        event_with_text.model_dump = lambda: {"type": "text"}

        acc, chunk = GoogleClient._process_stream_event(event_with_text, "")

        assert acc == "hello"
        assert isinstance(chunk, ChimericStreamChunk)
        assert chunk.common.content == "hello"
        assert chunk.common.delta == "hello"
        assert chunk.common.metadata == {"type": "text"}

        # Test with initially accumulated content
        acc, chunk = GoogleClient._process_stream_event(event_with_text, "initial ")

        assert acc == "initial hello"
        assert chunk.common.content == "initial hello"
        assert chunk.common.delta == "hello"

        # Test with event that has no text attribute - Mock should return None when a text doesn't exist
        event_no_text = Mock(spec=[])  # Empty spec means no attributes
        event_no_text.model_dump = lambda: {"type": "other"}

        acc, chunk = GoogleClient._process_stream_event(event_no_text, "prev ")

        assert acc == "prev "  # Should remain unchanged
        assert chunk.common.content == "prev "
        assert chunk.common.delta == ""  # Empty delta when no text

    def test_stream(self, chimeric_google_client, monkeypatch):
        """Test that _stream correctly processes and yields stream chunks."""
        client = chimeric_google_client

        # Create test events
        event1 = Mock(spec=GenerateContentResponse)
        event1.text = "Hello"
        event1.model_dump = lambda: {"part": 1}

        event2 = Mock(spec=GenerateContentResponse)
        event2.text = " world"
        event2.model_dump = lambda: {"part": 2}

        event3 = Mock(spec=GenerateContentResponse)
        event3.text = "!"
        event3.model_dump = lambda: {"part": 3}

        # Mock _process_stream_event to track calls - note this is an instance method, not static
        original_process = GoogleClient._process_stream_event
        process_calls = []

        def mock_process(event, acc):
            process_calls.append((event, acc))
            return original_process(event, acc)

        monkeypatch.setattr(GoogleClient, "_process_stream_event", staticmethod(mock_process))

        # Process events through _stream
        chunks = list(client._stream([event1, event2, event3]))

        # Verify correct processing sequence
        assert len(chunks) == 3
        assert chunks[0].common.content == "Hello"
        assert chunks[1].common.content == "Hello world"
        assert chunks[2].common.content == "Hello world!"

        # Verify each event was processed with correct accumulated text
        assert len(process_calls) == 3
        assert process_calls[0][0] is event1
        assert process_calls[0][1] == ""  # Initial accumulator is empty
        assert process_calls[1][0] is event2
        assert process_calls[1][1] == "Hello"  # Second event gets text from first
        assert process_calls[2][0] is event3
        assert process_calls[2][1] == "Hello world"  # Third event gets combined text

    async def test_astream(self, chimeric_google_client, monkeypatch):
        """Test that _astream correctly processes async stream chunks."""
        client = chimeric_google_client

        # Create test events
        event1 = Mock()
        event1.text = "Async"
        event1.model_dump = lambda: {"async": 1}

        event2 = Mock()
        event2.text = " response"
        event2.model_dump = lambda: {"async": 2}

        # Create async generator
        async def async_event_generator():
            yield event1
            yield event2

        # Process events through _astream
        chunks = []
        async for chunk in client._astream(async_event_generator()):
            chunks.append(chunk)

        # Verify correct processing
        assert len(chunks) == 2
        assert chunks[0].common.content == "Async"
        assert chunks[0].common.delta == "Async"
        assert chunks[0].common.metadata == {"async": 1}

        assert chunks[1].common.content == "Async response"
        assert chunks[1].common.delta == " response"
        assert chunks[1].common.metadata == {"async": 2}

    def test_encode_tools(self, chimeric_google_client):
        """Test that _encode_tools correctly formats tools for the Google API."""
        client = chimeric_google_client

        # Test with None tools
        assert client._encode_tools(None) is None

        # Create a tool with a function
        def weather_function(location: str):
            return f"Weather in {location}"

        tool_with_function = Tool(
            name="get_weather",
            description="Get weather information",
            function=weather_function,
        )

        # Create a tool-like object (not an instance of Tool)
        tool_like_obj = {"name": "calculator", "description": "Calculate things"}

        # Test mixed tools
        encoded_tools = client._encode_tools([tool_with_function, tool_like_obj])

        # Verify encoding
        assert isinstance(encoded_tools, list)
        assert len(encoded_tools) == 2
        assert encoded_tools[0] == weather_function  # Should extract function
        assert encoded_tools[1] == tool_like_obj  # Should keep as-is

    def test_convert_usage_metadata_with_data(self, chimeric_google_client):
        """Test _convert_usage_metadata with full metadata."""
        # Create mock usage metadata with all fields
        usage_meta = Mock(spec=GenerateContentResponseUsageMetadata)
        usage_meta.prompt_token_count = 15
        usage_meta.candidates_token_count = 25
        usage_meta.total_token_count = 40
        usage_meta.cache_tokens_details = {"cache": "info"}
        usage_meta.cached_content_token_count = 5
        usage_meta.candidates_tokens_details = {"candidates": "details"}
        usage_meta.prompt_tokens_details = {"prompt": "details"}
        usage_meta.thoughts_token_count = 10
        usage_meta.tool_use_prompt_token_count = 5
        usage_meta.tool_use_prompt_tokens_details = {"tool": "details"}
        usage_meta.traffic_type = "INTERACTIVE"

        # Convert metadata
        usage = GoogleClient._convert_usage_metadata(usage_meta)

        # Verify core fields
        assert usage.prompt_tokens == 15
        assert usage.completion_tokens == 25
        assert usage.total_tokens == 40

        # Verify Google-specific fields were added
        assert usage.cache_tokens_details == {"cache": "info"}
        assert usage.cached_content_token_count == 5
        assert usage.candidates_tokens_details == {"candidates": "details"}
        assert usage.prompt_tokens_details == {"prompt": "details"}
        assert usage.thoughts_token_count == 10
        assert usage.tool_use_prompt_token_count == 5
        assert usage.tool_use_prompt_tokens_details == {"tool": "details"}
        assert usage.traffic_type == "INTERACTIVE"

    def test_convert_usage_metadata_edge_cases(self, chimeric_google_client):
        """Test _convert_usage_metadata edge cases like None values and missing fields."""
        # Test with None metadata
        usage = GoogleClient._convert_usage_metadata(None)
        assert isinstance(usage, Usage)
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

        # Test with missing required fields
        sparse_meta = Mock(spec=GenerateContentResponseUsageMetadata)
        usage = GoogleClient._convert_usage_metadata(sparse_meta)
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

        # Test with zero/None values that should use fallbacks
        zero_meta = Mock(spec=GenerateContentResponseUsageMetadata)
        zero_meta.prompt_token_count = 0
        zero_meta.candidates_token_count = None
        zero_meta.total_token_count = 0
        usage = GoogleClient._convert_usage_metadata(zero_meta)
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

        # Test when total is missing, but individual counts exist
        partial_meta = Mock(spec=GenerateContentResponseUsageMetadata)
        partial_meta.prompt_token_count = 10
        partial_meta.candidates_token_count = 15
        partial_meta.total_token_count = None
        usage = GoogleClient._convert_usage_metadata(partial_meta)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 15
        assert usage.total_tokens == 25  # Should compute sum

    def test_chat_completion_impl_non_streaming(self, chimeric_google_client, monkeypatch):
        """Test _chat_completion_impl with non-streaming response."""
        client = chimeric_google_client

        # Create a mock response
        mock_response = MockGoogleResponse()

        # Mock the client's generate_content method
        mock_generate = Mock(return_value=mock_response)
        client._client.models.generate_content = mock_generate

        # Test the implementation
        result = client._chat_completion_impl(
            messages=["Hello, Gemini!"],
            model="gemini-pro",
            temperature=0.7,
        )

        # Verify the result
        assert isinstance(result, ChimericCompletionResponse)
        assert result.native is mock_response
        assert isinstance(result.common, CompletionResponse)
        assert result.common.content == "Hello from Gemini"
        assert result.common.model == "gemini-pro"
        assert result.common.usage.prompt_tokens == 10
        assert result.common.usage.completion_tokens == 5
        assert result.common.usage.total_tokens == 15
        assert result.common.metadata == {"model": "gemini-pro", "dumped": True}

        # Verify the client was called correctly - use ANY for the config object
        # since it's created internally, and we can't predict its exact Mock ID
        mock_generate.assert_called_once_with(
            model="gemini-pro",
            contents=["Hello, Gemini!"],
            config=ANY,
        )

    def test_chat_completion_impl_streaming(self, chimeric_google_client, monkeypatch):
        """Test _chat_completion_impl with streaming response."""
        client = chimeric_google_client

        # Create a mock stream response
        mock_stream_list = [Mock(), Mock()]

        # Mock the client's generate_content_stream method
        mock_stream_generate = Mock(return_value=mock_stream_list)
        client._client.models.generate_content_stream = mock_stream_generate

        stream_input = None

        def mock_stream_method(stream_obj):
            nonlocal stream_input
            stream_input = stream_obj
            yield ChimericStreamChunk(
                native=Mock(),
                common=StreamChunk(content="Streaming", delta="Streaming", metadata={}),
            )
            yield ChimericStreamChunk(
                native=Mock(),
                common=StreamChunk(content="Streaming response", delta=" response", metadata={}),
            )

        monkeypatch.setattr(client, "_stream", mock_stream_method)

        # Test the implementation
        result = client._chat_completion_impl(
            messages=["Stream request"],
            model="gemini-pro",
            stream=True,
        )

        # Verify the result is a generator
        assert hasattr(result, "__iter__")

        # Collect all chunks
        chunks = list(result)
        assert len(chunks) == 2
        assert all(isinstance(chunk, ChimericStreamChunk) for chunk in chunks)
        assert chunks[0].common.content == "Streaming"
        assert chunks[1].common.content == "Streaming response"

        # Verify the client was called correctly
        mock_stream_generate.assert_called_once_with(
            model="gemini-pro",
            contents=["Stream request"],
            config=ANY,
        )

        # Verify _stream was called with correct stream object
        # stream_input should be the mock_stream_list returned by generate_content_stream
        assert stream_input is mock_stream_list

    def test_chat_completion_impl_with_tools(self, chimeric_google_client, monkeypatch):
        """Test *chat*completion_impl with tools parameter."""
        client = chimeric_google_client

        # Create mock response
        mock_response = MockGoogleResponse()
        mock_response.text = "Function called"

        # Mock generate_content
        mock_generate = Mock(return_value=mock_response)
        client._client.models.generate_content = mock_generate

        # Test function that should be passed
        def get_time():
            return "10:30 AM"

        # Create a tool with the function
        tool = Tool(name="get_time", description="Get current time", function=get_time)

        # Make sure encode_tools works as expected
        original_encode = client._encode_tools
        encode_called_with = None

        def mock_encode_tools(tools):
            nonlocal encode_called_with
            encode_called_with = tools
            return original_encode(tools)

        monkeypatch.setattr(client, "_encode_tools", mock_encode_tools)

        # Since we're calling _chat_completion_impl directly, we need to manually process tools
        # like the chat_completion method would do
        processed_tools = client._process_tools(auto_tool=False, tools=[tool])

        # Test with processed tools
        result = client._chat_completion_impl(
            messages=["What time is it?"],
            model="gemini-pro",
            tools=processed_tools,
            temperature=0.5,
        )

        # Verify encode_tools was called with the tool during _process_tools
        assert encode_called_with == [tool]

        # Verify the result
        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Function called"
        assert result.common.model == "gemini-pro"

        # Verify the API was called correctly
        mock_generate.assert_called_once()
        call_args = mock_generate.call_args
        assert call_args[1]["model"] == "gemini-pro"
        assert call_args[1]["contents"] == ["What time is it?"]

        # Verify tools were passed to the config
        config = call_args[1]["config"]
        assert config.tools is not None
        assert len(config.tools) == 1
        # The tool should be the function itself (after encoding)
        assert config.tools[0] == get_time

    def test_chat_completion_impl_tools_type_check(self, chimeric_google_client):
        """Test _chat_completion_impl raises TypeError for non-list tools."""
        client = chimeric_google_client

        # Test with non-list tools
        with pytest.raises(TypeError) as excinfo:
            client._chat_completion_impl(
                messages=["Test"],
                model="gemini-pro",
                tools={"name": "not_a_list"},  # Dictionary instead of a list
            )

        assert "Google expects tools to be a list" in str(excinfo.value)

    async def test_achat_completion_impl_non_streaming(self, chimeric_google_client, monkeypatch):
        """Test _achat_completion_impl with non-streaming response."""
        client = chimeric_google_client

        # Create mock async response
        mock_response = MockGoogleResponse()
        mock_response.text = "Async hello"

        # Mock the async client's generate_content method
        mock_async_generate = AsyncMock(return_value=mock_response)
        client._async_client.models.generate_content = mock_async_generate

        # Test the async implementation
        result = await client._achat_completion_impl(
            messages=["Async hello"],
            model="gemini-pro",
            temperature=0.5,
        )

        # Verify the result
        assert isinstance(result, ChimericCompletionResponse)
        assert result.native is mock_response
        assert result.common.content == "Async hello"
        assert result.common.model == "gemini-pro"

        # Verify the async client was called correctly
        mock_async_generate.assert_called_once_with(
            model="gemini-pro",
            contents=["Async hello"],
            config=ANY,
        )

    async def test_achat_completion_impl_streaming(self, chimeric_google_client, monkeypatch):
        """Test _achat_completion_impl with streaming response."""
        client = chimeric_google_client

        # Create mock async stream
        mock_stream = Mock()

        # Mock the async client's generate_content_stream method
        mock_async_stream_generate = AsyncMock(return_value=mock_stream)
        client._async_client.models.generate_content_stream = mock_async_stream_generate

        # Mock _astream method to return an async generator
        astream_input = None

        async def mock_astream(stream_obj):
            nonlocal astream_input
            astream_input = stream_obj
            yield ChimericStreamChunk(
                native=Mock(),
                common=StreamChunk(content="Async", delta="Async", metadata={}),
            )
            yield ChimericStreamChunk(
                native=Mock(),
                common=StreamChunk(content="Async streaming", delta=" streaming", metadata={}),
            )

        monkeypatch.setattr(client, "_astream", mock_astream)

        # Test the async implementation
        result = await client._achat_completion_impl(
            messages=["Stream test"],
            model="gemini-pro",
            stream=True,
        )

        # Verify the result is an async generator
        assert hasattr(result, "__aiter__")

        # Collect all chunks
        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].common.content == "Async"
        assert chunks[1].common.content == "Async streaming"

        # Verify the async client was called correctly
        mock_async_stream_generate.assert_called_once_with(
            model="gemini-pro",
            contents=["Stream test"],
            config=ANY,
        )

        # Verify _astream was called with the correct stream object
        assert astream_input is mock_stream

    async def test_achat_completion_impl_with_tools(self, chimeric_google_client, monkeypatch):
        """Test _achat_completion_impl with tools parameter."""
        client = chimeric_google_client

        # Create mock response
        mock_response = MockGoogleResponse()
        mock_response.text = "Async tool response"

        # Mock async generate_content
        mock_async_generate = AsyncMock(return_value=mock_response)
        client._async_client.models.generate_content = mock_async_generate

        # Create a tool
        tool = Tool(name="async_tool", description="Tool for testing")

        def mock_tool_function() -> str:
            return "Tool function called"

        tool2 = {"name": "dict_tool", "function": mock_tool_function}  # Non-Tool object

        # Test with tools
        result = await client._achat_completion_impl(
            messages=["Test async tools"],
            model="gemini-pro",
            tools=[tool, tool2],
        )

        # Verify the result
        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Async tool response"

        # Verify client was called with the correct tools configuration
        mock_async_generate.assert_called_once()
        call_args = mock_async_generate.call_args
        assert len(call_args[1]["config"].tools) == 2

    async def test_achat_completion_impl_tools_type_check(self, chimeric_google_client):
        """Test _achat_completion_impl raises TypeError for non-list tools."""
        client = chimeric_google_client

        # Test with non-list tools
        with pytest.raises(TypeError) as excinfo:
            await client._achat_completion_impl(
                messages=["Test"],
                model="gemini-pro",
                tools="not_a_list",  # String instead of a list
            )

        assert "Google expects tools to be a list" in str(excinfo.value)

    def test_upload_file(self, chimeric_google_client, monkeypatch):
        """Test successful file upload."""
        client = chimeric_google_client

        # Create a mock file
        mock_file = MockGoogleFile()

        # Mock the upload method
        mock_upload = Mock(return_value=mock_file)
        client._client.files.upload = mock_upload

        # Test the upload
        result = client._upload_file(path="/path/to/file.txt", mime_type="text/plain")

        # Verify the result
        assert isinstance(result, ChimericFileUploadResponse)
        assert result.native is mock_file
        assert isinstance(result.common, FileUploadResponse)
        assert result.common.file_id == "file-123456"
        assert result.common.filename == "test_file.txt"
        assert result.common.bytes == 1024
        assert result.common.created_at is not None
        assert result.common.metadata == {"file_metadata": True}

        # Verify file upload was called with filtered kwargs
        mock_upload.assert_called_once_with(path="/path/to/file.txt", mime_type="text/plain")

    def test_upload_file_with_missing_fields(self, chimeric_google_client, monkeypatch):
        """Test file upload with missing metadata fields."""
        client = chimeric_google_client

        # Create a mock file with missing fields
        mock_file = Mock()
        mock_file.name = None
        mock_file.display_name = None
        mock_file.size_bytes = None
        mock_file.create_time = None
        mock_file.model_dump = lambda: {"sparse": True}

        # Mock the upload method
        mock_upload = Mock(return_value=mock_file)
        client._client.files.upload = mock_upload

        # Test the upload
        result = client._upload_file(path="/path/to/file.txt")

        # Verify the result uses default values for missing fields
        common = result.common
        assert common.file_id == "unknown"
        assert common.filename == "unknown"
        assert common.bytes == 0
        assert common.created_at is None

    def test_upload_file_error(self, chimeric_google_client, monkeypatch):
        """Test file upload error handling."""
        client = chimeric_google_client

        # Mock upload to raise an error
        def mock_upload_error(**kwargs):
            raise ValueError("File upload failed")

        client._client.files.upload = mock_upload_error

        # Test the error is properly propagated
        with pytest.raises(ProviderError):
            client.upload_file(path="/nonexistent/file.txt")

    def test_chat_completion_counts_errors(self, chimeric_google_client, monkeypatch):
        """Test that chat_completion properly tracks request and error counts."""
        client = chimeric_google_client

        # Test successful request path
        def mock_impl(messages, model, stream=False, tools=None, **kwargs):
            return "SUCCESS"

        monkeypatch.setattr(client, "_chat_completion_impl", mock_impl)
        before_req = client.request_count
        before_err = client.error_count

        # Test successful call
        out = client.chat_completion("Hi Gemini", "gemini-pro")
        assert out == "SUCCESS"
        assert client.request_count == before_req + 1
        assert client.error_count == before_err

        # Test error handling
        def mock_error_impl(*args, **kwargs):
            raise ValueError("Test error")

        monkeypatch.setattr(client, "_chat_completion_impl", mock_error_impl)

        # Should raise the error but also increment error count
        with pytest.raises(ValueError):
            client.chat_completion("Error test", "gemini-pro")

        # Verify counts
        assert client.request_count == before_req + 2
        assert client.error_count == before_err + 1

    async def test_achat_completion_counts_errors(self, chimeric_google_client, monkeypatch):
        """Test that achat_completion properly tracks request and error counts."""
        client = chimeric_google_client

        # Mock successful async implementation
        async def mock_async_impl(*args, **kwargs):
            return "ASYNC_SUCCESS"

        monkeypatch.setattr(client, "_achat_completion_impl", mock_async_impl)
        before_req = client.request_count
        before_err = client.error_count

        # Test successful async call
        result = await client.achat_completion("Hi async", "gemini-pro")
        assert result == "ASYNC_SUCCESS"
        assert client.request_count == before_req + 1
        assert client.error_count == before_err

        # Mock error implementation
        async def mock_async_error(*args, **kwargs):
            raise RuntimeError("Async error")

        monkeypatch.setattr(client, "_achat_completion_impl", mock_async_error)

        # Test error handling
        with pytest.raises(RuntimeError):
            await client.achat_completion("Async error test", "gemini-pro")

        # Verify counts
        assert client.request_count == before_req + 2
        assert client.error_count == before_err + 1

    def test_get_model_info(self, chimeric_google_client, monkeypatch):
        """Test get_model_info for both found and not found models."""
        client = chimeric_google_client

        # Create mock model summaries
        models = [
            ModelSummary(id="gemini-pro", name="Gemini Pro", created_at=0, owned_by="Google"),
            ModelSummary(id="gemini-vision", name="Gemini Vision", created_at=0, owned_by="Google"),
        ]

        # Mock list_models
        monkeypatch.setattr(client, "list_models", lambda: models)

        # Test finding existing model
        info = client.get_model_info("gemini-pro")
        assert isinstance(info, ModelSummary)
        assert info.id == "gemini-pro"
        assert info.name == "Gemini Pro"

        # Test error when model not found
        with pytest.raises(ValueError) as excinfo:
            client.get_model_info("nonexistent-model")

        assert "Model nonexistent-model not found" in str(excinfo.value)

    def test_provider_name_and_repr(self, chimeric_google_client):
        """Test that repr and str methods include provider name and request counts."""
        client = chimeric_google_client

        # Check provider name is set
        assert client._provider_name == "Google"

        # Test repr
        repr_str = repr(client)
        assert "GoogleClient" in repr_str
        assert "requests=" in repr_str
        assert "errors=" in repr_str

        # Test str
        str_output = str(client)
        assert "GoogleClient" in str_output
        assert "- Requests:" in str_output
        assert "- Errors:" in str_output
