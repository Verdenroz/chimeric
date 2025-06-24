from collections.abc import AsyncGenerator, Generator
import json
import os
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

from groq import NOT_GIVEN
import httpx
import pytest

from chimeric import Chimeric
from chimeric.exceptions import ProviderError, ToolRegistrationError
import chimeric.providers.groq.client as client_module
from chimeric.providers.groq.client import GroqClient
from chimeric.types import (
    ChimericCompletionResponse,
    ChimericStreamChunk,
    CompletionResponse,
    ModelSummary,
    StreamChunk,
    Tool,
    ToolParameters,
)


@pytest.fixture(scope="module")
def chimeric_groq():
    """Create a Chimeric instance configured for Groq."""
    return Chimeric(
        groq_api_key=os.getenv("GROQ_API_KEY", "test_key"),
        base_url="https://api.groq.com/v1",
        timeout=120,
        max_retries=2,
        project="chimeric_groq_test",
    )


@pytest.fixture(scope="module")
def chimeric_groq_client(chimeric_groq) -> GroqClient:
    """Get the GroqClient from the Chimeric wrapper."""
    return cast("GroqClient", chimeric_groq.get_provider_client("groq"))


class MockResponse:
    """Mock implementation of a non-streaming Groq response."""

    def __init__(self) -> None:
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(content="hello", tool_calls=None),
                index=0,
                finish_reason="stop",
            )
        ]
        self.usage = SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        self.model = "mixtral-8x7b"

    @staticmethod
    def model_dump() -> dict[str, Any]:
        return {"dumped": True}


class MockStreamChunk:
    """Mock implementation of a streaming Groq response chunk."""

    def __init__(self, content="", finish_reason=None):
        self.choices = [
            SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish_reason)
        ]

    @staticmethod
    def model_dump() -> dict[str, Any]:
        return {"chunked": True}


class MockFile:
    """Mock implementation of file upload return value."""

    def __init__(self) -> None:
        self.id = "file-id"
        self.filename = "file.txt"
        self.bytes = 123
        self.purpose = "test"
        self.created_at = 456
        self.status = "ready"


# noinspection PyUnusedLocal,PyTypeChecker
class TestGroqClient:
    """Tests for the GroqClient class."""

    def test_capabilities_and_supports(self, chimeric_groq_client):
        """Test that client reports correct capabilities and support methods."""
        client = chimeric_groq_client
        caps = client.capabilities

        # Verify all expected capabilities are enabled/disabled
        assert caps.multimodal
        assert caps.streaming
        assert caps.tools
        assert not caps.agents  # Groq doesn't support agents
        assert caps.files

        # Verify the supports_* API methods return correct values
        assert client.supports_multimodal()
        assert client.supports_streaming()
        assert client.supports_tools()
        assert not client.supports_agents()
        assert client.supports_files()

    def test_list_models_maps_to_summary(self, chimeric_groq_client, monkeypatch):
        """Test that list_models correctly maps raw model data to ModelSummary objects."""
        client = chimeric_groq_client

        class MockModel:
            def __init__(self, id):
                self.id = id
                self.owned_by = "groq"
                self.created = 161803398

        class MockModelList:
            @property
            def data(self):
                return [MockModel("llama3-8b"), MockModel("mixtral-8x7b")]

        client._client.models = SimpleNamespace(list=lambda: MockModelList())

        models = client.list_models()

        # Verify models are returned and properly mapped
        assert len(models) == 2
        assert isinstance(models[0], ModelSummary)
        assert models[0].id == "llama3-8b"
        assert models[0].owned_by == "groq"
        assert models[0].created_at == 161803398
        assert models[1].id == "mixtral-8x7b"

    def test_normalize_messages(self, chimeric_groq_client):
        """Test that _normalize_messages correctly formats input messages."""
        client = chimeric_groq_client

        # Test with string input
        string_input = "Hello, Groq!"
        normalized = client._normalize_messages(string_input)
        assert isinstance(normalized, list)
        assert len(normalized) == 1
        assert normalized[0]["role"] == "user"
        assert normalized[0]["content"] == "Hello, Groq!"

        # Test with list of messages
        message_list = [
            {"role": "system", "content": "You are an assistant"},
            {"role": "user", "content": "Hi there"},
        ]
        normalized = client._normalize_messages(message_list)
        assert normalized == message_list

    def test_process_stream_chunk_empty_content_no_finish_reason(self, chimeric_groq_client):
        """Test _process_stream_chunk with empty content and no finish_reason (line 134-135)."""
        client = chimeric_groq_client

        # Create a chunk with empty content and no finish_reason
        class MockStreamChunkEmpty:
            def __init__(self):
                self.choices = [
                    SimpleNamespace(
                        delta=SimpleNamespace(content=""),  # Empty content
                        finish_reason=None,  # No finish reason
                    )
                ]

            @staticmethod
            def model_dump() -> dict[str, Any]:
                return {"empty_chunk": True}

        empty_chunk = MockStreamChunkEmpty()
        accumulated = "some text"

        # Test that empty content with no finish_reason returns None
        new_accumulated, result_chunk = client._process_stream_chunk(empty_chunk, accumulated)

        # Verify accumulated text is unchanged and no chunk is returned
        assert new_accumulated == accumulated
        assert result_chunk is None

    def test_process_stream_chunk(self, chimeric_groq_client):
        """Test _process_stream_chunk correctly processes streaming chunks."""
        client = chimeric_groq_client

        # Test regular content chunk
        content_chunk = MockStreamChunk(content="Hello")
        accumulated, chunk = client._process_stream_chunk(content_chunk, "")
        assert accumulated == "Hello"
        assert chunk is not None
        assert chunk.common.content == "Hello"
        assert chunk.common.delta == "Hello"

        # Test accumulated content
        accumulated, chunk = client._process_stream_chunk(
            MockStreamChunk(content=" world"), "Hello"
        )
        assert accumulated == "Hello world"
        assert chunk.common.content == "Hello world"
        assert chunk.common.delta == " world"

        # Test finish chunk
        finish_chunk = MockStreamChunk(content="", finish_reason="stop")
        accumulated, chunk = client._process_stream_chunk(finish_chunk, "Hello world")
        assert accumulated == "Hello world"
        assert chunk.common.content == "Hello world"
        assert chunk.common.finish_reason == "stop"

        # Test empty choices
        empty_chunk = MockStreamChunk()
        empty_chunk.choices = []
        accumulated, chunk = client._process_stream_chunk(empty_chunk, "text")
        assert accumulated == "text"
        assert chunk is None

    def test_stream_skips_empty_chunks(self, chimeric_groq_client):
        """Ensure _stream skips chunks where processed_chunk is None."""
        client = chimeric_groq_client

        # Chunk with empty content and no finish_reason -> should be skipped
        class EmptyChunk:
            def __init__(self):
                self.choices = [
                    SimpleNamespace(delta=SimpleNamespace(content=""), finish_reason=None)
                ]

            @staticmethod
            def model_dump():
                return {"empty": True}

        # Valid chunk to follow
        valid_chunk = MockStreamChunk(content="OK")

        class Stream:
            def __iter__(self):
                yield EmptyChunk()
                yield valid_chunk

        chunks = list(client._stream(Stream()))
        assert len(chunks) == 1
        assert chunks[0].common.content == "OK"

    async def test_astream_skips_empty_chunks(self, chimeric_groq_client):
        """Ensure _astream skips chunks where processed_chunk is None."""
        client = chimeric_groq_client

        # Async chunk with empty content and no finish_reason -> skipped
        class EmptyChunk:
            def __init__(self):
                self.choices = [
                    SimpleNamespace(delta=SimpleNamespace(content=""), finish_reason=None)
                ]

            @staticmethod
            def model_dump():
                return {"empty": True}

        valid_chunk = MockStreamChunk(content="OK")

        class AsyncStream:
            def __init__(self):
                self.chunks = [EmptyChunk(), valid_chunk]

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.chunks:
                    raise StopAsyncIteration
                return self.chunks.pop(0)

        collected = []
        async for chunk in client._astream(AsyncStream()):
            collected.append(chunk)

        assert len(collected) == 1
        assert collected[0].common.content == "OK"

    def test_stream_method(self, chimeric_groq_client):
        """Test the _stream method processes chunks correctly from a synchronous stream."""
        client = chimeric_groq_client

        # Create a mock stream of chunks
        chunk1 = MockStreamChunk(content="Hello")
        chunk2 = MockStreamChunk(content=" world")
        chunk3 = MockStreamChunk(content="", finish_reason="stop")

        # Mock a Stream object that yields these chunks when iterated
        class MockStream:
            def __iter__(self):
                yield chunk1
                yield chunk2
                yield chunk3

        mock_stream = MockStream()

        # Get chunks from the _stream method
        chunks = list(client._stream(mock_stream))

        # Verify the correct number of chunks were processed
        # Note: empty chunks without finish_reason are filtered out
        assert len(chunks) == 3

        # Verify chunk contents and accumulated content
        assert chunks[0].common.content == "Hello"
        assert chunks[0].common.delta == "Hello"
        assert chunks[0].common.finish_reason is None

        assert chunks[1].common.content == "Hello world"
        assert chunks[1].common.delta == " world"
        assert chunks[1].common.finish_reason is None

        assert chunks[2].common.content == "Hello world"
        assert chunks[2].common.delta == ""
        assert chunks[2].common.finish_reason == "stop"

    async def test_astream_method(self, chimeric_groq_client):
        """Test the _astream method processes chunks correctly from an asynchronous stream."""
        client = chimeric_groq_client

        # Create a mock stream of chunks
        chunk1 = MockStreamChunk(content="Async")
        chunk2 = MockStreamChunk(content=" test")
        chunk3 = MockStreamChunk(content=" complete", finish_reason=None)
        chunk4 = MockStreamChunk(content="", finish_reason="stop")

        # Create a mock AsyncStream that yields chunks when async iterated
        class MockAsyncStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                if not hasattr(self, "chunks"):
                    self.chunks = [chunk1, chunk2, chunk3, chunk4]

                if not self.chunks:
                    raise StopAsyncIteration

                return self.chunks.pop(0)

        mock_async_stream = MockAsyncStream()

        # Collect chunks from the _astream method
        chunks = []
        async for chunk in client._astream(mock_async_stream):
            chunks.append(chunk)

        # Verify the correct number of chunks were processed
        assert len(chunks) == 4

        # Verify chunk contents and accumulated content
        assert chunks[0].common.content == "Async"
        assert chunks[0].common.delta == "Async"

        assert chunks[1].common.content == "Async test"
        assert chunks[1].common.delta == " test"

        assert chunks[2].common.content == "Async test complete"
        assert chunks[2].common.delta == " complete"

        assert chunks[3].common.content == "Async test complete"
        assert chunks[3].common.delta == ""
        assert chunks[3].common.finish_reason == "stop"

    def test_create_chimeric_response_no_choices(self, chimeric_groq_client):
        """Cover the branch where response.choices is empty in _create_chimeric_response."""
        client = chimeric_groq_client

        # Simulate a response with no choices
        resp = SimpleNamespace(
            choices=[],
            usage=None,
            model="test-model",
            model_dump=lambda: {"dumped": True},
        )

        result = client._create_chimeric_response(resp, [])
        assert isinstance(result, ChimericCompletionResponse)
        # Since there were no choices, content should default to empty string
        assert result.common.content == ""

    def test_create_chimeric_response(self, chimeric_groq_client):
        """Test _create_chimeric_response correctly converts native responses."""
        client = chimeric_groq_client

        # Create a mock response
        mock_response = MockResponse()

        # Test with no tool calls
        result = client._create_chimeric_response(mock_response, [])
        assert isinstance(result, ChimericCompletionResponse)
        assert result.native is mock_response
        assert isinstance(result.common, CompletionResponse)
        assert result.common.content == "hello"
        assert result.common.model == "mixtral-8x7b"

        # Test with tool calls
        tool_calls = [
            {"call_id": "123", "name": "tool", "arguments": "{}", "result": "tool_result"}
        ]
        result = client._create_chimeric_response(mock_response, tool_calls)
        assert "tool_calls" in result.common.metadata
        assert result.common.metadata["tool_calls"] == tool_calls

    def test_encode_tools(self, chimeric_groq_client):
        """Test that _encode_tools correctly formats tools for the Groq API."""
        client = chimeric_groq_client

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
        assert encoded_tools[0]["function"]["name"] == "get_weather"
        assert (
            encoded_tools[0]["function"]["description"] == "Get weather information for a location"
        )
        assert encoded_tools[0]["function"]["parameters"] == tool_params.model_dump()

        # Verify the second tool (without parameters)
        assert encoded_tools[1]["type"] == "function"
        assert encoded_tools[1]["function"]["name"] == "get_time"
        assert encoded_tools[1]["function"]["description"] == "Get current server time"
        assert encoded_tools[1]["function"]["parameters"] == {}

        # Test with pre-formatted tool dictionary
        pre_formatted_tool = {
            "type": "function",
            "function": {
                "name": "custom_tool",
                "description": "A pre-formatted tool",
                "parameters": {"type": "object"},
            },
        }

        # Test with mix of Tool instances and pre-formatted tools
        mixed_tools = client._encode_tools([tool_with_params, pre_formatted_tool])
        assert isinstance(mixed_tools, list)
        assert len(mixed_tools) == 2
        assert mixed_tools[0]["function"]["name"] == "get_weather"
        assert mixed_tools[1] == pre_formatted_tool  # Should be passed through unchanged

    def test_chat_completion_counts_and_errors(self, chimeric_groq_client, monkeypatch):
        """Test that chat_completion properly tracks request and error counts."""
        client = chimeric_groq_client

        # Test successful request path
        def mock_impl(messages: Any, model: Any, stream: Any, tools: Any = None, **kw: Any) -> str:
            return "DONE"

        monkeypatch.setattr(client, "_chat_completion_impl", mock_impl)
        before_req = client.request_count
        before_err = client.error_count
        out = client.chat_completion("hi", "llama3-8b")

        # Verify success increments request count but not error count
        assert out == "DONE"
        assert client.request_count == before_req + 1
        assert client.error_count == 0

        # Test error handling path
        def bad_impl(*args: Any, **kw: Any):
            raise ValueError("fail")

        monkeypatch.setattr(client, "_chat_completion_impl", bad_impl)
        with pytest.raises(ValueError):
            client.chat_completion([], "mixtral-8x7b")

        # Verify error increments both request and error counts
        assert client.request_count == before_req + 2
        assert client.error_count == before_err + 1

    async def test_achat_completion_counts_and_errors(self, chimeric_groq_client, monkeypatch):
        """Test that achat_completion properly tracks request and error counts."""
        client = chimeric_groq_client

        # Test successful async request
        async def ok_impl(*args: Any, **kw: Any) -> str:
            return "OK"

        monkeypatch.setattr(client, "_achat_completion_impl", ok_impl)
        before_req = client.request_count
        before_err = client.error_count

        # Verify successful async call increments request count
        assert await client.achat_completion("x", "llama3-8b") == "OK"
        assert client.request_count == before_req + 1

        # Test async error handling
        async def err_impl(*args: Any, **kw: Any):
            raise RuntimeError("err")

        monkeypatch.setattr(client, "_achat_completion_impl", err_impl)
        with pytest.raises(RuntimeError):
            await client.achat_completion([], "mixtral-8x7b")

        # Verify error increments error count
        assert client.error_count == before_err + 1

    def test_process_function_call(self, chimeric_groq_client, monkeypatch):
        """Test successful function call processing."""
        client = chimeric_groq_client

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
        mock_function = SimpleNamespace(
            name="get_weather", arguments=json.dumps({"location": "New York", "unit": "fahrenheit"})
        )
        mock_call = SimpleNamespace(id="call_123", function=mock_function)

        # Test the function call processing
        result = client._process_function_call(mock_call)

        # Verify the result structure
        assert result["call_id"] == "call_123"
        assert result["name"] == "get_weather"
        assert result["arguments"] == mock_function.arguments
        assert result["result"] == "Weather in New York is 22°F"

    def test_handle_function_tool_calls_no_calls(self, chimeric_groq_client):
        """Test _handle_function_tool_calls when response has no tool calls."""
        client = chimeric_groq_client

        # Create a response with no tool calls
        mock_response = Mock()
        mock_message = Mock()
        mock_message.tool_calls = None  # No tool calls
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        messages = [{"role": "user", "content": "Hello"}]

        # Test handling when no tool calls are present
        tool_calls, updated_messages = client._handle_function_tool_calls(mock_response, messages)

        assert tool_calls == []
        assert updated_messages == messages  # Should be unchanged

    def test_process_function_call_tool_not_callable(self, chimeric_groq_client, monkeypatch):
        """Test function call processing when tool has a non-callable function."""
        client = chimeric_groq_client

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

        # Create a mock function call with the correct structure
        mock_function = SimpleNamespace(
            name="broken_tool", arguments=json.dumps({"param": "value"})
        )
        mock_call = SimpleNamespace(id="call_789", function=mock_function)

        # Test that ToolRegistrationError is raised with the correct message
        with pytest.raises(ToolRegistrationError) as exc_info:
            client._process_function_call(mock_call)

        assert "Tool 'broken_tool' is not callable" in str(exc_info.value)

    def test_process_function_call_tool_not_registered(self, chimeric_groq_client):
        """Test function call processing when tool is not registered."""
        client = chimeric_groq_client

        # Create a mock function call for unregistered tool
        mock_function = SimpleNamespace(
            name="unregistered_tool", arguments=json.dumps({"param": "value"})
        )
        mock_call = SimpleNamespace(id="call_456", function=mock_function)

        # Test that ToolRegistrationError is raised
        with pytest.raises(ToolRegistrationError) as exc_info:
            client._process_function_call(mock_call)

        assert "unregistered_tool" in str(exc_info.value)
        assert "No tool registered" in str(exc_info.value)

    def test_handle_function_tool_calls_no_choices(self, chimeric_groq_client):
        """Test _handle_function_tool_calls with response having no choices (line 293-294)."""
        client = chimeric_groq_client

        # Create a response with empty choices list
        class MockResponseNoChoices:
            def __init__(self):
                self.choices = []  # Empty choices list

        mock_response = MockResponseNoChoices()
        messages = [{"role": "user", "content": "Test message"}]

        # Test handling when response has no choices
        tool_calls, updated_messages = client._handle_function_tool_calls(mock_response, messages)

        # Should return empty tool calls and unchanged messages
        assert tool_calls == []
        assert updated_messages == messages

    def test_handle_function_tool_calls_with_calls(self, chimeric_groq_client, monkeypatch):
        """Test _handle_function_tool_calls with actual tool calls."""
        client = chimeric_groq_client

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
        mock_call1 = SimpleNamespace(
            id="call_add",
            function=SimpleNamespace(
                name="calculator", arguments=json.dumps({"operation": "add", "a": 5, "b": 3})
            ),
        )

        mock_call2 = SimpleNamespace(
            id="call_mult",
            function=SimpleNamespace(
                name="calculator", arguments=json.dumps({"operation": "multiply", "a": 4, "b": 7})
            ),
        )

        # Create a response with tool calls
        mock_response = MockResponse()
        mock_response.choices[0].message.tool_calls = [mock_call1, mock_call2]

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
        assert len(updated_messages) == 4  # original + assistant message + 2 tool results
        assert updated_messages[0] == {"role": "user", "content": "Calculate 5+3 and 4*7"}

        # Check assistant message with tool calls
        assert updated_messages[1]["role"] == "assistant"
        assert "tool_calls" in updated_messages[1]
        assert len(updated_messages[1]["tool_calls"]) == 2

        # Check tool result messages
        assert updated_messages[2]["role"] == "tool"
        assert updated_messages[2]["tool_call_id"] == "call_add"
        assert updated_messages[2]["content"] == "8"

        assert updated_messages[3]["role"] == "tool"
        assert updated_messages[3]["tool_call_id"] == "call_mult"
        assert updated_messages[3]["content"] == "28"

    def test_chat_completion_impl_non_streaming(self, chimeric_groq_client, monkeypatch):
        """Test _chat_completion_impl with non-streaming response."""
        client = chimeric_groq_client

        # Create a mock response object
        mock_response = MockResponse()

        # Mock the client's chat.completions.create method
        mock_create = Mock(return_value=mock_response)
        client._client.chat = SimpleNamespace(completions=SimpleNamespace(create=mock_create))

        # Mock _filter_kwargs to properly filter out invalid parameters
        def mock_filter_kwargs(func, kwargs):
            # Only allow known valid parameters
            allowed = {"temperature", "max_tokens"}
            return {k: v for k, v in kwargs.items() if k in allowed}

        monkeypatch.setattr(client, "_filter_kwargs", mock_filter_kwargs)

        # Test the implementation
        result = client._chat_completion_impl(
            messages=[{"role": "user", "content": "Hello"}],
            model="llama3-8b",
            temperature=0.7,
            max_tokens=100,
            invalid_param="should_be_filtered",
        )

        # Verify the result
        assert isinstance(result, ChimericCompletionResponse)
        assert result.native is mock_response
        assert isinstance(result.common, CompletionResponse)
        assert result.common.content == "hello"
        assert result.common.model == "mixtral-8x7b"

        # Verify the client was called correctly
        mock_create.assert_called_once_with(
            model="llama3-8b",
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
            tools=NOT_GIVEN,
            temperature=0.7,
            max_tokens=100,
        )

    def test_chat_completion_impl_streaming(self, chimeric_groq_client, monkeypatch):
        """Test _chat_completion_impl with streaming response."""
        client = chimeric_groq_client

        # Create mock stream response
        mock_stream = Mock(spec=client_module.Stream)

        # Mock the client's chat.completions.create method to return stream
        mock_create = Mock(return_value=mock_stream)
        client._client.chat = SimpleNamespace(completions=SimpleNamespace(create=mock_create))

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
            messages=[{"role": "user", "content": "Hello"}], model="mixtral-8x7b", stream=True
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
            model="mixtral-8x7b",
            messages=[{"role": "user", "content": "Hello"}],
            tools=NOT_GIVEN,
            stream=True,
        )

    def test_chat_completion_impl_with_tools(self, chimeric_groq_client, monkeypatch):
        """Test _chat_completion_impl with tool parameter."""
        client = chimeric_groq_client

        # Create mock response
        mock_response = MockResponse()

        mock_create = Mock(return_value=mock_response)
        client._client.chat = SimpleNamespace(completions=SimpleNamespace(create=mock_create))

        # Test with tools
        tool = Tool(name="example_tool", description="Example tool")
        encoded_tool = client._encode_tools([tool])

        client._chat_completion_impl(
            messages=[{"role": "user", "content": "What's the weather?"}],
            model="llama3-8b",
            tools=encoded_tool,
        )

        # Verify tools parameter is handled correctly
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert call_args[1]["tools"] == encoded_tool

    async def test_achat_completion_impl_non_streaming(self, chimeric_groq_client, monkeypatch):
        """Test _achat_completion_impl with non-streaming response."""
        client = chimeric_groq_client

        # Create mock async response
        mock_response = MockResponse()

        # Mock the async client's chat.completions.create method
        mock_create = AsyncMock(return_value=mock_response)
        client._async_client.chat = SimpleNamespace(completions=SimpleNamespace(create=mock_create))

        # Test the async implementation
        result = await client._achat_completion_impl(
            messages=[{"role": "user", "content": "Async hello"}],
            model="llama3-8b",
            temperature=0.5,
        )

        # Verify the result
        assert isinstance(result, ChimericCompletionResponse)
        assert result.native is mock_response
        assert result.common.content == "hello"

        # Verify the async client was called correctly
        mock_create.assert_called_once_with(
            model="llama3-8b",
            messages=[{"role": "user", "content": "Async hello"}],
            stream=False,
            tools=NOT_GIVEN,
            temperature=0.5,
        )

    async def test_achat_completion_impl_streaming(self, chimeric_groq_client, monkeypatch):
        """Test _achat_completion_impl with streaming response."""
        client = chimeric_groq_client

        # Create mock async stream
        mock_stream = Mock(spec=client_module.AsyncStream)

        mock_create = AsyncMock(return_value=mock_stream)
        client._async_client.chat = SimpleNamespace(completions=SimpleNamespace(create=mock_create))

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
            messages=[{"role": "user", "content": "Stream test"}], model="mixtral-8x7b", stream=True
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
            model="mixtral-8x7b",
            messages=[{"role": "user", "content": "Stream test"}],
            tools=NOT_GIVEN,
            stream=True,
        )

    def test_upload_file_success(self, chimeric_groq_client, monkeypatch):
        """Test successful file upload."""
        client = chimeric_groq_client

        # Mock the httpx response
        mock_response = SimpleNamespace(
            is_success=True,
            text="",
            status_code=200,
            json=lambda: {
                "id": "file-id",
                "filename": "file.txt",
                "bytes": 123,
                "purpose": "batch",
                "status": "ready",
                "created_at": 456,
            },
        )

        def mock_post(*args: Any, **kwargs: Any):
            return mock_response

        # Create a proper context manager class
        class MockHttpxClient:
            def __init__(self):
                self.post = mock_post

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return

        monkeypatch.setattr(httpx, "Client", MockHttpxClient)

        # Test with file_path
        with open("test_file.txt", "w") as f:
            f.write("test content")

        try:
            # Perform upload with file_path
            resp = client.upload_file(file_path="test_file.txt")

            # Verify the response
            assert resp.common.file_id == "file-id"
            assert resp.common.filename == "file.txt"
            assert resp.common.bytes == 123
            assert resp.common.purpose == "batch"
            assert resp.common.status == "ready"
            assert resp.common.created_at == 456

        finally:
            # Clean up the test file
            if os.path.exists("test_file.txt"):
                os.remove("test_file.txt")

    def test_upload_file_error(self, chimeric_groq_client, monkeypatch):
        """Test file upload error handling."""
        client = chimeric_groq_client

        # Mock httpx error response
        mock_response = SimpleNamespace(
            is_success=False, text="File upload failed", status_code=400
        )

        def mock_post(*args: Any, **kwargs: Any):
            return mock_response

        # Create a proper context manager class for error case
        class MockHttpxClientError:
            def __init__(self):
                self.post = mock_post

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return

        monkeypatch.setattr(httpx, "Client", MockHttpxClientError)

        # Verify error is wrapped in ProviderError
        with pytest.raises(ProviderError) as ei:
            client.upload_file(file_object="content", filename="test.txt")

        # Check error details
        assert ei.value.provider == "Groq"
        assert ei.value.endpoint == "file_upload"
        assert ei.value.status_code == 400

    def test_upload_file_missing_args(self, chimeric_groq_client):
        """Test file upload with missing required arguments."""
        client = chimeric_groq_client

        # Verify ValueError when no file is provided
        with pytest.raises(ValueError) as ei:
            client.upload_file(purpose="test")

        assert "Either 'file_path' or 'file_object' must be provided" in str(ei.value)

    def test_upload_file_with_file_object_no_filename(self, chimeric_groq_client, monkeypatch):
        """Test file upload with file_object but no explicit filename (line 516-517)."""
        client = chimeric_groq_client

        # Mock successful httpx response
        mock_response = SimpleNamespace(
            is_success=True,
            text="",
            status_code=200,
            json=lambda: {
                "id": "file-obj-no-name",
                "filename": "uploaded_file.txt",
                "bytes": 123,
                "purpose": "batch",
                "status": "ready",
                "created_at": 999,
            },
        )

        class MockHttpxClient:
            def __init__(self):
                self.post_calls = []

            def post(self, *args, **kwargs):
                self.post_calls.append((args, kwargs))
                return mock_response

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        mock_client_instance = MockHttpxClient()
        monkeypatch.setattr(httpx, "Client", lambda: mock_client_instance)

        # Create a file-like object without a 'name' attribute
        class FileObjectWithoutName:
            def __init__(self, content):
                self.content = content
                self.position = 0

            def read(self):
                return self.content

            def seek(self, pos):
                self.position = pos

        file_obj = FileObjectWithoutName(b"file object content")

        # Test upload with file_object but no explicit filename
        resp = client.upload_file(file_object=file_obj)  # No filename parameter

        # Verify the response
        assert resp.common.file_id == "file-obj-no-name"

        # Verify that default filename was used
        post_call = mock_client_instance.post_calls[0]
        files_param = post_call[1]["files"]
        assert files_param["file"][0] == "uploaded_file"  # Default filename

    def test_upload_file_with_file_object_readable(self, chimeric_groq_client, monkeypatch):
        """Test file upload with file_object that has read method (line 520-528)."""
        client = chimeric_groq_client

        # Mock successful httpx response
        mock_response = SimpleNamespace(
            is_success=True,
            text="",
            status_code=200,
            json=lambda: {
                "id": "file-readable",
                "filename": "readable.txt",
                "bytes": 200,
                "purpose": "batch",
                "status": "ready",
                "created_at": 1234,
            },
        )

        class MockHttpxClient:
            def __init__(self):
                self.post_calls = []

            def post(self, *args, **kwargs):
                self.post_calls.append((args, kwargs))
                return mock_response

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        mock_client_instance = MockHttpxClient()
        monkeypatch.setattr(httpx, "Client", lambda: mock_client_instance)

        # Test case 1: File object with read() and seek() methods returning string content
        class ReadableFileObject:
            def __init__(self, content):
                self.content = content
                self.name = "readable_file.txt"
                self.read_called = False
                self.seek_called = False

            def read(self):
                self.read_called = True
                return self.content  # String content

            def seek(self, pos):
                self.seek_called = True

        file_obj_str = ReadableFileObject("string content from file object")

        # Upload with readable file object (string content)
        client.upload_file(file_object=file_obj_str, filename="custom_name.txt")

        # Verify the file object's methods were called
        assert file_obj_str.read_called
        assert file_obj_str.seek_called

        # Verify content was encoded to bytes
        post_call = mock_client_instance.post_calls[0]
        files_param = post_call[1]["files"]
        file_content = files_param["file"][1]
        assert isinstance(file_content, bytes)
        assert file_content == b"string content from file object"

        # Test case 2: File object with read() returning bytes content
        class ReadableFileObjectBytes:
            def __init__(self, content):
                self.content = content
                self.name = "bytes_file.txt"

            def read(self):
                return self.content  # Bytes content

            def seek(self, pos):
                pass

        file_obj_bytes = ReadableFileObjectBytes(b"bytes content from file object")

        # Clear previous calls
        mock_client_instance.post_calls.clear()

        # Upload with readable file object (bytes content)
        client.upload_file(file_object=file_obj_bytes)

        # Verify bytes content was used directly
        post_call2 = mock_client_instance.post_calls[0]
        files_param2 = post_call2[1]["files"]
        file_content2 = files_param2["file"][1]
        assert isinstance(file_content2, bytes)
        assert file_content2 == b"bytes content from file object"

    def test_upload_file_with_non_readable_file_object(self, chimeric_groq_client, monkeypatch):
        """Test file upload with file_object that doesn't have read method (line 529-535)."""
        client = chimeric_groq_client

        # Mock successful httpx response
        mock_response = SimpleNamespace(
            is_success=True,
            text="",
            status_code=200,
            json=lambda: {
                "id": "file-direct",
                "filename": "direct.txt",
                "bytes": 100,
                "purpose": "batch",
                "status": "ready",
                "created_at": 5678,
            },
        )

        class MockHttpxClient:
            def __init__(self):
                self.post_calls = []

            def post(self, *args, **kwargs):
                self.post_calls.append((args, kwargs))
                return mock_response

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        mock_client_instance = MockHttpxClient()
        monkeypatch.setattr(httpx, "Client", lambda: mock_client_instance)

        # Test case 1: file_object is bytes content directly
        bytes_content = b"direct bytes content"
        client.upload_file(file_object=bytes_content, filename="direct_bytes.txt")

        # Verify bytes were used directly
        post_call = mock_client_instance.post_calls[0]
        files_param = post_call[1]["files"]
        file_content = files_param["file"][1]
        assert isinstance(file_content, bytes)
        assert file_content == bytes_content

        # Test case 2: file_object is string content directly
        mock_client_instance.post_calls.clear()
        string_content = "direct string content"
        client.upload_file(file_object=string_content, filename="direct_string.txt")

        # Verify string was encoded to bytes
        post_call2 = mock_client_instance.post_calls[0]
        files_param2 = post_call2[1]["files"]
        file_content2 = files_param2["file"][1]
        assert isinstance(file_content2, bytes)
        assert file_content2 == b"direct string content"

    def test_upload_file_with_file_path_and_filename(
        self, chimeric_groq_client, monkeypatch, tmp_path
    ):
        """Cover the branch where filename is provided for a file_path upload."""
        client = chimeric_groq_client

        # Create a temp file
        file_path = tmp_path / "input.txt"
        file_path.write_text("hello")

        captured = {}

        class CaptureClient:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                pass

            def post(self, *args: Any, **kwargs: Any):
                captured["files"] = kwargs["files"]
                return SimpleNamespace(
                    is_success=True,
                    text="",
                    status_code=200,
                    json=lambda: {
                        "id": "id",
                        "filename": "ignored",
                        "bytes": 0,
                        "purpose": "batch",
                        "status": "ready",
                        "created_at": 0,
                    },
                )

        monkeypatch.setattr(httpx, "Client", lambda: CaptureClient())

        # Provide an explicit filename
        client.upload_file(file_path=str(file_path), filename="custom_name.txt")
        files = captured["files"]
        # The tuple's first element is the filename
        assert files["file"][0] == "custom_name.txt"

    def test_upload_file_file_object_read_only_without_seek(
        self, chimeric_groq_client, monkeypatch
    ):
        """Cover the branch where file_object has read() but no seek()."""
        client = chimeric_groq_client

        # File-like with read only (no seek)
        class ReadNoSeek:
            def __init__(self):
                self.read_called = False

            def read(self):
                self.read_called = True
                return "data"

        obj = ReadNoSeek()
        captured = {}

        class CaptureClient:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                pass

            def post(self, *args: Any, **kwargs: Any):
                captured["files"] = kwargs["files"]
                return SimpleNamespace(
                    is_success=True,
                    text="",
                    status_code=200,
                    json=lambda: {
                        "id": "id",
                        "filename": "fn",
                        "bytes": 0,
                        "purpose": "batch",
                        "status": "ready",
                        "created_at": 0,
                    },
                )

        monkeypatch.setattr(httpx, "Client", lambda: CaptureClient())

        client.upload_file(file_object=obj)
        # Ensure read() was called
        assert obj.read_called
        files = captured["files"]
        # Content should be encoded to bytes
        assert files["file"][1] == b"data"

    def test_chat_completion_impl_tool_calls_two_pass(self, chimeric_groq_client, monkeypatch):
        """Test _chat_completion_impl with tool calls requiring second API call."""
        client = chimeric_groq_client

        # Register a tool
        def mock_search(query: str) -> str:
            return f"Search results for: {query}"

        tool = Tool(name="search", description="Search the web", function=mock_search)
        client.tool_manager.register(
            func=tool.function,
            name=tool.name,
            description=tool.description,
        )

        # Create first response with tool call
        first_response = MockResponse()
        mock_function = SimpleNamespace(
            name="search", arguments=json.dumps({"query": "Python testing"})
        )
        mock_call = SimpleNamespace(id="call_search", function=mock_function)
        first_response.choices[0].message.tool_calls = [mock_call]
        first_response.choices[0].message.content = None

        # Create second response with final content
        second_response = MockResponse()
        second_response.choices[
            0
        ].message.content = "Based on the search results, here's the answer..."

        # Mock the client create method to return different responses
        responses = [first_response, second_response]
        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            result = responses[call_count]
            call_count += 1
            return result

        client._client.chat = SimpleNamespace(completions=SimpleNamespace(create=mock_create))

        # Test the implementation
        messages = [{"role": "user", "content": "Search for Python testing"}]
        result = client._chat_completion_impl(messages=messages, model="llama3-8b", tools=[tool])

        # Verify the result contains tool call metadata
        assert "tool_calls" in result.common.metadata
        assert len(result.common.metadata["tool_calls"]) == 1
        assert (
            result.common.metadata["tool_calls"][0]["result"]
            == "Search results for: Python testing"
        )

        # Verify both API calls were made
        assert call_count == 2

    async def test_achat_completion_impl_with_tool_calls_two_pass(
        self, chimeric_groq_client, monkeypatch
    ):
        """Test _achat_completion_impl with tool calls requiring second API call."""
        client = chimeric_groq_client

        # Register a tool
        def mock_translate(text: str, target_lang: str) -> str:
            return f"Translated '{text}' to {target_lang}"

        tool = Tool(name="translate", description="Translate text", function=mock_translate)
        client.tool_manager.register(
            func=tool.function,
            name=tool.name,
            description=tool.description,
        )

        # First response with tool call
        first_response = MockResponse()
        mock_function = SimpleNamespace(
            name="translate", arguments=json.dumps({"text": "Hello", "target_lang": "Spanish"})
        )
        mock_call = SimpleNamespace(id="call_translate", function=mock_function)
        first_response.choices[0].message.tool_calls = [mock_call]
        first_response.choices[0].message.content = None

        # Second response with final answer
        second_response = MockResponse()
        second_response.choices[0].message.content = "The translation is complete."

        # Mock the async client create method
        responses = [first_response, second_response]
        call_count = 0

        async def mock_async_create(**kwargs: Any):
            nonlocal call_count
            result = responses[call_count]
            call_count += 1
            return result

        client._async_client.chat = SimpleNamespace(
            completions=SimpleNamespace(create=mock_async_create)
        )

        # Test the async implementation
        messages = [{"role": "user", "content": "Translate 'Hello' to Spanish"}]

        result = await client._achat_completion_impl(
            messages=messages, model="llama3-8b", tools=[tool]
        )

        # Verify the result contains tool call metadata
        assert result.common.metadata.get("tool_calls") is not None
        assert len(result.common.metadata["tool_calls"]) == 1
        assert result.common.metadata["tool_calls"][0]["result"] == "Translated 'Hello' to Spanish"

        # Verify both async API calls were made
        assert call_count == 2

    def test_repr_and_str_contains_counts(self, chimeric_groq_client):
        """Test that repr and str methods include request/error counts."""
        # Verify repr contains expected components
        r = repr(chimeric_groq_client)
        assert "GroqClient" in r
        assert "requests=" in r
        assert "errors=" in r

        # Verify str contains expected components
        s = str(chimeric_groq_client)
        assert "GroqClient Client" in s
        assert "- Requests:" in s
        assert "- Errors:" in s
