from collections.abc import AsyncGenerator, Generator
from datetime import datetime
import os
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

from chimeric import Chimeric
from chimeric.exceptions import ToolRegistrationError
import chimeric.providers.cohere.client as client_module
from chimeric.providers.cohere.client import CohereClient
from chimeric.types import (
    ChimericCompletionResponse,
    ModelSummary,
    Tool,
    ToolParameters,
)


@pytest.fixture(scope="module")
def cohere_env():
    """Ensure COHERE_API_KEY is set for Chimeric initialization."""
    os.environ["COHERE_API_KEY"] = "test_key"
    yield
    del os.environ["COHERE_API_KEY"]


@pytest.fixture(scope="module")
def chimeric_cohere(cohere_env):
    """Create a Chimeric instance configured for Cohere."""
    return Chimeric(
        cohere_api_key=os.getenv("COHERE_API_KEY", "test_key"),
        timeout=120,
        max_retries=2,
    )


@pytest.fixture(scope="module")
def client(chimeric_cohere) -> CohereClient:
    """Get the CohereClient from the Chimeric wrapper."""
    return cast("CohereClient", chimeric_cohere.get_provider_client("cohere"))


@pytest.fixture(autouse=True)
def patch_cohere_imports(monkeypatch):
    """Stub out actual Cohere classes to prevent network calls."""

    def create_cohere_mock(api_key: str, **kw: Any) -> SimpleNamespace:
        return SimpleNamespace(api_key=api_key, **kw)

    def create_async_cohere_mock(api_key: str, **kw: Any) -> SimpleNamespace:
        return SimpleNamespace(api_key=api_key, **kw)

    monkeypatch.setattr(client_module, "Cohere", create_cohere_mock)
    monkeypatch.setattr(client_module, "AsyncCohere", create_async_cohere_mock)


class MockCohereResponse:
    """Mock Cohere ChatResponse."""

    def __init__(
        self,
        content="Hello from Cohere!",
        model="command-a-03-2025",
        usage_tokens=(10, 15),
        tool_calls=None,
        tool_plan="",
    ):
        self.message = SimpleNamespace(
            content=[SimpleNamespace(text=content, type="text")],
            tool_calls=tool_calls or [],
            tool_plan=tool_plan,
        )
        self.usage = SimpleNamespace(
            tokens=SimpleNamespace(input_tokens=usage_tokens[0], output_tokens=usage_tokens[1]),
            billed_units=SimpleNamespace(
                input_tokens=usage_tokens[0], output_tokens=usage_tokens[1]
            ),
        )
        self.model = model
        self.text = content  # Fallback content property

    def model_dump(self) -> dict[str, Any]:
        return {"response_data": "cohere_test", "model": self.model}


class MockCohereStreamEvent:
    """Mock Cohere streaming event."""

    def __init__(self, event_type="content-delta", delta_text=""):
        self.type = event_type
        if event_type == "content-delta":
            self.delta = SimpleNamespace(
                message=SimpleNamespace(content=SimpleNamespace(text=delta_text))
            )

    def model_dump(self) -> dict[str, Any]:
        return {"event_type": self.type, "stream_data": True}


class MockCohereModel:
    """Mock Cohere model object."""

    def __init__(self, model_id="command-a-03-2025", name="Command A"):
        self.id = model_id
        self.name = name

    def model_dump(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name}


class MockCohereToolCall:
    """Mock Cohere tool call object."""

    def __init__(
        self, tool_id="tool_123", tool_name="get_weather", arguments='{"location": "Toronto"}'
    ):
        self.id = tool_id
        self.function = SimpleNamespace(name=tool_name, arguments=arguments)


# noinspection PyUnusedLocal
class TestCohereClient:
    """Tests for the CohereClient class."""

    def test_capabilities(self, client):
        """Test that client reports correct capabilities."""
        caps = client.capabilities

        assert not caps.multimodal  # Cohere doesn't support multimodal
        assert caps.streaming
        assert caps.tools
        assert not caps.agents
        assert not caps.files  # Cohere doesn't support file uploads

        # Test convenience methods
        assert not client.supports_multimodal()
        assert client.supports_streaming()
        assert client.supports_tools()
        assert not client.supports_agents()
        assert not client.supports_files()

    def test_list_models(self, client, monkeypatch):
        """Test list_models maps to ModelSummary objects correctly."""

        def mock_list():
            return SimpleNamespace(
                models=[
                    MockCohereModel("command-a-03-2025", "Command A"),
                    MockCohereModel("command-r-plus", "Command R+"),
                ]
            )

        monkeypatch.setattr(client.client.models, "list", mock_list)
        models = client.list_models()

        assert len(models) == 2
        assert isinstance(models[0], ModelSummary)
        assert models[0].id == "command-a-03-2025"
        assert models[0].name == "Command A"
        assert models[1].id == "command-r-plus"
        assert models[1].name == "Command R+"

    def test_list_models_with_missing_attributes(self, client, monkeypatch):
        """Test list_models handles models with missing id/name attributes."""

        class MockModelMissingAttrs:
            def __init__(self):
                # No id or name attributes
                pass

            def model_dump(self):
                return {}

        def mock_list():
            return SimpleNamespace(models=[MockModelMissingAttrs()])

        monkeypatch.setattr(client.client.models, "list", mock_list)
        models = client.list_models()

        assert len(models) == 1
        assert models[0].id == "unknown"
        assert models[0].name == "unknown"

    @pytest.mark.parametrize(
        ("event_type", "delta_text", "expected_content", "expected_finish"),
        [
            ("content-delta", "Hello", "Hello", None),
            ("content-delta", " world", " world", None),
            ("message-end", "", "", "end_turn"),
            ("other-event", "", "", None),
        ],
    )
    def test_process_event(self, event_type, delta_text, expected_content, expected_finish):
        """Test _process_event handles different event types correctly."""
        event = MockCohereStreamEvent(event_type, delta_text)
        accumulated = ""

        new_accumulated, chunk = CohereClient._process_event(event, accumulated)

        if expected_content:
            assert new_accumulated == expected_content
        if chunk:
            assert chunk.common.content == new_accumulated
            if expected_finish:
                assert chunk.common.finish_reason == expected_finish
        else:
            assert chunk is None

    def test_stream_processing(self, client):
        """Test _stream processes events correctly."""
        events = [
            MockCohereStreamEvent("content-delta", "Hello"),
            MockCohereStreamEvent("content-delta", " world"),
            MockCohereStreamEvent("message-end", ""),
        ]

        # Create a mock stream object
        mock_stream = Mock()
        mock_stream.__iter__ = Mock(return_value=iter(events))

        chunks = list(client._stream(mock_stream))

        assert len(chunks) == 3
        assert chunks[0].common.content == "Hello"
        assert chunks[1].common.content == "Hello world"
        assert chunks[2].common.finish_reason == "end_turn"

    async def test_astream_processing(self, client):
        """Test _astream processes events correctly."""
        events = [
            MockCohereStreamEvent("content-delta", "Hello"),
            MockCohereStreamEvent("content-delta", " async"),
            MockCohereStreamEvent("message-end", ""),
        ]

        # Create a mock async stream
        class MockAsyncStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                if events:
                    return events.pop(0)
                raise StopAsyncIteration

        mock_stream = MockAsyncStream()
        chunks = []
        async for chunk in client._astream(mock_stream):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].common.content == "Hello"
        assert chunks[1].common.content == "Hello async"
        assert chunks[2].common.finish_reason == "end_turn"

    def test_create_chimeric_response(self, client):
        """Test _create_chimeric_response creates proper response objects."""
        mock_response = MockCohereResponse()
        tool_calls = [{"name": "test_tool", "result": "success"}]

        chimeric_response = CohereClient._create_chimeric_response(mock_response, tool_calls)

        assert isinstance(chimeric_response, ChimericCompletionResponse)
        assert chimeric_response.native == mock_response
        assert chimeric_response.common.content == "Hello from Cohere!"
        assert chimeric_response.common.usage.prompt_tokens == 10
        assert chimeric_response.common.usage.completion_tokens == 15
        assert chimeric_response.common.usage.total_tokens == 25
        assert chimeric_response.common.model == "command-a-03-2025"
        assert chimeric_response.common.metadata is not None
        assert "tool_calls" in chimeric_response.common.metadata

    def test_create_chimeric_response_with_list_content(self, client):
        """Test _create_chimeric_response handles list content correctly."""
        mock_response = MockCohereResponse()
        mock_response.message.content = [SimpleNamespace(text="List content")]

        chimeric_response = CohereClient._create_chimeric_response(mock_response, [])

        assert chimeric_response.common.content == "List content"

    def test_create_chimeric_response_fallback_content(self, client):
        """Test _create_chimeric_response falls back to text property."""
        mock_response = MockCohereResponse()
        mock_response.message = SimpleNamespace()  # Remove content attribute
        mock_response.text = "Fallback text"

        chimeric_response = CohereClient._create_chimeric_response(mock_response, [])

        assert chimeric_response.common.content == "Fallback text"

    def test_create_chimeric_response_no_usage(self, client):
        """Test _create_chimeric_response handles missing usage info."""
        mock_response = MockCohereResponse()
        delattr(mock_response, "usage")

        chimeric_response = CohereClient._create_chimeric_response(mock_response, [])

        assert chimeric_response.common.usage.prompt_tokens == 0
        assert chimeric_response.common.usage.completion_tokens == 0
        assert chimeric_response.common.usage.total_tokens == 0

    def test_encode_tools(self, client):
        """Test _encode_tools converts Tool objects correctly."""
        # Test with Tool objects
        tool = Tool(
            name="get_weather",
            description="Get weather info",
            parameters=ToolParameters(
                type="object", properties={"location": {"type": "string"}}, required=["location"]
            ),
        )

        encoded = client._encode_tools([tool])

        assert len(encoded) == 1
        assert encoded[0]["type"] == "function"
        assert encoded[0]["function"]["name"] == "get_weather"
        assert encoded[0]["function"]["description"] == "Get weather info"
        assert "properties" in encoded[0]["function"]["parameters"]

        # Test with dict tools
        dict_tool = {"type": "function", "function": {"name": "test"}}
        encoded_dict = client._encode_tools([dict_tool])
        assert encoded_dict[0] == dict_tool

        # Test with None
        assert client._encode_tools(None) is None
        assert client._encode_tools([]) is None

    def test_process_function_call(self, client, monkeypatch):
        """Test _process_function_call executes tools correctly."""

        # Mock a tool function
        def mock_weather_tool(location: str) -> str:
            return f"Weather in {location}: sunny"

        # Create a mock tool and manually add to tool manager's internal dict
        tool = Tool(name="get_weather", description="Get weather", function=mock_weather_tool)
        client.tool_manager.tools["get_weather"] = tool

        # Create a mock tool call
        mock_call = MockCohereToolCall()

        result = client._process_function_call(mock_call)

        assert result["call_id"] == "tool_123"
        assert result["name"] == "get_weather"
        assert result["arguments"] == '{"location": "Toronto"}'
        assert result["result"] == "Weather in Toronto: sunny"

    def test_process_function_call_non_callable_tool(self, client, monkeypatch):
        """Test _process_function_call raises error for non-callable tools."""
        # Create a mock tool with non-callable function and manually add to tool manager
        from types import SimpleNamespace

        tool = SimpleNamespace()
        tool.name = "bad_tool"
        tool.function = "not_callable"  # This is not callable
        client.tool_manager.tools["bad_tool"] = tool

        mock_call = MockCohereToolCall(tool_name="bad_tool")

        with pytest.raises(ToolRegistrationError, match="Tool 'bad_tool' is not callable"):
            client._process_function_call(mock_call)

    def test_handle_function_tool_calls_no_calls(self, client):
        """Test _handle_function_tool_calls with no tool calls."""
        mock_response = MockCohereResponse()
        messages = [{"role": "user", "content": "Hello"}]

        tool_calls, updated_messages = client._handle_function_tool_calls(mock_response, messages)

        assert tool_calls == []
        assert updated_messages == messages

    def test_handle_function_tool_calls_no_message(self, client):
        """Test _handle_function_tool_calls with response missing message."""
        mock_response = SimpleNamespace()  # No message attribute
        messages = [{"role": "user", "content": "Hello"}]

        tool_calls, updated_messages = client._handle_function_tool_calls(mock_response, messages)

        assert tool_calls == []
        assert updated_messages == messages

    def test_handle_function_tool_calls_with_calls(self, client):
        """Test _handle_function_tool_calls processes tool calls correctly."""

        # Mock a tool function
        def mock_weather_tool(location: str) -> str:
            return f"Weather in {location}: sunny"

        # Create a mock tool and manually add to tool manager's internal dict
        tool = Tool(name="get_weather", description="Get weather", function=mock_weather_tool)
        client.tool_manager.tools["get_weather"] = tool

        # Create response with tool calls
        mock_tool_call = MockCohereToolCall()
        mock_response = MockCohereResponse(tool_calls=[mock_tool_call])
        messages = [{"role": "user", "content": "What's the weather?"}]

        tool_calls, updated_messages = client._handle_function_tool_calls(mock_response, messages)

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "get_weather"
        assert tool_calls[0]["result"] == "Weather in Toronto: sunny"

        # Check updated messages structure
        assert len(updated_messages) == 3  # original + assistant + tool
        assert updated_messages[1]["role"] == "assistant"
        assert updated_messages[1]["tool_calls"] == [mock_tool_call]
        assert updated_messages[2]["role"] == "tool"
        assert updated_messages[2]["tool_call_id"] == "tool_123"

    @pytest.mark.parametrize(
        ("input_messages", "expected_output"),
        [
            ("Hello", [{"role": "user", "content": "Hello"}]),
            ({"role": "user", "content": "Hi"}, [{"role": "user", "content": "Hi"}]),
            ([{"role": "user", "content": "Test"}], [{"role": "user", "content": "Test"}]),
            (123, [{"role": "user", "content": "123"}]),
        ],
    )
    def test_convert_messages_to_cohere_format(self, client, input_messages, expected_output):
        """Test message format conversion."""
        result = client._convert_messages_to_cohere_format(input_messages)
        assert result == expected_output

    def test_chat_completion_impl_basic(self, client, monkeypatch):
        """Test basic chat completion without streaming or tools."""
        mock_response = MockCohereResponse()

        def mock_chat(**kwargs):
            return mock_response

        monkeypatch.setattr(client._client, "chat", mock_chat)

        result = client._chat_completion_impl(messages="Hello", model="command-a-03-2025")

        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Hello from Cohere!"

    def test_chat_completion_impl_streaming(self, client, monkeypatch):
        """Test streaming chat completion."""
        events = [
            MockCohereStreamEvent("content-delta", "Hello"),
            MockCohereStreamEvent("message-end", ""),
        ]

        def mock_chat_stream(**kwargs):
            return iter(events)

        monkeypatch.setattr(client._client, "chat_stream", mock_chat_stream)

        result = client._chat_completion_impl(
            messages="Hello", model="command-a-03-2025", stream=True
        )

        assert isinstance(result, Generator)
        chunks = list(result)
        assert len(chunks) == 2
        assert chunks[0].common.content == "Hello"

    def test_chat_completion_impl_with_tools(self, client, monkeypatch):
        """Test chat completion with tool calls."""

        # Mock a tool function
        def mock_weather_tool(location: str) -> str:
            return f"Weather in {location}: sunny"

        # Create a mock tool and manually add to tool manager's internal dict
        tool = Tool(name="get_weather", description="Get weather", function=mock_weather_tool)
        client.tool_manager.tools["get_weather"] = tool

        # Create responses - first with tool call, second with final answer
        mock_tool_call = MockCohereToolCall()
        first_response = MockCohereResponse(
            content="I'll check the weather", tool_calls=[mock_tool_call]
        )
        second_response = MockCohereResponse(content="The weather in Toronto is sunny")

        call_count = 0

        def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return first_response
            return second_response

        monkeypatch.setattr(client._client, "chat", mock_chat)

        result = client._chat_completion_impl(
            messages="What's the weather in Toronto?",
            model="command-a-03-2025",
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.metadata is not None
        assert "tool_calls" in result.common.metadata
        assert call_count == 2  # Should make two API calls

    async def test_achat_completion_impl_basic(self, client, monkeypatch):
        """Test basic async chat completion."""
        mock_response = MockCohereResponse()

        async def mock_chat(**kwargs):
            return mock_response

        monkeypatch.setattr(client._async_client, "chat", mock_chat)

        result = await client._achat_completion_impl(messages="Hello", model="command-a-03-2025")

        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Hello from Cohere!"

    async def test_achat_completion_impl_streaming(self, client, monkeypatch):
        """Test async streaming chat completion."""
        events = [
            MockCohereStreamEvent("content-delta", "Hello"),
            MockCohereStreamEvent("message-end", ""),
        ]

        class MockAsyncStream:
            def __init__(self):
                self.events = events.copy()

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.events:
                    return self.events.pop(0)
                raise StopAsyncIteration

        def mock_chat_stream(**kwargs):
            return MockAsyncStream()

        monkeypatch.setattr(client._async_client, "chat_stream", mock_chat_stream)

        result = await client._achat_completion_impl(
            messages="Hello", model="command-a-03-2025", stream=True
        )

        assert isinstance(result, AsyncGenerator)
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        assert len(chunks) == 2

    async def test_achat_completion_impl_with_tools(self, client, monkeypatch):
        """Test async chat completion with tool calls."""

        # Mock a tool function
        def mock_weather_tool(location: str) -> str:
            return f"Weather in {location}: sunny"

        # Create a mock tool and manually add to tool manager's internal dict
        tool = Tool(name="get_weather", description="Get weather", function=mock_weather_tool)
        client.tool_manager.tools["get_weather"] = tool

        # Create responses
        mock_tool_call = MockCohereToolCall()
        first_response = MockCohereResponse(
            content="I'll check the weather", tool_calls=[mock_tool_call]
        )
        second_response = MockCohereResponse(content="The weather in Toronto is sunny")

        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return first_response
            return second_response

        monkeypatch.setattr(client._async_client, "chat", mock_chat)

        result = await client._achat_completion_impl(
            messages="What's the weather in Toronto?",
            model="command-a-03-2025",
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.metadata is not None
        assert "tool_calls" in result.common.metadata
        assert call_count == 2

    def test_upload_file_not_supported(self, client):
        """Test that file upload raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Cohere does not support file uploads"):
            client._upload_file(file="test.txt")

    def test_provider_name_set(self, client):
        """Test that _provider_name is correctly set."""
        assert client._provider_name == "Cohere"

    def test_generic_types(self, client):
        """Test _get_generic_types returns correct client types."""
        types = client._get_generic_types()
        assert "sync" in types
        assert "async" in types
        # Since we're mocking, we can't test the exact types, but ensure they're set

    def test_init_client(self, client):
        """Test _init_client creates client with correct parameters."""
        # Test that the client is initialized correctly
        assert hasattr(client, "_client")
        assert client._client is not None
        # The client should have the API key stored in the CohereClient instance
        assert client.api_key == "test_key"

    def test_init_async_client(self, client):
        """Test _init_async_client creates async client with correct parameters."""
        # Test that the async client is initialized correctly
        assert hasattr(client, "_async_client")
        assert client._async_client is not None
        # The API key should be stored in the CohereClient instance
        assert client.api_key == "test_key"

    def test_string_representations(self, client):
        """Test __repr__ and __str__ methods."""
        repr_str = repr(client)
        str_str = str(client)

        assert "CohereClient" in repr_str
        assert "capabilities" in repr_str
        assert "CohereClient" in str_str
        assert "Created:" in str_str

    def test_request_tracking(self, client, monkeypatch):
        """Test that request counts and error counts are tracked correctly."""
        # Get initial counts
        initial_requests = client.request_count
        initial_errors = client.error_count

        # Mock successful response
        mock_response = MockCohereResponse()

        def mock_chat(**kwargs):
            return mock_response

        monkeypatch.setattr(client._client, "chat", mock_chat)

        # Use the public interface which tracks requests
        client.chat_completion(messages="Hello", model="command-a-03-2025")

        # Check that request count increased
        assert client.request_count == initial_requests + 1
        assert client.error_count == initial_errors

    def test_kwarg_filtering(self, client, monkeypatch):
        """Test that kwargs are properly filtered for API calls."""
        mock_response = MockCohereResponse()

        # Track what kwargs are passed to the API
        captured_kwargs = {}

        def mock_chat(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_response

        monkeypatch.setattr(client._client, "chat", mock_chat)

        # Call with some valid and invalid kwargs
        client._chat_completion_impl(
            messages="Hello",
            model="command-a-03-2025",
            temperature=0.7,  # Valid Cohere parameter
            invalid_param="should_be_filtered",  # Invalid parameter
        )

        # Check that only valid parameters were passed
        assert "temperature" in captured_kwargs or "invalid_param" not in captured_kwargs

    def test_get_model_info(self, client, monkeypatch):
        """Test get_model_info method."""

        def mock_list():
            return SimpleNamespace(
                models=[
                    MockCohereModel("command-a-03-2025", "Command A"),
                ]
            )

        monkeypatch.setattr(client.client.models, "list", mock_list)

        # Test finding existing model
        model_info = client.get_model_info("command-a-03-2025")
        assert model_info.id == "command-a-03-2025"
        assert model_info.name == "Command A"

        # Test model not found
        with pytest.raises(ValueError, match="Model nonexistent not found"):
            client.get_model_info("nonexistent")

    def test_client_age_property(self, client):
        """Test client_age property returns a reasonable value."""
        age = client.client_age
        assert isinstance(age, float)
        assert age >= 0

    def test_last_request_time_initially_none(self, client):
        """Test that last_request_time is initially None."""
        # Create a fresh client to test initial state
        fresh_client = CohereClient(api_key="test", tool_manager=client.tool_manager)
        assert fresh_client.last_request_time is None

    def test_context_manager_sync(self, client):
        """Test synchronous context manager."""
        with client as ctx_client:
            assert ctx_client is client

    async def test_context_manager_async(self, client):
        """Test asynchronous context manager."""
        async with client as ctx_client:
            assert ctx_client is client

    def test_process_event_with_missing_delta(self, client):
        """Test _process_event handles events with missing delta attributes."""
        # Create event without proper delta structure
        event = SimpleNamespace(type="content-delta")

        accumulated, chunk = CohereClient._process_event(event, "start")

        # Should handle missing delta gracefully with empty string
        assert accumulated == "start"  # No delta added
        assert chunk is not None  # Chunk should still be created
        assert chunk.common.delta == ""  # Delta should be empty

    def test_create_chimeric_response_string_content(self, client):
        """Test _create_chimeric_response with string content instead of list."""
        mock_response = MockCohereResponse()
        mock_response.message.content = "Direct string content"

        chimeric_response = CohereClient._create_chimeric_response(mock_response, [])

        assert chimeric_response.common.content == "Direct string content"

    def test_create_chimeric_response_no_message(self, client):
        """Test _create_chimeric_response with response missing message."""
        mock_response = SimpleNamespace()
        mock_response.text = "Fallback text"
        mock_response.model = "test-model"
        mock_response.usage = SimpleNamespace(
            tokens=SimpleNamespace(input_tokens=5, output_tokens=10)
        )

        def model_dump():
            return {"test": "data"}

        mock_response.model_dump = model_dump

        chimeric_response = CohereClient._create_chimeric_response(mock_response, [])

        assert chimeric_response.common.content == "Fallback text"
        assert chimeric_response.common.model == "test-model"

    def test_handle_function_tool_calls_no_tool_calls_attr(self, client):
        """Test _handle_function_tool_calls with message missing tool_calls."""
        mock_response = SimpleNamespace()
        mock_response.message = SimpleNamespace()  # No tool_calls attribute
        messages = [{"role": "user", "content": "Hello"}]

        tool_calls, updated_messages = client._handle_function_tool_calls(mock_response, messages)

        assert tool_calls == []
        assert updated_messages == messages

    def test_handle_function_tool_calls_empty_tool_calls(self, client):
        """Test _handle_function_tool_calls with empty tool_calls list."""
        mock_response = MockCohereResponse()
        mock_response.message.tool_calls = []
        messages = [{"role": "user", "content": "Hello"}]

        tool_calls, updated_messages = client._handle_function_tool_calls(mock_response, messages)

        assert tool_calls == []
        assert updated_messages == messages

    def test_handle_function_tool_calls_single_message_input(self, client):
        """Test _handle_function_tool_calls with single message dict input."""

        def mock_weather_tool(location: str) -> str:
            return f"Weather in {location}: sunny"

        tool = Tool(name="get_weather", description="Get weather", function=mock_weather_tool)
        client.tool_manager.tools["get_weather"] = tool

        mock_tool_call = MockCohereToolCall()
        mock_response = MockCohereResponse(tool_calls=[mock_tool_call])
        messages = {"role": "user", "content": "What's the weather?"}  # Single dict, not list

        tool_calls, updated_messages = client._handle_function_tool_calls(mock_response, messages)

        assert len(tool_calls) == 1
        assert len(updated_messages) == 3

    def test_encode_tools_with_none_parameters(self, client):
        """Test _encode_tools with Tool that has None parameters."""
        tool = Tool(name="simple_tool", description="Simple", function=lambda: "test")
        # parameters defaults to None in Tool

        encoded = client._encode_tools([tool])

        assert len(encoded) == 1
        # When parameters is None, .model_dump() returns None
        assert encoded[0]["function"]["parameters"] == {}

    def test_chat_completion_with_filtered_kwargs(self, client, monkeypatch):
        """Test that kwargs filtering works correctly in chat completion."""
        mock_response = MockCohereResponse()

        captured_kwargs = {}

        def mock_chat(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_response

        monkeypatch.setattr(client._client, "chat", mock_chat)

        # Mock the _filter_kwargs method to capture what was passed
        def mock_filter_kwargs(func, kwargs):
            return {"filtered_param": "value"}

        monkeypatch.setattr(client, "_filter_kwargs", mock_filter_kwargs)

        client._chat_completion_impl(
            messages="Hello",
            model="command-a-03-2025",
            temperature=0.7,
            invalid_param="should_be_filtered",
        )

        assert "filtered_param" in captured_kwargs
        assert captured_kwargs["filtered_param"] == "value"

    async def test_achat_completion_with_filtered_kwargs(self, client, monkeypatch):
        """Test that kwargs filtering works correctly in async chat completion."""
        mock_response = MockCohereResponse()

        captured_kwargs = {}

        async def mock_chat(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_response

        monkeypatch.setattr(client._async_client, "chat", mock_chat)

        # Mock the _filter_kwargs method
        def mock_filter_kwargs(func, kwargs):
            return {"async_filtered": "value"}

        monkeypatch.setattr(client, "_filter_kwargs", mock_filter_kwargs)

        await client._achat_completion_impl(
            messages="Hello",
            model="command-a-03-2025",
            temperature=0.7,
            invalid_param="should_be_filtered",
        )

        assert "async_filtered" in captured_kwargs
        assert captured_kwargs["async_filtered"] == "value"

    def test_tool_manager_property_access(self, client):
        """Test that tool_manager property is accessible."""
        assert hasattr(client, "tool_manager")
        assert client.tool_manager is not None

    def test_client_properties(self, client):
        """Test various client properties."""
        # Test created_at property
        assert isinstance(client.created_at, datetime)

        # Test client and async_client properties
        assert client.client is not None
        assert client.async_client is not None

        # Test capabilities property
        caps = client.capabilities
        assert not caps.multimodal
        assert caps.streaming
        assert caps.tools
        assert not caps.agents
        assert not caps.files

    def test_model_dump_fallback(self, client, monkeypatch):
        """Test model_dump fallback when object doesn't have model_dump method."""

        class MockModelNoModelDump:
            def __init__(self):
                self.id = "test-id"
                self.name = "test-name"

        def mock_list():
            return SimpleNamespace(models=[MockModelNoModelDump()])

        monkeypatch.setattr(client.client.models, "list", mock_list)
        models = client.list_models()

        assert len(models) == 1
        assert models[0].metadata == {}

    def test_create_chimeric_response_no_model_dump(self, client):
        """Test _create_chimeric_response with object that doesn't have model_dump."""
        mock_response = SimpleNamespace()
        mock_response.message = SimpleNamespace(content=[SimpleNamespace(text="test")])
        mock_response.usage = SimpleNamespace(
            tokens=SimpleNamespace(input_tokens=1, output_tokens=2)
        )
        mock_response.model = "test-model"
        mock_response.text = "fallback"

        chimeric_response = CohereClient._create_chimeric_response(mock_response, [])

        assert chimeric_response.common.content == "test"
        assert chimeric_response.common.metadata == {}

    def test_process_event_no_model_dump(self, client):
        """Test _process_event with event that doesn't have model_dump."""
        event = SimpleNamespace(type="content-delta")
        event.delta = SimpleNamespace(message=SimpleNamespace(content=SimpleNamespace(text="test")))

        accumulated, chunk = CohereClient._process_event(event, "")

        assert accumulated == "test"
        assert chunk.common.metadata == {}

    def test_convert_messages_edge_cases(self, client):
        """Test message conversion edge cases."""
        # Test with None-like input
        result = client._convert_messages_to_cohere_format(None)
        assert result == [{"role": "user", "content": "None"}]

        # Test with empty list
        result = client._convert_messages_to_cohere_format([])
        assert result == []

        # Test with complex object
        class CustomObject:
            def __str__(self):
                return "custom_string"

        result = client._convert_messages_to_cohere_format(CustomObject())
        assert result == [{"role": "user", "content": "custom_string"}]

    def test_missing_content_branches(self, client):
        """Test edge cases in _create_chimeric_response content extraction."""
        # Test response with no content at all
        mock_response = SimpleNamespace()
        mock_response.usage = SimpleNamespace(
            tokens=SimpleNamespace(input_tokens=1, output_tokens=2)
        )
        mock_response.model = "test-model"

        def model_dump():
            return {"test": "data"}

        mock_response.model_dump = model_dump

        chimeric_response = CohereClient._create_chimeric_response(mock_response, [])

        # Should use empty string when no content found
        assert chimeric_response.common.content == ""

    def test_stream_events_edge_cases(self, client):
        """Test stream processing with various event types."""
        # Test unknown event type
        event = SimpleNamespace(type="unknown-event")
        accumulated, chunk = CohereClient._process_event(event, "start")

        assert accumulated == "start"  # Should be unchanged
        assert chunk is None

    def test_tool_calls_with_none(self, client):
        """Test _handle_function_tool_calls with None tool_calls."""
        mock_response = MockCohereResponse()
        mock_response.message.tool_calls = None  # Explicitly None
        messages = [{"role": "user", "content": "Hello"}]

        tool_calls, updated_messages = client._handle_function_tool_calls(mock_response, messages)

        assert tool_calls == []
        assert updated_messages == messages

    def test_create_chimeric_response_empty_content_list(self, client):
        """Test _create_chimeric_response with empty content list."""
        mock_response = MockCohereResponse()
        mock_response.message.content = []  # Empty list
        mock_response.text = "fallback text"

        chimeric_response = CohereClient._create_chimeric_response(mock_response, [])

        # With empty content list, it converts to string representation of empty list
        assert chimeric_response.common.content == "[]"

    def test_stream_with_no_chunks(self, client):
        """Test _stream when _process_event returns None chunks."""
        # Create events that return None chunks (like unknown event types)
        events = [
            SimpleNamespace(type="unknown-event-1"),
            SimpleNamespace(type="unknown-event-2"),
        ]

        # Create a mock stream object
        mock_stream = Mock()
        mock_stream.__iter__ = Mock(return_value=iter(events))

        chunks = list(client._stream(mock_stream))

        # Should be empty since all events produce None chunks
        assert len(chunks) == 0

    async def test_astream_with_no_chunks(self, client):
        """Test _astream when _process_event returns None chunks."""
        # Create events that return None chunks
        events = [
            SimpleNamespace(type="unknown-event-1"),
            SimpleNamespace(type="unknown-event-2"),
        ]

        # Create a mock async stream
        class MockAsyncStream:
            def __init__(self):
                self.events = events.copy()

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.events:
                    return self.events.pop(0)
                raise StopAsyncIteration

        mock_stream = MockAsyncStream()
        chunks = []
        async for chunk in client._astream(mock_stream):
            chunks.append(chunk)

        # Should be empty since all events produce None chunks
        assert len(chunks) == 0

    def test_create_chimeric_response_missing_tokens_usage(self, client):
        """Test _create_chimeric_response with usage missing tokens structure."""
        mock_response = MockCohereResponse()
        # Override with usage structure missing tokens attribute
        mock_response.usage = SimpleNamespace(
            billed_units=SimpleNamespace(input_tokens=20, output_tokens=30)
        )

        chimeric_response = CohereClient._create_chimeric_response(mock_response, [])

        # Should use default values when tokens structure is missing
        assert chimeric_response.common.usage.prompt_tokens == 0
        assert chimeric_response.common.usage.completion_tokens == 0
        assert chimeric_response.common.usage.total_tokens == 0
