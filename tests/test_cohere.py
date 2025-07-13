from collections.abc import AsyncGenerator, Generator
from datetime import datetime
import os
from types import SimpleNamespace
from typing import Any, cast

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

    def __call__(self, **kwargs: Any) -> "MockCohereResponse":
        """Make the instance callable to replace lambda functions."""
        return self

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


# noinspection PyUnusedLocal,PyTypeChecker
class TestCohereClient:
    """Tests for the CohereClient class."""

    def test_client_initialization_and_properties(self, client):
        """Test client initialization, capabilities, and properties."""
        # Test capabilities
        caps = client.capabilities
        assert not caps.multimodal
        assert caps.streaming
        assert caps.tools
        assert not caps.agents
        assert not caps.files

        # Test convenience methods
        assert not client.supports_multimodal()
        assert client.supports_streaming()
        assert client.supports_tools()
        assert not client.supports_agents()
        assert not client.supports_files()

        # Test provider name
        assert client._provider_name == "Cohere"

        # Test generic types
        types = client._get_generic_types()
        assert "sync" in types
        assert "async" in types

        # Test client properties
        assert client.client is not None
        assert client.async_client is not None
        assert isinstance(client.created_at, datetime)
        assert isinstance(client.client_age, float)
        assert client.client_age >= 0
        assert client.last_request_time is None  # Initially None
        assert client.tool_manager is not None

        # Test string representations
        repr_str = repr(client)
        str_str = str(client)
        assert "CohereClient" in repr_str
        assert "capabilities" in repr_str
        assert "CohereClient" in str_str
        assert "Created:" in str_str

        # Test context managers
        with client as ctx_client:
            assert ctx_client is client

    async def test_async_context_manager(self, client):
        """Test asynchronous context manager."""
        async with client as ctx_client:
            assert ctx_client is client

    def test_list_models_comprehensive(self, client, monkeypatch):
        """Test list_models with various edge cases."""

        # Test normal models
        def mock_list_normal():
            return SimpleNamespace(
                models=[
                    MockCohereModel("command-a-03-2025", "Command A"),
                    MockCohereModel("command-r-plus", "Command R+"),
                ]
            )

        monkeypatch.setattr(client.client.models, "list", mock_list_normal)
        models = client.list_models()

        assert len(models) == 2
        assert isinstance(models[0], ModelSummary)
        assert models[0].id == "command-a-03-2025"
        assert models[0].name == "Command A"
        assert models[1].id == "command-r-plus"
        assert models[1].name == "Command R+"

        # Test model with missing attributes
        class MockModelMissingAttrs:
            @staticmethod
            def model_dump():
                return {}

        def mock_list_missing():
            return SimpleNamespace(models=[MockModelMissingAttrs()])

        monkeypatch.setattr(client.client.models, "list", mock_list_missing)
        models = client.list_models()

        assert len(models) == 1
        assert models[0].id == "unknown"
        assert models[0].name == "unknown"


        # Test model without model_dump
        class MockModelNoModelDump:
            def __init__(self):
                self.id = "test-id"
                self.name = "test-name"

        def mock_list_no_dump():
            return SimpleNamespace(models=[MockModelNoModelDump()])

        monkeypatch.setattr(client.client.models, "list", mock_list_no_dump)
        models = client.list_models()

        assert len(models) == 1
        assert models[0].metadata == {}

        # Test get_model_info
        monkeypatch.setattr(client.client.models, "list", mock_list_normal)
        model_info = client.get_model_info("command-a-03-2025")
        assert model_info.id == "command-a-03-2025"
        assert model_info.name == "Command A"

        with pytest.raises(ValueError, match="Model nonexistent not found"):
            client.get_model_info("nonexistent")

    def test_process_event_comprehensive(self):
        """Test _process_event with all edge cases including None chunk generation."""
        # Test content-delta event
        event = MockCohereStreamEvent("content-delta", "Hello")
        accumulated, chunk = CohereClient._process_event(event, "")
        assert accumulated == "Hello"
        assert chunk.common.content == "Hello"
        assert chunk.common.delta == "Hello"

        # Test message-end event
        event = MockCohereStreamEvent("message-end", "")
        accumulated, chunk = CohereClient._process_event(event, "existing")
        assert accumulated == "existing"
        assert chunk.common.finish_reason == "end_turn"

        # Test unknown event type (should return None chunk)
        event = MockCohereStreamEvent("unknown-event", "")
        accumulated, chunk = CohereClient._process_event(event, "start")
        assert accumulated == "start"
        assert chunk is None  # This is the missing branch

        # Test event with missing delta structure
        event = SimpleNamespace(type="content-delta")
        accumulated, chunk = CohereClient._process_event(event, "start")
        assert accumulated == "start"
        assert chunk.common.delta == ""

        # Test event without model_dump
        event = SimpleNamespace(type="content-delta")
        event.delta = SimpleNamespace(message=SimpleNamespace(content=SimpleNamespace(text="test")))
        accumulated, chunk = CohereClient._process_event(event, "")
        assert accumulated == "test"
        assert chunk.common.metadata == {}

        # Test event with type attribute error
        event_no_type = SimpleNamespace()
        with pytest.raises(AttributeError):
            CohereClient._process_event(event_no_type, "start")

    def test_streaming_comprehensive(self, client, monkeypatch):
        """Test streaming with all edge cases, including None chunks."""
        # Test basic streaming with None chunks
        events = [
            MockCohereStreamEvent("unknown-event-1", ""),  # Will produce None chunk
            MockCohereStreamEvent("content-delta", "Hello"),
            MockCohereStreamEvent("unknown-event-2", ""),  # Will produce None chunk
            MockCohereStreamEvent("content-delta", " world"),
            MockCohereStreamEvent("message-end", ""),
        ]

        def mock_chat_stream(**kwargs: Any) -> Any:
            return iter(events)

        monkeypatch.setattr(client._client, "chat_stream", mock_chat_stream)

        chunks = list(
            client._stream(
                messages=[{"role": "user", "content": "Hello"}],
                model="command-a-03-2025",
                tools=None,
                tools_enabled=False,
            )
        )

        # Should only have 3 chunks (2 content-delta + 1 message-end), not 5
        assert len(chunks) == 3
        assert chunks[0].common.content == "Hello"
        assert chunks[1].common.content == "Hello world"
        assert chunks[2].common.finish_reason == "end_turn"

        # Test streaming with tools
        def mock_weather_tool(location: str) -> str:
            return f"Weather in {location}: sunny"

        client.tool_manager.register(
            mock_weather_tool, name="get_weather", description="Get weather"
        )

        mock_tool_call = MockCohereToolCall()
        response_with_tools = MockCohereResponse(tool_calls=[mock_tool_call])
        response_final = MockCohereResponse()

        events_first = [
            MockCohereStreamEvent("processing-event", ""),  # Will produce None chunk
            MockCohereStreamEvent("content-delta", "Processing"),
            MockCohereStreamEvent("message-end", ""),
        ]
        events_final = [
            MockCohereStreamEvent("content-delta", "Done"),
            MockCohereStreamEvent("message-end", ""),
        ]

        call_count = 0
        stream_call_count = 0

        def mock_chat_stream_with_tools(**kwargs: Any):
            nonlocal stream_call_count
            stream_call_count += 1
            if stream_call_count == 1:
                return iter(events_first)
            return iter(events_final)

        def mock_chat(**kwargs: Any):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return response_with_tools
            return response_final

        monkeypatch.setattr(client._client, "chat_stream", mock_chat_stream_with_tools)
        monkeypatch.setattr(client._client, "chat", mock_chat)

        chunks = list(
            client._stream(
                messages=[{"role": "user", "content": "Use tool"}],
                model="command-a-03-2025",
                tools=[{"name": "get_weather", "description": "Test", "parameters": {}}],
                tools_enabled=True,
            )
        )

        # Should have chunks from both streaming responses, excluding None chunks
        assert len(chunks) >= 2
        assert call_count >= 1

    async def test_async_streaming_comprehensive(self, client, monkeypatch):
        """Test async streaming with all edge cases, including None chunks."""
        # Test basic async streaming with None chunks
        events = [
            MockCohereStreamEvent("unknown-async-1", ""),  # Will produce None chunk
            MockCohereStreamEvent("content-delta", "Hello"),
            MockCohereStreamEvent("unknown-async-2", ""),  # Will produce None chunk
            MockCohereStreamEvent("content-delta", " async"),
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

            def __call__(self, **kwargs: Any) -> Any:
                return self

        mock_stream = MockAsyncStream()
        monkeypatch.setattr(client._async_client, "chat_stream", mock_stream)

        chunks = []
        async for chunk in client._astream(
            messages=[{"role": "user", "content": "Hello"}],
            model="command-a-03-2025",
            tools=None,
            tools_enabled=False,
        ):
            chunks.append(chunk)

        # Should only have 3 chunks (2 content-delta + 1 message-end), not 5
        assert len(chunks) == 3
        assert chunks[0].common.content == "Hello"
        assert chunks[1].common.content == "Hello async"
        assert chunks[2].common.finish_reason == "end_turn"

        # Test async streaming with tools
        def async_test_tool(query: str) -> str:
            return f"Async tool result for: {query}"

        client.tool_manager.register(
            async_test_tool, name="async_test_tool", description="Async test tool"
        )

        mock_tool_call = MockCohereToolCall(
            tool_name="async_test_tool", arguments='{"query": "async_test"}'
        )
        response_with_tools = MockCohereResponse(tool_calls=[mock_tool_call])
        response_final = MockCohereResponse()

        events_first = [
            MockCohereStreamEvent("async-processing", ""),  # Will produce None chunk
            MockCohereStreamEvent("content-delta", "Async processing"),
            MockCohereStreamEvent("message-end", ""),
        ]
        events_final = [
            MockCohereStreamEvent("content-delta", "Async done"),
            MockCohereStreamEvent("message-end", ""),
        ]

        call_count = 0
        stream_call_count = 0

        async def mock_async_stream(**kwargs: Any):
            nonlocal stream_call_count
            stream_call_count += 1
            if stream_call_count == 1:
                for event in events_first:
                    yield event
            else:
                for event in events_final:
                    yield event

        async def mock_async_chat(**kwargs: Any):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return response_with_tools
            return response_final

        monkeypatch.setattr(client._async_client, "chat_stream", mock_async_stream)
        monkeypatch.setattr(client._async_client, "chat", mock_async_chat)

        chunks = []
        async for chunk in client._astream(
            messages=[{"role": "user", "content": "Use async tool"}],
            model="command-a-03-2025",
            tools=[{"name": "async_test_tool", "description": "Async test", "parameters": {}}],
            tools_enabled=True,
        ):
            chunks.append(chunk)

        assert len(chunks) >= 2
        assert call_count >= 1

    def test_create_chimeric_response_comprehensive(self, client):
        """Test _create_chimeric_response with all edge cases."""
        # Test normal response
        mock_response = MockCohereResponse()
        tool_calls = [{"name": "test_tool", "result": "success"}]
        chimeric_response = CohereClient._create_chimeric_response(mock_response, tool_calls)

        assert isinstance(chimeric_response, ChimericCompletionResponse)
        assert chimeric_response.common.content == "Hello from Cohere!"
        assert chimeric_response.common.usage.prompt_tokens == 10
        assert chimeric_response.common.usage.completion_tokens == 15
        assert chimeric_response.common.usage.total_tokens == 25
        assert chimeric_response.common.model == "command-a-03-2025"
        assert isinstance(chimeric_response.common.metadata, dict)
        assert "tool_calls" in chimeric_response.common.metadata

        # Test with list content
        mock_response.message.content = [SimpleNamespace(text="List content")]
        chimeric_response = CohereClient._create_chimeric_response(mock_response, [])
        assert chimeric_response.common.content == "List content"

        # Test with string content
        mock_response.message.content = "Direct string content"
        chimeric_response = CohereClient._create_chimeric_response(mock_response, [])
        assert chimeric_response.common.content == "Direct string content"

        # Test with an empty content list
        mock_response.message.content = []
        mock_response.text = "fallback text"
        chimeric_response = CohereClient._create_chimeric_response(mock_response, [])
        assert chimeric_response.common.content == "[]"

        # Test fallback to text property
        mock_response.message = SimpleNamespace()
        chimeric_response = CohereClient._create_chimeric_response(mock_response, [])
        assert chimeric_response.common.content == "fallback text"

        # Test no message at all
        mock_response = SimpleNamespace()
        mock_response.text = "Fallback text"
        mock_response.model = "test-model"
        mock_response.usage = SimpleNamespace(
            tokens=SimpleNamespace(input_tokens=5, output_tokens=10)
        )
        mock_response.model_dump = lambda: {"test": "data"}
        chimeric_response = CohereClient._create_chimeric_response(mock_response, [])
        assert chimeric_response.common.content == "Fallback text"

        # Test no content at all
        mock_response = SimpleNamespace()
        mock_response.usage = SimpleNamespace(
            tokens=SimpleNamespace(input_tokens=1, output_tokens=2)
        )
        mock_response.model = "test-model"
        mock_response.model_dump = lambda: {"test": "data"}
        chimeric_response = CohereClient._create_chimeric_response(mock_response, [])
        assert chimeric_response.common.content == ""

        # Test missing usage
        mock_response = MockCohereResponse()
        delattr(mock_response, "usage")
        chimeric_response = CohereClient._create_chimeric_response(mock_response, [])
        assert chimeric_response.common.usage.prompt_tokens == 0
        assert chimeric_response.common.usage.completion_tokens == 0
        assert chimeric_response.common.usage.total_tokens == 0

        # Test missing tokens structure
        mock_response = MockCohereResponse()
        mock_response.usage = SimpleNamespace(
            billed_units=SimpleNamespace(input_tokens=20, output_tokens=30)
        )
        chimeric_response = CohereClient._create_chimeric_response(mock_response, [])
        assert chimeric_response.common.usage.prompt_tokens == 0
        assert chimeric_response.common.usage.completion_tokens == 0

        # Test no model_dump
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

    def test_encode_tools_comprehensive(self, client):
        """Test _encode_tools with various inputs."""
        # Test with Tool objects
        tool = Tool(
            name="get_weather",
            description="Get weather info",
            parameters=ToolParameters(
                type="object", properties={"location": {"type": "string"}}, required=["location"]
            ),
        )
        encoded = client._encode_tools([tool])
        assert isinstance(encoded, list)
        assert len(encoded) == 1
        assert encoded[0]["type"] == "function"
        assert encoded[0]["function"]["name"] == "get_weather"
        assert encoded[0]["function"]["description"] == "Get weather info"
        assert "properties" in encoded[0]["function"]["parameters"]

        # Test with None parameters
        tool_no_params = Tool(name="simple_tool", description="Simple", function=lambda: "test")
        encoded = client._encode_tools([tool_no_params])
        assert isinstance(encoded, list)
        assert len(encoded) == 1
        assert encoded[0]["function"]["parameters"] == {}

        # Test with dict tools
        dict_tool = {"type": "function", "function": {"name": "test"}}
        encoded_dict = client._encode_tools([dict_tool])
        assert isinstance(encoded_dict, list)
        assert encoded_dict[0] == dict_tool

        # Test with None and empty list
        assert client._encode_tools(None) is None
        assert client._encode_tools([]) is None

    def test_tool_processing_comprehensive(self, client, monkeypatch):
        """Test comprehensive tool processing including errors."""

        # Setup tools
        def mock_weather_tool(location: str) -> str:
            return f"Weather in {location}: sunny"

        def error_tool(arg: str) -> str:
            raise ValueError("Tool execution failed intentionally")

        tool = Tool(name="get_weather", description="Get weather", function=mock_weather_tool)
        error_tool_obj = Tool(name="error_tool", description="Error tool", function=error_tool)
        client.tool_manager.tools["get_weather"] = tool
        client.tool_manager.tools["error_tool"] = error_tool_obj

        # Test successful tool call
        mock_call = MockCohereToolCall()
        result = client._process_function_call(mock_call)
        assert result["call_id"] == "tool_123"
        assert result["name"] == "get_weather"
        assert result["arguments"] == '{"location": "Toronto"}'
        assert result["result"] == "Weather in Toronto: sunny"

        # Test tool with error
        error_call = MockCohereToolCall(tool_name="error_tool", arguments='{"arg": "test"}')
        result = client._process_function_call(error_call)
        assert result["name"] == "error_tool"
        assert "error" in result
        assert result["error"] is True
        assert "Tool execution failed intentionally" in result["result"]

        # Test invalid JSON arguments
        json_error_call = MockCohereToolCall(tool_name="get_weather", arguments='{"invalid": json}')
        result = client._process_function_call(json_error_call)
        assert "error" in result
        assert result["error"] is True
        assert "Error executing tool" in result["result"]

        # Test non-callable tool
        tool_non_callable = SimpleNamespace()
        tool_non_callable.name = "bad_tool"
        tool_non_callable.function = "not_callable"
        client.tool_manager.tools["bad_tool"] = tool_non_callable
        bad_call = MockCohereToolCall(tool_name="bad_tool")
        with pytest.raises(ToolRegistrationError, match="Tool 'bad_tool' is not callable"):
            client._process_function_call(bad_call)

    def test_handle_function_tool_calls_comprehensive(self, client, monkeypatch):
        """Test _handle_function_tool_calls with all edge cases."""

        # Setup tools
        def tool1(arg: str) -> str:
            return f"Tool1 result: {arg}"

        def tool2(arg: str) -> str:
            return f"Tool2 result: {arg}"

        client.tool_manager.register(tool1, name="tool1", description="First tool")
        client.tool_manager.register(tool2, name="tool2", description="Second tool")

        # Test no tool calls
        mock_response = MockCohereResponse()
        messages = [{"role": "user", "content": "Hello"}]
        monkeypatch.setattr(client._client, "chat", mock_response)
        tool_calls, final_response = client._handle_function_tool_calls(
            model=mock_response.model, response=mock_response, messages=messages
        )
        assert tool_calls == []
        assert final_response == mock_response

        # Test response with no message
        mock_response_no_msg = SimpleNamespace()
        tool_calls, final_response = client._handle_function_tool_calls(
            model="command-a-03-2025", response=mock_response_no_msg, messages=messages
        )
        assert tool_calls == []
        assert final_response == mock_response_no_msg

        # Test message with no tool_calls attribute
        mock_response_no_tc = SimpleNamespace()
        mock_response_no_tc.message = SimpleNamespace()
        tool_calls, final_response = client._handle_function_tool_calls(
            model="command-a-03-2025", response=mock_response_no_tc, messages=messages
        )
        assert tool_calls == []
        assert final_response == mock_response_no_tc

        # Test empty tool_calls list
        mock_response_empty = MockCohereResponse()
        mock_response_empty.message.tool_calls = []
        tool_calls, final_response = client._handle_function_tool_calls(
            model=mock_response_empty.model, response=mock_response_empty, messages=messages
        )
        assert tool_calls == []
        assert final_response == mock_response_empty

        # Test with None tool_calls
        mock_response_none = MockCohereResponse()
        mock_response_none.message.tool_calls = None
        tool_calls, final_response = client._handle_function_tool_calls(
            model=mock_response_none.model, response=mock_response_none, messages=messages
        )
        assert tool_calls == []
        assert final_response == mock_response_none

        # Test multiple tool call iterations
        tool_call1 = MockCohereToolCall(
            tool_id="call1", tool_name="tool1", arguments='{"arg": "first"}'
        )
        tool_call2 = MockCohereToolCall(
            tool_id="call2", tool_name="tool2", arguments='{"arg": "second"}'
        )
        response1 = MockCohereResponse(tool_calls=[tool_call1])
        response2 = MockCohereResponse(tool_calls=[tool_call2])
        response_final = MockCohereResponse()

        call_count = 0

        def mock_chat(**kwargs: Any):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return response2
            return response_final

        monkeypatch.setattr(client._client, "chat", mock_chat)
        tool_calls, final_response = client._handle_function_tool_calls(
            model=response1.model,
            response=response1,
            messages=messages,
            tools=client.tool_manager.tools,
        )
        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "tool1"
        assert tool_calls[1]["name"] == "tool2"
        assert final_response == response_final

        # Test single message dict input
        messages_dict = {"role": "user", "content": "What's the weather?"}
        tool_calls, final_response = client._handle_function_tool_calls(
            model=response1.model,
            response=response1,
            messages=messages_dict,
            tools=client.tool_manager.tools,
        )
        assert len(tool_calls) >= 1

    async def test_handle_function_tool_calls_async_comprehensive(self, client, monkeypatch):
        """Test async _handle_function_tool_calls_async with all cases."""

        # Setup tools
        def async_tool1(arg: str) -> str:
            return f"Async Tool1 result: {arg}"

        def async_tool2(arg: str) -> str:
            return f"Async Tool2 result: {arg}"

        client.tool_manager.register(
            async_tool1, name="async_tool1", description="First async tool"
        )
        client.tool_manager.register(
            async_tool2, name="async_tool2", description="Second async tool"
        )

        # Test no tool calls
        mock_response = MockCohereResponse()
        messages = [{"role": "user", "content": "Hello"}]

        async def mock_async_chat(**kwargs: Any):
            return mock_response

        monkeypatch.setattr(client._async_client, "chat", mock_async_chat)
        tool_calls, final_response = await client._handle_function_tool_calls_async(
            model=mock_response.model, response=mock_response, messages=messages
        )
        assert tool_calls == []
        assert final_response == mock_response

        # Test multiple tool call iterations
        tool_call1 = MockCohereToolCall(
            tool_id="async_call1", tool_name="async_tool1", arguments='{"arg": "first"}'
        )
        tool_call2 = MockCohereToolCall(
            tool_id="async_call2", tool_name="async_tool2", arguments='{"arg": "second"}'
        )
        response1 = MockCohereResponse(tool_calls=[tool_call1])
        response2 = MockCohereResponse(tool_calls=[tool_call2])
        response_final = MockCohereResponse()

        call_count = 0

        async def mock_async_chat_multi(**kwargs: Any):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return response2
            return response_final

        monkeypatch.setattr(client._async_client, "chat", mock_async_chat_multi)
        tool_calls, final_response = await client._handle_function_tool_calls_async(
            model=response1.model, response=response1, messages=messages
        )
        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "async_tool1"
        assert tool_calls[1]["name"] == "async_tool2"
        assert final_response == response_final

    def test_message_conversion_comprehensive(self, client):
        """Test message conversion with all edge cases."""
        # Test string
        result = client._convert_messages_to_cohere_format("Hello")
        assert result == [{"role": "user", "content": "Hello"}]

        # Test dict
        result = client._convert_messages_to_cohere_format({"role": "user", "content": "Hi"})
        assert result == [{"role": "user", "content": "Hi"}]

        # Test list
        result = client._convert_messages_to_cohere_format([{"role": "user", "content": "Test"}])
        assert result == [{"role": "user", "content": "Test"}]

        # Test numeric
        result = client._convert_messages_to_cohere_format(123)
        assert result == [{"role": "user", "content": "123"}]

        # Test None
        result = client._convert_messages_to_cohere_format(None)
        assert result == [{"role": "user", "content": "None"}]

        # Test boolean
        result = client._convert_messages_to_cohere_format(False)
        assert result == [{"role": "user", "content": "False"}]

        # Test zero
        result = client._convert_messages_to_cohere_format(0)
        assert result == [{"role": "user", "content": "0"}]

        # Test empty string
        result = client._convert_messages_to_cohere_format("")
        assert result == [{"role": "user", "content": ""}]

        # Test empty list
        result = client._convert_messages_to_cohere_format([])
        assert result == []

        # Test custom object
        class CustomObject:
            def __str__(self):
                return "custom_string"

        result = client._convert_messages_to_cohere_format(CustomObject())
        assert result == [{"role": "user", "content": "custom_string"}]

    def test_chat_completion_comprehensive(self, client, monkeypatch):
        """Test chat completion with all variations."""
        # Mock response
        mock_response = MockCohereResponse()

        # Test basic chat completion
        def mock_chat(**kwargs: Any):
            return mock_response

        monkeypatch.setattr(client._client, "chat", mock_chat)
        result = client._chat_completion_impl(messages="Hello", model="command-a-03-2025")
        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Hello from Cohere!"

        # Test with kwargs filtering
        captured_kwargs = {}

        def mock_chat_capture(**kwargs: Any):
            captured_kwargs.update(kwargs)
            return mock_response

        def mock_filter_kwargs(func, kwargs):
            return {"filtered_param": "value"}

        monkeypatch.setattr(client._client, "chat", mock_chat_capture)
        monkeypatch.setattr(client, "_filter_kwargs", mock_filter_kwargs)

        client._chat_completion_impl(
            messages="Hello",
            model="command-a-03-2025",
            temperature=0.7,
            invalid_param="should_be_filtered",
        )
        assert "filtered_param" in captured_kwargs
        assert captured_kwargs["filtered_param"] == "value"

        # Test streaming
        events = [
            MockCohereStreamEvent("content-delta", "Hello"),
            MockCohereStreamEvent("message-end", ""),
        ]

        def mock_chat_stream(**kwargs: Any):
            return iter(events)

        monkeypatch.setattr(client._client, "chat_stream", mock_chat_stream)
        result = client._chat_completion_impl(
            messages="Hello", model="command-a-03-2025", stream=True
        )
        assert isinstance(result, Generator)
        chunks = list(result)
        assert len(chunks) == 2

        # Test with tools
        def mock_weather_tool(location: str) -> str:
            return f"Weather in {location}: sunny"

        tool = Tool(name="get_weather", description="Get weather", function=mock_weather_tool)
        client.tool_manager.tools["get_weather"] = tool

        mock_tool_call = MockCohereToolCall()
        first_response = MockCohereResponse(
            content="I'll check the weather", tool_calls=[mock_tool_call]
        )
        second_response = MockCohereResponse(content="The weather in Toronto is sunny")

        call_count = 0

        def mock_chat_tools(**kwargs: Any):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return first_response
            return second_response

        monkeypatch.setattr(client._client, "chat", mock_chat_tools)
        result = client._chat_completion_impl(
            messages="What's the weather in Toronto?",
            model="command-a-03-2025",
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )
        assert isinstance(result, ChimericCompletionResponse)
        assert isinstance(result.common.metadata, dict)
        assert "tool_calls" in result.common.metadata
        assert call_count == 2

        # Test request tracking
        initial_requests = client.request_count
        initial_errors = client.error_count
        client.chat_completion(messages="Hello", model="command-a-03-2025")
        assert client.request_count == initial_requests + 1
        assert client.error_count == initial_errors

    async def test_achat_completion_comprehensive(self, client, monkeypatch):
        """Test async chat completion with all variations."""
        # Mock response
        mock_response = MockCohereResponse()

        # Test basic async chat completion
        async def mock_chat(**kwargs: Any):
            return mock_response

        monkeypatch.setattr(client._async_client, "chat", mock_chat)
        result = await client._achat_completion_impl(messages="Hello", model="command-a-03-2025")
        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Hello from Cohere!"

        # Test with kwargs filtering
        captured_kwargs = {}

        async def mock_chat_capture(**kwargs: Any):
            captured_kwargs.update(kwargs)
            return mock_response

        def mock_filter_kwargs(func, kwargs):
            return {"async_filtered": "value"}

        monkeypatch.setattr(client._async_client, "chat", mock_chat_capture)
        monkeypatch.setattr(client, "_filter_kwargs", mock_filter_kwargs)

        await client._achat_completion_impl(
            messages="Hello",
            model="command-a-03-2025",
            temperature=0.7,
            invalid_param="should_be_filtered",
        )
        assert "async_filtered" in captured_kwargs
        assert captured_kwargs["async_filtered"] == "value"

        # Test async streaming
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

        def mock_chat_stream(**kwargs: Any):
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

        # Test with tools
        def mock_weather_tool(location: str) -> str:
            return f"Weather in {location}: sunny"

        tool = Tool(name="get_weather", description="Get weather", function=mock_weather_tool)
        client.tool_manager.tools["get_weather"] = tool

        mock_tool_call = MockCohereToolCall()
        first_response = MockCohereResponse(
            content="I'll check the weather", tool_calls=[mock_tool_call]
        )
        second_response = MockCohereResponse(content="The weather in Toronto is sunny")

        call_count = 0

        async def mock_chat_tools(**kwargs: Any):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return first_response
            return second_response

        monkeypatch.setattr(client._async_client, "chat", mock_chat_tools)
        result = await client._achat_completion_impl(
            messages="What's the weather in Toronto?",
            model="command-a-03-2025",
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )
        assert isinstance(result, ChimericCompletionResponse)
        assert isinstance(result.common.metadata, dict)
        assert "tool_calls" in result.common.metadata
        assert call_count == 2

    def test_upload_file_not_supported(self, client):
        """Test that file upload raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Cohere does not support file uploads"):
            client._upload_file(file="test.txt")
