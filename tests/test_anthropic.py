from collections.abc import AsyncGenerator, Generator
import os
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

from chimeric import Chimeric
from chimeric.exceptions import ToolRegistrationError
import chimeric.providers.anthropic.client as client_module
from chimeric.providers.anthropic.client import AnthropicClient
from chimeric.types import (
    ChimericCompletionResponse,
    ChimericStreamChunk,
    FileUploadResponse,
    ModelSummary,
    StreamChunk,
    Tool,
    ToolParameters,
)


@pytest.fixture(scope="module")
def anthropic_env():
    """Ensure ANTHROPIC_API_KEY is set for Chimeric initialization."""
    os.environ["ANTHROPIC_API_KEY"] = "test_key"
    yield
    del os.environ["ANTHROPIC_API_KEY"]


@pytest.fixture(scope="module")
def chimeric_anthropic(anthropic_env):
    """Create a Chimeric instance configured for Anthropic."""
    return Chimeric(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "test_key"),
        base_url="https://api.anthropic.com",
        timeout=120,
        max_retries=2,
    )


@pytest.fixture(scope="module")
def client(chimeric_anthropic) -> AnthropicClient:
    """Get the AnthropicClient from the Chimeric wrapper."""
    return cast("AnthropicClient", chimeric_anthropic.get_provider_client("anthropic"))


@pytest.fixture(autouse=True)
def patch_anthropic_imports(monkeypatch):
    """Stub out actual Anthropic classes to prevent network calls."""

    def create_anthropic_mock(api_key: str, **kw: Any) -> SimpleNamespace:
        return SimpleNamespace(api_key=api_key, **kw)

    def create_async_anthropic_mock(api_key: str, **kw: Any) -> SimpleNamespace:
        return SimpleNamespace(api_key=api_key, **kw)

    monkeypatch.setattr(client_module, "Anthropic", create_anthropic_mock)
    monkeypatch.setattr(client_module, "AsyncAnthropic", create_async_anthropic_mock)

    # Create stub Stream types for isinstance checks
    class MockStreamType:
        pass

    class MockAsyncStreamType:
        pass

    monkeypatch.setattr(client_module, "Stream", MockStreamType)
    monkeypatch.setattr(client_module, "AsyncStream", MockAsyncStreamType)


class MockResponse:
    """Mock Anthropic Message response."""

    def __init__(
        self, content="Hello from Claude!", model="claude-3-sonnet", usage_tokens=(10, 15)
    ):
        self.content = [SimpleNamespace(text=content, type="text")]
        self.usage = SimpleNamespace(input_tokens=usage_tokens[0], output_tokens=usage_tokens[1])
        self.model = model

    def model_dump(self) -> dict[str, Any]:
        return {"response_data": "anthropic_test", "model": self.model}


class MockStreamEvent:
    """Mock Anthropic streaming event."""

    def __init__(self, event_type="content_block_delta", delta_text=""):
        self.type = event_type
        if event_type == "content_block_delta":
            self.delta = SimpleNamespace(text=delta_text, type="text")

    def model_dump(self) -> dict[str, Any]:
        return {"event_type": self.type, "stream_data": True}


class MockFile:
    """Mock Anthropic file upload response."""

    def __init__(self):
        self.id = "file-anthro-123"
        self.filename = "test_file.txt"
        self.size_bytes = 456
        self.created_at = 789

    def model_dump(self) -> dict[str, Any]:
        return {"file_metadata": "anthropic"}


# noinspection PyUnusedLocal
class TestAnthropicClient:
    """Tests for the AnthropicClient class."""

    def test_capabilities(self, client):
        """Test that client reports correct capabilities."""
        caps = client.capabilities

        assert caps.multimodal
        assert caps.streaming
        assert caps.tools
        assert not caps.agents
        assert caps.files

        # Test convenience methods
        assert client.supports_multimodal()
        assert client.supports_streaming()
        assert client.supports_tools()
        assert not client.supports_agents()
        assert client.supports_files()

    def test_list_models(self, client, monkeypatch):
        """Test list_models maps to ModelSummary objects correctly."""

        class MockModel:
            def __init__(self, id, display_name=None, created_at=None):
                self.id = id
                self.display_name = display_name
                self.created_at = created_at

            def model_dump(self):
                return {"id": self.id, "display_name": self.display_name}

        def mock_list():
            return SimpleNamespace(
                data=[
                    MockModel("claude-3-opus", "Claude 3 Opus", 1234567890),
                    MockModel("claude-3-sonnet", created_at=1234567891),
                ]
            )

        monkeypatch.setattr(client.client.models, "list", mock_list)
        models = client.list_models()

        assert len(models) == 2
        assert isinstance(models[0], ModelSummary)
        assert models[0].id == "claude-3-opus"
        assert models[0].name == "Claude 3 Opus"
        assert models[1].name == "claude-3-sonnet"  # Falls back to id

    @pytest.mark.parametrize(
        ("event_type", "delta_text", "expected_content", "expected_finish"),
        [
            ("content_block_delta", " world", "Hello world", None),
            ("message_stop", "", "Hello", "end_turn"),
            ("unknown_event", "", "Hello", None),
        ],
    )
    def test_process_event(self, event_type, delta_text, expected_content, expected_finish):
        """Test _process_event with different event types."""
        accumulated = "Hello"
        event = MockStreamEvent(event_type, delta_text)

        new_accumulated, chunk = AnthropicClient._process_event(event, accumulated)  # type: ignore[arg-type]

        if event_type == "unknown_event":
            assert chunk is None
            assert new_accumulated == "Hello"
        else:
            assert new_accumulated == expected_content
            assert chunk.common.content == expected_content
            assert chunk.common.finish_reason == expected_finish

    def test_stream_processing(self, client):
        """Test stream filtering and processing."""
        events = [
            MockStreamEvent("content_block_delta", "Hello"),
            MockStreamEvent("unknown_event"),
            MockStreamEvent("message_stop"),
        ]

        chunks = list(client._stream(events))
        assert len(chunks) == 2  # Unknown event filtered out
        assert chunks[0].common.content == "Hello"
        assert chunks[1].common.finish_reason == "end_turn"

    async def test_astream_processing(self, client):
        """Test async stream filtering and processing."""

        async def agen():
            for event in [
                MockStreamEvent("content_block_delta", "Async"),
                MockStreamEvent("unknown_event"),
                MockStreamEvent("message_stop"),
            ]:
                yield event

        chunks = []
        async for chunk in client._astream(agen()):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].common.content == "Async"
        assert chunks[1].common.finish_reason == "end_turn"

    def test_create_chimeric_response(self, client):
        """Test response creation with and without tool calls."""
        response = MockResponse("Test response")

        # Without tool calls
        result = client._create_chimeric_response(response, [])
        assert result.common.content == "Test response"
        assert "tool_calls" not in result.common.metadata

        # With tool calls
        tool_calls = [{"call_id": "call_1", "name": "test_tool", "result": "result"}]
        result = client._create_chimeric_response(response, tool_calls)
        assert result.common.metadata["tool_calls"] == tool_calls

    def test_encode_tools(self, client):
        """Test tool encoding for different input types."""
        # None input
        assert client._encode_tools(None) is None

        # Tool object
        tool_params = ToolParameters(properties={"query": {"type": "string"}}, required=["query"])
        tool = Tool(name="search", description="Search tool", parameters=tool_params)

        encoded = client._encode_tools([tool])
        assert len(encoded) == 1
        assert encoded[0]["name"] == "search"
        assert encoded[0]["input_schema"] == tool_params.model_dump()

        # Dict object
        tool_dict = {"name": "dict_tool", "description": "Dict tool"}
        encoded = client._encode_tools([tool_dict])
        assert encoded[0] == tool_dict

    def test_function_call_non_callable_tool(self, client, monkeypatch):
        """Test function call with non-callable tool raises ToolRegistrationError."""

        # Create a mock tool that has a non-callable function
        mock_tool = Mock()
        mock_tool.function = "not_callable_string"  # String instead of function

        def mock_get_tool(name):
            return mock_tool

        monkeypatch.setattr(client.tool_manager, "get_tool", mock_get_tool)

        tool_use_block = SimpleNamespace(id="call_123", name="non_callable_tool", input={})

        with pytest.raises(ToolRegistrationError) as exc_info:
            client._process_function_call(tool_use_block)
        assert "Tool 'non_callable_tool' is not callable" in str(exc_info.value)

    def test_handle_function_tool_calls_with_tool_use_blocks(self, client):
        """Test _handle_function_tool_calls with tool_use blocks to cover the elif branch."""

        # Register a test tool
        def mock_calculator(operation: str, a: int, b: int) -> str:
            if operation == "add":
                return str(a + b)
            return "unknown operation"

        tool = Tool(name="calculator", description="Calculate", function=mock_calculator)
        client.tool_manager.register(
            func=tool.function, name=tool.name, description=tool.description
        )

        # Create response with mixed content including tool_use
        response = MockResponse("I'll help you calculate")
        response.content = [
            SimpleNamespace(type="text", text="I'll help you calculate."),
            SimpleNamespace(
                type="tool_use",
                id="call_calc",
                name="calculator",
                input={"operation": "add", "a": 10, "b": 5},
            ),
        ]

        messages = [{"role": "user", "content": "Calculate 10 + 5"}]

        tool_calls, updated_messages = client._handle_function_tool_calls(response, messages)

        # Verify tool call was processed
        assert len(tool_calls) == 1
        assert tool_calls[0]["result"] == "15"

        # Verify messages were updated correctly
        assert len(updated_messages) == 3  # original + assistant + user with tool result

        # Check assistant message structure (this tests the elif branch)
        assistant_msg = updated_messages[1]
        assert assistant_msg["role"] == "assistant"
        assert len(assistant_msg["content"]) == 2

        # Check text content
        text_content = assistant_msg["content"][0]
        assert text_content["type"] == "text"
        assert text_content["text"] == "I'll help you calculate."

        # Check tool_use content (this tests the elif block_type == "tool_use" branch - line 317)
        tool_use_content = assistant_msg["content"][1]
        assert tool_use_content["type"] == "tool_use"
        assert tool_use_content["id"] == "call_calc"
        assert tool_use_content["name"] == "calculator"
        assert tool_use_content["input"] == {"operation": "add", "a": 10, "b": 5}

        # Check tool result message
        tool_result_msg = updated_messages[2]
        assert tool_result_msg["role"] == "user"
        assert len(tool_result_msg["content"]) == 1
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "call_calc"
        assert tool_result_msg["content"][0]["content"] == "15"

    def test_handle_function_tool_calls_error_handling(self, client):
        """Test tool call error handling when tool execution fails."""

        # Register a tool that will raise an exception
        def failing_tool(value: str) -> str:
            raise ValueError(f"Tool failed with value: {value}")

        tool = Tool(name="failing_tool", description="Fails", function=failing_tool)
        client.tool_manager.register(
            func=tool.function, name=tool.name, description=tool.description
        )

        # Create response with tool use that will fail
        response = MockResponse("Using failing tool")
        response.content = [
            SimpleNamespace(
                type="tool_use", id="call_fail", name="failing_tool", input={"value": "test"}
            )
        ]

        messages = [{"role": "user", "content": "Use the failing tool"}]

        tool_calls, _ = client._handle_function_tool_calls(response, messages)

        # Verify error was handled gracefully
        assert len(tool_calls) == 1
        assert "error" in tool_calls[0]
        assert tool_calls[0]["error"] is True
        assert "Error executing tool 'failing_tool'" in tool_calls[0]["result"]
        assert "Tool failed with value: test" in tool_calls[0]["result"]

    def test_handle_function_tool_calls_unknown_content_types(self, client):
        """Test handling of unknown content block types in tool call processing."""

        # Register a test tool
        def mock_calculator(a: int, b: int) -> int:
            return a + b

        tool = Tool(name="add", description="Add numbers", function=mock_calculator)
        client.tool_manager.register(
            func=tool.function, name=tool.name, description=tool.description
        )

        # Create response with mixed content including unknown types
        response = MockResponse("Mixed content")
        response.content = [
            SimpleNamespace(type="text", text="I'll help with that."),
            SimpleNamespace(type="image", data="base64data"),  # Unknown type - should be skipped
            SimpleNamespace(type="tool_use", id="call_add", name="add", input={"a": 5, "b": 3}),
            SimpleNamespace(
                type="audio", url="audio.mp3"
            ),  # Another unknown type - should be skipped
        ]

        messages = [{"role": "user", "content": "Calculate 5 + 3"}]

        tool_calls, updated_messages = client._handle_function_tool_calls(response, messages)

        # Verify tool call was processed
        assert len(tool_calls) == 1
        assert tool_calls[0]["result"] == "8"

        # Verify assistant message only includes text and tool_use content (unknown types skipped)
        assistant_msg = updated_messages[1]
        assert assistant_msg["role"] == "assistant"
        assert len(assistant_msg["content"]) == 2  # Only text and tool_use, image and audio skipped

        # Check that content is properly structured
        content_types = [content["type"] for content in assistant_msg["content"]]
        assert content_types == ["text", "tool_use"]

    def test_function_call_processing_comprehensive(self, client, monkeypatch):
        """Test function call processing including tool not found error handling."""

        # First register a test tool for the successful case
        def mock_weather(location: str) -> str:
            return f"Weather in {location}: Sunny"

        tool = Tool(name="get_weather", description="Get weather", function=mock_weather)
        client.tool_manager.register(
            func=tool.function, name=tool.name, description=tool.description
        )

        # Test successful execution
        tool_use_block = SimpleNamespace(
            id="call_123", name="get_weather", input={"location": "Paris"}
        )

        result = client._process_function_call(tool_use_block)
        assert result["call_id"] == "call_123"
        assert result["result"] == "Weather in Paris: Sunny"
        assert "error" not in result

        # Test tool not registered - raises ToolRegistrationError
        unknown_block = SimpleNamespace(id="call_456", name="unknown_tool", input={})

        with pytest.raises(ToolRegistrationError) as exc_info:
            client._process_function_call(unknown_block)
        assert "No tool registered with name 'unknown_tool'" in str(exc_info.value)

    def test_function_call_execution_errors(self, client):
        """Test function call execution with various error scenarios."""

        # Register a test tool
        def mock_weather(location: str) -> str:
            return f"Weather in {location}: Sunny"

        tool = Tool(name="test_weather", description="Get weather", function=mock_weather)
        client.tool_manager.register(
            func=tool.function, name=tool.name, description=tool.description
        )

        # Test successful execution
        tool_use_block = SimpleNamespace(
            id="call_123", name="test_weather", input={"location": "Paris"}
        )

        result = client._process_function_call(tool_use_block)
        assert result["call_id"] == "call_123"
        assert result["result"] == "Weather in Paris: Sunny"
        assert "error" not in result

        # Test tool not registered - raises ToolRegistrationError
        unknown_block = SimpleNamespace(id="call_456", name="unknown_tool", input={})
        with pytest.raises(ToolRegistrationError) as exc_info:
            client._process_function_call(unknown_block)
        assert "No tool registered with name 'unknown_tool'" in str(exc_info.value)

    def test_handle_function_tool_calls(self, client):
        """Test tool call handling and message updating."""

        # Register tool
        def mock_calc(a: int, b: int) -> int:
            return a + b

        tool = Tool(name="add_numbers", description="Add numbers", function=mock_calc)
        client.tool_manager.register(
            func=tool.function, name=tool.name, description=tool.description
        )

        # No tool calls
        response = MockResponse("No tools")
        response.content = [SimpleNamespace(type="text", text="Regular response")]
        messages = [{"role": "user", "content": "Hello"}]

        tool_calls, updated = client._handle_function_tool_calls(response, messages)
        assert tool_calls == []
        assert updated == messages

        # With tool calls
        response.content = [
            SimpleNamespace(type="text", text="I'll calculate that."),
            SimpleNamespace(
                type="tool_use", id="call_add", name="add_numbers", input={"a": 5, "b": 3}
            ),
        ]

        tool_calls, updated = client._handle_function_tool_calls(response, messages)
        assert len(tool_calls) == 1
        assert tool_calls[0]["result"] == "8"
        assert len(updated) == 3  # original + assistant + user with tool result

    @pytest.mark.parametrize(
        ("input_messages", "expected_formatted", "expected_system"),
        [
            ("Hello", [{"role": "user", "content": "Hello"}], None),
            (42, [{"role": "user", "content": "42"}], None),
            (
                [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hi"},
                ],
                [{"role": "user", "content": "Hi"}],
                "You are helpful",
            ),
            (
                ["string message", {"role": "user", "content": "dict message"}],
                [
                    {"role": "user", "content": "string message"},
                    {"role": "user", "content": "dict message"},
                ],
                None,
            ),
        ],
    )
    def test_format_messages(self, client, input_messages, expected_formatted, expected_system):
        """Test message formatting for different input types."""
        formatted, system = client._format_messages(input_messages)
        assert formatted == expected_formatted
        assert system == expected_system

    def test_chat_completion_impl_basic(self, client, monkeypatch):
        """Test basic chat completion implementation."""
        mock_response = MockResponse("Hello from Anthropic")

        captured_params = {}

        def mock_create(**kwargs):
            captured_params.update(kwargs)
            return mock_response

        client._client.messages = SimpleNamespace(create=mock_create)

        def mock_filter_kwargs(func, kwargs):
            return kwargs

        monkeypatch.setattr(client, "_filter_kwargs", mock_filter_kwargs)

        # Test basic call without system prompt
        result = client._chat_completion_impl(
            messages=[{"role": "user", "content": "Hello"}], model="claude-3-sonnet"
        )

        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Hello from Anthropic"
        assert "system" not in captured_params

        # Test with system prompt - branch coverage
        captured_params.clear()
        messages_with_system = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

        client._chat_completion_impl(messages=messages_with_system, model="claude-3-sonnet")

        assert captured_params["system"] == "You are helpful"
        assert len(captured_params["messages"]) == 1  # System message extracted

        # Test with tools - branch coverage
        captured_params.clear()
        tools = [{"name": "test_tool", "description": "Test tool"}]

        client._chat_completion_impl(
            messages=[{"role": "user", "content": "Use tool"}], model="claude-3-sonnet", tools=tools
        )

        assert captured_params["tools"] == tools

        # Test with optional parameters - branch coverage
        captured_params.clear()
        client._chat_completion_impl(
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-3-sonnet",
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            stop_sequences=["STOP"],
            metadata={"test": "value"},
            unsupported_param="ignored",  # This should be filtered out
        )

        assert captured_params["temperature"] == 0.8
        assert captured_params["top_p"] == 0.9
        assert captured_params["top_k"] == 50
        assert captured_params["stop_sequences"] == ["STOP"]
        assert captured_params["metadata"] == {"test": "value"}
        assert "unsupported_param" not in captured_params

    def test_chat_completion_impl_streaming(self, client, monkeypatch):
        """Test streaming chat completion."""
        mock_stream = Mock(spec=client_module.Stream)

        def mock_create(**kwargs):
            return mock_stream

        client._client.messages = SimpleNamespace(create=mock_create)

        def mock_stream_method(stream):
            yield ChimericStreamChunk(
                native=Mock(), common=StreamChunk(content="Hello", delta="Hello", metadata={})
            )

        monkeypatch.setattr(client, "_stream", mock_stream_method)

        result = client._chat_completion_impl(
            messages=[{"role": "user", "content": "Hello"}], model="claude-3-sonnet", stream=True
        )

        assert isinstance(result, Generator)
        chunks = list(result)
        assert len(chunks) == 1

    def test_chat_completion_impl_with_tool_calls(self, client, monkeypatch):
        """Test two-pass tool calling mechanism."""

        def mock_search(query: str) -> str:
            return f"Results for: {query}"

        tool = Tool(name="search_web", description="Search web", function=mock_search)
        client.tool_manager.register(
            func=tool.function, name=tool.name, description=tool.description
        )

        # First response with tool call, second with final answer
        first_response = MockResponse("Tool call")
        first_response.content = [
            SimpleNamespace(
                type="tool_use", id="call_1", name="search_web", input={"query": "test"}
            )
        ]
        second_response = MockResponse("Final answer")

        responses = [first_response, second_response]
        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            result = responses[call_count]
            call_count += 1
            return result

        client._client.messages = SimpleNamespace(create=mock_create)

        def mock_filter_kwargs(func, kwargs):
            return kwargs

        monkeypatch.setattr(client, "_filter_kwargs", mock_filter_kwargs)

        result = client._chat_completion_impl(
            messages=[{"role": "user", "content": "Search for test"}], model="claude-3-sonnet"
        )

        assert "tool_calls" in result.common.metadata
        assert result.common.metadata["tool_calls"][0]["result"] == "Results for: test"
        assert call_count == 2

    def test_chat_completion_impl_exception_handling(self, client, monkeypatch):
        """Test that exceptions are not caught and are allowed to propagate to base class."""

        def mock_create(**kwargs):
            error = Exception("API Error")
            error.status_code = 429
            raise error

        client._client.messages = SimpleNamespace(create=mock_create)

        def mock_filter_kwargs(func, kwargs):
            return kwargs

        monkeypatch.setattr(client, "_filter_kwargs", mock_filter_kwargs)

        # Exception should propagate to base class (not caught here)
        with pytest.raises(Exception) as exc_info:
            client._chat_completion_impl(
                messages=[{"role": "user", "content": "Hello"}], model="claude-3-sonnet"
            )

        assert "API Error" in str(exc_info.value)
        assert hasattr(exc_info.value, "status_code")
        assert exc_info.value.status_code == 429

    async def test_achat_completion_impl_basic(self, client, monkeypatch):
        """Test basic async chat completion."""
        mock_response = MockResponse("Async hello")

        captured_params = {}

        async def mock_create(**kwargs):
            captured_params.update(kwargs)
            return mock_response

        client._async_client.messages = SimpleNamespace(create=mock_create)

        def mock_filter_kwargs(func, kwargs):
            return kwargs

        monkeypatch.setattr(client, "_filter_kwargs", mock_filter_kwargs)

        # Test basic async call
        result = await client._achat_completion_impl(
            messages=[{"role": "user", "content": "Hello"}], model="claude-3-sonnet"
        )

        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Async hello"

    async def test_achat_completion_impl_with_tool_calls(self, client, monkeypatch):
        """Test async chat completion with tool calls requiring second API call."""

        def mock_translate(text: str) -> str:
            return f"Translated: {text}"

        tool = Tool(name="translate", description="Translate", function=mock_translate)
        client.tool_manager.register(
            func=tool.function, name=tool.name, description=tool.description
        )

        # First response with tool call
        first_response = MockResponse("Tool call")
        first_response.content = [
            SimpleNamespace(type="tool_use", id="call_1", name="translate", input={"text": "hello"})
        ]

        # Second response after tool execution
        second_response = MockResponse("Translation complete")

        responses = [first_response, second_response]
        call_count = 0
        captured_params_list = []

        async def mock_create(**kwargs):
            nonlocal call_count
            captured_params_list.append(kwargs)
            result = responses[call_count]
            call_count += 1
            return result

        client._async_client.messages = SimpleNamespace(create=mock_create)

        def mock_filter_kwargs(func, kwargs):
            return kwargs

        monkeypatch.setattr(client, "_filter_kwargs", mock_filter_kwargs)

        result = await client._achat_completion_impl(
            messages=[{"role": "user", "content": "Translate hello"}], model="claude-3-sonnet"
        )

        # Verify tool call metadata is included
        assert "tool_calls" in result.common.metadata
        assert len(result.common.metadata["tool_calls"]) == 1
        assert result.common.metadata["tool_calls"][0]["result"] == "Translated: hello"
        assert call_count == 2

        # Verify that the second call has stream=False (branch coverage)
        assert captured_params_list[1]["stream"] is False

    async def test_achat_completion_impl_exception_handling(self, client, monkeypatch):
        """Test that async exceptions are not caught and propagate to base class."""

        async def mock_create(**kwargs):
            error = RuntimeError("Async error")
            error.status_code = 500
            raise error

        client._async_client.messages = SimpleNamespace(create=mock_create)

        def mock_filter_kwargs(func, kwargs):
            return kwargs

        monkeypatch.setattr(client, "_filter_kwargs", mock_filter_kwargs)

        # Exception should propagate to base class (not caught here)
        with pytest.raises(RuntimeError) as exc_info:
            await client._achat_completion_impl(
                messages=[{"role": "user", "content": "Hello"}], model="claude-3-sonnet"
            )

        assert "Async error" in str(exc_info.value)
        assert hasattr(exc_info.value, "status_code")
        assert exc_info.value.status_code == 500

    async def test_achat_completion_impl_streaming(self, client, monkeypatch):
        """Test async streaming chat completion."""
        mock_stream = Mock(spec=client_module.AsyncStream)

        async def mock_create(**kwargs):
            return mock_stream

        client._async_client.messages = SimpleNamespace(create=mock_create)

        async def mock_astream_method(stream):
            yield ChimericStreamChunk(
                native=Mock(), common=StreamChunk(content="Async", delta="Async", metadata={})
            )

        monkeypatch.setattr(client, "_astream", mock_astream_method)

        result = await client._achat_completion_impl(
            messages=[{"role": "user", "content": "Hello"}], model="claude-3-sonnet", stream=True
        )

        assert isinstance(result, AsyncGenerator)
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        assert len(chunks) == 1

    def test_upload_file_success(self, client):
        """Test successful file upload."""
        mock_file = MockFile()

        def mock_upload(**kwargs):
            return mock_file

        client._client.beta = SimpleNamespace(files=SimpleNamespace(upload=mock_upload))

        result = client._upload_file(file="test_file_content")

        assert isinstance(result.common, FileUploadResponse)
        assert result.common.file_id == "file-anthro-123"
        assert result.common.filename == "test_file.txt"

    def test_upload_file_error_handling(self, client):
        """Test file upload error handling."""
        # Missing file parameter - should raise ValueError
        with pytest.raises(ValueError, match="'file' parameter is required"):
            client._upload_file(purpose="test")

        # API error - should propagate to base class, not caught here
        def mock_upload(**kwargs):
            error = Exception("Upload failed")
            error.status_code = 413
            raise error

        client._client.beta = SimpleNamespace(files=SimpleNamespace(upload=mock_upload))

        with pytest.raises(Exception) as exc_info:
            client._upload_file(file="large_file")

        assert "Upload failed" in str(exc_info.value)
        assert hasattr(exc_info.value, "status_code")
        assert exc_info.value.status_code == 413

    def test_request_and_error_tracking(self, client, monkeypatch):
        """Test that requests and errors are properly tracked."""

        # Successful request
        def mock_impl(*args, **kwargs):
            return "SUCCESS"

        monkeypatch.setattr(client, "_chat_completion_impl", mock_impl)

        before_req = client.request_count
        before_err = client.error_count

        client.chat_completion("Hello", "claude-3-sonnet")

        assert client.request_count == before_req + 1
        assert client.error_count == before_err

        # Failed request
        def error_impl(*args, **kwargs):
            raise RuntimeError("Request failed")

        monkeypatch.setattr(client, "_chat_completion_impl", error_impl)

        with pytest.raises(RuntimeError):
            client.chat_completion("Hello", "claude-3-sonnet")

        assert client.request_count == before_req + 2
        assert client.error_count == before_err + 1

    def test_get_model_info(self, client, monkeypatch):
        """Test model info retrieval."""
        models = [
            ModelSummary(
                id="claude-3-opus", name="Claude 3 Opus", created_at=123, owned_by="anthropic"
            ),
            ModelSummary(
                id="claude-3-sonnet", name="Claude 3 Sonnet", created_at=456, owned_by="anthropic"
            ),
        ]

        def mock_list_models():
            return models

        monkeypatch.setattr(client, "list_models", mock_list_models)

        # Found model
        info = client.get_model_info("claude-3-opus")
        assert info.id == "claude-3-opus"

        # Not found
        with pytest.raises(ValueError, match="Model nonexistent-model not found"):
            client.get_model_info("nonexistent-model")

    def test_string_representations(self, client):
        """Test __repr__ and __str__ methods."""
        r = repr(client)
        assert "AnthropicClient" in r
        assert "requests=" in r
        assert "errors=" in r

        s = str(client)
        assert "AnthropicClient Client" in s
        assert "- Requests:" in s
        assert "- Errors:" in s
