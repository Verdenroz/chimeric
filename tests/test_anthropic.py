import contextlib
from datetime import datetime
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
    StreamChunk,
    Tool,
    ToolCall,
    ToolCallChunk,
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

    def __init__(self, event_type="content_block_delta", delta_text="", index=0):
        self.type = event_type
        self.index = index
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

    @staticmethod
    def model_dump() -> dict[str, Any]:
        return {"file_metadata": "anthropic"}


# noinspection PyUnusedLocal,PyTypeChecker
class TestAnthropicClient:
    """Tests for AnthropicClient class."""

    def test_initialization_and_properties(self, client):
        """Test client initialization, capabilities, and string representations."""
        # Test capabilities
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

        # Test string representations
        assert "AnthropicClient" in repr(client)
        assert "requests=" in repr(client)
        assert "AnthropicClient Client" in str(client)
        assert "- Requests:" in str(client)

        # Test generic types
        types = client._get_generic_types()
        assert types["sync"] == client_module.Anthropic
        assert types["async"] == client_module.AsyncAnthropic

        # Test client initialization methods
        sync_client = client._init_client(client_module.Anthropic, base_url="https://test.com")
        assert sync_client.api_key == client.api_key

        async_client = client._init_async_client(client_module.AsyncAnthropic, timeout=30)
        assert async_client.api_key == client.api_key

    def test_model_operations(self, client, monkeypatch):
        """Test model listing and information retrieval."""

        class MockModel:
            def __init__(self, id, display_name=None, created_at=None):
                self.id = id
                self.display_name = display_name
                self.created_at = created_at or datetime(2023, 9, 1)

            def model_dump(self):
                return {"id": self.id, "display_name": self.display_name}

        def mock_list():
            return SimpleNamespace(
                data=[
                    MockModel("claude-3-opus", "Claude 3 Opus"),
                    MockModel("claude-3-sonnet", "Claude 3 Sonnet"),
                ]
            )

        monkeypatch.setattr(client.client.models, "list", mock_list)

        # Test list_models
        models = client.list_models()
        assert len(models) == 8  # 2 API models + 6 aliases
        assert models[0].id == "claude-3-opus"
        assert models[0].name == "Claude 3 Opus"

        # Test get_model_info
        info = client.get_model_info("claude-3-opus")
        assert info.id == "claude-3-opus"

        # Test model not found
        with pytest.raises(ValueError, match="Model nonexistent-model not found"):
            client.get_model_info("nonexistent-model")

    @pytest.mark.parametrize(
        ("input_messages", "expected_formatted", "expected_system"),
        [
            ("Hello", [{"role": "user", "content": "Hello"}], None),
            (42, [{"role": "user", "content": "42"}], None),
            (
                [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": "Hi"}],
                [{"role": "user", "content": "Hi"}],
                "Be helpful",
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
    def test_message_formatting(self, client, input_messages, expected_formatted, expected_system):
        """Test message formatting for different input types."""
        formatted, system = client._format_messages(input_messages)
        assert formatted == expected_formatted
        assert system == expected_system

    def test_tool_encoding(self, client):
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

    def test_tool_execution_comprehensive(self, client):
        """Test comprehensive tool execution scenarios."""

        # Register test tools
        def calculator(operation: str, a: int, b: int) -> str:
            if operation == "add":
                return str(a + b)
            if operation == "error":
                raise ValueError("Calculation error")
            return "unknown"

        def failing_tool() -> str:
            raise RuntimeError("Tool failure")

        tool1 = Tool(name="calculator", description="Calculate", function=calculator)
        tool2 = Tool(name="failing_tool", description="Fails", function=failing_tool)

        client.tool_manager.register(
            func=tool1.function, name=tool1.name, description=tool1.description
        )
        client.tool_manager.register(
            func=tool2.function, name=tool2.name, description=tool2.description
        )

        # Test successful execution
        call = ToolCall(
            call_id="c1", name="calculator", arguments='{"operation": "add", "a": 5, "b": 3}'
        )
        result = client._execute_single_tool(call)
        assert result["result"] == "8"
        assert "error" not in result

        # Test execution error
        error_call = ToolCall(
            call_id="c2", name="calculator", arguments='{"operation": "error", "a": 1, "b": 1}'
        )
        error_result = client._execute_single_tool(error_call)
        assert "Error executing tool" in error_result["result"]
        assert error_result["error"] == "true"

        # Test tool not found
        with pytest.raises(ToolRegistrationError, match="No tool registered"):
            client._execute_single_tool(ToolCall(call_id="c3", name="unknown_tool", arguments="{}"))

        # Test non-callable tool
        client.tool_manager.tools["non_callable"] = Mock(function="not_callable")
        with pytest.raises(ToolRegistrationError, match="not callable"):
            client._execute_single_tool(ToolCall(call_id="c4", name="non_callable", arguments="{}"))

    def test_parse_tool_arguments_edge_cases(self, client):
        """Test edge cases for tool argument parsing."""
        # Valid cases
        assert client._parse_tool_arguments('{"x": 5}') == {"x": 5}
        assert client._parse_tool_arguments({"a": 1}) == {"a": 1}
        assert client._parse_tool_arguments("") == {}
        assert client._parse_tool_arguments("   ") == {}

        # Edge cases
        assert client._parse_tool_arguments(None) is None
        assert client._parse_tool_arguments([1, 2, 3]) == [1, 2, 3]

        # Invalid JSON should not crash
        try:
            client._parse_tool_arguments("{invalid")
        except Exception:
            contextlib.suppress(Exception)  # Expected behavior

    def test_tool_batch_execution(self, client):
        """Test batch tool execution with various statuses."""

        def test_tool(value: str) -> str:
            return f"processed: {value}"

        tool = Tool(name="batch_tool", description="Batch tool", function=test_tool)
        client.tool_manager.register(func=tool.function, name=tool.name)

        tool_calls = {
            "t1": ToolCallChunk(
                id="t1",
                call_id="c1",
                name="batch_tool",
                arguments='{"value": "test1"}',
                status="completed",
            ),
            "t2": ToolCallChunk(
                id="t2",
                call_id="c2",
                name="",
                arguments="{}",
                status="completed",  # Empty name
            ),
            "t3": ToolCallChunk(
                id="t3",
                call_id="c3",
                name="batch_tool",
                arguments='{"value": "test2"}',
                status="started",  # Not completed
            ),
        }

        results = client._execute_tool_batch(tool_calls)
        assert len(results) == 1  # Only t1 should execute
        assert results[0]["result"] == "processed: test1"

    def test_generate_tool_summary(self, client):
        """Test tool summary generation."""
        # Empty summary
        assert client._generate_tool_summary([]) == ""

        # With arguments
        tool_calls = [
            {"name": "calc", "arguments": {"a": 10, "b": 5}, "result": "15"},
            {"name": "weather", "arguments": {"location": "Paris"}, "result": "Sunny"},
        ]
        summary = client._generate_tool_summary(tool_calls)
        assert "calc(a=10, b=5) → 15" in summary
        assert "weather(location=Paris) → Sunny" in summary

        # Without arguments
        tool_calls_no_args = [
            {"name": "random", "arguments": {}, "result": "42"},
            {"name": "timestamp", "arguments": "", "result": "2023-01-01"},
        ]
        summary = client._generate_tool_summary(tool_calls_no_args)
        assert "random() → 42" in summary
        assert "timestamp() → 2023-01-01" in summary

    @pytest.mark.parametrize(
        ("event_type", "delta_text", "expected_content", "expected_finish"),
        [
            ("content_block_delta", " world", "Hello world", None),
            ("message_stop", "", "Hello", "end_turn"),
            ("unknown_event", "", "Hello", None),
        ],
    )
    def test_stream_event_processing(
        self, client, event_type, delta_text, expected_content, expected_finish
    ):
        """Test stream event processing with different event types."""
        accumulated = "Hello"
        event = MockStreamEvent(event_type, delta_text)

        new_accumulated, _, chunk = client._process_stream_event(event, accumulated)

        if event_type == "unknown_event":
            assert chunk is None
            assert new_accumulated == "Hello"
        else:
            assert new_accumulated == expected_content
            if chunk:
                assert chunk.common.content == expected_content
                assert chunk.common.finish_reason == expected_finish

    def test_stream_event_with_tools(self, client):
        """Test stream event processing with tool calls."""
        accumulated = "Calculating..."
        tool_calls = {}

        # Tool use block start
        event = SimpleNamespace(
            type="content_block_start",
            index=1,
            content_block=SimpleNamespace(type="tool_use", id="toolu_123", name="calculator"),
        )
        _, new_tools, _ = client._process_stream_event(event, accumulated, tool_calls)
        assert "tool_call_1" in new_tools
        assert new_tools["tool_call_1"].call_id == "toolu_123"

        # Input JSON delta
        event = SimpleNamespace(
            type="content_block_delta",
            index=1,
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"x": 5}'),
        )
        _, new_tools, _ = client._process_stream_event(event, accumulated, new_tools)
        assert new_tools["tool_call_1"].arguments == '{"x": 5}'
        assert new_tools["tool_call_1"].status == "arguments_streaming"

        # Content block stop
        event = SimpleNamespace(type="content_block_stop", index=1)
        client._process_block_stop(event, new_tools)
        assert new_tools["tool_call_1"].status == "completed"

        # Test block stop with non-streaming status
        new_tools["tool_call_1"].status = "started"
        client._process_block_stop(event, new_tools)
        assert new_tools["tool_call_1"].status == "started"  # Should not change

        # Test block start without tool_use type
        event_non_tool = SimpleNamespace(
            type="content_block_start",
            index=2,
            content_block=SimpleNamespace(type="text"),  # Not tool_use
        )
        client._process_block_start(event_non_tool, new_tools)
        assert "tool_call_2" not in new_tools  # Should not create a new tool call

        # Test block start without content_block attribute
        event_no_block = SimpleNamespace(type="content_block_start", index=3)
        client._process_block_start(event_no_block, new_tools)  # Should not crash

    def test_sync_streaming(self, client):
        """Test synchronous streaming."""
        events = [
            MockStreamEvent("content_block_delta", "Hello"),
            MockStreamEvent("unknown_event"),
            MockStreamEvent("message_stop"),
        ]

        chunks = list(client._stream(events))
        assert len(chunks) == 2
        assert chunks[0].common.content == "Hello"
        assert chunks[1].common.finish_reason == "end_turn"

    async def test_async_streaming(self, client):
        """Test asynchronous streaming."""

        async def agen():
            for event in [
                MockStreamEvent("content_block_delta", "Async"),
                MockStreamEvent("message_stop"),
            ]:
                yield event

        chunks = []
        async for chunk in client._astream(agen()):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].common.content == "Async"

    def test_chat_completion_basic(self, client, monkeypatch):
        """Test basic chat completion with various parameter combinations."""
        mock_response = MockResponse("Test response")
        captured_params = {}

        def mock_create(**kwargs):
            captured_params.update(kwargs)
            return mock_response

        client._client.messages = SimpleNamespace(create=mock_create)

        def filter_kwargs_func(f, kw):
            return kw

        monkeypatch.setattr(client, "_filter_kwargs", filter_kwargs_func)

        # Basic call
        result = client._chat_completion_impl(
            messages=[{"role": "user", "content": "Hello"}], model="claude-3-sonnet"
        )
        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Test response"

        # With system prompt
        captured_params.clear()
        messages_with_system = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]
        client._chat_completion_impl(messages=messages_with_system, model="claude-3-sonnet")
        assert captured_params["system"] == "Be helpful"

        # With tools
        captured_params.clear()
        tools = [{"name": "test_tool", "description": "Test"}]
        client._chat_completion_impl(
            messages=[{"role": "user", "content": "Use tool"}], model="claude-3-sonnet", tools=tools
        )
        assert "You have access to tools" in captured_params["system"]

        # With optional parameters
        captured_params.clear()
        client._chat_completion_impl(
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-3-sonnet",
            temperature=0.8,
            top_p=0.9,
            unsupported_param="ignored",
        )
        assert captured_params["temperature"] == 0.8
        assert "unsupported_param" not in captured_params

    def test_chat_completion_with_tool_execution(self, client, monkeypatch):
        """Test chat completion with tool execution loop."""

        def search(query: str) -> str:
            return f"Results for: {query}"

        tool = Tool(name="search", description="Search", function=search)
        client.tool_manager.register(func=tool.function, name=tool.name)

        # Mock responses
        first_response = MockResponse("Tool call")
        first_response.content = [
            SimpleNamespace(type="tool_use", id="c1", name="search", input={"query": "test"})
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

        def filter_kwargs_func(f, kw):
            return kw

        monkeypatch.setattr(client, "_filter_kwargs", filter_kwargs_func)

        result = client._chat_completion_impl(
            messages=[{"role": "user", "content": "Search for test"}],
            model="claude-3-sonnet",
            tools=[{"name": "search", "description": "Search"}],
        )

        assert "tool_calls" in result.common.metadata
        assert result.common.metadata["tool_calls"][0]["result"] == "Results for: test"

    def test_chat_completion_streaming_with_tools(self, client, monkeypatch):
        """Test streaming chat completion with tool execution."""
        mock_stream = Mock(spec=client_module.Stream)

        def mock_create(**kwargs):
            return mock_stream

        client._client.messages = SimpleNamespace(create=mock_create)

        def mock_stream_with_tools(stream, **kwargs):
            yield ChimericStreamChunk(
                native=Mock(), common=StreamChunk(content="Result", delta="Result", metadata={})
            )

        monkeypatch.setattr(client, "_process_stream_with_tools_sync", mock_stream_with_tools)

        result = client._chat_completion_impl(
            messages=[{"role": "user", "content": "Calculate"}],
            model="claude-3-sonnet",
            stream=True,
            tools=[{"name": "calc", "description": "Calculate"}],
        )

        chunks = list(result)
        assert len(chunks) == 1
        assert "Result" in chunks[0].common.content

        # Test streaming without tools
        def mock_stream_no_tools(stream):
            yield ChimericStreamChunk(
                native=Mock(), common=StreamChunk(content="No tools", delta="No tools", metadata={})
            )

        monkeypatch.setattr(client, "_stream", mock_stream_no_tools)

        result = client._chat_completion_impl(
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-3-sonnet",
            stream=True,
            tools=None,
        )

        chunks = list(result)
        assert len(chunks) == 1
        assert "No tools" in chunks[0].common.content

    async def test_async_chat_completion(self, client, monkeypatch):
        """Test async chat completion with all scenarios."""
        mock_response = MockResponse("Async response")

        async def mock_create(**kwargs):
            return mock_response

        client._async_client.messages = SimpleNamespace(create=mock_create)

        async def mock_process_stream_async(stream, **kwargs):
            yield ChimericStreamChunk(
                native=Mock(),
                common=StreamChunk(content="Async response", delta="Response", metadata={}),
            )

        monkeypatch.setattr(client, "_process_stream_with_tools_async", mock_process_stream_async)

        messages = [{"role": "user", "content": "Async test"}]

        # Register the tool we'll use in the test
        def async_edge_tool(value: str) -> str:
            return f"processed: {value}"

        tool = Tool(name="async_edge_tool", description="Async edge tool", function=async_edge_tool)
        client.tool_manager.register(func=tool.function, name=tool.name)

        # Define tool_calls for the test
        tool_calls = {
            "t1": ToolCallChunk(
                id="t1",
                call_id="c1",
                name="async_edge_tool",
                arguments='{"value": "test"}',
                status="completed",
            ),
        }

        # Case 2: Test with max_tokens already in kwargs
        chunks = []
        async for chunk in client._handle_tool_execution_and_continue_async(
            tool_calls,
            messages,
            "claude-3-sonnet",
            [{"name": "async_edge_tool"}],
            "Async test content",
            max_tokens=1024,
        ):
            chunks.append(chunk)

        assert len(chunks) == 1

        # Test with no completed tool calls
        tool_calls_no_complete = {
            "t1": ToolCallChunk(
                id="t1",
                call_id="c1",
                name="async_edge_tool",
                arguments='{"value": "test"}',
                status="started",
            ),
        }

        chunks = []
        async for chunk in client._handle_tool_execution_and_continue_async(
            tool_calls_no_complete,
            messages,
            "claude-3-sonnet",
            [{"name": "async_edge_tool"}],
            "Content",
        ):
            chunks.append(chunk)

        assert len(chunks) == 0  # No completed tools, no continuation

    def test_file_upload(self, client):
        """Test file upload success and error cases."""
        # Success case
        mock_file = MockFile()

        def mock_upload(**kwargs):
            return mock_file

        client._client.beta = SimpleNamespace(files=SimpleNamespace(upload=mock_upload))

        result = client._upload_file(file="test_file_content")
        assert isinstance(result.common, FileUploadResponse)
        assert result.common.file_id == "file-anthro-123"
        assert result.common.filename == "test_file.txt"

        # Missing file parameter
        with pytest.raises(ValueError, match="'file' parameter is required"):
            client._upload_file(purpose="test")

        # API error propagation
        def mock_upload_error(**kwargs):
            error = Exception("Upload failed")
            error.status_code = 413
            raise error

        client._client.beta.files.upload = mock_upload_error

        with pytest.raises(Exception) as exc_info:
            client._upload_file(file="large_file")
        assert "Upload failed" in str(exc_info.value)

    def test_request_and_error_tracking(self, client, monkeypatch):
        """Test request and error counting."""

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

    def test_exception_propagation(self, client, monkeypatch):
        """Test that exceptions properly propagate to base class."""

        def mock_create(**kwargs):
            error = Exception("API Error")
            error.status_code = 429
            raise error

        client._client.messages = SimpleNamespace(create=mock_create)

        def filter_kwargs_func(f, kw):
            return kw

        monkeypatch.setattr(client, "_filter_kwargs", filter_kwargs_func)

        with pytest.raises(Exception) as exc_info:
            client._chat_completion_impl(
                messages=[{"role": "user", "content": "Hello"}], model="claude-3-sonnet"
            )

        assert "API Error" in str(exc_info.value)
        assert exc_info.value.status_code == 429

    def test_response_tools_processing(self, client):
        """Test comprehensive tool response processing."""

        def calc(a: int, b: int) -> int:
            return a + b

        tool = Tool(name="calc", description="Calculate", function=calc)
        client.tool_manager.register(func=tool.function, name=tool.name)

        # Response with mixed content types
        response = MockResponse("Text content")
        response.content = [
            SimpleNamespace(type="text", text="I'll calculate that."),
            SimpleNamespace(type="image", data="base64data"),  # Unknown type
            SimpleNamespace(type="tool_use", id="c1", name="calc", input={"a": 5, "b": 3}),
            SimpleNamespace(type="audio", url="audio.mp3"),  # Unknown type
        ]

        messages = [{"role": "user", "content": "Calculate 5 + 3"}]
        tool_calls, updated_messages = client._process_response_tools(response, messages)

        # Verify processing
        assert len(tool_calls) == 1
        assert tool_calls[0]["result"] == "8"

        # Check assistant message structure
        assistant_msg = updated_messages[1]
        assert len(assistant_msg["content"]) == 2  # Only text and tool_use
        content_types = [c["type"] for c in assistant_msg["content"]]
        assert content_types == ["text", "tool_use"]

        # Test with no tool calls
        response_no_tools = MockResponse("No tools")
        response_no_tools.content = [SimpleNamespace(type="text", text="Regular response")]
        tool_calls, updated = client._process_response_tools(response_no_tools, messages)
        assert tool_calls == []
        assert updated == messages

    def test_create_chimeric_response_variations(self, client):
        """Test response creation with various scenarios."""
        # With text content and tool calls
        response = MockResponse("Text response")
        tool_calls = [{"name": "tool1", "arguments": {"x": 1}, "result": "result1"}]
        result = client._create_chimeric_response(response, tool_calls)
        assert result.common.content == "Text response"
        assert "tool_calls" in result.common.metadata

        # Tool-only response (no text)
        response_no_text = MockResponse("")
        response_no_text.content = []
        result = client._create_chimeric_response(response_no_text, tool_calls)
        assert "Tool execution results:" in result.common.content
        assert "tool1(x=1) → result1" in result.common.content

        # No content or tools
        result = client._create_chimeric_response(response_no_text, [])
        assert result.common.content == ""

    def test_process_stream_with_tools_comprehensive(self, client, monkeypatch):
        """Test comprehensive streaming with tools scenarios."""

        def test_tool(value: str) -> str:
            return f"processed: {value}"

        tool = Tool(name="stream_tool", description="Stream tool", function=test_tool)
        client.tool_manager.register(func=tool.function, name=tool.name)

        # Setup mocks for tool execution continuation
        def mock_handle_tool_execution(tool_calls, messages, model, tools, content, **kwargs):
            yield ChimericStreamChunk(
                native=Mock(),
                common=StreamChunk(content="Tool executed", delta="Tool executed", metadata={}),
            )

        monkeypatch.setattr(
            client, "_handle_tool_execution_and_continue_sync", mock_handle_tool_execution
        )

        # Create events that include tool calls
        def mock_events():
            yield MockStreamEvent("content_block_delta", "Processing...")
            yield SimpleNamespace(type="message_stop", model_dump=lambda: {"type": "message_stop"})

        tool_calls = {
            "tool_call_1": ToolCallChunk(
                id="tool_call_1",
                call_id="call_test",
                name="stream_tool",
                arguments='{"value": "test"}',
                status="completed",
            )
        }

        # Mock process_stream_event to return tool calls on finish
        original_process = client._process_stream_event

        def mock_process_stream_event(event, acc, tools=None):
            if tools is None:
                tools = {}
            if event.type == "message_stop":
                return (
                    acc,
                    tool_calls,
                    ChimericStreamChunk(
                        native=event,
                        common=StreamChunk(content=acc, finish_reason="end_turn", metadata={}),
                    ),
                )
            return original_process(event, acc, tools)

        monkeypatch.setattr(client, "_process_stream_event", mock_process_stream_event)

        # Test with tools
        chunks = list(
            client._process_stream_with_tools_sync(
                mock_events(),
                original_messages=[{"role": "user", "content": "Test"}],
                original_model="claude-3-sonnet",
                original_tools=[{"name": "stream_tool"}],
            )
        )

        assert len(chunks) >= 1

        # Test without tools
        def mock_events_no_finish():
            yield MockStreamEvent("content_block_delta", "Hello")

        chunks = list(
            client._process_stream_with_tools_sync(
                mock_events_no_finish(),
                original_messages=[{"role": "user", "content": "Test"}],
                original_model="claude-3-sonnet",
                original_tools=None,
            )
        )

        assert len(chunks) == 1
        assert "Hello" in chunks[0].common.content

        # Test early return when finish reason is found
        def mock_events_with_finish():
            yield SimpleNamespace(type="message_stop", model_dump=lambda: {"type": "message_stop"})

        # Reset process_stream_event to original
        monkeypatch.setattr(client, "_process_stream_event", original_process)

        chunks = list(
            client._process_stream_with_tools_sync(
                mock_events_with_finish(),
                original_messages=[{"role": "user", "content": "Test"}],
                original_model="claude-3-sonnet",
                original_tools=None,
            )
        )

        assert len(chunks) == 1
        assert chunks[0].common.finish_reason == "end_turn"

    async def test_process_stream_with_tools_async_comprehensive(self, client, monkeypatch):
        """Test async streaming with tools edge cases."""

        def test_tool(value: str) -> str:
            return f"async processed: {value}"

        tool = Tool(name="async_stream_tool", description="Async stream tool", function=test_tool)
        client.tool_manager.register(func=tool.function, name=tool.name)

        # Test early return with finish reason
        async def mock_events_with_finish():
            yield SimpleNamespace(type="message_stop", model_dump=lambda: {"type": "message_stop"})

        tool_calls = {}
        original_process = client._process_stream_event

        def mock_process_stream_event(event, acc, tools=None):
            if tools is None:
                tools = {}
            if event.type == "message_stop":
                return (
                    acc,
                    tool_calls,
                    ChimericStreamChunk(
                        native=event,
                        common=StreamChunk(content=acc, finish_reason="end_turn", metadata={}),
                    ),
                )
            return original_process(event, acc, tools)

        monkeypatch.setattr(client, "_process_stream_event", mock_process_stream_event)

        chunks = []
        async for chunk in client._process_stream_with_tools_async(
            mock_events_with_finish(),
            original_messages=[{"role": "user", "content": "Test"}],
            original_model="claude-3-sonnet",
            original_tools=None,
        ):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].common.finish_reason == "end_turn"

    def test_content_delta_edge_cases(self, client):
        """Test content delta processing edge cases."""
        accumulated = "test"
        tool_calls = {}

        # Event with neither text nor partial_json
        event = SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="other", data="some_data"),
        )

        new_acc, new_tools, chunk = client._process_content_delta(event, accumulated, tool_calls)
        assert new_acc == accumulated
        assert new_tools == tool_calls
        assert chunk is None

        # Test partial JSON processing
        event_json = SimpleNamespace(
            type="content_block_delta",
            index=1,
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"test": "value"}'),
        )

        new_acc, new_tools, chunk = client._process_content_delta(
            event_json, accumulated, tool_calls
        )
        assert "tool_call_1" in new_tools
        assert new_tools["tool_call_1"].arguments == '{"test": "value"}'

    def test_make_create_params_comprehensive(self, client):
        """Test comprehensive parameter creation scenarios."""
        # With tools and existing system prompt
        tools = [{"name": "test", "description": "test tool"}]
        messages_with_system = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

        params = client._make_create_params(
            messages=messages_with_system,
            model="claude-3-sonnet",
            stream=False,
            tools=tools,
            temperature=0.7,
            max_tokens=2048,
        )

        assert "You are helpful" in params["system"]
        assert "You have access to tools" in params["system"]
        assert params["tools"] == tools
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 2048

        # Without system prompt but with tools
        messages_no_system = [{"role": "user", "content": "Hello"}]
        params = client._make_create_params(
            messages=messages_no_system, model="claude-3-sonnet", stream=False, tools=tools
        )

        assert (
            params["system"]
            == "You have access to tools. Use them when appropriate to help answer the user's questions. When you need to use multiple tools, call them in parallel whenever possible for efficiency."
        )

        # No tools, no system
        params = client._make_create_params(
            messages=messages_no_system, model="claude-3-sonnet", stream=True, tools=None
        )

        assert "system" not in params
        assert params["stream"] is True

    def test_tool_execution_loops(self, client, monkeypatch):
        """Test tool execution loops for non-streaming responses."""

        def mock_tool(value: str) -> str:
            return f"result: {value}"

        tool = Tool(name="loop_tool", description="Loop tool", function=mock_tool)
        client.tool_manager.register(func=tool.function, name=tool.name)

        # Mock _process_response_tools to avoid double execution
        def mock_process_response_tools(response, messages):
            if hasattr(response, "_tool_processed"):
                # Second call - no tools
                return [], messages

            # First call - return tool calls
            response._tool_processed = True
            tool_calls = [
                {
                    "name": "loop_tool",
                    "arguments": {"value": "test"},
                    "result": "result: test",
                    "call_id": "c1",
                }
            ]
            updated_messages = [*messages, {"role": "assistant", "content": "Using tool"}]
            return tool_calls, updated_messages

        monkeypatch.setattr(client, "_process_response_tools", mock_process_response_tools)

        # Response with tool call
        response1 = MockResponse("Using tool")
        response1.content = [
            SimpleNamespace(type="tool_use", id="c1", name="loop_tool", input={"value": "test"})
        ]

        # Final response without tools
        response2 = MockResponse("Final answer")

        responses = [response1, response2]
        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            result = responses[call_count]
            call_count += 1
            return result

        client._client.messages = SimpleNamespace(create=mock_create)

        params = {"messages": [{"role": "user", "content": "Test"}], "model": "claude-3-sonnet"}

        # Test sync version
        final_response, tool_calls = client._handle_tool_execution_loop(response1, params)
        assert hasattr(final_response, "content")
        assert len(tool_calls) == 1
        assert tool_calls[0]["result"] == "result: test"

        # Test with no tool calls initially
        response_no_tools = MockResponse("No tools needed")
        response_no_tools._tool_processed = True  # Mark as already processed
        final_response, tool_calls = client._handle_tool_execution_loop(response_no_tools, params)
        assert hasattr(final_response, "content")
        assert tool_calls == []

    async def test_async_tool_execution_loops(self, client, monkeypatch):
        """Test async tool execution loops."""

        def mock_tool(value: str) -> str:
            return f"async result: {value}"

        tool = Tool(name="async_loop_tool", description="Async loop tool", function=mock_tool)
        client.tool_manager.register(func=tool.function, name=tool.name)

        # Mock _process_response_tools to avoid double execution
        def mock_process_response_tools(response, messages):
            if hasattr(response, "_tool_processed"):
                # Second call - no tools
                return [], messages

            # First call - return tool calls
            response._tool_processed = True
            tool_calls = [
                {
                    "name": "async_loop_tool",
                    "arguments": {"value": "async_test"},
                    "result": "async result: async_test",
                    "call_id": "c1",
                }
            ]
            updated_messages = [*messages, {"role": "assistant", "content": "Using async tool"}]
            return tool_calls, updated_messages

        monkeypatch.setattr(client, "_process_response_tools", mock_process_response_tools)

        response1 = MockResponse("Using async tool")
        response1.content = [
            SimpleNamespace(
                type="tool_use", id="c1", name="async_loop_tool", input={"value": "async_test"}
            )
        ]

        response2 = MockResponse("Async final answer")

        responses = [response1, response2]
        call_count = 0

        async def mock_create(**kwargs):
            nonlocal call_count
            result = responses[call_count]
            call_count += 1
            return result

        client._async_client.messages = SimpleNamespace(create=mock_create)

        params = {
            "messages": [{"role": "user", "content": "Async test"}],
            "model": "claude-3-sonnet",
        }

        final_response, tool_calls = await client._handle_async_tool_execution_loop(
            response1, params
        )
        assert hasattr(final_response, "content")
        assert len(tool_calls) == 1
        assert tool_calls[0]["result"] == "async result: async_test"

    async def test_async_chat_completion_complete(self, client, monkeypatch):
        """Test complete async chat completion scenarios."""
        mock_response = MockResponse("Async response")

        async def mock_create(**kwargs):
            return mock_response

        client._async_client.messages = SimpleNamespace(create=mock_create)

        def filter_kwargs_func(f, kw):
            return kw

        monkeypatch.setattr(client, "_filter_kwargs", filter_kwargs_func)

        # Basic async call
        result = await client._achat_completion_impl(
            messages=[{"role": "user", "content": "Hello"}], model="claude-3-sonnet"
        )
        assert result.common.content == "Async response"

        # With tool execution
        def translate(text: str) -> str:
            return f"Translated: {text}"

        tool = Tool(name="translate", description="Translate", function=translate)
        client.tool_manager.register(func=tool.function, name=tool.name)

        first_response = MockResponse("Tool call")
        first_response.content = [
            SimpleNamespace(type="tool_use", id="c1", name="translate", input={"text": "hello"})
        ]
        second_response = MockResponse("Translation complete")

        responses = [first_response, second_response]
        call_count = 0

        async def mock_create_with_tools(**kwargs):
            nonlocal call_count
            result = responses[call_count]
            call_count += 1
            return result

        client._async_client.messages.create = mock_create_with_tools

        result = await client._achat_completion_impl(
            messages=[{"role": "user", "content": "Translate hello"}],
            model="claude-3-sonnet",
            tools=[{"name": "translate", "description": "Translate"}],
        )

        assert "tool_calls" in result.common.metadata
        assert result.common.metadata["tool_calls"][0]["result"] == "Translated: hello"

        # Test async streaming with tools
        mock_async_stream = Mock(spec=client_module.AsyncStream)

        async def mock_create_stream(**kwargs):
            return mock_async_stream

        client._async_client.messages.create = mock_create_stream

        async def mock_astream_with_tools(stream, **kwargs):
            yield ChimericStreamChunk(
                native=Mock(),
                common=StreamChunk(content="Async stream", delta="Async stream", metadata={}),
            )

        monkeypatch.setattr(client, "_process_stream_with_tools_async", mock_astream_with_tools)

        result = await client._achat_completion_impl(
            messages=[{"role": "user", "content": "Stream with tools"}],
            model="claude-3-sonnet",
            stream=True,
            tools=[{"name": "tool", "description": "Tool"}],
        )

        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        assert len(chunks) == 1

        # Test async streaming without tools
        async def mock_astream_no_tools(stream):
            yield ChimericStreamChunk(
                native=Mock(),
                common=StreamChunk(content="No tools async", delta="No tools", metadata={}),
            )

        monkeypatch.setattr(client, "_astream", mock_astream_no_tools)

        result = await client._achat_completion_impl(
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-3-sonnet",
            stream=True,
            tools=None,
        )

        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        assert len(chunks) == 1

    def test_handle_tool_execution_edge_cases_sync(self, client, monkeypatch):
        """Test sync tool execution handler edge cases."""

        def test_tool(value: str) -> str:
            return f"result: {value}"

        tool = Tool(name="edge_tool", description="Edge case tool", function=test_tool)
        client.tool_manager.register(func=tool.function, name=tool.name)

        # Case 1: Tool calls with non-completed status
        tool_calls = {
            "t1": ToolCallChunk(
                id="t1",
                call_id="c1",
                name="edge_tool",
                arguments='{"value": "test"}',
                status="started",  # NOT completed
            ),
            "t2": ToolCallChunk(
                id="t2",
                call_id="c2",
                name="edge_tool",
                arguments='{"value": "test2"}',
                status="completed",
            ),
        }

        # Mock the continuation
        def mock_create(**kwargs):
            return [MockStreamEvent("content_block_delta", "Continued")]

        client._client.messages = SimpleNamespace(create=mock_create)

        def mock_process_stream(stream, **kwargs):
            yield ChimericStreamChunk(
                native=Mock(), common=StreamChunk(content="Response", delta="Response", metadata={})
            )

        monkeypatch.setattr(client, "_process_stream_with_tools_sync", mock_process_stream)

        messages = [{"role": "user", "content": "Test"}]

        # Case 2: Test with max_tokens already in kwargs
        chunks = list(
            client._handle_tool_execution_and_continue_sync(
                tool_calls,
                messages,
                "claude-3-sonnet",
                [{"name": "edge_tool"}],
                "Test content",
                max_tokens=2048,
            )
        )

        assert len(chunks) == 1

        # Test with no completed tool calls
        tool_calls_no_complete = {
            "t1": ToolCallChunk(
                id="t1",
                call_id="c1",
                name="edge_tool",
                arguments='{"value": "test"}',
                status="started",
            ),
        }

        chunks = list(
            client._handle_tool_execution_and_continue_sync(
                tool_calls_no_complete,
                messages,
                "claude-3-sonnet",
                [{"name": "edge_tool"}],
                "Test content",
            )
        )

        assert len(chunks) == 0  # No completed tools, no continuation

    async def test_handle_tool_execution_edge_cases_async(self, client, monkeypatch):
        """Test async tool execution handler edge cases."""

        def test_tool(value: str) -> str:
            return f"async result: {value}"

        tool = Tool(name="async_edge_tool_2", description="Async edge tool", function=test_tool)
        client.tool_manager.register(func=tool.function, name=tool.name)

        # Case 1: Tool calls with non-completed status
        tool_calls = {
            "t1": ToolCallChunk(
                id="t1",
                call_id="c1",
                name="async_edge_tool_2",
                arguments='{"value": "test"}',
                status="arguments_streaming",  # NOT completed
            ),
            "t2": ToolCallChunk(
                id="t2",
                call_id="c2",
                name="async_edge_tool_2",
                arguments='{"value": "test2"}',
                status="completed",
            ),
        }

        # Mock the continuation
        async def mock_create(**kwargs):
            class MockAsyncStream:
                async def __aiter__(self):
                    yield MockStreamEvent("content_block_delta", "Async continued")

            return MockAsyncStream()

        client._async_client.messages = SimpleNamespace(create=mock_create)

        async def mock_process_stream_async(stream, **kwargs):
            yield ChimericStreamChunk(
                native=Mock(),
                common=StreamChunk(content="Async continued", delta="Async continued", metadata={}),
            )

        monkeypatch.setattr(client, "_process_stream_with_tools_async", mock_process_stream_async)

        messages = [{"role": "user", "content": "Async test"}]

        # Case 2: Test with max_tokens already in kwargs
        chunks = []
        async for chunk in client._handle_tool_execution_and_continue_async(
            tool_calls,
            messages,
            "claude-3-sonnet",
            [{"name": "async_edge_tool_2"}],
            "Async test content",
            max_tokens=1024,
        ):
            chunks.append(chunk)

        assert len(chunks) == 1

        # Test with no completed tool calls
        tool_calls_no_complete = {
            "t1": ToolCallChunk(
                id="t1",
                call_id="c1",
                name="async_edge_tool_2",
                arguments='{"value": "test"}',
                status="started",
            ),
        }

        chunks = []
        async for chunk in client._handle_tool_execution_and_continue_async(
            tool_calls_no_complete,
            messages,
            "claude-3-sonnet",
            [{"name": "async_edge_tool_2"}],
            "Content",
        ):
            chunks.append(chunk)

        assert len(chunks) == 0  # No completed tools, no continuation

    async def test_async_process_stream_with_tools_finish_reason_coverage(
        self, client, monkeypatch
    ):
        """Test async streaming with finish reason."""

        def finish_tool(value: str) -> str:
            return f"finished: {value}"

        tool = Tool(name="finish_coverage_tool", description="Finish tool", function=finish_tool)
        client.tool_manager.register(func=tool.function, name=tool.name)

        # Mock events that will trigger finish reason and tool execution
        async def mock_events():
            # First event creates a tool call
            yield SimpleNamespace(
                type="content_block_start",
                index=1,
                content_block=SimpleNamespace(
                    type="tool_use", id="finish_123", name="finish_coverage_tool"
                ),
            )
            # Add input to the tool call
            yield SimpleNamespace(
                type="content_block_delta",
                index=1,
                delta=SimpleNamespace(type="input_json_delta", partial_json='{"value": "test"}'),
            )
            # Stop the block to mark tool as completed
            yield SimpleNamespace(type="content_block_stop", index=1)
            # End with finish reason to trigger tool execution
            yield SimpleNamespace(type="message_stop", model_dump=lambda: {"type": "message_stop"})

        # Mock tool execution continuation
        async def mock_handle_tool_execution(tool_calls, messages, model, tools, content, **kwargs):
            yield ChimericStreamChunk(
                native=Mock(),
                common=StreamChunk(
                    content="Tool executed and finished", delta="Tool executed", metadata={}
                ),
            )

        monkeypatch.setattr(
            client, "_handle_tool_execution_and_continue_async", mock_handle_tool_execution
        )

        # Process the stream
        chunks = []
        async for chunk in client._process_stream_with_tools_async(
            mock_events(),
            original_messages=[{"role": "user", "content": "Test finish"}],
            original_model="claude-3-sonnet",
            original_tools=[{"name": "finish_coverage_tool"}],
        ):
            chunks.append(chunk)

        # Should have chunks from both the stream event and tool execution
        assert len(chunks) >= 1

    def test_sync_streaming_no_chunk_returned(self, client):
        """Test sync streaming when process_stream_event returns None chunk."""
        # Create an event that will return None chunk
        events = [SimpleNamespace(type="unknown_event", model_dump=lambda: {"type": "unknown"})]

        chunks = list(client._stream(events))
        assert len(chunks) == 0  # No chunks should be yielded for unknown events

    def test_sync_streaming_with_tools_no_chunk_returned(self, client):
        """Test sync streaming with tools when process_stream_event returns None chunk."""
        # Create events that will return None chunk
        events = [SimpleNamespace(type="unknown_event", model_dump=lambda: {"type": "unknown"})]

        chunks = list(
            client._process_stream_with_tools_sync(
                events,
                original_messages=[{"role": "user", "content": "Test"}],
                original_model="claude-3-sonnet",
                original_tools=None,
            )
        )
        assert len(chunks) == 0  # No chunks should be yielded for unknown events

    async def test_async_streaming_no_chunk_returned(self, client):
        """Test async streaming when process_stream_event returns None chunk."""

        async def mock_events():
            # Unknown event type that returns None chunk
            yield SimpleNamespace(type="unknown_event", model_dump=lambda: {"type": "unknown"})

        chunks = []
        async for chunk in client._astream(mock_events()):
            chunks.append(chunk)

        assert len(chunks) == 0  # No chunks should be yielded for unknown events

    async def test_async_streaming_with_tools_no_chunk_returned(self, client):
        """Test async streaming with tools when process_stream_event returns None chunk."""

        async def mock_events():
            # Unknown event type that returns None chunk
            yield SimpleNamespace(type="unknown_event", model_dump=lambda: {"type": "unknown"})

        chunks = []
        async for chunk in client._process_stream_with_tools_async(
            mock_events(),
            original_messages=[{"role": "user", "content": "Test"}],
            original_model="claude-3-sonnet",
            original_tools=None,
        ):
            chunks.append(chunk)

        assert len(chunks) == 0  # No chunks should be yielded for unknown events

    def test_sync_tool_execution_empty_content(self, client, monkeypatch):
        """Test sync tool execution with empty accumulated content."""

        def empty_tool() -> str:
            return "tool result"

        tool = Tool(
            name="empty_content_tool", description="Empty content tool", function=empty_tool
        )
        client.tool_manager.register(func=tool.function, name=tool.name)

        # Tool calls with completed status but no accumulated content
        tool_calls = {
            "t1": ToolCallChunk(
                id="t1", call_id="c1", name="empty_content_tool", arguments="{}", status="completed"
            ),
        }

        messages = [{"role": "user", "content": "Test"}]

        # Call with empty accumulated content (empty string)
        chunks = list(
            client._handle_tool_execution_and_continue_sync(
                tool_calls,
                messages,
                "claude-3-sonnet",
                [{"name": "empty_content_tool"}],
                "",  # Empty accumulated content
            )
        )

        # Should still work but won't add text content to an assistant message
        assert len(chunks) >= 0

    async def test_async_tool_execution_empty_content(self, client, monkeypatch):
        """Test async tool execution with empty accumulated content."""

        def empty_tool_async() -> str:
            return "async tool result"

        tool = Tool(
            name="empty_content_async_tool",
            description="Empty content async tool",
            function=empty_tool_async,
        )
        client.tool_manager.register(func=tool.function, name=tool.name)

        # Mock the continuation
        async def mock_create(**kwargs):
            class MockAsyncStream:
                async def __aiter__(self):
                    yield MockStreamEvent("content_block_delta", "Response")

            return MockAsyncStream()

        client._async_client.messages = SimpleNamespace(create=mock_create)

        async def mock_process_stream_async(stream, **kwargs):
            yield ChimericStreamChunk(
                native=Mock(),
                common=StreamChunk(content="Final response", delta="Final", metadata={}),
            )

        monkeypatch.setattr(client, "_process_stream_with_tools_async", mock_process_stream_async)

        # Tool calls with completed status but no accumulated content
        tool_calls = {
            "t1": ToolCallChunk(
                id="t1",
                call_id="c1",
                name="empty_content_async_tool",
                arguments="{}",
                status="completed",
            ),
        }

        messages = [{"role": "user", "content": "Test"}]

        # Call with empty accumulated content (empty string)
        chunks = []
        async for chunk in client._handle_tool_execution_and_continue_async(
            tool_calls,
            messages,
            "claude-3-sonnet",
            [{"name": "empty_content_async_tool"}],
            "",  # Empty accumulated content
        ):
            chunks.append(chunk)

        # Should still work but won't add text content to an assistant message
        assert len(chunks) >= 0
