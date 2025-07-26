from chimeric.types import (
    Message,
    Tool,
    ToolParameters,
    Usage,
)
from chimeric.utils import (
    StreamProcessor,
    create_completion_response,
    normalize_tools,
)


class TestStreamProcessor:
    """Test StreamProcessor functionality for tool calls."""

    def test_process_tool_call_lifecycle(self):
        """Test complete tool call lifecycle through StreamProcessor."""
        processor = StreamProcessor()

        # Start a tool call
        processor.process_tool_call_start("call-1", "test_function", "output-id-1")

        # Verify tool call was created
        assert "call-1" in processor.state.tool_calls
        tool_call = processor.state.tool_calls["call-1"]
        assert tool_call.id == "call-1"
        assert tool_call.call_id == "output-id-1"
        assert tool_call.name == "test_function"
        assert tool_call.arguments == ""
        assert tool_call.status == "started"

        # Process arguments in chunks
        processor.process_tool_call_delta("call-1", '{"x": ')
        assert tool_call.arguments == '{"x": '
        assert tool_call.arguments_delta == '{"x": '
        assert tool_call.status == "arguments_streaming"

        processor.process_tool_call_delta("call-1", "42}")
        assert tool_call.arguments == '{"x": 42}'
        assert tool_call.arguments_delta == "42}"

        # Complete the tool call
        processor.process_tool_call_complete("call-1")
        assert tool_call.status == "completed"
        assert tool_call.arguments_delta is None

        # Get completed tool calls
        completed = processor.get_completed_tool_calls()
        assert len(completed) == 1
        assert completed[0].id == "call-1"
        assert completed[0].call_id == "output-id-1"
        assert completed[0].arguments == '{"x": 42}'

    def test_process_tool_call_without_custom_call_id(self):
        """Test tool call with default call_id (same as id)."""
        processor = StreamProcessor()

        # Start without providing custom call_id
        processor.process_tool_call_start("call-2", "another_function")

        tool_call = processor.state.tool_calls["call-2"]
        assert tool_call.id == "call-2"
        assert tool_call.call_id == "call-2"  # Should default to id

    def test_multiple_tool_calls(self):
        """Test handling multiple concurrent tool calls."""
        processor = StreamProcessor()

        # Start multiple tool calls
        processor.process_tool_call_start("call-1", "func1")
        processor.process_tool_call_start("call-2", "func2")
        processor.process_tool_call_start("call-3", "func3")

        # Process arguments for different calls
        processor.process_tool_call_delta("call-1", '{"a": 1}')
        processor.process_tool_call_delta("call-2", '{"b": 2}')
        processor.process_tool_call_delta("call-3", '{"c": 3}')

        # Complete only some calls
        processor.process_tool_call_complete("call-1")
        processor.process_tool_call_complete("call-3")

        # Check completed calls
        completed = processor.get_completed_tool_calls()
        assert len(completed) == 2
        completed_ids = {tc.id for tc in completed}
        assert completed_ids == {"call-1", "call-3"}

        # call-2 should still be incomplete
        assert processor.state.tool_calls["call-2"].status == "arguments_streaming"

    def test_process_tool_call_delta_nonexistent_call(self):
        """Test processing delta for non-existent tool call."""
        processor = StreamProcessor()

        # Process delta without starting the call - should be silently ignored
        processor.process_tool_call_delta("nonexistent", "some args")

        # Verify no tool calls were created
        assert len(processor.state.tool_calls) == 0

    def test_process_tool_call_complete_nonexistent_call(self):
        """Test completing non-existent tool call."""
        processor = StreamProcessor()

        # Complete without starting - should be silently ignored
        processor.process_tool_call_complete("nonexistent")

        # Verify no errors and no tool calls
        assert len(processor.state.tool_calls) == 0

    def test_get_completed_tool_calls_empty(self):
        """Test getting completed calls when none exist."""
        processor = StreamProcessor()

        # No tool calls at all
        assert processor.get_completed_tool_calls() == []

        # With incomplete tool calls
        processor.process_tool_call_start("call-1", "func1")
        processor.process_tool_call_delta("call-1", '{"incomplete": ')

        assert processor.get_completed_tool_calls() == []


class TestNormalizeTools:
    """Test normalize_tools function."""

    def test_normalize_tools_empty(self):
        """Test normalizing empty tools list."""
        assert normalize_tools(None) == []
        assert normalize_tools([]) == []

    def test_normalize_tools_with_tool_objects(self):
        """Test normalizing list of Tool objects."""
        tools = [
            Tool(name="tool1", description="First tool"),
            Tool(name="tool2", description="Second tool", parameters=ToolParameters(properties={})),
        ]

        normalized = normalize_tools(tools)
        assert len(normalized) == 2
        assert all(isinstance(t, Tool) for t in normalized)
        assert normalized[0].name == "tool1"
        assert normalized[1].name == "tool2"

    def test_normalize_tools_with_dicts(self):
        """Test normalizing list of dictionaries."""
        tools = [
            {"name": "tool1", "description": "First tool"},
            {
                "name": "tool2",
                "description": "Second tool",
                "parameters": {"properties": {"x": {"type": "string"}}},
            },
        ]

        normalized = normalize_tools(tools)
        assert len(normalized) == 2
        assert all(isinstance(t, Tool) for t in normalized)
        assert normalized[0].name == "tool1"
        assert normalized[1].parameters.properties == {"x": {"type": "string"}}

    def test_normalize_tools_with_objects(self):
        """Test normalizing list of arbitrary objects with attributes."""

        # Create mock objects with attributes
        class MockTool:
            def __init__(self, name, description, parameters=None, function=None):
                self.name = name
                self.description = description
                self.parameters = parameters
                self.function = function

        def test_func():
            return "test"

        tools = [
            MockTool("tool1", "First tool"),
            MockTool("tool2", "Second tool", {"type": "object"}, test_func),
        ]

        normalized = normalize_tools(tools)
        assert len(normalized) == 2
        assert normalized[0].name == "tool1"
        assert normalized[0].parameters is None
        assert normalized[0].function is None
        assert normalized[1].name == "tool2"
        assert normalized[1].parameters.type == "object"
        assert normalized[1].function is test_func

    def test_normalize_tools_with_missing_attributes(self):
        """Test normalizing objects without expected attributes."""

        # Object without any expected attributes
        class EmptyObject:
            pass

        tools = [EmptyObject()]

        normalized = normalize_tools(tools)
        assert len(normalized) == 1
        assert normalized[0].name == "unknown"
        assert normalized[0].description == ""
        assert normalized[0].parameters is None
        assert normalized[0].function is None

    def test_normalize_tools_mixed_types(self):
        """Test normalizing mixed types of tools."""

        def func1():
            return "result"

        # Create a simple object with a function attribute
        class MockTool:
            def __init__(self):
                self.name = "tool3"
                self.description = "Mock"
                self.function = func1

        tools = [
            Tool(name="tool1", description="Tool object"),
            {"name": "tool2", "description": "Dict tool"},
            MockTool(),
        ]

        normalized = normalize_tools(tools)
        assert len(normalized) == 3
        assert all(isinstance(t, Tool) for t in normalized)
        assert normalized[0].name == "tool1"
        assert normalized[1].name == "tool2"
        assert normalized[2].name == "tool3"
        assert normalized[2].function is func1


class TestCreateCompletionResponse:
    """Test create_completion_response function."""

    def test_basic_completion_response(self):
        """Test creating basic completion response."""
        native = {"id": "123", "model": "test"}

        response = create_completion_response(
            native_response=native, content="Test response", model="test-model"
        )

        assert response.native == native
        assert response.common.content == "Test response"
        assert response.common.model == "test-model"
        assert response.common.usage.total_tokens == 0  # Default usage
        assert response.common.metadata == {}

    def test_completion_response_with_usage(self):
        """Test creating response with usage information."""
        native = {"id": "123"}
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

        response = create_completion_response(
            native_response=native, content="Response with usage", usage=usage, model="test-model"
        )

        assert response.common.usage == usage
        assert response.common.usage.total_tokens == 30

    def test_completion_response_with_tool_calls(self):
        """Test creating response with tool calls."""
        from chimeric.types import ToolExecutionResult

        native = {"id": "123"}
        tool_calls = [
            ToolExecutionResult(
                call_id="1", name="test_tool", arguments='{"x": 42}', result="Result: 42"
            ),
            ToolExecutionResult(
                call_id="2", name="error_tool", arguments="{}", error="Tool failed", is_error=True
            ),
        ]

        response = create_completion_response(
            native_response=native,
            content="Response with tools",
            tool_calls=tool_calls,
            model="test-model",
        )

        # Tool calls should be in metadata
        assert "tool_calls" in response.common.metadata
        assert len(response.common.metadata["tool_calls"]) == 2

        # Check serialized tool calls
        tc1 = response.common.metadata["tool_calls"][0]
        assert tc1["call_id"] == "1"
        assert tc1["name"] == "test_tool"
        assert tc1["result"] == "Result: 42"
        assert tc1["is_error"] is False

        tc2 = response.common.metadata["tool_calls"][1]
        assert tc2["call_id"] == "2"
        assert tc2["error"] == "Tool failed"
        assert tc2["is_error"] is True

    def test_completion_response_with_metadata(self):
        """Test creating response with custom metadata."""
        native = {"id": "123"}
        metadata = {"custom_field": "value", "number": 42, "nested": {"key": "value"}}

        response = create_completion_response(
            native_response=native,
            content="Response with metadata",
            metadata=metadata,
            model="test-model",
        )

        assert response.common.metadata == metadata
        assert response.common.metadata["custom_field"] == "value"

    def test_completion_response_with_list_content(self):
        """Test creating response with list content."""
        native = {"id": "123"}
        content = [
            {"type": "text", "text": "Part 1"},
            {"type": "image", "url": "http://example.com/image.jpg"},
        ]

        response = create_completion_response(
            native_response=native, content=content, model="test-model"
        )

        assert response.common.content == content
        assert len(response.common.content) == 2


class TestNormalizeMessages:
    """Test normalize_messages function."""

    def test_normalize_single_message_object(self):
        """Test normalizing a single Message object."""
        from chimeric.utils import normalize_messages

        msg = Message(role="user", content="Hello")
        result = normalize_messages(msg)
        assert len(result) == 1
        assert result[0] == msg

    def test_normalize_single_string(self):
        """Test normalizing a single string."""
        from chimeric.utils import normalize_messages

        result = normalize_messages("Hello world")
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content == "Hello world"

    def test_normalize_single_dict(self):
        """Test normalizing a single dict."""
        from chimeric.utils import normalize_messages

        msg_dict = {"role": "assistant", "content": "Hi there"}
        result = normalize_messages(msg_dict)
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].content == "Hi there"

    def test_normalize_list_with_string(self):
        """Test normalizing list containing string."""
        from chimeric.utils import normalize_messages

        messages = ["Hello", Message(role="assistant", content="Hi")]
        result = normalize_messages(messages)
        assert len(result) == 2
        assert result[0].role == "user"
        assert result[0].content == "Hello"
        assert result[1].role == "assistant"

    def test_normalize_list_with_dict(self):
        """Test normalizing list containing dict."""
        from chimeric.utils import normalize_messages

        messages = [{"role": "user", "content": "Test"}]
        result = normalize_messages(messages)
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content == "Test"

    def test_normalize_list_with_other_object(self):
        """Test normalizing list containing arbitrary object."""
        from chimeric.utils import normalize_messages

        class CustomObj:
            def __str__(self):
                return "custom object"

        messages = [CustomObj()]
        result = normalize_messages(messages)
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content == "custom object"


class TestCreateStreamChunk:
    """Test create_stream_chunk function"""

    def test_create_stream_chunk_without_content_delta(self):
        """Test create_stream_chunk when content_delta is None."""
        from chimeric.utils import StreamProcessor, create_stream_chunk

        processor = StreamProcessor()
        processor.state.accumulated_content = "existing content"
        processor.state.metadata = {"test": "value"}

        # Test the else branch where content_delta is None
        chunk = create_stream_chunk(
            native_event={"event": "finish"},
            processor=processor,
            content_delta=None,
            finish_reason="stop",
            metadata={"custom": "meta"},
        )

        assert chunk.common.content == "existing content"
        assert chunk.common.finish_reason == "stop"
        assert chunk.common.metadata == {"custom": "meta"}
