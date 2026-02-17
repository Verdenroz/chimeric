"""Tests for utility functions in chimeric.utils."""

from chimeric.types import Message, Tool, ToolCall, Usage
from chimeric.utils import (
    create_completion_response,
    create_stream_chunk,
    normalize_messages,
    normalize_tools,
)


class TestNormalizeMessages:
    def test_string_becomes_user_message(self):
        messages = normalize_messages("Hello")
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"

    def test_single_message_object_wrapped(self):
        msg = Message(role="assistant", content="Hi")
        messages = normalize_messages(msg)
        assert messages == [msg]

    def test_dict_converted(self):
        messages = normalize_messages({"role": "user", "content": "Hey"})
        assert len(messages) == 1
        assert messages[0].role == "user"

    def test_list_of_dicts(self):
        raw = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Second"},
        ]
        messages = normalize_messages(raw)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    def test_list_of_message_objects(self):
        raw = [Message(role="user", content="a"), Message(role="assistant", content="b")]
        messages = normalize_messages(raw)
        assert len(messages) == 2

    def test_mixed_list(self):
        raw = [
            "plain string",
            {"role": "user", "content": "dict form"},
            Message(role="assistant", content="message object"),
        ]
        messages = normalize_messages(raw)
        assert len(messages) == 3
        assert messages[0].role == "user"
        assert messages[0].content == "plain string"


class TestNormalizeTools:
    def test_none_returns_empty(self):
        assert normalize_tools(None) == []

    def test_empty_list(self):
        assert normalize_tools([]) == []

    def test_tool_object_passthrough(self):
        tool = Tool(name="fn", description="A function")
        result = normalize_tools([tool])
        assert result == [tool]

    def test_dict_converted(self):
        raw = {"name": "fn", "description": "A function"}
        result = normalize_tools([raw])
        assert result[0].name == "fn"

    def test_object_with_attrs(self):
        class FakeTool:
            name = "fn"
            description = "Desc"
            parameters = None
            function = None

        result = normalize_tools([FakeTool()])
        assert result[0].name == "fn"


class TestCreateCompletionResponse:
    def test_basic_response(self):
        usage = Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8)
        response = create_completion_response("Hello", usage=usage, model="gpt-4o")
        assert response.content == "Hello"
        assert response.usage.total_tokens == 8
        assert response.model == "gpt-4o"

    def test_tool_calls_in_metadata(self):
        tool_call = ToolCall(call_id="c1", name="fn", arguments="{}")
        response = create_completion_response("", tool_calls=[tool_call])
        assert response.metadata is not None
        assert "tool_calls" in response.metadata

    def test_no_metadata_when_empty(self):
        response = create_completion_response("Hi")
        assert response.metadata is None


class TestCreateStreamChunk:
    def test_basic_chunk(self):
        chunk = create_stream_chunk("Hello world", delta="world")
        assert chunk.content == "Hello world"
        assert chunk.delta == "world"

    def test_finish_reason(self):
        chunk = create_stream_chunk("done", finish_reason="stop")
        assert chunk.finish_reason == "stop"

    def test_metadata(self):
        chunk = create_stream_chunk("", metadata={"id": "req_123"})
        assert chunk.metadata == {"id": "req_123"}
