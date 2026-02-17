"""Tests for the Anthropic Messages API adapter."""

import json

import pytest

from chimeric.adapters.anthropic import AnthropicAdapter
from chimeric.adapters.base import StreamState
from chimeric.types import Message, Tool, ToolCall, ToolParameters


@pytest.fixture
def adapter():
    return AnthropicAdapter()


@pytest.fixture
def messages():
    return [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello"),
    ]


class TestBuildRequestBody:
    def test_system_extracted_to_top_level(self, adapter, messages):
        """System messages must be removed from the message list and placed in 'system'."""
        body = adapter.build_request_body(messages, "claude-3-5-sonnet", False, None)
        assert "system" in body
        assert body["system"] == "You are helpful."
        # The remaining message list must not contain system messages
        roles = [m["role"] for m in body["messages"]]
        assert "system" not in roles

    def test_no_system_message(self, adapter):
        msgs = [Message(role="user", content="Hi")]
        body = adapter.build_request_body(msgs, "claude-3-5-sonnet", False, None)
        assert "system" not in body

    def test_tools_use_input_schema(self, adapter, messages):
        tool = Tool(
            name="fn",
            description="A function",
            parameters=ToolParameters(properties={"x": {"type": "integer"}}, required=["x"]),
        )
        body = adapter.build_request_body(messages, "claude-3-5-sonnet", False, [tool])
        assert "tools" in body
        assert "input_schema" in body["tools"][0]
        assert "parameters" not in body["tools"][0]

    def test_multiple_system_messages_joined(self, adapter):
        msgs = [
            Message(role="system", content="Part 1."),
            Message(role="system", content="Part 2."),
            Message(role="user", content="Hi"),
        ]
        body = adapter.build_request_body(msgs, "claude-3-5-sonnet", False, None)
        assert "Part 1." in body["system"]
        assert "Part 2." in body["system"]


class TestParseResponse:
    def test_text_content(self, adapter):
        data = {
            "content": [{"type": "text", "text": "Hello there!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "claude-3-5-sonnet",
            "stop_reason": "end_turn",
            "id": "msg_123",
        }
        response = adapter.parse_response(data, "claude-3-5-sonnet")
        assert response.content == "Hello there!"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15

    def test_multiple_text_blocks_joined(self, adapter):
        data = {
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ],
            "usage": {},
        }
        response = adapter.parse_response(data, "claude-3-5-sonnet")
        assert response.content == "Hello world"


class TestParseToolCalls:
    def test_tool_use_blocks(self, adapter):
        data = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {"city": "NYC"},
                },
            ]
        }
        calls = adapter.parse_tool_calls(data)
        assert calls is not None
        assert len(calls) == 1
        assert calls[0].call_id == "toolu_01"
        assert calls[0].name == "get_weather"
        assert json.loads(calls[0].arguments) == {"city": "NYC"}

    def test_no_tool_calls(self, adapter):
        data = {"content": [{"type": "text", "text": "Hi"}]}
        assert adapter.parse_tool_calls(data) is None


class TestBuildToolResultMessages:
    def test_structure(self, adapter):
        messages = [Message(role="user", content="Hi")]
        tool_calls = [ToolCall(call_id="toolu_01", name="get_weather", arguments='{"city": "NYC"}')]
        results = ["Sunny"]

        new_msgs = adapter.build_tool_result_messages(messages, tool_calls, results)

        assert len(new_msgs) == 3
        assert new_msgs[1].role == "assistant"
        assert isinstance(new_msgs[1].content, list)
        assert new_msgs[1].content[0]["type"] == "tool_use"

        assert new_msgs[2].role == "user"
        assert isinstance(new_msgs[2].content, list)
        assert new_msgs[2].content[0]["type"] == "tool_result"
        assert new_msgs[2].content[0]["tool_use_id"] == "toolu_01"

    def test_immutability(self, adapter):
        messages = [Message(role="user", content="Hi")]
        tool_calls = [ToolCall(call_id="t1", name="fn", arguments="{}")]
        original_len = len(messages)
        adapter.build_tool_result_messages(messages, tool_calls, ["result"])
        assert len(messages) == original_len


class TestParseSSEEvent:
    def test_text_delta(self, adapter):
        state = StreamState()
        event = json.dumps(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            }
        )
        chunk = adapter.parse_sse_event(event, state)
        assert chunk is not None
        assert chunk.delta == "Hello"
        assert state.accumulated_content == "Hello"

    def test_tool_use_start(self, adapter):
        state = StreamState()
        event = json.dumps(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_01", "name": "get_weather"},
            }
        )
        chunk = adapter.parse_sse_event(event, state)
        assert chunk is None
        assert "0" in state.tool_calls
        assert state.tool_calls["0"]["name"] == "get_weather"

    def test_input_json_delta(self, adapter):
        state = StreamState()
        state.tool_calls["0"] = {"id": "t1", "name": "fn", "arguments": ""}
        event = json.dumps(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
            }
        )
        adapter.parse_sse_event(event, state)
        assert state.tool_calls["0"]["arguments"] == '{"city":'

    def test_invalid_json_returns_none(self, adapter):
        assert adapter.parse_sse_event("bad-json", StreamState()) is None


class TestParseModelsResponse:
    def test_parses_display_name(self, adapter):
        data = {
            "data": [
                {"id": "claude-3-5-sonnet-20241022", "display_name": "Claude 3.5 Sonnet"},
            ]
        }
        models = adapter.parse_models_response(data)
        assert len(models) == 1
        assert models[0].id == "claude-3-5-sonnet-20241022"
        assert models[0].name == "Claude 3.5 Sonnet"
