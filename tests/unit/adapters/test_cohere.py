"""Tests for the Cohere v2 Chat API adapter."""

import json

import pytest

from chimeric.adapters.base import StreamState
from chimeric.adapters.cohere import CohereAdapter
from chimeric.types import Message, Tool, ToolCall, ToolParameters


@pytest.fixture
def adapter():
    return CohereAdapter()


@pytest.fixture
def messages():
    return [Message(role="user", content="Hello")]


class TestBuildRequestBody:
    def test_basic_structure(self, adapter, messages):
        body = adapter.build_request_body(messages, "command-r-plus", False, None)
        assert body["model"] == "command-r-plus"
        assert body["stream"] is False
        assert len(body["messages"]) == 1

    def test_tools_included(self, adapter, messages):
        tool = Tool(
            name="fn",
            description="A function",
            parameters=ToolParameters(properties={"x": {"type": "integer"}}, required=["x"]),
        )
        body = adapter.build_request_body(messages, "command-r-plus", False, [tool])
        assert "tools" in body
        assert body["tools"][0]["function"]["name"] == "fn"


class TestParseResponse:
    def test_text_content(self, adapter):
        data = {
            "message": {"content": [{"type": "text", "text": "Hello!"}]},
            "usage": {"tokens": {"input_tokens": 10, "output_tokens": 5}},
            "model": "command-r-plus",
            "finish_reason": "COMPLETE",
        }
        response = adapter.parse_response(data, "command-r-plus")
        assert response.content == "Hello!"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5

    def test_empty_content(self, adapter):
        data = {"message": {"content": []}, "usage": {}}
        response = adapter.parse_response(data, "command-r-plus")
        assert response.content == ""


class TestParseToolCalls:
    def test_tool_calls_in_message(self, adapter):
        data = {
            "message": {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": {"city": "NYC"}},
                    }
                ]
            }
        }
        calls = adapter.parse_tool_calls(data)
        assert calls is not None
        assert len(calls) == 1
        assert calls[0].call_id == "call_1"
        assert calls[0].name == "get_weather"

    def test_no_tool_calls(self, adapter):
        data = {"message": {"content": [{"type": "text", "text": "Hi"}]}}
        assert adapter.parse_tool_calls(data) is None


class TestBuildToolResultMessages:
    def test_structure(self, adapter):
        messages = [Message(role="user", content="Hi")]
        tool_calls = [ToolCall(call_id="call_1", name="fn", arguments='{"x": 1}')]
        results = ["42"]

        new_msgs = adapter.build_tool_result_messages(messages, tool_calls, results)
        assert len(new_msgs) == 3
        assert new_msgs[1].role == "assistant"
        assert new_msgs[2].role == "tool"
        assert new_msgs[2].tool_call_id == "call_1"

    def test_immutability(self, adapter):
        messages = [Message(role="user", content="Hi")]
        tool_calls = [ToolCall(call_id="c1", name="fn", arguments="{}")]
        original_len = len(messages)
        adapter.build_tool_result_messages(messages, tool_calls, ["r"])
        assert len(messages) == original_len


class TestParseSSEEvent:
    def test_content_delta(self, adapter):
        state = StreamState()
        event = json.dumps(
            {
                "type": "content-delta",
                "delta": {"message": {"content": {"text": "Hello"}}},
            }
        )
        chunk = adapter.parse_sse_event(event, state)
        assert chunk is not None
        assert chunk.delta == "Hello"
        assert state.accumulated_content == "Hello"

    def test_tool_call_start(self, adapter):
        state = StreamState()
        event = json.dumps(
            {
                "type": "tool-call-start",
                "index": 0,
                "delta": {"message": {"tool_calls": {"id": "tc_1", "function": {"name": "fn"}}}},
            }
        )
        adapter.parse_sse_event(event, state)
        assert "0" in state.tool_calls
        assert state.tool_calls["0"]["name"] == "fn"

    def test_message_end(self, adapter):
        state = StreamState()
        event = json.dumps(
            {
                "type": "message-end",
                "delta": {"finish_reason": "COMPLETE"},
            }
        )
        chunk = adapter.parse_sse_event(event, state)
        assert chunk is not None
        assert chunk.finish_reason == "COMPLETE"

    def test_invalid_json_returns_none(self, adapter):
        assert adapter.parse_sse_event("bad", StreamState()) is None


class TestParseModelsResponse:
    def test_basic_parsing(self, adapter):
        data = {
            "models": [
                {"name": "command-r-plus"},
                {"name": "command-r"},
            ]
        }
        models = adapter.parse_models_response(data)
        assert len(models) == 2
        assert models[0].id == "command-r-plus"

    def test_empty_response(self, adapter):
        assert adapter.parse_models_response({}) == []
