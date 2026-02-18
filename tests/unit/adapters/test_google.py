"""Tests for the Google Gemini REST API adapter."""

import json

import pytest

from chimeric.adapters.base import StreamState
from chimeric.adapters.google import GoogleAdapter
from chimeric.types import Message, Tool, ToolCall, ToolParameters


@pytest.fixture
def adapter():
    return GoogleAdapter()


@pytest.fixture
def messages():
    return [Message(role="user", content="Hello")]


class TestGetCompletionPath:
    def test_non_streaming(self, adapter):
        path = adapter.get_completion_path("gemini-2.0-flash", False)
        assert path == "/models/gemini-2.0-flash:generateContent"

    def test_streaming(self, adapter):
        path = adapter.get_completion_path("gemini-2.0-flash", True)
        assert path == "/models/gemini-2.0-flash:streamGenerateContent?alt=sse"


class TestBuildRequestBody:
    def test_basic_structure(self, adapter, messages):
        body = adapter.build_request_body(messages, "gemini-2.0-flash", False, None)
        assert "contents" in body
        assert body["contents"][0]["role"] == "user"
        assert body["contents"][0]["parts"][0]["text"] == "Hello"

    def test_system_extracted_to_instruction(self, adapter):
        msgs = [
            Message(role="system", content="Be helpful."),
            Message(role="user", content="Hi"),
        ]
        body = adapter.build_request_body(msgs, "gemini-2.0-flash", False, None)
        assert "system_instruction" in body
        assert body["system_instruction"]["parts"][0]["text"] == "Be helpful."
        # System should not appear in contents
        roles = [c["role"] for c in body["contents"]]
        assert "system" not in roles

    def test_assistant_mapped_to_model(self, adapter):
        msgs = [
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello!"),
        ]
        body = adapter.build_request_body(msgs, "gemini-2.0-flash", False, None)
        roles = [c["role"] for c in body["contents"]]
        assert "model" in roles
        assert "assistant" not in roles

    def test_tools_use_function_declarations(self, adapter, messages):
        tool = Tool(
            name="get_weather",
            description="Get weather",
            parameters=ToolParameters(properties={"city": {"type": "string"}}, required=["city"]),
        )
        body = adapter.build_request_body(messages, "gemini-2.0-flash", False, [tool])
        assert "tools" in body
        assert "functionDeclarations" in body["tools"][0]


class TestParseResponse:
    def test_text_response(self, adapter):
        data = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello!"}], "role": "model"},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }
        response = adapter.parse_response(data, "gemini-2.0-flash")
        assert response.content == "Hello!"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5

    def test_empty_candidates(self, adapter):
        data = {"candidates": [{"content": {"parts": []}}]}
        response = adapter.parse_response(data, "gemini-2.0-flash")
        assert response.content == ""


class TestParseToolCalls:
    def test_function_call_parts(self, adapter):
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "get_weather", "args": {"city": "NYC"}}}
                        ]
                    }
                }
            ]
        }
        calls = adapter.parse_tool_calls(data)
        assert calls is not None
        assert len(calls) == 1
        assert calls[0].name == "get_weather"
        assert json.loads(calls[0].arguments) == {"city": "NYC"}

    def test_no_function_calls(self, adapter):
        data = {"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]}
        assert adapter.parse_tool_calls(data) is None


class TestBuildToolResultMessages:
    def test_uses_function_response_format(self, adapter):
        messages = [Message(role="user", content="Hi")]
        tool_calls = [
            ToolCall(call_id="call_0_get_weather", name="get_weather", arguments='{"city": "NYC"}')
        ]
        results = ["Sunny"]

        new_msgs = adapter.build_tool_result_messages(messages, tool_calls, results)
        assert len(new_msgs) == 3
        assert new_msgs[1].role == "assistant"
        assert isinstance(new_msgs[1].content, list)
        assert "functionCall" in new_msgs[1].content[0]

        assert new_msgs[2].role == "user"
        assert isinstance(new_msgs[2].content, list)
        assert "functionResponse" in new_msgs[2].content[0]

    def test_immutability(self, adapter):
        messages = [Message(role="user", content="Hi")]
        tool_calls = [ToolCall(call_id="c0", name="fn", arguments="{}")]
        original_len = len(messages)
        adapter.build_tool_result_messages(messages, tool_calls, ["r"])
        assert len(messages) == original_len


class TestParseSSEEvent:
    def test_text_content(self, adapter):
        state = StreamState()
        event = json.dumps(
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Hi"}]},
                        "finishReason": None,
                    }
                ]
            }
        )
        chunk = adapter.parse_sse_event(event, state)
        assert chunk is not None
        assert chunk.delta == "Hi"

    def test_invalid_json(self, adapter):
        assert adapter.parse_sse_event("bad", StreamState()) is None

    def test_usage_metadata_in_chunk(self, adapter):
        state = StreamState()
        event = json.dumps(
            {
                "candidates": [{"content": {"parts": [{"text": ""}]}, "finishReason": "STOP"}],
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 3,
                    "totalTokenCount": 8,
                },
            }
        )
        chunk = adapter.parse_sse_event(event, state)
        assert chunk is not None
        assert chunk.metadata is not None
        assert "usage" in chunk.metadata


class TestParseModelsResponse:
    def test_strips_models_prefix(self, adapter):
        data = {
            "models": [
                {"name": "models/gemini-2.0-flash", "displayName": "Gemini 2.0 Flash"},
                {"name": "models/gemini-1.5-pro"},
            ]
        }
        models = adapter.parse_models_response(data)
        assert len(models) == 2
        assert models[0].id == "gemini-2.0-flash"
        assert models[0].name == "Gemini 2.0 Flash"
        assert models[1].id == "gemini-1.5-pro"
