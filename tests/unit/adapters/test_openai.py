"""Tests for the OpenAI Chat Completions adapter."""

import json

import pytest

from chimeric.adapters.base import StreamState
from chimeric.adapters.openai import OpenAIAdapter
from chimeric.types import EmbeddingResponse, Message, Tool, ToolCall, ToolParameters


@pytest.fixture
def adapter():
    return OpenAIAdapter()


@pytest.fixture
def messages():
    return [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello"),
    ]


@pytest.fixture
def tool():
    return Tool(
        name="get_weather",
        description="Get weather for a city",
        parameters=ToolParameters(
            properties={"city": {"type": "string"}},
            required=["city"],
        ),
    )


class TestBuildRequestBody:
    def test_basic_structure(self, adapter, messages):
        body = adapter.build_request_body(messages, "gpt-4o", False, None)
        assert body["model"] == "gpt-4o"
        assert body["stream"] is False
        assert len(body["messages"]) == 2
        assert "tools" not in body

    def test_streaming_flag(self, adapter, messages):
        body = adapter.build_request_body(messages, "gpt-4o", True, None)
        assert body["stream"] is True

    def test_tools_included(self, adapter, messages, tool):
        body = adapter.build_request_body(messages, "gpt-4o", False, [tool])
        assert "tools" in body
        assert body["tools"][0]["type"] == "function"
        assert body["tools"][0]["function"]["name"] == "get_weather"

    def test_kwargs_passed_through(self, adapter, messages):
        body = adapter.build_request_body(messages, "gpt-4o", False, None, temperature=0.5)
        assert body["temperature"] == 0.5

    def test_system_message_preserved(self, adapter, messages):
        body = adapter.build_request_body(messages, "gpt-4o", False, None)
        roles = [m["role"] for m in body["messages"]]
        assert "system" in roles


class TestParseResponse:
    def test_basic_response(self, adapter):
        data = {
            "choices": [
                {"message": {"content": "Hello!", "role": "assistant"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "gpt-4o",
            "id": "chatcmpl-123",
        }
        response = adapter.parse_response(data, "gpt-4o")
        assert response.content == "Hello!"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.model == "gpt-4o"
        assert response.metadata["finish_reason"] == "stop"

    def test_empty_content(self, adapter):
        data = {
            "choices": [
                {"message": {"content": None, "role": "assistant"}, "finish_reason": "tool_calls"}
            ],
            "usage": {},
        }
        response = adapter.parse_response(data, "gpt-4o")
        assert response.content == ""


class TestParseToolCalls:
    def test_returns_none_when_no_calls(self, adapter):
        data = {"choices": [{"message": {"content": "Hello", "role": "assistant"}}]}
        assert adapter.parse_tool_calls(data) is None

    def test_parses_tool_calls(self, adapter):
        data = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                            }
                        ]
                    }
                }
            ]
        }
        calls = adapter.parse_tool_calls(data)
        assert calls is not None
        assert len(calls) == 1
        assert calls[0].call_id == "call_abc"
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == '{"city": "NYC"}'


class TestBuildToolResultMessages:
    def test_appends_assistant_and_tool_messages(self, adapter):
        messages = [Message(role="user", content="What's the weather?")]
        tool_calls = [ToolCall(call_id="call_1", name="get_weather", arguments='{"city": "NYC"}')]
        results = ["Sunny, 72°F"]

        new_messages = adapter.build_tool_result_messages(messages, tool_calls, results)

        assert len(new_messages) == 3
        assert new_messages[0].role == "user"  # original
        assert new_messages[1].role == "assistant"  # tool call turn
        assert new_messages[2].role == "tool"  # result
        assert new_messages[2].tool_call_id == "call_1"
        assert new_messages[2].content == "Sunny, 72°F"

    def test_does_not_mutate_input(self, adapter):
        messages = [Message(role="user", content="Hi")]
        original_len = len(messages)
        tool_calls = [ToolCall(call_id="c1", name="fn", arguments="{}")]
        adapter.build_tool_result_messages(messages, tool_calls, ["result"])
        assert len(messages) == original_len


class TestParseSSEEvent:
    def test_content_delta(self, adapter):
        state = StreamState()
        event = json.dumps({"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]})
        chunk = adapter.parse_sse_event(event, state)
        assert chunk is not None
        assert chunk.delta == "Hello"
        assert state.accumulated_content == "Hello"

    def test_done_signal_returns_none(self, adapter):
        state = StreamState()
        assert adapter.parse_sse_event("[DONE]", state) is None

    def test_invalid_json_returns_none(self, adapter):
        state = StreamState()
        assert adapter.parse_sse_event("not-json", state) is None

    def test_tool_call_accumulation(self, adapter):
        state = StreamState()
        # First chunk: tool call start
        event1 = json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "get_weather", "arguments": '{"ci'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )
        adapter.parse_sse_event(event1, state)

        # Second chunk: argument continuation
        event2 = json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": 'ty": "NYC"}'}}]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )
        adapter.parse_sse_event(event2, state)

        assert state.tool_calls["0"]["arguments"] == '{"city": "NYC"}'

    def test_finish_reason_propagated(self, adapter):
        state = StreamState()
        event = json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        chunk = adapter.parse_sse_event(event, state)
        assert chunk is not None
        assert chunk.finish_reason == "stop"


class TestFinalizeStream:
    def test_returns_none_when_no_tool_calls(self, adapter):
        state = StreamState()
        assert adapter.finalize_stream(state) is None

    def test_converts_tool_calls(self, adapter):
        state = StreamState()
        state.tool_calls["0"] = {
            "id": "call_1",
            "name": "get_weather",
            "arguments": '{"city": "NYC"}',
        }
        calls = adapter.finalize_stream(state)
        assert calls is not None
        assert len(calls) == 1
        assert calls[0].call_id == "call_1"
        assert calls[0].name == "get_weather"


class TestParseModelsResponse:
    def test_basic_parsing(self, adapter):
        data = {
            "data": [
                {"id": "gpt-4o", "owned_by": "openai", "created": 1700000000},
                {"id": "gpt-4o-mini", "owned_by": "openai"},
            ]
        }
        models = adapter.parse_models_response(data)
        assert len(models) == 2
        assert models[0].id == "gpt-4o"
        assert models[0].owned_by == "openai"
        assert models[1].id == "gpt-4o-mini"

    def test_empty_response(self, adapter):
        assert adapter.parse_models_response({}) == []


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------


class TestBuildEmbeddingRequest:
    def test_single_input(self, adapter):
        body = adapter.build_embedding_request("hello", "text-embedding-3-small")
        assert body == {"input": "hello", "model": "text-embedding-3-small"}

    def test_batch_input(self, adapter):
        body = adapter.build_embedding_request(["a", "b"], "text-embedding-3-small")
        assert body["input"] == ["a", "b"]
        assert body["model"] == "text-embedding-3-small"

    def test_kwargs_passed_through(self, adapter):
        body = adapter.build_embedding_request(
            "hello", "model", dimensions=512, encoding_format="float"
        )
        assert body["dimensions"] == 512
        assert body["encoding_format"] == "float"


class TestParseEmbeddingResponse:
    def test_single_embedding(self, adapter):
        data = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        result = adapter.parse_embedding_response(data, "text-embedding-3-small", batch=False)
        assert isinstance(result, EmbeddingResponse)
        assert result.embedding == [0.1, 0.2, 0.3]
        assert result.embeddings == []
        assert result.model == "text-embedding-3-small"
        assert result.usage.prompt_tokens == 5
        assert result.usage.total_tokens == 5

    def test_batch_embeddings(self, adapter):
        data = {
            "data": [
                {"embedding": [0.1, 0.2], "index": 0},
                {"embedding": [0.3, 0.4], "index": 1},
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }
        result = adapter.parse_embedding_response(data, "text-embedding-3-small", batch=True)
        assert result.embedding is None
        assert len(result.embeddings) == 2
        assert result.embeddings[0] == [0.1, 0.2]
        assert result.embeddings[1] == [0.3, 0.4]

    def test_batch_sorts_by_index(self, adapter):
        data = {
            "data": [
                {"embedding": [0.3, 0.4], "index": 1},
                {"embedding": [0.1, 0.2], "index": 0},
            ],
            "model": "model",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        result = adapter.parse_embedding_response(data, "model", batch=True)
        assert result.embeddings[0] == [0.1, 0.2]
        assert result.embeddings[1] == [0.3, 0.4]

    def test_single_item_list_uses_batch_flag(self, adapter):
        """When batch=True but only one embedding returned, populate embeddings not embedding."""
        data = {
            "data": [{"embedding": [0.1, 0.2], "index": 0}],
            "model": "model",
            "usage": {},
        }
        result = adapter.parse_embedding_response(data, "model", batch=True)
        assert result.embedding is None
        assert result.embeddings == [[0.1, 0.2]]

    def test_missing_usage_defaults(self, adapter):
        data = {"data": [{"embedding": [0.5], "index": 0}], "model": "model"}
        result = adapter.parse_embedding_response(data, "model", batch=False)
        assert result.usage.prompt_tokens == 0
        assert result.usage.total_tokens == 0
