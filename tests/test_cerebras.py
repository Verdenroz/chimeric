from collections.abc import AsyncGenerator
import os
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

from cerebras.cloud.sdk.types.chat.chat_completion import (
    ChatChunkResponse,
    ChatCompletionResponse,
    ChatCompletionResponseChoiceMessageToolCall,
)
import pytest

from chimeric import Chimeric
from chimeric.exceptions import ToolRegistrationError
import chimeric.providers.cerebras.client as client_module
from chimeric.providers.cerebras.client import CerebrasClient
from chimeric.types import (
    ChimericCompletionResponse,
    ModelSummary,
    Tool,
    ToolParameters,
)


@pytest.fixture(scope="module")
def chimeric_cerebras():
    """Create a Chimeric instance configured for Cerebras."""
    return Chimeric(
        cerebras_api_key=os.getenv("CEREBRAS_API_KEY", "test_key"),
        timeout=120,
        max_retries=2,
    )


@pytest.fixture(scope="module")
def chimeric_cerebras_client(chimeric_cerebras) -> CerebrasClient:
    """Get the CerebrasClient from the Chimeric wrapper."""
    return cast("CerebrasClient", chimeric_cerebras.get_provider_client("cerebras"))


@pytest.fixture(autouse=True)
def patch_cerebras_imports(monkeypatch):
    """Stub out actual Cerebras classes to prevent network calls."""

    def create_cerebras_mock(api_key: str, **kw: Any) -> SimpleNamespace:
        return SimpleNamespace(api_key=api_key, **kw)

    def create_async_cerebras_mock(api_key: str, **kw: Any) -> SimpleNamespace:
        return SimpleNamespace(api_key=api_key, **kw)

    monkeypatch.setattr(client_module, "Cerebras", create_cerebras_mock)
    monkeypatch.setattr(client_module, "AsyncCerebras", create_async_cerebras_mock)
    return


# Helper functions for creating mock objects
def create_mock_chat_completion_response(
    content: str = "hello", model: str = "llama-3.1-8b", tool_calls: Any = None
) -> Mock:
    """Create a properly mocked ChatCompletionResponse."""
    mock_message = Mock()
    mock_message.role = "assistant"
    mock_message.content = content
    mock_message.tool_calls = tool_calls

    mock_choice = Mock()
    mock_choice.index = 0
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_usage = Mock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_usage.total_tokens = 15

    mock_response = Mock(spec=ChatCompletionResponse)
    mock_response.id = "chatcmpl-123"
    mock_response.object = "chat.completion"
    mock_response.created = 1234567890
    mock_response.model = model
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_response.model_dump.return_value = {"dumped": True}

    return mock_response


def create_mock_chat_chunk_response(
    content: str = "hello",
    finish_reason: str | None = None,
    choices: list[Any] | None = None,
    tool_calls: list[Any] | None = None,
) -> Mock:
    """Create a properly mocked ChatChunkResponse."""
    if choices is not None:
        mock_choices = choices
    else:
        mock_delta = Mock()
        mock_delta.role = "assistant"
        mock_delta.content = content
        mock_delta.tool_calls = tool_calls

        mock_choice = Mock()
        mock_choice.index = 0
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = finish_reason

        mock_choices = [mock_choice]

    mock_chunk = Mock(spec=ChatChunkResponse)
    mock_chunk.id = "chatcmpl-123"
    mock_chunk.object = "chat.completion.chunk"
    mock_chunk.created = 1234567890
    mock_chunk.model = "llama-3.1-8b"
    mock_chunk.choices = mock_choices
    mock_chunk.model_dump.return_value = {"dumped": True, "chunk": True}

    return mock_chunk


def create_mock_tool_call(
    tool_id: str = "call_123", name: str = "test_tool", arguments: str = '{"param1": "value1"}'
) -> Mock:
    """Create a properly mocked ChatCompletionResponseChoiceMessageToolCall."""
    mock_function = Mock()
    mock_function.name = name
    mock_function.arguments = arguments

    mock_tool_call = Mock(spec=ChatCompletionResponseChoiceMessageToolCall)
    mock_tool_call.id = tool_id
    mock_tool_call.function = mock_function

    return mock_tool_call


class MockModel:
    """Mock implementation of Cerebras Model."""

    def __init__(self, model_id: str = "llama-3.1-8b") -> None:
        self.id = model_id
        self.object = "model"
        self.created = 1234567890
        self.owned_by = "cerebras"


# noinspection PyUnusedLocal,PyTypeChecker
class TestCerebrasClient:
    """Test suite for the CerebrasClient implementation."""

    def test_initialization_and_basics(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test client initialization, capabilities, and basic methods."""
        # Verify initialization
        assert chimeric_cerebras_client.api_key == "test_key"
        assert chimeric_cerebras_client._provider_name == "Cerebras"
        assert chimeric_cerebras_client.client.api_key == "test_key"
        assert chimeric_cerebras_client.async_client.api_key == "test_key"

        # Verify generic types
        types = chimeric_cerebras_client._get_generic_types()
        assert "sync" in types
        assert "async" in types

        # Verify capabilities
        capabilities = chimeric_cerebras_client._get_capabilities()
        assert not capabilities.multimodal
        assert capabilities.streaming
        assert capabilities.tools
        assert not capabilities.agents
        assert not capabilities.files

        # Verify file upload raises NotImplementedError
        with pytest.raises(NotImplementedError, match="Cerebras does not support file uploads"):
            chimeric_cerebras_client._upload_file()

    def test_list_models_variations(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test model listing with different scenarios."""
        # Test with normal models
        mock_models = [MockModel("llama-3.1-8b"), MockModel("llama-3.1-70b")]
        mock_response = SimpleNamespace(data=mock_models)
        chimeric_cerebras_client.client.models = SimpleNamespace(list=lambda: mock_response)

        models = chimeric_cerebras_client._list_models_impl()
        assert len(models) == 2
        assert all(isinstance(m, ModelSummary) for m in models)
        assert models[0].id == "llama-3.1-8b"

        # Test with models missing attributes
        class MinimalModel:
            def __init__(self, model_id: str):
                self.id = model_id

        minimal_models = [MinimalModel("model1"), MinimalModel("model2")]
        chimeric_cerebras_client.client.models.list = lambda: SimpleNamespace(data=minimal_models)

        models = chimeric_cerebras_client._list_models_impl()
        assert all(m.owned_by == "cerebras" for m in models)
        assert all(m.created_at is None for m in models)

    def test_tool_encoding_scenarios(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test various tool encoding scenarios."""
        # Test None/empty tools
        assert chimeric_cerebras_client._encode_tools(None) is None
        assert chimeric_cerebras_client._encode_tools([]) is None

        # Test Tool with ToolParameters
        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters=ToolParameters(
                type="object",
                properties={"param1": {"type": "string"}},
                required=["param1"],
                strict=True,
            ),
        )
        encoded = chimeric_cerebras_client._encode_tools([tool])
        assert encoded is not None
        encoded_list = list(encoded)
        assert encoded_list[0]["function"]["strict"] is True
        assert "strict" not in encoded_list[0]["function"]["parameters"]

        # Test Tool with dict parameters - use explicit cast
        from typing import cast

        tool_dict = Tool(
            name="test_tool",
            description="A test tool",
            parameters=cast(
                "ToolParameters", {"type": "object", "properties": {"x": {"type": "int"}}}
            ),
        )
        encoded = chimeric_cerebras_client._encode_tools([tool_dict])
        assert encoded is not None
        encoded_list = list(encoded)
        assert "type" in encoded_list[0]["function"]["parameters"]

        # Test already formatted tools
        pre_formatted = [{"type": "function", "function": {"name": "existing"}}]
        result = chimeric_cerebras_client._encode_tools(pre_formatted)
        assert result is not None
        assert list(result) == pre_formatted

    def test_chimeric_response_creation_edge_cases(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test creating ChimericCompletionResponse with edge cases."""
        # Test normal response
        mock_response = create_mock_chat_completion_response("Hello!")
        response = chimeric_cerebras_client._create_chimeric_response(mock_response, [])
        assert response.common.content == "Hello!"

        # Test with tool calls
        tool_calls = [{"call_id": "123", "name": "test", "result": "success"}]
        response = chimeric_cerebras_client._create_chimeric_response(mock_response, tool_calls)
        assert response.common.metadata is not None
        assert response.common.metadata["tool_calls"] == tool_calls

        # Test empty choices
        mock_response.choices = []
        response = chimeric_cerebras_client._create_chimeric_response(mock_response, [])
        assert response.common.content == ""

        # Test no message
        mock_response = create_mock_chat_completion_response()
        mock_response.choices[0].message = None
        response = chimeric_cerebras_client._create_chimeric_response(mock_response, [])
        assert response.common.content == ""

        # Test None content
        mock_response = create_mock_chat_completion_response()
        mock_response.choices[0].message.content = None
        response = chimeric_cerebras_client._create_chimeric_response(mock_response, [])
        assert response.common.content == ""

        # Test no/None usage
        mock_response = create_mock_chat_completion_response()
        mock_response.usage = None
        response = chimeric_cerebras_client._create_chimeric_response(mock_response, [])
        assert response.common.usage.total_tokens == 0

        mock_response = create_mock_chat_completion_response()
        mock_response.usage.prompt_tokens = None
        mock_response.usage.completion_tokens = None
        mock_response.usage.total_tokens = None
        response = chimeric_cerebras_client._create_chimeric_response(mock_response, [])
        assert response.common.usage.total_tokens == 0

    def test_sync_chat_completion_variations(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test synchronous chat completion with different scenarios."""
        # Test basic completion
        mock_response = create_mock_chat_completion_response("Hello, sync!")
        chimeric_cerebras_client.client.chat = SimpleNamespace(
            completions=SimpleNamespace(create=Mock(return_value=mock_response))
        )

        result = chimeric_cerebras_client._chat_completion_impl(
            messages=[{"role": "user", "content": "Hello"}],
            model="llama-3.1-8b",
            stream=False,
        )
        assert result.common.content == "Hello, sync!"

        # Test message as string
        result = chimeric_cerebras_client._chat_completion_impl(
            messages="Single message string",
            model="llama-3.1-8b",
            stream=False,
        )
        assert isinstance(result, ChimericCompletionResponse)

        # Test with tools but no tool calls
        tools = [{"type": "function", "function": {"name": "test"}}]
        result = chimeric_cerebras_client._chat_completion_impl(
            messages=[{"role": "user", "content": "Hello"}],
            model="llama-3.1-8b",
            stream=False,
            tools=tools,
        )
        assert result.common.content == "Hello, sync!"

    async def test_async_chat_completion_variations(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test asynchronous chat completion with different scenarios."""
        mock_response = create_mock_chat_completion_response("Hello, async!")
        chimeric_cerebras_client.async_client.chat = SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=mock_response))
        )

        result = await chimeric_cerebras_client._achat_completion_impl(
            messages=[{"role": "user", "content": "Hello"}],
            model="llama-3.1-8b",
            stream=False,
        )
        assert result.common.content == "Hello, async!"

        # Test message as string
        result = await chimeric_cerebras_client._achat_completion_impl(
            messages="Single async message string",
            model="llama-3.1-8b",
            stream=False,
        )
        assert result.common.content == "Hello, async!"

    def test_streaming_edge_cases(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test streaming with various edge cases."""
        # Test empty content chunks
        mock_chunks = [
            create_mock_chat_chunk_response(""),
            create_mock_chat_chunk_response("Hello"),
            create_mock_chat_chunk_response("", finish_reason="stop"),
        ]

        def mock_generator():
            yield from mock_chunks

        result = chimeric_cerebras_client._stream(mock_generator())
        chunks = list(result)
        assert len(chunks) == 1
        assert chunks[0].common.content == "Hello"

        # Test empty choices
        mock_chunks = [
            create_mock_chat_chunk_response("Hello", choices=[]),
            create_mock_chat_chunk_response("World", choices=[]),
        ]

        def mock_generator2():
            yield from mock_chunks

        result = chimeric_cerebras_client._stream(mock_generator2())
        chunks = list(result)
        assert len(chunks) == 0

        # Test no delta content
        mock_chunks = [
            create_mock_chat_chunk_response(
                "",
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(role="assistant", content=None),
                        finish_reason=None,
                    )
                ],
            ),
            create_mock_chat_chunk_response("Hello"),
        ]

        def mock_generator3():
            yield from mock_chunks

        result = chimeric_cerebras_client._stream(mock_generator3())
        chunks = list(result)
        assert len(chunks) == 1

    def test_sync_streaming_with_varied_chunks(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test synchronous streaming with a variety of chunk structures."""
        # Create chunks that test all branches
        mock_chunks = [
            # Chunk with choices but no content (false on second condition)
            create_mock_chat_chunk_response(
                "",
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(role="assistant", content=None),
                        finish_reason=None,
                    )
                ],
            ),
            # Chunk with no choices (false on first condition)
            create_mock_chat_chunk_response("", choices=[]),
            # Chunk with choices and content (both conditions true)
            create_mock_chat_chunk_response("Hello"),
            # Another chunk with no choices
            create_mock_chat_chunk_response("", choices=None),
            # Final chunk with content
            create_mock_chat_chunk_response(" World", finish_reason="stop"),
        ]

        def mock_generator4():
            yield from mock_chunks

        result = chimeric_cerebras_client._stream(mock_generator4())
        chunks = list(result)

        # Should only get 2 chunks (the ones with actual content)
        assert len(chunks) == 2
        assert chunks[0].common.delta == "Hello"
        assert chunks[1].common.delta == " World"
        assert chunks[1].common.content == "Hello World"

    async def test_async_streaming_with_varied_chunks(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test asynchronous streaming with a variety of chunk structures."""

        async def mock_stream():
            # Chunk with choices but no content (false on second condition)
            yield create_mock_chat_chunk_response(
                "",
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(role="assistant", content=None),
                        finish_reason=None,
                    )
                ],
            )
            # Chunk with no choices (false on first condition)
            yield create_mock_chat_chunk_response("", choices=[])
            # Chunk with choices and content (both conditions true)
            yield create_mock_chat_chunk_response("Async")
            # Another chunk with no choices
            yield create_mock_chat_chunk_response("", choices=None)
            # Final chunk with content
            yield create_mock_chat_chunk_response(" Hello", finish_reason="stop")

        result = chimeric_cerebras_client._astream(mock_stream())
        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        # Should only get 2 chunks (the ones with actual content)
        assert len(chunks) == 2
        assert chunks[0].common.delta == "Async"
        assert chunks[1].common.delta == " Hello"
        assert chunks[1].common.content == "Async Hello"

    def test_tool_execution_scenarios(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test various tool execution scenarios."""

        # Test normal tool execution
        def mock_tool(param1: str) -> str:
            return f"Result: {param1}"

        chimeric_cerebras_client.tool_manager.register(mock_tool, "test_tool")
        mock_call = create_mock_tool_call("call_123", "test_tool", '{"param1": "value1"}')

        result = chimeric_cerebras_client._process_function_call(mock_call)
        assert result["result"] == "Result: value1"

        # Test tool not found
        mock_call = create_mock_tool_call("call_123", "nonexistent", "{}")
        with pytest.raises(ToolRegistrationError):
            chimeric_cerebras_client._process_function_call(mock_call)

        # Test non-callable tool
        bad_tool = Tool(
            name="bad_tool", description="bad tool", parameters=cast("ToolParameters", {})
        )
        bad_tool.function = "not_callable"
        chimeric_cerebras_client.tool_manager.tools["bad_tool"] = bad_tool
        mock_call = create_mock_tool_call("call_123", "bad_tool", "{}")
        with pytest.raises(ToolRegistrationError):
            chimeric_cerebras_client._process_function_call(mock_call)

        # Test tool with error
        def error_tool(x: int) -> int:
            raise ValueError("Tool error")

        chimeric_cerebras_client.tool_manager.register(error_tool, "error_tool")
        mock_call = create_mock_tool_call("call_123", "error_tool", '{"x": 5}')

        result = chimeric_cerebras_client._process_function_call(mock_call)
        assert "error" in result
        assert "Tool error" in result["error"]

        # Test invalid JSON arguments
        mock_call = create_mock_tool_call("call_123", "test_tool", "invalid json")
        result = chimeric_cerebras_client._process_function_call(mock_call)
        assert "error" in result

    def test_tool_execution_loops(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test tool execution loops for non-streaming responses."""

        # Register tools
        def tool1(x: int) -> int:
            return x * 2

        def tool2(x: int) -> int:
            return x + 10

        chimeric_cerebras_client.tool_manager.register(tool1, "multiply")
        chimeric_cerebras_client.tool_manager.register(tool2, "add_ten")

        # Mock responses
        mock_response_with_tools = create_mock_chat_completion_response("Using tools")
        mock_response_with_tools.choices[0].message.tool_calls = [
            create_mock_tool_call("call_1", "multiply", '{"x": 5}'),
            create_mock_tool_call("call_2", "add_ten", '{"x": 3}'),
        ]

        mock_final_response = create_mock_chat_completion_response("Done")
        mock_final_response.choices[0].message.tool_calls = None

        messages = [{"role": "user", "content": "Calculate"}]
        tools = [{"type": "function", "function": {"name": "multiply"}}]

        # Test without tool calls
        _, tool_calls = chimeric_cerebras_client._handle_tool_execution_loop(
            mock_final_response, messages.copy(), "model", tools
        )
        assert tool_calls == []

        # Test with empty choices
        mock_empty = create_mock_chat_completion_response()
        mock_empty.choices = []
        _, tool_calls = chimeric_cerebras_client._handle_tool_execution_loop(
            mock_empty, messages.copy(), "model", tools
        )
        assert tool_calls == []

        # Test with tool calls
        call_count = 0

        def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_final_response

        chimeric_cerebras_client.client.chat = SimpleNamespace(
            completions=SimpleNamespace(create=mock_create)
        )

        _, tool_calls = chimeric_cerebras_client._handle_tool_execution_loop(
            mock_response_with_tools, messages.copy(), "model", tools
        )
        assert len(tool_calls) == 2
        assert any(tc["result"] == "10" for tc in tool_calls)
        assert any(tc["result"] == "13" for tc in tool_calls)

    async def test_async_tool_execution_loops(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test async tool execution loops."""

        def tool1(x: int) -> int:
            return x * 3

        chimeric_cerebras_client.tool_manager.register(tool1, "triple")

        mock_response_with_tools = create_mock_chat_completion_response("Using tools")
        mock_response_with_tools.choices[0].message.tool_calls = [
            create_mock_tool_call("call_1", "triple", '{"x": 4}'),
        ]

        mock_final_response = create_mock_chat_completion_response("Done")

        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_final_response

        chimeric_cerebras_client.async_client.chat = SimpleNamespace(
            completions=SimpleNamespace(create=mock_create)
        )

        messages = [{"role": "user", "content": "Calculate"}]
        tools = [{"type": "function", "function": {"name": "triple"}}]

        _, tool_calls = await chimeric_cerebras_client._handle_async_tool_execution_loop(
            mock_response_with_tools, messages.copy(), "model", tools
        )
        assert len(tool_calls) == 1
        assert tool_calls[0]["result"] == "12"

    async def test_async_chat_completion_with_tools_non_streaming(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test async chat completion with tools and stream=False to hit missing coverage."""

        def tool_multiply(x: int) -> int:
            return x * 5

        chimeric_cerebras_client.tool_manager.register(tool_multiply, "multiply_by_five")

        # Mock response with tool calls
        mock_response_with_tools = create_mock_chat_completion_response("I'll use the tool")
        mock_response_with_tools.choices[0].message.tool_calls = [
            create_mock_tool_call("call_1", "multiply_by_five", '{"x": 6}'),
        ]

        # Mock final response after tool execution
        mock_final_response = create_mock_chat_completion_response("Result is 30")

        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_response_with_tools
            return mock_final_response

        chimeric_cerebras_client.async_client.chat = SimpleNamespace(
            completions=SimpleNamespace(create=mock_create)
        )

        messages = [{"role": "user", "content": "Multiply 6 by 5"}]
        tools = [{"type": "function", "function": {"name": "multiply_by_five"}}]

        # This should hit lines 781-784 in the async implementation
        result = await chimeric_cerebras_client._achat_completion_impl(
            messages=messages,
            model="llama-3.1-8b",
            stream=False,
            tools=tools,
        )

        assert result.common.content == "Result is 30"
        assert call_count == 2  # Should have made 2 calls

    def test_streaming_with_tools(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test streaming with tool execution."""

        def mock_tool(param1: str) -> str:
            return f"Result: {param1}"

        chimeric_cerebras_client.tool_manager.register(mock_tool, "stream_tool")

        # Initial chunks with tool call
        mock_tool_call = create_mock_tool_call("call_123", "stream_tool", '{"param1": "test"}')
        initial_chunks = [
            create_mock_chat_chunk_response("Using", tool_calls=[mock_tool_call]),
            create_mock_chat_chunk_response(" tool", finish_reason="tool_calls"),
        ]

        # Continuation chunks
        continuation_chunks = [
            create_mock_chat_chunk_response("Tool result"),
            create_mock_chat_chunk_response(" processed", finish_reason="stop"),
        ]

        call_count = 0

        def mock_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            yield from (initial_chunks if call_count == 1 else continuation_chunks)

        chimeric_cerebras_client.client.chat = SimpleNamespace(
            completions=SimpleNamespace(create=mock_stream)
        )

        messages = [{"role": "user", "content": "Test"}]
        tools = [{"type": "function", "function": {"name": "stream_tool"}}]

        # Test sync streaming
        result = chimeric_cerebras_client._chat_completion_impl(
            messages=messages, model="model", stream=True, tools=tools
        )
        chunks = list(result)
        assert len(chunks) >= 2
        assert call_count == 2

        # Test without tools
        call_count = 0

        def continuation_generator():
            yield from continuation_chunks

        result = chimeric_cerebras_client._process_stream_with_tools_sync(
            continuation_generator(),
            messages,
            "model",
            None,
        )
        chunks = list(result)
        assert len(chunks) == 2

    def test_streaming_without_tools(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test streaming without tools"""
        mock_chunks = [
            create_mock_chat_chunk_response("Hello"),
            create_mock_chat_chunk_response(" World", finish_reason="stop"),
        ]

        def mock_stream(*args, **kwargs):
            yield from mock_chunks

        chimeric_cerebras_client.client.chat = SimpleNamespace(
            completions=SimpleNamespace(create=mock_stream)
        )

        # Test sync streaming without tools
        result = chimeric_cerebras_client._chat_completion_impl(
            messages=[{"role": "user", "content": "Hello"}],
            model="llama-3.1-8b",
            stream=True,
            tools=None,
        )
        chunks = list(result)
        assert len(chunks) == 2
        assert chunks[1].common.content == "Hello World"

    async def test_async_streaming_without_tools(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test async streaming without tools"""
        mock_chunks = [
            create_mock_chat_chunk_response("Async"),
            create_mock_chat_chunk_response(" Hello", finish_reason="stop"),
        ]

        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk

        async def mock_create(*args, **kwargs):
            return mock_stream()

        chimeric_cerebras_client.async_client.chat = SimpleNamespace(
            completions=SimpleNamespace(create=mock_create)
        )

        # Test async streaming without tools
        result = await chimeric_cerebras_client._achat_completion_impl(
            messages=[{"role": "user", "content": "Hello"}],
            model="llama-3.1-8b",
            stream=True,
            tools=None,
        )
        chunks = []
        # When stream=True, result should be an AsyncGenerator
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                chunks.append(chunk)
        else:
            # Handle case where result is not async iterable
            chunks = [result]
        assert len(chunks) == 2
        assert chunks[1].common.content == "Async Hello"

    async def test_async_streaming_with_tools_full_loop(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test async streaming with tools to ensure full loop coverage."""

        def mock_tool(param1: str) -> str:
            return f"Processed: {param1}"

        chimeric_cerebras_client.tool_manager.register(mock_tool, "async_full_tool")

        mock_tool_call = create_mock_tool_call("call_123", "async_full_tool", '{"param1": "test"}')

        initial_chunks = [
            create_mock_chat_chunk_response("I'll use", tool_calls=[mock_tool_call]),
            create_mock_chat_chunk_response(" the tool", finish_reason="tool_calls"),
        ]

        continuation_chunks = [
            create_mock_chat_chunk_response("The tool"),
            create_mock_chat_chunk_response(" returned"),
            create_mock_chat_chunk_response(" the result:", finish_reason=None),
            create_mock_chat_chunk_response(" Processed: test", finish_reason="stop"),
        ]

        call_count = 0

        async def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:

                async def first_stream():
                    for chunk in initial_chunks:
                        yield chunk

                return first_stream()

            async def continuation_stream():
                for chunk in continuation_chunks:
                    yield chunk

            return continuation_stream()

        chimeric_cerebras_client.async_client.chat = SimpleNamespace(
            completions=SimpleNamespace(create=mock_create)
        )

        messages = [{"role": "user", "content": "Process this"}]
        tools = [
            {"type": "function", "function": {"name": "async_full_tool", "description": "test"}}
        ]

        result = await chimeric_cerebras_client._achat_completion_impl(
            messages=messages,
            model="llama-3.1-8b",
            stream=True,
            tools=tools,
        )

        chunks = []
        # When stream=True, result should be an AsyncGenerator
        if isinstance(result, AsyncGenerator):
            async for chunk in result:
                chunks.append(chunk)
        else:
            # Handle case where result is not async iterable
            chunks = [result]

        assert len(chunks) >= 6 if isinstance(result, AsyncGenerator) else len(chunks) >= 1
        assert call_count == 2

        content_parts = [chunk.common.delta for chunk in chunks if chunk.common.delta]
        assert "I'll use" in content_parts
        assert " the tool" in content_parts
        assert "The tool" in content_parts
        assert " Processed: test" in content_parts

        assert chunks[-1].common.finish_reason == "stop"

    async def test_async_streaming_without_tools_full_loop(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test async streaming without tools to ensure full loop coverage"""
        mock_chunks = [
            create_mock_chat_chunk_response("Hello"),
            create_mock_chat_chunk_response(" async", finish_reason="stop"),
        ]

        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk

        result = chimeric_cerebras_client._process_stream_with_tools_async(
            mock_stream(),
            [{"role": "user", "content": "Hi"}],
            "model",
            None,
        )

        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[1].common.content == "Hello async"

    def test_direct_stream_tool_execution_methods(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test direct execution of stream tool handling methods."""

        def mock_tool(param1: str) -> str:
            return f"Direct: {param1}"

        chimeric_cerebras_client.tool_manager.register(mock_tool, "direct_tool")

        mock_tool_call = create_mock_tool_call("call_123", "direct_tool", '{"param1": "test"}')

        continuation_chunks = [
            create_mock_chat_chunk_response("Final"),
            create_mock_chat_chunk_response(" result", finish_reason="stop"),
        ]

        def mock_continuation(*args, **kwargs):
            yield from continuation_chunks

        chimeric_cerebras_client.client.chat = SimpleNamespace(
            completions=SimpleNamespace(create=mock_continuation)
        )

        result = chimeric_cerebras_client._handle_stream_tool_execution_sync(
            [mock_tool_call], [{"role": "user", "content": "Test"}], "model", None, "Initial"
        )
        chunks = list(result)
        assert len(chunks) >= 1

    async def test_direct_async_stream_tool_execution(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test direct async stream tool execution method."""

        def mock_tool(param1: str) -> str:
            return f"Async Direct: {param1}"

        chimeric_cerebras_client.tool_manager.register(mock_tool, "async_direct_tool")

        mock_tool_call = create_mock_tool_call(
            "call_123", "async_direct_tool", '{"param1": "test"}'
        )

        continuation_chunks = [
            create_mock_chat_chunk_response("Async"),
            create_mock_chat_chunk_response(" final"),
            create_mock_chat_chunk_response(" result", finish_reason="stop"),
        ]

        async def mock_continuation(*args, **kwargs):
            for chunk in continuation_chunks:
                yield chunk

        chimeric_cerebras_client.async_client.chat = SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=mock_continuation()))
        )

        result = chimeric_cerebras_client._handle_stream_tool_execution_async(
            [mock_tool_call],
            [{"role": "user", "content": "Test async"}],
            "model",
            None,
            "Initial async",
        )

        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].common.delta == "Async"
        assert chunks[1].common.delta == " final"
        assert chunks[2].common.delta == " result"

    def test_sync_stream_processing_with_tools_and_varied_chunks(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test synchronous stream processing with tools handles a variety of chunk types."""

        def mock_tool(x: int) -> int:
            return x * 2

        chimeric_cerebras_client.tool_manager.register(mock_tool, "coverage_tool")

        mock_tool_call = create_mock_tool_call("call_123", "coverage_tool", '{"x": 5}')

        # Create a stream that tests all branches
        mock_chunks = [
            # Chunk with no choices (false on first condition)
            create_mock_chat_chunk_response("", choices=[]),
            # Chunk with choices but no content (false on second condition)
            create_mock_chat_chunk_response(
                "",
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(role="assistant", content=None, tool_calls=None),
                        finish_reason=None,
                    )
                ],
            ),
            # Chunk with content
            create_mock_chat_chunk_response("Processing"),
            # Chunk with tool calls
            create_mock_chat_chunk_response(" now", tool_calls=[mock_tool_call]),
            # Final chunk with finish reason and tool calls
            create_mock_chat_chunk_response("", finish_reason="tool_calls", tool_calls=[]),
        ]

        # Mock the continuation
        continuation_chunks = [
            create_mock_chat_chunk_response("Tool executed"),
            create_mock_chat_chunk_response(" successfully", finish_reason="stop"),
        ]

        def mock_create(*args, **kwargs):
            yield from continuation_chunks

        chimeric_cerebras_client.client.chat = SimpleNamespace(
            completions=SimpleNamespace(create=mock_create)
        )

        def sync_generator():
            yield from mock_chunks

        result = chimeric_cerebras_client._process_stream_with_tools_sync(
            sync_generator(),
            [{"role": "user", "content": "Test"}],
            "model",
            [{"type": "function", "function": {"name": "coverage_tool"}}],
        )

        chunks = list(result)
        # Should get chunks from initial stream + continuation
        assert len(chunks) >= 4

    async def test_async_stream_processing_with_tools_and_varied_chunks(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test asynchronous stream processing with tools handles a variety of chunk types."""

        def mock_tool(x: int) -> int:
            return x * 3

        chimeric_cerebras_client.tool_manager.register(mock_tool, "async_coverage_tool")

        mock_tool_call = create_mock_tool_call("call_456", "async_coverage_tool", '{"x": 7}')

        # Create a stream that tests all branches
        async def mock_stream():
            # Chunk with no choices (false on first condition)
            yield create_mock_chat_chunk_response("", choices=[])
            # Chunk with choices but no content (false on second condition)
            yield create_mock_chat_chunk_response(
                "",
                choices=[
                    SimpleNamespace(
                        index=0,
                        delta=SimpleNamespace(role="assistant", content=None, tool_calls=None),
                        finish_reason=None,
                    )
                ],
            )
            # Chunk with content
            yield create_mock_chat_chunk_response("Async processing")
            # Chunk with tool calls
            yield create_mock_chat_chunk_response(" in progress", tool_calls=[mock_tool_call])
            # Final chunk with finish reason and tool calls
            yield create_mock_chat_chunk_response("", finish_reason="tool_calls", tool_calls=[])

        # Mock the continuation
        async def mock_continuation():
            yield create_mock_chat_chunk_response("Async tool")
            yield create_mock_chat_chunk_response(" completed", finish_reason="stop")

        async def mock_create(*args, **kwargs):
            return mock_continuation()

        chimeric_cerebras_client.async_client.chat = SimpleNamespace(
            completions=SimpleNamespace(create=mock_create)
        )

        result = chimeric_cerebras_client._process_stream_with_tools_async(
            mock_stream(),
            [{"role": "user", "content": "Test async"}],
            "model",
            [{"type": "function", "function": {"name": "async_coverage_tool"}}],
        )

        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        # Should get chunks from initial stream + continuation
        assert len(chunks) >= 4
