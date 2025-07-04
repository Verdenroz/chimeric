from collections.abc import AsyncGenerator, Generator
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
    ChimericStreamChunk,
    CompletionResponse,
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

    # Create mock implementations for sync and async SDK entrypoints
    def create_cerebras_mock(api_key: str, **kw: Any) -> SimpleNamespace:
        return SimpleNamespace(api_key=api_key, **kw)

    def create_async_cerebras_mock(api_key: str, **kw: Any) -> SimpleNamespace:
        return SimpleNamespace(api_key=api_key, **kw)

    monkeypatch.setattr(client_module, "Cerebras", create_cerebras_mock)
    monkeypatch.setattr(client_module, "AsyncCerebras", create_async_cerebras_mock)
    return


def create_mock_chat_completion_response(
    content: str = "hello", model: str = "llama-3.1-8b", tool_calls: Any = None
) -> Mock:
    """Create a properly mocked ChatCompletionResponse."""
    # Create mock choice message
    mock_message = Mock()
    mock_message.role = "assistant"
    mock_message.content = content
    mock_message.tool_calls = tool_calls

    # Create mock choice
    mock_choice = Mock()
    mock_choice.index = 0
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    # Create mock usage
    mock_usage = Mock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_usage.total_tokens = 15

    # Create the main mock response
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
    content: str = "hello", finish_reason: str | None = None, choices: list[Any] | None = None
) -> Mock:
    """Create a properly mocked ChatChunkResponse."""
    if choices is not None:
        mock_choices = choices
    else:
        # Create mock delta
        mock_delta = Mock()
        mock_delta.role = "assistant"
        mock_delta.content = content

        # Create mock choice
        mock_choice = Mock()
        mock_choice.index = 0
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = finish_reason

        mock_choices = [mock_choice]

    # Create the main mock chunk response
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
    # Create mock function
    mock_function = Mock()
    mock_function.name = name
    mock_function.arguments = arguments

    # Create the main mock tool call
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


# noinspection PyUnusedLocal
class TestCerebrasClient:
    """Test suite for the CerebrasClient implementation."""

    def test_initialization(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test that the client initializes correctly."""
        assert chimeric_cerebras_client.api_key == "test_key"
        assert chimeric_cerebras_client._provider_name == "Cerebras"
        assert chimeric_cerebras_client.client.api_key == "test_key"
        assert chimeric_cerebras_client.async_client.api_key == "test_key"

    def test_get_generic_types(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test that generic types are returned correctly."""
        types = chimeric_cerebras_client._get_generic_types()
        assert "sync" in types
        assert "async" in types

    def test_get_capabilities(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test that capabilities are returned correctly."""
        capabilities = chimeric_cerebras_client._get_capabilities()
        assert capabilities.multimodal is False
        assert capabilities.streaming is True
        assert capabilities.tools is True
        assert capabilities.agents is False
        assert capabilities.files is False

    def test_list_models(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test that models are listed correctly."""
        # Mock the models.list() call - CerebrasClient expects .data attribute
        mock_models = [MockModel("llama-3.1-8b"), MockModel("llama-3.1-70b")]
        mock_response = SimpleNamespace(data=mock_models)
        chimeric_cerebras_client.client.models = SimpleNamespace(list=lambda: mock_response)

        models = chimeric_cerebras_client._list_models_impl()
        assert len(models) == 2
        assert isinstance(models[0], ModelSummary)
        assert models[0].id == "llama-3.1-8b"
        assert models[0].owned_by == "cerebras"

    def test_encode_tools(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test that tools are encoded correctly."""
        # Test with None
        assert chimeric_cerebras_client._encode_tools(None) is None

        # Test with empty list
        assert chimeric_cerebras_client._encode_tools([]) is None

        # Test with Tool objects
        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters=ToolParameters(
                type="object",
                properties={"param1": {"type": "string"}},
                required=["param1"],
            ),
        )

        encoded = chimeric_cerebras_client._encode_tools([tool])
        assert isinstance(encoded, list)
        assert len(encoded) == 1
        assert encoded[0]["type"] == "function"
        assert encoded[0]["function"]["name"] == "test_tool"
        assert encoded[0]["function"]["strict"] is True

    def test_create_chimeric_response(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test that ChimericCompletionResponse is created correctly."""
        mock_response = create_mock_chat_completion_response("Hello, world!")
        tool_calls = []

        chimeric_response = chimeric_cerebras_client._create_chimeric_response(
            mock_response, tool_calls
        )

        assert isinstance(chimeric_response, ChimericCompletionResponse)
        assert chimeric_response.native == mock_response
        assert isinstance(chimeric_response.common, CompletionResponse)
        assert chimeric_response.common.content == "Hello, world!"
        assert chimeric_response.common.usage.prompt_tokens == 10
        assert chimeric_response.common.usage.completion_tokens == 5
        assert chimeric_response.common.usage.total_tokens == 15

    def test_chat_completion_sync(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test synchronous chat completion."""
        # Mock the chat.completions.create call
        mock_response = create_mock_chat_completion_response("Hello, sync!")
        chimeric_cerebras_client.client.chat = SimpleNamespace(
            completions=SimpleNamespace(create=Mock(return_value=mock_response))
        )

        messages = [{"role": "user", "content": "Hello"}]
        result = chimeric_cerebras_client._chat_completion_impl(
            messages=messages,
            model="llama-3.1-8b",
            stream=False,
        )

        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Hello, sync!"

        # Verify the mock was called with correct parameters
        chimeric_cerebras_client.client.chat.completions.create.assert_called_once()

    async def test_chat_completion_async(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test asynchronous chat completion."""
        # Mock the async chat.completions.create call
        mock_response = create_mock_chat_completion_response("Hello, async!")
        chimeric_cerebras_client.async_client.chat = SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=mock_response))
        )

        messages = [{"role": "user", "content": "Hello"}]
        result = await chimeric_cerebras_client._achat_completion_impl(
            messages=messages,
            model="llama-3.1-8b",
            stream=False,
        )

        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Hello, async!"

        # Verify the mock was called with correct parameters
        chimeric_cerebras_client.async_client.chat.completions.create.assert_called_once()

    def test_chat_completion_streaming(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test streaming chat completion."""
        # Mock the streaming response
        mock_chunks = [
            create_mock_chat_chunk_response("Hello"),
            create_mock_chat_chunk_response(" world"),
            create_mock_chat_chunk_response("!", finish_reason="stop"),
        ]

        def mock_stream():
            yield from mock_chunks

        chimeric_cerebras_client.client.chat = SimpleNamespace(
            completions=SimpleNamespace(create=Mock(return_value=mock_stream()))
        )

        messages = [{"role": "user", "content": "Hello"}]
        result = chimeric_cerebras_client._chat_completion_impl(
            messages=messages,
            model="llama-3.1-8b",
            stream=True,
        )

        assert isinstance(result, Generator)

        # Collect all chunks
        chunks = list(result)
        assert len(chunks) == 3

        # Check first chunk
        assert isinstance(chunks[0], ChimericStreamChunk)
        assert chunks[0].common.content == "Hello"
        assert chunks[0].common.delta == "Hello"

        # Check accumulated content
        assert chunks[2].common.content == "Hello world!"
        assert chunks[2].common.finish_reason == "stop"

    async def test_chat_completion_streaming_async(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test async streaming chat completion."""
        # Mock the async streaming response
        mock_chunks = [
            create_mock_chat_chunk_response("Hello"),
            create_mock_chat_chunk_response(" async"),
            create_mock_chat_chunk_response("!", finish_reason="stop"),
        ]

        async def mock_async_stream():
            for chunk in mock_chunks:
                yield chunk

        chimeric_cerebras_client.async_client.chat = SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=mock_async_stream()))
        )

        messages = [{"role": "user", "content": "Hello"}]
        result = await chimeric_cerebras_client._achat_completion_impl(
            messages=messages,
            model="llama-3.1-8b",
            stream=True,
        )

        assert isinstance(result, AsyncGenerator)

        # Collect all chunks
        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) == 3

        # Check first chunk
        assert isinstance(chunks[0], ChimericStreamChunk)
        assert chunks[0].common.content == "Hello"
        assert chunks[0].common.delta == "Hello"

        # Check accumulated content
        assert chunks[2].common.content == "Hello async!"
        assert chunks[2].common.finish_reason == "stop"

    def test_tool_handling_no_tools(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test that responses without tool calls are handled correctly."""
        mock_response = create_mock_chat_completion_response("No tools here")
        messages = [{"role": "user", "content": "Hello"}]

        tool_calls, updated_messages = chimeric_cerebras_client._handle_function_tool_calls(
            mock_response, messages
        )

        assert tool_calls == []
        assert updated_messages == messages

    def test_parallel_tool_calls_for_scout_models(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test that scout models automatically get parallel_tool_calls=False."""
        # Mock the chat.completions.create call
        mock_response = create_mock_chat_completion_response("Hello!")
        mock_create = Mock(return_value=mock_response)
        chimeric_cerebras_client.client.chat = SimpleNamespace(
            completions=SimpleNamespace(create=mock_create)
        )

        # Add a simple tool to trigger the parallel_tool_calls logic
        tools = [{"type": "function", "function": {"name": "test", "description": "test"}}]

        messages = [{"role": "user", "content": "Hello"}]
        chimeric_cerebras_client._chat_completion_impl(
            messages=messages,
            model="llama-4-scout-17b-16e-instruct",
            stream=False,
            tools=tools,
        )

        # Check that parallel_tool_calls was set to False
        call_args = mock_create.call_args
        assert call_args[1]["parallel_tool_calls"] is False

    def test_create_chimeric_response_with_tool_calls(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test creating response with tool calls metadata."""
        mock_response = create_mock_chat_completion_response("Hello!")
        tool_calls = [{"call_id": "123", "name": "test_tool", "result": "success"}]

        chimeric_response = chimeric_cerebras_client._create_chimeric_response(
            mock_response, tool_calls
        )

        assert isinstance(chimeric_response, ChimericCompletionResponse)
        assert chimeric_response.native == mock_response
        assert chimeric_response.common.metadata is not None
        assert chimeric_response.common.metadata["tool_calls"] == tool_calls

    def test_create_chimeric_response_with_empty_choices(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test creating response with no choices."""
        mock_response = create_mock_chat_completion_response("Hello!")
        mock_response.choices = []

        chimeric_response = chimeric_cerebras_client._create_chimeric_response(mock_response, [])
        assert chimeric_response.common.content == ""

    def test_create_chimeric_response_with_no_usage(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test creating response with no usage data."""
        mock_response = create_mock_chat_completion_response("Hello!")
        mock_response.usage = None

        chimeric_response = chimeric_cerebras_client._create_chimeric_response(mock_response, [])
        assert chimeric_response.common.usage.prompt_tokens == 0
        assert chimeric_response.common.usage.completion_tokens == 0
        assert chimeric_response.common.usage.total_tokens == 0

    def test_encode_tools_with_dict_tools(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test encoding tools that are already dictionaries."""
        tools = [{"type": "function", "function": {"name": "existing_tool"}}]

        encoded = chimeric_cerebras_client._encode_tools(tools)

        assert encoded == tools

    def test_process_function_call(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test processing a function call."""

        # Register a mock tool
        def mock_tool(param1: str) -> str:
            return f"Result: {param1}"

        chimeric_cerebras_client.tool_manager.register(mock_tool, "test_tool_1")

        # Create a mock tool call
        mock_call = create_mock_tool_call("call_123", "test_tool_1", '{"param1": "value1"}')

        result = chimeric_cerebras_client._process_function_call(mock_call)

        assert result["call_id"] == "call_123"
        assert result["name"] == "test_tool_1"
        assert result["arguments"] == '{"param1": "value1"}'
        assert result["result"] == "Result: value1"

    def test_process_function_call_not_callable(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test processing a function call with non-callable tool."""
        # Register a non-callable tool
        chimeric_cerebras_client.tool_manager.tools["bad_tool"] = SimpleNamespace(  # type: ignore[assignment]
            name="bad_tool", function="not_callable"
        )

        mock_call = create_mock_tool_call("call_123", "bad_tool", '{"param1": "value1"}')

        with pytest.raises(ToolRegistrationError):
            chimeric_cerebras_client._process_function_call(mock_call)

    def test_handle_function_tool_calls_with_calls(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test handling function tool calls with actual tool calls."""

        # Register a mock tool
        def mock_tool(param1: str) -> str:
            return f"Result: {param1}"

        chimeric_cerebras_client.tool_manager.register(mock_tool, "test_tool_2")

        # Create a mock response with tool calls
        mock_response = create_mock_chat_completion_response("Hello!")
        mock_response.choices[0].message.tool_calls = [
            create_mock_tool_call("call_123", "test_tool_2", '{"param1": "value1"}')
        ]

        messages = [{"role": "user", "content": "Hello"}]
        tool_calls, updated_messages = chimeric_cerebras_client._handle_function_tool_calls(
            mock_response, messages
        )

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "test_tool_2"
        assert len(updated_messages) == 3  # Original + assistant + tool result

    def test_handle_function_tool_calls_with_empty_choices(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test handling function tool calls with empty choices."""
        mock_response = create_mock_chat_completion_response("Hello!")
        mock_response.choices = []

        messages = [{"role": "user", "content": "Hello"}]
        tool_calls, updated_messages = chimeric_cerebras_client._handle_function_tool_calls(
            mock_response, messages
        )

        assert tool_calls == []
        assert updated_messages == messages

    def test_stream_with_empty_content(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test streaming with chunks that have no content."""
        mock_chunks = [
            create_mock_chat_chunk_response(""),  # Empty content
            create_mock_chat_chunk_response("Hello"),
            create_mock_chat_chunk_response("", finish_reason="stop"),  # Empty content with finish
        ]

        def mock_stream():
            yield from mock_chunks

        result = chimeric_cerebras_client._stream(mock_stream())
        chunks = list(result)

        # Should only yield chunks with content
        assert len(chunks) == 1
        assert chunks[0].common.content == "Hello"

    async def test_astream_with_empty_content(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test async streaming with chunks that have no content."""
        mock_chunks = [
            create_mock_chat_chunk_response(""),  # Empty content
            create_mock_chat_chunk_response("Hello"),
            create_mock_chat_chunk_response("", finish_reason="stop"),  # Empty content with finish
        ]

        async def mock_async_stream():
            for chunk in mock_chunks:
                yield chunk

        result = chimeric_cerebras_client._astream(mock_async_stream())
        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        # Should only yield chunks with content
        assert len(chunks) == 1
        assert chunks[0].common.content == "Hello"

    def test_chat_completion_with_tool_calls_and_retry(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test chat completion with tool calls that triggers a retry."""

        # Register a mock tool
        def mock_tool(param1: str) -> str:
            return f"Result: {param1}"

        chimeric_cerebras_client.tool_manager.register(mock_tool, "test_tool_3")

        # First response with tool calls
        mock_response_with_tools = create_mock_chat_completion_response("I'll use a tool")
        mock_response_with_tools.choices[0].message.tool_calls = [
            create_mock_tool_call("call_123", "test_tool_3", '{"param1": "value1"}')
        ]

        # Second response after tool execution
        mock_final_response = create_mock_chat_completion_response("Tool result processed")

        # Mock the create method to return different responses
        call_count = 0

        def mock_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_response_with_tools
            return mock_final_response

        chimeric_cerebras_client.client.chat = SimpleNamespace(
            completions=SimpleNamespace(create=mock_create)
        )

        messages = [{"role": "user", "content": "Hello"}]
        result = chimeric_cerebras_client._chat_completion_impl(
            messages=messages,
            model="llama-3.1-8b",
            stream=False,
        )

        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Tool result processed"
        assert call_count == 2  # Should make two calls

    async def test_achat_completion_with_tool_calls_and_retry(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test async chat completion with tool calls that triggers a retry."""

        # Register a mock tool
        def mock_tool(param1: str) -> str:
            return f"Result: {param1}"

        chimeric_cerebras_client.tool_manager.register(mock_tool, "test_tool_4")

        # First response with tool calls
        mock_response_with_tools = create_mock_chat_completion_response("I'll use a tool")
        mock_response_with_tools.choices[0].message.tool_calls = [
            create_mock_tool_call("call_123", "test_tool_4", '{"param1": "value1"}')
        ]

        # Second response after tool execution
        mock_final_response = create_mock_chat_completion_response("Tool result processed")

        # Mock the create method to return different responses
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

        messages = [{"role": "user", "content": "Hello"}]
        result = await chimeric_cerebras_client._achat_completion_impl(
            messages=messages,
            model="llama-3.1-8b",
            stream=False,
        )

        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Tool result processed"
        assert call_count == 2  # Should make two calls

    async def test_achat_completion_with_scout_model_and_tools(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test async chat completion with scout model and tools."""
        mock_response = create_mock_chat_completion_response("Hello!")
        mock_create = AsyncMock(return_value=mock_response)
        chimeric_cerebras_client.async_client.chat = SimpleNamespace(
            completions=SimpleNamespace(create=mock_create)
        )

        # Add tools to trigger the parallel_tool_calls logic
        tools = [{"type": "function", "function": {"name": "test", "description": "test"}}]

        messages = [{"role": "user", "content": "Hello"}]
        await chimeric_cerebras_client._achat_completion_impl(
            messages=messages,
            model="llama-4-scout-17b-16e-instruct",
            stream=False,
            tools=tools,
        )

        # Check that parallel_tool_calls was set to False
        call_args = mock_create.call_args
        assert call_args[1]["parallel_tool_calls"] is False

    def test_upload_file_not_implemented(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test that file upload raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Cerebras does not support file uploads"):
            chimeric_cerebras_client._upload_file()

    def test_stream_with_empty_choices(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test streaming with chunks that have no choices."""
        mock_chunks = [
            create_mock_chat_chunk_response("Hello", choices=[]),
            create_mock_chat_chunk_response("World", choices=[]),
        ]

        def mock_stream():
            yield from mock_chunks

        result = chimeric_cerebras_client._stream(mock_stream())
        chunks = list(result)

        # Should not yield any chunks since there are no choices
        assert len(chunks) == 0

    async def test_astream_with_empty_choices(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test async streaming with chunks that have no choices."""
        mock_chunks = [
            create_mock_chat_chunk_response("Hello", choices=[]),
            create_mock_chat_chunk_response("World", choices=[]),
        ]

        async def mock_async_stream():
            for chunk in mock_chunks:
                yield chunk

        result = chimeric_cerebras_client._astream(mock_async_stream())
        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        # Should not yield any chunks since there are no choices
        assert len(chunks) == 0

    def test_stream_with_no_delta_content(self, chimeric_cerebras_client: CerebrasClient) -> None:
        """Test streaming with chunks that have no delta content."""
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

        def mock_stream():
            yield from mock_chunks

        result = chimeric_cerebras_client._stream(mock_stream())
        chunks = list(result)

        # Should only yield chunks with content
        assert len(chunks) == 1
        assert chunks[0].common.content == "Hello"

    async def test_astream_with_no_delta_content(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test async streaming with chunks that have no delta content."""
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

        async def mock_async_stream():
            for chunk in mock_chunks:
                yield chunk

        result = chimeric_cerebras_client._astream(mock_async_stream())
        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        # Should only yield chunks with content
        assert len(chunks) == 1
        assert chunks[0].common.content == "Hello"

    def test_handle_function_tool_calls_with_non_list_messages(
        self, chimeric_cerebras_client: CerebrasClient
    ) -> None:
        """Test handling function tool calls with non-list messages."""

        # Register a mock tool
        def mock_tool(param1: str) -> str:
            return f"Result: {param1}"

        chimeric_cerebras_client.tool_manager.register(mock_tool, "test_tool_5")

        # Create a mock response with tool calls
        mock_response = create_mock_chat_completion_response("Hello!")
        mock_response.choices[0].message.tool_calls = [
            create_mock_tool_call("call_123", "test_tool_5", '{"param1": "value1"}')
        ]

        # Pass a single message (not a list) to test the conversion
        messages = {"role": "user", "content": "Hello"}
        tool_calls, updated_messages = chimeric_cerebras_client._handle_function_tool_calls(
            mock_response,
            messages,  # type: ignore[arg-type]
        )

        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "test_tool_5"
        assert len(updated_messages) == 3  # Original + assistant + tool result
