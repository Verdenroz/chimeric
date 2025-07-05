from collections.abc import AsyncGenerator, Generator
import os
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from chimeric import Chimeric
from chimeric.exceptions import ToolRegistrationError
import chimeric.providers.grok.client as client_module
from chimeric.providers.grok.client import GrokClient
from chimeric.types import (
    ChimericCompletionResponse,
    ChimericStreamChunk,
    CompletionResponse,
    Tool,
    ToolParameters,
)


@pytest.fixture
def chimeric_grok():
    """Create a Chimeric instance configured for Grok."""
    return Chimeric(
        grok_api_key=os.getenv("GROK_API_KEY", "test_key"),
    )


@pytest.fixture
def chimeric_grok_client(chimeric_grok) -> GrokClient:
    """Get the GrokClient from the Chimeric wrapper."""
    return cast("GrokClient", chimeric_grok.get_provider_client("grok"))


@pytest.fixture(autouse=True)
def patch_grok_imports(monkeypatch):
    """Stub out actual xai-sdk classes to prevent network calls."""

    # Create mock models
    def create_mock_models():
        mock_model1 = SimpleNamespace()
        mock_model1.name = "grok-3"
        mock_model1.version = "1.0"
        mock_model1.input_modalities = ["TEXT"]
        mock_model1.output_modalities = ["TEXT"]
        mock_model1.max_prompt_length = 131072
        mock_model1.system_fingerprint = "fp_test1"
        mock_model1.aliases = ["grok-3-latest"]
        mock_model1.prompt_text_token_price = 30000
        mock_model1.completion_text_token_price = 150000
        mock_model1.created = SimpleNamespace(seconds=1743724800)

        mock_model2 = SimpleNamespace()
        mock_model2.name = "grok-2-vision"
        mock_model2.version = "1.0"
        mock_model2.input_modalities = ["TEXT", "IMAGE"]
        mock_model2.output_modalities = ["TEXT"]
        mock_model2.max_prompt_length = 32768
        mock_model2.system_fingerprint = "fp_test2"
        mock_model2.aliases = ["grok-2-vision-latest"]
        mock_model2.prompt_text_token_price = 20000
        mock_model2.prompt_image_token_price = 20000
        mock_model2.completion_text_token_price = 100000
        mock_model2.created = SimpleNamespace(seconds=1733961600)

        # Add a third model without aliases to test branch coverage
        mock_model3 = SimpleNamespace()
        mock_model3.name = "grok-basic"
        mock_model3.version = "1.0"
        mock_model3.input_modalities = ["TEXT"]
        mock_model3.output_modalities = ["TEXT"]
        mock_model3.max_prompt_length = 8192
        mock_model3.system_fingerprint = "fp_test3"
        # No aliases attribute to test the other branch
        mock_model3.prompt_text_token_price = 15000
        mock_model3.completion_text_token_price = 75000
        mock_model3.created = SimpleNamespace(seconds=1733961600)

        return [mock_model1, mock_model2, mock_model3]

    # Create mock implementations for sync and async SDK entrypoints
    def create_grok_client_mock(api_key: str, **kw: Any) -> SimpleNamespace:
        mock_models_response = SimpleNamespace()
        mock_models_response.models = create_mock_models()

        mock_models_api = SimpleNamespace()
        mock_models_api.list_language_models = lambda: mock_models_response.models

        return SimpleNamespace(api_key=api_key, models=mock_models_api, **kw)

    def create_async_grok_client_mock(api_key: str, **kw: Any) -> SimpleNamespace:
        mock_models_response = SimpleNamespace()
        mock_models_response.models = create_mock_models()

        mock_models_api = SimpleNamespace()
        mock_models_api.list_language_models = lambda: mock_models_response.models

        return SimpleNamespace(api_key=api_key, models=mock_models_api, **kw)

    monkeypatch.setattr(client_module, "Client", create_grok_client_mock)
    monkeypatch.setattr(client_module, "AsyncClient", create_async_grok_client_mock)

    # Create stub Response and Chunk types for isinstance checks
    class MockResponseType:
        pass

    class MockChunkType:
        pass

    monkeypatch.setattr(client_module, "Response", MockResponseType)
    monkeypatch.setattr(client_module, "Chunk", MockChunkType)


# noinspection PyUnusedLocal,PyTypeChecker
class TestGrokClient:
    """Test cases for GrokClient functionality."""

    def test_client_initialization(self, chimeric_grok_client):
        """Test that GrokClient initializes correctly."""
        assert chimeric_grok_client is not None
        assert chimeric_grok_client._provider_name == "Grok"
        assert chimeric_grok_client.api_key == "test_key"

    def test_capabilities(self, chimeric_grok_client):
        """Test GrokClient capabilities."""
        capabilities = chimeric_grok_client.capabilities
        assert capabilities.multimodal is True
        assert capabilities.streaming is True
        assert capabilities.tools is True
        assert capabilities.agents is False
        assert capabilities.files is False

    def test_list_models(self, chimeric_grok_client):
        """Test listing available models."""
        models = chimeric_grok_client.list_models()
        assert len(models) >= 4  # Original models + aliases

        model_ids = [model.id for model in models]
        assert "grok-3" in model_ids
        assert "grok-2-vision" in model_ids
        assert "grok-3-latest" in model_ids  # alias
        assert "grok-2-vision-latest" in model_ids  # alias

        # Check model details
        grok3_model = next(m for m in models if m.id == "grok-3")
        assert grok3_model.name == "grok-3"
        assert grok3_model.metadata is not None
        assert grok3_model.metadata["version"] == "1.0"
        assert grok3_model.metadata["max_prompt_length"] == 131072
        assert grok3_model.metadata["system_fingerprint"] == "fp_test1"
        assert grok3_model.provider == "grok"

        # Check alias details
        grok3_alias = next(m for m in models if m.id == "grok-3-latest")
        assert grok3_alias.name == "grok-3-latest"
        assert grok3_alias.metadata is not None
        assert grok3_alias.metadata["canonical_name"] == "grok-3"
        assert grok3_alias.provider == "grok"

    def test_model_aliases(self, chimeric_grok_client):
        """Test that model aliases are returned in the model list."""
        models = chimeric_grok_client.list_models()
        alias_models = [m for m in models if "canonical_name" in m.metadata]
        assert len(alias_models) >= 2

        # Check that aliases have canonical_name in metadata
        for alias_model in alias_models:
            assert "canonical_name" in alias_model.metadata
            assert alias_model.metadata["canonical_name"] in ["grok-3", "grok-2-vision"]

    def test_generic_types(self, chimeric_grok_client):
        """Test that generic types are returned correctly."""
        types = chimeric_grok_client._get_generic_types()
        assert "sync" in types
        assert "async" in types

    def test_message_conversion(self, chimeric_grok_client):
        """Test message conversion for different input types."""
        from xai_sdk.chat import assistant, user

        # Test string input
        messages = chimeric_grok_client._convert_messages("Hello")
        assert len(messages) == 1
        assert isinstance(messages[0], type(user("test")))

        # Test list input
        messages = chimeric_grok_client._convert_messages(
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]
        )
        assert len(messages) == 2
        assert isinstance(messages[0], type(user("test")))
        assert isinstance(messages[1], type(assistant("test")))

    def test_tool_encoding(self, chimeric_grok_client):
        """Test tool encoding for Grok API format."""
        # Test with Tool objects
        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters=ToolParameters(
                type="object", properties={"query": {"type": "string"}}, required=["query"]
            ),
        )

        encoded = chimeric_grok_client._encode_tools([tool])
        assert len(encoded) == 1
        assert encoded[0]["name"] == "test_tool"
        assert encoded[0]["description"] == "A test tool"
        assert "parameters" in encoded[0]

        # Test with None
        encoded = chimeric_grok_client._encode_tools(None)
        assert encoded is None

    def test_sync_chat_completion(self, chimeric_grok_client, monkeypatch):
        """Test synchronous chat completion."""
        client = chimeric_grok_client

        # Create a mock response object
        mock_response = SimpleNamespace()
        mock_response.content = "Test response from Grok"
        mock_response.model = "grok-3"
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_response.tool_calls = None

        # Create a mock chat object
        mock_chat = Mock()
        mock_chat.sample.return_value = mock_response

        # Mock the client's chat.create method
        mock_create = Mock(return_value=mock_chat)
        client._client.chat = SimpleNamespace(create=mock_create)

        # Test the implementation
        result = client._chat_completion_impl(
            messages="What is the capital of France?", model="grok-3"
        )

        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Test response from Grok"
        assert result.common.model == "grok-3"
        assert result.common.usage.total_tokens == 30

    @pytest.mark.asyncio
    async def test_async_chat_completion(self, chimeric_grok_client, monkeypatch):
        """Test asynchronous chat completion."""
        client = chimeric_grok_client

        # Create a mock response object
        mock_response = SimpleNamespace()
        mock_response.content = "Test response from Grok"
        mock_response.model = "grok-3"
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_response.tool_calls = None

        # Create a mock chat object
        mock_chat = Mock()
        mock_chat.sample = AsyncMock(return_value=mock_response)

        # Mock the async client's chat.create method
        mock_create = Mock(return_value=mock_chat)
        client._async_client.chat = SimpleNamespace(create=mock_create)

        # Test the async implementation
        result = await client._achat_completion_impl(
            messages="What is the capital of France?", model="grok-3"
        )

        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Test response from Grok"
        assert result.common.model == "grok-3"

    def test_sync_streaming_chat_completion(self, chimeric_grok_client, monkeypatch):
        """Test synchronous streaming chat completion."""
        client = chimeric_grok_client

        # Create mock chunks
        chunk1 = SimpleNamespace()
        chunk1.content = "Tell"
        chunk2 = SimpleNamespace()
        chunk2.content = " me"
        chunk3 = SimpleNamespace()
        chunk3.content = " a story"

        mock_response = SimpleNamespace()
        mock_response.content = "Tell me a story"

        def mock_stream():
            yield mock_response, chunk1
            yield mock_response, chunk2
            yield mock_response, chunk3

        # Create a mock chat object
        mock_chat = Mock()
        mock_chat.stream.return_value = mock_stream()

        # Mock the client's chat.create method
        mock_create = Mock(return_value=mock_chat)
        client._client.chat = SimpleNamespace(create=mock_create)

        # Test streaming
        result = client._chat_completion_impl(
            messages="Tell me a story", model="grok-3", stream=True
        )

        assert isinstance(result, Generator)
        chunks = list(result)
        assert len(chunks) == 3

        # Check that chunks are ChimericStreamChunk instances
        for chunk in chunks:
            assert isinstance(chunk, ChimericStreamChunk)
            assert hasattr(chunk.common, "content")
            assert hasattr(chunk.common, "delta")

    @pytest.mark.asyncio
    async def test_async_streaming_chat_completion(self, chimeric_grok_client, monkeypatch):
        """Test asynchronous streaming chat completion."""
        client = chimeric_grok_client

        # Create mock chunks
        chunk1 = SimpleNamespace()
        chunk1.content = "Tell"
        chunk2 = SimpleNamespace()
        chunk2.content = " me"
        chunk3 = SimpleNamespace()
        chunk3.content = " a story"

        mock_response = SimpleNamespace()
        mock_response.content = "Tell me a story"

        async def mock_async_stream():
            yield mock_response, chunk1
            yield mock_response, chunk2
            yield mock_response, chunk3

        # Create a mock chat object
        mock_chat = Mock()
        mock_chat.stream.return_value = mock_async_stream()

        # Mock the async client's chat.create method
        mock_create = Mock(return_value=mock_chat)
        client._async_client.chat = SimpleNamespace(create=mock_create)

        # Test async streaming
        result = await client._achat_completion_impl(
            messages="Tell me a story", model="grok-3", stream=True
        )

        assert isinstance(result, AsyncGenerator)
        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) == 3

        # Check that chunks are ChimericStreamChunk instances
        for chunk in chunks:
            assert isinstance(chunk, ChimericStreamChunk)
            assert hasattr(chunk.common, "content")
            assert hasattr(chunk.common, "delta")

    def test_tool_function_calling(self, chimeric_grok_client, monkeypatch):
        """Test function calling with tools."""
        client = chimeric_grok_client

        # Register a test tool
        def get_weather(city: str) -> str:
            return f"The weather in {city} is sunny"

        client.tool_manager.register(
            func=get_weather, name="get_weather", description="Get weather for a city"
        )

        # Mock a tool call response
        mock_tool_call = SimpleNamespace()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = SimpleNamespace()
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"city": "Paris"}'

        # Test the _process_function_call method directly
        response = client._process_function_call(mock_tool_call)
        assert response["name"] == "get_weather"
        assert response["call_id"] == "call_123"
        assert "sunny" in response["result"]

    def test_error_handling(self, chimeric_grok_client, monkeypatch):
        """Test error handling in chat completion."""
        client = chimeric_grok_client

        # Mock an exception from the xai-sdk
        class MockAPIError(Exception):
            pass

        def mock_create_with_error(*args, **kwargs):
            raise MockAPIError("API Error")

        client._client.chat = SimpleNamespace(create=mock_create_with_error)

        with pytest.raises(MockAPIError) as exc_info:
            client._chat_completion_impl(messages="Test message", model="grok-3")

        assert "API Error" in str(exc_info.value)

    def test_message_conversion_edge_cases(self, chimeric_grok_client):
        """Test message conversion for edge cases."""
        from xai_sdk.chat import user

        client = chimeric_grok_client

        # Test with list containing mixed types
        messages = client._convert_messages(
            [
                "string message",
                {"role": "user", "content": "dict message"},
                123,  # non-string, non-dict object
            ]
        )
        assert len(messages) == 3
        # All should be converted to xai-sdk message objects
        for msg in messages:
            assert isinstance(msg, type(user("test")))

        # Test with non-string, non-list input
        messages = client._convert_messages(42)
        assert len(messages) == 1
        assert isinstance(messages[0], type(user("test")))

    def test_tool_registration_error(self, chimeric_grok_client):
        """Test tool function calling with invalid tool."""
        client = chimeric_grok_client

        # Register a non-callable tool (this should fail)
        tool_name = "bad_tool"

        # Mock tool manager to return a tool with non-callable function
        mock_tool = Mock()
        mock_tool.function = "not a function"  # Not callable
        client.tool_manager.get_tool = Mock(return_value=mock_tool)

        # Mock a tool call
        mock_tool_call = SimpleNamespace()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = SimpleNamespace()
        mock_tool_call.function.name = tool_name
        mock_tool_call.function.arguments = '{"arg": "value"}'

        with pytest.raises(ToolRegistrationError) as exc_info:
            client._process_function_call(mock_tool_call)

        assert f"Tool '{tool_name}' is not callable" in str(exc_info.value)

    def test_sync_chat_completion_without_tool_calls(self, chimeric_grok_client, monkeypatch):
        """Test sync chat completion without tool calls path."""
        client = chimeric_grok_client

        # Mock response without tool calls
        mock_response = SimpleNamespace()
        mock_response.content = "Simple response"
        mock_response.model = "grok-3"
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        mock_response.tool_calls = None

        # Create a mock chat object
        mock_chat = Mock()
        mock_chat.sample.return_value = mock_response

        # Mock the client's chat.create method
        mock_create = Mock(return_value=mock_chat)
        client._client.chat = SimpleNamespace(create=mock_create)

        # Test the implementation - this should exercise the else path on line 272-278
        result = client._chat_completion_impl(messages="What's the weather?", model="grok-3")

        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Simple response"

    @pytest.mark.asyncio
    async def test_async_chat_completion_without_tool_calls(
        self, chimeric_grok_client, monkeypatch
    ):
        """Test async chat completion without tool calls path."""
        client = chimeric_grok_client

        # Mock response without tool calls
        mock_response = SimpleNamespace()
        mock_response.content = "Simple response"
        mock_response.model = "grok-3"
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        mock_response.tool_calls = None

        # Create a mock chat object
        mock_chat = Mock()
        mock_chat.sample = AsyncMock(return_value=mock_response)

        # Mock the async client's chat.create method
        mock_create = Mock(return_value=mock_chat)
        client._async_client.chat = SimpleNamespace(create=mock_create)

        # Test the async implementation - this should exercise the else path on line 332-338
        result = await client._achat_completion_impl(messages="What's the weather?", model="grok-3")

        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Simple response"

    @pytest.mark.asyncio
    async def test_async_error_handling(self, chimeric_grok_client, monkeypatch):
        """Test error handling in async chat completion."""
        client = chimeric_grok_client

        # Mock an exception from the xai-sdk
        class MockAsyncAPIError(Exception):
            pass

        def mock_create_with_error(*args, **kwargs):
            raise MockAsyncAPIError("Async API Error")

        client._async_client.chat = SimpleNamespace(create=mock_create_with_error)

        with pytest.raises(MockAsyncAPIError) as exc_info:
            await client._achat_completion_impl(messages="Test message", model="grok-3")

        assert "Async API Error" in str(exc_info.value)

    def test_upload_file_not_implemented(self, chimeric_grok_client):
        """Test that upload_file raises NotImplementedError."""
        client = chimeric_grok_client

        with pytest.raises(NotImplementedError) as exc_info:
            client._upload_file(file_path="/tmp/test.txt")

        assert "Grok does not support file uploads" in str(exc_info.value)

    def test_usage_extraction_with_object(self, chimeric_grok_client):
        """Test usage extraction when usage_data is an object (not dict)."""
        client = chimeric_grok_client

        # Create mock response with object-based usage
        mock_response = SimpleNamespace()
        mock_response.content = "Test response"
        mock_response.model = "grok-3"

        # Create usage as an object with attributes
        mock_usage = SimpleNamespace()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage

        result = client._create_chimeric_response(mock_response, [])

        assert result.common.usage.prompt_tokens == 5
        assert result.common.usage.completion_tokens == 10
        assert result.common.usage.total_tokens == 15

    def test_tool_encoding_with_dict(self, chimeric_grok_client):
        """Test tool encoding when tool is already a dictionary."""
        client = chimeric_grok_client

        # Test with pre-formatted tool dictionary
        tool_dict = {"name": "test_tool", "description": "A test", "parameters": {}}
        encoded = client._encode_tools([tool_dict])

        assert len(encoded) == 1
        assert encoded[0] == tool_dict

    def test_tool_execution_error_handling(self, chimeric_grok_client):
        """Test error handling during tool execution."""
        client = chimeric_grok_client

        # Register a tool that will raise an exception
        def failing_tool(query: str) -> str:
            raise ValueError("Tool execution failed")

        client.tool_manager.register(
            failing_tool, name="failing_tool", description="A tool that fails"
        )

        # Create mock tool call
        mock_tool_call = SimpleNamespace()
        mock_tool_call.id = "call_fail"
        mock_tool_call.function = SimpleNamespace()
        mock_tool_call.function.name = "failing_tool"
        mock_tool_call.function.arguments = '{"query": "test"}'

        result = client._process_function_call(mock_tool_call)

        assert result["call_id"] == "call_fail"
        assert result["name"] == "failing_tool"
        assert "error" in result
        assert result["error"] is True
        assert "Tool execution failed" in result["result"]

    def test_usage_extraction_no_usage(self, chimeric_grok_client):
        """Test usage extraction when no usage data is present."""
        client = chimeric_grok_client

        # Create mock response without usage
        mock_response = SimpleNamespace()
        mock_response.content = "Test response"
        mock_response.model = "grok-3"
        # No usage attribute

        result = client._create_chimeric_response(mock_response, [])

        # Should get default usage values
        assert result.common.usage.prompt_tokens == 0
        assert result.common.usage.completion_tokens == 0
        assert result.common.usage.total_tokens == 0

    def test_sync_chat_completion_with_tools_parameter(self, chimeric_grok_client, monkeypatch):
        """Test sync chat completion when tools parameter is provided."""
        client = chimeric_grok_client

        # Create mock response
        mock_response = SimpleNamespace()
        mock_response.content = "I can help with that"
        mock_response.model = "grok-3"
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_response.tool_calls = None

        # Create mock chat
        mock_chat = Mock()
        mock_chat.sample.return_value = mock_response

        # Mock the client's chat.create method
        mock_create = Mock(return_value=mock_chat)
        client._client.chat = SimpleNamespace(create=mock_create)

        # Create a tool dict (as would come from base class encoding)
        tool_dict = {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        }

        # Test with tools parameter to trigger tool conversion
        result = client._chat_completion_impl(
            messages="Hello",
            model="grok-3",
            tools=[tool_dict],  # This should trigger lines 413-415
        )

        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "I can help with that"

    @pytest.mark.asyncio
    async def test_async_chat_completion_with_tools_parameter(
        self, chimeric_grok_client, monkeypatch
    ):
        """Test async chat completion when tools parameter is provided."""
        client = chimeric_grok_client

        # Create mock response
        mock_response = SimpleNamespace()
        mock_response.content = "I can help with that async"
        mock_response.model = "grok-3"
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_response.tool_calls = None

        # Create mock chat
        mock_chat = AsyncMock()
        mock_chat.sample.return_value = mock_response

        # Mock the client's async chat.create method
        mock_create = Mock(return_value=mock_chat)
        client._async_client.chat = SimpleNamespace(create=mock_create)

        # Create a tool dict (as would come from base class encoding)
        tool_dict = {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        }

        # Test with tools parameter to trigger tool conversion
        result = await client._achat_completion_impl(
            messages="Hello",
            model="grok-3",
            tools=[tool_dict],  # This should trigger lines 451-453
        )

        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "I can help with that async"

    def test_sync_chat_completion_with_tool_calls(self, chimeric_grok_client, monkeypatch):
        """Test sync chat completion with tool calls that trigger tool execution."""
        client = chimeric_grok_client

        # Create mock tool call response
        mock_tool_call = SimpleNamespace()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = SimpleNamespace()
        mock_tool_call.function.name = "get_weather_sync_tool_execution"
        mock_tool_call.function.arguments = '{"city": "Paris"}'

        # Mock first response with tool calls
        mock_response_with_tools = SimpleNamespace()
        mock_response_with_tools.content = "I'll get the weather for you"
        mock_response_with_tools.model = "grok-3"
        mock_response_with_tools.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }
        mock_response_with_tools.tool_calls = [mock_tool_call]

        # Mock final response after tool execution
        mock_final_response = SimpleNamespace()
        mock_final_response.content = "Based on the weather data, it's sunny in Paris"
        mock_final_response.model = "grok-3"
        mock_final_response.usage = {
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40,
        }
        mock_final_response.tool_calls = None

        # Create mock chat object
        mock_chat = Mock()
        mock_chat.sample.side_effect = [mock_response_with_tools, mock_final_response]
        mock_chat.append = Mock()

        # Mock the client's chat.create method
        mock_create = Mock(return_value=mock_chat)
        client._client.chat = SimpleNamespace(create=mock_create)

        # Mock the tool manager and tool execution
        mock_tool = Mock()
        mock_tool.function = Mock(return_value="The weather in Paris is sunny")
        client.tool_manager.get_tool = Mock(return_value=mock_tool)

        # Test the implementation
        result = client._chat_completion_impl(
            messages="What's the weather in Paris?", model="grok-3"
        )

        # Verify tool was called and final response is returned
        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Based on the weather data, it's sunny in Paris"
        assert result.common.model == "grok-3"
        assert result.common.usage.total_tokens == 40
        assert result.common.metadata is not None
        assert "tool_calls" in result.common.metadata
        assert len(result.common.metadata["tool_calls"]) == 1
        assert result.common.metadata["tool_calls"][0]["name"] == "get_weather_sync_tool_execution"
        assert "sunny" in result.common.metadata["tool_calls"][0]["result"]

        # Verify chat.append was called multiple times (message, response, tool result)
        assert mock_chat.append.call_count == 3
        # Check the last call was the tool result
        last_call_args = mock_chat.append.call_args_list[-1][0][0]
        # The tool result should contain the result text
        assert "sunny" in str(last_call_args)

    @pytest.mark.asyncio
    async def test_async_chat_completion_with_tool_calls(self, chimeric_grok_client, monkeypatch):
        """Test async chat completion with tool calls that trigger tool execution."""
        client = chimeric_grok_client

        # Create mock tool call response
        mock_tool_call = SimpleNamespace()
        mock_tool_call.id = "call_456"
        mock_tool_call.function = SimpleNamespace()
        mock_tool_call.function.name = "get_weather_async_tool_execution"
        mock_tool_call.function.arguments = '{"city": "London"}'

        # Mock first response with tool calls
        mock_response_with_tools = SimpleNamespace()
        mock_response_with_tools.content = "I'll get the weather for you"
        mock_response_with_tools.model = "grok-3"
        mock_response_with_tools.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }
        mock_response_with_tools.tool_calls = [mock_tool_call]

        # Mock final response after tool execution
        mock_final_response = SimpleNamespace()
        mock_final_response.content = "Based on the weather data, it's cloudy in London"
        mock_final_response.model = "grok-3"
        mock_final_response.usage = {
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40,
        }
        mock_final_response.tool_calls = None

        # Create mock chat object
        mock_chat = Mock()
        mock_chat.sample = AsyncMock(side_effect=[mock_response_with_tools, mock_final_response])
        mock_chat.append = Mock()

        # Mock the async client's chat.create method
        mock_create = Mock(return_value=mock_chat)
        client._async_client.chat = SimpleNamespace(create=mock_create)

        # Mock the tool manager and tool execution
        mock_tool = Mock()
        mock_tool.function = Mock(return_value="The weather in London is cloudy")
        client.tool_manager.get_tool = Mock(return_value=mock_tool)

        # Test the async implementation
        result = await client._achat_completion_impl(
            messages="What's the weather in London?", model="grok-3"
        )

        # Verify tool was called and final response is returned
        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Based on the weather data, it's cloudy in London"
        assert result.common.model == "grok-3"
        assert result.common.usage.total_tokens == 40
        assert result.common.metadata is not None
        assert "tool_calls" in result.common.metadata
        assert len(result.common.metadata["tool_calls"]) == 1
        assert result.common.metadata["tool_calls"][0]["name"] == "get_weather_async_tool_execution"
        assert "cloudy" in result.common.metadata["tool_calls"][0]["result"]

        # Verify chat.append was called multiple times (message, response, tool result)
        assert mock_chat.append.call_count == 3
        # Check the last call was the tool result
        last_call_args = mock_chat.append.call_args_list[-1][0][0]
        # The tool result should contain the result text
        assert "cloudy" in str(last_call_args)


class TestGrokIntegration:
    """Integration tests for Grok client with Chimeric."""

    def test_grok_provider_registration(self, chimeric_grok):
        """Test that Grok provider is properly registered."""
        assert "grok" in chimeric_grok.available_providers

        grok_client = chimeric_grok.get_provider_client("grok")
        assert isinstance(grok_client, GrokClient)

    def test_grok_model_detection(self, chimeric_grok):
        """Test automatic model detection for Grok models."""
        # This would normally query the API, but we're mocking it
        models = chimeric_grok.list_models(provider="grok")
        model_ids = [m.id for m in models]
        assert "grok-3" in model_ids
        assert "grok-2-vision" in model_ids
        assert "grok-3-latest" in model_ids  # alias
        assert "grok-2-vision-latest" in model_ids  # alias

    def test_grok_capabilities_integration(self, chimeric_grok):
        """Test capabilities integration."""
        capabilities = chimeric_grok.get_capabilities(provider="grok")
        assert capabilities.multimodal is True
        assert capabilities.tools is True
        assert capabilities.streaming is True

    def test_grok_generate_wrapper(self, chimeric_grok, monkeypatch):
        """Test the high-level generate method with Grok."""
        grok_client = chimeric_grok.get_provider_client("grok")

        # Mock the response
        mock_response = SimpleNamespace()
        mock_response.content = "Test response from Grok"
        mock_response.model = "grok-3"
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_response.tool_calls = None

        # Mock the chat
        mock_chat = Mock()
        mock_chat.sample.return_value = mock_response

        # Mock the client's chat.create method
        mock_create = Mock(return_value=mock_chat)
        grok_client._client.chat = SimpleNamespace(create=mock_create)

        response = chimeric_grok.generate(model="grok-3", messages="Hello, Grok!", provider="grok")

        assert isinstance(response, CompletionResponse)
        assert response.content == "Test response from Grok"

    @pytest.mark.asyncio
    async def test_grok_agenerate_wrapper(self, chimeric_grok, monkeypatch):
        """Test the high-level async generate method with Grok."""
        grok_client = chimeric_grok.get_provider_client("grok")

        # Mock the response
        mock_response = SimpleNamespace()
        mock_response.content = "Test response from Grok"
        mock_response.model = "grok-3"
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        mock_response.tool_calls = None

        # Mock the chat
        mock_chat = Mock()
        mock_chat.sample = AsyncMock(return_value=mock_response)

        # Mock the async client's chat.create method
        mock_create = Mock(return_value=mock_chat)
        grok_client._async_client.chat = SimpleNamespace(create=mock_create)

        response = await chimeric_grok.agenerate(
            model="grok-3", messages="Hello, Grok!", provider="grok"
        )

        assert isinstance(response, CompletionResponse)
        assert response.content == "Test response from Grok"
