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

        # Add a third model without aliases
        mock_model3 = SimpleNamespace()
        mock_model3.name = "grok-basic"
        mock_model3.version = "1.0"
        mock_model3.input_modalities = ["TEXT"]
        mock_model3.output_modalities = ["TEXT"]
        mock_model3.max_prompt_length = 8192
        mock_model3.system_fingerprint = "fp_test3"
        # No aliases attribute
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
    """Consolidated test cases for GrokClient functionality."""

    def test_client_initialization_and_capabilities(self, chimeric_grok_client):
        """Test client initialization and capabilities."""
        assert chimeric_grok_client is not None
        assert chimeric_grok_client._provider_name == "Grok"
        assert chimeric_grok_client.api_key == "test_key"

        # Test capabilities
        capabilities = chimeric_grok_client.capabilities
        assert capabilities.multimodal is True
        assert capabilities.streaming is True
        assert capabilities.tools is True
        assert capabilities.agents is False
        assert capabilities.files is False

        # Test generic types
        types = chimeric_grok_client._get_generic_types()
        assert "sync" in types
        assert "async" in types

    def test_model_listing_and_aliases(self, chimeric_grok_client):
        """Test listing models including aliases and metadata handling."""
        models = chimeric_grok_client.list_models()
        assert len(models) >= 4  # Original models + aliases

        model_ids = [model.id for model in models]
        assert "grok-3" in model_ids
        assert "grok-2-vision" in model_ids
        assert "grok-basic" in model_ids  # Model without aliases
        assert "grok-3-latest" in model_ids  # alias
        assert "grok-2-vision-latest" in model_ids  # alias

        # Check model details
        grok3_model = next(m for m in models if m.id == "grok-3")
        assert grok3_model.name == "grok-3"
        assert grok3_model.metadata is not None
        assert grok3_model.metadata["version"] == "1.0"
        assert grok3_model.metadata["max_prompt_length"] == 131072
        assert grok3_model.provider == "grok"

        # Check alias details (tests canonical_name metadata)
        grok3_alias = next(m for m in models if m.id == "grok-3-latest")
        assert grok3_alias.name == "grok-3-latest"
        assert grok3_alias.metadata["canonical_name"] == "grok-3"

        # Verify model without aliases doesn't have canonical_name
        basic_model = next(m for m in models if m.id == "grok-basic")
        assert "canonical_name" not in basic_model.metadata

    def test_message_conversion_comprehensive(self, chimeric_grok_client):
        """Test message conversion for all input types and edge cases."""
        from xai_sdk.chat import user

        client = chimeric_grok_client

        # Test string input
        messages = client._convert_messages("Hello")
        assert len(messages) == 1
        assert isinstance(messages[0], type(user("test")))

        # Test list with different message types
        messages = client._convert_messages(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "system", "content": "You are helpful"},
                "string message",  # Direct string
                123,  # Non-string, non-dict object
                {"role": "unknown", "content": "fallback to user"},  # Unknown role
            ]
        )
        assert len(messages) == 6
        # All should be converted to xai-sdk message objects
        for msg in messages:
            assert isinstance(msg, type(user("test")))

        # Test non-string, non-list input
        messages = client._convert_messages(42)
        assert len(messages) == 1
        assert isinstance(messages[0], type(user("test")))

        # Test empty dict message
        messages = client._convert_messages([{"content": "no role"}])
        assert len(messages) == 1

    def test_tool_encoding_and_processing(self, chimeric_grok_client):
        """Test tool encoding, processing, and error handling."""
        client = chimeric_grok_client

        # Test with Tool objects
        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters=ToolParameters(
                type="object", properties={"query": {"type": "string"}}, required=["query"]
            ),
        )

        encoded = client._encode_tools([tool])
        assert len(encoded) == 1
        assert isinstance(encoded, list)
        assert encoded[0]["name"] == "test_tool"
        assert encoded[0]["description"] == "A test tool"
        assert "parameters" in encoded[0]

        # Test with pre-formatted tool dictionary
        tool_dict = {"name": "dict_tool", "description": "A dict tool", "parameters": {}}
        encoded = client._encode_tools([tool_dict])
        assert len(encoded) == 1
        assert isinstance(encoded, list)
        assert encoded[0] == tool_dict

        # Test with None
        encoded = client._encode_tools(None)
        assert encoded is None

        # Test with an empty list
        encoded = client._encode_tools([])
        assert encoded is None

        # Test tool function calling with successful execution
        def get_weather(city: str) -> str:
            return f"The weather in {city} is sunny"

        client.tool_manager.register(
            func=get_weather, name="get_weather", description="Get weather for a city"
        )

        mock_tool_call = SimpleNamespace()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = SimpleNamespace()
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"city": "Paris"}'

        response = client._process_function_call(mock_tool_call)
        assert response["name"] == "get_weather"
        assert response["call_id"] == "call_123"
        assert "sunny" in response["result"]
        assert "error" not in response

        # Test tool execution error handling
        def failing_tool(query: str) -> str:
            raise ValueError("Tool execution failed")

        client.tool_manager.register(
            failing_tool, name="failing_tool", description="A tool that fails"
        )

        mock_tool_call.function.name = "failing_tool"
        mock_tool_call.function.arguments = '{"query": "test"}'

        result = client._process_function_call(mock_tool_call)
        assert result["call_id"] == "call_123"
        assert result["name"] == "failing_tool"
        assert "error" in result
        assert result["error"] is True
        assert "Tool execution failed" in result["result"]

        # Test tool registration error (non-callable)
        mock_tool = Mock()
        mock_tool.function = "not a function"  # Not callable
        client.tool_manager.get_tool = Mock(return_value=mock_tool)

        mock_tool_call.function.name = "bad_tool"
        with pytest.raises(ToolRegistrationError) as exc_info:
            client._process_function_call(mock_tool_call)
        assert "Tool 'bad_tool' is not callable" in str(exc_info.value)

    def test_response_creation_and_usage_extraction(self, chimeric_grok_client):
        """Test response creation with different usage data formats."""
        client = chimeric_grok_client

        # Test with dict-based usage
        mock_response = SimpleNamespace()
        mock_response.content = "Test response"
        mock_response.model = "grok-3"
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

        result = client._create_chimeric_response(mock_response, [])
        assert result.common.content == "Test response"
        assert result.common.model == "grok-3"
        assert result.common.usage.total_tokens == 30

        # Test with object-based usage
        mock_usage = SimpleNamespace()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage

        result = client._create_chimeric_response(mock_response, [])
        assert result.common.usage.prompt_tokens == 5
        assert result.common.usage.completion_tokens == 10
        assert result.common.usage.total_tokens == 15

        # Test with no usage data
        delattr(mock_response, "usage")
        result = client._create_chimeric_response(mock_response, [])
        assert result.common.usage.prompt_tokens == 0
        assert result.common.usage.completion_tokens == 0
        assert result.common.usage.total_tokens == 0

        # Test with tool calls metadata
        tool_calls = [{"name": "test_tool", "result": "success"}]
        result = client._create_chimeric_response(mock_response, tool_calls)
        assert result.common.metadata["tool_calls"] == tool_calls

        # Test with None content
        mock_response.content = None
        result = client._create_chimeric_response(mock_response, [])
        assert result.common.content == ""

    def test_sync_chat_completion_comprehensive(self, chimeric_grok_client):
        """Test synchronous chat completion"""
        client = chimeric_grok_client

        # Test basic completion without tools
        mock_response = SimpleNamespace()
        mock_response.content = "Simple response"
        mock_response.model = "grok-3"
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        mock_response.tool_calls = None

        mock_chat = Mock()
        mock_chat.sample.return_value = mock_response
        mock_chat.append = Mock()

        mock_create = Mock(return_value=mock_chat)
        client._client.chat = SimpleNamespace(create=mock_create)

        # Test without tools
        result = client._chat_completion_impl(messages="What's the weather?", model="grok-3")
        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Simple response"

        # Test with tool parameter (triggers tool conversion)
        tool_dict = {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        }

        result = client._chat_completion_impl(messages="Hello", model="grok-3", tools=[tool_dict])
        assert isinstance(result, ChimericCompletionResponse)

        # Test with tool calls that require execution
        mock_tool_call = SimpleNamespace()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = SimpleNamespace()
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"city": "Paris"}'

        mock_response_with_tools = SimpleNamespace()
        mock_response_with_tools.content = "I'll get the weather"
        mock_response_with_tools.model = "grok-3"
        mock_response_with_tools.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }
        mock_response_with_tools.tool_calls = [mock_tool_call]

        mock_final_response = SimpleNamespace()
        mock_final_response.content = "It's sunny in Paris"
        mock_final_response.model = "grok-3"
        mock_final_response.usage = {
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40,
        }
        mock_final_response.tool_calls = None

        mock_chat.sample.side_effect = [mock_response_with_tools, mock_final_response]

        # Register a test tool
        def get_weather(city: str) -> str:
            return f"The weather in {city} is sunny"

        client.tool_manager.register(get_weather, name="get_weather", description="Get weather")

        result = client._chat_completion_impl(
            messages="What's the weather in Paris?", model="grok-3"
        )
        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "It's sunny in Paris"
        assert result.common.metadata is not None
        assert result.common.metadata["tool_calls"][0]["name"] == "get_weather"

    async def test_async_chat_completion_comprehensive(self, chimeric_grok_client):
        """Test asynchronous chat completion"""
        client = chimeric_grok_client

        # Test basic async completion without tools
        mock_response = SimpleNamespace()
        mock_response.content = "Async response"
        mock_response.model = "grok-3"
        mock_response.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        mock_response.tool_calls = None

        mock_chat = Mock()
        mock_chat.sample = AsyncMock(return_value=mock_response)
        mock_chat.append = Mock()

        mock_create = Mock(return_value=mock_chat)
        client._async_client.chat = SimpleNamespace(create=mock_create)

        result = await client._achat_completion_impl(messages="Hello", model="grok-3")
        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "Async response"

        # Test async with tool calls
        mock_tool_call = SimpleNamespace()
        mock_tool_call.id = "call_456"
        mock_tool_call.function = SimpleNamespace()
        mock_tool_call.function.name = "async_weather"
        mock_tool_call.function.arguments = '{"city": "London"}'

        mock_response_with_tools = SimpleNamespace()
        mock_response_with_tools.content = "Getting async weather"
        mock_response_with_tools.model = "grok-3"
        mock_response_with_tools.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }
        mock_response_with_tools.tool_calls = [mock_tool_call]

        mock_final_response = SimpleNamespace()
        mock_final_response.content = "It's cloudy in London"
        mock_final_response.model = "grok-3"
        mock_final_response.usage = {
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40,
        }
        mock_final_response.tool_calls = None

        mock_chat.sample = AsyncMock(side_effect=[mock_response_with_tools, mock_final_response])

        # Register async tool
        def async_weather(city: str) -> str:
            return f"The weather in {city} is cloudy"

        client.tool_manager.register(
            async_weather, name="async_weather", description="Get async weather"
        )

        result = await client._achat_completion_impl(messages="Weather in London?", model="grok-3")
        assert isinstance(result, ChimericCompletionResponse)
        assert result.common.content == "It's cloudy in London"
        assert result.common.metadata is not None
        assert result.common.metadata["tool_calls"][0]["name"] == "async_weather"

    def test_streaming_comprehensive(self, chimeric_grok_client):
        """Test streaming with and without tools"""
        client = chimeric_grok_client

        # Test basic streaming without tools
        chunk1 = SimpleNamespace()
        chunk1.content = "Hello"
        chunk2 = SimpleNamespace()
        chunk2.content = " world"

        mock_response = SimpleNamespace()

        def mock_stream():
            yield mock_response, chunk1
            yield mock_response, chunk2

        mock_chat = Mock()
        mock_chat.stream.return_value = mock_stream()
        mock_chat.append = Mock()

        mock_create = Mock(return_value=mock_chat)
        client._client.chat = SimpleNamespace(create=mock_create)

        result = client._chat_completion_impl(
            messages="Tell me a story", model="grok-3", stream=True
        )

        assert isinstance(result, Generator)
        chunks = list(result)
        assert len(chunks) == 2

        for chunk in chunks:
            assert isinstance(chunk, ChimericStreamChunk)
            assert hasattr(chunk.common, "content")
            assert hasattr(chunk.common, "delta")

        # Test streaming with tools enabled but no tool calls
        mock_chat_no_tools = Mock()
        mock_chat_no_tools.stream.return_value = mock_stream()
        mock_chat_no_tools.append = Mock()
        mock_create.return_value = mock_chat_no_tools

        # Create a response without tool calls for tools_enabled=True
        response_no_tools = SimpleNamespace()
        response_no_tools.tool_calls = None  # No tool calls

        def mock_stream_no_tool_calls():
            yield response_no_tools, chunk1
            yield response_no_tools, chunk2

        mock_chat_no_tools.stream.return_value = mock_stream_no_tool_calls()

        result = client._chat_completion_impl(
            messages="Tell story",
            model="grok-3",
            stream=True,
            tools=[{"name": "dummy_tool", "description": "A dummy tool", "parameters": {}}],
        )

        chunks = list(result)
        assert len(chunks) == 2

        # Test streaming with tools that actually get called
        chunk_tool1 = SimpleNamespace()
        chunk_tool1.content = "I'll"
        chunk_tool2 = SimpleNamespace()
        chunk_tool2.content = " help"

        response_with_tools = SimpleNamespace()
        response_with_tools.tool_calls = [
            SimpleNamespace(
                id="call_123",
                function=SimpleNamespace(name="stream_tool", arguments='{"arg": "value"}'),
            )
        ]

        # Second stream - final response after tool execution
        chunk_final1 = SimpleNamespace()
        chunk_final1.content = "Based"
        chunk_final2 = SimpleNamespace()
        chunk_final2.content = " on results"

        response_final = SimpleNamespace()
        response_final.tool_calls = None

        # Create a complex streaming scenario that exercises the while loop
        call_count = 0

        def mock_complex_tool_stream():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call - stream with tool calls
                yield response_with_tools, chunk_tool1
                yield response_with_tools, chunk_tool2
            else:
                # Second call - final response without tool calls
                yield response_final, chunk_final1
                yield response_final, chunk_final2

        mock_chat_tools = Mock()
        mock_chat_tools.stream.side_effect = mock_complex_tool_stream
        mock_chat_tools.append = Mock()
        mock_create.return_value = mock_chat_tools

        # Register tool for streaming test
        def stream_tool(arg: str) -> str:
            return f"Tool result for {arg}"

        client.tool_manager.register(stream_tool, name="stream_tool", description="Stream tool")

        result = client._chat_completion_impl(
            messages="Use tool",
            model="grok-3",
            stream=True,
            tools=[{"name": "stream_tool", "description": "Stream tool", "parameters": {}}],
        )

        assert isinstance(result, Generator)
        chunks = list(result)
        assert len(chunks) == 4  # 2 from tool response + 2 from final response

        # Test chunk without content attribute
        chunk_no_content = SimpleNamespace()

        # No content attribute

        def mock_stream_no_content():
            yield mock_response, chunk_no_content

        mock_chat_no_content = Mock()
        mock_chat_no_content.stream.return_value = mock_stream_no_content()
        mock_create.return_value = mock_chat_no_content

        result = client._chat_completion_impl(messages="Test", model="grok-3", stream=True)

        chunks = list(result)
        assert len(chunks) == 1
        assert chunks[0].common.delta == ""

    def test_create_chat_params_comprehensive(self, chimeric_grok_client):
        """Test chat parameter creation with various scenarios."""
        client = chimeric_grok_client

        # Mock the chat.create methods
        mock_chat = Mock()
        mock_create = Mock(return_value=mock_chat)
        client._client.chat = SimpleNamespace(create=mock_create)
        client._async_client.chat = SimpleNamespace(create=mock_create)

        # Test with no tools
        chat, messages = client._create_chat_params(messages="Hello", model="grok-3", tools=None)
        assert chat is not None
        assert len(messages) == 1

        # Test with tools
        tools = [{"name": "test_tool", "description": "A tool", "parameters": {}}]
        chat, messages = client._create_chat_params(
            messages="Hello", model="grok-3", tools=tools, extra_param="value"
        )
        assert chat is not None

        # Test async version
        chat, messages = client._create_async_chat_params(
            messages="Hello", model="grok-3", tools=tools, extra_param="value"
        )
        assert chat is not None

        # Test async with no tools
        chat, messages = client._create_async_chat_params(
            messages="Hello", model="grok-3", tools=None
        )
        assert chat is not None

    def test_metadata_extraction_comprehensive(self, chimeric_grok_client):
        """Test metadata extraction with different chunk types."""
        client = chimeric_grok_client

        # Test chunk with metadata
        chunk_with_metadata = SimpleNamespace()
        chunk_with_metadata.content = "test"
        chunk_with_metadata.metadata = {"key": "value"}

        mock_response = SimpleNamespace()

        def mock_stream_with_metadata():
            yield mock_response, chunk_with_metadata

        mock_chat = Mock()
        mock_chat.stream.return_value = mock_stream_with_metadata()

        result = client._stream(mock_chat, tools_enabled=False)
        chunks = list(result)
        assert len(chunks) == 1
        assert chunks[0].common.metadata == {"key": "value"}

        # Test chunk without metadata
        chunk_without_metadata = SimpleNamespace()
        chunk_without_metadata.content = "test"

        def mock_stream_without_metadata():
            yield mock_response, chunk_without_metadata

        mock_chat.stream.return_value = mock_stream_without_metadata()

        result = client._stream(mock_chat, tools_enabled=False)
        chunks = list(result)
        assert len(chunks) == 1
        assert chunks[0].common.metadata == {}

    async def test_async_metadata_extraction_comprehensive(self, chimeric_grok_client):
        """Test async metadata extraction with different chunk types."""
        client = chimeric_grok_client

        # Test async chunk with metadata
        chunk_with_metadata = SimpleNamespace()
        chunk_with_metadata.content = "test"
        chunk_with_metadata.metadata = {"async_key": "async_value"}

        mock_response = SimpleNamespace()

        async def mock_async_stream_with_metadata():
            yield mock_response, chunk_with_metadata

        mock_chat = Mock()
        mock_chat.stream.return_value = mock_async_stream_with_metadata()

        result = client._astream(mock_chat, tools_enabled=False)
        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].common.metadata == {"async_key": "async_value"}

        # Test async chunk without metadata
        chunk_without_metadata = SimpleNamespace()
        chunk_without_metadata.content = "test"

        async def mock_async_stream_without_metadata():
            yield mock_response, chunk_without_metadata

        mock_chat.stream.return_value = mock_async_stream_without_metadata()

        result = client._astream(mock_chat, tools_enabled=False)
        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].common.metadata == {}

    def test_model_listing_edge_cases(self, chimeric_grok_client):
        """Test model listing with various edge cases."""
        client = chimeric_grok_client

        # Test model without created attribute
        def create_models_without_created():
            mock_model = SimpleNamespace()
            mock_model.name = "test-model"
            mock_model.version = "1.0"
            # No created attribute
            return [mock_model]

        original_list_models = client._client.models.list_language_models
        client._client.models.list_language_models = create_models_without_created

        models = client.list_models()
        assert len(models) >= 1
        test_model = next(m for m in models if m.id == "test-model")
        assert test_model.created_at is None

        # Restore original method
        client._client.models.list_language_models = original_list_models

    def test_tool_argument_parsing_edge_cases(self, chimeric_grok_client):
        """Test tool argument parsing with invalid JSON and edge cases."""
        client = chimeric_grok_client

        # Register a test tool
        def edge_case_tool(arg: str) -> str:
            return f"Received: {arg}"

        client.tool_manager.register(edge_case_tool, name="edge_tool", description="Edge case tool")

        # Test with invalid JSON arguments
        mock_tool_call = SimpleNamespace()
        mock_tool_call.id = "call_invalid_json"
        mock_tool_call.function = SimpleNamespace()
        mock_tool_call.function.name = "edge_tool"
        mock_tool_call.function.arguments = '{"invalid": json}'  # Invalid JSON

        result = client._process_function_call(mock_tool_call)
        assert result["call_id"] == "call_invalid_json"
        assert result["name"] == "edge_tool"
        assert "error" in result
        assert result["error"] is True

        # Test with empty arguments
        mock_tool_call.function.arguments = "{}"

        # This should cause an error because the tool expects an 'arg' parameter
        result = client._process_function_call(mock_tool_call)
        assert "error" in result
        assert result["error"] is True

        # Test with correct arguments
        mock_tool_call.function.arguments = '{"arg": "test_value"}'
        result = client._process_function_call(mock_tool_call)
        assert "error" not in result or not result.get("error", False)
        assert "test_value" in result["result"]

    def test_response_creation_edge_cases(self, chimeric_grok_client):
        """Test response creation with various edge cases."""
        client = chimeric_grok_client

        # Test response without model attribute
        mock_response = SimpleNamespace()
        mock_response.content = "Test"
        # No model attribute
        # No usage attribute

        result = client._create_chimeric_response(mock_response, [])
        assert result.common.model is None
        assert result.common.usage.total_tokens == 0

        # Test response with an empty tool calls list
        result = client._create_chimeric_response(mock_response, [])
        assert result.common.metadata == {}

        # Test response with non-empty tool calls
        tool_calls = [{"name": "test", "result": "success"}]
        result = client._create_chimeric_response(mock_response, tool_calls)
        assert result.common.metadata == {"tool_calls": tool_calls}

    def test_tool_parameters_with_none(self, chimeric_grok_client):
        """Test tool encoding with None parameters."""
        client = chimeric_grok_client

        # Test Tool with None parameters
        tool_with_none_params = Tool(
            name="none_params_tool", description="Tool with None parameters", parameters=None
        )

        encoded = client._encode_tools([tool_with_none_params])
        assert len(encoded) == 1
        assert isinstance(encoded, list)
        assert encoded[0]["parameters"] == {}

    def test_comprehensive_edge_cases(self, chimeric_grok_client):
        """Test various edge cases not covered elsewhere."""
        client = chimeric_grok_client

        # Test _convert_messages with an empty list
        messages = client._convert_messages([])
        assert len(messages) == 0

        # Test _convert_messages with dict missing content
        messages = client._convert_messages([{"role": "user"}])
        assert len(messages) == 1

        # Test _encode_tools with an empty tools list
        encoded = client._encode_tools([])
        assert encoded is None

        # Test message conversion with complex nested data
        complex_msg = {"role": "user", "content": "Hello", "extra_field": {"nested": "data"}}
        messages = client._convert_messages([complex_msg])
        assert len(messages) == 1

        # Test model with created but no seconds attribute
        def create_models_with_invalid_created():
            mock_model = SimpleNamespace()
            mock_model.name = "test-model-2"
            mock_model.version = "1.0"
            mock_model.created = SimpleNamespace()  # No seconds attribute
            return [mock_model]

        original_list_models = client._client.models.list_language_models
        client._client.models.list_language_models = create_models_with_invalid_created

        models = client.list_models()
        test_model2 = next(m for m in models if m.id == "test-model-2")
        assert test_model2.created_at is None

        # Restore original method
        client._client.models.list_language_models = original_list_models

    async def test_async_streaming_comprehensive(self, chimeric_grok_client):
        """Test async streaming with and without tools, covering all async streaming"""
        client = chimeric_grok_client

        # Test basic async streaming without tools
        chunk1 = SimpleNamespace()
        chunk1.content = "Async"
        chunk2 = SimpleNamespace()
        chunk2.content = " stream"

        mock_response = SimpleNamespace()

        async def mock_async_stream():
            yield mock_response, chunk1
            yield mock_response, chunk2

        mock_chat = Mock()
        mock_chat.stream.return_value = mock_async_stream()
        mock_chat.append = Mock()

        mock_create = Mock(return_value=mock_chat)
        client._async_client.chat = SimpleNamespace(create=mock_create)

        result = await client._achat_completion_impl(
            messages="Tell async story", model="grok-3", stream=True
        )

        assert isinstance(result, AsyncGenerator)
        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) == 2
        for chunk in chunks:
            assert isinstance(chunk, ChimericStreamChunk)

        # Test async streaming with tools enabled but no tool calls
        response_no_tools = SimpleNamespace()
        response_no_tools.tool_calls = None

        async def mock_async_stream_no_tool_calls():
            yield response_no_tools, chunk1
            yield response_no_tools, chunk2

        mock_chat_no_tools = Mock()
        mock_chat_no_tools.stream.return_value = mock_async_stream_no_tool_calls()
        mock_create.return_value = mock_chat_no_tools

        result = await client._achat_completion_impl(
            messages="Tell story",
            model="grok-3",
            stream=True,
            tools=[{"name": "dummy_tool", "description": "A dummy tool", "parameters": {}}],
        )

        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        assert len(chunks) == 2

        # Test async streaming with tools that get called
        chunk_tool = SimpleNamespace()
        chunk_tool.content = "Processing"

        chunk_final = SimpleNamespace()
        chunk_final.content = "Done"

        response_with_tools = SimpleNamespace()
        response_with_tools.tool_calls = [
            SimpleNamespace(
                id="async_call",
                function=SimpleNamespace(name="async_stream_tool", arguments='{"data": "test"}'),
            )
        ]

        response_final = SimpleNamespace()
        response_final.tool_calls = None

        # Create a complex async streaming scenario
        call_count = 0

        async def mock_async_tool_stream():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield response_with_tools, chunk_tool
            else:
                yield response_final, chunk_final

        mock_chat_async_tools = Mock()
        mock_chat_async_tools.stream.side_effect = mock_async_tool_stream
        mock_chat_async_tools.append = Mock()
        mock_create.return_value = mock_chat_async_tools

        def async_stream_tool(data: str) -> str:
            return f"Processed {data}"

        client.tool_manager.register(
            async_stream_tool, name="async_stream_tool", description="Async tool"
        )

        result = await client._achat_completion_impl(
            messages="Use async tool",
            model="grok-3",
            stream=True,
            tools=[{"name": "async_stream_tool", "description": "Async tool", "parameters": {}}],
        )

        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) == 2

        # Test async chunk without a content attribute
        chunk_no_content = SimpleNamespace()

        async def mock_async_stream_no_content():
            yield mock_response, chunk_no_content

        mock_chat_no_content = Mock()
        mock_chat_no_content.stream.return_value = mock_async_stream_no_content()
        mock_create.return_value = mock_chat_no_content

        result = await client._achat_completion_impl(messages="Test", model="grok-3", stream=True)

        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].common.delta == ""

    def test_error_handling_comprehensive(self, chimeric_grok_client):
        """Test error handling for sync and async operations."""
        client = chimeric_grok_client

        # Test sync error handling
        class MockAPIError(Exception):
            pass

        def mock_create_with_error(*args, **kwargs):
            raise MockAPIError("Sync API Error")

        client._client.chat = SimpleNamespace(create=mock_create_with_error)

        with pytest.raises(MockAPIError) as exc_info:
            client._chat_completion_impl(messages="Test message", model="grok-3")
        assert "Sync API Error" in str(exc_info.value)

    async def test_async_error_handling(self, chimeric_grok_client):
        """Test async error handling."""
        client = chimeric_grok_client

        class MockAsyncAPIError(Exception):
            pass

        def mock_async_create_with_error(*args, **kwargs):
            raise MockAsyncAPIError("Async API Error")

        client._async_client.chat = SimpleNamespace(create=mock_async_create_with_error)

        with pytest.raises(MockAsyncAPIError) as exc_info:
            await client._achat_completion_impl(messages="Test message", model="grok-3")
        assert "Async API Error" in str(exc_info.value)

    def test_file_upload_not_implemented(self, chimeric_grok_client):
        """Test that file upload raises NotImplementedError."""
        client = chimeric_grok_client

        with pytest.raises(NotImplementedError) as exc_info:
            client._upload_file(file_path="/tmp/test.txt")
        assert "Grok does not support file uploads" in str(exc_info.value)

    def test_tool_call_not_callable_error(self, chimeric_grok_client):
        """Test _process_function_call with non-callable tool to trigger ToolRegistrationError."""
        client = chimeric_grok_client

        # Mock a tool that isn't callable
        mock_tool = Mock()
        mock_tool.function = "not_callable"  # String, not callable
        client.tool_manager.get_tool = Mock(return_value=mock_tool)

        mock_tool_call = SimpleNamespace()
        mock_tool_call.id = "call_not_callable"
        mock_tool_call.function = SimpleNamespace()
        mock_tool_call.function.name = "non_callable_tool"
        mock_tool_call.function.arguments = "{}"

        with pytest.raises(ToolRegistrationError) as exc_info:
            client._process_function_call(mock_tool_call)
        assert "Tool 'non_callable_tool' is not callable" in str(exc_info.value)

    def test_stream_with_tool_calls_multiple_iterations(self, chimeric_grok_client):
        """Test _stream method with multiple tool call iterations to cover while loop"""
        client = chimeric_grok_client

        # Setup chunks for first response (with tool calls)
        chunk1 = SimpleNamespace()
        chunk1.content = "I'll"
        chunk2 = SimpleNamespace()
        chunk2.content = " help"

        # Setup chunks for final response (no tool calls)
        final_chunk1 = SimpleNamespace()
        final_chunk1.content = "Here's"
        final_chunk2 = SimpleNamespace()
        final_chunk2.content = " the result"

        # First response with tool calls
        response_with_tools = SimpleNamespace()
        response_with_tools.tool_calls = [
            SimpleNamespace(
                id="call_123",
                function=SimpleNamespace(name="test_tool", arguments='{"arg": "value"}'),
            )
        ]

        # Final response without tool calls
        response_final = SimpleNamespace()
        response_final.tool_calls = None

        call_count = 0

        def mock_complex_stream():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First stream with tool calls
                yield response_with_tools, chunk1
                yield response_with_tools, chunk2
            else:
                # Second stream without tool calls
                yield response_final, final_chunk1
                yield response_final, final_chunk2

        mock_chat = Mock()
        mock_chat.stream.side_effect = mock_complex_stream
        mock_chat.append = Mock()

        # Register test tool
        def test_tool(arg: str) -> str:
            return f"Tool result for {arg}"

        client.tool_manager.register(test_tool, name="test_tool", description="Test tool")

        # Test the streaming with tools enabled
        result = client._stream(mock_chat, tools_enabled=True)
        chunks = list(result)

        # Should have chunks from both iterations
        assert len(chunks) == 4
        assert all(isinstance(chunk, ChimericStreamChunk) for chunk in chunks)

        # Verify chat.append was called for both response and tool result
        assert mock_chat.append.call_count >= 2

    async def test_astream_with_tool_calls_multiple_iterations(self, chimeric_grok_client):
        """Test _astream method with multiple tool call iterations to cover async while loop"""
        client = chimeric_grok_client

        chunk1 = SimpleNamespace()
        chunk1.content = "Async"
        chunk2 = SimpleNamespace()
        chunk2.content = " processing"

        final_chunk1 = SimpleNamespace()
        final_chunk1.content = "Final"
        final_chunk2 = SimpleNamespace()
        final_chunk2.content = " result"

        response_with_tools = SimpleNamespace()
        response_with_tools.tool_calls = [
            SimpleNamespace(
                id="async_call",
                function=SimpleNamespace(name="async_tool", arguments='{"data": "test"}'),
            )
        ]

        response_final = SimpleNamespace()
        response_final.tool_calls = None

        call_count = 0

        async def mock_async_complex_stream():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield response_with_tools, chunk1
                yield response_with_tools, chunk2
            else:
                yield response_final, final_chunk1
                yield response_final, final_chunk2

        mock_chat = Mock()
        mock_chat.stream.side_effect = mock_async_complex_stream
        mock_chat.append = Mock()

        def async_tool(data: str) -> str:
            return f"Async result for {data}"

        client.tool_manager.register(async_tool, name="async_tool", description="Async test tool")

        result = client._astream(mock_chat, tools_enabled=True)
        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) == 4
        assert all(isinstance(chunk, ChimericStreamChunk) for chunk in chunks)
        assert mock_chat.append.call_count >= 2

    def test_create_chat_params_none_tools(self, chimeric_grok_client):
        """Test _create_chat_params with None tools"""
        client = chimeric_grok_client

        # Mock the chat.create method properly
        mock_chat = Mock()
        mock_create = Mock(return_value=mock_chat)
        client._client.chat = SimpleNamespace(create=mock_create)

        chat, messages = client._create_chat_params(messages="Hello", model="grok-3", tools=None)

        # Verify chat was created properly
        assert chat is not None
        assert len(messages) == 1

        # Verify the create method was called with tools=None and tool_choice=None
        mock_create.assert_called_with(model="grok-3", tools=None, tool_choice=None)

    def test_create_async_chat_params_none_tools(self, chimeric_grok_client):
        """Test _create_async_chat_params with None tools"""
        client = chimeric_grok_client

        # Mock the async chat.create method properly
        mock_chat = Mock()
        mock_create = Mock(return_value=mock_chat)
        client._async_client.chat = SimpleNamespace(create=mock_create)

        chat, messages = client._create_async_chat_params(
            messages="Hello", model="grok-3", tools=None
        )

        assert chat is not None
        assert len(messages) == 1

        mock_create.assert_called_with(model="grok-3", tools=None, tool_choice=None)
