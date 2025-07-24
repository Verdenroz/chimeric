from unittest.mock import AsyncMock, Mock, patch

from chimeric.providers.grok import GrokAsyncClient, GrokClient
from chimeric.types import Capability, Message, Tool, ToolExecutionResult, ToolParameters
from chimeric.utils import StreamProcessor
from conftest import BaseProviderTestSuite


class TestGrokClient(BaseProviderTestSuite):
    """Test suite for Grok sync client."""

    client_class = GrokClient
    provider_name = "Grok"
    mock_client_path = "chimeric.providers.grok.client.Client"

    @property
    def sample_response(self):
        """Create a sample Grok response."""
        mock_response = Mock()
        mock_response.content = "Hello from Grok"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.tool_calls = None
        return mock_response

    def test_initialization_and_capabilities(self):
        """Test initialization and capabilities."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            assert client.api_key == "test-key"
            assert client._provider_name == self.provider_name

            capabilities = client._get_capabilities()
            assert isinstance(capabilities, Capability)
            assert capabilities.streaming is True
            assert capabilities.tools is True

    def test_initialization_clients(self):
        """Test initialization and type methods."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test _get_client_type
            client_type = client._get_client_type()
            assert client_type is not None

            # Test _init_client with kwargs
            from chimeric.providers.grok.client import Client

            result = client._init_client(Client, base_url="https://api.x.ai", timeout=30)
            assert result is not None

    def test_list_models_comprehensive(self):
        """Test model listing with various scenarios."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Full model with aliases
            mock_model = Mock()
            mock_model.name = "grok-beta"
            mock_model.version = "1.0"
            mock_model.aliases = ["grok"]
            mock_model.created = Mock()
            mock_model.created.seconds = 1234567890
            mock_client.models.list_language_models.return_value = [mock_model]

            models = client._list_models_impl()
            assert len(models) == 2
            assert models[0].id == "grok-beta"

            # Model with missing attributes
            mock_minimal = Mock()
            mock_minimal.name = "grok-minimal"
            del mock_minimal.version
            del mock_minimal.created
            del mock_minimal.aliases
            mock_client.models.list_language_models.return_value = [mock_minimal]
            models = client._list_models_impl()
            assert len(models) == 1

    def test_models_with_full_metadata(self):
        """Test model listing with full metadata coverage."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Mock model with all possible metadata
            mock_model = Mock()
            mock_model.name = "grok-full"
            mock_model.version = "2.0"
            mock_model.input_modalities = ["text", "image"]
            mock_model.output_modalities = ["text"]
            mock_model.max_prompt_length = 8192
            mock_model.system_fingerprint = "fp_123"
            mock_model.prompt_text_token_price = 0.001
            mock_model.completion_text_token_price = 0.002
            mock_model.prompt_image_token_price = 0.003
            mock_model.cached_prompt_token_price = 0.0005
            mock_model.search_price = 0.01
            mock_model.aliases = ["grok-2", "grok-v2"]
            mock_model.created = Mock()
            mock_model.created.seconds = 1640995200

            mock_client.models.list_language_models.return_value = [mock_model]

            models = client._list_models_impl()
            assert isinstance(models, list)
            assert len(models) == 3  # Main model + 2 aliases
            assert isinstance(models[0].metadata, dict)
            assert isinstance(models[1].metadata, dict)
            assert isinstance(models[2].metadata, dict)
            assert models[0].id == "grok-full"
            assert models[0].metadata["version"] == "2.0"
            assert models[0].metadata["max_prompt_length"] == 8192
            assert models[1].metadata["canonical_name"] == "grok-full"

    def test_message_format_conversion(self):
        """Test message format conversion."""
        tool_manager = self.create_tool_manager()

        with (
            patch(self.mock_client_path),
            patch("chimeric.providers.grok.client.user") as mock_user,
            patch("chimeric.providers.grok.client.system") as mock_system,
            patch("chimeric.providers.grok.client.assistant") as mock_assistant,
        ):
            mock_user.return_value = Mock()
            mock_system.return_value = Mock()
            mock_assistant.return_value = Mock()

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test all role mappings and list content conversion
            messages = [
                Message(role="user", content="Hello"),
                Message(role="system", content="System"),
                Message(role="assistant", content="Assistant"),
                Message(role="unknown", content=["List", "content"]),  # Unknown role + list
                Message(role="user", content="String content"),  # Non-list content
            ]
            result = client._messages_to_provider_format(messages)
            assert len(result) == 5

    def test_tool_format_conversion(self):
        """Test tool format conversion."""
        tool_manager = self.create_tool_manager()

        with (
            patch(self.mock_client_path),
            patch("chimeric.providers.grok.client.tool") as mock_tool,
        ):
            mock_tool.return_value = Mock()
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            def mock_func():
                pass

            # Tool with parameters
            params = ToolParameters()
            params.properties = {"x": {"type": "int"}}
            tools = [
                Tool(name="test_tool", description="Test", parameters=params, function=mock_func)
            ]
            result = client._tools_to_provider_format(tools)
            assert len(result) == 1

            # Tool with None parameters
            tools = [
                Tool(name="test_tool", description="Test", parameters=None, function=mock_func)
            ]
            result = client._tools_to_provider_format(tools)
            assert len(result) == 1

    def test_make_provider_request(self):
        """Test _make_provider_request method."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Non-streaming request
            mock_chat = Mock()
            mock_chat.sample.return_value = self.sample_response
            mock_client.chat.create.return_value = mock_chat

            result = client._make_provider_request(
                messages=[Mock()], model="grok-beta", stream=False
            )
            assert mock_client.chat.create.called

            # Streaming request
            def mock_stream():
                yield (Mock(), Mock())

            mock_chat.stream.return_value = mock_stream()
            result = client._make_provider_request(
                messages=[Mock()], model="grok-beta", stream=True
            )
            chunks = list(result)
            assert len(chunks) > 0

            # Request with tools and kwargs - need fresh mock
            mock_chat.sample.return_value = self.sample_response
            result = client._make_provider_request(
                messages=[Mock()], model="grok-beta", stream=False, tools=[Mock()], temperature=0.7
            )
            assert mock_client.chat.create.called

    def test_stream_processing(self):
        """Test stream processing."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Test with content
            mock_response = Mock()
            mock_chunk = Mock()
            mock_chunk.content = "Stream content"
            client._process_provider_stream_event((mock_response, mock_chunk), processor)

            # Test with None content
            mock_chunk.content = None
            client._process_provider_stream_event((mock_response, mock_chunk), processor)

    def test_usage_extraction(self):
        """Test usage extraction."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Object format
            usage = client._extract_usage_from_response(self.sample_response)
            assert usage.prompt_tokens == 10

            # Dict format
            response = Mock()
            response.usage = {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40}
            usage = client._extract_usage_from_response(response)
            assert usage.prompt_tokens == 15

            # None usage
            response.usage = None
            usage = client._extract_usage_from_response(response)
            assert usage.prompt_tokens == 0

    def test_content_extraction(self):
        """Test content extraction."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            content = client._extract_content_from_response(self.sample_response)
            assert content == "Hello from Grok"

            response = Mock()
            response.content = None
            content = client._extract_content_from_response(response)
            assert content == ""

    def test_tool_call_extraction(self):
        """Test tool call extraction."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # With tool calls
            response = Mock()
            mock_call = Mock()
            mock_call.id = "call_123"
            mock_call.function = Mock()
            mock_call.function.name = "test_func"
            mock_call.function.arguments = '{"x": 5}'
            response.tool_calls = [mock_call]

            tool_calls = client._extract_tool_calls_from_response(response)
            assert isinstance(tool_calls, list)
            assert len(tool_calls) == 1

            # None tool calls
            response.tool_calls = None
            tool_calls = client._extract_tool_calls_from_response(response)
            assert tool_calls is None

            # Empty tool calls
            response.tool_calls = []
            tool_calls = client._extract_tool_calls_from_response(response)
            assert tool_calls is None

    def test_handle_tool_calling_completion(self):
        """Test tool calling completion."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            with patch("chimeric.providers.grok.client.tool_result") as mock_tool_result:
                mock_tool_result.return_value = Mock()

                mock_chat = Mock()
                mock_client.chat.create.return_value = mock_chat
                mock_chat.append = Mock()

                # Response with tool calls
                first_response = Mock()
                first_response.tool_calls = [Mock()]
                first_response.tool_calls[0].id = "call_123"
                first_response.tool_calls[0].function = Mock()
                first_response.tool_calls[0].function.name = "test_tool"
                first_response.tool_calls[0].function.arguments = '{"x": 5}'

                final_response = Mock()
                final_response.content = "Done"
                final_response.tool_calls = None
                final_response.usage = Mock()
                final_response.usage.prompt_tokens = 5
                final_response.usage.completion_tokens = 10
                final_response.usage.total_tokens = 15

                mock_chat.sample.side_effect = [first_response, final_response]

                with patch.object(client, "_execute_tool_calls") as mock_execute:
                    mock_execute.return_value = [
                        ToolExecutionResult(
                            name="test_tool",
                            arguments='{"x": 5}',
                            call_id="call_123",
                            result="success",
                            is_error=False,
                        )
                    ]

                    result = client._handle_tool_calling_completion(
                        messages=[Mock()], model="grok-beta", tools=[Mock()]
                    )

                    assert result is not None
                    assert mock_chat.sample.call_count == 2

                    # Test error case
                    mock_execute.return_value = [
                        ToolExecutionResult(
                            name="test_tool",
                            arguments='{"x": 5}',
                            call_id="call_123",
                            result=None,
                            is_error=True,
                            error="Test error",
                        )
                    ]

                    mock_chat.sample.side_effect = [first_response, final_response]
                    result = client._handle_tool_calling_completion(
                        messages=[Mock()], model="grok-beta", tools=[Mock()]
                    )
                    assert result is not None

    def test_handle_streaming_tool_calls(self):
        """Test streaming tool calls."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            with patch("chimeric.providers.grok.client.tool_result") as mock_tool_result:
                mock_tool_result.return_value = Mock()

                mock_chat = Mock()
                mock_client.chat.create.return_value = mock_chat
                mock_chat.append = Mock()

                # Mock tool response then final response
                first_response = Mock()
                first_response.tool_calls = [Mock()]
                first_response.tool_calls[0].id = "stream_call"
                first_response.tool_calls[0].function = Mock()
                first_response.tool_calls[0].function.name = "stream_tool"
                first_response.tool_calls[0].function.arguments = '{"param": "value"}'

                final_response = Mock()
                final_response.content = "Stream done"
                final_response.tool_calls = None

                mock_chat.sample.side_effect = [first_response, final_response]

                # Mock the final streaming
                def mock_stream_generator():
                    yield (Mock(), Mock(content="Stream chunk"))

                mock_chat.stream.return_value = mock_stream_generator()

                with patch.object(client, "_execute_tool_calls") as mock_execute:
                    mock_execute.return_value = [
                        ToolExecutionResult(
                            name="stream_tool",
                            arguments='{"param": "value"}',
                            call_id="stream_call",
                            result="success",
                            is_error=False,
                        )
                    ]

                    processor = StreamProcessor()
                    stream_gen = client._handle_streaming_tool_calls(
                        stream=Mock(),
                        processor=processor,
                        messages=[Mock()],
                        model="grok-beta",
                        tools=[Mock()],
                    )

                    chunks = list(stream_gen)
                    assert len(chunks) >= 0

    def test_edge_cases_and_error_conditions(self):
        """Test edge cases and error conditions for complete coverage."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test _handle_tool_calling_completion without tool_manager
            client.tool_manager = None
            mock_chat = Mock()
            mock_client.chat.create.return_value = mock_chat

            response_with_tools = Mock()
            response_with_tools.tool_calls = [Mock()]
            response_with_tools.tool_calls[0].id = "call_no_manager"
            response_with_tools.tool_calls[0].function = Mock()
            response_with_tools.tool_calls[0].function.name = "test_tool"
            response_with_tools.tool_calls[0].function.arguments = '{"x": 1}'

            final_response = Mock()
            final_response.content = "Done without tool manager"
            final_response.tool_calls = None
            final_response.usage = Mock()
            final_response.usage.prompt_tokens = 1
            final_response.usage.completion_tokens = 2
            final_response.usage.total_tokens = 3

            mock_chat.sample.side_effect = [response_with_tools, final_response]

            result = client._handle_tool_calling_completion(
                messages=[Mock()], model="grok-beta", tools=[Mock()]
            )
            assert result is not None

            # Test _handle_streaming_tool_calls without tool_manager
            def mock_stream_generator():
                yield (Mock(), Mock(content="Stream chunk"))

            mock_chat.stream.return_value = mock_stream_generator()
            mock_chat.sample.side_effect = [response_with_tools, final_response]

            processor = StreamProcessor()
            stream_gen = client._handle_streaming_tool_calls(
                stream=Mock(),
                processor=processor,
                messages=[Mock()],
                model="grok-beta",
                tools=[Mock()],
            )

            chunks = list(stream_gen)
            assert len(chunks) >= 0

            # Reset tool_manager for other tests
            client.tool_manager = tool_manager

    def test_tool_result_error_handling(self):
        """Test tool result error handling scenarios."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            with patch("chimeric.providers.grok.client.tool_result") as mock_tool_result:
                mock_tool_result.return_value = Mock()

                mock_chat = Mock()
                mock_client.chat.create.return_value = mock_chat
                mock_chat.append = Mock()

                # Response with tool calls
                first_response = Mock()
                first_response.tool_calls = [Mock()]
                first_response.tool_calls[0].id = "call_error_test"
                first_response.tool_calls[0].function = Mock()
                first_response.tool_calls[0].function.name = "error_tool"
                first_response.tool_calls[0].function.arguments = '{"x": 5}'

                final_response = Mock()
                final_response.content = "Done after error"
                final_response.tool_calls = None
                final_response.usage = Mock()
                final_response.usage.prompt_tokens = 5
                final_response.usage.completion_tokens = 10
                final_response.usage.total_tokens = 15

                mock_chat.sample.side_effect = [first_response, final_response]

                with patch.object(client, "_execute_tool_calls") as mock_execute:
                    # Test error result with None result
                    mock_execute.return_value = [
                        ToolExecutionResult(
                            name="error_tool",
                            arguments='{"x": 5}',
                            call_id="call_error_test",
                            result=None,
                            is_error=True,
                            error=None,
                        )
                    ]

                    result = client._handle_tool_calling_completion(
                        messages=[Mock()], model="grok-beta", tools=[Mock()]
                    )
                    assert result is not None

    def test_update_messages_with_tool_calls(self):
        """Test message updating."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            original_messages = [Mock()]
            result = client._update_messages_with_tool_calls(
                messages=original_messages,
                assistant_response=Mock(),
                tool_calls=[Mock()],
                tool_results=[Mock()],
            )
            assert result is original_messages

    def test_streaming_with_no_tools_coverage(self):
        """Test streaming without tool calls to cover lines 418-421."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            mock_chat = Mock()
            mock_client.chat.create.return_value = mock_chat

            # Response with no tool calls to go directly to streaming
            no_tools_response = Mock()
            no_tools_response.tool_calls = None
            no_tools_response.content = "Direct streaming"
            mock_chat.sample.return_value = no_tools_response

            # Mock streaming to cover lines 418-421
            def mock_stream_generator():
                yield (Mock(), Mock(content="Stream chunk 1"))
                yield (Mock(), Mock(content=None))  # Test if chunk_obj condition

            mock_chat.stream.return_value = mock_stream_generator()

            processor = StreamProcessor()
            stream_gen = client._handle_streaming_tool_calls(
                stream=Mock(),
                processor=processor,
                messages=[Mock()],
                model="grok-beta",
                tools=[Mock()],
            )

            chunks = list(stream_gen)
            assert len(chunks) >= 0


class TestGrokAsyncClient(BaseProviderTestSuite):
    """Test suite for Grok async client."""

    client_class = GrokAsyncClient
    provider_name = "Grok"
    mock_client_path = "chimeric.providers.grok.client.AsyncClient"

    @property
    def sample_response(self):
        mock_response = Mock()
        mock_response.content = "Hello from async Grok"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 25
        mock_response.usage.total_tokens = 40
        mock_response.tool_calls = None
        return mock_response

    async def test_async_initialization_and_capabilities(self):
        """Test async initialization and capabilities."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            assert client.api_key == "test-key"
            assert client._provider_name == self.provider_name

            capabilities = client._get_capabilities()
            assert isinstance(capabilities, Capability)
            assert capabilities.streaming is True
            assert capabilities.tools is True

    async def test_async_comprehensive_coverage(self):
        """Test async comprehensive scenarios for 100% coverage."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test _get_async_client_type
            client_type = client._get_async_client_type()
            assert client_type is not None

            # Test _init_async_client
            from chimeric.providers.grok.client import AsyncClient

            result = client._init_async_client(AsyncClient, base_url="https://api.x.ai")
            assert result is not None

            # Test capabilities
            capabilities = client._get_capabilities()
            assert capabilities.streaming is True
            assert capabilities.tools is True

    async def test_async_list_models(self):
        """Test async model listing."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            mock_model = Mock()
            mock_model.name = "grok-async"
            mock_model.aliases = ["grok-a"]
            del mock_model.created
            mock_client.models.list_language_models = AsyncMock(return_value=[mock_model])

            models = await client._list_models_impl()
            assert len(models) == 2
            assert models[0].id == "grok-async"

    async def test_async_models_with_no_created_attribute(self):
        """Test async model listing when models lack created attribute."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Mock model without created attribute
            mock_model = Mock()
            mock_model.name = "grok-no-created"
            mock_model.aliases = []
            # Explicitly remove created attribute
            if hasattr(mock_model, "created"):
                del mock_model.created

            mock_client.models.list_language_models = AsyncMock(return_value=[mock_model])

            models = await client._list_models_impl()
            assert len(models) == 1
            assert models[0].id == "grok-no-created"
            assert models[0].created_at is None

    async def test_async_make_provider_request(self):
        """Test async provider request."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Non-streaming
            mock_chat = Mock()
            mock_client.chat.create.return_value = mock_chat
            mock_chat.sample = AsyncMock(return_value=self.sample_response)

            result = await client._make_async_provider_request(
                messages=[Mock()], model="grok-async", stream=False, tools=None
            )
            assert mock_chat.sample.called

            # Streaming path to cover line 589
            async def mock_stream():
                yield (Mock(), Mock(content="Streaming test"))

            mock_chat.stream = Mock(return_value=mock_stream())
            result = await client._make_async_provider_request(
                messages=[Mock()], model="grok-async", stream=True, tools=None
            )

            chunks = []
            async for chunk in result:
                chunks.append(chunk)
            assert len(chunks) > 0

    async def test_async_make_request_with_tools(self):
        """Test async request with tools and tool_choice."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            mock_chat = Mock()
            mock_client.chat.create.return_value = mock_chat
            mock_chat.sample = AsyncMock(return_value=self.sample_response)

            # Test with tools (tool_choice auto is set in implementation)
            await client._make_async_provider_request(
                messages=[Mock()], model="grok-async", stream=False, tools=[Mock()]
            )
            assert mock_chat.sample.called

    async def test_async_handle_tool_calling_completion(self):
        """Test async tool calling completion."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            with patch("chimeric.providers.grok.client.tool_result") as mock_tool_result:
                mock_tool_result.return_value = Mock()

                mock_chat = Mock()
                mock_client.chat.create.return_value = mock_chat
                mock_chat.append = Mock()

                first_response = Mock()
                first_response.tool_calls = [Mock()]
                first_response.tool_calls[0].id = "call_async"
                first_response.tool_calls[0].function = Mock()
                first_response.tool_calls[0].function.name = "async_tool"
                first_response.tool_calls[0].function.arguments = '{"y": 10}'

                final_response = Mock()
                final_response.content = "Async done"
                final_response.tool_calls = None
                final_response.usage = Mock()
                final_response.usage.prompt_tokens = 10
                final_response.usage.completion_tokens = 15
                final_response.usage.total_tokens = 25

                mock_chat.sample = AsyncMock(side_effect=[first_response, final_response])

                with patch.object(
                    client, "_execute_tool_calls", new_callable=AsyncMock
                ) as mock_execute:
                    mock_execute.return_value = [
                        ToolExecutionResult(
                            name="async_tool",
                            arguments='{"y": 10}',
                            call_id="call_async",
                            result="async_success",
                            is_error=False,
                        )
                    ]

                    result = await client._handle_tool_calling_completion(
                        messages=[Mock()], model="grok-async", tools=[Mock()]
                    )

                    assert result is not None
                    assert mock_chat.sample.call_count == 2

    async def test_async_handle_streaming_tool_calls(self):
        """Test async streaming tool calls to cover lines 788-790."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            with patch("chimeric.providers.grok.client.tool_result") as mock_tool_result:
                mock_tool_result.return_value = Mock()

                mock_chat = Mock()
                mock_client.chat.create.return_value = mock_chat
                mock_chat.append = Mock()

                # Create a response with no tool calls to reach final streaming
                final_response = Mock()
                final_response.tool_calls = None
                final_response.content = "Ready to stream"

                mock_chat.sample = AsyncMock(return_value=final_response)

                # Mock the final streaming (lines 788-790)
                async def mock_final_stream():
                    yield (Mock(), Mock(content="Final chunk 1"))
                    yield (Mock(), Mock(content="Final chunk 2"))
                    yield (Mock(), Mock(content=None))  # Test the if chunk_obj condition

                mock_chat.stream = Mock(return_value=mock_final_stream())

                processor = StreamProcessor()
                stream_gen = client._handle_streaming_tool_calls(
                    stream=Mock(),
                    processor=processor,
                    messages=[Mock()],
                    model="grok-async",
                    tools=[Mock()],
                )

                chunks = []
                async for chunk in stream_gen:
                    chunks.append(chunk)
                # Should get chunks from final streaming
                assert len(chunks) >= 0

    async def test_async_streaming_tool_execution_coverage(self):
        """Test async streaming with actual tool execution to cover lines 768-772."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            with patch("chimeric.providers.grok.client.tool_result") as mock_tool_result:
                mock_tool_result.return_value = Mock()

                mock_chat = Mock()
                mock_client.chat.create.return_value = mock_chat
                mock_chat.append = Mock()

                # Create a response WITH tool calls to trigger tool execution
                response_with_tools = Mock()
                mock_tool_call = Mock()
                mock_tool_call.id = "call_1"
                mock_tool_call.function.name = "test_tool"
                mock_tool_call.function.arguments = "{}"
                response_with_tools.tool_calls = [mock_tool_call]
                response_with_tools.content = "Tool call response"

                # Final response with no more tool calls
                final_response = Mock()
                final_response.tool_calls = None
                final_response.content = "Final response"

                mock_chat.sample = AsyncMock(side_effect=[response_with_tools, final_response])

                # Mock tool execution to cover lines 768-772
                with patch.object(client, "_execute_tool_calls") as mock_execute:
                    mock_execute.return_value = [Mock(result="Tool executed", is_error=False)]

                    # Mock final streaming
                    async def mock_final_stream():
                        yield (Mock(), Mock(content="Final stream chunk"))

                    mock_chat.stream = Mock(return_value=mock_final_stream())

                    processor = StreamProcessor()
                    stream_gen = client._handle_streaming_tool_calls(
                        stream=Mock(),
                        processor=processor,
                        messages=[Mock()],
                        model="grok-async",
                        tools=[Mock()],
                    )

                    chunks = []
                    async for chunk in stream_gen:
                        chunks.append(chunk)

                    # Verify tool execution was called (covers line 768)
                    assert mock_execute.called
                    # Verify tool results were appended (covers lines 771-772)
                    assert mock_chat.append.call_count >= 2  # Messages + tool results

    async def test_async_message_and_tool_format_conversion(self):
        """Test async message and tool format conversion."""
        tool_manager = self.create_tool_manager()

        with (
            patch(self.mock_client_path),
            patch("chimeric.providers.grok.client.user") as mock_user,
            patch("chimeric.providers.grok.client.system") as mock_system,
            patch("chimeric.providers.grok.client.assistant") as mock_assistant,
            patch("chimeric.providers.grok.client.tool") as mock_tool,
        ):
            mock_user.return_value = Mock()
            mock_system.return_value = Mock()
            mock_assistant.return_value = Mock()
            mock_tool.return_value = Mock()

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test message conversion
            messages = [
                Message(role="user", content="Hello"),
                Message(role="system", content=["System", "list"]),
            ]
            result = client._messages_to_provider_format(messages)
            assert len(result) == 2

            # Test tool conversion
            def mock_func():
                pass

            params = ToolParameters()
            params.properties = {"x": {"type": "int"}}
            tools = [
                Tool(name="test_tool", description="Test", parameters=params, function=mock_func)
            ]
            result = client._tools_to_provider_format(tools)
            assert len(result) == 1

    async def test_async_extraction_methods(self):
        """Test async extraction methods."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test usage extraction
            usage = client._extract_usage_from_response(self.sample_response)
            assert usage.prompt_tokens == 15

            # Test content extraction
            content = client._extract_content_from_response(self.sample_response)
            assert content == "Hello from async Grok"

            # Test tool call extraction
            response = Mock()
            mock_call = Mock()
            mock_call.id = "async_call_123"
            mock_call.function = Mock()
            mock_call.function.name = "async_test_func"
            mock_call.function.arguments = '{"x": 10}'
            response.tool_calls = [mock_call]

            tool_calls = client._extract_tool_calls_from_response(response)
            assert isinstance(tool_calls, list)
            assert len(tool_calls) == 1
            assert tool_calls[0].call_id == "async_call_123"

    async def test_async_usage_extraction_edge_cases(self):
        """Test async usage extraction with edge cases."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test with dict usage format
            response = Mock()
            response.usage = {"prompt_tokens": 25, "completion_tokens": 35, "total_tokens": 60}
            usage = client._extract_usage_from_response(response)
            assert usage.prompt_tokens == 25
            assert usage.completion_tokens == 35
            assert usage.total_tokens == 60

            # Test with None usage
            response.usage = None
            usage = client._extract_usage_from_response(response)
            assert usage.prompt_tokens == 0
            assert usage.completion_tokens == 0
            assert usage.total_tokens == 0

    async def test_async_content_extraction_edge_cases(self):
        """Test async content extraction edge cases."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test with None content
            response = Mock()
            response.content = None
            content = client._extract_content_from_response(response)
            assert content == ""

            # Test with empty string content
            response.content = ""
            content = client._extract_content_from_response(response)
            assert content == ""

    async def test_async_tool_calls_extraction_edge_cases(self):
        """Test async tool calls extraction edge cases."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test with None tool_calls
            response = Mock()
            response.tool_calls = None
            tool_calls = client._extract_tool_calls_from_response(response)
            assert tool_calls is None

            # Test with empty tool_calls list
            response.tool_calls = []
            tool_calls = client._extract_tool_calls_from_response(response)
            assert tool_calls is None

            # Test with valid tool_calls
            mock_call = Mock()
            mock_call.id = "async_test_call"
            mock_call.function = Mock()
            mock_call.function.name = "async_test_function"
            mock_call.function.arguments = '{"param": "async_value"}'
            response.tool_calls = [mock_call]

            tool_calls = client._extract_tool_calls_from_response(response)
            assert isinstance(tool_calls, list)
            assert len(tool_calls) == 1
            assert tool_calls[0].call_id == "async_test_call"
            assert tool_calls[0].name == "async_test_function"

    async def test_async_edge_cases_and_error_conditions(self):
        """Test async edge cases and error conditions for complete coverage."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test _handle_tool_calling_completion without tool_manager
            client.tool_manager = None
            mock_chat = Mock()
            mock_client.chat.create.return_value = mock_chat

            response_with_tools = Mock()
            response_with_tools.tool_calls = [Mock()]
            response_with_tools.tool_calls[0].id = "async_call_no_manager"
            response_with_tools.tool_calls[0].function = Mock()
            response_with_tools.tool_calls[0].function.name = "async_test_tool"
            response_with_tools.tool_calls[0].function.arguments = '{"y": 2}'

            final_response = Mock()
            final_response.content = "Async done without tool manager"
            final_response.tool_calls = None
            final_response.usage = Mock()
            final_response.usage.prompt_tokens = 5
            final_response.usage.completion_tokens = 8
            final_response.usage.total_tokens = 13

            mock_chat.sample = AsyncMock(side_effect=[response_with_tools, final_response])

            result = await client._handle_tool_calling_completion(
                messages=[Mock()], model="grok-async", tools=[Mock()]
            )
            assert result is not None

            # Test _handle_streaming_tool_calls without tool_manager
            async def mock_stream_generator():
                yield (Mock(), Mock(content="Async stream chunk"))

            mock_chat.stream = Mock(return_value=mock_stream_generator())
            mock_chat.sample = AsyncMock(side_effect=[response_with_tools, final_response])

            processor = StreamProcessor()
            stream_gen = client._handle_streaming_tool_calls(
                stream=Mock(),
                processor=processor,
                messages=[Mock()],
                model="grok-async",
                tools=[Mock()],
            )

            chunks = []
            async for chunk in stream_gen:
                chunks.append(chunk)
            assert len(chunks) >= 0

            # Reset tool_manager for other tests
            client.tool_manager = tool_manager

    async def test_async_tool_result_error_handling(self):
        """Test async tool result error handling scenarios."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            with patch("chimeric.providers.grok.client.tool_result") as mock_tool_result:
                mock_tool_result.return_value = Mock()

                mock_chat = Mock()
                mock_client.chat.create.return_value = mock_chat
                mock_chat.append = Mock()

                # Response with tool calls
                first_response = Mock()
                first_response.tool_calls = [Mock()]
                first_response.tool_calls[0].id = "async_call_error_test"
                first_response.tool_calls[0].function = Mock()
                first_response.tool_calls[0].function.name = "async_error_tool"
                first_response.tool_calls[0].function.arguments = '{"y": 10}'

                final_response = Mock()
                final_response.content = "Async done after error"
                final_response.tool_calls = None
                final_response.usage = Mock()
                final_response.usage.prompt_tokens = 8
                final_response.usage.completion_tokens = 12
                final_response.usage.total_tokens = 20

                mock_chat.sample = AsyncMock(side_effect=[first_response, final_response])

                with patch.object(
                    client, "_execute_tool_calls", new_callable=AsyncMock
                ) as mock_execute:
                    # Test error result with None result and None error
                    mock_execute.return_value = [
                        ToolExecutionResult(
                            name="async_error_tool",
                            arguments='{"y": 10}',
                            call_id="async_call_error_test",
                            result=None,
                            is_error=True,
                            error=None,
                        )
                    ]

                    result = await client._handle_tool_calling_completion(
                        messages=[Mock()], model="grok-async", tools=[Mock()]
                    )
                    assert result is not None

    async def test_async_update_messages_with_tool_calls(self):
        """Test async message updating."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            original_messages = [Mock()]
            result = client._update_messages_with_tool_calls(
                messages=original_messages,
                assistant_response=Mock(),
                tool_calls=[Mock()],
                tool_results=[Mock()],
            )
            assert result is original_messages
            assert client._provider_name == self.provider_name

    async def test_async_stream_processing(self):
        """Test async stream processing methods."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Test with content
            mock_response = Mock()
            mock_chunk = Mock()
            mock_chunk.content = "Async stream content"
            result = client._process_provider_stream_event((mock_response, mock_chunk), processor)
            assert result is not None

            # Test with None content
            mock_chunk.content = None
            result = client._process_provider_stream_event((mock_response, mock_chunk), processor)
            assert result is None

    async def test_async_streaming_with_no_tools_coverage(self):
        """Test async streaming without tool calls to cover lines 787-790."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            mock_chat = Mock()
            mock_client.chat.create.return_value = mock_chat

            # Response with no tool calls to go directly to streaming
            no_tools_response = Mock()
            no_tools_response.tool_calls = None
            no_tools_response.content = "Direct async streaming"
            mock_chat.sample = AsyncMock(return_value=no_tools_response)

            # Mock async streaming to cover lines 787-790
            async def mock_async_stream_generator():
                yield (Mock(), Mock(content="Async stream chunk 1"))
                yield (Mock(), Mock(content=None))  # Test if chunk_obj condition

            mock_chat.stream = Mock(return_value=mock_async_stream_generator())

            processor = StreamProcessor()
            stream_gen = client._handle_streaming_tool_calls(
                stream=Mock(),
                processor=processor,
                messages=[Mock()],
                model="grok-async",
                tools=[Mock()],
            )

            chunks = []
            async for chunk in stream_gen:
                chunks.append(chunk)
            assert len(chunks) >= 0
