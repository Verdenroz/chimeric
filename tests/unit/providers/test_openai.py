from unittest.mock import AsyncMock, MagicMock, Mock, patch

from openai.types.responses import Response, ResponseFunctionToolCall, ResponseStreamEvent

from chimeric.providers.openai import OpenAIAsyncClient, OpenAIClient
from chimeric.types import Capability, Message, Tool, ToolCall, ToolExecutionResult, ToolParameters
from chimeric.utils import StreamProcessor
from conftest import BaseProviderTestSuite


class TestOpenAIClient(BaseProviderTestSuite):
    """Test suite for OpenAI sync client"""

    client_class = OpenAIClient
    provider_name = "OpenAI"
    mock_client_path = "chimeric.providers.openai.client.OpenAI"

    @property
    def sample_response(self):
        """Create a sample OpenAI response."""
        response = Mock(spec=Response)
        response.output_text = "Hello there"
        response.output = []
        response.usage = Mock(input_tokens=10, output_tokens=20, total_tokens=30)
        return response

    @property
    def sample_stream_events(self):
        """Create sample OpenAI stream events."""
        events = []

        # Text delta event
        text_event = Mock(spec=ResponseStreamEvent)
        text_event.type = "response.output_text.delta"
        text_event.delta = "Hello"
        events.append(text_event)

        # Tool call start event
        tool_start = Mock(spec=ResponseStreamEvent)
        tool_start.type = "response.output_item.added"
        tool_item = Mock()
        tool_item.type = "function_call"
        tool_item.id = "fc_123"
        tool_item.call_id = "call_456"
        tool_item.name = "test_tool"
        tool_start.item = tool_item
        events.append(tool_start)

        # Non-tool output item event
        non_tool_start = Mock(spec=ResponseStreamEvent)
        non_tool_start.type = "response.output_item.added"
        non_tool_item = Mock()
        non_tool_item.type = "text"  # Not "function_call"
        non_tool_start.item = non_tool_item
        events.append(non_tool_start)

        # Tool call arguments delta
        args_delta = Mock(spec=ResponseStreamEvent)
        args_delta.type = "response.function_call_arguments.delta"
        args_delta.item_id = "fc_123"
        args_delta.delta = '{"x": 1'
        events.append(args_delta)

        # More arguments
        args_delta2 = Mock(spec=ResponseStreamEvent)
        args_delta2.type = "response.function_call_arguments.delta"
        args_delta2.item_id = "fc_123"
        args_delta2.delta = "0}"
        events.append(args_delta2)

        # Tool call complete
        args_done = Mock(spec=ResponseStreamEvent)
        args_done.type = "response.function_call_arguments.done"
        args_done.item_id = "fc_123"
        events.append(args_done)

        # Function arguments done with None item_id
        args_done_none = Mock(spec=ResponseStreamEvent)
        args_done_none.type = "response.function_call_arguments.done"
        args_done_none.item_id = None
        events.append(args_done_none)

        # Response completed
        complete = Mock(spec=ResponseStreamEvent)
        complete.type = "response.completed"
        complete.response = Mock(status="completed")
        events.append(complete)

        return events

    # ===== Initialization Tests =====

    def test_client_initialization_success(self):
        """Test successful client initialization with all parameters."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_openai:
            client = self.client_class(
                api_key="test-key", tool_manager=tool_manager, organization="test-org", timeout=30
            )

            assert client.api_key == "test-key"
            assert client.tool_manager == tool_manager
            assert client._provider_name == self.provider_name
            mock_openai.assert_called_once_with(
                api_key="test-key", organization="test-org", timeout=30
            )

    def test_client_initialization_minimal(self):
        """Test client initialization with minimal parameters."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            assert client.api_key == "test-key"

    # ===== Capability Tests =====

    def test_capabilities(self):
        """Test provider capabilities."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            capabilities = client._get_capabilities()

            assert isinstance(capabilities, Capability)
            assert capabilities.streaming is True
            assert capabilities.tools is True

    # ===== Model Listing Tests =====

    def test_list_models_success(self):
        """Test successful model listing."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            # Mock model list response
            mock_model = Mock(id="gpt-4", owned_by="openai", created=1234567890)
            mock_client.models.list.return_value = [mock_model]

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            models = client._list_models_impl()

            assert len(models) == 1
            assert models[0].id == "gpt-4"
            assert models[0].owned_by == "openai"
            assert models[0].created_at == 1234567890

    def test_list_models_empty(self):
        """Test model listing with no models."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.models.list.return_value = []

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            models = client._list_models_impl()

            assert models == []

    # ===== Message Formatting Tests =====

    def test_messages_to_provider_format_regular(self):
        """Test formatting regular messages."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            messages = [
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there"),
            ]

            formatted = client._messages_to_provider_format(messages)

            assert len(formatted) == 2
            assert formatted[0]["role"] == "user"
            assert formatted[0]["content"] == "Hello"

    def test_messages_to_provider_format_with_tool_results(self):
        """Test formatting messages with tool results."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            messages = [
                Message(role="user", content="Hello"),
                Message(role="tool", content="Result", tool_call_id="call_123"),
            ]

            formatted = client._messages_to_provider_format(messages)

            assert len(formatted) == 2
            assert formatted[1]["type"] == "function_call_output"
            assert formatted[1]["call_id"] == "call_123"
            assert formatted[1]["output"] == "Result"

    def test_messages_to_provider_format_with_tool_calls(self):
        """Test formatting messages with assistant tool calls."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            tool_call = ToolCall(call_id="call_123", name="test_tool", arguments='{"x": 10}')
            messages = [Message(role="assistant", content="", tool_calls=[tool_call])]

            formatted = client._messages_to_provider_format(messages)

            assert len(formatted) == 1
            assert formatted[0]["type"] == "function_call"
            assert formatted[0]["call_id"] == "call_123"
            assert formatted[0]["name"] == "test_tool"

    # ===== Tool Formatting Tests =====

    def test_tools_to_provider_format(self):
        """Test formatting tools for provider."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            tools = [
                Tool(
                    name="test_tool",
                    description="Test tool",
                    parameters=ToolParameters(type="object", properties={}),
                ),
                Tool(name="simple_tool", description="Simple", parameters=None),
            ]

            formatted = client._tools_to_provider_format(tools)

            assert len(formatted) == 2
            assert formatted[0]["type"] == "function"
            assert formatted[0]["name"] == "test_tool"
            assert "parameters" in formatted[0]
            assert formatted[1]["parameters"] is None

    # ===== API Request Tests =====

    def test_make_provider_request_no_tools(self):
        """Test making API request without tools."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            sample_response = self.sample_response
            mock_client.responses.create.return_value = sample_response

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            response = client._make_provider_request(
                messages=[{"role": "user", "content": "Hello"}], model="gpt-4", stream=False
            )

            assert response is sample_response
            mock_client.responses.create.assert_called_once()

            # Verify NOT_GIVEN was used for tools
            call_args = mock_client.responses.create.call_args
            from openai import NOT_GIVEN

            assert call_args.kwargs["tools"] is NOT_GIVEN

    def test_make_provider_request_with_tools_streaming(self):
        """Test making streaming API request with tools."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            tools = [{"type": "function", "name": "test"}]

            client._make_provider_request(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4",
                stream=True,
                tools=tools,
                temperature=0.7,
            )

            mock_client.responses.create.assert_called_once_with(
                model="gpt-4",
                input=[{"role": "user", "content": "Hello"}],
                stream=True,
                tools=tools,
                temperature=0.7,
            )

    # ===== Stream Processing Tests =====

    def test_process_stream_events_all_types(self):
        """Test processing all types of stream events."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            chunks = []
            for event in self.sample_stream_events:
                chunk = client._process_provider_stream_event(event, processor)
                if chunk:
                    chunks.append(chunk)

            # Should have text chunk and completion chunk
            assert len(chunks) == 2
            assert chunks[0].common.content == "Hello"
            assert chunks[1].common.finish_reason == "completed"

            # Check tool calls were processed
            tool_calls = processor.get_completed_tool_calls()
            assert len(tool_calls) == 1
            assert tool_calls[0].name == "test_tool"
            assert tool_calls[0].arguments == '{"x": 10}'

    def test_process_stream_event_unknown_type(self):
        """Test processing unknown event type."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            unknown_event = Mock()
            unknown_event.type = "unknown.event.type"

            chunk = client._process_provider_stream_event(unknown_event, processor)
            assert chunk is None

    def test_process_stream_event_no_type_attribute(self):
        """Test processing event without type attribute."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            event_no_type = Mock(spec=[])  # No attributes
            chunk = client._process_provider_stream_event(event_no_type, processor)
            assert chunk is None

    def test_process_stream_event_empty_delta(self):
        """Test processing event with empty delta."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Event with None delta
            event = Mock()
            event.type = "response.output_text.delta"
            event.delta = None

            chunk = client._process_provider_stream_event(event, processor)
            assert chunk.common.content == ""

    def test_process_stream_event_tool_without_id(self):
        """Test processing tool event without required IDs."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Tool start without ID
            event = Mock()
            event.type = "response.output_item.added"
            event.item = Mock(type="function_call", id=None, call_id=None, name="test")

            chunk = client._process_provider_stream_event(event, processor)
            assert chunk is None

            # Tool delta without item_id
            event2 = Mock()
            event2.type = "response.function_call_arguments.delta"
            event2.item_id = None
            event2.delta = "test"

            chunk2 = client._process_provider_stream_event(event2, processor)
            assert chunk2 is None

    # ===== Response Extraction Tests =====

    def test_extract_usage_from_response(self):
        """Test extracting usage information."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Response with usage
            response = Mock()
            response.usage = Mock(input_tokens=10, output_tokens=20, total_tokens=30)

            usage = client._extract_usage_from_response(response)
            assert usage.prompt_tokens == 10
            assert usage.completion_tokens == 20
            assert usage.total_tokens == 30

            # Response without usage
            response_no_usage = Mock()
            response_no_usage.usage = None

            usage_empty = client._extract_usage_from_response(response_no_usage)
            assert usage_empty.prompt_tokens == 0
            assert usage_empty.completion_tokens == 0

    def test_extract_content_from_response_variations(self):
        """Test extracting content from different response formats."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Response with output_text
            response1 = Mock()
            response1.output_text = "Hello"
            response1.output = None
            content1 = client._extract_content_from_response(response1)
            assert content1 == "Hello"

            # Response with output list
            response2 = Mock()
            response2.output_text = None
            response2.output = ["Item1", "Item2"]
            content2 = client._extract_content_from_response(response2)
            assert content2 == ["Item1", "Item2"]

            # Response with nested text
            response3 = Mock()
            response3.output_text = None
            response3.output = None
            response3.response = Mock(text="Nested text")
            content3 = client._extract_content_from_response(response3)
            assert content3 == "Nested text"

            # Empty response
            response4 = Mock()
            response4.output_text = None
            response4.output = None
            response4.response = Mock(spec=[])  # No text attribute
            content4 = client._extract_content_from_response(response4)
            assert content4 == ""

    def test_extract_tool_calls_from_response(self):
        """Test extracting tool calls from response."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Response with tool calls
            tool_call = ResponseFunctionToolCall(
                type="function_call",
                call_id="call_123",
                name="test_tool",
                arguments='{"x": 10}',
            )
            response = Mock()
            response.output = [tool_call, "other_output"]

            tool_calls = client._extract_tool_calls_from_response(response)
            assert tool_calls is not None
            assert len(tool_calls) == 1
            assert tool_calls[0].call_id == "call_123"
            assert tool_calls[0].name == "test_tool"

            # Response without output
            response_no_output = Mock(spec=["output_text"])
            tool_calls_none = client._extract_tool_calls_from_response(response_no_output)
            assert tool_calls_none is None

            # Response with empty output
            response_empty = Mock()
            response_empty.output = []
            tool_calls_empty = client._extract_tool_calls_from_response(response_empty)
            assert tool_calls_empty is None

    # ===== Message Update Tests =====

    def test_update_messages_with_tool_calls_non_streaming(self):
        """Test updating messages after tool execution (non-streaming)."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Setup assistant response with tool calls in output
            tool_output = Mock(
                type="function_call",
                id="fc_123",
                call_id="call_456",
                name="test_tool",
                arguments='{"x": 10}',
            )
            # Add a non-function_call item
            non_tool_output = Mock(type="text_output", content="some text")
            assistant_response = Mock()
            assistant_response.output = [tool_output, non_tool_output]

            tool_calls = [ToolCall(call_id="call_456", name="test_tool", arguments='{"x": 10}')]
            tool_results = [
                ToolExecutionResult(
                    call_id="call_456", name="test_tool", arguments='{"x": 10}', result="Result: 10"
                )
            ]

            messages = [{"role": "user", "content": "Hello"}]
            updated = client._update_messages_with_tool_calls(
                messages, assistant_response, tool_calls, tool_results
            )

            assert (
                len(updated) == 3
            )  # Original + tool call + tool result (non-tool item filtered out)
            assert updated[1] == tool_output
            assert updated[2]["type"] == "function_call_output"
            assert updated[2]["output"] == "Result: 10"

    def test_update_messages_with_tool_calls_streaming(self):
        """Test updating messages after tool execution (streaming)."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Streaming response (no output attribute)
            assistant_response = Mock(spec=[])

            # Tool calls with metadata containing original IDs
            tool_calls = [
                ToolCall(
                    call_id="call_456",
                    name="test_tool",
                    arguments='{"x": 10}',
                    metadata={"original_id": "fc_123"},
                )
            ]
            tool_results = [
                ToolExecutionResult(
                    call_id="call_456", name="test_tool", arguments='{"x": 10}', result="Result: 10"
                )
            ]

            messages = [{"role": "user", "content": "Hello"}]
            updated = client._update_messages_with_tool_calls(
                messages, assistant_response, tool_calls, tool_results
            )

            assert len(updated) == 3
            assert updated[1]["type"] == "function_call"
            assert updated[1]["id"] == "fc_123"  # Original ID
            assert updated[1]["call_id"] == "call_456"
            assert updated[2]["type"] == "function_call_output"

    def test_update_messages_with_tool_error(self):
        """Test updating messages with tool execution error."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            assistant_response = Mock(spec=[])
            tool_calls = [ToolCall(call_id="call_456", name="error_tool", arguments='{"x": 10}')]
            tool_results = [
                ToolExecutionResult(
                    call_id="call_456",
                    name="error_tool",
                    arguments='{"x": 10}',
                    error="Tool failed",
                    is_error=True,
                )
            ]

            messages = []
            updated = client._update_messages_with_tool_calls(
                messages, assistant_response, tool_calls, tool_results
            )

            # Check error formatting
            assert updated[-1]["output"] == "Error: Tool failed"

    # ===== Error Handling Tests =====

    def test_extract_content_fallback_chain(self):
        """Test content extraction fallback chain."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test all fallback scenarios
            response = Mock()
            response.output_text = None
            response.output = None
            response.response = None
            content = client._extract_content_from_response(response)
            assert content == ""

    def test_extract_tool_calls_no_output_attribute(self):
        """Test tool call extraction when output attribute is missing."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Response with no output attribute
            response = Mock(spec=[])  # No attributes
            tool_calls = client._extract_tool_calls_from_response(response)
            assert tool_calls is None


class TestOpenAIAsyncClient(BaseProviderTestSuite):
    """Test suite for OpenAI async client"""

    client_class = OpenAIAsyncClient
    provider_name = "OpenAI"
    mock_async_client_path = "chimeric.providers.openai.client.AsyncOpenAI"

    @property
    def sample_response(self):
        """Create a sample OpenAI response."""
        response = Mock(spec=Response)
        response.output_text = "Hello there"
        response.output = []
        response.usage = Mock(input_tokens=10, output_tokens=20, total_tokens=30)
        return response

    @property
    def sample_stream_events(self):
        """Create sample OpenAI stream events."""
        events = []

        # Text delta event
        text_event = Mock(spec=ResponseStreamEvent)
        text_event.type = "response.output_text.delta"
        text_event.delta = "Hello"
        events.append(text_event)

        # Response completed
        complete = Mock(spec=ResponseStreamEvent)
        complete.type = "response.completed"
        complete.response = Mock(status="completed")
        events.append(complete)

        return events

    # ===== Async Initialization Tests =====

    async def test_async_client_initialization(self):
        """Test async client initialization."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            assert client.api_key == "test-key"
            assert client._provider_name == self.provider_name

    async def test_async_list_models(self):
        """Test async model listing."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path) as mock_async_openai:
            mock_client = AsyncMock()
            mock_async_openai.return_value = mock_client

            # Mock async model list
            mock_model = Mock(id="gpt-4", owned_by="openai", created=1234567890)
            mock_models_response = Mock()
            mock_models_response.data = [mock_model]
            mock_client.models.list.return_value = mock_models_response

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            models = await client._list_models_impl()

            assert len(models) == 1
            assert models[0].id == "gpt-4"

    async def test_async_make_request(self):
        """Test async API request."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path) as mock_async_openai:
            mock_client = AsyncMock()
            mock_async_openai.return_value = mock_client
            sample_response = self.sample_response
            mock_client.responses.create.return_value = sample_response

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            response = await client._make_async_provider_request(
                messages=[{"role": "user", "content": "Hello"}], model="gpt-4", stream=False
            )

            assert response is sample_response

    async def test_async_capabilities(self):
        """Test async provider capabilities."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            capabilities = client._get_capabilities()

            assert isinstance(capabilities, Capability)
            assert capabilities.streaming is True
            assert capabilities.tools is True

    async def test_async_tools_to_provider_format(self):
        """Test async formatting tools for provider."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            tools = [
                Tool(
                    name="test_tool",
                    description="Test tool",
                    parameters=ToolParameters(type="object", properties={}),
                ),
                Tool(name="simple_tool", description="Simple", parameters=None),
            ]

            formatted = client._tools_to_provider_format(tools)

            assert len(formatted) == 2
            assert formatted[0]["type"] == "function"
            assert formatted[0]["name"] == "test_tool"

    async def test_async_stream_processing(self):
        """Test async stream processing."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            chunks = []
            for event in self.sample_stream_events:
                chunk = client._process_provider_stream_event(event, processor)
                if chunk:
                    chunks.append(chunk)

            assert len(chunks) == 2
            assert chunks[0].common.content == "Hello"
            assert chunks[1].common.finish_reason == "completed"

    async def test_async_extract_tool_calls_missing_output(self):
        """Test async tool call extraction with missing output attribute."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Response with no output attribute
            response = Mock(spec=[])
            tool_calls = client._extract_tool_calls_from_response(response)
            assert tool_calls is None

    async def test_async_extract_tool_calls_empty_output(self):
        """Test async tool call extraction with empty output."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Response with empty output
            response = Mock()
            response.output = []
            tool_calls = client._extract_tool_calls_from_response(response)
            assert tool_calls is None

    async def test_async_usage_extraction_no_usage(self):
        """Test async usage extraction when no usage data."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = Mock()
            response.usage = None
            usage = client._extract_usage_from_response(response)
            assert usage.prompt_tokens == 0
            assert usage.completion_tokens == 0
            assert usage.total_tokens == 0

    async def test_async_content_extraction_all_none(self):
        """Test async content extraction when all sources are None."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = Mock()
            response.output_text = None
            response.output = None
            response.response = None
            content = client._extract_content_from_response(response)
            assert content == ""

    async def test_async_messages_formatting_edge_cases(self):
        """Test async message formatting edge cases."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Empty messages list
            messages = []
            formatted = client._messages_to_provider_format(messages)
            assert formatted == []

            # System message
            messages = [Message(role="system", content="You are helpful")]
            formatted = client._messages_to_provider_format(messages)
            assert len(formatted) == 1
            assert formatted[0]["role"] == "system"

    async def test_async_tool_calls_update_no_metadata(self):
        """Test async tool calls update with missing metadata."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            assistant_response = Mock(spec=[])
            tool_calls = [ToolCall(call_id="call_456", name="test_tool", arguments='{"x": 10}')]
            tool_results = [
                ToolExecutionResult(
                    call_id="call_456", name="test_tool", arguments='{"x": 10}', result="Result: 10"
                )
            ]

            messages = []
            updated = client._update_messages_with_tool_calls(
                messages, assistant_response, tool_calls, tool_results
            )

            # Should only have tool result since no metadata for function call
            assert len(updated) == 1
            assert updated[0]["type"] == "function_call_output"

    async def test_async_messages_with_tool_calls_and_results(self):
        """Test async message formatting with tool calls and results."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test tool message formatting
            messages = [
                Message(role="user", content="Hello"),
                Message(role="tool", content="Tool result", tool_call_id="call_123"),
            ]
            formatted = client._messages_to_provider_format(messages)
            assert len(formatted) == 2
            assert formatted[1]["type"] == "function_call_output"
            assert formatted[1]["call_id"] == "call_123"
            assert formatted[1]["output"] == "Tool result"

            # Test assistant message with tool calls
            tool_call = ToolCall(call_id="call_456", name="test_tool", arguments='{"x": 20}')
            messages_with_calls = [Message(role="assistant", content="", tool_calls=[tool_call])]
            formatted_calls = client._messages_to_provider_format(messages_with_calls)
            assert len(formatted_calls) == 1
            assert formatted_calls[0]["type"] == "function_call"
            assert formatted_calls[0]["call_id"] == "call_456"
            assert formatted_calls[0]["name"] == "test_tool"
            assert formatted_calls[0]["arguments"] == '{"x": 20}'

    async def test_async_stream_processing_complete_flow(self):
        """Test async stream processing with full tool call flow."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Complete set of events
            events = []

            # Text delta
            text_event = Mock(spec=ResponseStreamEvent)
            text_event.type = "response.output_text.delta"
            text_event.delta = "Hello"
            events.append(text_event)

            # Tool call start
            tool_start = Mock(spec=ResponseStreamEvent)
            tool_start.type = "response.output_item.added"
            tool_item = Mock()
            tool_item.type = "function_call"
            tool_item.id = "fc_456"
            tool_item.call_id = "call_789"
            tool_item.name = "async_tool"
            tool_start.item = tool_item
            events.append(tool_start)

            # Non-tool output item event
            non_tool_start = Mock(spec=ResponseStreamEvent)
            non_tool_start.type = "response.output_item.added"
            non_tool_item = Mock()
            non_tool_item.type = "message"  # Not "function_call"
            non_tool_start.item = non_tool_item
            events.append(non_tool_start)

            # Tool call start event with None
            tool_start_no_id = Mock(spec=ResponseStreamEvent)
            tool_start_no_id.type = "response.output_item.added"
            tool_item_no_id = Mock()
            tool_item_no_id.type = "function_call"
            tool_item_no_id.id = None  # This will cause tool_call_id to be None
            tool_item_no_id.call_id = "call_999"
            tool_item_no_id.name = "no_id_tool"
            tool_start_no_id.item = tool_item_no_id
            events.append(tool_start_no_id)

            # Tool call arguments delta
            args_delta = Mock(spec=ResponseStreamEvent)
            args_delta.type = "response.function_call_arguments.delta"
            args_delta.item_id = "fc_456"
            args_delta.delta = '{"value":'
            events.append(args_delta)

            # More arguments
            args_delta2 = Mock(spec=ResponseStreamEvent)
            args_delta2.type = "response.function_call_arguments.delta"
            args_delta2.item_id = "fc_456"
            args_delta2.delta = " 42}"
            events.append(args_delta2)

            # Tool call complete
            args_done = Mock(spec=ResponseStreamEvent)
            args_done.type = "response.function_call_arguments.done"
            args_done.item_id = "fc_456"
            events.append(args_done)

            # Function arguments done with None item_id
            args_done_none = Mock(spec=ResponseStreamEvent)
            args_done_none.type = "response.function_call_arguments.done"
            args_done_none.item_id = None
            events.append(args_done_none)

            # Response completed
            complete = Mock(spec=ResponseStreamEvent)
            complete.type = "response.completed"
            complete.response = Mock(status="finished")
            events.append(complete)

            chunks = []
            for event in events:
                chunk = client._process_provider_stream_event(event, processor)
                if chunk:
                    chunks.append(chunk)

            # Should have text and completion chunks
            assert len(chunks) == 2
            assert chunks[0].common.content == "Hello"
            assert chunks[1].common.finish_reason == "finished"

            # Check completed tool calls
            tool_calls = processor.get_completed_tool_calls()
            assert len(tool_calls) == 1
            assert tool_calls[0].name == "async_tool"
            assert tool_calls[0].arguments == '{"value": 42}'

    async def test_async_tool_calls_update_with_response_output(self):
        """Test async tool calls update with response containing output."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Assistant response with output containing tool calls
            tool_output = Mock()
            tool_output.type = "function_call"
            tool_output.id = "fc_789"
            tool_output.call_id = "call_987"
            tool_output.name = "async_tool"
            tool_output.arguments = '{"param": "value"}'

            # Add non-function call output
            text_output = Mock()
            text_output.type = "text_output"
            text_output.content = "some text"

            assistant_response = Mock()
            assistant_response.output = [tool_output, text_output]

            tool_calls = [
                ToolCall(call_id="call_987", name="async_tool", arguments='{"param": "value"}')
            ]
            tool_results = [
                ToolExecutionResult(
                    call_id="call_987",
                    name="async_tool",
                    arguments='{"param": "value"}',
                    result="Success",
                )
            ]

            messages = [{"role": "user", "content": "Test"}]
            updated = client._update_messages_with_tool_calls(
                messages, assistant_response, tool_calls, tool_results
            )

            assert (
                len(updated) == 3
            )  # Original + tool call + tool result (text_output filtered out)
            assert updated[1] == tool_output  # Original tool call from response
            assert updated[2]["type"] == "function_call_output"
            assert updated[2]["call_id"] == "call_987"
            assert updated[2]["output"] == "Success"

    async def test_async_tool_calls_update_streaming_with_metadata(self):
        """Test async tool calls update for streaming with metadata."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Streaming response (no output)
            assistant_response = Mock(spec=[])

            # Tool call with metadata
            tool_calls = [
                ToolCall(
                    call_id="call_111",
                    name="streaming_tool",
                    arguments='{"data": "test"}',
                    metadata={"original_id": "fc_111"},
                )
            ]
            tool_results = [
                ToolExecutionResult(
                    call_id="call_111",
                    name="streaming_tool",
                    arguments='{"data": "test"}',
                    result="Streaming result",
                )
            ]

            messages = []
            updated = client._update_messages_with_tool_calls(
                messages, assistant_response, tool_calls, tool_results
            )

            assert len(updated) == 2
            # Check function call reconstruction
            assert updated[0]["type"] == "function_call"
            assert updated[0]["id"] == "fc_111"  # Original ID from metadata
            assert updated[0]["call_id"] == "call_111"
            assert updated[0]["name"] == "streaming_tool"
            assert updated[0]["arguments"] == '{"data": "test"}'
            # Check function result
            assert updated[1]["type"] == "function_call_output"
            assert updated[1]["call_id"] == "call_111"
            assert updated[1]["output"] == "Streaming result"

    async def test_async_extract_tool_calls_with_calls(self):
        """Test async tool call extraction with actual tool calls."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Response with actual tool calls
            tool_call = ResponseFunctionToolCall(
                type="function_call",
                call_id="call_async_123",
                name="async_test_tool",
                arguments='{"async_param": "value"}',
            )
            response = Mock()
            response.output = [tool_call, "other_item"]

            tool_calls = client._extract_tool_calls_from_response(response)
            assert tool_calls is not None
            assert len(tool_calls) == 1
            assert tool_calls[0].call_id == "call_async_123"
            assert tool_calls[0].name == "async_test_tool"
            assert tool_calls[0].arguments == '{"async_param": "value"}'

    async def test_async_usage_extraction_with_usage(self):
        """Test async usage extraction with actual usage data."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = Mock()
            response.usage = Mock(input_tokens=50, output_tokens=75, total_tokens=125)
            usage = client._extract_usage_from_response(response)
            assert usage.prompt_tokens == 50
            assert usage.completion_tokens == 75
            assert usage.total_tokens == 125

    async def test_async_stream_processing_edge_cases(self):
        """Test async stream processing edge cases."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Test tool event with no item
            event1 = Mock()
            event1.type = "response.output_item.added"
            event1.item = None
            result1 = client._process_provider_stream_event(event1, processor)
            assert result1 is None

            # Test tool event with wrong type
            event2 = Mock()
            event2.type = "response.output_item.added"
            wrong_item = Mock()
            wrong_item.type = "text"
            event2.item = wrong_item
            result2 = client._process_provider_stream_event(event2, processor)
            assert result2 is None

            # Test function arguments delta with None item_id
            event3 = Mock()
            event3.type = "response.function_call_arguments.delta"
            event3.item_id = None
            event3.delta = "test"
            result3 = client._process_provider_stream_event(event3, processor)
            assert result3 is None

            # Test function arguments done with None item_id
            event4 = Mock()
            event4.type = "response.function_call_arguments.done"
            event4.item_id = None
            result4 = client._process_provider_stream_event(event4, processor)
            assert result4 is None

            # Test unrecognized event type (should hit final return None)
            event5 = Mock()
            event5.type = "unrecognized.event.type"
            result5 = client._process_provider_stream_event(event5, processor)
            assert result5 is None
