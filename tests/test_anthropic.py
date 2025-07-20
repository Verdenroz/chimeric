from unittest.mock import AsyncMock, MagicMock, Mock, patch

from anthropic.types import Message, MessageStreamEvent

from chimeric.providers.anthropic import AnthropicAsyncClient, AnthropicClient
from chimeric.types import Capability, Message as ChimericMessage, Tool, ToolCall, ToolExecutionResult, ToolParameters
from chimeric.utils import StreamProcessor
from conftest import BaseProviderTestSuite


class TestAnthropicClient(BaseProviderTestSuite):
    """Test suite for Anthropic sync client."""

    client_class = AnthropicClient
    provider_name = "Anthropic"
    mock_client_path = "chimeric.providers.anthropic.client.Anthropic"

    @property
    def sample_response(self):
        """Create a sample Anthropic response."""
        response = Mock(spec=Message)
        content_block = Mock()
        content_block.type = "text"
        content_block.text = "Hello there"
        response.content = [content_block]
        response.usage = Mock(input_tokens=10, output_tokens=20, total_tokens=30)
        response.stop_reason = "end_turn"
        return response

    @property
    def sample_stream_events(self):
        """Create sample Anthropic stream events."""
        events = []

        # Text content delta event
        text_event = Mock()  # Remove spec to allow dynamic attributes
        text_event.type = "content_block_delta"
        # Use configure_mock to set the nested attributes correctly
        text_event.configure_mock(**{"delta.text": "Hello"})
        events.append(text_event)

        # Tool call start event
        tool_start = Mock()
        tool_start.type = "content_block_start"
        tool_start.index = 0
        tool_content_block = Mock()
        tool_content_block.type = "tool_use"
        tool_content_block.name = "test_tool"
        tool_start.content_block = tool_content_block
        events.append(tool_start)

        # Non-tool content block start
        non_tool_start = Mock()
        non_tool_start.type = "content_block_start"
        non_tool_start.index = 1
        non_tool_content_block = Mock()
        non_tool_content_block.type = "text"  # Not "tool_use"
        non_tool_start.content_block = non_tool_content_block
        events.append(non_tool_start)

        # Tool call arguments delta
        args_delta = Mock()
        args_delta.type = "content_block_delta"
        args_delta.index = 0
        args_delta.configure_mock(**{"delta.partial_json": '{"x": 1'})
        events.append(args_delta)

        # More arguments
        args_delta2 = Mock()
        args_delta2.type = "content_block_delta"
        args_delta2.index = 0
        args_delta2.configure_mock(**{"delta.partial_json": '0}'})
        events.append(args_delta2)

        # Tool call complete
        tool_stop = Mock()
        tool_stop.type = "content_block_stop"
        tool_stop.index = 0
        events.append(tool_stop)

        # Message completion
        message_stop = Mock()
        message_stop.type = "message_stop"
        events.append(message_stop)

        return events

    # ===== Initialization Tests =====

    def test_client_initialization_success(self):
        """Test successful client initialization with all parameters."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_anthropic:
            client = self.client_class(
                api_key="test-key",
                tool_manager=tool_manager,
                base_url="https://custom.api.com",
                timeout=30
            )

            assert client.api_key == "test-key"
            assert client.tool_manager == tool_manager
            assert client._provider_name == self.provider_name
            mock_anthropic.assert_called_once_with(
                api_key="test-key",
                base_url="https://custom.api.com",
                timeout=30
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

    def test_model_aliases(self):
        """Test model aliases."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            aliases = client._get_model_aliases()

            assert isinstance(aliases, list)
            assert "claude-opus-4-0" in aliases
            assert "claude-sonnet-4-0" in aliases
            assert "claude-3-7-sonnet-latest" in aliases

    # ===== Model Listing Tests =====

    def test_list_models_success(self):
        """Test successful model listing."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            # Mock model list response
            mock_timestamp = Mock()
            mock_timestamp.timestamp.return_value = 1709251200.0  # Feb 29, 2024
            
            mock_model = Mock()
            mock_model.id = "claude-3-sonnet-20240229"
            mock_model.display_name = "Claude 3 Sonnet"
            mock_model.created_at = mock_timestamp
            mock_model.model_dump.return_value = {"id": "claude-3-sonnet-20240229"}
            
            mock_models_response = Mock(data=[mock_model])
            mock_client.models.list.return_value = mock_models_response

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            models = client._list_models_impl()

            assert len(models) == 1
            assert models[0].id == "claude-3-sonnet-20240229"
            assert models[0].name == "Claude 3 Sonnet"
            assert models[0].created_at == 1709251200

    # ===== Message Formatting Tests =====

    def test_messages_to_provider_format_regular(self):
        """Test formatting regular messages."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            messages = [
                ChimericMessage(role="user", content="Hello"),
                ChimericMessage(role="assistant", content="Hi there")
            ]

            formatted = client._messages_to_provider_format(messages)

            assert len(formatted) == 2
            assert formatted[0]["role"] == "user"
            assert formatted[0]["content"] == "Hello"

    def test_messages_to_provider_format_with_system(self):
        """Test formatting messages with system message (should be filtered out)."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            messages = [
                ChimericMessage(role="system", content="You are helpful"),
                ChimericMessage(role="user", content="Hello"),
                ChimericMessage(role="assistant", content="Hi there")
            ]

            formatted = client._messages_to_provider_format(messages)

            # System message should be filtered out
            assert len(formatted) == 2
            assert formatted[0]["role"] == "user"
            assert formatted[1]["role"] == "assistant"

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
                    parameters=ToolParameters(type="object", properties={"x": {"type": "number"}})
                ),
                Tool(name="simple_tool", description="Simple", parameters=None)
            ]

            formatted = client._tools_to_provider_format(tools)

            assert len(formatted) == 2
            assert formatted[0]["name"] == "test_tool"
            assert formatted[0]["description"] == "Test tool"
            assert formatted[0]["input_schema"]["type"] == "object"
            assert formatted[1]["input_schema"] == {}

    # ===== API Request Tests =====

    def test_make_provider_request_no_tools(self):
        """Test making API request without tools."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client
            sample_response = self.sample_response
            mock_client.messages.create.return_value = sample_response

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            response = client._make_provider_request(
                messages=[{"role": "user", "content": "Hello"}],
                model="claude-3-sonnet-20240229",
                stream=False
            )

            assert response is sample_response
            mock_client.messages.create.assert_called_once()

            # Verify NOT_GIVEN was used for tools
            call_args = mock_client.messages.create.call_args
            from anthropic import NOT_GIVEN
            assert call_args.kwargs["tools"] is NOT_GIVEN

    def test_make_provider_request_with_tools_streaming(self):
        """Test making streaming API request with tools."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            tools = [{"name": "test", "description": "Test tool", "input_schema": {}}]

            client._make_provider_request(
                messages=[{"role": "user", "content": "Hello"}],
                model="claude-3-sonnet-20240229",
                stream=True,
                tools=tools,
                temperature=0.7
            )

            mock_client.messages.create.assert_called_once_with(
                model="claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
                tools=tools,
                temperature=0.7,
                max_tokens=4096
            )

    # ===== Stream Processing Tests =====

    def test_process_stream_events_all_types(self):
        """Test processing all types of stream events."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Create a simple text delta event directly in the test
            text_event = Mock()
            text_event.type = "content_block_delta"
            text_event.delta = Mock()
            text_event.delta.text = "Hello"
            
            # Create message stop event
            message_stop = Mock()
            message_stop.type = "message_stop"

            chunks = []
            # Process text event
            chunk = client._process_provider_stream_event(text_event, processor)
            if chunk:
                chunks.append(chunk)
            
            # Process completion event
            chunk = client._process_provider_stream_event(message_stop, processor)
            if chunk:
                chunks.append(chunk)

            # Should have text chunk and completion chunk
            assert len(chunks) == 2
            assert chunks[0].common.content == "Hello"
            assert chunks[1].common.finish_reason == "end_turn"

    def test_process_stream_events_tool_calls(self):
        """Test processing tool call stream events to achieve full coverage."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Tool call start event
            tool_start = Mock()
            tool_start.type = "content_block_start"
            tool_start.index = 0
            tool_start.content_block = Mock()
            tool_start.content_block.type = "tool_use"
            tool_start.content_block.name = "test_tool"

            # Tool call arguments delta - only has partial_json, NOT text
            args_delta = Mock()
            args_delta.type = "content_block_delta"
            args_delta.index = 0
            args_delta.delta = Mock(spec=['partial_json'])  # Only partial_json attribute
            args_delta.delta.partial_json = '{"x": 1'

            # More arguments
            args_delta2 = Mock()
            args_delta2.type = "content_block_delta"
            args_delta2.index = 0
            args_delta2.delta = Mock(spec=['partial_json'])  # Only partial_json attribute
            args_delta2.delta.partial_json = '0}'

            # Tool call complete
            tool_stop = Mock()
            tool_stop.type = "content_block_stop"
            tool_stop.index = 0

            # Process events to cover tool call branches
            client._process_provider_stream_event(tool_start, processor)
            client._process_provider_stream_event(args_delta, processor)
            client._process_provider_stream_event(args_delta2, processor)
            client._process_provider_stream_event(tool_stop, processor)

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
            unknown_event.type = "unknown_event_type"

            chunk = client._process_provider_stream_event(unknown_event, processor)
            assert chunk is None

    def test_process_stream_event_content_delta_without_text(self):
        """Test processing content delta event without text attribute."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Event with delta but no text attribute
            event = Mock()
            event.type = "content_block_delta"
            event.delta = Mock(spec=[])  # No text attribute

            chunk = client._process_provider_stream_event(event, processor)
            assert chunk is None

    def test_process_stream_event_content_start_without_content_block(self):
        """Test processing content start event without content_block attribute."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Event without content_block attribute
            event = Mock()
            event.type = "content_block_start"
            event.index = 0
            # Missing content_block attribute

            chunk = client._process_provider_stream_event(event, processor)
            assert chunk is None

            # Event with content_block but type != 'tool_use'
            event2 = Mock()
            event2.type = "content_block_start"
            event2.content_block = Mock()
            event2.content_block.type = "text"  # Not 'tool_use'
            event2.index = 1

            chunk2 = client._process_provider_stream_event(event2, processor)
            assert chunk2 is None
            # Verify no tool calls were started since it's not tool_use type
            assert len(processor.state.tool_calls) == 0

            # Event with content_block_start but content_block has no type attribute
            event3 = Mock()
            event3.type = "content_block_start"
            event3.content_block = Mock(spec=[])  # Mock with no attributes
            event3.index = 2

            chunk3 = client._process_provider_stream_event(event3, processor)
            assert chunk3 is None
            # Verify no tool calls were started
            assert len(processor.state.tool_calls) == 0

    def test_process_stream_event_delta_without_partial_json(self):
        """Test processing delta event without partial_json attribute."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Event with delta but no partial_json attribute
            event = Mock()
            event.type = "content_block_delta"
            event.index = 0
            event.delta = Mock(spec=[])  # No partial_json attribute

            chunk = client._process_provider_stream_event(event, processor)
            assert chunk is None

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

    def test_extract_content_from_response(self):
        """Test extracting content from response."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Response with text content
            response = Mock()
            response.content = [Mock(type="text", text="Hello world")]

            content = client._extract_content_from_response(response)
            assert content == "Hello world"

            # Response with multiple content blocks
            response_multi = Mock()
            response_multi.content = [
                Mock(type="text", text="Hello"),
                Mock(type="text", text=" world")
            ]

            content_multi = client._extract_content_from_response(response_multi)
            assert content_multi == "Hello world"

            # Response with no content
            response_empty = Mock()
            response_empty.content = []

            content_empty = client._extract_content_from_response(response_empty)
            assert content_empty == ""

    def test_extract_tool_calls_from_response(self):
        """Test extracting tool calls from response."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Response with tool calls
            tool_block = Mock()
            tool_block.type = "tool_use"
            tool_block.id = "call_123"
            tool_block.name = "test_tool"
            tool_block.input = {"x": 10}

            # Add non-tool block to test branch condition
            text_block = Mock()
            text_block.type = "text"
            text_block.text = "Some text"

            response = Mock()
            response.content = [tool_block, text_block]

            tool_calls = client._extract_tool_calls_from_response(response)
            assert tool_calls is not None
            assert len(tool_calls) == 1
            assert tool_calls[0].call_id == "call_123"
            assert tool_calls[0].name == "test_tool"
            assert tool_calls[0].arguments == '{"x": 10}'

            # Response without tool calls
            response_no_tools = Mock()
            response_no_tools.content = [text_block]

            tool_calls_none = client._extract_tool_calls_from_response(response_no_tools)
            assert tool_calls_none is None

    def test_extract_tool_calls_with_non_dict_input(self):
        """Test extracting tool calls with non-dict input."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Tool block with string input (not dict)
            tool_block = Mock()
            tool_block.type = "tool_use"
            tool_block.id = "call_456"
            tool_block.name = "string_tool"
            tool_block.input = "string_input"

            response = Mock()
            response.content = [tool_block]

            tool_calls = client._extract_tool_calls_from_response(response)
            assert tool_calls is not None
            assert len(tool_calls) == 1
            assert tool_calls[0].arguments == "string_input"

    # ===== Message Update Tests =====

    def test_update_messages_with_tool_calls_simple(self):
        """Test updating messages with tool calls and results."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Mock assistant response with tool calls
            tool_block = Mock()
            tool_block.type = "tool_use"
            tool_block.id = "call_123"
            tool_block.name = "test_tool"
            tool_block.input = {"x": 10}

            assistant_response = Mock()
            assistant_response.content = [tool_block]

            tool_calls = [ToolCall(call_id="call_123", name="test_tool", arguments='{"x": 10}')]
            tool_results = [ToolExecutionResult(call_id="call_123", name="test_tool", arguments='{"x": 10}', result="Result: 10")]

            messages = [{"role": "user", "content": "Hello"}]
            updated = client._update_messages_with_tool_calls(
                messages, assistant_response, tool_calls, tool_results
            )

            assert len(updated) == 3  # Original + assistant + tool results
            assert updated[1]["role"] == "assistant"
            assert updated[2]["role"] == "user"  # Tool results become user messages

    def test_update_messages_with_tool_error(self):
        """Test updating messages with tool execution error."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            assistant_response = Mock()
            assistant_response.content = []

            tool_calls = [ToolCall(call_id="call_456", name="error_tool", arguments='{"x": 10}')]
            tool_results = [
                ToolExecutionResult(
                    call_id="call_456",
                    name="error_tool",
                    arguments='{"x": 10}',
                    error="Tool failed",
                    is_error=True
                )
            ]

            messages = []
            updated = client._update_messages_with_tool_calls(
                messages, assistant_response, tool_calls, tool_results
            )

            # Check error formatting in tool result content
            tool_result_content = updated[-1]["content"][0]["content"]
            assert "Error: Tool failed" in tool_result_content

    def test_update_messages_with_mixed_content_response(self):
        """Test updating messages with response containing both text and tool blocks."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Response with both text and tool_use blocks
            text_block = Mock()
            text_block.type = "text"
            text_block.text = "I'll help with that."

            tool_block = Mock()
            tool_block.type = "tool_use"
            tool_block.id = "call_123"
            tool_block.name = "test_tool"
            tool_block.input = {"x": 10}

            assistant_response = Mock()
            assistant_response.content = [text_block, tool_block]

            tool_calls = [ToolCall(call_id="call_123", name="test_tool", arguments='{"x": 10}')]
            tool_results = [ToolExecutionResult(call_id="call_123", name="test_tool", arguments='{"x": 10}', result="Result: 10")]

            messages = [{"role": "user", "content": "Hello"}]
            updated = client._update_messages_with_tool_calls(
                messages, assistant_response, tool_calls, tool_results
            )

            # Check that both text and tool blocks are included
            assert len(updated) == 3  # Original + assistant + tool results

            # Response with content blocks that aren't text or tool_use
            unknown_block = Mock()
            unknown_block.type = "unknown_type"  # This should be ignored

            text_block2 = Mock()
            text_block2.type = "text"
            text_block2.text = "Valid text"

            assistant_response2 = Mock()
            assistant_response2.content = [unknown_block, text_block2]

            updated2 = client._update_messages_with_tool_calls(
                messages, assistant_response2, [], []
            )

            # Should only have the text block, unknown_type should be ignored
            assistant_msg = updated2[1]
            assert len(assistant_msg["content"]) == 1
            assert assistant_msg["content"][0]["type"] == "text"
            assert assistant_msg["content"][0]["text"] == "Valid text"
            assistant_msg = updated[1]
            assert assistant_msg["role"] == "assistant"
            assert len(assistant_msg["content"]) == 2  # Text + tool blocks
            assert assistant_msg["content"][0]["type"] == "text"
            assert assistant_msg["content"][0]["text"] == "I'll help with that."
            assert assistant_msg["content"][1]["type"] == "tool_use"

    def test_update_messages_with_streaming_response(self):
        """Test updating messages with streaming response format"""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Streaming response WITH accumulated_content
            assistant_response = Mock()
            assistant_response.accumulated_content = "Let me help you."
            if hasattr(assistant_response, 'content'):
                del assistant_response.content

            tool_calls = [ToolCall(call_id="call_789", name="helper_tool", arguments='{"action": "help"}')]
            tool_results = [ToolExecutionResult(call_id="call_789", name="helper_tool", arguments='{"action": "help"}', result="Success")]

            messages = []
            updated = client._update_messages_with_tool_calls(
                messages, assistant_response, tool_calls, tool_results
            )

            # Check streaming response handling - should have text content
            assistant_msg = updated[0]
            assert assistant_msg["role"] == "assistant"
            assert len(assistant_msg["content"]) == 2  # Text + tool blocks
            assert assistant_msg["content"][0]["type"] == "text"
            assert assistant_msg["content"][0]["text"] == "Let me help you."

            # Streaming response WITHOUT accumulated_content
            assistant_response2 = Mock()
            # Explicitly remove accumulated_content to test the false branch
            if hasattr(assistant_response2, 'accumulated_content'):
                del assistant_response2.accumulated_content
            if hasattr(assistant_response2, 'content'):
                del assistant_response2.content

            tool_calls2 = [ToolCall(call_id="call_999", name="no_text_tool", arguments='{"test": true}')]
            tool_results2 = [ToolExecutionResult(call_id="call_999", name="no_text_tool", arguments='{"test": true}', result="OK")]

            messages2 = []
            updated2 = client._update_messages_with_tool_calls(
                messages2, assistant_response2, tool_calls2, tool_results2
            )

            # Check streaming response without accumulated_content - should only have tool blocks
            assistant_msg2 = updated2[0]
            assert assistant_msg2["role"] == "assistant"
            assert len(assistant_msg2["content"]) == 1  # Only tool blocks, no text
            assert assistant_msg2["content"][0]["type"] == "tool_use"

            # Streaming response WITH accumulated_content but empty string
            assistant_response3 = Mock()
            assistant_response3.accumulated_content = ""  # Empty string should be skipped
            if hasattr(assistant_response3, 'content'):
                del assistant_response3.content

            tool_calls3 = [ToolCall(call_id="call_empty", name="empty_tool", arguments='{"empty": true}')]
            tool_results3 = [ToolExecutionResult(call_id="call_empty", name="empty_tool", arguments='{"empty": true}', result="Empty OK")]

            messages3 = []
            updated3 = client._update_messages_with_tool_calls(
                messages3, assistant_response3, tool_calls3, tool_results3
            )

            # Check streaming response with empty accumulated_content - should only have tool blocks
            assistant_msg3 = updated3[0]
            assert assistant_msg3["role"] == "assistant"
            assert len(assistant_msg3["content"]) == 1  # Only tool blocks, no text since content is empty
            assert assistant_msg3["content"][0]["type"] == "tool_use"

    def test_update_messages_with_invalid_tool_arguments(self):
        """Test updating messages with invalid JSON tool arguments."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Streaming response without content attribute, but with accumulated content
            assistant_response = Mock()
            assistant_response.accumulated_content = ""  # Empty string
            if hasattr(assistant_response, 'content'):
                del assistant_response.content

            # Tool call with invalid JSON arguments
            tool_calls = [ToolCall(call_id="call_bad", name="bad_tool", arguments="invalid_json")]
            tool_results = [ToolExecutionResult(call_id="call_bad", name="bad_tool", arguments="invalid_json", result="Fixed")]

            messages = []
            updated = client._update_messages_with_tool_calls(
                messages, assistant_response, tool_calls, tool_results
            )

            # Check that invalid JSON is handled gracefully
            assistant_msg = updated[0]
            # With empty accumulated_content, only tool_use should be present
            tool_content = assistant_msg["content"][0]
            assert tool_content["type"] == "tool_use"
            assert tool_content["input"] == {}  # Should default to empty dict


class TestAnthropicAsyncClient(BaseProviderTestSuite):
    """Test suite for Anthropic async client"""

    client_class = AnthropicAsyncClient
    provider_name = "Anthropic"
    mock_async_client_path = "chimeric.providers.anthropic.client.AsyncAnthropic"

    @property
    def sample_response(self):
        """Create a sample Anthropic response."""
        response = Mock(spec=Message)
        content_block = Mock()
        content_block.type = "text"
        content_block.text = "Hello there"
        response.content = [content_block]
        response.usage = Mock(input_tokens=10, output_tokens=20, total_tokens=30)
        response.stop_reason = "end_turn"
        return response

    @property
    def sample_stream_events(self):
        """Create sample Anthropic stream events."""
        events = []

        # Text content delta event
        text_event = Mock()  # Remove spec to allow dynamic attributes
        text_event.type = "content_block_delta"
        # Use configure_mock to set the nested attributes correctly
        text_event.configure_mock(**{"delta.text": "Hello"})
        events.append(text_event)

        # Message completion
        message_stop = Mock()
        message_stop.type = "message_stop"
        events.append(message_stop)

        return events

    # ===== Async Initialization Tests =====

    async def test_async_client_initialization(self):
        """Test async client initialization."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(
                api_key="test-key",
                tool_manager=tool_manager
            )

            assert client.api_key == "test-key"
            assert client._provider_name == self.provider_name

    async def test_async_capabilities(self):
        """Test async provider capabilities."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            capabilities = client._get_capabilities()

            assert isinstance(capabilities, Capability)
            assert capabilities.streaming is True
            assert capabilities.tools is True

    async def test_async_model_aliases(self):
        """Test async model aliases."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            aliases = client._get_model_aliases()

            assert isinstance(aliases, list)
            assert "claude-opus-4-0" in aliases

    async def test_async_list_models(self):
        """Test async model listing."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path) as mock_async_anthropic:
            mock_client = AsyncMock()
            mock_async_anthropic.return_value = mock_client

            # Mock async model list
            mock_timestamp = Mock()
            mock_timestamp.timestamp.return_value = 1709251200.0  # Feb 29, 2024
            
            mock_model = Mock()
            mock_model.id = "claude-3-opus-20240229"
            mock_model.display_name = "Claude 3 Opus"
            mock_model.created_at = mock_timestamp
            mock_model.model_dump.return_value = {"id": "claude-3-opus-20240229"}
            
            mock_models_response = Mock(data=[mock_model])
            mock_client.models.list.return_value = mock_models_response

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            models = await client._list_models_impl()

            assert len(models) == 1
            assert models[0].id == "claude-3-opus-20240229"

    async def test_async_make_request(self):
        """Test async API request."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path) as mock_async_anthropic:
            mock_client = AsyncMock()
            mock_async_anthropic.return_value = mock_client
            sample_response = self.sample_response
            mock_client.messages.create.return_value = sample_response

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            response = await client._make_async_provider_request(
                messages=[{"role": "user", "content": "Hello"}],
                model="claude-3-sonnet-20240229",
                stream=False
            )

            assert response is sample_response

    async def test_async_tools_to_provider_format(self):
        """Test async formatting tools for provider."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            tools = [
                Tool(
                    name="test_tool",
                    description="Test tool",
                    parameters=ToolParameters(type="object", properties={})
                ),
                Tool(name="simple_tool", description="Simple", parameters=None)
            ]

            formatted = client._tools_to_provider_format(tools)

            assert len(formatted) == 2
            assert formatted[0]["name"] == "test_tool"
            assert formatted[1]["input_schema"] == {}

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
            assert chunks[1].common.finish_reason == "end_turn"

    async def test_async_messages_formatting_edge_cases(self):
        """Test async message formatting edge cases."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Empty messages list
            messages = []
            formatted = client._messages_to_provider_format(messages)
            assert formatted == []

            # Mixed message types
            messages = [
                ChimericMessage(role="system", content="System message"),
                ChimericMessage(role="user", content="User message"),
                ChimericMessage(role="assistant", content="Assistant message")
            ]
            formatted = client._messages_to_provider_format(messages)
            assert len(formatted) == 2  # System message filtered out
            assert formatted[0]["role"] == "user"
            assert formatted[1]["role"] == "assistant"

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

    async def test_async_content_extraction_variations(self):
        """Test async content extraction variations."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Single text block
            response1 = Mock()
            response1.content = [Mock(type="text", text="Single")]
            content1 = client._extract_content_from_response(response1)
            assert content1 == "Single"

            # Multiple text blocks
            response2 = Mock()
            response2.content = [
                Mock(type="text", text="Multi"),
                Mock(type="text", text="ple")
            ]
            content2 = client._extract_content_from_response(response2)
            assert content2 == "Multiple"

            # Empty content
            response3 = Mock()
            response3.content = []
            content3 = client._extract_content_from_response(response3)
            assert content3 == ""

    async def test_async_tool_calls_extraction_with_calls(self):
        """Test async tool call extraction with actual tool calls."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Response with tool calls
            tool_block = Mock()
            tool_block.type = "tool_use"
            tool_block.id = "call_async_123"
            tool_block.name = "async_test_tool"
            tool_block.input = {"async_param": "value"}

            response = Mock()
            response.content = [tool_block]

            tool_calls = client._extract_tool_calls_from_response(response)
            assert tool_calls is not None
            assert len(tool_calls) == 1
            assert tool_calls[0].call_id == "call_async_123"
            assert tool_calls[0].name == "async_test_tool"
            assert tool_calls[0].arguments == '{"async_param": "value"}'

    async def test_async_tool_calls_extraction_no_calls(self):
        """Test async tool call extraction with no tool calls."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Response with only text blocks
            text_block = Mock()
            text_block.type = "text"
            text_block.text = "No tools here"

            response = Mock()
            response.content = [text_block]

            tool_calls = client._extract_tool_calls_from_response(response)
            assert tool_calls is None

    async def test_async_update_messages_comprehensive(self):
        """Test async comprehensive message update scenarios."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Response with text and tool_use blocks
            text_block = Mock()
            text_block.type = "text"
            text_block.text = "I'll use a tool:"

            tool_block = Mock()
            tool_block.type = "tool_use"
            tool_block.id = "call_async_456"
            tool_block.name = "async_tool"
            tool_block.input = {"param": "test"}

            assistant_response = Mock()
            assistant_response.content = [text_block, tool_block]

            tool_calls = [ToolCall(call_id="call_async_456", name="async_tool", arguments='{"param": "test"}')]
            tool_results = [
                ToolExecutionResult(
                    call_id="call_async_456",
                    name="async_tool", 
                    arguments='{"param": "test"}',
                    result="Tool executed successfully"
                )
            ]

            messages = [{"role": "user", "content": "Test async"}]
            updated = client._update_messages_with_tool_calls(
                messages, assistant_response, tool_calls, tool_results
            )

            assert len(updated) == 3  # Original + assistant + tool results
            assert updated[1]["role"] == "assistant"
            assert len(updated[1]["content"]) == 2  # Text + tool blocks
            assert updated[2]["role"] == "user"
            assert "Tool executed successfully" in updated[2]["content"][0]["content"]

            # Response with content blocks that aren't text or tool_use
            unknown_block = Mock()
            unknown_block.type = "unknown_async_type"  # This should be ignored

            text_block2 = Mock()
            text_block2.type = "text"
            text_block2.text = "Valid async text"

            assistant_response2 = Mock()
            assistant_response2.content = [unknown_block, text_block2]

            updated2 = client._update_messages_with_tool_calls(
                messages, assistant_response2, [], []
            )

            # Should only have the text block, unknown_async_type should be ignored
            assistant_msg2 = updated2[1]
            assert len(assistant_msg2["content"]) == 1
            assert assistant_msg2["content"][0]["type"] == "text"
            assert assistant_msg2["content"][0]["text"] == "Valid async text"

    async def test_async_stream_tool_processing(self):
        """Test async stream tool call processing for coverage including unknown events."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Tool call start event
            tool_start = Mock()
            tool_start.type = "content_block_start"
            tool_start.index = 1
            tool_start.content_block = Mock()
            tool_start.content_block.type = "tool_use"
            tool_start.content_block.name = "async_tool"

            # Tool call arguments delta
            args_delta = Mock()
            args_delta.type = "content_block_delta"
            args_delta.index = 1
            args_delta.delta = Mock(spec=['partial_json'])
            args_delta.delta.partial_json = '{"async": true}'

            # Tool call complete
            tool_stop = Mock()
            tool_stop.type = "content_block_stop"
            tool_stop.index = 1

            # Unknown event type to hit the return None path
            unknown_event = Mock()
            unknown_event.type = "unknown_async_event_type"

            # Process events to cover async tool call branches
            client._process_provider_stream_event(tool_start, processor)
            client._process_provider_stream_event(args_delta, processor)
            client._process_provider_stream_event(tool_stop, processor)
            result = client._process_provider_stream_event(unknown_event, processor)
            assert result is None  # Unknown event should return None

            # Check tool calls were processed
            tool_calls = processor.get_completed_tool_calls()
            assert len(tool_calls) == 1
            assert tool_calls[0].name == "async_tool"
            assert tool_calls[0].arguments == '{"async": true}'

            # content_block_start with non-tool_use type
            non_tool_event = Mock()
            non_tool_event.type = "content_block_start"
            non_tool_event.content_block = Mock()
            non_tool_event.content_block.type = "text"  # Not tool_use
            non_tool_event.index = 2

            result2 = client._process_provider_stream_event(non_tool_event, processor)
            assert result2 is None  # Should return None without processing as tool

    async def test_async_streaming_response_paths(self):
        """Test async streaming response with accumulated content and JSON error handling."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test streaming response with accumulated content
            assistant_response = Mock()
            assistant_response.accumulated_content = "Async response"
            del assistant_response.content  # Trigger streaming path

            # Test invalid JSON arguments
            tool_calls = [ToolCall(call_id="async_call", name="async_tool", arguments="{invalid json")]
            tool_results = [ToolExecutionResult(call_id="async_call", name="async_tool", arguments="{invalid json", result="OK")]

            messages = []
            updated = client._update_messages_with_tool_calls(
                messages, assistant_response, tool_calls, tool_results
            )

            assistant_msg = updated[0]
            assert assistant_msg["content"][0]["text"] == "Async response"
            assert assistant_msg["content"][1]["input"] == {}  # Invalid JSON should become empty dict

            # Async streaming response WITHOUT accumulated_content
            assistant_response2 = Mock()
            if hasattr(assistant_response2, 'accumulated_content'):
                del assistant_response2.accumulated_content
            if hasattr(assistant_response2, 'content'):
                del assistant_response2.content

            tool_calls2 = [ToolCall(call_id="async_no_content", name="async_tool", arguments='{"test": 1}')]
            tool_results2 = [ToolExecutionResult(call_id="async_no_content", name="async_tool", arguments='{"test": 1}', result="No content OK")]

            updated2 = client._update_messages_with_tool_calls(
                [], assistant_response2, tool_calls2, tool_results2
            )

            # Should only have tool blocks, no text
            assistant_msg2 = updated2[0]
            assert len(assistant_msg2["content"]) == 1  # Only tool blocks
            assert assistant_msg2["content"][0]["type"] == "tool_use"

            # Async streaming response WITH accumulated_content but empty string
            assistant_response3 = Mock()
            assistant_response3.accumulated_content = ""  # Empty string should be skipped
            if hasattr(assistant_response3, 'content'):
                del assistant_response3.content

            tool_calls3 = [ToolCall(call_id="async_empty", name="async_tool", arguments='{"empty": true}')]
            tool_results3 = [ToolExecutionResult(call_id="async_empty", name="async_tool", arguments='{"empty": true}', result="Empty async OK")]

            updated3 = client._update_messages_with_tool_calls(
                [], assistant_response3, tool_calls3, tool_results3
            )

            # Should only have tool blocks, no text since content is empty
            assistant_msg3 = updated3[0]
            assert len(assistant_msg3["content"]) == 1  # Only tool blocks
            assert assistant_msg3["content"][0]["type"] == "tool_use"