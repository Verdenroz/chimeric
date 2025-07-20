from unittest.mock import Mock, patch

from cohere import ChatResponse

from chimeric.providers.cohere import CohereAsyncClient, CohereClient
from chimeric.types import Capability, Message as ChimericMessage, Tool, ToolCall, ToolExecutionResult, ToolParameters
from chimeric.utils import StreamProcessor
from conftest import BaseProviderTestSuite


class TestCohereClient(BaseProviderTestSuite):
    """Test suite for Cohere sync client."""

    client_class = CohereClient
    provider_name = "Cohere"
    mock_client_path = "chimeric.providers.cohere.client.Cohere"

    @property
    def sample_response(self):
        """Create a sample Cohere response."""
        mock_response = Mock(spec=ChatResponse)
        mock_response.message = Mock()
        mock_response.message.content = [Mock()]
        mock_response.message.content[0].text = "Hello from Cohere"
        mock_response.message.tool_calls = None
        mock_response.usage = Mock()
        mock_response.usage.tokens = Mock()
        mock_response.usage.tokens.input_tokens = 15
        mock_response.usage.tokens.output_tokens = 25
        return mock_response

    @property
    def sample_stream_events(self):
        """Create sample Cohere stream events."""
        events = []

        # Message start event
        message_start = Mock()
        message_start.type = "message-start"
        events.append(message_start)

        # Tool plan delta event
        tool_plan_delta = Mock()
        tool_plan_delta.type = "tool-plan-delta"
        tool_plan_delta.delta = Mock()
        tool_plan_delta.delta.message = Mock()
        tool_plan_delta.delta.message.tool_plan = "I'll help you with that."
        events.append(tool_plan_delta)

        # Tool call start event
        tool_call_start = Mock()
        tool_call_start.type = "tool-call-start"
        tool_call_start.delta = Mock()
        tool_call_start.delta.message = Mock()
        tool_call_start.delta.message.tool_calls = Mock()
        tool_call_start.delta.message.tool_calls.id = "call_123"
        tool_call_start.delta.message.tool_calls.function = Mock()
        tool_call_start.delta.message.tool_calls.function.name = "test_tool"
        events.append(tool_call_start)

        # Tool call delta (arguments)
        tool_call_delta = Mock()
        tool_call_delta.type = "tool-call-delta"
        tool_call_delta.index = 0
        tool_call_delta.delta = Mock()
        tool_call_delta.delta.message = Mock()
        tool_call_delta.delta.message.tool_calls = Mock()
        tool_call_delta.delta.message.tool_calls.function = Mock()
        tool_call_delta.delta.message.tool_calls.function.arguments = '{"x": 10}'
        events.append(tool_call_delta)

        # Tool call end event
        tool_call_end = Mock()
        tool_call_end.type = "tool-call-end"
        tool_call_end.index = 0
        events.append(tool_call_end)

        # Content delta event
        content_delta = Mock()
        content_delta.type = "content-delta"
        content_delta.delta = Mock()
        content_delta.delta.message = Mock()
        content_delta.delta.message.content = Mock()
        content_delta.delta.message.content.text = "Here's your result"
        events.append(content_delta)

        # Message end event with complete finish reason
        message_end_complete = Mock()
        message_end_complete.type = "message-end"
        message_end_complete.delta = Mock()
        message_end_complete.delta.finish_reason = "COMPLETE"
        events.append(message_end_complete)

        return events

    # ===== Initialization Tests =====

    def test_client_initialization_success(self):
        """Test successful client initialization with all parameters."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client:
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            assert client._provider_name == "Cohere"
            assert client.api_key == "test-key"
            mock_client.assert_called_with(api_key="test-key")

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

        with patch(self.mock_client_path) as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test model with full attributes
            mock_model_full = Mock()
            mock_model_full.id = "command-r-plus"
            mock_model_full.name = "Command R+"
            mock_model_full.model_dump.return_value = {"id": "command-r-plus", "name": "Command R+"}

            mock_response = Mock()
            mock_response.models = [mock_model_full]
            mock_instance.models.list.return_value = mock_response

            models = client._list_models_impl()
            assert len(models) == 1
            assert models[0].id == "command-r-plus"
            assert models[0].owned_by == "cohere"

    def test_list_models_fallback_scenarios(self):
        """Test model listing with fallback scenarios."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            mock_response = Mock()
            mock_instance.models.list.return_value = mock_response

            # Test model with name only
            mock_model_name_only = Mock(spec=['name'])
            mock_model_name_only.name = "command-r"
            mock_response.models = [mock_model_name_only]
            models = client._list_models_impl()
            assert models[0].id == "command-r"

            # Test empty model
            mock_model_empty = Mock(spec=[])
            mock_response.models = [mock_model_empty]
            models = client._list_models_impl()
            assert models[0].id == "unknown"

    # ===== Message Formatting Tests =====

    def test_messages_to_provider_format_regular(self):
        """Test formatting regular messages."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            messages = [
                ChimericMessage(role="user", content="Hello"),
                ChimericMessage(role="assistant", content="Hi there"),
            ]
            formatted = client._messages_to_provider_format(messages)
            assert len(formatted) == 2
            assert formatted[0]["role"] == "user"
            assert formatted[0]["content"] == "Hello"

    # ===== Tool Formatting Tests =====

    def test_tools_to_provider_format(self):
        """Test formatting tools for provider."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            params = ToolParameters(type="object", properties={"x": {"type": "integer"}})
            tool = Tool(name="test_tool", description="Test tool", parameters=params)
            formatted_tools = client._tools_to_provider_format([tool])
            assert len(formatted_tools) == 1
            assert formatted_tools[0]["type"] == "function"
            assert formatted_tools[0]["function"]["name"] == "test_tool"

            # Test tool without parameters
            simple_tool = Tool(name="simple_tool", description="Simple tool")
            formatted_simple = client._tools_to_provider_format([simple_tool])
            assert formatted_simple[0]["function"]["parameters"] == {}

    # ===== API Request Tests =====

    def test_make_provider_request_no_tools(self):
        """Test making API request without tools."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            messages = [{"role": "user", "content": "Hello"}]
            client._make_provider_request(
                messages=messages,
                model="command-r",
                stream=False,
                temperature=0.7
            )
            mock_instance.chat.assert_called_once_with(
                model="command-r",
                messages=messages,
                tools=None,
                temperature=0.7
            )

    def test_make_provider_request_with_tools_streaming(self):
        """Test making streaming API request with tools."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            messages = [{"role": "user", "content": "Hello"}]
            tools = [{"type": "function", "function": {"name": "test"}}]

            client._make_provider_request(
                messages=messages,
                model="command-r",
                stream=True,
                tools=tools,
                temperature=0.5
            )
            mock_instance.chat_stream.assert_called_once_with(
                model="command-r",
                messages=messages,
                tools=tools,
                temperature=0.5
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
            assert len(chunks) > 0

    def test_process_stream_event_unknown_type(self):
        """Test processing unknown event type."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            unknown_event = Mock()
            unknown_event.type = "unknown-event-type"
            chunk = client._process_provider_stream_event(unknown_event, processor)
            assert chunk is None

    def test_process_stream_event_missing_attributes(self):
        """Test processing events with missing attributes."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Test event missing delta attribute
            tool_plan_no_delta = Mock(spec=['type'])
            tool_plan_no_delta.type = "tool-plan-delta"
            chunk = client._process_provider_stream_event(tool_plan_no_delta, processor)
            assert chunk is not None
            
            content_no_delta = Mock(spec=['type'])
            content_no_delta.type = "content-delta"
            chunk = client._process_provider_stream_event(content_no_delta, processor)
            assert chunk is not None
            
            message_no_delta = Mock(spec=['type'])
            message_no_delta.type = "message-end"
            chunk = client._process_provider_stream_event(message_no_delta, processor)
            assert chunk is not None
            assert chunk.common.finish_reason == "stop"

    def test_process_stream_event_tool_variations(self):
        """Test processing tool events with various scenarios."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()
            
            # Test tool start event missing delta attribute
            tool_start_no_delta = Mock(spec=['type'])
            tool_start_no_delta.type = "tool-call-start"
            chunk = client._process_provider_stream_event(tool_start_no_delta, processor)
            assert chunk is not None
            
            # Test tool delta without existing tool calls
            empty_processor = StreamProcessor()
            tool_delta = Mock()
            tool_delta.type = "tool-call-delta"
            tool_delta.index = 0
            tool_delta.delta = Mock()
            tool_delta.delta.message = Mock()
            tool_delta.delta.message.tool_calls = Mock()
            tool_delta.delta.message.tool_calls.function = Mock()
            tool_delta.delta.message.tool_calls.function.arguments = '{"test": true}'
            chunk = client._process_provider_stream_event(tool_delta, empty_processor)
            assert chunk is not None

    def test_process_stream_event_message_end_variations(self):
        """Test processing message-end events with different finish reasons."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Test message-end with TOOL_CALL finish reason
            message_end_tool_call = Mock()
            message_end_tool_call.type = "message-end"
            message_end_tool_call.delta = Mock()
            message_end_tool_call.delta.finish_reason = "TOOL_CALL"
            chunk = client._process_provider_stream_event(message_end_tool_call, processor)
            assert chunk is not None
            assert chunk.common.finish_reason == "tool_calls"

            # Test message-end with unknown finish reason
            message_end_unknown = Mock()
            message_end_unknown.type = "message-end"
            message_end_unknown.delta = Mock()
            message_end_unknown.delta.finish_reason = "UNKNOWN_REASON"
            chunk = client._process_provider_stream_event(message_end_unknown, processor)
            assert chunk is not None
            assert chunk.common.finish_reason == "stop"

            # Test message-end without finish reason
            message_end_no_finish = Mock()
            message_end_no_finish.type = "message-end"
            message_end_no_finish.delta = Mock(spec=[])
            chunk = client._process_provider_stream_event(message_end_no_finish, processor)
            assert chunk is not None
            assert chunk.common.finish_reason == "stop"
            
            # Test message-end with finish reason that's not "COMPLETE"
            message_end_other_reason = Mock()
            message_end_other_reason.type = "message-end"
            message_end_other_reason.delta = Mock()
            message_end_other_reason.delta.finish_reason = "OTHER_REASON"
            chunk = client._process_provider_stream_event(message_end_other_reason, processor)
            assert chunk is not None
            assert chunk.common.finish_reason == "stop"
            
            # Test message-end with finish reason "MAX_TOKENS" (another non-COMPLETE reason)
            message_end_max_tokens = Mock()
            message_end_max_tokens.type = "message-end"
            message_end_max_tokens.delta = Mock()
            message_end_max_tokens.delta.finish_reason = "MAX_TOKENS"
            chunk = client._process_provider_stream_event(message_end_max_tokens, processor)
            assert chunk is not None
            assert chunk.common.finish_reason == "stop"

    def test_process_event_method(self):
        """Test _process_event method functionality."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test content-delta event
            event = Mock()
            event.type = "content-delta"
            event.delta = Mock()
            event.delta.message = Mock()
            event.delta.message.content = Mock()
            event.delta.message.content.text = "test content"
            event.model_dump.return_value = {"type": "content-delta"}

            accumulated, chunk = client._process_event(event, "")
            assert accumulated == "test content"
            assert chunk is not None
            assert chunk.common.delta == "test content"

            # Test message-end event
            event_end = Mock()
            event_end.type = "message-end"
            event_end.model_dump.return_value = {"type": "message-end"}
            accumulated, chunk = client._process_event(event_end, "final")
            assert accumulated == "final"
            assert chunk.common.finish_reason == "end_turn"

            # Test unknown event type
            event_unknown = Mock()
            event_unknown.type = "unknown-type"
            accumulated, chunk = client._process_event(event_unknown, "unchanged")
            assert accumulated == "unchanged"
            assert chunk is None

    def test_process_event_fallback_scenarios(self):
        """Test _process_event method fallback scenarios."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test event missing delta attribute
            event_no_delta = Mock(spec=['type', 'model_dump'])
            event_no_delta.type = "content-delta"
            event_no_delta.model_dump.return_value = {"type": "content-delta"}
            accumulated, chunk = client._process_event(event_no_delta, "previous")
            assert accumulated == "previous"
            assert chunk.common.delta == ""

            # Test unknown event type without model_dump attribute
            event_unknown_no_dump = Mock(spec=['type'])
            event_unknown_no_dump.type = "another-unknown-type"
            accumulated, chunk = client._process_event(event_unknown_no_dump, "test")
            assert accumulated == "test"
            assert chunk is None
            
            # Test unknown event type that triggers fallback return statement
            event_fallback = Mock(spec=['type'])
            event_fallback.type = "completely-unknown-type"
            accumulated, chunk = client._process_event(event_fallback, "fallback_test")
            assert accumulated == "fallback_test"
            assert chunk is None
            
            # Test another unknown event type to ensure fallback works
            event_fallback2 = Mock()
            event_fallback2.type = "tool-call-start"  # Neither content-delta nor message-end
            accumulated2, chunk2 = client._process_event(event_fallback2, "fallback_test2")
            assert accumulated2 == "fallback_test2" 
            assert chunk2 is None

    # ===== Response Extraction Tests =====

    def test_extract_usage_from_response(self):
        """Test extracting usage information."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = self.sample_response
            usage = client._extract_usage_from_response(response)
            assert usage.prompt_tokens == 15
            assert usage.completion_tokens == 25
            assert usage.total_tokens == 40

    def test_extract_usage_edge_cases(self):
        """Test usage extraction edge cases."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Response without usage attribute
            response_no_usage = Mock(spec=[])
            usage = client._extract_usage_from_response(response_no_usage)
            assert usage.prompt_tokens == 0

            # Response with usage but no tokens
            response_no_tokens = Mock()
            response_no_tokens.usage = Mock(spec=[])
            usage = client._extract_usage_from_response(response_no_tokens)
            assert usage.prompt_tokens == 0

    def test_extract_content_from_response_variations(self):
        """Test extracting content from different response formats."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test standard response
            response = self.sample_response
            content = client._extract_content_from_response(response)
            assert content == "Hello from Cohere"

            # Test response with string content
            response_string = Mock()
            response_string.message = Mock()
            response_string.message.content = "Direct string content"
            content = client._extract_content_from_response(response_string)
            assert content == "Direct string content"

            # Test response with empty content list
            response_empty = Mock()
            response_empty.message = Mock()
            response_empty.message.content = []
            content = client._extract_content_from_response(response_empty)
            assert content == "[]"

    def test_extract_content_fallback_scenarios(self):
        """Test content extraction fallback scenarios."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test response with text attribute only (no message attribute)
            response_text_only = Mock(spec=['text'])
            response_text_only.text = "Text only response"
            content = client._extract_content_from_response(response_text_only)
            assert content == "Text only response"
            
            # Test additional edge case for text extraction
            response_with_message_no_content = Mock()
            response_with_message_no_content.message = Mock(spec=[])  # message exists but no content
            response_with_message_no_content.text = "Fallback to text"
            content = client._extract_content_from_response(response_with_message_no_content)
            assert content == "Fallback to text"

            # Test response with neither message nor text
            response_none = Mock(spec=[])
            content = client._extract_content_from_response(response_none)
            assert content == ""

    def test_extract_tool_calls_from_response(self):
        """Test extracting tool calls from response."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Response with tool calls
            response = Mock()
            response.message = Mock()
            tool_call = Mock()
            tool_call.id = "call_789"
            tool_call.function = Mock()
            tool_call.function.name = "test_function"
            tool_call.function.arguments = '{"param": "value"}'
            response.message.tool_calls = [tool_call]

            tool_calls = client._extract_tool_calls_from_response(response)
            assert len(tool_calls) == 1
            assert tool_calls[0].call_id == "call_789"
            assert tool_calls[0].name == "test_function"

    def test_extract_tool_calls_edge_cases(self):
        """Test tool call extraction edge cases."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Response without message attribute
            response_no_message = Mock(spec=[])
            tool_calls = client._extract_tool_calls_from_response(response_no_message)
            assert tool_calls is None

            # Response with message but no tool_calls
            response_no_calls = Mock()
            response_no_calls.message = Mock()
            response_no_calls.message.tool_calls = None
            tool_calls = client._extract_tool_calls_from_response(response_no_calls)
            assert tool_calls is None

    # ===== Message Update Tests =====

    def test_update_messages_with_tool_calls_success(self):
        """Test updating messages after tool execution."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            messages = [{"role": "user", "content": "Hello"}]
            assistant_response = Mock()
            assistant_response.message = Mock()
            assistant_response.message.tool_plan = "I'll help you"
            assistant_response.message.tool_calls = [Mock()]

            tool_calls = [ToolCall(call_id="call_update", name="update_tool", arguments='{"x": 5}')]
            tool_results = [ToolExecutionResult(
                call_id="call_update",
                name="update_tool",
                arguments='{"x": 5}',
                result="Updated successfully"
            )]

            updated = client._update_messages_with_tool_calls(
                messages, assistant_response, tool_calls, tool_results
            )
            assert len(updated) == 3
            assert updated[1]["role"] == "assistant"
            assert updated[1]["tool_plan"] == "I'll help you"
            assert updated[2]["role"] == "tool"

    def test_update_messages_with_tool_error(self):
        """Test updating messages with tool execution error."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            assistant_response = Mock()
            assistant_response.message = Mock()
            assistant_response.message.tool_calls = [Mock()]

            tool_results_error = [ToolExecutionResult(
                call_id="call_error",
                name="error_tool",
                arguments='{"x": 1}',
                error="Tool execution failed",
                is_error=True
            )]

            updated_error = client._update_messages_with_tool_calls(
                [], assistant_response, [ToolCall(call_id="call_error", name="error_tool", arguments='{"x": 1}')], tool_results_error
            )
            assert updated_error[-1]["content"][0]["document"]["data"] == "Tool execution failed"

    def test_update_messages_missing_attributes(self):
        """Test message updates with missing attributes."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            messages = [{"role": "user", "content": "Test"}]
            
            # Test assistant response missing message attribute
            assistant_no_message_attr = Mock(spec=[])
            updated = client._update_messages_with_tool_calls(messages, assistant_no_message_attr, [], [])
            assert len(updated) == 1  # Only original message, no assistant message added

            # Test assistant message missing tool_calls attribute
            assistant_message_no_tool_calls_attr = Mock()
            assistant_message_no_tool_calls_attr.message = Mock(spec=[])
            updated = client._update_messages_with_tool_calls(messages, assistant_message_no_tool_calls_attr, [], [])
            assert len(updated) == 1  # Only original message, no assistant message added

    # ===== Streaming Tool Calls Tests =====

    def test_streaming_tool_calls_workflow(self):
        """Test complete streaming tool calls workflow."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            
            # Test stream without tool calls
            processor = StreamProcessor()
            content_event = Mock()
            content_event.type = "content-delta"
            content_event.delta = Mock()
            content_event.delta.message = Mock()
            content_event.delta.message.content = Mock()
            content_event.delta.message.content.text = "No tool calls here"
            content_event.model_dump.return_value = {"type": "content-delta"}
            
            end_event = Mock()
            end_event.type = "message-end"
            end_event.model_dump.return_value = {"type": "message-end"}
            
            chunks = list(client._handle_streaming_tool_calls(
                iter([content_event, end_event]), processor, [], "command-r", None
            ))
            assert len(chunks) == 2

    def test_streaming_tool_calls_with_tools(self):
        """Test streaming with actual tool calls."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            
            processor = StreamProcessor()
            tool_start_event = Mock()
            tool_start_event.type = "tool-call-start"
            
            # Mock tool response
            mock_tool_response = Mock()
            mock_tool_response.message = Mock()
            mock_tool_response.message.tool_plan = "I'll use a tool"
            mock_tool_call = Mock()
            mock_tool_call.id = "call_123"
            mock_tool_call.function = Mock()
            mock_tool_call.function.name = "test_function"
            mock_tool_call.function.arguments = '{"param": "value"}'
            mock_tool_response.message.tool_calls = [mock_tool_call]
            
            mock_final_response = Mock()
            mock_final_response.message = Mock()
            mock_final_response.message.tool_calls = None
            
            content_event = Mock()
            content_event.type = "content-delta"
            content_event.delta = Mock()
            content_event.delta.message = Mock()
            content_event.delta.message.content = Mock()
            content_event.delta.message.content.text = "Result"
            content_event.model_dump.return_value = {"type": "content-delta"}
            
            end_event = Mock()
            end_event.type = "message-end"
            end_event.model_dump.return_value = {"type": "message-end"}
            
            with patch.object(client, '_make_provider_request') as mock_make_request:
                with patch.object(client, '_execute_tool_calls') as mock_execute:
                    with patch.object(client, '_process_provider_stream_event') as mock_process:
                        mock_make_request.side_effect = [
                            mock_tool_response,
                            mock_final_response,
                            iter([content_event, end_event])
                        ]
                        
                        mock_result = ToolExecutionResult(
                            call_id="call_123",
                            name="test_function",
                            arguments='{"param": "value"}',
                            result="Tool executed successfully"
                        )
                        mock_execute.return_value = [mock_result]
                        mock_process.return_value = Mock()
                        
                        chunks = list(client._handle_streaming_tool_calls(
                            iter([tool_start_event]), processor, [], "command-r", None
                        ))
                        
                        assert mock_make_request.call_count == 3
                        mock_execute.assert_called_once()

    # ===== Edge Case Tests =====

    def test_tool_event_edge_cases(self):
        """Test tool event processing edge cases."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Test tool call delta missing function attribute
            tool_delta_no_function = Mock()
            tool_delta_no_function.type = "tool-call-delta"
            tool_delta_no_function.index = 0
            tool_delta_no_function.delta = Mock()
            tool_delta_no_function.delta.message = Mock()
            tool_delta_no_function.delta.message.tool_calls = Mock(spec=[])  # No function attribute
            chunk = client._process_provider_stream_event(tool_delta_no_function, processor)
            assert chunk is not None

            # Test tool call delta with function but no arguments
            tool_delta_no_args = Mock()
            tool_delta_no_args.type = "tool-call-delta"
            tool_delta_no_args.index = 0
            tool_delta_no_args.delta = Mock()
            tool_delta_no_args.delta.message = Mock()
            tool_delta_no_args.delta.message.tool_calls = Mock()
            tool_delta_no_args.delta.message.tool_calls.function = Mock(spec=[])  # No arguments
            chunk = client._process_provider_stream_event(tool_delta_no_args, processor)
            assert chunk is not None

            # Test tool call end with empty tool_calls list
            empty_processor = StreamProcessor()
            tool_end_empty = Mock()
            tool_end_empty.type = "tool-call-end"
            tool_end_empty.index = 0
            chunk = client._process_provider_stream_event(tool_end_empty, empty_processor)
            assert chunk is not None

    def test_streaming_edge_cases(self):
        """Test streaming edge cases."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Test case where _process_event returns None
            with patch.object(client, '_process_event') as mock_process_event:
                mock_process_event.return_value = ("accumulated", None)  # Returns None
                test_event = Mock()
                test_event.type = "unknown-event"
                
                chunks = list(client._handle_streaming_tool_calls(
                    iter([test_event]), processor, [], "command-r", None
                ))
                assert len(chunks) == 0
                mock_process_event.assert_called_once()

            # Test streaming with no chunks generated
            with patch.object(client, '_process_provider_stream_event') as mock_process_stream:
                with patch.object(client, '_make_provider_request') as mock_make_request:
                    mock_process_stream.return_value = None  # Returns None
                    mock_final_response = Mock()
                    mock_final_response.message = Mock()
                    mock_final_response.message.tool_calls = None
                    
                    final_event = Mock()
                    final_event.type = "content-delta"
                    
                    mock_make_request.side_effect = [
                        mock_final_response,
                        iter([final_event])
                    ]
                    
                    tool_start_event = Mock()
                    tool_start_event.type = "tool-call-start"
                    
                    chunks = list(client._handle_streaming_tool_calls(
                        iter([tool_start_event]), processor, [], "command-r", None
                    ))
                    assert len(chunks) == 0
                    mock_process_stream.assert_called()

    def test_tool_call_attribute_variations(self):
        """Test tool call events with various attribute configurations."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Test event missing delta attribute
            tool_start_no_delta = Mock(spec=['type'])
            tool_start_no_delta.type = "tool-call-start"
            chunk = client._process_provider_stream_event(tool_start_no_delta, processor)
            assert chunk is not None

            # Test event delta missing message attribute
            tool_start_delta_no_message = Mock()
            tool_start_delta_no_message.type = "tool-call-start"
            tool_start_delta_no_message.delta = Mock(spec=[])  # No message attribute
            chunk = client._process_provider_stream_event(tool_start_delta_no_message, processor)
            assert chunk is not None

            # Test event message missing tool_calls attribute
            tool_start_message_no_tool_calls_attr = Mock()
            tool_start_message_no_tool_calls_attr.type = "tool-call-start"
            tool_start_message_no_tool_calls_attr.delta = Mock()
            tool_start_message_no_tool_calls_attr.delta.message = Mock(spec=[])  # No tool_calls attribute
            chunk = client._process_provider_stream_event(tool_start_message_no_tool_calls_attr, processor)
            assert chunk is not None

            # Test event with empty tool_calls value
            tool_start_tool_calls_falsy = Mock()
            tool_start_tool_calls_falsy.type = "tool-call-start"
            tool_start_tool_calls_falsy.delta = Mock()
            tool_start_tool_calls_falsy.delta.message = Mock()
            tool_start_tool_calls_falsy.delta.message.tool_calls = None  # Falsy value
            chunk = client._process_provider_stream_event(tool_start_tool_calls_falsy, processor)
            assert chunk is not None
            
            # Test tool-call-delta event missing required attributes
            tool_delta_missing_attrs = Mock(spec=['type'])
            tool_delta_missing_attrs.type = "tool-call-delta"
            chunk = client._process_provider_stream_event(tool_delta_missing_attrs, processor)
            assert chunk is not None


class TestCohereAsyncClient(BaseProviderTestSuite):
    """Test suite for Cohere async client."""

    client_class = CohereAsyncClient
    provider_name = "Cohere"
    mock_async_client_path = "chimeric.providers.cohere.client.AsyncCohere"

    @property
    def sample_response(self):
        """Create a sample async Cohere response."""
        mock_response = Mock(spec=ChatResponse)
        mock_response.message = Mock()
        mock_response.message.content = [Mock()]
        mock_response.message.content[0].text = "Hello from async Cohere"
        mock_response.usage = Mock()
        mock_response.usage.tokens = Mock()
        mock_response.usage.tokens.input_tokens = 20
        mock_response.usage.tokens.output_tokens = 30
        return mock_response

    @property
    def sample_stream_events(self):
        """Create sample async Cohere stream events."""
        events = []

        # Message start event
        message_start = Mock()
        message_start.type = "message-start"
        events.append(message_start)

        # Tool plan delta event
        tool_plan_delta = Mock()
        tool_plan_delta.type = "tool-plan-delta"
        tool_plan_delta.delta = Mock()
        tool_plan_delta.delta.message = Mock()
        tool_plan_delta.delta.message.tool_plan = "Async planning task."
        events.append(tool_plan_delta)

        # Tool call start event
        tool_call_start = Mock()
        tool_call_start.type = "tool-call-start"
        tool_call_start.delta = Mock()
        tool_call_start.delta.message = Mock()
        tool_call_start.delta.message.tool_calls = Mock()
        tool_call_start.delta.message.tool_calls.id = "async_call_456"
        tool_call_start.delta.message.tool_calls.function = Mock()
        tool_call_start.delta.message.tool_calls.function.name = "async_tool"
        events.append(tool_call_start)

        # Tool call delta (arguments)
        tool_call_delta = Mock()
        tool_call_delta.type = "tool-call-delta"
        tool_call_delta.index = 0
        tool_call_delta.delta = Mock()
        tool_call_delta.delta.message = Mock()
        tool_call_delta.delta.message.tool_calls = Mock()
        tool_call_delta.delta.message.tool_calls.function = Mock()
        tool_call_delta.delta.message.tool_calls.function.arguments = '{"y": 20}'
        events.append(tool_call_delta)

        # Tool call end event
        tool_call_end = Mock()
        tool_call_end.type = "tool-call-end"
        tool_call_end.index = 0
        events.append(tool_call_end)

        # Content delta event
        content_delta = Mock()
        content_delta.type = "content-delta"
        content_delta.delta = Mock()
        content_delta.delta.message = Mock()
        content_delta.delta.message.content = Mock()
        content_delta.delta.message.content.text = "Async result content"
        events.append(content_delta)

        # Message end event with complete finish reason
        message_end_complete = Mock()
        message_end_complete.type = "message-end"
        message_end_complete.delta = Mock()
        message_end_complete.delta.finish_reason = "COMPLETE"
        events.append(message_end_complete)

        return events

    # ===== Async Initialization Tests =====

    async def test_async_client_initialization(self):
        """Test async client initialization."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path) as mock_client:
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            assert client._provider_name == "Cohere"
            assert client.api_key == "test-key"
            mock_client.assert_called_with(api_key="test-key")

    async def test_async_capabilities(self):
        """Test async provider capabilities."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            capabilities = client._get_capabilities()
            assert isinstance(capabilities, Capability)
            assert capabilities.streaming is True
            assert capabilities.tools is True

    # ===== Async Model Listing Tests =====

    async def test_async_list_models(self):
        """Test async model listing."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path) as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            mock_model = Mock()
            mock_model.id = "command-r-async"
            mock_model.name = "Command R Async"
            mock_model.model_dump.return_value = {"id": "command-r-async", "async": True}

            mock_response = Mock()
            mock_response.models = [mock_model]

            async def async_list():
                return mock_response

            mock_instance.models.list = async_list

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            models = await client._list_models_impl()

            assert len(models) == 1
            assert models[0].id == "command-r-async"

    async def test_async_list_models_edge_cases(self):
        """Test async model listing edge cases."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path) as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            mock_response = Mock()

            async def async_list():
                return mock_response

            mock_instance.models.list = async_list
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test model with name only
            mock_model_name_only = Mock(spec=['name'])
            mock_model_name_only.name = "async-command-r"
            mock_response.models = [mock_model_name_only]
            models = await client._list_models_impl()
            assert models[0].id == "async-command-r"

            # Test empty model
            mock_model_empty = Mock(spec=[])
            mock_response.models = [mock_model_empty]
            models = await client._list_models_impl()
            assert models[0].id == "unknown"

    # ===== Async Message Formatting Tests =====

    async def test_async_messages_formatting(self):
        """Test async message formatting."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            messages = [ChimericMessage(role="user", content="Test async message")]
            formatted = client._messages_to_provider_format(messages)
            assert len(formatted) == 1
            assert formatted[0]["role"] == "user"

    async def test_async_tools_formatting(self):
        """Test async tool formatting."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            params = ToolParameters(type="object", properties={"x": {"type": "integer"}})
            tool = Tool(name="async_tool", description="Async tool", parameters=params)
            formatted_tools = client._tools_to_provider_format([tool])
            assert len(formatted_tools) == 1
            assert formatted_tools[0]["function"]["name"] == "async_tool"

    # ===== Async API Request Tests =====

    async def test_async_make_request(self):
        """Test async API request."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path) as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            # Test non-streaming async request
            async def async_chat(**kwargs):
                return Mock()

            mock_instance.chat = async_chat

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            result = await client._make_async_provider_request(
                messages=[{"role": "user", "content": "Hello"}],
                model="command-r",
                stream=False
            )
            assert result is not None

    async def test_async_make_request_streaming(self):
        """Test async streaming API request."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path) as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            # Test streaming async request
            async def mock_async_generator():
                yield Mock(type="content-delta")

            mock_instance.chat_stream.return_value = mock_async_generator()
            
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            result = await client._make_async_provider_request(
                messages=[{"role": "user", "content": "Stream async"}],
                model="command-r",
                stream=True
            )
            assert result is not None

    # ===== Async Stream Processing Tests =====

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
            assert len(chunks) > 0

    async def test_async_stream_processing_edge_cases(self):
        """Test async stream processing edge cases."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()
            
            # Test message-end with TOOL_CALL finish reason
            message_end_tool = Mock()
            message_end_tool.type = "message-end"
            message_end_tool.delta = Mock()
            message_end_tool.delta.finish_reason = "TOOL_CALL"
            chunk = client._process_provider_stream_event(message_end_tool, processor)
            assert chunk is not None
            assert chunk.common.finish_reason == "tool_calls"

            # Test unknown event type
            unknown_event = Mock()
            unknown_event.type = "unknown-event-type"
            chunk = client._process_provider_stream_event(unknown_event, processor)
            assert chunk is None

            # Test events with missing attributes
            tool_plan_no_delta = Mock(spec=['type'])
            tool_plan_no_delta.type = "tool-plan-delta"
            chunk = client._process_provider_stream_event(tool_plan_no_delta, processor)
            assert chunk is not None
            
            content_no_delta = Mock(spec=['type'])
            content_no_delta.type = "content-delta"
            chunk = client._process_provider_stream_event(content_no_delta, processor)
            assert chunk is not None
            
            message_no_delta = Mock(spec=['type'])
            message_no_delta.type = "message-end"
            chunk = client._process_provider_stream_event(message_no_delta, processor)
            assert chunk is not None
            assert chunk.common.finish_reason == "stop"
            
            # Test message-end with finish reason that's not "COMPLETE"
            message_end_other_async = Mock()
            message_end_other_async.type = "message-end"
            message_end_other_async.delta = Mock()
            message_end_other_async.delta.finish_reason = "TIMEOUT"
            chunk = client._process_provider_stream_event(message_end_other_async, processor)
            assert chunk is not None
            assert chunk.common.finish_reason == "stop"

    # ===== Async Response Extraction Tests =====

    async def test_async_extract_usage(self):
        """Test async usage extraction."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = self.sample_response
            usage = client._extract_usage_from_response(response)
            assert usage.prompt_tokens == 20
            assert usage.completion_tokens == 30
            assert usage.total_tokens == 50

    async def test_async_extract_usage_edge_cases(self):
        """Test async usage extraction edge cases."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response_no_usage = Mock(spec=[])
            usage = client._extract_usage_from_response(response_no_usage)
            assert usage.prompt_tokens == 0

    async def test_async_extract_content(self):
        """Test async content extraction."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = self.sample_response
            content = client._extract_content_from_response(response)
            assert content == "Hello from async Cohere"

    async def test_async_extract_content_edge_cases(self):
        """Test async content extraction edge cases."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response_no_message = Mock(spec=[])
            content = client._extract_content_from_response(response_no_message)
            assert content == ""
            
            tool_calls = client._extract_tool_calls_from_response(response_no_message)
            assert tool_calls is None

            response_string = Mock()
            response_string.message = Mock()
            response_string.message.content = "direct string"
            content = client._extract_content_from_response(response_string)
            assert content == "direct string"
            
            # Test async client text fallback (no message, only text)
            response_text_only_async = Mock(spec=['text'])
            response_text_only_async.text = "Async text only"
            content = client._extract_content_from_response(response_text_only_async)
            assert content == "Async text only"

    async def test_async_extract_tool_calls(self):
        """Test async tool call extraction."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = Mock()
            response.message = Mock()
            tool_call = Mock()
            tool_call.id = "async_call_999"
            tool_call.function = Mock()
            tool_call.function.name = "async_test_tool"
            tool_call.function.arguments = '{"async": true}'
            response.message.tool_calls = [tool_call]

            tool_calls = client._extract_tool_calls_from_response(response)
            assert len(tool_calls) == 1
            assert tool_calls[0].call_id == "async_call_999"

    # ===== Async Message Update Tests =====

    async def test_async_update_messages_with_tool_calls(self):
        """Test async message updates with tool calls."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            messages = [{"role": "user", "content": "Async test"}]
            assistant_response = Mock()
            assistant_response.message = Mock()
            assistant_response.message.tool_plan = "Async planning"
            assistant_response.message.tool_calls = [Mock()]

            tool_calls = [ToolCall(call_id="async_call", name="async_tool", arguments='{"x": 10}')]
            tool_results = [ToolExecutionResult(
                call_id="async_call",
                name="async_tool",
                arguments='{"x": 10}',
                result="Async result: 10"
            )]

            updated = client._update_messages_with_tool_calls(
                messages, assistant_response, tool_calls, tool_results
            )
            assert len(updated) == 3
            assert updated[1]["role"] == "assistant"
            assert updated[1]["tool_plan"] == "Async planning"

    async def test_async_update_messages_with_error(self):
        """Test async message updates with tool execution error."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            assistant_response = Mock()
            assistant_response.message = Mock()
            assistant_response.message.tool_calls = [Mock()]

            error_result = ToolExecutionResult(
                call_id="error_call",
                name="error_tool",
                arguments='{}',
                error="Test error",
                is_error=True
            )

            updated = client._update_messages_with_tool_calls(
                [], assistant_response, [], [error_result]
            )
            assert "Test error" in updated[-1]["content"][0]["document"]["data"]

    # ===== Async Process Event Tests =====

    async def test_async_process_event(self):
        """Test async _process_event method."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            
            content_event = Mock()
            content_event.type = "content-delta"
            content_event.delta = Mock()
            content_event.delta.message = Mock()
            content_event.delta.message.content = Mock()
            content_event.delta.message.content.text = "async test content"
            content_event.model_dump.return_value = {"type": "content-delta"}
            
            accumulated, chunk = client._process_event(content_event, "")
            assert accumulated == "async test content"
            assert chunk is not None
            
            end_event = Mock()
            end_event.type = "message-end"
            end_event.model_dump.return_value = {"type": "message-end"}
            accumulated, chunk = client._process_event(end_event, "test")
            assert accumulated == "test"
            assert chunk.common.finish_reason == "end_turn"
            
            # Test async unknown event fallback
            unknown_async_event = Mock()
            unknown_async_event.type = "unknown-async-type"
            async_accumulated, async_chunk = client._process_event(unknown_async_event, "async_fallback")
            assert async_accumulated == "async_fallback"
            assert async_chunk is None

    # ===== Async Streaming Tool Calls Tests =====

    async def test_async_streaming_tool_calls_workflow(self):
        """Test async streaming tool calls workflow."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path) as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            
            # Test stream without tool calls
            processor = StreamProcessor()
            content_event = Mock()
            content_event.type = "content-delta"
            content_event.delta = Mock()
            content_event.delta.message = Mock()
            content_event.delta.message.content = Mock()
            content_event.delta.message.content.text = "No async tool calls"
            content_event.model_dump.return_value = {"type": "content-delta"}
            
            end_event = Mock()
            end_event.type = "message-end"
            end_event.model_dump.return_value = {"type": "message-end"}
            
            async def mock_stream():
                yield content_event
                yield end_event
            
            chunks = []
            async for chunk in client._handle_streaming_tool_calls(
                mock_stream(), processor, [], "command-r", None
            ):
                chunks.append(chunk)
            assert len(chunks) == 2

    async def test_async_streaming_tool_calls_with_tools(self):
        """Test async streaming with tool calls."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path) as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            
            processor = StreamProcessor()
            tool_start_event = Mock()
            tool_start_event.type = "tool-call-start"
            
            async def mock_stream_with_tools():
                yield tool_start_event
            
            # Mock tool response
            mock_tool_response = Mock()
            mock_tool_response.message = Mock()
            mock_tool_response.message.tool_plan = "I'll use an async tool"
            mock_tool_call = Mock()
            mock_tool_call.id = "async_call_123"
            mock_tool_call.function = Mock()
            mock_tool_call.function.name = "async_test_function"
            mock_tool_call.function.arguments = '{"param": "async_value"}'
            mock_tool_response.message.tool_calls = [mock_tool_call]
            
            mock_final_response = Mock()
            mock_final_response.message = Mock()
            mock_final_response.message.tool_calls = None
            
            content_event = Mock()
            content_event.type = "content-delta"
            content_event.delta = Mock()
            content_event.delta.message = Mock()
            content_event.delta.message.content = Mock()
            content_event.delta.message.content.text = "Async result"
            content_event.model_dump.return_value = {"type": "content-delta"}
            
            end_event = Mock()
            end_event.type = "message-end"
            end_event.model_dump.return_value = {"type": "message-end"}
            
            async def mock_final_stream():
                yield content_event
                yield end_event
            
            with patch.object(client, '_make_async_provider_request') as mock_make_request:
                with patch.object(client, '_execute_tool_calls') as mock_execute:
                    with patch.object(client, '_process_provider_stream_event') as mock_process:
                        mock_make_request.side_effect = [
                            mock_tool_response,
                            mock_final_response,
                            mock_final_stream()
                        ]
                        
                        mock_result = ToolExecutionResult(
                            call_id="async_call_123",
                            name="async_test_function",
                            arguments='{"param": "async_value"}',
                            result="Async tool executed successfully"
                        )
                        mock_execute.return_value = [mock_result]
                        mock_process.return_value = Mock()
                        
                        chunks = []
                        async for chunk in client._handle_streaming_tool_calls(
                            mock_stream_with_tools(), processor, [], "command-r", None
                        ):
                            chunks.append(chunk)
                        
                        assert mock_make_request.call_count == 3
                        mock_execute.assert_called_once()

    # ===== Async Edge Case Tests =====

    async def test_async_tool_call_attribute_variations(self):
        """Test async tool call events with various attribute configurations."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Test event missing delta attribute
            async_tool_start_no_delta = Mock(spec=['type'])
            async_tool_start_no_delta.type = "tool-call-start"
            chunk = client._process_provider_stream_event(async_tool_start_no_delta, processor)
            assert chunk is not None

            # Test event delta missing message attribute
            async_tool_start_delta_no_message = Mock()
            async_tool_start_delta_no_message.type = "tool-call-start"
            async_tool_start_delta_no_message.delta = Mock(spec=[])  # No message attribute
            chunk = client._process_provider_stream_event(async_tool_start_delta_no_message, processor)
            assert chunk is not None

            # Test event message missing tool_calls attribute
            async_tool_start_message_no_tool_calls_attr = Mock()
            async_tool_start_message_no_tool_calls_attr.type = "tool-call-start"
            async_tool_start_message_no_tool_calls_attr.delta = Mock()
            async_tool_start_message_no_tool_calls_attr.delta.message = Mock(spec=[])  # No tool_calls attribute
            chunk = client._process_provider_stream_event(async_tool_start_message_no_tool_calls_attr, processor)
            assert chunk is not None

            # Test event with empty tool_calls value
            async_tool_start_tool_calls_falsy = Mock()
            async_tool_start_tool_calls_falsy.type = "tool-call-start"
            async_tool_start_tool_calls_falsy.delta = Mock()
            async_tool_start_tool_calls_falsy.delta.message = Mock()
            async_tool_start_tool_calls_falsy.delta.message.tool_calls = None  # Falsy value
            chunk = client._process_provider_stream_event(async_tool_start_tool_calls_falsy, processor)
            assert chunk is not None

    async def test_async_tool_delta_edge_cases(self):
        """Test async tool delta events with missing attributes."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Test tool delta events with missing attributes
            async_tool_delta_no_delta = Mock(spec=['type'])
            async_tool_delta_no_delta.type = "tool-call-delta"
            chunk = client._process_provider_stream_event(async_tool_delta_no_delta, processor)
            assert chunk is not None

            async_tool_delta_delta_no_message = Mock()
            async_tool_delta_delta_no_message.type = "tool-call-delta"
            async_tool_delta_delta_no_message.delta = Mock(spec=[])  # No message attribute
            chunk = client._process_provider_stream_event(async_tool_delta_delta_no_message, processor)
            assert chunk is not None

            async_tool_delta_message_no_tool_calls_attr = Mock()
            async_tool_delta_message_no_tool_calls_attr.type = "tool-call-delta"
            async_tool_delta_message_no_tool_calls_attr.delta = Mock()
            async_tool_delta_message_no_tool_calls_attr.delta.message = Mock(spec=[])  # No tool_calls attribute
            chunk = client._process_provider_stream_event(async_tool_delta_message_no_tool_calls_attr, processor)
            assert chunk is not None

            # Test tool call delta missing function attribute
            async_tool_delta_no_function_attr = Mock()
            async_tool_delta_no_function_attr.type = "tool-call-delta"
            async_tool_delta_no_function_attr.delta = Mock()
            async_tool_delta_no_function_attr.delta.message = Mock()
            async_tool_delta_no_function_attr.delta.message.tool_calls = Mock(spec=[])  # No function attribute
            chunk = client._process_provider_stream_event(async_tool_delta_no_function_attr, processor)
            assert chunk is not None

            # Test tool call delta function missing arguments attribute
            async_tool_delta_function_no_args_attr = Mock()
            async_tool_delta_function_no_args_attr.type = "tool-call-delta"
            async_tool_delta_function_no_args_attr.delta = Mock()
            async_tool_delta_function_no_args_attr.delta.message = Mock()
            async_tool_delta_function_no_args_attr.delta.message.tool_calls = Mock()
            async_tool_delta_function_no_args_attr.delta.message.tool_calls.function = Mock(spec=[])  # No arguments attribute
            chunk = client._process_provider_stream_event(async_tool_delta_function_no_args_attr, processor)
            assert chunk is not None

    async def test_async_tool_end_and_message_update_edge_cases(self):
        """Test async tool end events and message update edge cases."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test tool call delta with empty processor state
            empty_processor = StreamProcessor()
            async_tool_delta_empty_processor = Mock()
            async_tool_delta_empty_processor.type = "tool-call-delta"
            async_tool_delta_empty_processor.delta = Mock()
            async_tool_delta_empty_processor.delta.message = Mock()
            async_tool_delta_empty_processor.delta.message.tool_calls = Mock()
            async_tool_delta_empty_processor.delta.message.tool_calls.function = Mock()
            async_tool_delta_empty_processor.delta.message.tool_calls.function.arguments = '{"test": true}'
            chunk = client._process_provider_stream_event(async_tool_delta_empty_processor, empty_processor)
            assert chunk is not None

            # Test tool call end with empty processor state
            empty_processor_762 = StreamProcessor()  # Empty processor with no tool calls
            async_tool_end_empty_processor = Mock()
            async_tool_end_empty_processor.type = "tool-call-end"
            async_tool_end_empty_processor.index = 0
            chunk = client._process_provider_stream_event(async_tool_end_empty_processor, empty_processor_762)
            assert chunk is not None

            # Test message updates with missing attributes
            messages = [{"role": "user", "content": "Test"}]
            
            # Test assistant response missing message attribute
            assistant_no_message_attr = Mock(spec=[])  # No message attribute
            updated = client._update_messages_with_tool_calls(messages, assistant_no_message_attr, [], [])
            assert len(updated) == 1  # Only original message, no assistant message added

            # Test assistant message missing tool_calls attribute
            assistant_message_no_tool_calls_attr = Mock()
            assistant_message_no_tool_calls_attr.message = Mock(spec=[])  # No tool_calls attribute
            updated = client._update_messages_with_tool_calls(messages, assistant_message_no_tool_calls_attr, [], [])
            assert len(updated) == 1  # Only original message, no assistant message added

    async def test_async_streaming_edge_cases(self):
        """Test async streaming edge cases."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Test async streaming with no chunks generated
            with patch.object(client, '_process_event') as mock_async_process_event:
                with patch.object(client, '_process_provider_stream_event') as mock_async_stream_event:
                    mock_async_process_event.return_value = ("content", None)  # Returns None
                    mock_async_stream_event.return_value = None  # Returns None
                    
                    with patch.object(client, '_make_async_provider_request') as mock_async_request:
                        mock_final_response = Mock()
                        mock_final_response.message = Mock()
                        mock_final_response.message.tool_calls = None
                        
                        async def mock_final_stream():
                            yield Mock(type="content-delta")
                        
                        mock_async_request.side_effect = [
                            mock_final_response,
                            mock_final_stream()
                        ]
                        
                        content_event = Mock()
                        content_event.type = "content-delta"
                        
                        async def mock_stream_no_tools():
                            yield content_event
                        
                        chunks1 = []
                        async for chunk in client._handle_streaming_tool_calls(
                            mock_stream_no_tools(), processor, [], "command-r", None
                        ):
                            chunks1.append(chunk)
                        
                        tool_start_event = Mock()
                        tool_start_event.type = "tool-call-start"
                        
                        async def mock_stream_with_tools():
                            yield tool_start_event
                        
                        chunks2 = []
                        async for chunk in client._handle_streaming_tool_calls(
                            mock_stream_with_tools(), processor, [], "command-r", None
                        ):
                            chunks2.append(chunk)
                        
                        assert len(chunks1) == 0  # No chunks
                        assert len(chunks2) == 0  # No chunks