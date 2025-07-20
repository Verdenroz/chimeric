from unittest.mock import Mock, patch

from chimeric.providers.cerebras import CerebrasAsyncClient, CerebrasClient
from chimeric.types import Capability, Message as ChimericMessage, Tool, ToolCall, ToolExecutionResult, ToolParameters
from chimeric.utils import StreamProcessor
from conftest import BaseProviderTestSuite


class TestCerebrasClient(BaseProviderTestSuite):
    """Test suite for Cerebras sync client."""

    client_class = CerebrasClient
    provider_name = "Cerebras"
    mock_client_path = "chimeric.providers.cerebras.client.Cerebras"

    @property
    def sample_response(self):
        """Create a sample Cerebras response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Hello from Cerebras"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        return mock_response

    @property
    def sample_stream_events(self):
        """Create sample Cerebras stream events."""
        events = []

        # Text content event
        text_event = Mock()
        text_event.choices = [Mock()]
        text_event.choices[0].delta = Mock()
        text_event.choices[0].delta.content = "Hello"
        text_event.choices[0].delta.tool_calls = None
        text_event.choices[0].finish_reason = None
        events.append(text_event)

        # Tool call start event
        tool_start = Mock()
        tool_start.choices = [Mock()]
        tool_start.choices[0].delta = Mock()
        tool_start.choices[0].delta.content = None
        tool_start.choices[0].delta.tool_calls = [Mock()]
        tool_start.choices[0].delta.tool_calls[0].id = "call_123"
        tool_start.choices[0].delta.tool_calls[0].index = 0
        tool_start.choices[0].delta.tool_calls[0].function = Mock()
        tool_start.choices[0].delta.tool_calls[0].function.name = "test_tool"
        tool_start.choices[0].delta.tool_calls[0].function.arguments = None
        tool_start.choices[0].finish_reason = None
        events.append(tool_start)

        # Tool call args event
        tool_args = Mock()
        tool_args.choices = [Mock()]
        tool_args.choices[0].delta = Mock()
        tool_args.choices[0].delta.content = None
        tool_args.choices[0].delta.tool_calls = [Mock()]
        tool_args.choices[0].delta.tool_calls[0].id = "call_123"
        tool_args.choices[0].delta.tool_calls[0].index = 0
        tool_args.choices[0].delta.tool_calls[0].function = Mock()
        tool_args.choices[0].delta.tool_calls[0].function.name = None
        tool_args.choices[0].delta.tool_calls[0].function.arguments = '{"x": 10}'
        tool_args.choices[0].finish_reason = None
        events.append(tool_args)

        # Completion event
        completion_event = Mock()
        completion_event.choices = [Mock()]
        completion_event.choices[0].delta = Mock()
        completion_event.choices[0].delta.content = None
        completion_event.choices[0].delta.tool_calls = None
        completion_event.choices[0].finish_reason = "stop"
        events.append(completion_event)

        # Event with no choices (edge case)
        no_choices_event = Mock()
        no_choices_event.choices = []
        events.append(no_choices_event)

        return events

    def test_client_initialization_success(self):
        """Test successful client initialization."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            assert client._provider_name == "Cerebras"
            assert client.api_key == "test-key"

    def test_client_initialization_minimal(self):
        """Test client initialization with minimal parameters."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            assert client.api_key == "test-key"

    def test_capabilities(self):
        """Test getting provider capabilities."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            capabilities = client._get_capabilities()
            assert isinstance(capabilities, Capability)
            assert capabilities.streaming is True
            assert capabilities.tools is True

    def test_list_models_success(self):
        """Test successful model listing."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            # Mock models response
            mock_model = Mock()
            mock_model.id = "llama3.1-8b"
            mock_model.owned_by = "cerebras"
            mock_model.created = 1234567890

            mock_response = Mock()
            mock_response.data = [mock_model]
            mock_instance.models.list.return_value = mock_response

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            models = client._list_models_impl()

            assert len(models) == 1
            assert models[0].id == "llama3.1-8b"
            assert models[0].name == "llama3.1-8b"
            assert models[0].owned_by == "cerebras"
            assert models[0].created_at == 1234567890

    def test_list_models_with_missing_attributes(self):
        """Test model listing with missing attributes."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            # Mock model without optional attributes
            mock_model = Mock(spec=['id'])  # Only has id attribute
            mock_model.id = "llama3.1-70b"
            # No owned_by or created attributes

            mock_response = Mock()
            mock_response.data = [mock_model]
            mock_instance.models.list.return_value = mock_response

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            models = client._list_models_impl()

            assert len(models) == 1
            assert models[0].id == "llama3.1-70b"
            assert models[0].owned_by == "cerebras"  # Default value
            assert models[0].created_at is None

    def test_messages_to_provider_format(self):
        """Test message format conversion."""
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
            assert formatted[1]["role"] == "assistant"
            assert formatted[1]["content"] == "Hi there"

    def test_tools_to_provider_format(self):
        """Test tool format conversion."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test with ToolParameters
            params = ToolParameters(
                type="object",
                properties={"x": {"type": "integer"}},
                required=["x"],
                strict=True
            )
            tool = Tool(name="test_tool", description="Test tool", parameters=params)

            formatted = client._tools_to_provider_format([tool])
            assert len(formatted) == 1
            assert formatted[0]["type"] == "function"
            assert formatted[0]["function"]["name"] == "test_tool"
            assert formatted[0]["function"]["strict"] is True
            assert "strict" not in formatted[0]["function"]["parameters"]

    def test_tools_to_provider_format_no_parameters(self):
        """Test tool format conversion without parameters."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            tool = Tool(name="simple_tool", description="Simple tool")
            formatted = client._tools_to_provider_format([tool])

            assert len(formatted) == 1
            assert formatted[0]["function"]["parameters"] == {}

    def test_make_provider_request(self):
        """Test making provider request."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            messages = [{"role": "user", "content": "Hello"}]
            tools = [{"type": "function", "function": {"name": "test"}}]

            client._make_provider_request(
                messages=messages,
                model="llama3.1-8b",
                stream=False,
                tools=tools,
                temperature=0.7
            )

            mock_instance.chat.completions.create.assert_called_once_with(
                model="llama3.1-8b",
                messages=messages,
                stream=False,
                tools=tools,
                temperature=0.7
            )

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

            # Should get text chunk and completion chunk
            assert len(chunks) == 2
            assert chunks[0].common.delta == "Hello"
            assert chunks[1].common.finish_reason == "stop"

    def test_process_stream_event_no_choices(self):
        """Test processing stream event with no choices."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Event with no choices
            event = Mock()
            event.choices = []

            chunk = client._process_provider_stream_event(event, processor)
            assert chunk is None

    def test_process_stream_event_no_content(self):
        """Test processing stream event with no content."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Event with content = None
            event = Mock()
            event.choices = [Mock()]
            event.choices[0].delta = Mock()
            event.choices[0].delta.content = None
            event.choices[0].delta.tool_calls = None
            event.choices[0].finish_reason = None

            chunk = client._process_provider_stream_event(event, processor)
            assert chunk is None

    def test_process_stream_event_tool_call_no_id(self):
        """Test processing tool call event without id."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Tool call without id
            event = Mock()
            event.choices = [Mock()]
            event.choices[0].delta = Mock()
            event.choices[0].delta.content = None
            event.choices[0].delta.tool_calls = [Mock()]
            event.choices[0].delta.tool_calls[0].id = None
            event.choices[0].delta.tool_calls[0].index = 5
            event.choices[0].delta.tool_calls[0].function = Mock()
            event.choices[0].delta.tool_calls[0].function.name = "test_tool"
            event.choices[0].delta.tool_calls[0].function.arguments = None
            event.choices[0].finish_reason = None

            chunk = client._process_provider_stream_event(event, processor)
            assert chunk is None
            # Should use index-based id
            assert "tool_call_5" in processor.state.tool_calls

    def test_process_stream_event_tool_call_no_function(self):
        """Test processing tool call event without function."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Tool call without function
            event = Mock()
            event.choices = [Mock()]
            event.choices[0].delta = Mock()
            event.choices[0].delta.content = None
            event.choices[0].delta.tool_calls = [Mock()]
            event.choices[0].delta.tool_calls[0].id = "call_test"
            event.choices[0].delta.tool_calls[0].function = None
            event.choices[0].finish_reason = None

            chunk = client._process_provider_stream_event(event, processor)
            assert chunk is None
            # Should not create tool call
            assert len(processor.state.tool_calls) == 0

    def test_process_stream_event_tool_call_branches(self):
        """Test tool call processing branch coverage."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Tool call with function but no name
            event1 = Mock()
            event1.choices = [Mock()]
            event1.choices[0].delta = Mock()
            event1.choices[0].delta.content = None
            event1.choices[0].delta.tool_calls = [Mock()]
            event1.choices[0].delta.tool_calls[0].id = "call_no_name"
            event1.choices[0].delta.tool_calls[0].function = Mock()
            event1.choices[0].delta.tool_calls[0].function.name = None  # No name
            event1.choices[0].delta.tool_calls[0].function.arguments = '{"test": true}'
            event1.choices[0].finish_reason = None

            chunk1 = client._process_provider_stream_event(event1, processor)
            assert chunk1 is None

            # Tool call with function but no arguments
            event2 = Mock()
            event2.choices = [Mock()]
            event2.choices[0].delta = Mock()
            event2.choices[0].delta.content = None
            event2.choices[0].delta.tool_calls = [Mock()]
            event2.choices[0].delta.tool_calls[0].id = "call_no_args"
            event2.choices[0].delta.tool_calls[0].function = Mock()
            event2.choices[0].delta.tool_calls[0].function.name = "test_tool"
            event2.choices[0].delta.tool_calls[0].function.arguments = None  # No arguments
            event2.choices[0].finish_reason = None

            chunk2 = client._process_provider_stream_event(event2, processor)
            assert chunk2 is None

    def test_extract_usage_from_response(self):
        """Test extracting usage from response."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = self.sample_response
            usage = client._extract_usage_from_response(response)

            assert usage.prompt_tokens == 10
            assert usage.completion_tokens == 20
            assert usage.total_tokens == 30

    def test_extract_usage_no_usage(self):
        """Test extracting usage when none exists."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = Mock()
            response.usage = None

            usage = client._extract_usage_from_response(response)
            assert usage.prompt_tokens == 0
            assert usage.completion_tokens == 0
            assert usage.total_tokens == 0

    def test_extract_usage_partial_none_values(self):
        """Test extracting usage with None values."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = Mock()
            response.usage = Mock()
            response.usage.prompt_tokens = None
            response.usage.completion_tokens = 15
            response.usage.total_tokens = None

            usage = client._extract_usage_from_response(response)
            assert usage.prompt_tokens == 0
            assert usage.completion_tokens == 15
            assert usage.total_tokens == 0

    def test_extract_content_from_response(self):
        """Test extracting content from response."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = self.sample_response
            content = client._extract_content_from_response(response)
            assert content == "Hello from Cerebras"

    def test_extract_content_no_choices(self):
        """Test extracting content when no choices exist."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = Mock()
            response.choices = []

            content = client._extract_content_from_response(response)
            assert content == ""

    def test_extract_content_no_message(self):
        """Test extracting content when no message exists."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = Mock()
            response.choices = [Mock()]
            response.choices[0].message = None

            content = client._extract_content_from_response(response)
            assert content == ""

    def test_extract_content_no_content(self):
        """Test extracting content when content is None."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = Mock()
            response.choices = [Mock()]
            response.choices[0].message = Mock()
            response.choices[0].message.content = None

            content = client._extract_content_from_response(response)
            assert content == ""

    def test_extract_tool_calls_from_response(self):
        """Test extracting tool calls from response."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = Mock()
            response.choices = [Mock()]
            response.choices[0].message = Mock()

            # Mock tool call
            tool_call = Mock()
            tool_call.id = "call_123"
            tool_call.function = Mock()
            tool_call.function.name = "test_tool"
            tool_call.function.arguments = '{"x": 10}'

            response.choices[0].message.tool_calls = [tool_call]

            tool_calls = client._extract_tool_calls_from_response(response)
            assert len(tool_calls) == 1
            assert tool_calls[0].call_id == "call_123"
            assert tool_calls[0].name == "test_tool"
            assert tool_calls[0].arguments == '{"x": 10}'

    def test_extract_tool_calls_no_choices(self):
        """Test extracting tool calls when no choices exist."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = Mock()
            response.choices = []

            tool_calls = client._extract_tool_calls_from_response(response)
            assert tool_calls is None

    def test_extract_tool_calls_no_tool_calls(self):
        """Test extracting tool calls when none exist."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = Mock()
            response.choices = [Mock()]
            response.choices[0].message = Mock()
            response.choices[0].message.tool_calls = None

            tool_calls = client._extract_tool_calls_from_response(response)
            assert tool_calls is None

    def test_update_messages_with_tool_calls(self):
        """Test updating messages with tool calls."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            messages = [{"role": "user", "content": "Hello"}]
            assistant_response = Mock()
            tool_calls = [ToolCall(call_id="call_123", name="test_tool", arguments='{"x": 10}')]
            tool_results = [ToolExecutionResult(
                call_id="call_123",
                name="test_tool",
                arguments='{"x": 10}',
                result="Result: 10"
            )]

            updated = client._update_messages_with_tool_calls(
                messages, assistant_response, tool_calls, tool_results
            )

            assert len(updated) == 3  # Original + assistant + tool result
            assert updated[1]["role"] == "assistant"
            assert updated[1]["tool_calls"][0]["id"] == "call_123"
            assert updated[2]["role"] == "tool"
            assert updated[2]["tool_call_id"] == "call_123"
            assert updated[2]["content"] == "Result: 10"

    def test_update_messages_with_tool_error(self):
        """Test updating messages with tool error."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            messages = []
            assistant_response = Mock()
            tool_calls = [ToolCall(call_id="call_error", name="error_tool", arguments='{"x": 5}')]
            tool_results = [ToolExecutionResult(
                call_id="call_error",
                name="error_tool",
                arguments='{"x": 5}',
                error="Tool failed",
                is_error=True
            )]

            updated = client._update_messages_with_tool_calls(
                messages, assistant_response, tool_calls, tool_results
            )

            # Check error formatting in tool result content
            tool_result_content = updated[-1]["content"]
            assert "Error: Tool failed" in tool_result_content


class TestCerebrasAsyncClient(BaseProviderTestSuite):
    """Test suite for Cerebras async client."""

    client_class = CerebrasAsyncClient
    provider_name = "Cerebras"
    mock_async_client_path = "chimeric.providers.cerebras.client.AsyncCerebras"

    @property
    def sample_response(self):
        """Create a sample Cerebras response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Hello from async Cerebras"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 25
        mock_response.usage.total_tokens = 40
        return mock_response

    @property
    def sample_stream_events(self):
        """Create sample Cerebras stream events."""
        events = []

        # Text content event
        text_event = Mock()
        text_event.choices = [Mock()]
        text_event.choices[0].delta = Mock()
        text_event.choices[0].delta.content = "Async hello"
        text_event.choices[0].delta.tool_calls = None
        text_event.choices[0].finish_reason = None
        events.append(text_event)

        # Completion event
        completion_event = Mock()
        completion_event.choices = [Mock()]
        completion_event.choices[0].delta = Mock()
        completion_event.choices[0].delta.content = None
        completion_event.choices[0].delta.tool_calls = None
        completion_event.choices[0].finish_reason = "stop"
        events.append(completion_event)

        return events

    async def test_async_client_initialization(self):
        """Test async client initialization."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            assert client._provider_name == "Cerebras"
            assert client.api_key == "test-key"

    async def test_async_capabilities(self):
        """Test async client capabilities."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            capabilities = client._get_capabilities()
            assert isinstance(capabilities, Capability)
            assert capabilities.streaming is True
            assert capabilities.tools is True

    async def test_async_list_models(self):
        """Test async model listing."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path) as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            # Mock async models response
            mock_model = Mock()
            mock_model.id = "llama3.1-8b-async"
            mock_model.owned_by = "cerebras"
            mock_model.created = 1234567899

            mock_response = Mock()
            mock_response.data = [mock_model]
            
            # Create async mock for the list method
            async def async_list():
                return mock_response
            
            mock_instance.models.list = async_list

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            models = await client._list_models_impl()

            assert len(models) == 1
            assert models[0].id == "llama3.1-8b-async"
            assert models[0].owned_by == "cerebras"

    async def test_async_make_request(self):
        """Test async provider request."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path) as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance

            # Create async mock for the create method
            async def async_create(**kwargs):
                return Mock()
            
            mock_instance.chat.completions.create = async_create

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            messages = [{"role": "user", "content": "Async hello"}]
            result = await client._make_async_provider_request(
                messages=messages,
                model="llama3.1-8b",
                stream=True,
                temperature=0.5
            )

            # Verify the result is returned
            assert result is not None

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
            assert chunks[0].common.delta == "Async hello"
            assert chunks[1].common.finish_reason == "stop"

    async def test_async_messages_formatting(self):
        """Test async message formatting."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            messages = [ChimericMessage(role="user", content="Async test")]
            formatted = client._messages_to_provider_format(messages)

            assert len(formatted) == 1
            assert formatted[0]["role"] == "user"
            assert formatted[0]["content"] == "Async test"

    async def test_async_tools_formatting(self):
        """Test async tools formatting."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            tool = Tool(name="async_tool", description="Async test tool")
            formatted = client._tools_to_provider_format([tool])

            assert len(formatted) == 1
            assert formatted[0]["function"]["name"] == "async_tool"
            assert formatted[0]["function"]["strict"] is True

    async def test_async_usage_extraction(self):
        """Test async usage extraction."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = self.sample_response
            usage = client._extract_usage_from_response(response)

            assert usage.prompt_tokens == 15
            assert usage.completion_tokens == 25
            assert usage.total_tokens == 40

    async def test_async_content_extraction(self):
        """Test async content extraction."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = self.sample_response
            content = client._extract_content_from_response(response)
            assert content == "Hello from async Cerebras"

    async def test_async_tool_calls_extraction(self):
        """Test async tool calls extraction."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = Mock()
            response.choices = [Mock()]
            response.choices[0].message = Mock()

            # Mock async tool call
            tool_call = Mock()
            tool_call.id = "async_call_456"
            tool_call.function = Mock()
            tool_call.function.name = "async_test_tool"
            tool_call.function.arguments = '{"x": 20}'

            response.choices[0].message.tool_calls = [tool_call]

            tool_calls = client._extract_tool_calls_from_response(response)
            assert len(tool_calls) == 1
            assert tool_calls[0].call_id == "async_call_456"
            assert tool_calls[0].name == "async_test_tool"

    async def test_async_update_messages(self):
        """Test async message updates with tool calls."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            messages = [{"role": "user", "content": "Async test"}]
            assistant_response = Mock()
            tool_calls = [ToolCall(call_id="async_call", name="async_tool", arguments='{"x": 30}')]
            tool_results = [ToolExecutionResult(
                call_id="async_call",
                name="async_tool",
                arguments='{"x": 30}',
                result="Async result: 30"
            )]

            updated = client._update_messages_with_tool_calls(
                messages, assistant_response, tool_calls, tool_results
            )

            assert len(updated) == 3
            assert updated[1]["role"] == "assistant"
            assert updated[2]["role"] == "tool"
            assert updated[2]["content"] == "Async result: 30"

    async def test_async_comprehensive_edge_cases(self):
        """Test comprehensive edge cases for async client."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test usage extraction with no usage
            response_no_usage = Mock()
            response_no_usage.usage = None
            usage = client._extract_usage_from_response(response_no_usage)
            assert usage.prompt_tokens == 0

            # Test content extraction edge cases
            response_no_choices = Mock()
            response_no_choices.choices = []
            content = client._extract_content_from_response(response_no_choices)
            assert content == ""

            # Test tool calls extraction edge cases
            tool_calls = client._extract_tool_calls_from_response(response_no_choices)
            assert tool_calls is None

    async def test_async_stream_tool_processing(self):
        """Test async stream tool call processing."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Tool call event with multiple conditions
            tool_event = Mock()
            tool_event.choices = [Mock()]
            tool_event.choices[0].delta = Mock()
            tool_event.choices[0].delta.content = None
            tool_event.choices[0].delta.tool_calls = [Mock()]
            tool_event.choices[0].delta.tool_calls[0].id = "async_tool_call"
            tool_event.choices[0].delta.tool_calls[0].function = Mock()
            tool_event.choices[0].delta.tool_calls[0].function.name = "async_tool"
            tool_event.choices[0].delta.tool_calls[0].function.arguments = '{"test": true}'
            tool_event.choices[0].finish_reason = None

            # Process tool event
            chunk = client._process_provider_stream_event(tool_event, processor)
            assert chunk is None

            # Completion event with tool call completion
            completion_event = Mock()
            completion_event.choices = [Mock()]
            completion_event.choices[0].delta = Mock()
            completion_event.choices[0].delta.content = None
            completion_event.choices[0].delta.tool_calls = None
            completion_event.choices[0].finish_reason = "tool_calls"

            chunk = client._process_provider_stream_event(completion_event, processor)
            assert chunk is not None
            assert chunk.common.finish_reason == "tool_calls"

            # Verify tool call was processed
            tool_calls = processor.get_completed_tool_calls()
            assert len(tool_calls) == 1
            assert tool_calls[0].name == "async_tool"

    async def test_async_stream_tool_call_branches(self):
        """Test async tool call processing branch coverage."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_async_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Tool call with function but no name
            event1 = Mock()
            event1.choices = [Mock()]
            event1.choices[0].delta = Mock()
            event1.choices[0].delta.content = None
            event1.choices[0].delta.tool_calls = [Mock()]
            event1.choices[0].delta.tool_calls[0].id = "async_call_no_name"
            event1.choices[0].delta.tool_calls[0].function = Mock()
            event1.choices[0].delta.tool_calls[0].function.name = None  # No name
            event1.choices[0].delta.tool_calls[0].function.arguments = '{"async": true}'
            event1.choices[0].finish_reason = None

            chunk1 = client._process_provider_stream_event(event1, processor)
            assert chunk1 is None

            # Test Case 2: Tool call with function but no arguments
            event2 = Mock()
            event2.choices = [Mock()]
            event2.choices[0].delta = Mock()
            event2.choices[0].delta.content = None
            event2.choices[0].delta.tool_calls = [Mock()]
            event2.choices[0].delta.tool_calls[0].id = "async_call_no_args"
            event2.choices[0].delta.tool_calls[0].function = Mock()
            event2.choices[0].delta.tool_calls[0].function.name = "async_test_tool"
            event2.choices[0].delta.tool_calls[0].function.arguments = None  # No arguments
            event2.choices[0].finish_reason = None

            chunk2 = client._process_provider_stream_event(event2, processor)
            assert chunk2 is None