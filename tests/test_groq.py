from unittest.mock import AsyncMock, Mock, patch

from chimeric.providers.groq import GroqAsyncClient, GroqClient
from chimeric.types import Capability, Message, Tool, ToolCall, ToolExecutionResult, ToolParameters
from chimeric.utils import StreamProcessor
from conftest import BaseProviderTestSuite


class TestGroqClient(BaseProviderTestSuite):
    """Test suite for Groq sync client."""

    client_class = GroqClient
    provider_name = "Groq"
    mock_client_path = "chimeric.providers.groq.client.Groq"

    @property
    def sample_response(self):
        """Create a sample Groq response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Hello from Groq"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
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

    def test_initialization_edge_cases(self):
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
            from groq import Groq

            result = client._init_client(Groq, base_url="https://api.groq.com", timeout=30)
            assert result is not None

    def test_list_models_comprehensive(self):
        """Test model listing with various scenarios."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Mock models list response
            mock_model1 = Mock()
            mock_model1.id = "llama3-8b-8192"
            mock_model1.owned_by = "Meta"
            mock_model1.created = 1640995200

            mock_model2 = Mock()
            mock_model2.id = "mixtral-8x7b-32768"
            mock_model2.owned_by = "Mistral"
            mock_model2.created = 1640995300

            mock_models_response = Mock()
            mock_models_response.data = [mock_model1, mock_model2]
            mock_client.models.list.return_value = mock_models_response

            models = client._list_models_impl()
            assert len(models) == 2
            assert models[0].id == "llama3-8b-8192"
            assert models[0].owned_by == "Meta"
            assert models[1].id == "mixtral-8x7b-32768"
            assert models[1].owned_by == "Mistral"

    def test_message_format_conversion(self):
        """Test message format conversion."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test all role mappings and content types
            messages = [
                Message(role="user", content="Hello"),
                Message(role="system", content="System message"),
                Message(role="assistant", content="Assistant message"),
                Message(role="user", content=["List", "content"]),  # List content
            ]
            result = client._messages_to_provider_format(messages)
            assert len(result) == 4
            assert all(isinstance(msg, dict) for msg in result)
            assert result[0]["role"] == "user"
            assert result[0]["content"] == "Hello"
            assert result[1]["role"] == "system"
            assert result[3]["content"] == ["List", "content"]

    def test_tool_format_conversion(self):
        """Test tool format conversion."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            def mock_func():
                pass

            # Tool with parameters
            params = ToolParameters()
            params.properties = {"x": {"type": "integer"}}
            tools = [
                Tool(
                    name="test_tool", description="Test tool", parameters=params, function=mock_func
                )
            ]
            result = client._tools_to_provider_format(tools)
            assert len(result) == 1
            assert result[0]["type"] == "function"
            assert result[0]["function"]["name"] == "test_tool"
            assert result[0]["function"]["description"] == "Test tool"
            assert result[0]["function"]["parameters"]["properties"]["x"]["type"] == "integer"

            # Tool with None parameters
            tools = [
                Tool(name="test_tool", description="Test", parameters=None, function=mock_func)
            ]
            result = client._tools_to_provider_format(tools)
            assert len(result) == 1
            assert result[0]["function"]["parameters"] == {}

    def test_make_provider_request(self):
        """Test provider request method."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Non-streaming request
            mock_client.chat.completions.create.return_value = self.sample_response

            result = client._make_provider_request(
                messages=[{"role": "user", "content": "Hello"}],
                model="llama3-8b-8192",
                stream=False,
            )
            assert mock_client.chat.completions.create.called
            call_args = mock_client.chat.completions.create.call_args
            assert call_args[1]["stream"] is False

            # Streaming request
            def mock_stream():
                yield Mock()

            mock_client.chat.completions.create.return_value = mock_stream()
            result = client._make_provider_request(
                messages=[{"role": "user", "content": "Hello"}], model="llama3-8b-8192", stream=True
            )
            chunks = list(result)
            assert len(chunks) > 0

            # Request with tools and kwargs
            mock_client.chat.completions.create.return_value = self.sample_response
            result = client._make_provider_request(
                messages=[{"role": "user", "content": "Hello"}],
                model="llama3-8b-8192",
                stream=False,
                tools=[{"type": "function", "function": {"name": "test"}}],
                temperature=0.7,
            )
            call_args = mock_client.chat.completions.create.call_args
            assert "tools" in call_args[1]
            assert call_args[1]["temperature"] == 0.7

    def test_stream_processing(self):
        """Test stream processing."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Test with content
            mock_event = Mock()
            mock_event.choices = [Mock()]
            mock_event.choices[0].delta = Mock()
            mock_event.choices[0].delta.content = "Stream content"
            mock_event.choices[0].delta.tool_calls = None
            mock_event.choices[0].finish_reason = None

            result = client._process_provider_stream_event(mock_event, processor)
            assert result is not None

            # Test with no choices
            mock_event.choices = []
            result = client._process_provider_stream_event(mock_event, processor)
            assert result is None

            # Test with tool calls
            mock_event = Mock()
            mock_event.choices = [Mock()]
            mock_event.choices[0].delta = Mock()
            mock_event.choices[0].delta.content = None
            mock_tool_call = Mock()
            mock_tool_call.id = "call_123"
            mock_tool_call.index = 0
            mock_tool_call.function = Mock()
            mock_tool_call.function.name = "test_tool"
            mock_tool_call.function.arguments = '{"x": 5}'
            mock_event.choices[0].delta.tool_calls = [mock_tool_call]
            mock_event.choices[0].finish_reason = None

            result = client._process_provider_stream_event(mock_event, processor)
            # Should process tool call but not return chunk for name/args only
            assert result is None

            # Test with finish reason
            mock_event = Mock()
            mock_event.choices = [Mock()]
            mock_event.choices[0].delta = Mock()
            mock_event.choices[0].delta.content = None
            mock_event.choices[0].delta.tool_calls = None
            mock_event.choices[0].finish_reason = "stop"

            result = client._process_provider_stream_event(mock_event, processor)
            assert result is not None

    def test_usage_extraction(self):
        """Test usage extraction."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # With usage object
            usage = client._extract_usage_from_response(self.sample_response)
            assert usage.prompt_tokens == 10
            assert usage.completion_tokens == 20
            assert usage.total_tokens == 30

            # None usage
            response = Mock()
            response.usage = None
            usage = client._extract_usage_from_response(response)
            assert usage.prompt_tokens == 0

    def test_content_extraction(self):
        """Test content extraction."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            content = client._extract_content_from_response(self.sample_response)
            assert content == "Hello from Groq"

            # No choices
            response = Mock()
            response.choices = []
            content = client._extract_content_from_response(response)
            assert content == ""

            # None content
            response = Mock()
            response.choices = [Mock()]
            response.choices[0].message = Mock()
            response.choices[0].message.content = None
            content = client._extract_content_from_response(response)
            assert content == ""

    def test_tool_call_extraction(self):
        """Test tool call extraction."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # With tool calls
            response = Mock()
            response.choices = [Mock()]
            response.choices[0].message = Mock()
            mock_call = Mock()
            mock_call.id = "call_123"
            mock_call.function = Mock()
            mock_call.function.name = "test_func"
            mock_call.function.arguments = '{"x": 5}'
            response.choices[0].message.tool_calls = [mock_call]

            tool_calls = client._extract_tool_calls_from_response(response)
            assert len(tool_calls) == 1
            assert tool_calls[0].call_id == "call_123"
            assert tool_calls[0].name == "test_func"
            assert tool_calls[0].arguments == '{"x": 5}'

            # None tool calls
            response.choices[0].message.tool_calls = None
            tool_calls = client._extract_tool_calls_from_response(response)
            assert tool_calls is None

            # Empty tool calls
            response.choices[0].message.tool_calls = []
            tool_calls = client._extract_tool_calls_from_response(response)
            assert tool_calls is None

            # No choices
            response.choices = []
            tool_calls = client._extract_tool_calls_from_response(response)
            assert tool_calls is None

    def test_update_messages_with_tool_calls(self):
        """Test message updating with tool calls."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            original_messages = [{"role": "user", "content": "Hello"}]
            tool_calls = [ToolCall(call_id="call_123", name="test_tool", arguments='{"x": 5}')]
            tool_results = [
                ToolExecutionResult(
                    name="test_tool",
                    arguments='{"x": 5}',
                    call_id="call_123",
                    result="Result: 5",
                    is_error=False,
                )
            ]

            result = client._update_messages_with_tool_calls(
                messages=original_messages,
                assistant_response=Mock(),
                tool_calls=tool_calls,
                tool_results=tool_results,
            )

            assert len(result) == 3  # original + assistant + tool result
            assert result[0]["role"] == "user"
            assert result[1]["role"] == "assistant"
            assert result[1]["tool_calls"][0]["id"] == "call_123"
            assert result[2]["role"] == "tool"
            assert result[2]["tool_call_id"] == "call_123"
            assert result[2]["content"] == "Result: 5"

            # Test with error result
            error_results = [
                ToolExecutionResult(
                    name="test_tool",
                    arguments='{"x": 5}',
                    call_id="call_123",
                    result=None,
                    is_error=True,
                    error="Test error",
                )
            ]

            result = client._update_messages_with_tool_calls(
                messages=original_messages,
                assistant_response=Mock(),
                tool_calls=tool_calls,
                tool_results=error_results,
            )
            assert result[2]["content"] == "Error: Test error"

    def test_stream_processing_tool_call_edge_cases(self):
        """Test stream processing edge cases for tool calls."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Test tool call with no ID
            mock_event = Mock()
            mock_event.choices = [Mock()]
            mock_event.choices[0].delta = Mock()
            mock_event.choices[0].delta.content = None
            mock_tool_call = Mock()
            mock_tool_call.id = None
            mock_tool_call.index = 1
            mock_tool_call.function = Mock()
            mock_tool_call.function.name = "test_tool"
            mock_tool_call.function.arguments = None
            mock_event.choices[0].delta.tool_calls = [mock_tool_call]
            mock_event.choices[0].finish_reason = None

            result = client._process_provider_stream_event(mock_event, processor)
            assert result is None

            # Test tool call with only arguments
            mock_tool_call.function.name = None
            mock_tool_call.function.arguments = '{"param": "value"}'

            result = client._process_provider_stream_event(mock_event, processor)
            assert result is None

    def test_stream_processing_completion_with_tool_calls(self):
        """Test stream processing when finishing with tool calls."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # First add some tool calls to processor state
            processor.process_tool_call_start("call_1", "test_tool")
            processor.process_tool_call_delta("call_1", '{"x": 5}')

            # Test finish with tool calls in state
            mock_event = Mock()
            mock_event.choices = [Mock()]
            mock_event.choices[0].delta = Mock()
            mock_event.choices[0].delta.content = None
            mock_event.choices[0].delta.tool_calls = None
            mock_event.choices[0].finish_reason = "tool_calls"

            result = client._process_provider_stream_event(mock_event, processor)
            assert result is not None


class TestGroqAsyncClient(BaseProviderTestSuite):
    """Test suite for Groq async client."""

    client_class = GroqAsyncClient
    provider_name = "Groq"
    mock_client_path = "chimeric.providers.groq.client.AsyncGroq"

    @property
    def sample_response(self):
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Hello from async Groq"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 25
        mock_response.usage.total_tokens = 40
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
        """Test async comprehensive scenarios for coverage."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test _get_async_client_type
            client_type = client._get_async_client_type()
            assert client_type is not None

            # Test _init_async_client
            from groq import AsyncGroq

            result = client._init_async_client(AsyncGroq, base_url="https://api.groq.com")
            assert result is not None

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
            mock_model.id = "llama3-async"
            mock_model.owned_by = "Meta"
            mock_model.created = 1640995400

            mock_models_response = Mock()
            mock_models_response.data = [mock_model]
            mock_client.models.list = AsyncMock(return_value=mock_models_response)

            models = await client._list_models_impl()
            assert len(models) == 1
            assert models[0].id == "llama3-async"
            assert models[0].owned_by == "Meta"

    async def test_async_make_provider_request(self):
        """Test async provider request."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Non-streaming
            mock_client.chat.completions.create = AsyncMock(return_value=self.sample_response)

            result = await client._make_async_provider_request(
                messages=[{"role": "user", "content": "Hello"}],
                model="llama3-async",
                stream=False,
                tools=None,
            )
            assert mock_client.chat.completions.create.called

            # Streaming path
            async def mock_stream():
                yield Mock()

            mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
            result = await client._make_async_provider_request(
                messages=[{"role": "user", "content": "Hello"}],
                model="llama3-async",
                stream=True,
                tools=None,
            )

            chunks = []
            async for chunk in result:
                chunks.append(chunk)
            assert len(chunks) > 0

    async def test_async_make_request_with_tools(self):
        """Test async request with tools."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            mock_client.chat.completions.create = AsyncMock(return_value=self.sample_response)

            # Test with tools
            await client._make_async_provider_request(
                messages=[{"role": "user", "content": "Hello"}],
                model="llama3-async",
                stream=False,
                tools=[{"type": "function", "function": {"name": "test"}}],
            )
            assert mock_client.chat.completions.create.called

    async def test_async_message_and_tool_format_conversion(self):
        """Test async message and tool format conversion."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test message conversion
            messages = [
                Message(role="user", content="Hello"),
                Message(role="system", content=["System", "list"]),
            ]
            result = client._messages_to_provider_format(messages)
            assert len(result) == 2
            assert result[0]["role"] == "user"
            assert result[1]["content"] == ["System", "list"]

            # Test tool conversion
            def mock_func():
                pass

            params = ToolParameters()
            params.properties = {"x": {"type": "integer"}}
            tools = [
                Tool(name="test_tool", description="Test", parameters=params, function=mock_func)
            ]
            result = client._tools_to_provider_format(tools)
            assert len(result) == 1
            assert result[0]["type"] == "function"

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
            assert content == "Hello from async Groq"

            # Test tool call extraction
            response = Mock()
            response.choices = [Mock()]
            response.choices[0].message = Mock()
            mock_call = Mock()
            mock_call.id = "async_call_123"
            mock_call.function = Mock()
            mock_call.function.name = "async_test_func"
            mock_call.function.arguments = '{"x": 10}'
            response.choices[0].message.tool_calls = [mock_call]

            tool_calls = client._extract_tool_calls_from_response(response)
            assert len(tool_calls) == 1
            assert tool_calls[0].call_id == "async_call_123"

    async def test_async_usage_extraction_edge_cases(self):
        """Test async usage extraction with edge cases."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test with None usage
            response = Mock()
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
            response.choices = [Mock()]
            response.choices[0].message = Mock()
            response.choices[0].message.content = None
            content = client._extract_content_from_response(response)
            assert content == ""

            # Test with empty choices
            response.choices = []
            content = client._extract_content_from_response(response)
            assert content == ""

    async def test_async_tool_calls_extraction_edge_cases(self):
        """Test async tool calls extraction edge cases."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Test with None tool_calls
            response = Mock()
            response.choices = [Mock()]
            response.choices[0].message = Mock()
            response.choices[0].message.tool_calls = None
            tool_calls = client._extract_tool_calls_from_response(response)
            assert tool_calls is None

            # Test with empty tool_calls list
            response.choices[0].message.tool_calls = []
            tool_calls = client._extract_tool_calls_from_response(response)
            assert tool_calls is None

            # Test with no choices
            response.choices = []
            tool_calls = client._extract_tool_calls_from_response(response)
            assert tool_calls is None

    async def test_async_update_messages_with_tool_calls(self):
        """Test async message updating."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            original_messages = [{"role": "user", "content": "Hello"}]
            tool_calls = [
                ToolCall(call_id="async_call_123", name="async_tool", arguments='{"x": 10}')
            ]
            tool_results = [
                ToolExecutionResult(
                    name="async_tool",
                    arguments='{"x": 10}',
                    call_id="async_call_123",
                    result="Async result: 10",
                    is_error=False,
                )
            ]

            result = client._update_messages_with_tool_calls(
                messages=original_messages,
                assistant_response=Mock(),
                tool_calls=tool_calls,
                tool_results=tool_results,
            )
            assert len(result) == 3
            assert result[1]["tool_calls"][0]["function"]["name"] == "async_tool"
            assert result[2]["content"] == "Async result: 10"
            assert client._provider_name == self.provider_name

    async def test_async_stream_processing(self):
        """Test async stream processing methods."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Test with content
            mock_event = Mock()
            mock_event.choices = [Mock()]
            mock_event.choices[0].delta = Mock()
            mock_event.choices[0].delta.content = "Async stream content"
            mock_event.choices[0].delta.tool_calls = None
            mock_event.choices[0].finish_reason = None
            result = client._process_provider_stream_event(mock_event, processor)
            assert result is not None

            # Test with None content
            mock_event.choices[0].delta.content = None
            mock_event.choices[0].delta.tool_calls = None
            result = client._process_provider_stream_event(mock_event, processor)
            assert result is None

    async def test_async_stream_processing_tool_calls(self):
        """Test async stream processing with tool calls."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Test tool call in stream
            mock_event = Mock()
            mock_event.choices = [Mock()]
            mock_event.choices[0].delta = Mock()
            mock_event.choices[0].delta.content = None
            mock_tool_call = Mock()
            mock_tool_call.id = "async_call"
            mock_tool_call.index = 0
            mock_tool_call.function = Mock()
            mock_tool_call.function.name = "async_tool"
            mock_tool_call.function.arguments = '{"param": "async_value"}'
            mock_event.choices[0].delta.tool_calls = [mock_tool_call]
            mock_event.choices[0].finish_reason = None

            result = client._process_provider_stream_event(mock_event, processor)
            assert result is None  # Tool call processing doesn't return chunk

            # Test completion with tool calls
            processor.process_tool_call_start("test_call", "test_tool")
            mock_event = Mock()
            mock_event.choices = [Mock()]
            mock_event.choices[0].delta = Mock()
            mock_event.choices[0].delta.content = None
            mock_event.choices[0].delta.tool_calls = None
            mock_event.choices[0].finish_reason = "tool_calls"

            result = client._process_provider_stream_event(mock_event, processor)
            assert result is not None

    async def test_async_stream_processing_edge_cases(self):
        """Test async stream processing edge cases."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Test with no choices
            mock_event = Mock()
            mock_event.choices = []
            result = client._process_provider_stream_event(mock_event, processor)
            assert result is None

            # Test tool call with missing attributes
            mock_event = Mock()
            mock_event.choices = [Mock()]
            mock_event.choices[0].delta = Mock()
            mock_event.choices[0].delta.content = None
            mock_event.choices[0].delta.tool_calls = None
            mock_event.choices[0].finish_reason = None

            result = client._process_provider_stream_event(mock_event, processor)
            assert result is None

    async def test_async_tool_call_stream_without_function_data(self):
        """Test async tool call stream without function data."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            # Test tool call delta without function
            mock_event = Mock()
            mock_event.choices = [Mock()]
            mock_event.choices[0].delta = Mock()
            mock_event.choices[0].delta.content = None
            mock_tool_call = Mock()
            mock_tool_call.id = "call_no_func"
            mock_tool_call.index = 0
            mock_tool_call.function = None
            mock_event.choices[0].delta.tool_calls = [mock_tool_call]
            mock_event.choices[0].finish_reason = None

            result = client._process_provider_stream_event(mock_event, processor)
            assert result is None

            # Test tool call with function but no name or arguments
            mock_tool_call.function = Mock()
            mock_tool_call.function.name = None
            mock_tool_call.function.arguments = None

            result = client._process_provider_stream_event(mock_event, processor)
            assert result is None
