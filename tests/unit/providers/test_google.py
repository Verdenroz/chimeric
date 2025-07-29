from unittest.mock import AsyncMock, Mock, patch

from google.genai.types import GenerateContentResponse, GenerateContentResponseUsageMetadata

from chimeric.providers.google import GoogleAsyncClient, GoogleClient
from chimeric.types import Capability, Message, Tool, ToolParameters
from chimeric.utils import StreamProcessor
from conftest import BaseProviderTestSuite


class TestGoogleClient(BaseProviderTestSuite):
    """Test suite for Google sync client."""

    client_class = GoogleClient
    provider_name = "Google"
    mock_client_path = "chimeric.providers.google.client.Client"

    @property
    def sample_response(self):
        """Create a sample Google response."""
        mock_response = Mock(spec=GenerateContentResponse)
        mock_response.text = "Hello from Google"
        mock_response.usage_metadata = Mock(spec=GenerateContentResponseUsageMetadata)
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20
        mock_response.usage_metadata.total_token_count = 30
        return mock_response

    @property
    def sample_stream_events(self):
        """Create sample Google stream events."""
        events = []

        # Text delta event
        text_event = Mock()
        text_event.text = "Hello from stream"
        events.append(text_event)

        # Empty text event
        empty_event = Mock()
        empty_event.text = ""
        events.append(empty_event)

        # Event with no text attribute
        no_text_event = Mock(spec=[])
        events.append(no_text_event)

        return events

    # ===== Initialization Tests =====

    def test_client_initialization_success(self):
        """Test successful client initialization with all parameters."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client:
            client = self.client_class(
                api_key="test-key", tool_manager=tool_manager, custom_param="value"
            )

            assert client.api_key == "test-key"
            assert client.tool_manager == tool_manager
            assert client._provider_name == self.provider_name
            mock_client.assert_called_once_with(api_key="test-key", custom_param="value")

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

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Mock model list response - create object with proper attributes
            mock_model = type(
                "MockModel",
                (),
                {"name": "gemini-pro", "display_name": "Gemini Pro", "description": "Test model"},
            )()
            mock_client.models.list.return_value = [mock_model]

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            models = client._list_models_impl()

            assert len(models) == 1
            assert models[0].id == "gemini-pro"
            assert models[0].name == "Gemini Pro"
            assert models[0].description == "Test model"

    def test_list_models_missing_attributes(self):
        """Test model listing with missing attributes."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Mock model with missing attributes
            mock_model = type(
                "MockModel",
                (),
                {"name": None, "display_name": None, "description": "Test description"},
            )()
            mock_client.models.list.return_value = [mock_model]

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            models = client._list_models_impl()

            assert len(models) == 1
            assert models[0].id == "unknown"
            assert models[0].name == "Unknown Model"

    # ===== Message Formatting Tests =====

    def test_messages_to_provider_format(self):
        """Test all message formatting scenarios."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            with (
                patch("chimeric.providers.google.client.Part") as mock_part,
                patch("chimeric.providers.google.client.Content") as mock_content,
            ):
                mock_part.from_text.return_value = Mock()
                mock_content.return_value = Mock()

                messages = [
                    # Regular string content
                    Message(role="user", content="Hello"),
                    # Assistant role (converts to 'model')
                    Message(role="assistant", content="Hi there"),
                    # Empty string content (filtered out)
                    Message(role="user", content=""),
                    # Whitespace-only content (filtered out)
                    Message(role="user", content="   "),
                    # List with valid strings
                    Message(role="user", content=["Hello", "world"]),
                    # List with mixed valid/empty strings
                    Message(role="user", content=["valid", "", "content"]),
                    # List with dict content
                    Message(role="user", content=[{"type": "text", "text": "Hello"}]),
                    # List with non-string, non-dict items (should be filtered out)
                    Message(role="user", content=["valid", 123, None, ["nested"]]),
                    # None content (filtered out)
                ]

                # Add a message with None content manually to bypass validation
                class MockMessage:
                    def __init__(self, role, content):
                        self.role = role
                        self.content = content

                messages.append(MockMessage(role="user", content=None))  # type: ignore

                client._messages_to_provider_format(messages)

                # Should only create Content objects for messages with valid parts
                expected_calls = 6  # Regular, assistant, list valid, list mixed, list dict, list with non-string items
                assert mock_content.call_count == expected_calls

                # Verify role conversion
                calls = mock_content.call_args_list
                assert calls[0][1]["role"] == "user"
                assert calls[1][1]["role"] == "model"  # assistant -> model

                # Verify parts counts
                assert len(calls[0][1]["parts"]) == 1  # "Hello"
                assert len(calls[1][1]["parts"]) == 1  # "Hi there"
                assert len(calls[2][1]["parts"]) == 2  # ["Hello", "world"]
                assert len(calls[3][1]["parts"]) == 2  # ["valid", "content"] - empty filtered
                assert len(calls[4][1]["parts"]) == 1  # dict converted to string
                assert len(calls[5][1]["parts"]) == 1  # only "valid", others filtered out

    # ===== Tool Formatting Tests =====

    def test_tools_to_provider_format_with_tools(self):
        """Test conversion of tools to Google format."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            def mock_func():
                pass

            tools = [
                Tool(
                    name="test", description="test", parameters=ToolParameters(), function=mock_func
                )
            ]

            result = client._tools_to_provider_format(tools)

            assert result is not None
            assert len(result) == 1
            assert result[0] == mock_func

    def test_tools_to_provider_format_empty_list(self):
        """Test handling of empty tools list."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            result = client._tools_to_provider_format([])

            assert result is None

    # ===== API Request Tests =====

    def test_make_provider_request_no_tools(self):
        """Test making API request without tools."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            sample_response = self.sample_response
            mock_client.models.generate_content.return_value = sample_response

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            response = client._make_provider_request(
                messages=[{"role": "user", "content": "Hello"}], model="gemini-pro", stream=False
            )

            assert response is sample_response
            mock_client.models.generate_content.assert_called_once()

    def test_make_provider_request_with_tools_streaming(self):
        """Test making streaming API request with tools."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            tools = [lambda: None]  # Mock function

            client._make_provider_request(
                messages=[{"role": "user", "content": "Hello"}],
                model="gemini-pro",
                stream=True,
                tools=tools,
                temperature=0.7,
            )

            mock_client.models.generate_content_stream.assert_called_once()

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

            # Should have one text chunk
            assert len(chunks) == 1
            assert chunks[0].common.delta == "Hello from stream"

    def test_process_stream_event_no_text(self):
        """Test processing stream event without text content."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            event = Mock()
            event.text = ""

            result = client._process_provider_stream_event(event, processor)

            assert result is None

    def test_process_stream_event_no_text_attribute(self):
        """Test processing stream event missing text attribute."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            event = Mock(spec=[])  # No text attribute

            result = client._process_provider_stream_event(event, processor)

            assert result is None

    # ===== Response Extraction Tests =====

    def test_extract_usage_from_response(self):
        """Test extracting usage information."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            usage = client._extract_usage_from_response(self.sample_response)

            assert usage.prompt_tokens == 10
            assert usage.completion_tokens == 20
            assert usage.total_tokens == 30

    def test_extract_usage_from_response_with_all_fields(self):
        """Test extracting usage information with all Google-specific fields."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Create response with complete usage metadata
            response = Mock(spec=GenerateContentResponse)
            response.usage_metadata = Mock(spec=GenerateContentResponseUsageMetadata)
            response.usage_metadata.prompt_token_count = 50
            response.usage_metadata.candidates_token_count = 30
            response.usage_metadata.total_token_count = 80
            # Add all Google-specific fields
            response.usage_metadata.cache_tokens_details = {"some": "data"}
            response.usage_metadata.cached_content_token_count = 5
            response.usage_metadata.candidates_tokens_details = {"details": "info"}
            response.usage_metadata.prompt_tokens_details = {"prompt": "details"}
            response.usage_metadata.thoughts_token_count = 3
            response.usage_metadata.tool_use_prompt_token_count = 7
            response.usage_metadata.tool_use_prompt_tokens_details = {"tool": "usage"}
            response.usage_metadata.traffic_type = "premium"

            usage = client._extract_usage_from_response(response)

            assert usage.prompt_tokens == 50
            assert usage.completion_tokens == 30
            assert usage.total_tokens == 80
            # Check extra fields
            assert usage.cached_content_token_count == 5
            assert usage.thoughts_token_count == 3

    def test_extract_content_from_response(self):
        """Test extracting content from Google response."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            content = client._extract_content_from_response(self.sample_response)

            assert content == "Hello from Google"

    def test_extract_content_from_response_no_text(self):
        """Test content extraction when response has no text."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = Mock()
            response.text = None

            content = client._extract_content_from_response(response)

            assert content == ""

    def test_extract_tool_calls_from_response(self):
        """Test extraction of tool calls from Google response."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            tool_calls = client._extract_tool_calls_from_response(self.sample_response)

            # Google client currently returns None for tool calls (placeholder implementation)
            assert tool_calls is None

    # ===== Usage Metadata Conversion Tests =====

    def test_convert_usage_metadata_complete(self):
        """Test conversion of complete usage metadata."""
        usage_metadata = Mock()
        usage_metadata.prompt_token_count = 100
        usage_metadata.candidates_token_count = 50
        usage_metadata.total_token_count = 150
        usage_metadata.cached_content_token_count = 10
        usage_metadata.thoughts_token_count = 5

        usage = GoogleClient._convert_usage_metadata(usage_metadata)

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert hasattr(usage, "cached_content_token_count")
        assert usage.cached_content_token_count == 10

    def test_convert_usage_metadata_none(self):
        """Test conversion of None usage metadata."""
        usage = GoogleClient._convert_usage_metadata(None)

        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_convert_usage_metadata_missing_fields(self):
        """Test conversion with missing metadata fields."""
        usage_metadata = Mock()
        usage_metadata.prompt_token_count = 100
        # Simulate missing attributes by raising AttributeError
        del usage_metadata.candidates_token_count
        del usage_metadata.total_token_count

        usage = GoogleClient._convert_usage_metadata(usage_metadata)

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 100  # Calculated from prompt + completion

    def test_convert_usage_metadata_all_extra_fields(self):
        """Test conversion with all Google-specific extra fields."""
        usage_metadata = Mock()
        usage_metadata.prompt_token_count = 50
        usage_metadata.candidates_token_count = 30
        usage_metadata.total_token_count = 80
        # Add all Google-specific fields
        usage_metadata.cache_tokens_details = {"some": "data"}
        usage_metadata.cached_content_token_count = 5
        usage_metadata.candidates_tokens_details = {"details": "info"}
        usage_metadata.prompt_tokens_details = {"prompt": "details"}
        usage_metadata.thoughts_token_count = 3
        usage_metadata.tool_use_prompt_token_count = 7
        usage_metadata.tool_use_prompt_tokens_details = {"tool": "usage"}
        usage_metadata.traffic_type = "premium"

        usage = GoogleClient._convert_usage_metadata(usage_metadata)

        assert usage.prompt_tokens == 50
        assert usage.completion_tokens == 30
        assert usage.total_tokens == 80
        # Check extra fields are added
        assert hasattr(usage, "cache_tokens_details")
        assert usage.cache_tokens_details == {"some": "data"}
        assert usage.cached_content_token_count == 5
        assert usage.candidates_tokens_details == {"details": "info"}
        assert usage.prompt_tokens_details == {"prompt": "details"}
        assert usage.thoughts_token_count == 3
        assert usage.tool_use_prompt_token_count == 7
        assert usage.tool_use_prompt_tokens_details == {"tool": "usage"}
        assert usage.traffic_type == "premium"

    def test_convert_usage_metadata_with_none_values(self):
        """Test conversion with Google-specific fields that are None (should not be added)."""
        usage_metadata = Mock()
        usage_metadata.prompt_token_count = 25
        usage_metadata.candidates_token_count = 15
        usage_metadata.total_token_count = 40
        # Add Google-specific fields with None values (should not be set on usage object)
        usage_metadata.cache_tokens_details = None
        usage_metadata.cached_content_token_count = None
        usage_metadata.candidates_tokens_details = None
        usage_metadata.prompt_tokens_details = None
        usage_metadata.thoughts_token_count = None
        usage_metadata.tool_use_prompt_token_count = None
        usage_metadata.tool_use_prompt_tokens_details = None
        usage_metadata.traffic_type = None

        usage = GoogleClient._convert_usage_metadata(usage_metadata)

        assert usage.prompt_tokens == 25
        assert usage.completion_tokens == 15
        assert usage.total_tokens == 40
        # None values should not be added as attributes
        assert not hasattr(usage, "cache_tokens_details")
        assert not hasattr(usage, "cached_content_token_count")
        assert not hasattr(usage, "candidates_tokens_details")
        assert not hasattr(usage, "prompt_tokens_details")
        assert not hasattr(usage, "thoughts_token_count")
        assert not hasattr(usage, "tool_use_prompt_token_count")
        assert not hasattr(usage, "tool_use_prompt_tokens_details")
        assert not hasattr(usage, "traffic_type")

    def test_convert_usage_metadata_mixed_none_and_valid_values(self):
        """Test conversion with mix of None and valid Google-specific fields."""
        usage_metadata = Mock()
        usage_metadata.prompt_token_count = 30
        usage_metadata.candidates_token_count = 20
        usage_metadata.total_token_count = 50
        # Mix of None and valid values
        usage_metadata.cache_tokens_details = {"valid": "data"}  # Valid
        usage_metadata.cached_content_token_count = None  # None
        usage_metadata.candidates_tokens_details = None  # None
        usage_metadata.prompt_tokens_details = {"prompt": "details"}  # Valid
        usage_metadata.thoughts_token_count = 5  # Valid
        usage_metadata.tool_use_prompt_token_count = None  # None
        usage_metadata.tool_use_prompt_tokens_details = None  # None
        usage_metadata.traffic_type = "standard"  # Valid

        usage = GoogleClient._convert_usage_metadata(usage_metadata)

        assert usage.prompt_tokens == 30
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 50
        # Only non-None values should be added as attributes
        assert hasattr(usage, "cache_tokens_details")
        assert usage.cache_tokens_details == {"valid": "data"}
        assert not hasattr(usage, "cached_content_token_count")
        assert not hasattr(usage, "candidates_tokens_details")
        assert hasattr(usage, "prompt_tokens_details")
        assert usage.prompt_tokens_details == {"prompt": "details"}
        assert hasattr(usage, "thoughts_token_count")
        assert usage.thoughts_token_count == 5
        assert not hasattr(usage, "tool_use_prompt_token_count")
        assert not hasattr(usage, "tool_use_prompt_tokens_details")
        assert hasattr(usage, "traffic_type")
        assert usage.traffic_type == "standard"

    def test_convert_usage_metadata_loop_coverage_explicit(self):
        """Explicit test to ensure the loop continues for None values (covers line 563->562)."""
        usage_metadata = Mock()
        usage_metadata.prompt_token_count = 35
        usage_metadata.candidates_token_count = 25
        usage_metadata.total_token_count = 60

        # Explicitly set some fields to None to ensure the loop continues
        usage_metadata.cache_tokens_details = None
        usage_metadata.cached_content_token_count = None
        usage_metadata.candidates_tokens_details = None
        usage_metadata.prompt_tokens_details = None
        usage_metadata.thoughts_token_count = None
        usage_metadata.tool_use_prompt_token_count = None
        usage_metadata.tool_use_prompt_tokens_details = None
        usage_metadata.traffic_type = None

        usage = GoogleClient._convert_usage_metadata(usage_metadata)

        # Basic assertions
        assert usage.prompt_tokens == 35
        assert usage.completion_tokens == 25
        assert usage.total_tokens == 60

        # Verify that all the None values were skipped by checking they don't exist as attributes
        google_specific_fields = [
            "cache_tokens_details",
            "cached_content_token_count",
            "candidates_tokens_details",
            "prompt_tokens_details",
            "thoughts_token_count",
            "tool_use_prompt_token_count",
            "tool_use_prompt_tokens_details",
            "traffic_type",
        ]

        for field in google_specific_fields:
            assert not hasattr(usage, field), f"Field {field} should not exist on usage object"


class TestGoogleAsyncClient(BaseProviderTestSuite):
    """Test suite for Google async client."""

    client_class = GoogleAsyncClient
    provider_name = "Google"
    mock_client_path = "chimeric.providers.google.client.Client"

    @property
    def sample_response(self):
        """Create a sample Google response."""
        mock_response = Mock(spec=GenerateContentResponse)
        mock_response.text = "Hello from Google async"
        mock_response.usage_metadata = Mock(spec=GenerateContentResponseUsageMetadata)
        mock_response.usage_metadata.prompt_token_count = 15
        mock_response.usage_metadata.candidates_token_count = 25
        mock_response.usage_metadata.total_token_count = 40
        return mock_response

    @property
    def sample_stream_events(self):
        """Create sample Google stream events."""
        events = []

        # Text delta event
        text_event = Mock()
        text_event.text = "Hello from async stream"
        events.append(text_event)

        # Empty text event
        empty_event = Mock()
        empty_event.text = ""
        events.append(empty_event)

        return events

    # ===== Async Initialization Tests =====

    async def test_async_client_initialization(self):
        """Test async client initialization."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client_instance = Mock()
            mock_client_instance.aio = Mock()
            mock_client_class.return_value = mock_client_instance

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            assert client.api_key == "test-key"
            assert client._provider_name == self.provider_name

    async def test_async_list_models(self):
        """Test async model listing."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client_instance = Mock()
            mock_async_client = AsyncMock()
            mock_client_instance.aio = mock_async_client
            mock_client_class.return_value = mock_client_instance

            # Mock async model list
            mock_model = type(
                "MockModel",
                (),
                {"name": "gemini-pro", "display_name": "Gemini Pro", "description": "Test model"},
            )()
            mock_async_client.models.list.return_value = [mock_model]

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            models = await client._list_models_impl()

            assert len(models) == 1
            assert models[0].id == "gemini-pro"

    async def test_async_make_request(self):
        """Test async API request."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client_instance = Mock()
            mock_async_client = AsyncMock()
            mock_client_instance.aio = mock_async_client
            mock_client_class.return_value = mock_client_instance

            sample_response = self.sample_response
            mock_async_client.models.generate_content.return_value = sample_response

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            response = await client._make_async_provider_request(
                messages=[{"role": "user", "content": "Hello"}], model="gemini-pro", stream=False
            )

            assert response is sample_response

    async def test_async_capabilities(self):
        """Test async provider capabilities."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client_instance = Mock()
            mock_client_instance.aio = Mock()
            mock_client_class.return_value = mock_client_instance

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            capabilities = client._get_capabilities()

            assert isinstance(capabilities, Capability)
            assert capabilities.streaming is True
            assert capabilities.tools is True

    async def test_async_stream_processing(self):
        """Test async stream processing."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client_instance = Mock()
            mock_client_instance.aio = Mock()
            mock_client_class.return_value = mock_client_instance

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            processor = StreamProcessor()

            chunks = []
            for event in self.sample_stream_events:
                chunk = client._process_provider_stream_event(event, processor)
                if chunk:
                    chunks.append(chunk)

            assert len(chunks) == 1
            assert chunks[0].common.delta == "Hello from async stream"

    async def test_async_usage_extraction_no_usage(self):
        """Test async usage extraction when no usage data."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client_instance = Mock()
            mock_client_instance.aio = Mock()
            mock_client_class.return_value = mock_client_instance

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = Mock()
            response.usage_metadata = None
            usage = client._extract_usage_from_response(response)
            assert usage.prompt_tokens == 0
            assert usage.completion_tokens == 0
            assert usage.total_tokens == 0

    async def test_async_usage_extraction_with_all_fields(self):
        """Test async usage extraction with all Google-specific fields."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client_instance = Mock()
            mock_client_instance.aio = Mock()
            mock_client_class.return_value = mock_client_instance

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            # Create response with complete usage metadata
            response = Mock(spec=GenerateContentResponse)
            response.usage_metadata = Mock(spec=GenerateContentResponseUsageMetadata)
            response.usage_metadata.prompt_token_count = 60
            response.usage_metadata.candidates_token_count = 40
            response.usage_metadata.total_token_count = 100
            # Add all Google-specific fields
            response.usage_metadata.cache_tokens_details = {"async": "data"}
            response.usage_metadata.cached_content_token_count = 8
            response.usage_metadata.candidates_tokens_details = {"async_details": "info"}
            response.usage_metadata.prompt_tokens_details = {"async_prompt": "details"}
            response.usage_metadata.thoughts_token_count = 4
            response.usage_metadata.tool_use_prompt_token_count = 9
            response.usage_metadata.tool_use_prompt_tokens_details = {"async_tool": "usage"}
            response.usage_metadata.traffic_type = "enterprise"

            usage = client._extract_usage_from_response(response)

            assert usage.prompt_tokens == 60
            assert usage.completion_tokens == 40
            assert usage.total_tokens == 100
            # Check extra fields through actual usage method call
            assert usage.cached_content_token_count == 8
            assert usage.thoughts_token_count == 4

    async def test_async_content_extraction_all_none(self):
        """Test async content extraction when text is None."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client_instance = Mock()
            mock_client_instance.aio = Mock()
            mock_client_class.return_value = mock_client_instance

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            response = Mock()
            response.text = None
            content = client._extract_content_from_response(response)
            assert content == ""

    async def test_async_messages_to_provider_format_with_dict_content(self):
        """Test async message formatting with dict content in list."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client_instance = Mock()
            mock_client_instance.aio = Mock()
            mock_client_class.return_value = mock_client_instance

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            messages = [
                Message(role="user", content=[{"type": "text", "text": "Hello"}, {"data": "test"}]),
                Message(
                    role="assistant", content="Response"
                ),  # Test assistant -> model role mapping
            ]

            with (
                patch("chimeric.providers.google.client.Part") as mock_part,
                patch("chimeric.providers.google.client.Content") as mock_content,
            ):
                mock_part.from_text.return_value = Mock()
                mock_content.return_value = Mock()

                client._messages_to_provider_format(messages)

                # Should create 2 Content objects
                assert mock_content.call_count == 2
                # Check role mapping for assistant -> model
                calls = mock_content.call_args_list
                assert calls[0][1]["role"] == "user"
                assert calls[1][1]["role"] == "model"

    async def test_async_tools_to_provider_format(self):
        """Test async tools formatting."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client_instance = Mock()
            mock_client_instance.aio = Mock()
            mock_client_class.return_value = mock_client_instance

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            def mock_func():
                pass

            tools = [
                Tool(
                    name="test", description="test", parameters=ToolParameters(), function=mock_func
                )
            ]

            result = client._tools_to_provider_format(tools)

            assert result is not None
            assert len(result) == 1
            assert result[0] == mock_func

    async def test_async_tools_to_provider_format_empty(self):
        """Test async tools formatting with empty list."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client_instance = Mock()
            mock_client_instance.aio = Mock()
            mock_client_class.return_value = mock_client_instance

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            result = client._tools_to_provider_format([])

            assert result is None

    async def test_async_make_request_streaming(self):
        """Test async streaming API request."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client_instance = Mock()
            mock_async_client = AsyncMock()
            mock_client_instance.aio = mock_async_client
            mock_client_class.return_value = mock_client_instance

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)
            tools = [lambda: None]  # Mock function

            await client._make_async_provider_request(
                messages=[{"role": "user", "content": "Hello"}],
                model="gemini-pro",
                stream=True,
                tools=tools,
                temperature=0.7,
            )

            mock_async_client.models.generate_content_stream.assert_called_once()

    async def test_async_extract_tool_calls_from_response(self):
        """Test async extraction of tool calls from Google response."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path) as mock_client_class:
            mock_client_instance = Mock()
            mock_client_instance.aio = Mock()
            mock_client_class.return_value = mock_client_instance

            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            tool_calls = client._extract_tool_calls_from_response(self.sample_response)

            # Google async client currently returns None for tool calls (placeholder implementation)
            assert tool_calls is None

    def test_messages_to_provider_format(self):
        """Test all message formatting scenarios."""
        tool_manager = self.create_tool_manager()

        with patch(self.mock_client_path):
            client = self.client_class(api_key="test-key", tool_manager=tool_manager)

            with (
                patch("chimeric.providers.google.client.Part") as mock_part,
                patch("chimeric.providers.google.client.Content") as mock_content,
            ):
                mock_part.from_text.return_value = Mock()
                mock_content.return_value = Mock()

                messages = [
                    # Regular string content
                    Message(role="user", content="Hello"),
                    # Assistant role (converts to 'model')
                    Message(role="assistant", content="Hi there"),
                    # Empty string content (filtered out)
                    Message(role="user", content=""),
                    # Whitespace-only content (filtered out)
                    Message(role="user", content="   "),
                    # List with valid strings
                    Message(role="user", content=["Hello", "world"]),
                    # List with mixed valid/empty strings
                    Message(role="user", content=["valid", "", "content"]),
                    # List with dict content
                    Message(role="user", content=[{"type": "text", "text": "Hello"}]),
                    # List with non-string, non-dict items (should be filtered out)
                    Message(role="user", content=["valid", 123, None, ["nested"]]),
                    # None content (filtered out)
                ]

                # Add a message with None content manually to bypass validation
                class MockMessage:
                    def __init__(self, role, content):
                        self.role = role
                        self.content = content

                messages.append(MockMessage(role="user", content=None))  # type: ignore

                client._messages_to_provider_format(messages)

                # Should only create Content objects for messages with valid parts
                expected_calls = 6  # Regular, assistant, list valid, list mixed, list dict, list with non-string items
                assert mock_content.call_count == expected_calls

                # Verify role conversion
                calls = mock_content.call_args_list
                assert calls[0][1]["role"] == "user"
                assert calls[1][1]["role"] == "model"  # assistant -> model

                # Verify parts counts
                assert len(calls[0][1]["parts"]) == 1  # "Hello"
                assert len(calls[1][1]["parts"]) == 1  # "Hi there"
                assert len(calls[2][1]["parts"]) == 2  # ["Hello", "world"]
                assert len(calls[3][1]["parts"]) == 2  # ["valid", "content"] - empty filtered
                assert len(calls[4][1]["parts"]) == 1  # dict converted to string
                assert len(calls[5][1]["parts"]) == 1  # only "valid", others filtered out

    async def test_async_convert_usage_metadata_loop_coverage_explicit(self):
        """Explicit test for async client's usage metadata loop coverage (line 563->562)."""
        usage_metadata = Mock()
        usage_metadata.prompt_token_count = 40
        usage_metadata.candidates_token_count = 30
        usage_metadata.total_token_count = 70

        # Set all Google-specific fields to None to test filtering behavior
        usage_metadata.cache_tokens_details = None
        usage_metadata.cached_content_token_count = None
        usage_metadata.candidates_tokens_details = None
        usage_metadata.prompt_tokens_details = None
        usage_metadata.thoughts_token_count = None
        usage_metadata.tool_use_prompt_token_count = None
        usage_metadata.tool_use_prompt_tokens_details = None
        usage_metadata.traffic_type = None

        # Call the async client's static method directly
        usage = GoogleAsyncClient._convert_usage_metadata(usage_metadata)

        # Basic assertions
        assert usage.prompt_tokens == 40
        assert usage.completion_tokens == 30
        assert usage.total_tokens == 70

        # Verify that None values were properly filtered out
        google_specific_fields = [
            "cache_tokens_details",
            "cached_content_token_count",
            "candidates_tokens_details",
            "prompt_tokens_details",
            "thoughts_token_count",
            "tool_use_prompt_token_count",
            "tool_use_prompt_tokens_details",
            "traffic_type",
        ]

        for field in google_specific_fields:
            assert not hasattr(usage, field), f"Field {field} should not exist on usage object"
