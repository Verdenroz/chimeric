from typing import Any

from google.genai import Client
from google.genai.client import AsyncClient
from google.genai.types import (
    Content,
    GenerateContentConfig,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    Part,
)

from chimeric.base import ChimericAsyncClient, ChimericClient
from chimeric.types import (
    Capability,
    ChimericStreamChunk,
    Message,
    ModelSummary,
    Tool,
    ToolCall,
    Usage,
)
from chimeric.utils import StreamProcessor, create_stream_chunk


class GoogleClient(ChimericClient[Client, GenerateContentResponse, GenerateContentResponse]):
    """Google Client for interacting with the Google Gemini API.

    This client provides a unified interface for synchronous interactions with
    Google's API via the `google-genai` library. It returns `chimeric` response
    objects that wrap the native Google responses.
    """

    def __init__(self, api_key: str, tool_manager, **kwargs: Any) -> None:
        """Initializes the synchronous Google client."""
        self._provider_name = "Google"
        super().__init__(api_key=api_key, tool_manager=tool_manager, **kwargs)

    def _get_client_type(self) -> type:
        """Get the sync Client class type.

        Returns:
            The Client class from google.genai.
        """
        return Client

    def _init_client(self, client_type: type, **kwargs: Any) -> Client:
        """Initialize the synchronous genai Client.

        Args:
            client_type: The Client class to instantiate.
            **kwargs: Additional keyword arguments including api_key.

        Returns:
            Configured synchronous Client instance.
        """
        return Client(api_key=self.api_key, **kwargs)

    def _get_capabilities(self) -> Capability:
        """Get supported features for the Google provider.

        Returns:
            Capability object indicating which features are supported:
            - streaming: True (supports real-time streaming responses)
            - tools: True (supports function calling)
        """
        return Capability(
            streaming=True,
            tools=True,
        )

    def _list_models_impl(self) -> list[ModelSummary]:
        """Lists available models from the Google API.

        Returns:
            List of ModelSummary objects containing model metadata from the API.
            Each summary includes id, name, and description.
        """
        models = []
        for model in self.client.models.list():
            model_id = model.name or "unknown"
            model_name = model.display_name or "Unknown Model"
            models.append(
                ModelSummary(
                    id=model_id,
                    name=model_name,
                    description=model.description,
                )
            )
        return models

    def _messages_to_provider_format(self, messages: list[Message]) -> Any:
        """Convert standardized messages to Google's format.

        Args:
            messages: List of standardized Message objects.

        Returns:
            Messages in Google's expected format (list of Content objects).
        """
        google_contents = []

        for message in messages:
            # Skip messages with empty content
            if not message.content:
                continue

            # Convert content to parts
            parts = []
            if isinstance(message.content, str):
                if message.content.strip():  # Only add non-empty text
                    parts.append(Part.from_text(text=message.content))
            elif isinstance(message.content, list):  # type: ignore[reportUnnecessaryIsInstance]
                for content_item in message.content:
                    if isinstance(content_item, str) and content_item.strip():
                        parts.append(Part.from_text(text=content_item))
                    elif isinstance(content_item, dict):
                        # Handle structured content - for now just convert to string
                        text_content = str(content_item)
                        if text_content.strip():
                            parts.append(Part.from_text(text=text_content))

            # Only create Content if we have parts
            if parts:
                # Map roles: assistant -> model for Google
                google_role = "model" if message.role == "assistant" else message.role
                google_contents.append(Content(role=google_role, parts=parts))

        return google_contents

    def _tools_to_provider_format(self, tools: list[Tool]) -> Any:
        """Convert standardized tools to Google's format.

        Args:
            tools: List of standardized Tool objects.

        Returns:
            Tools in Google's expected format (list of function definitions).
        """
        if not tools:
            return None

        # We just need the function definitions for Google
        return [tool.function for tool in tools]

    def _make_provider_request(
        self,
        messages: Any,
        model: str,
        stream: bool,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Make the actual API request to Google.

        Args:
            messages: Messages in Google's format.
            model: Model identifier.
            stream: Whether to stream the response.
            tools: Tools in Google's format.
            **kwargs: Additional parameters.

        Returns:
            Raw response from Google's API.
        """
        # Prepare configuration
        config: GenerateContentConfig = kwargs.pop("config", None) or GenerateContentConfig(
            **kwargs
        )
        config.tools = tools

        # Make the request
        if stream:
            return self.client.models.generate_content_stream(
                model=model,
                contents=messages,
                config=config,
            )
        return self.client.models.generate_content(
            model=model,
            contents=messages,
            config=config,
        )

    def _process_provider_stream_event(
        self, event: GenerateContentResponse, processor: StreamProcessor
    ) -> ChimericStreamChunk[Any] | None:
        """Process a single streaming response chunk.

        Args:
            event: GenerateContentResponse from the streaming API.
            processor: StreamProcessor to track state.

        Returns:
            ChimericStreamChunk with processed content or None.
        """
        delta = getattr(event, "text", "") or ""
        if delta:
            return create_stream_chunk(
                native_event=event,
                processor=processor,
                content_delta=delta,
            )
        return None

    def _extract_usage_from_response(self, response: GenerateContentResponse) -> Usage:
        """Extract usage information from Google's response.

        Args:
            response: Google's response object.

        Returns:
            Standardized Usage object.
        """
        return self._convert_usage_metadata(response.usage_metadata)

    def _extract_content_from_response(self, response: GenerateContentResponse) -> str:
        """Extract content from Google's response.

        Args:
            response: Google's response object.

        Returns:
            Text content from the response.
        """
        return response.text or ""

    def _extract_tool_calls_from_response(
        self, response: GenerateContentResponse
    ) -> list[ToolCall] | None:
        """Extract tool calls from Google's response.

        Args:
            response: Google's response object.

        Returns:
            List of ToolCall objects or None if no tool calls.
        """
        # Google handles tool calls natively in the response
        return None

    @staticmethod
    def _convert_usage_metadata(
        usage_metadata: GenerateContentResponseUsageMetadata | None,
    ) -> Usage:
        """Convert Google's usage metadata to standardized Usage format.

        Args:
            usage_metadata: Usage metadata from Google's API response.

        Returns:
            Usage object with core fields and Google-specific fields as extras.
        """
        if not usage_metadata:
            return Usage()

        # Extract core usage fields with safe defaults
        prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage_metadata, "total_token_count", 0) or (
            prompt_tokens + completion_tokens
        )

        # Create base Usage object
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        # Add Google-specific fields as extras (only non-None values)
        google_specific_fields = {
            "cache_tokens_details": getattr(usage_metadata, "cache_tokens_details", None),
            "cached_content_token_count": getattr(
                usage_metadata, "cached_content_token_count", None
            ),
            "candidates_tokens_details": getattr(usage_metadata, "candidates_tokens_details", None),
            "prompt_tokens_details": getattr(usage_metadata, "prompt_tokens_details", None),
            "thoughts_token_count": getattr(usage_metadata, "thoughts_token_count", None),
            "tool_use_prompt_token_count": getattr(
                usage_metadata, "tool_use_prompt_token_count", None
            ),
            "tool_use_prompt_tokens_details": getattr(
                usage_metadata, "tool_use_prompt_tokens_details", None
            ),
            "traffic_type": getattr(usage_metadata, "traffic_type", None),
        }

        for key, value in google_specific_fields.items():
            if value is not None:
                setattr(usage, key, value)

        return usage


class GoogleAsyncClient(
    ChimericAsyncClient[AsyncClient, GenerateContentResponse, GenerateContentResponse]
):
    """Async Google Client for interacting with the Google Gemini API.

    This client provides a unified interface for asynchronous interactions with
    Google's API via the `google-genai` library. It returns `chimeric` response
    objects that wrap the native Google responses.
    """

    def __init__(self, api_key: str, tool_manager, **kwargs: Any) -> None:
        """Initializes the asynchronous Google client."""
        self._provider_name = "Google"
        super().__init__(api_key=api_key, tool_manager=tool_manager, **kwargs)

    def _get_async_client_type(self) -> type:
        """Get the async AsyncClient class type.

        Returns:
            The AsyncClient class from google.genai.client.
        """
        return AsyncClient

    def _init_async_client(self, async_client_type: type, **kwargs: Any) -> AsyncClient:
        """Initialize the asynchronous genai AsyncClient.

        Args:
            async_client_type: The AsyncClient class to instantiate.
            **kwargs: Additional keyword arguments including api_key.

        Returns:
            Configured asynchronous AsyncClient instance.
        """
        return Client(api_key=self.api_key, **kwargs).aio

    def _get_capabilities(self) -> Capability:
        """Get supported features for the Google provider.

        Returns:
            Capability object indicating which features are supported:
            - streaming: True (supports real-time streaming responses)
            - tools: True (supports function calling)
        """
        return Capability(
            streaming=True,
            tools=True,
        )

    async def _list_models_impl(self) -> list[ModelSummary]:
        """Lists available models from the Google API.

        Returns:
            List of ModelSummary objects containing model metadata from the API.
            Each summary includes id, name, and description.
        """
        models = []
        for model in await self.async_client.models.list():
            model_id = model.name or "unknown"
            model_name = model.display_name or "Unknown Model"
            models.append(
                ModelSummary(
                    id=model_id,
                    name=model_name,
                    description=model.description,
                )
            )
        return models

    def _messages_to_provider_format(self, messages: list[Message]) -> Any:
        """Convert standardized messages to Google's format.

        Args:
            messages: List of standardized Message objects.

        Returns:
            Messages in Google's expected format (list of Content objects).
        """
        google_contents = []

        for message in messages:
            # Skip messages with empty content
            if not message.content:
                continue

            # Convert content to parts
            parts = []
            if isinstance(message.content, str):
                if message.content.strip():  # Only add non-empty text
                    parts.append(Part.from_text(text=message.content))
            elif isinstance(message.content, list):  # type: ignore[reportUnnecessaryIsInstance]
                for content_item in message.content:
                    if isinstance(content_item, str) and content_item.strip():
                        parts.append(Part.from_text(text=content_item))
                    elif isinstance(content_item, dict):
                        # Handle structured content - for now just convert to string
                        text_content = str(content_item)
                        if text_content.strip():
                            parts.append(Part.from_text(text=text_content))

            # Only create Content if we have parts
            if parts:
                # Map roles: assistant -> model for Google
                google_role = "model" if message.role == "assistant" else message.role
                google_contents.append(Content(role=google_role, parts=parts))

        return google_contents

    def _tools_to_provider_format(self, tools: list[Tool]) -> Any:
        """Convert standardized tools to Google's format.

        Args:
            tools: List of standardized Tool objects.

        Returns:
            Tools in Google's expected format (list of function definitions).
        """
        if not tools:
            return None

        # We just need the function definitions for Google
        return [tool.function for tool in tools]

    async def _make_async_provider_request(
        self,
        messages: Any,
        model: str,
        stream: bool,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Make the actual async API request to Google.

        Args:
            messages: Messages in Google's format.
            model: Model identifier.
            stream: Whether to stream the response.
            tools: Tools in Google's format.
            **kwargs: Additional parameters.

        Returns:
            Raw response from Google's async API.
        """
        # Prepare configuration
        config: GenerateContentConfig = kwargs.pop("config", None) or GenerateContentConfig(
            **kwargs
        )
        config.tools = tools

        # Make the async request
        if stream:
            return await self.async_client.models.generate_content_stream(
                model=model,
                contents=messages,
                config=config,
            )
        return await self.async_client.models.generate_content(
            model=model,
            contents=messages,
            config=config,
        )

    def _process_provider_stream_event(
        self, event: GenerateContentResponse, processor: StreamProcessor
    ) -> ChimericStreamChunk[Any] | None:
        """Process a single streaming response chunk.

        Args:
            event: GenerateContentResponse from the streaming API.
            processor: StreamProcessor to track state.

        Returns:
            ChimericStreamChunk with processed content or None.
        """
        delta = getattr(event, "text", "") or ""
        if delta:
            return create_stream_chunk(
                native_event=event,
                processor=processor,
                content_delta=delta,
            )
        return None

    def _extract_usage_from_response(self, response: GenerateContentResponse) -> Usage:
        """Extract usage information from Google's response.

        Args:
            response: Google's response object.

        Returns:
            Standardized Usage object.
        """
        return self._convert_usage_metadata(response.usage_metadata)

    def _extract_content_from_response(self, response: GenerateContentResponse) -> str:
        """Extract content from Google's response.

        Args:
            response: Google's response object.

        Returns:
            Text content from the response.
        """
        return response.text or ""

    def _extract_tool_calls_from_response(
        self, response: GenerateContentResponse
    ) -> list[ToolCall] | None:
        """Extract tool calls from Google's response.

        Args:
            response: Google's response object.

        Returns:
            List of ToolCall objects or None if no tool calls.
        """
        # This is a placeholder - implement based on Google's actual tool call format
        # You'll need to inspect the response structure to extract tool calls
        return None

    @staticmethod
    def _convert_usage_metadata(
        usage_metadata: GenerateContentResponseUsageMetadata | None,
    ) -> Usage:
        """Convert Google's usage metadata to standardized Usage format.

        Args:
            usage_metadata: Usage metadata from Google's API response.

        Returns:
            Usage object with core fields and Google-specific fields as extras.
        """
        if not usage_metadata:
            return Usage()

        # Extract core usage fields with safe defaults
        prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage_metadata, "total_token_count", 0) or (
            prompt_tokens + completion_tokens
        )

        # Create base Usage object
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        # Add Google-specific fields as extras (only non-None values)
        google_specific_fields = {
            "cache_tokens_details": getattr(usage_metadata, "cache_tokens_details", None),
            "cached_content_token_count": getattr(
                usage_metadata, "cached_content_token_count", None
            ),
            "candidates_tokens_details": getattr(usage_metadata, "candidates_tokens_details", None),
            "prompt_tokens_details": getattr(usage_metadata, "prompt_tokens_details", None),
            "thoughts_token_count": getattr(usage_metadata, "thoughts_token_count", None),
            "tool_use_prompt_token_count": getattr(
                usage_metadata, "tool_use_prompt_token_count", None
            ),
            "tool_use_prompt_tokens_details": getattr(
                usage_metadata, "tool_use_prompt_tokens_details", None
            ),
            "traffic_type": getattr(usage_metadata, "traffic_type", None),
        }

        for key, value in google_specific_fields.items():
            if value is not None:
                setattr(usage, key, value)

        return usage
