from collections.abc import AsyncGenerator, AsyncIterator, Generator, Iterator
from typing import Any

from google.genai import Client
from google.genai.client import AsyncClient
from google.genai.types import (
    File,
    GenerateContentConfig,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
)

from chimeric.base import BaseClient
from chimeric.types import (
    Capability,
    ChimericCompletionResponse,
    ChimericFileUploadResponse,
    ChimericStreamChunk,
    CompletionResponse,
    FileUploadResponse,
    Input,
    ModelSummary,
    StreamChunk,
    Tool,
    Tools,
    Usage,
)


class GoogleClient(
    BaseClient[Client, AsyncClient, GenerateContentResponse, GenerateContentResponse, File]
):
    """Google Gemini API client with sync/async support.

    This client provides a unified interface for interacting with Google's Gemini
    models through the genai SDK. It supports:
    - Text generation with streaming
    - Multimodal inputs (text, images, etc.)
    - Function calling with automatic tool handling
    - File uploads and management
    - Usage tracking and metadata

    Example:
        ```python
        client = GoogleClient(api_key="your-api-key")
        response = client.chat_completion(
            messages="Hello, world!",
            model="gemini-pro"
        )
        print(response.common.content)
        ```
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the GoogleClient with API credentials.

        Args:
            **kwargs: Keyword arguments passed to BaseClient. Must include 'api_key'.
                Additional arguments are forwarded to the underlying Client.

        Raises:
            ValueError: If api_key is not provided.
        """
        self._provider_name = "Google"
        super().__init__(**kwargs)

    def _get_generic_types(self) -> dict[str, type]:
        """Get mapping of sync and async client classes.

        Returns:
            Dictionary mapping 'sync' to Client and 'async' to AsyncClient.
        """
        return {"sync": Client, "async": AsyncClient}

    def _init_client(self, client_type: type, **kwargs: Any) -> Client:
        """Initialize the synchronous genai Client.

        Args:
            client_type: The Client class to instantiate.
            **kwargs: Additional keyword arguments including api_key.

        Returns:
            Configured synchronous Client instance.
        """
        return Client(api_key=self.api_key, **kwargs)

    def _init_async_client(self, async_client_type: type, **kwargs: Any) -> AsyncClient:
        """Get the asynchronous client interface.

        Args:
            async_client_type: The AsyncClient class type (unused).
            **kwargs: Additional keyword arguments.

        Returns:
            AsyncClient instance available on self.client.aio.
        """
        return self.client.aio

    def _get_capabilities(self) -> Capability:
        """Get supported features for the Google provider.

        Returns:
            Capability object indicating which features are supported:
            - multimodal: True (supports text, images, audio, etc.)
            - streaming: True (supports real-time streaming responses)
            - tools: True (supports function calling)
            - agents: False (agent workflows not supported)
            - files: True (supports file uploads and management)
        """
        return Capability(
            multimodal=True,
            streaming=True,
            tools=True,
            agents=False,
            files=True,
        )

    def list_models(self) -> list[ModelSummary]:
        """List available Gemini models.

        Returns:
            List of ModelSummary objects containing model metadata.
            Each summary includes id, name, and description.

        Raises:
            google.genai.errors.GoogleGenAIError: If API request fails.
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

    @staticmethod
    def _process_stream_event(
        event: GenerateContentResponse, accumulated: str
    ) -> tuple[str, ChimericStreamChunk[GenerateContentResponse]]:
        """Process a single streaming response chunk.

        Args:
            event: GenerateContentResponse from the streaming API.
            accumulated: Previously accumulated text content.

        Returns:
            Tuple of (updated_accumulated_text, stream_chunk).
            The stream chunk contains both the delta and full accumulated content.
        """
        delta = getattr(event, "text", "") or ""
        accumulated += delta

        chunk = ChimericStreamChunk(
            native=event,
            common=StreamChunk(
                content=accumulated,
                delta=delta,
                metadata=event.model_dump(),
            ),
        )
        return accumulated, chunk

    def _stream(
        self, stream: Iterator[GenerateContentResponse]
    ) -> Generator[ChimericStreamChunk[GenerateContentResponse], None, None]:
        """Convert SDK iterator to ChimericStreamChunk iterator.

        Args:
            stream: Iterator yielding GenerateContentResponse objects.

        Yields:
            ChimericStreamChunk with accumulated content and delta information.
        """
        accumulated = ""
        for event in stream:
            accumulated, chunk = self._process_stream_event(event, accumulated)
            yield chunk

    async def _astream(
        self, stream: AsyncIterator[GenerateContentResponse]
    ) -> AsyncGenerator[ChimericStreamChunk[GenerateContentResponse], None]:
        """Convert SDK async iterator to ChimericStreamChunk async generator.

        Args:
            stream: AsyncIterator yielding GenerateContentResponse objects.

        Yields:
            ChimericStreamChunk with accumulated content and delta information.
        """
        accumulated = ""
        async for event in stream:
            accumulated, chunk = self._process_stream_event(event, accumulated)
            yield chunk

    def _encode_tools(self, tools: Tools = None) -> Tools:
        """Encode tools for Gemini function calling.

        Args:
            tools: Optional list of Tool objects to encode.

        Returns:
            List of function definitions extracted from Tool objects,
            or None if no tools provided.

        Note:
            Gemini expects a list of function definitions rather than
            Tool wrapper objects.
        """
        if not tools:
            return None

        # Converts Tool objects to their function definitions otherwise returns the tools as
        # This is with the assumption that tools are already in the correct format as this means they are passed directly
        return [tool.function if isinstance(tool, Tool) else tool for tool in tools]

    @staticmethod
    def _convert_usage_metadata(
        usage_metadata: GenerateContentResponseUsageMetadata | None,
    ) -> Usage:
        """Convert Google's usage metadata to standardized Usage format.

        Args:
            usage_metadata: Usage metadata from Google's API response.

        Returns:
            Usage object with core fields (prompt_tokens, completion_tokens, total_tokens) and Google-specific fields as extras.
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

    def _chat_completion_impl(
        self,
        messages: Input,
        model: str,
        stream: bool = False,
        tools: Tools = None,
        **kwargs: Any,
    ) -> (
        ChimericCompletionResponse[GenerateContentResponse]
        | Generator[ChimericStreamChunk[GenerateContentResponse], None, None]
    ):
        """Internal synchronous chat completion implementation.

        Args:
            messages: Input messages (text, images, etc.).
            model: Model identifier (e.g., "gemini-pro").
            stream: Whether to return streaming response.
            tools: Optional list of Tool objects for function calling.
            **kwargs: Additional parameters forwarded to the SDK.

        Returns:
            ChimericCompletionResponse for non-streaming requests,
            or Generator of ChimericStreamChunk for streaming requests.

        Raises:
            TypeError: If tools is not a list when provided.
            google.genai.errors.GoogleGenAIError: If API request fails.
        """
        # Prepare configuration, ensuring tools are properly set
        config: GenerateContentConfig = kwargs.pop("config", None) or GenerateContentConfig()

        # Adds tools to the config if provided (already encoded in BaseClient)
        if tools:
            if not isinstance(tools, list):
                raise TypeError("Google expects tools to be a list")
            config.tools = tools

        # If stream is True, returns an iterator for streaming responses
        if stream:
            iterator = self.client.models.generate_content_stream(
                model=model,
                contents=messages,
                config=config,
            )
            return self._stream(iterator)

        response = self.client.models.generate_content(
            model=model,
            contents=messages,
            config=config,
        )

        # Convert response to ChimericCompletionResponse format
        return ChimericCompletionResponse(
            native=response,
            common=CompletionResponse(
                content=response.text or "",
                usage=self._convert_usage_metadata(response.usage_metadata),
                model=model,
                metadata=response.model_dump(),
            ),
        )

    async def _achat_completion_impl(
        self,
        messages: Input,
        model: str,
        stream: bool = False,
        tools: Tools = None,
        **kwargs: Any,
    ) -> (
        ChimericCompletionResponse[GenerateContentResponse]
        | AsyncGenerator[ChimericStreamChunk[GenerateContentResponse], None]
    ):
        """Internal asynchronous chat completion implementation.

        Args:
            messages: Input messages for the async call.
            model: Model identifier (e.g., "gemini-pro").
            stream: Whether to return streaming async generator.
            tools: Optional list of Tool objects for function calling.
            **kwargs: Additional parameters forwarded to the SDK.

        Returns:
            ChimericCompletionResponse for non-streaming requests,
            or AsyncGenerator of ChimericStreamChunk for streaming requests.

        Raises:
            TypeError: If tools is not a list when provided.
            google.genai.errors.GoogleGenAIError: If API request fails.
        """
        # Prepare configuration, ensuring tools are properly set
        config: GenerateContentConfig = kwargs.pop("config", None) or GenerateContentConfig()

        # Adds tools to the config if provided (already encoded in BaseClient)
        if tools:
            if not isinstance(tools, list):
                raise TypeError("Google expects tools to be a list")
            config.tools = tools

        # If stream is True, returns an async iterator for streaming responses
        if stream:
            async_iterator = await self.async_client.models.generate_content_stream(
                model=model,
                contents=messages,
                config=config,
            )
            return self._astream(async_iterator)

        response = await self.async_client.models.generate_content(
            model=model,
            contents=messages,
            config=config,
        )

        # Convert response to ChimericCompletionResponse format
        return ChimericCompletionResponse(
            native=response,
            common=CompletionResponse(
                content=response.text or "",
                usage=self._convert_usage_metadata(response.usage_metadata),
                model=model,
                metadata=response.model_dump(),
            ),
        )

    def _upload_file(self, **kwargs: Any) -> ChimericFileUploadResponse[File]:
        """Upload a file using Gemini's file service.

        Args:
            **kwargs: Parameters forwarded to client.files.upload.
                Common parameters include:
                - path: File path to upload
                - mime_type: MIME type of the file
                - display_name: Optional display name

        Returns:
            ChimericFileUploadResponse containing file metadata and upload details.

        Raises:
            google.genai.errors.GoogleGenAIError: If file upload fails.
        """
        filtered_kwargs = self._filter_kwargs(self.client.files.upload, kwargs)
        file_obj = self.client.files.upload(**filtered_kwargs)

        # Extract file metadata with safe defaults
        file_id = file_obj.name or "unknown"
        filename = file_obj.display_name or "unknown"
        file_size = file_obj.size_bytes or 0

        # Convert datetime to timestamp if available
        created_at = None
        if file_obj.create_time and hasattr(file_obj.create_time, "timestamp"):
            created_at = int(file_obj.create_time.timestamp())

        return ChimericFileUploadResponse(
            native=file_obj,
            common=FileUploadResponse(
                file_id=file_id,
                filename=filename,
                bytes=file_size,
                created_at=created_at,
                metadata=file_obj.model_dump(),
            ),
        )
