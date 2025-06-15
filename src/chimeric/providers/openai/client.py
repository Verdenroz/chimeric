from collections.abc import AsyncGenerator, Generator
from typing import Any

from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from openai.types import FileObject
from openai.types.responses import Response, ResponseStreamEvent

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
    Usage,
)


class OpenAIClient(BaseClient[OpenAI, AsyncOpenAI, Response, ResponseStreamEvent, FileObject]):
    """OpenAI Client for interacting with OpenAI services using the Responses API.

    Returns native OpenAI response types as thin wrappers around the actual API responses.
    """

    def __init__(
        self,
        api_key: str,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key.
            **kwargs: Additional OpenAI client options.
        """
        self._provider_name = "OpenAI"  # Add provider name for better error messages

        super().__init__(
            api_key=api_key,
            **kwargs,
        )

    def _get_generic_types(self) -> dict[str, type]:
        """Return the actual OpenAI client types for kwargs filtering."""
        return {
            "sync": OpenAI,
            "async": AsyncOpenAI,
        }

    def _init_client(self, client_type: type, **kwargs: Any) -> OpenAI:
        """Initialize the OpenAI client with filtered kwargs."""
        # At this point, kwargs are already filtered
        return OpenAI(api_key=self.api_key, **kwargs)

    def _init_async_client(self, async_client_type: type, **kwargs: Any) -> AsyncOpenAI:
        """Initialize the AsyncOpenAI client with filtered kwargs."""
        # At this point, kwargs are already filtered
        return AsyncOpenAI(api_key=self.api_key, **kwargs)

    def _get_capabilities(self) -> Capability:
        """Get OpenAI provider capabilities.

        Returns:
            Capability object indicating supported features.
        """
        return Capability(multimodal=True, streaming=True, tools=True, agents=True, files=True)

    def list_models(self) -> list[ModelSummary]:
        """List available OpenAI models.

        Returns:
            List of ModelSummary objects for available models.
        """
        return [
            ModelSummary(
                id=model.id, name=model.id, owned_by=model.owned_by, created_at=model.created
            )
            for model in self.client.models.list()
        ]

    @staticmethod
    def _process_event(
        event: ResponseStreamEvent,
        accumulated: str,
    ) -> tuple[str, ChimericStreamChunk[ResponseStreamEvent] | None]:
        """Process a single response event from OpenAI.

        Args:
            event: The response event to process.
            accumulated: The accumulated text from previous events.

        Returns:
            Tuple containing the updated accumulated text and a ChimericStreamChunk if applicable.
        """
        event_type = getattr(event, "type", None)

        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "") or ""
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

        if event_type == "response.completed":
            response = getattr(event, "response", None)
            outputs = getattr(response, "output", []) or []
            first = outputs[0] if outputs else None
            contents = getattr(first, "content", []) or []
            final_text = getattr(contents[0], "text", "") if contents else None
            finish_reason = getattr(response, "status", None)
            chunk = ChimericStreamChunk(
                native=event,
                common=StreamChunk(
                    content=final_text or accumulated,
                    finish_reason=finish_reason,
                    metadata=event.model_dump(),
                ),
            )
            return accumulated, chunk

        # other event types are ignored
        return accumulated, None

    def _stream(
        self,
        stream: Stream[ResponseStreamEvent],
    ) -> Generator[ChimericStreamChunk[ResponseStreamEvent], None, None]:
        """Stream response events from OpenAI (sync)."""
        accumulated = ""
        for event in stream:
            accumulated, chunk = self._process_event(event, accumulated)
            if chunk:
                yield chunk

    async def _astream(
        self,
        stream: AsyncStream[ResponseStreamEvent],
    ) -> AsyncGenerator[ChimericStreamChunk[ResponseStreamEvent], None]:
        """Stream response events from OpenAI (async)."""
        accumulated = ""
        async for event in stream:
            accumulated, chunk = self._process_event(event, accumulated)
            if chunk:
                yield chunk

    def _format_response(
        self, response: Response | Stream[ResponseStreamEvent]
    ) -> (
        ChimericCompletionResponse[Response]
        | Generator[ChimericStreamChunk[ResponseStreamEvent], None, None]
    ):
        """Format the synchronous OpenAI response into a standardized format."""
        if isinstance(response, Stream):
            return self._stream(response)

        return ChimericCompletionResponse(
            native=response,
            common=CompletionResponse(
                content=response.output_text or response.output or response.response.text,
                usage=Usage(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.total_tokens,
                ),
                model=response.model,
                metadata=response.model_dump(),
            ),
        )

    async def _aformat_response(
        self, response: Response | AsyncStream[ResponseStreamEvent]
    ) -> (
        ChimericCompletionResponse[Response]
        | AsyncGenerator[ChimericStreamChunk[ResponseStreamEvent], None]
    ):
        """Format the asynchronous OpenAI response into a standardized format."""
        if isinstance(response, AsyncStream):
            return self._astream(response)

        return ChimericCompletionResponse(
            native=response,
            common=CompletionResponse(
                content=response.output_text or response.output or response.response.text,
                usage=Usage(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.total_tokens,
                ),
                model=response.model,
                metadata=response.model_dump(),
            ),
        )

    def _chat_completion_impl(
        self,
        messages: Input,
        model: str,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> (
        ChimericCompletionResponse[Response]
        | Generator[ChimericStreamChunk[ResponseStreamEvent], None, None]
    ):
        """Implement synchronous chat completion using OpenAI Responses API."""
        filtered_kwargs = self._filter_kwargs(self._client.responses.create, kwargs)
        response = self._client.responses.create(model=model, input=messages, **filtered_kwargs)

        return self._format_response(response)

    async def _achat_completion_impl(
        self,
        messages: Input,
        model: str,
        tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> (
        ChimericCompletionResponse[Response]
        | AsyncGenerator[ChimericStreamChunk[ResponseStreamEvent], None]
    ):
        """Implement asynchronous chat completion using OpenAI Responses API."""
        filtered_kwargs = self._filter_kwargs(self._async_client.responses.create, kwargs)
        response = await self._async_client.responses.create(
            model=model, input=messages, **filtered_kwargs
        )

        return await self._aformat_response(response)

    def _upload_file(self, **kwargs: Any) -> ChimericFileUploadResponse[FileObject]:
        """Upload a file to OpenAI.

        Args:
            native: If True, returns the file object in its native format.
            **kwargs: Additional provider-specific arguments for file upload.

        Returns:
            Either a FileObject or ChimericFileUploadResponse depending on the native flag.
        """
        filtered_kwargs = self._filter_kwargs(self._client.files.create, kwargs)
        file_object = self._client.files.create(**filtered_kwargs)

        return ChimericFileUploadResponse(
            native=file_object,
            common=FileUploadResponse(
                file_id=file_object.id,
                filename=file_object.filename,
                bytes=file_object.bytes,
                purpose=file_object.purpose,
                created_at=file_object.created_at,
                status=file_object.status,
                metadata=file_object.model_dump(),
            ),
        )
