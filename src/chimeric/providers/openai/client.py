from collections.abc import AsyncGenerator, Generator
import json
from typing import Any

from openai import NOT_GIVEN, AsyncOpenAI, AsyncStream, OpenAI, Stream
from openai.types import FileObject
from openai.types.responses import Response, ResponseFunctionToolCall, ResponseStreamEvent

from chimeric.base import BaseClient
from chimeric.exceptions import ToolRegistrationError
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


class OpenAIClient(BaseClient[OpenAI, AsyncOpenAI, Response, ResponseStreamEvent, FileObject]):
    """OpenAI Client for interacting with the OpenAI API.

    This client provides a unified interface for synchronous and asynchronous
    interactions with OpenAI's API via the `openai` library. It returns `chimeric`
    response objects that wrap the native OpenAI responses.
    """

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        """Initializes the OpenAI client.

        Args:
            api_key: The OpenAI API key for authentication.
            **kwargs: Additional keyword arguments to pass to the OpenAI client constructor.
        """
        self._provider_name = "OpenAI"
        super().__init__(api_key=api_key, **kwargs)

    def _get_generic_types(self) -> dict[str, type]:
        """Returns the concrete OpenAI client types for `kwargs` filtering.

        Returns:
            A dictionary mapping client type names to their respective classes.
        """
        return {
            "sync": OpenAI,
            "async": AsyncOpenAI,
        }

    def _init_client(self, client_type: type, **kwargs: Any) -> OpenAI:
        """Initializes the synchronous OpenAI client.

        Args:
            client_type: The OpenAI client class.
            **kwargs: Additional arguments for the client constructor.

        Returns:
            An instance of the synchronous OpenAI client.
        """
        return OpenAI(api_key=self.api_key, **kwargs)

    def _init_async_client(self, async_client_type: type, **kwargs: Any) -> AsyncOpenAI:
        """Initializes the asynchronous OpenAI client.

        Args:
            async_client_type: The AsyncOpenAI client class.
            **kwargs: Additional arguments for the client constructor.

        Returns:
            An instance of the asynchronous OpenAI client.
        """
        return AsyncOpenAI(api_key=self.api_key, **kwargs)

    def _get_capabilities(self) -> Capability:
        """Gets OpenAI provider capabilities.

        Returns:
            A Capability object indicating all supported features.
        """
        return Capability(multimodal=True, streaming=True, tools=True, agents=True, files=True)

    def list_models(self) -> list[ModelSummary]:
        """Lists available models from OpenAI.

        Returns:
            A list of ModelSummary objects for all available models.
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
        """Processes a single event from an OpenAI response stream.

        Args:
            event: The response stream event from the OpenAI API.
            accumulated: The accumulated content from previous events.

        Returns:
            A tuple containing the updated accumulated content and an optional
            ChimericStreamChunk to be yielded.
        """
        event_type = getattr(event, "type", None)

        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "") or ""
            accumulated += delta
            return accumulated, ChimericStreamChunk(
                native=event,
                common=StreamChunk(
                    content=accumulated,
                    delta=delta,
                    metadata=event.model_dump(),
                ),
            )

        if event_type == "response.completed":
            response = getattr(event, "response", None)
            outputs = getattr(response, "output", []) or []
            first_output = outputs[0] if outputs else None
            contents = getattr(first_output, "content", []) or []
            # Fallback to accumulated content if the final response is empty
            final_content = getattr(contents[0], "text", "") if contents else accumulated
            finish_reason = getattr(response, "status", None)

            return accumulated, ChimericStreamChunk(
                native=event,
                common=StreamChunk(
                    content=final_content,
                    finish_reason=finish_reason,
                    metadata=event.model_dump(),
                ),
            )

        return accumulated, None

    def _stream(
        self,
        stream: Stream[ResponseStreamEvent],
    ) -> Generator[ChimericStreamChunk[ResponseStreamEvent], None, None]:
        """Yields processed chunks from a synchronous OpenAI stream.

        Args:
            stream: The synchronous stream of response events from `client.responses.create`.

        Yields:
            ChimericStreamChunk containing the processed event data.
        """
        accumulated = ""
        for event in stream:
            accumulated, chunk = self._process_event(event, accumulated)
            if chunk:
                yield chunk

    async def _astream(
        self,
        stream: AsyncStream[ResponseStreamEvent],
    ) -> AsyncGenerator[ChimericStreamChunk[ResponseStreamEvent], None]:
        """Yields processed chunks from an asynchronous OpenAI stream.

        Args:
            stream: The asynchronous stream of response events from `async_client.responses.create`.

        Yields:
            ChimericStreamChunk containing the processed event data.
        """
        accumulated = ""
        async for event in stream:
            accumulated, chunk = self._process_event(event, accumulated)
            if chunk:
                yield chunk

    @staticmethod
    def _create_chimeric_response(
        response: Response, tool_calls: list[dict[str, Any]]
    ) -> ChimericCompletionResponse[Response]:
        """Creates a ChimericCompletionResponse from a native OpenAI Response.

        Args:
            response: The OpenAI response object from `client.responses.create`.
            tool_calls: A list of tool calls made and executed during the request.

        Returns:
            A ChimericCompletionResponse wrapping the native response.
        """
        metadata = response.model_dump()
        if tool_calls:
            metadata["tool_calls"] = tool_calls

        # The response object can have content in different attributes depending
        # on the response type (e.g., text vs. function call).
        content = response.output_text or response.output or getattr(response.response, "text", "")

        return ChimericCompletionResponse(
            native=response,
            common=CompletionResponse(
                content=str(content),
                usage=Usage(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.total_tokens,
                ),
                model=response.model,
                metadata=metadata,
            ),
        )

    def _format_response(
        self, response: Response | Stream[ResponseStreamEvent], tool_calls: list[dict[str, Any]]
    ) -> (
        ChimericCompletionResponse[Response]
        | Generator[ChimericStreamChunk[ResponseStreamEvent], None, None]
    ):
        """Formats a synchronous OpenAI response into a standardized `chimeric` type.

        Args:
            response: The OpenAI response object or stream.
            tool_calls: A list of tool calls made during the request.

        Returns:
            A ChimericCompletionResponse for a single response, or a generator
            of ChimericStreamChunk for a streaming response.
        """
        if isinstance(response, Stream):
            return self._stream(response)
        return self._create_chimeric_response(response, tool_calls)

    async def _aformat_response(
        self,
        response: Response | AsyncStream[ResponseStreamEvent],
        tool_calls: list[dict[str, Any]],
    ) -> (
        ChimericCompletionResponse[Response]
        | AsyncGenerator[ChimericStreamChunk[ResponseStreamEvent], None]
    ):
        """Formats an asynchronous OpenAI response into a standardized `chimeric` type.

        Args:
            response: The OpenAI response object or async stream.
            tool_calls: A list of tool calls made during the request.

        Returns:
            A ChimericCompletionResponse for a single response, or an async
            generator of ChimericStreamChunk for a streaming response.
        """
        if isinstance(response, AsyncStream):
            return self._astream(response)
        return self._create_chimeric_response(response, tool_calls)

    def _encode_tools(self, tools: Tools = None) -> Tools:
        """Encodes a list of Tool objects into the format expected by the OpenAI API.

        Args:
            tools: A list of Tool objects or dictionaries.

        Returns:
            A list of tool dictionaries formatted for the OpenAI API, or None.
        """
        if not tools:
            return None

        encoded_tools = []
        for tool in tools:
            if isinstance(tool, Tool):
                encoded_tools.append(
                    {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters.model_dump() if tool.parameters else None,
                    }
                )
            else:
                encoded_tools.append(tool)  # Assumes tool is already a dict
        return encoded_tools

    def _process_function_call(self, call: Any) -> dict[str, Any]:
        """Executes a function call from the model and returns metadata.

        Args:
            call: The tool call object from the OpenAI response.

        Returns:
            A dictionary containing the tool call ID, name, arguments, and result.

        Raises:
            ToolRegistrationError: If the requested tool is not registered or not callable.
        """
        tool_name = call.name
        tool_call_info = {
            "call_id": call.call_id,
            "name": tool_name,
            "arguments": call.arguments,
        }

        tool = self.tool_manager.get_tool(tool_name)
        if not tool or not tool.function:
            raise ToolRegistrationError(
                tool_name=tool_name,
                reason=f"Tool '{tool_name}' is not registered or has no callable function.",
            )

        args = json.loads(call.arguments)
        result = tool.function(**args)
        tool_call_info["result"] = str(result)
        return tool_call_info

    def _handle_function_tool_calls(
        self, response: Response, messages: Input
    ) -> tuple[list[dict[str, Any]], Input]:
        """Processes tool calls from a response and updates the message history.

        If the response contains tool calls, this method executes them, appends the
        call and result to the message list, and returns metadata about the calls.

        Args:
            response: The OpenAI response containing potential tool calls.
            messages: The current list of messages to append to.

        Returns:
            A tuple containing (tool_calls_metadata, updated_messages).
        """
        calls = [
            output
            for output in getattr(response, "output", [])
            if isinstance(output, ResponseFunctionToolCall)
        ]
        if not calls:
            return [], messages

        # Ensure messages is a mutable list of dictionaries for appending
        messages_list: list[Any] = (
            list(messages) if isinstance(messages, list) else [messages]
        )
        tool_calls_metadata = []

        for call in calls:
            tool_call_info = self._process_function_call(call)
            tool_calls_metadata.append(tool_call_info)

            # Append the function call and its output to the message history
            # for the next turn in the conversation.
            messages_list.append(
                {
                    "type": "function_call",
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
            )
            messages_list.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": tool_call_info["result"],
                }
            )

        return tool_calls_metadata, messages_list

    def _chat_completion_impl(
        self,
        messages: Input,
        model: str,
        tools: Tools = None,
        **kwargs: Any,
    ) -> (
        ChimericCompletionResponse[Response]
        | Generator[ChimericStreamChunk[ResponseStreamEvent], None, None]
    ):
        """Sends a synchronous chat completion request to the OpenAI API.

        This method supports a two-pass mechanism for tool calls. If the model
        responds with a tool call, the tool is executed and a second request
        is made with the tool's result.

        Args:
            messages: The input messages for the chat completion.
            model: The model to use for the completion.
            tools: A list of tools (already encoded for the API).
            **kwargs: Additional arguments for the `client.responses.create` method.

        Returns:
            A ChimericCompletionResponse for a single response, or a generator
            of ChimericStreamChunk for a streaming response.
        """
        filtered_kwargs = self._filter_kwargs(self._client.responses.create, kwargs)
        tools_param = NOT_GIVEN if tools is None else tools

        # Send the initial user request.
        response = self._client.responses.create(
            model=model, input=messages, tools=tools_param, **filtered_kwargs
        )

        # Check for and handle any tool calls requested by the model.
        tool_calls_metadata, updated_messages = self._handle_function_tool_calls(response, messages)

        # If tool calls were made, send their results back to the model.
        if tool_calls_metadata:
            response = self._client.responses.create(
                model=model, input=updated_messages, tools=tools_param, **filtered_kwargs
            )

        return self._format_response(response, tool_calls_metadata)

    async def _achat_completion_impl(
        self,
        messages: Input,
        model: str,
        tools: Tools = None,
        **kwargs: Any,
    ) -> (
        ChimericCompletionResponse[Response]
        | AsyncGenerator[ChimericStreamChunk[ResponseStreamEvent], None]
    ):
        """Sends an asynchronous chat completion request to the OpenAI API.

        This method supports a two-pass mechanism for tool calls. If the model
        responds with a tool call, the tool is executed and a second request
        is made with the tool's result.

        Args:
            messages: The input messages for the chat completion.
            model: The model to use for the completion.
            tools: A list of tools (already encoded for the API).
            **kwargs: Additional arguments for the `async_client.responses.create` method.

        Returns:
            A ChimericCompletionResponse for a single response, or an async
            generator of ChimericStreamChunk for a streaming response.
        """
        filtered_kwargs = self._filter_kwargs(self._async_client.responses.create, kwargs)
        tools_param = NOT_GIVEN if tools is None else tools

        # Send the initial user request.
        response = await self._async_client.responses.create(
            model=model, input=messages, tools=tools_param, **filtered_kwargs
        )

        # Check for and handle any tool calls requested by the model.
        tool_calls_metadata, updated_messages = self._handle_function_tool_calls(response, messages)

        # If tool calls were made, send their results back to the model.
        if tool_calls_metadata:
            response = await self._async_client.responses.create(
                model=model, input=updated_messages, tools=tools_param, **filtered_kwargs
            )

        return await self._aformat_response(response, tool_calls_metadata)

    def _upload_file(self, **kwargs: Any) -> ChimericFileUploadResponse[FileObject]:
        """Uploads a file to OpenAI.

        Args:
            **kwargs: Provider-specific arguments for file upload, passed to `client.files.create`.

        Returns:
            A ChimericFileUploadResponse containing the native response and
            common file upload information.
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
