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
    ToolCall,
    ToolCallChunk,
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

    def _list_models_impl(self) -> list[ModelSummary]:
        """Lists available models from the OpenAI API.

        Returns:
            A list of ModelSummary objects for all available models from the API.
        """
        return [
            ModelSummary(
                id=model.id, name=model.id, owned_by=model.owned_by, created_at=model.created
            )
            for model in self.client.models.list()
        ]

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

    # ====================================================================
    # Unified tool execution methods
    # ====================================================================

    def _execute_tool_call(self, call: ToolCall) -> dict[str, Any]:
        """Executes a single tool call and returns metadata.

        Args:
            call: The ToolCall object containing call information.

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
        if not callable(tool.function):
            raise ToolRegistrationError(f"Tool '{tool_name}' is not callable.")

        try:
            args = json.loads(call.arguments)
            result = tool.function(**args)
            tool_call_info["result"] = str(result)
        except Exception as e:
            tool_call_info["error"] = str(e)

        return tool_call_info

    def _execute_accumulated_tool_calls(
        self, tool_calls: dict[str, ToolCallChunk]
    ) -> list[dict[str, Any]]:
        """Executes all accumulated tool calls and returns their results.

        Args:
            tool_calls: Dictionary of accumulated tool calls.

        Returns:
            List of tool call results with metadata.
        """
        results = []
        for tool_call in tool_calls.values():
            if tool_call.status == "completed":
                # Create a proper ToolCall object for execution
                tool_call_obj = ToolCall(
                    call_id=tool_call.call_id or tool_call.id,
                    name=tool_call.name,
                    arguments=tool_call.arguments,
                )
                result = self._execute_tool_call(tool_call_obj)
                results.append(result)
        return results

    @staticmethod
    def _update_messages_with_tool_results(
        messages: Input, tool_results: list[dict[str, Any]]
    ) -> list[Any]:
        """Updates message history with tool call results.

        Args:
            messages: Original message history.
            tool_results: List of tool call results.

        Returns:
            Updated messages with tool calls and results.
        """
        # Ensure messages is a mutable list
        messages_list: list[Any] = list(messages) if isinstance(messages, list) else [messages]

        for tool_result in tool_results:
            # Add the function call message
            messages_list.append(
                {
                    "type": "function_call",
                    "call_id": tool_result["call_id"],
                    "name": tool_result["name"],
                    "arguments": tool_result["arguments"],
                }
            )
            # Add the function result message
            messages_list.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_result["call_id"],
                    "output": tool_result.get("result", tool_result.get("error", "")),
                }
            )

        return messages_list

    def _handle_tool_calls_in_response(
        self, response: Response, messages: Input
    ) -> tuple[list[dict[str, Any]], Input]:
        """Processes tool calls from a non-streaming response.

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

        # Convert to ToolCall objects and execute
        tool_calls_metadata = []
        for call in calls:
            tool_call_obj = ToolCall(call_id=call.call_id, name=call.name, arguments=call.arguments)
            result = self._execute_tool_call(tool_call_obj)
            tool_calls_metadata.append(result)

        # Update message history
        updated_messages = self._update_messages_with_tool_results(messages, tool_calls_metadata)

        return tool_calls_metadata, updated_messages

    # ====================================================================
    # Streaming event processing
    # ====================================================================

    @staticmethod
    def _process_event(
        event: ResponseStreamEvent,
        accumulated: str,
        tool_calls: dict[str, ToolCallChunk] | None = None,
    ) -> tuple[str, dict[str, ToolCallChunk], ChimericStreamChunk[ResponseStreamEvent] | None]:
        """Processes a single event from an OpenAI response stream.

        Args:
            event: The response stream event from the OpenAI API.
            accumulated: The accumulated content from previous events.
            tool_calls: Dictionary of accumulated tool calls indexed by ID.

        Returns:
            A tuple containing the updated accumulated content, updated tool calls,
            and an optional ChimericStreamChunk to be yielded.
        """
        if tool_calls is None:
            tool_calls = {}

        event_type = getattr(event, "type", None)

        # Handle text content deltas
        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "") or ""
            accumulated += delta
            return (
                accumulated,
                tool_calls,
                ChimericStreamChunk(
                    native=event,
                    common=StreamChunk(
                        content=accumulated,
                        delta=delta,
                        metadata=event.model_dump(),
                    ),
                ),
            )

        # Handle tool call events (processed internally, no chunks returned)
        if event_type == "response.output_item.added":
            item = getattr(event, "item", None)
            if item and hasattr(item, "type") and item.type == "function_call":
                tool_call_id = getattr(item, "id", None)
                if tool_call_id:
                    tool_calls[tool_call_id] = ToolCallChunk(
                        id=tool_call_id,
                        call_id=getattr(item, "call_id", None),
                        name=getattr(item, "name", ""),
                        arguments="",
                        status="started",
                    )
                    return accumulated, tool_calls, None

        if event_type == "response.function_call_arguments.delta":
            tool_call_id = getattr(event, "item_id", None)
            delta = getattr(event, "delta", "") or ""
            if tool_call_id and tool_call_id in tool_calls:
                tool_calls[tool_call_id].arguments += delta
                tool_calls[tool_call_id].arguments_delta = delta
                tool_calls[tool_call_id].status = "arguments_streaming"
                return accumulated, tool_calls, None

        if event_type == "response.function_call_arguments.done":
            tool_call_id = getattr(event, "item_id", None)
            if tool_call_id and tool_call_id in tool_calls:
                tool_calls[tool_call_id].status = "completed"
                tool_calls[tool_call_id].arguments_delta = None
                return accumulated, tool_calls, None

        # Handle response completion
        if event_type == "response.completed":
            response = getattr(event, "response", None)
            outputs = getattr(response, "output", []) or []
            first_output = outputs[0] if outputs else None
            contents = getattr(first_output, "content", []) or []
            # Fallback to accumulated content if the final response is empty
            final_content = getattr(contents[0], "text", "") if contents else accumulated
            finish_reason = getattr(response, "status", None)

            return (
                accumulated,
                tool_calls,
                ChimericStreamChunk(
                    native=event,
                    common=StreamChunk(
                        content=final_content,
                        finish_reason=finish_reason,
                        metadata=event.model_dump(),
                    ),
                ),
            )

        return accumulated, tool_calls, None

    def _stream(
        self, stream: Stream[ResponseStreamEvent]
    ) -> Generator[ChimericStreamChunk[ResponseStreamEvent], None, None]:
        """Process a synchronous stream of events into ChimericStreamChunk objects.

        Args:
            stream: The synchronous stream of events.

        Yields:
            ChimericStreamChunk objects for valid events.
        """
        accumulated = ""
        tool_calls = {}

        for event in stream:
            accumulated, tool_calls, chunk = self._process_event(event, accumulated, tool_calls)
            if chunk:
                yield chunk

    async def _astream(
        self, stream: AsyncStream[ResponseStreamEvent]
    ) -> AsyncGenerator[ChimericStreamChunk[ResponseStreamEvent], None]:
        """Process an asynchronous stream of events into ChimericStreamChunk objects.

        Args:
            stream: The asynchronous stream of events.

        Yields:
            ChimericStreamChunk objects for valid events.
        """
        accumulated = ""
        tool_calls = {}

        async for event in stream:
            accumulated, tool_calls, chunk = self._process_event(event, accumulated, tool_calls)
            if chunk:
                yield chunk

    def _process_stream_with_tools_sync(
        self,
        stream: Stream[ResponseStreamEvent],
        original_messages: Input,
        original_model: str,
        original_tools: Tools | None = None,
        **original_kwargs: Any,
    ) -> Generator[ChimericStreamChunk[ResponseStreamEvent], None, None]:
        """Process a synchronous stream with automatic tool execution.

        Args:
            stream: The synchronous stream object.
            original_messages: Original messages for tool call continuation.
            original_model: Original model for tool call continuation.
            original_tools: Original tools for tool call continuation.
            **original_kwargs: Original kwargs for tool call continuation.

        Yields:
            ChimericStreamChunk containing the processed event data.
        """
        accumulated = ""
        tool_calls = {}

        for event in stream:
            accumulated, tool_calls, chunk = self._process_event(event, accumulated, tool_calls)
            if chunk:
                # Check if we need to execute tools and continue
                if (
                    chunk.common.finish_reason
                    and tool_calls
                    and original_messages
                    and original_model
                ):
                    yield from self._handle_tool_execution_and_continue_sync(
                        tool_calls,
                        original_messages,
                        original_model,
                        original_tools,
                        **original_kwargs,
                    )
                    return
                yield chunk

    async def _process_stream_with_tools_async(
        self,
        stream: AsyncStream[ResponseStreamEvent],
        original_messages: Input,
        original_model: str,
        original_tools: Tools | None = None,
        **original_kwargs: Any,
    ) -> AsyncGenerator[ChimericStreamChunk[ResponseStreamEvent], None]:
        """Process an asynchronous stream with automatic tool execution.

        Args:
            stream: The asynchronous stream object.
            original_messages: Original messages for tool call continuation.
            original_model: Original model for tool call continuation.
            original_tools: Original tools for tool call continuation.
            **original_kwargs: Original kwargs for tool call continuation.

        Yields:
            ChimericStreamChunk containing the processed event data.
        """
        accumulated = ""
        tool_calls = {}

        async for event in stream:
            accumulated, tool_calls, chunk = self._process_event(event, accumulated, tool_calls)
            if chunk:
                # Check if we need to execute tools and continue
                if (
                    chunk.common.finish_reason
                    and tool_calls
                    and original_messages
                    and original_model
                ):
                    async for tool_chunk in self._handle_tool_execution_and_continue_async(
                        tool_calls,
                        original_messages,
                        original_model,
                        original_tools,
                        **original_kwargs,
                    ):
                        yield tool_chunk
                    return
                yield chunk

    def _handle_tool_execution_and_continue_sync(
        self,
        tool_calls: dict[str, ToolCallChunk],
        messages: Input,
        model: str,
        tools: Tools | None,
        **kwargs: Any,
    ) -> Generator[ChimericStreamChunk[ResponseStreamEvent], None, None]:
        """Execute tools and continue streaming synchronously."""
        # Execute completed tool calls
        completed_tool_calls = self._execute_accumulated_tool_calls(tool_calls)
        if completed_tool_calls:
            # Update messages with tool results
            updated_messages = self._update_messages_with_tool_results(
                messages, completed_tool_calls
            )
            # Create a new stream for the continuation
            tools_param = tools if tools is not None else NOT_GIVEN
            continuation_stream = self._client.responses.create(
                model=model, input=updated_messages, tools=tools_param, stream=True, **kwargs
            )
            # Yield from the continuation stream
            yield from self._process_stream_with_tools_sync(
                continuation_stream,
                original_messages=messages,
                original_model=model,
                original_tools=tools,
                **kwargs,
            )

    async def _handle_tool_execution_and_continue_async(
        self,
        tool_calls: dict[str, ToolCallChunk],
        messages: Input,
        model: str,
        tools: Tools | None,
        **kwargs: Any,
    ) -> AsyncGenerator[ChimericStreamChunk[ResponseStreamEvent], None]:
        """Execute tools and continue streaming asynchronously."""
        # Execute completed tool calls
        completed_tool_calls = self._execute_accumulated_tool_calls(tool_calls)
        if completed_tool_calls:
            # Update messages with tool results
            updated_messages = self._update_messages_with_tool_results(
                messages, completed_tool_calls
            )
            # Create a new stream for the continuation
            tools_param = tools if tools is not None else NOT_GIVEN
            continuation_stream = await self._async_client.responses.create(
                model=model, input=updated_messages, tools=tools_param, stream=True, **kwargs
            )
            # Yield from the continuation stream
            async for chunk in self._process_stream_with_tools_async(
                continuation_stream,
                original_messages=messages,
                original_model=model,
                original_tools=tools,
                **kwargs,
            ):
                yield chunk

    # ====================================================================
    # Main completion methods
    # ====================================================================

    def _chat_completion_impl(
        self,
        messages: Input,
        model: str,
        stream: bool = False,
        tools: Tools = None,
        **kwargs: Any,
    ) -> (
        ChimericCompletionResponse[Response]
        | Generator[ChimericStreamChunk[ResponseStreamEvent], None, None]
    ):
        """Sends a synchronous chat completion request to the OpenAI API."""
        filtered_kwargs = self._filter_kwargs(self._client.responses.create, kwargs)
        tools_param = NOT_GIVEN if tools is None else tools

        # Send the initial user request
        response = self._client.responses.create(
            model=model, stream=stream, input=messages, tools=tools_param, **filtered_kwargs
        )

        if isinstance(response, Stream):
            # Handle streaming response with automatic tool execution
            if tools:
                return self._process_stream_with_tools_sync(
                    response,
                    original_messages=messages,
                    original_model=model,
                    original_tools=tools,
                    **filtered_kwargs,
                )
            return self._stream(response)

        # Handle non-streaming response
        tool_calls_metadata, updated_messages = self._handle_tool_calls_in_response(
            response, messages
        )

        # If tool calls were made, send their results back to the model
        if tool_calls_metadata:
            response = self._client.responses.create(
                model=model, input=updated_messages, tools=tools_param, **filtered_kwargs
            )

        return self._create_chimeric_response(response, tool_calls_metadata)

    async def _achat_completion_impl(
        self,
        messages: Input,
        model: str,
        stream: bool = False,
        tools: Tools = None,
        **kwargs: Any,
    ) -> (
        ChimericCompletionResponse[Response]
        | AsyncGenerator[ChimericStreamChunk[ResponseStreamEvent], None]
    ):
        """Sends an asynchronous chat completion request to the OpenAI API."""
        filtered_kwargs = self._filter_kwargs(self._async_client.responses.create, kwargs)
        tools_param = NOT_GIVEN if tools is None else tools

        # Send the initial user request
        response = await self._async_client.responses.create(
            model=model, input=messages, stream=stream, tools=tools_param, **filtered_kwargs
        )

        if isinstance(response, AsyncStream):
            # Handle streaming response with automatic tool execution
            if tools:
                return self._process_stream_with_tools_async(
                    response,
                    original_messages=messages,
                    original_model=model,
                    original_tools=tools,
                    **filtered_kwargs,
                )
            return self._astream(response)

        # Handle non-streaming response
        tool_calls_metadata, updated_messages = self._handle_tool_calls_in_response(
            response, messages
        )

        # If tool calls were made, send their results back to the model
        if tool_calls_metadata:
            response = await self._async_client.responses.create(
                model=model, input=updated_messages, tools=tools_param, **filtered_kwargs
            )

        return self._create_chimeric_response(response, tool_calls_metadata)

    # ====================================================================
    # Helper methods
    # ====================================================================

    @staticmethod
    def _create_chimeric_response(
        response: Response, tool_calls: list[dict[str, Any]]
    ) -> ChimericCompletionResponse[Response]:
        """Creates a ChimericCompletionResponse from a native OpenAI Response."""
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

    def _upload_file(self, **kwargs: Any) -> ChimericFileUploadResponse[FileObject]:
        """Uploads a file to OpenAI."""
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
