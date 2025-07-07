from collections.abc import AsyncGenerator, Generator
import json
from typing import Any

from anthropic import NOT_GIVEN, Anthropic, AsyncAnthropic, AsyncStream, Stream
from anthropic.types import Message, MessageStreamEvent
from anthropic.types.beta import FileMetadata

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


class AnthropicClient(
    BaseClient[Anthropic, AsyncAnthropic, Message, MessageStreamEvent, FileMetadata]
):
    """Anthropic Client for interacting with Claude models via the Anthropic API.

    This client provides a unified interface for synchronous and asynchronous
    interactions with Anthropic's API via the `anthropic` library. It returns `chimeric`
    response objects that wrap the native Anthropic responses.
    """

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        """Initializes the Anthropic client.

        Args:
            api_key: The Anthropic API key for authentication.
            **kwargs: Additional keyword arguments to pass to the Anthropic client constructor.
        """
        self._provider_name = "Anthropic"
        super().__init__(api_key=api_key, **kwargs)

    def _get_generic_types(self) -> dict[str, type]:
        """Returns the concrete Anthropic client types for `kwargs` filtering.

        Returns:
            A dictionary mapping client type names to their respective classes.
        """
        return {
            "sync": Anthropic,
            "async": AsyncAnthropic,
        }

    def _init_client(self, client_type: type, **kwargs: Any) -> Anthropic:
        """Initializes the synchronous Anthropic client.

        Args:
            client_type: The Anthropic client class.
            **kwargs: Additional arguments for the client constructor.

        Returns:
            An instance of the synchronous Anthropic client.
        """
        return Anthropic(api_key=self.api_key, **kwargs)

    def _init_async_client(self, async_client_type: type, **kwargs: Any) -> AsyncAnthropic:
        """Initializes the asynchronous Anthropic client.

        Args:
            async_client_type: The AsyncAnthropic client class.
            **kwargs: Additional arguments for the client constructor.

        Returns:
            An instance of the asynchronous Anthropic client.
        """
        return AsyncAnthropic(api_key=self.api_key, **kwargs)

    def _get_capabilities(self) -> Capability:
        """Gets Anthropic provider capabilities.

        Returns:
            A Capability object indicating all supported features.
        """
        return Capability(multimodal=True, streaming=True, tools=True, agents=False, files=True)

    def _get_model_aliases(self) -> list[str]:
        """Returns Anthropic model aliases.

        Returns:
            A list of alias model names.
        """
        return [
            # Claude 4 Models
            "claude-opus-4-0",
            "claude-sonnet-4-0",
            # Claude 3.7 Models
            "claude-3-7-sonnet-latest",
            # Claude 3.5 Models
            "claude-3-5-haiku-latest",
            "claude-3-5-sonnet-latest",
            # Claude 3 Models
            "claude-3-opus-latest",
        ]

    def _list_models_impl(self) -> list[ModelSummary]:
        """Lists available models from Anthropic API.

        Returns:
            A list of ModelSummary objects for all available models from the API.
        """
        models_response = self.client.models.list()
        return [
            ModelSummary(
                id=model.id,
                name=model.display_name,
                created_at=int(model.created_at.timestamp()),
                metadata=model.model_dump() if hasattr(model, "model_dump") else {},
            )
            for model in models_response.data
        ]

    @staticmethod
    def _process_event(
        event: MessageStreamEvent,
        accumulated: str,
        tool_calls: dict[str, ToolCallChunk] | None = None,
    ) -> tuple[str, dict[str, ToolCallChunk], ChimericStreamChunk[MessageStreamEvent] | None]:
        """Processes a single event from an Anthropic response stream.

        Args:
            event: The response stream event from the Anthropic API.
            accumulated: The accumulated content from previous events.
            tool_calls: Dictionary of accumulated tool calls indexed by ID.

        Returns:
            A tuple containing the updated accumulated content, updated tool calls,
            and an optional ChimericStreamChunk to be yielded.
        """
        if tool_calls is None:
            tool_calls = {}

        event_type = event.type

        # Handle text content deltas
        if event_type == "content_block_delta":
            if hasattr(event.delta, "text"):
                delta = event.delta.text
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
            # Handle tool use input JSON deltas
            if hasattr(event.delta, "partial_json"):
                # Get the content block index from the event
                block_index = getattr(event, "index", 0)
                tool_call_id = f"tool_call_{block_index}"

                if tool_call_id not in tool_calls:
                    tool_calls[tool_call_id] = ToolCallChunk(
                        id=tool_call_id,
                        call_id=tool_call_id,
                        name="",  # Will be set when we get content_block_start
                        arguments="",
                        status="started",
                    )

                # Accumulate the partial JSON
                tool_calls[tool_call_id].arguments += event.delta.partial_json
                tool_calls[tool_call_id].arguments_delta = event.delta.partial_json
                tool_calls[tool_call_id].status = "arguments_streaming"
                return accumulated, tool_calls, None

        # Handle content block start events
        elif event_type == "content_block_start":
            if (
                hasattr(event, "content_block")
                and getattr(event.content_block, "type", None) == "tool_use"
            ):
                block_index = getattr(event, "index", 0)
                tool_call_id = f"tool_call_{block_index}"

                tool_calls[tool_call_id] = ToolCallChunk(
                    id=tool_call_id,
                    call_id=getattr(event.content_block, "id", tool_call_id),
                    name=getattr(event.content_block, "name", ""),
                    arguments="",
                    status="started",
                )
                return accumulated, tool_calls, None

        # Handle content block stop events
        elif event_type == "content_block_stop":
            block_index = getattr(event, "index", 0)
            tool_call_id = f"tool_call_{block_index}"

            if (
                tool_call_id in tool_calls
                and tool_calls[tool_call_id].status == "arguments_streaming"
            ):
                tool_calls[tool_call_id].status = "completed"
                tool_calls[tool_call_id].arguments_delta = None
                return accumulated, tool_calls, None

        # Handle message stop events
        elif event_type == "message_stop":
            return (
                accumulated,
                tool_calls,
                ChimericStreamChunk(
                    native=event,
                    common=StreamChunk(
                        content=accumulated,
                        finish_reason="end_turn",
                        metadata=event.model_dump(),
                    ),
                ),
            )

        return accumulated, tool_calls, None

    def _stream(
        self,
        stream: Stream[MessageStreamEvent],
    ) -> Generator[ChimericStreamChunk[MessageStreamEvent], None, None]:
        """Yields processed chunks from a synchronous Anthropic stream.

        Args:
            stream: The synchronous stream of response events from `client.messages.create`.

        Yields:
            ChimericStreamChunk containing the processed event data.
        """
        accumulated = ""
        tool_calls = {}
        for event in stream:
            accumulated, tool_calls, chunk = self._process_event(event, accumulated, tool_calls)
            if chunk:
                yield chunk

    async def _astream(
        self,
        stream: AsyncStream[MessageStreamEvent],
    ) -> AsyncGenerator[ChimericStreamChunk[MessageStreamEvent], None]:
        """Yields processed chunks from an asynchronous Anthropic stream.

        Args:
            stream: The asynchronous stream of response events from `async_client.messages.create`.

        Yields:
            ChimericStreamChunk containing the processed event data.
        """
        accumulated = ""
        tool_calls = {}
        async for event in stream:
            accumulated, tool_calls, chunk = self._process_event(event, accumulated, tool_calls)
            if chunk:
                yield chunk

    @staticmethod
    def _create_chimeric_response(
        response: Message, tool_calls: list[dict[str, Any]]
    ) -> ChimericCompletionResponse[Message]:
        """Creates a ChimericCompletionResponse from a native Anthropic Message.

        Args:
            response: The Anthropic message object from `client.messages.create`.
            tool_calls: A list of tool calls made and executed during the request.

        Returns:
            A ChimericCompletionResponse wrapping the native response.
        """
        metadata = response.model_dump()
        if tool_calls:
            metadata["tool_calls"] = tool_calls

        # Extract content from the response
        content = "".join(
            content_block.text
            for content_block in response.content
            if hasattr(content_block, "text")
        )

        return ChimericCompletionResponse(
            native=response,
            common=CompletionResponse(
                content=content,
                usage=Usage(
                    prompt_tokens=response.usage.input_tokens if response.usage else 0,
                    completion_tokens=response.usage.output_tokens if response.usage else 0,
                    total_tokens=(
                        response.usage.input_tokens + response.usage.output_tokens
                        if response.usage
                        else 0
                    ),
                ),
                model=response.model,
                metadata=metadata,
            ),
        )

    def _encode_tools(self, tools: Tools = None) -> Tools:
        """Encodes a list of Tool objects into the format expected by the Anthropic API.

        Args:
            tools: A list of Tool objects or dictionaries.

        Returns:
            A list of tool dictionaries formatted for the Anthropic API, or None.
        """
        if not tools:
            return None

        encoded_tools = []
        for tool in tools:
            if isinstance(tool, Tool):
                encoded_tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.parameters.model_dump() if tool.parameters else {},
                    }
                )
            else:
                encoded_tools.append(tool)  # Assumes tool is already a dict
        return encoded_tools

    def _process_function_call(self, tool_use_block: Any) -> dict[str, Any]:
        """Executes a function call from the model and returns metadata.

        Args:
            tool_use_block: The tool use content block from the Anthropic response.

        Returns:
            A dictionary containing the tool call ID, name, arguments, and result.

        Raises:
            ToolRegistrationError: If the requested tool is not registered or not callable.
        """
        tool_name = tool_use_block.name
        tool_call_info = {
            "call_id": tool_use_block.id,
            "name": tool_name,
            "arguments": tool_use_block.input,
        }

        tool = self.tool_manager.get_tool(tool_name)
        if not callable(tool.function):
            raise ToolRegistrationError(
                f"Tool '{tool_name}' is not callable or not registered properly."
            )

        try:
            # Enhanced error handling for tool execution
            result = tool.function(**tool_use_block.input)
            tool_call_info["result"] = str(result)

        except Exception as e:
            # Handle tool execution errors gracefully
            error_msg = f"Error executing tool '{tool_name}': {e!s}"
            tool_call_info["result"] = error_msg
            tool_call_info["error"] = "true"

        return tool_call_info

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
            error_msg = f"Error executing tool '{tool_name}': {e!s}"
            tool_call_info["result"] = error_msg
            tool_call_info["error"] = "true"

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
            if tool_call.status == "completed" and tool_call.name:
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
        """Updates message history with tool call results in Anthropic format.

        Args:
            messages: Original message history.
            tool_results: List of tool call results.

        Returns:
            Updated messages with tool calls and results.
        """
        messages_list: list[Any] = list(messages) if isinstance(messages, list) else [messages]

        if tool_results:
            # Add the assistant message with tool uses
            assistant_content = []
            for tool_result in tool_results:
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": tool_result["call_id"],
                        "name": tool_result["name"],
                        "input": json.loads(tool_result["arguments"]),
                    }
                )

            messages_list.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                }
            )

            # Add tool result messages
            tool_results_content = []
            for tool_result in tool_results:
                tool_results_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_result["call_id"],
                        "content": tool_result.get("result", tool_result.get("error", "")),
                    }
                )

            messages_list.append(
                {
                    "role": "user",
                    "content": tool_results_content,
                }
            )

        return messages_list

    def _process_stream_with_tools_sync(
        self,
        stream: Stream[MessageStreamEvent],
        original_messages: Input,
        original_model: str,
        original_tools: Tools | None = None,
        **original_kwargs: Any,
    ) -> Generator[ChimericStreamChunk[MessageStreamEvent], None, None]:
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
        stream: AsyncStream[MessageStreamEvent],
        original_messages: Input,
        original_model: str,
        original_tools: Tools | None = None,
        **original_kwargs: Any,
    ) -> AsyncGenerator[ChimericStreamChunk[MessageStreamEvent], None]:
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
    ) -> Generator[ChimericStreamChunk[MessageStreamEvent], None, None]:
        """Execute tools and continue streaming synchronously."""
        completed_tool_calls = self._execute_accumulated_tool_calls(tool_calls)
        if completed_tool_calls:
            updated_messages = self._update_messages_with_tool_results(
                messages, completed_tool_calls
            )
            tools_param = tools if tools is not None else NOT_GIVEN
            # Ensure max_tokens is provided (required by Anthropic API)
            continuation_kwargs = kwargs.copy()
            if "max_tokens" not in continuation_kwargs:
                continuation_kwargs["max_tokens"] = 4096

            continuation_stream = self._client.messages.create(
                model=model,
                messages=updated_messages,
                tools=tools_param,
                stream=True,
                **continuation_kwargs,
            )
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
    ) -> AsyncGenerator[ChimericStreamChunk[MessageStreamEvent], None]:
        """Execute tools and continue streaming asynchronously."""
        completed_tool_calls = self._execute_accumulated_tool_calls(tool_calls)
        if completed_tool_calls:
            updated_messages = self._update_messages_with_tool_results(
                messages, completed_tool_calls
            )
            tools_param = tools if tools is not None else NOT_GIVEN
            # Ensure max_tokens is provided (required by Anthropic API)
            continuation_kwargs = kwargs.copy()
            if "max_tokens" not in continuation_kwargs:
                continuation_kwargs["max_tokens"] = 4096

            continuation_stream = await self._async_client.messages.create(
                model=model,
                messages=updated_messages,
                tools=tools_param,
                stream=True,
                **continuation_kwargs,
            )
            async for chunk in self._process_stream_with_tools_async(
                continuation_stream,
                original_messages=messages,
                original_model=model,
                original_tools=tools,
                **kwargs,
            ):
                yield chunk

    def _handle_function_tool_calls(
        self, response: Message, messages: Input
    ) -> tuple[list[dict[str, Any]], Input]:
        """Processes tool calls from a response and updates the message history.

        If the response contains tool calls, this method executes them, appends the
        call and result to the message list, and returns metadata about the calls.

        Args:
            response: The Anthropic response containing potential tool calls.
            messages: The current list of messages to append to.

        Returns:
            A tuple containing (tool_calls_metadata, updated_messages).
        """
        tool_use_blocks = [
            block for block in response.content if getattr(block, "type", None) == "tool_use"
        ]
        if not tool_use_blocks:
            return [], messages

        # Ensure messages is a mutable list
        messages_list = list(messages) if isinstance(messages, list) else [messages]
        tool_calls_metadata = []

        # Add the assistant's message with tool uses
        assistant_content = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block_type == "tool_use":
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        messages_list.append({"role": "assistant", "content": assistant_content})  # type: ignore[arg-type]

        # Process tool calls and add results
        tool_results = []
        for block in tool_use_blocks:
            tool_call_info = self._process_function_call(block)
            tool_calls_metadata.append(tool_call_info)

            content = tool_call_info["result"]
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                }
            )

        messages_list.append({"role": "user", "content": tool_results})  # type: ignore[arg-type]

        return tool_calls_metadata, messages_list

    @staticmethod
    def _format_messages(messages: Input) -> tuple[list[dict[str, Any]], str | None]:
        """Formats messages for the Anthropic API.

        Args:
            messages: Input messages to format.

        Returns:
            A tuple of (formatted_messages, system_prompt).
        """
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}], None

        if not hasattr(messages, "__iter__") or isinstance(messages, str):
            return [{"role": "user", "content": str(messages)}], None

        formatted_messages = []
        system_prompt = None

        for message in messages:
            if isinstance(message, dict):
                if message.get("role") == "system":
                    system_prompt = message.get("content", "")
                else:
                    formatted_messages.append(message)
            else:
                # Handle string messages
                formatted_messages.append({"role": "user", "content": str(message)})

        return formatted_messages, system_prompt

    def _create_params(
        self,
        messages: Input,
        model: str,
        stream: bool,
        tools: Tools,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Creates API parameters for the Anthropic messages.create call.

        Args:
            messages: Input messages.
            model: Model to use.
            stream: Whether to stream the response.
            tools: Tools to include.
            **kwargs: Additional API parameters.

        Returns:
            Dictionary of parameters for the API call.
        """
        formatted_messages, system_prompt = self._format_messages(messages)
        filtered_kwargs = self._filter_kwargs(self._client.messages.create, kwargs)

        params = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": filtered_kwargs.get("max_tokens", 4096),
            "stream": stream,
            "tools": tools if tools else NOT_GIVEN,
        }

        if system_prompt:
            params["system"] = system_prompt

        # Add optional parameters with validation
        optional_params = ["temperature", "top_p", "top_k", "stop_sequences", "metadata"]
        for key in optional_params:
            if key in filtered_kwargs:
                params[key] = filtered_kwargs[key]

        return params

    def _chat_completion_impl(
        self,
        messages: Input,
        model: str,
        stream: bool = False,
        tools: Tools = None,
        **kwargs: Any,
    ) -> (
        ChimericCompletionResponse[Message]
        | Generator[ChimericStreamChunk[MessageStreamEvent], None, None]
    ):
        """Sends a synchronous chat completion request to the Anthropic API.

        This method supports a two-pass mechanism for tool calls. If the model
        responds with a tool call, the tool is executed and a second request
        is made with the tool's result.

        Args:
            messages: The input messages for the chat completion.
            model: The model to use for the completion.
            stream: Whether to return a streaming response.
            tools: A list of tools (already encoded for the API).
            **kwargs: Additional arguments for the `client.messages.create` method.

        Returns:
            A ChimericCompletionResponse for a single response, or a generator
            of ChimericStreamChunk for a streaming response.
        """
        params = self._create_params(messages, model, stream, tools, **kwargs)
        response = self._client.messages.create(**params)

        if isinstance(response, Stream):
            # Handle streaming response with automatic tool execution
            if tools:
                return self._process_stream_with_tools_sync(
                    response,
                    original_messages=params["messages"],
                    original_model=model,
                    original_tools=tools,
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ["model", "messages", "tools", "stream"]
                    },
                )
            return self._stream(response)

        # Handle tool calls
        tool_calls_metadata, updated_messages = self._handle_function_tool_calls(
            response, params["messages"]
        )

        # If tool calls were made, send their results back to the model
        if tool_calls_metadata:
            final_params = params.copy()
            final_params["messages"] = updated_messages
            final_params["stream"] = False
            response = self._client.messages.create(**final_params)

        return self._create_chimeric_response(response, tool_calls_metadata)

    async def _achat_completion_impl(
        self,
        messages: Input,
        model: str,
        stream: bool = False,
        tools: Tools = None,
        **kwargs: Any,
    ) -> (
        ChimericCompletionResponse[Message]
        | AsyncGenerator[ChimericStreamChunk[MessageStreamEvent], None]
    ):
        """Sends an asynchronous chat completion request to the Anthropic API.

        This method supports a two-pass mechanism for tool calls. If the model
        responds with a tool call, the tool is executed and a second request
        is made with the tool's result.

        Args:
            messages: The input messages for the chat completion.
            model: The model to use for the completion.
            stream: Whether to return a streaming response.
            tools: A list of tools (already encoded for the API).
            **kwargs: Additional arguments for the `async_client.messages.create` method.

        Returns:
            A ChimericCompletionResponse for a single response, or an async
            generator of ChimericStreamChunk for a streaming response.
        """
        params = self._create_params(messages, model, stream, tools, **kwargs)
        response = await self._async_client.messages.create(**params)

        if isinstance(response, AsyncStream):
            # Handle streaming response with automatic tool execution
            if tools:
                return self._process_stream_with_tools_async(
                    response,
                    original_messages=params["messages"],
                    original_model=model,
                    original_tools=tools,
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ["model", "messages", "tools", "stream"]
                    },
                )
            return self._astream(response)

        # Handle tool calls
        tool_calls_metadata, updated_messages = self._handle_function_tool_calls(
            response, params["messages"]
        )

        # If tool calls were made, send their results back to the model
        if tool_calls_metadata:
            final_params = params.copy()
            final_params["messages"] = updated_messages
            final_params["stream"] = False
            response = await self._async_client.messages.create(**final_params)

        return self._create_chimeric_response(response, tool_calls_metadata)

    def _upload_file(self, **kwargs: Any) -> ChimericFileUploadResponse[FileMetadata]:
        """Uploads a file to Anthropic.

        Args:
            **kwargs: Provider-specific arguments for file upload, passed to `client.beta.files.upload`.

        Returns:
            A ChimericFileUploadResponse containing the native response and
            common file upload information.
        """
        if "file" not in kwargs:
            raise ValueError("'file' parameter is required for file upload")

        file_object = self._client.beta.files.upload(file=kwargs["file"])

        return ChimericFileUploadResponse(
            native=file_object,
            common=FileUploadResponse(
                file_id=file_object.id,
                filename=file_object.filename,
                bytes=file_object.size_bytes,
                created_at=getattr(file_object, "created_at", None),
                metadata=file_object.model_dump(),
            ),
        )
