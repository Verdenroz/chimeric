from collections.abc import AsyncGenerator, Generator
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

    def list_models(self) -> list[ModelSummary]:
        """Lists available models from Anthropic.

        Returns:
            A list of ModelSummary objects for all available models.
        """
        models_response = self.client.models.list()
        return [
            ModelSummary(
                id=model.id,
                name=model.display_name or model.id,
                created_at=getattr(model, "created_at", None),
                metadata=model.model_dump() if hasattr(model, "model_dump") else {},
            )
            for model in models_response.data
        ]

    @staticmethod
    def _process_event(
        event: MessageStreamEvent,
        accumulated: str,
    ) -> tuple[str, ChimericStreamChunk[MessageStreamEvent] | None]:
        """Processes a single event from an Anthropic response stream.

        Args:
            event: The response stream event from the Anthropic API.
            accumulated: The accumulated content from previous events.

        Returns:
            A tuple containing the updated accumulated content and an optional
            ChimericStreamChunk to be yielded.
        """
        event_type = event.type

        if event_type == "content_block_delta":
            delta = event.delta.text if hasattr(event.delta, "text") else ""
            accumulated += delta
            return accumulated, ChimericStreamChunk(
                native=event,
                common=StreamChunk(
                    content=accumulated,
                    delta=delta,
                    metadata=event.model_dump(),
                ),
            )

        if event_type == "message_stop":
            return accumulated, ChimericStreamChunk(
                native=event,
                common=StreamChunk(
                    content=accumulated,
                    finish_reason="end_turn",
                    metadata=event.model_dump(),
                ),
            )

        return accumulated, None

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
        for event in stream:
            accumulated, chunk = self._process_event(event, accumulated)
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
        async for event in stream:
            accumulated, chunk = self._process_event(event, accumulated)
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
            tool_call_info["error"] = True

        return tool_call_info

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
