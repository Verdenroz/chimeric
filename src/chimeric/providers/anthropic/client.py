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
    """Client for interacting with Anthropic's Claude models.

    This client provides a unified interface for synchronous and asynchronous
    interactions with the Anthropic API. It supports features like automatic
    tool execution, streaming, and file uploads.
    """

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        """Initializes the AnthropicClient.

        Args:
            api_key: The Anthropic API key for authentication.
            **kwargs: Additional arguments to pass to the Anthropic client.
        """
        self._provider_name = "Anthropic"
        super().__init__(api_key=api_key, **kwargs)

    def _get_generic_types(self) -> dict[str, type]:
        """Gets the concrete Anthropic client types for `kwargs` filtering.

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
        """Gets a list of Anthropic model aliases.

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
        """Lists available models from the Anthropic API.

        Returns:
            A list of ModelSummary objects for available models.
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

    # =====================================
    # Message and Tool Processing
    # =====================================

    @staticmethod
    def _format_messages(messages: Input) -> tuple[list[dict[str, Any]], str | None]:
        """Formats input messages for the Anthropic API.

        Args:
            messages: Input messages in various formats.

        Returns:
            A tuple containing the list of formatted messages and an optional
            system prompt.
        """
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}], None

        if not hasattr(messages, "__iter__"):
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
                formatted_messages.append({"role": "user", "content": str(message)})

        return formatted_messages, system_prompt

    def _encode_tools(self, tools: Tools = None) -> Tools:
        """Encodes tools into the format expected by the Anthropic API.

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
                encoded_tools.append(tool) # Assuming tool is already a dict

        return encoded_tools

    def _make_create_params(
        self, messages: Input, model: str, stream: bool, tools: Tools, **kwargs: Any
    ) -> dict[str, Any]:
        """Make API parameters for the Anthropic messages.create call.

        Args:
            messages: Input messages.
            model: Model identifier.
            stream: Whether to stream the response.
            tools: Tools to include in the request.
            **kwargs: Additional API parameters.

        Returns:
            A dictionary of parameters for the API call.
        """
        formatted_messages, system_prompt = self._format_messages(messages)
        filtered_kwargs = self._filter_kwargs(self._client.messages.create, kwargs)

        # Add tool usage instruction to system prompt if tools are provided
        if tools:
            tool_instruction = "You have access to tools. Always use the appropriate tools to perform calculations, look up information, or complete tasks rather than doing them manually. When a user asks for calculations, use the calculator tool. When they ask for information retrieval, use the appropriate search or data tools."
            system_prompt = (
                f"{system_prompt} {tool_instruction}" if system_prompt else tool_instruction
            )

        params = {
            "model": model,
            "messages": formatted_messages,
            "max_tokens": filtered_kwargs.get("max_tokens", 4096),
            "stream": stream,
            "tools": tools if tools else NOT_GIVEN,
        }

        if system_prompt:
            params["system"] = system_prompt

        # Add optional parameters
        for param in ["temperature", "top_p", "top_k", "stop_sequences", "metadata"]:
            if param in filtered_kwargs:
                params[param] = filtered_kwargs[param]

        return params

    # =====================================
    # Tool Execution Methods
    # =====================================

    @staticmethod
    def _parse_tool_arguments(arguments: str | dict[str, Any]) -> dict[str, Any]:
        """Parses tool arguments from string or dict format.

        Args:
            arguments: The tool arguments, which can be a JSON string or a dict.

        Returns:
            A dictionary of parsed arguments.
        """
        if isinstance(arguments, str):
            return json.loads(arguments.strip()) if arguments.strip() else {}
        return arguments if not isinstance(arguments, str) else {}

    def _execute_single_tool(self, call: ToolCall) -> dict[str, Any]:
        """Executes a single tool call and returns its metadata.

        Args:
            call: The tool call to execute.

        Returns:
            A dictionary containing call metadata and results.

        Raises:
            ToolRegistrationError: If the tool is not callable.
        """
        tool = self.tool_manager.get_tool(call.name)
        if not callable(tool.function):
            raise ToolRegistrationError(f"Tool '{call.name}' is not callable")

        tool_metadata = {
            "call_id": call.call_id,
            "name": call.name,
            "arguments": call.arguments,
        }

        try:
            args = self._parse_tool_arguments(call.arguments)
            result = tool.function(**args)
            tool_metadata["result"] = str(result)
        except Exception as e:
            tool_metadata["result"] = f"Error executing tool '{call.name}': {e!s}"
            tool_metadata["error"] = "true"

        return tool_metadata

    def _execute_tool_batch(self, tool_calls: dict[str, ToolCallChunk]) -> list[dict[str, Any]]:
        """Executes all completed tool calls in a batch.

        Args:
            tool_calls: A dictionary of accumulated tool calls.

        Returns:
            A list of tool execution results.
        """
        completed_calls = [
            call for call in tool_calls.values() if call.status == "completed" and call.name
        ]

        results = []
        for tool_call in completed_calls:
            call_obj = ToolCall(
                call_id=tool_call.call_id or tool_call.id,
                name=tool_call.name,
                arguments=tool_call.arguments,
            )
            result = self._execute_single_tool(call_obj)
            results.append(result)

        return results

    def _process_response_tools(
        self, response: Message, messages: Input
    ) -> tuple[list[dict[str, Any]], Input]:
        """Processes tool calls from a response and updates message history.

        Args:
            response: The Anthropic response containing tool calls.
            messages: The current message history.

        Returns:
            A tuple containing tool call metadata and the updated messages.
        """
        tool_use_blocks = [
            block for block in response.content if getattr(block, "type", None) == "tool_use"
        ]

        if not tool_use_blocks:
            return [], messages

        messages_list: list[Any] = list(messages) if isinstance(messages, list) else [messages]

        # Build assistant message with tool uses
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

        messages_list.append({"role": "assistant", "content": assistant_content})

        # Execute tools and format results
        tool_metadata = []
        tool_results = []

        for block in tool_use_blocks:
            # Execute tool - serialize arguments to string for ToolCall validation
            serialized_args = (
                json.dumps(block.input) if isinstance(block.input, dict) else str(block.input)
            )
            tool_call = ToolCall(
                call_id=block.id,
                name=block.name,
                arguments=serialized_args,
            )
            metadata = self._execute_single_tool(tool_call)
            tool_metadata.append(metadata)

            # Format result for API
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": metadata["result"],
                }
            )

        messages_list.append({"role": "user", "content": tool_results})
        return tool_metadata, messages_list

    # =====================================
    # Stream Processing Methods
    # =====================================

    @staticmethod
    def _process_content_delta(
        event: MessageStreamEvent, accumulated: str, tool_calls: dict[str, ToolCallChunk]
    ) -> tuple[str, dict[str, ToolCallChunk], ChimericStreamChunk[MessageStreamEvent] | None]:
        """Processes content delta events from the stream.

        Args:
            event: The message stream event.
            accumulated: The accumulated content string.
            tool_calls: The dictionary of in-progress tool calls.

        Returns:
            A tuple containing the updated accumulated content, tool calls,
            and an optional ChimericStreamChunk.
        """
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

        if hasattr(event.delta, "partial_json"):
            block_index = getattr(event, "index", 0)
            tool_call_id = f"tool_call_{block_index}"

            if tool_call_id not in tool_calls:
                tool_calls[tool_call_id] = ToolCallChunk(
                    id=tool_call_id,
                    call_id=tool_call_id,
                    name="",
                    arguments="",
                    status="started",
                )

            tool_calls[tool_call_id].arguments += event.delta.partial_json
            tool_calls[tool_call_id].arguments_delta = event.delta.partial_json
            tool_calls[tool_call_id].status = "arguments_streaming"

        return accumulated, tool_calls, None

    @staticmethod
    def _process_block_start(
        event: MessageStreamEvent, tool_calls: dict[str, ToolCallChunk]
    ) -> None:
        """Processes content block start events from the stream.

        This method identifies tool_use blocks and initializes a new
        ToolCallChunk.

        Args:
            event: The message stream event.
            tool_calls: The dictionary of in-progress tool calls.
        """
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

    @staticmethod
    def _process_block_stop(
        event: MessageStreamEvent, tool_calls: dict[str, ToolCallChunk]
    ) -> None:
        """Processes content block stop events from the stream.

        This method marks a tool call as completed when its block ends.

        Args:
            event: The message stream event.
            tool_calls: The dictionary of in-progress tool calls.
        """
        block_index = getattr(event, "index", 0)
        tool_call_id = f"tool_call_{block_index}"

        if tool_call_id in tool_calls and tool_calls[tool_call_id].status == "arguments_streaming":
            tool_calls[tool_call_id].status = "completed"
            tool_calls[tool_call_id].arguments_delta = None

    def _process_stream_event(
        self,
        event: MessageStreamEvent,
        accumulated: str,
        tool_calls: dict[str, ToolCallChunk] | None = None,
    ) -> tuple[str, dict[str, ToolCallChunk], ChimericStreamChunk[MessageStreamEvent] | None]:
        """Processes a single event from an Anthropic response stream.

        Args:
            event: The response stream event from the Anthropic API.
            accumulated: The accumulated content from previous events.
            tool_calls: A dictionary of accumulated tool calls indexed by ID.

        Returns:
            A tuple containing updated accumulated content, updated tool calls,
            and an optional ChimericStreamChunk to be yielded.
        """
        if tool_calls is None:
            tool_calls = {}

        event_type = event.type

        if event_type == "content_block_delta":
            return self._process_content_delta(event, accumulated, tool_calls)

        if event_type == "content_block_start":
            self._process_block_start(event, tool_calls)

        elif event_type == "content_block_stop":
            self._process_block_stop(event, tool_calls)

        elif event_type == "message_stop":
            return (
                accumulated,
                tool_calls,
                ChimericStreamChunk(
                    native=event,
                    common=StreamChunk(
                        content=accumulated,
                        finish_reason="end_turn",
                        metadata={
                            **event.model_dump(),
                            "tool_calls_found": len(tool_calls),
                            "completed_tool_calls": len(
                                [c for c in tool_calls.values() if c.status == "completed"]
                            ),
                        },
                    ),
                ),
            )

        return accumulated, tool_calls, None

    def _stream(
        self, stream: Stream[MessageStreamEvent]
    ) -> Generator[ChimericStreamChunk[MessageStreamEvent], None, None]:
        """Processes a synchronous Anthropic stream.

        Args:
            stream: The synchronous stream of MessageStreamEvent objects.

        Yields:
            ChimericStreamChunk objects.
        """
        accumulated = ""
        tool_calls = {}
        for event in stream:
            accumulated, tool_calls, chunk = self._process_stream_event(
                event, accumulated, tool_calls
            )
            if chunk:
                yield chunk

    async def _astream(
        self, stream: AsyncStream[MessageStreamEvent]
    ) -> AsyncGenerator[ChimericStreamChunk[MessageStreamEvent], None]:
        """Processes an asynchronous Anthropic stream.

        Args:
            stream: The asynchronous stream of MessageStreamEvent objects.

        Yields:
            ChimericStreamChunk objects.
        """
        accumulated = ""
        tool_calls = {}
        async for event in stream:
            accumulated, tool_calls, chunk = self._process_stream_event(
                event, accumulated, tool_calls
            )
            if chunk:
                yield chunk

    # =====================================
    # Response Creation Methods
    # =====================================

    @staticmethod
    def _generate_tool_summary(tool_calls: list[dict[str, Any]]) -> str:
        """Generates a summary for responses with only tool calls.

        Args:
            tool_calls: A list of executed tool call metadata dictionaries.

        Returns:
            A formatted string summarizing the tool execution results.
        """
        results = []
        for tool_call in tool_calls:
            name = tool_call.get("name", "unknown")
            args = tool_call.get("arguments", {})
            result = tool_call.get("result", "")

            if isinstance(args, dict) and args:
                arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
                results.append(f"{name}({arg_str}) → {result}")
            else:
                results.append(f"{name}() → {result}")

        return "Tool execution results:\n" + "\n".join(results) if results else ""

    def _create_chimeric_response(
        self, response: Message, tool_calls: list[dict[str, Any]]
    ) -> ChimericCompletionResponse[Message]:
        """Creates a ChimericCompletionResponse from a native Anthropic Message.

        Args:
            response: The Anthropic message object.
            tool_calls: A list of executed tool calls.

        Returns:
            A ChimericCompletionResponse wrapping the native response.
        """
        metadata = response.model_dump()
        if tool_calls:
            metadata["tool_calls"] = tool_calls

        # Extract text content
        content = "".join(block.text for block in response.content if hasattr(block, "text"))

        # Generate summary for tool-only responses
        if not content.strip() and tool_calls:
            content = self._generate_tool_summary(tool_calls)

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

    def _process_stream_with_tools_sync(
        self,
        stream: Stream[MessageStreamEvent],
        original_messages: Input,
        original_model: str,
        original_tools: Tools | None = None,
        **original_kwargs: Any,
    ) -> Generator[ChimericStreamChunk[MessageStreamEvent], None, None]:
        """Processes a synchronous stream with automatic tool execution.

        This method handles the streaming response, executes any tool calls
        that appear, and continues the chat completion with the tool results.

        Args:
            stream: The synchronous stream from the Anthropic API.
            original_messages: The initial list of messages.
            original_model: The model being used.
            original_tools: The list of available tools.
            **original_kwargs: Additional arguments for the API call.

        Yields:
            ChimericStreamChunk objects from the initial and subsequent streams.
        """
        accumulated = ""
        tool_calls = {}

        for event in stream:
            accumulated, tool_calls, chunk = self._process_stream_event(
                event, accumulated, tool_calls
            )
            if chunk:
                yield chunk
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

    async def _process_stream_with_tools_async(
        self,
        stream: AsyncStream[MessageStreamEvent],
        original_messages: Input,
        original_model: str,
        original_tools: Tools | None = None,
        **original_kwargs: Any,
    ) -> AsyncGenerator[ChimericStreamChunk[MessageStreamEvent], None]:
        """Processes an asynchronous stream with automatic tool execution.

        This method handles the async streaming response, executes any tool calls
        that appear, and continues the chat completion with the tool results.

        Args:
            stream: The asynchronous stream from the Anthropic API.
            original_messages: The initial list of messages.
            original_model: The model being used.
            original_tools: The list of available tools.
            **original_kwargs: Additional arguments for the API call.

        Yields:
            ChimericStreamChunk objects from the initial and subsequent streams.
        """
        accumulated = ""
        tool_calls = {}

        async for event in stream:
            accumulated, tool_calls, chunk = self._process_stream_event(
                event, accumulated, tool_calls
            )
            if chunk:
                yield chunk
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

    def _handle_tool_execution_and_continue_sync(
        self,
        tool_calls: dict[str, ToolCallChunk],
        messages: Input,
        model: str,
        tools: Tools | None,
        **kwargs: Any,
    ) -> Generator[ChimericStreamChunk[MessageStreamEvent], None, None]:
        """Executes tools and continues the conversation stream synchronously.

        Args:
            tool_calls: A dictionary of completed tool call chunks.
            messages: The current list of messages.
            model: The model to use for continuation.
            tools: The list of available tools.
            **kwargs: Additional arguments for the continuation API call.

        Yields:
            ChimericStreamChunk objects from the continuation stream.
        """
        completed_tool_calls = self._execute_tool_batch(tool_calls)
        if completed_tool_calls:
            updated_messages = AnthropicClient._update_messages_with_tool_results(
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
                original_messages=updated_messages,
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
        """Executes tools and continues the conversation stream asynchronously.

        Args:
            tool_calls: A dictionary of completed tool call chunks.
            messages: The current list of messages.
            model: The model to use for continuation.
            tools: The list of available tools.
            **kwargs: Additional arguments for the continuation API call.

        Yields:
            ChimericStreamChunk objects from the continuation stream.
        """
        completed_tool_calls = self._execute_tool_batch(tool_calls)
        if completed_tool_calls:
            updated_messages = AnthropicClient._update_messages_with_tool_results(
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
                original_messages=updated_messages,
                original_model=model,
                original_tools=tools,
                **kwargs,
            ):
                yield chunk

    @staticmethod
    def _update_messages_with_tool_results(
        messages: Input, tool_results: list[dict[str, Any]]
    ) -> list[Any]:
        """Updates the message history with tool call results.

        This method constructs the assistant's tool_use message and the
        user's tool_result message to continue the conversation.

        Args:
            messages: The current list of messages.
            tool_results: A list of dictionaries containing tool execution results.

        Returns:
            The updated list of messages.
        """
        if not tool_results:
            return list(messages) if isinstance(messages, list) else [messages]

        messages_list: list[Any] = list(messages) if isinstance(messages, list) else [messages]

        # Add assistant message with tool uses
        assistant_content = []
        for tool_result in tool_results:
            input_data = AnthropicClient._parse_tool_arguments(tool_result["arguments"])
            assistant_content.append(
                {
                    "type": "tool_use",
                    "id": tool_result["call_id"],
                    "name": tool_result["name"],
                    "input": input_data,
                }
            )

        messages_list.append({"role": "assistant", "content": assistant_content})

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

        messages_list.append({"role": "user", "content": tool_results_content})

        return messages_list

    def _handle_tool_execution_loop(
        self, response: Message, params: dict[str, Any]
    ) -> tuple[Message, list[dict[str, Any]]]:
        """Handles the tool execution loop for non-streaming responses.

        This method repeatedly executes tools and calls the API until the model
        provides a final response instead of more tool calls.

        Args:
            response: The initial response from the API.
            params: API parameters for continuation calls.

        Returns:
            A tuple containing the final API response and a list of all
            executed tool call metadata.
        """
        all_tool_calls = []
        current_messages = params["messages"]
        max_iterations = 5

        for _ in range(max_iterations):
            tool_calls, updated_messages = self._process_response_tools(response, current_messages)

            if not tool_calls:
                break

            all_tool_calls.extend(tool_calls)
            current_messages = updated_messages

            # Request final response
            current_messages.append(
                {
                    "role": "user",
                    "content": "Please provide your final response based on the tool results above.",
                }
            )

            final_params = params.copy()
            final_params.update({"messages": current_messages, "stream": False})
            response = self._client.messages.create(**final_params)

        return response, all_tool_calls

    async def _handle_async_tool_execution_loop(
        self, response: Message, params: dict[str, Any]
    ) -> tuple[Message, list[dict[str, Any]]]:
        """Handles the asynchronous tool execution loop for non-streaming responses.

        This method repeatedly executes tools and calls the API asynchronously
        until the model provides a final response.

        Args:
            response: The initial response from the API.
            params: API parameters for continuation calls.

        Returns:
            A tuple containing the final API response and a list of all
            executed tool call metadata.
        """
        all_tool_calls = []
        current_messages = params["messages"]
        max_iterations = 5

        for _ in range(max_iterations):
            tool_calls, updated_messages = self._process_response_tools(response, current_messages)

            if not tool_calls:
                break

            all_tool_calls.extend(tool_calls)
            current_messages = updated_messages

            current_messages.append(
                {
                    "role": "user",
                    "content": "Please provide your final response based on the tool results above.",
                }
            )

            final_params = params.copy()
            final_params.update({"messages": current_messages, "stream": False})
            response = await self._async_client.messages.create(**final_params)

        return response, all_tool_calls

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

        Args:
            messages: Input messages for the chat completion.
            model: The model to use for the completion.
            stream: Whether to return a streaming response.
            tools: A list of available tools.
            **kwargs: Additional arguments for the API call.

        Returns:
            A ChimericCompletionResponse for a single response or a generator
            for a streaming response.
        """
        params = self._make_create_params(messages, model, stream, tools, **kwargs)
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

        # Handle tool execution if tools are available
        if tools:
            response, all_tool_calls = self._handle_tool_execution_loop(response, params)
            return self._create_chimeric_response(response, all_tool_calls)

        return self._create_chimeric_response(response, [])

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

        Args:
            messages: Input messages for the chat completion.
            model: The model to use for the completion.
            stream: Whether to return a streaming response.
            tools: A list of available tools.
            **kwargs: Additional arguments for the API call.

        Returns:
            A ChimericCompletionResponse for a single response or an async
            generator for a streaming response.
        """
        params = self._make_create_params(messages, model, stream, tools, **kwargs)
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

        # Handle tool execution if tools are available
        if tools:
            response, all_tool_calls = await self._handle_async_tool_execution_loop(
                response, params
            )
            return self._create_chimeric_response(response, all_tool_calls)

        return self._create_chimeric_response(response, [])

    def _upload_file(self, **kwargs: Any) -> ChimericFileUploadResponse[FileMetadata]:
        """Uploads a file to Anthropic.

        Args:
            **kwargs: Provider-specific arguments for file upload. Must include
                the 'file' parameter.

        Returns:
            A ChimericFileUploadResponse containing upload information.

        Raises:
            ValueError: If the 'file' parameter is missing from kwargs.
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
