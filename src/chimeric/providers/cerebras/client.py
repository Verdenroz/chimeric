from collections.abc import AsyncGenerator, Generator
import json
from typing import Any, cast

from cerebras.cloud.sdk import AsyncCerebras, Cerebras
from cerebras.cloud.sdk.types.chat.chat_completion import (
    ChatChunkResponse,
    ChatCompletionResponse,
    ChatCompletionResponseChoiceMessageToolCall,
)

from chimeric.base import BaseClient
from chimeric.exceptions import ToolRegistrationError
from chimeric.types import (
    Capability,
    ChimericCompletionResponse,
    ChimericFileUploadResponse,
    ChimericStreamChunk,
    CompletionResponse,
    Input,
    ModelSummary,
    StreamChunk,
    Tool,
    ToolParameters,
    Tools,
    Usage,
)


class CerebrasClient(
    BaseClient[Cerebras, AsyncCerebras, ChatCompletionResponse, ChatChunkResponse, None]
):
    """Cerebras Client for interacting with Cerebras Cloud API.

    This client provides a unified interface for synchronous and asynchronous
    interactions with Cerebras's API via the `cerebras-cloud-sdk` library. It returns
    `chimeric` response objects that wrap the native Cerebras responses.
    """

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        """Initializes the Cerebras client.

        Args:
            api_key: The Cerebras API key for authentication.
            **kwargs: Additional keyword arguments to pass to the Cerebras client constructor.
        """
        self._provider_name = "Cerebras"
        super().__init__(api_key=api_key, **kwargs)

    def _get_generic_types(self) -> dict[str, type]:
        """Returns the concrete Cerebras client types for `kwargs` filtering.

        Returns:
            A dictionary mapping client type names to their respective classes.
        """
        return {
            "sync": Cerebras,
            "async": AsyncCerebras,
        }

    def _init_client(self, client_type: type, **kwargs: Any) -> Cerebras:
        """Initializes the synchronous Cerebras client.

        Args:
            client_type: The Cerebras client class.
            **kwargs: Additional arguments for the client constructor.

        Returns:
            An instance of the synchronous Cerebras client.
        """
        return Cerebras(api_key=self.api_key, **kwargs)

    def _init_async_client(self, async_client_type: type, **kwargs: Any) -> AsyncCerebras:
        """Initializes the asynchronous Cerebras client.

        Args:
            async_client_type: The AsyncCerebras client class.
            **kwargs: Additional arguments for the client constructor.

        Returns:
            An instance of the asynchronous Cerebras client.
        """
        return AsyncCerebras(api_key=self.api_key, **kwargs)

    def _get_capabilities(self) -> Capability:
        """Gets Cerebras provider capabilities.

        Returns:
            A Capability object indicating supported features.
        """
        return Capability(multimodal=False, streaming=True, tools=True, agents=False, files=False)

    def _list_models_impl(self) -> list[ModelSummary]:
        """Lists available models from the Cerebras API.

        Returns:
            A list of ModelSummary objects for all available models from the API.
        """
        models = self.client.models.list()
        return [
            ModelSummary(
                id=model.id,
                name=model.id,
                owned_by=getattr(model, "owned_by", "cerebras"),
                created_at=getattr(model, "created", None),
            )
            for model in models.data
        ]

    @staticmethod
    def _create_chimeric_response(
        response: ChatCompletionResponse, tool_calls: list[dict[str, Any]]
    ) -> ChimericCompletionResponse[ChatCompletionResponse]:
        """Creates a ChimericCompletionResponse from a native Cerebras ChatCompletion.

        Args:
            response: The Cerebras response object from `client.chat.completions.create`.
            tool_calls: A list of tool calls made and executed during the request.

        Returns:
            A ChimericCompletionResponse wrapping the native response.
        """
        metadata = response.model_dump()
        if tool_calls:
            metadata["tool_calls"] = tool_calls

        # Extract content from the first choice
        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice and choice.message else ""

        return ChimericCompletionResponse(
            native=response,
            common=CompletionResponse(
                content=content or "",
                usage=Usage(
                    prompt_tokens=(response.usage.prompt_tokens if response.usage else 0) or 0,
                    completion_tokens=(response.usage.completion_tokens if response.usage else 0)
                    or 0,
                    total_tokens=(response.usage.total_tokens if response.usage else 0) or 0,
                ),
                model=response.model,
                metadata=metadata,
            ),
        )

    def _encode_tools(self, tools: Tools = None) -> Tools:
        """Encodes a list of Tool objects into the format expected by the Cerebras API.

        Args:
            tools: A list of Tool objects or dictionaries.

        Returns:
            A list of tool dictionaries formatted for the Cerebras API, or None.
        """
        if not tools:
            return None

        encoded_tools = []
        for tool in tools:
            if isinstance(tool, Tool):
                # Get parameters and remove 'strict' from the parameters schema since
                # Cerebras expects it only in the function object
                parameters = (
                    tool.parameters.model_dump()
                    if isinstance(tool.parameters, ToolParameters)
                    else {}
                )
                parameters.pop("strict", None)  # Remove strict from parameters

                encoded_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "strict": True,
                            "description": tool.description,
                            "parameters": parameters,
                        },
                    }
                )
            else:
                encoded_tools.append(tool)  # Assumes user already formats tool
        return encoded_tools

    def _process_function_call(
        self, call: ChatCompletionResponseChoiceMessageToolCall
    ) -> dict[str, Any]:
        """Executes a function call from the model and returns metadata.

        Args:
            call: The tool call object from the Cerebras response.

        Returns:
            A dictionary containing the tool call ID, name, arguments, and result.

        Raises:
            ToolRegistrationError: If the requested tool is not registered or not callable.
        """
        tool_name = call.function.name
        tool_call_info = {
            "call_id": call.id,
            "name": tool_name,
            "arguments": call.function.arguments,
        }

        tool = self.tool_manager.get_tool(tool_name)
        if not callable(tool.function):
            raise ToolRegistrationError(f"Tool '{tool_name}' is not callable.")

        try:
            args = json.loads(call.function.arguments)
            result = tool.function(**args)
            tool_call_info["result"] = str(result)
        except Exception as e:
            tool_call_info["error"] = str(e)
            tool_call_info["result"] = f"Error: {e!s}"

        return tool_call_info

    @staticmethod
    def _stream(
        stream: Generator[ChatChunkResponse, None, None],
    ) -> Generator[ChimericStreamChunk[ChatChunkResponse], None, None]:
        """Yields processed chunks from a synchronous Cerebras stream.

        Args:
            stream: The synchronous stream of response chunks from `client.chat.completions.create`.

        Yields:
            ChimericStreamChunk containing the processed chunk data.
        """
        accumulated = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                accumulated += delta
                yield ChimericStreamChunk(
                    native=chunk,
                    common=StreamChunk(
                        content=accumulated,
                        delta=delta,
                        finish_reason=chunk.choices[0].finish_reason,
                        metadata=chunk.model_dump(),
                    ),
                )

    @staticmethod
    async def _astream(
        stream: AsyncGenerator[ChatChunkResponse, None],
    ) -> AsyncGenerator[ChimericStreamChunk[ChatChunkResponse], None]:
        """Yields processed chunks from an asynchronous Cerebras stream.

        Args:
            stream: The asynchronous stream of response chunks from `async_client.chat.completions.create`.

        Yields:
            ChimericStreamChunk containing the processed chunk data.
        """
        accumulated = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                accumulated += delta
                yield ChimericStreamChunk(
                    native=chunk,
                    common=StreamChunk(
                        content=accumulated,
                        delta=delta,
                        finish_reason=chunk.choices[0].finish_reason,
                        metadata=chunk.model_dump(),
                    ),
                )

    def _process_stream_with_tools_sync(
        self,
        stream: Generator[ChatChunkResponse, None, None],
        original_messages: list[dict[str, Any]],
        original_model: str,
        original_tools: Tools | None = None,
        **original_kwargs: Any,
    ) -> Generator[ChimericStreamChunk[ChatChunkResponse], None, None]:
        """Processes a synchronous stream with automatic tool execution.

        Args:
            stream: The synchronous stream from the Cerebras API.
            original_messages: The initial list of messages.
            original_model: The model being used.
            original_tools: The list of available tools.
            **original_kwargs: Additional arguments for the API call.

        Yields:
            ChimericStreamChunk objects from the initial and subsequent streams.
        """
        accumulated = ""
        accumulated_tool_calls = []

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                accumulated += delta

            # Check for tool calls in the delta
            if chunk.choices and chunk.choices[0].delta.tool_calls:
                accumulated_tool_calls.extend(chunk.choices[0].delta.tool_calls)

            yield ChimericStreamChunk(
                native=chunk,
                common=StreamChunk(
                    content=accumulated,
                    delta=chunk.choices[0].delta.content if chunk.choices else "",
                    finish_reason=chunk.choices[0].finish_reason if chunk.choices else None,
                    metadata=chunk.model_dump(),
                ),
            )

            # If the stream is complete, and we have tool calls, execute them
            if chunk.choices and chunk.choices[0].finish_reason and accumulated_tool_calls:
                # Execute tools and continue conversation
                yield from self._handle_stream_tool_execution_sync(
                    accumulated_tool_calls,
                    original_messages,
                    original_model,
                    original_tools,
                    accumulated,
                    **original_kwargs,
                )
                return

    async def _process_stream_with_tools_async(
        self,
        stream: AsyncGenerator[ChatChunkResponse, None],
        original_messages: list[dict[str, Any]],
        original_model: str,
        original_tools: Tools | None = None,
        **original_kwargs: Any,
    ) -> AsyncGenerator[ChimericStreamChunk[ChatChunkResponse], None]:
        """Processes an asynchronous stream with automatic tool execution.

        Args:
            stream: The asynchronous stream from the Cerebras API.
            original_messages: The initial list of messages.
            original_model: The model being used.
            original_tools: The list of available tools.
            **original_kwargs: Additional arguments for the API call.

        Yields:
            ChimericStreamChunk objects from the initial and subsequent streams.
        """
        accumulated = ""
        accumulated_tool_calls = []

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                accumulated += delta

            # Check for tool calls in the delta
            if chunk.choices and chunk.choices[0].delta.tool_calls:
                accumulated_tool_calls.extend(chunk.choices[0].delta.tool_calls)

            yield ChimericStreamChunk(
                native=chunk,
                common=StreamChunk(
                    content=accumulated,
                    delta=chunk.choices[0].delta.content if chunk.choices else "",
                    finish_reason=chunk.choices[0].finish_reason if chunk.choices else None,
                    metadata=chunk.model_dump(),
                ),
            )

            # If the stream is complete, and we have tool calls, execute them
            if chunk.choices and chunk.choices[0].finish_reason and accumulated_tool_calls:
                # Execute tools and continue conversation
                async for tool_chunk in self._handle_stream_tool_execution_async(
                    accumulated_tool_calls,
                    original_messages,
                    original_model,
                    original_tools,
                    accumulated,
                    **original_kwargs,
                ):
                    yield tool_chunk
                return

    def _handle_stream_tool_execution_sync(
        self,
        tool_calls: list[Any],
        messages: list[dict[str, Any]],
        model: str,
        tools: Tools | None,
        accumulated_content: str,
        **kwargs: Any,
    ) -> Generator[ChimericStreamChunk[ChatChunkResponse], None, None]:
        """Executes tools from a stream and continues the conversation.

        Args:
            tool_calls: Tool calls from the stream.
            messages: Current message history.
            model: Model to use.
            tools: Available tools.
            accumulated_content: Text content accumulated from the stream.
            **kwargs: Additional API parameters.

        Yields:
            ChimericStreamChunk objects from the continuation.
        """
        # Build the assistant message with content and tool calls
        assistant_message = {
            "role": "assistant",
            "content": accumulated_content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ],
        }
        messages.append(assistant_message)

        # Execute tools and add results
        for call in tool_calls:
            tool_call_info = self._process_function_call(call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": tool_call_info["result"],
                }
            )

        # Continue conversation
        filtered_kwargs = self._filter_kwargs(self._client.chat.completions.create, kwargs)
        continuation = self._client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            stream=True,
            **filtered_kwargs,
        )

        yield from self._process_stream_with_tools_sync(
            cast("Generator[ChatChunkResponse, None, None]", continuation),
            messages,
            model,
            tools,
            **kwargs,
        )

    async def _handle_stream_tool_execution_async(
        self,
        tool_calls: list[Any],
        messages: list[dict[str, Any]],
        model: str,
        tools: Tools | None,
        accumulated_content: str,
        **kwargs: Any,
    ) -> AsyncGenerator[ChimericStreamChunk[ChatChunkResponse], None]:
        """Executes tools from a stream and continues the conversation asynchronously.

        Args:
            tool_calls: Tool calls from the stream.
            messages: Current message history.
            model: Model to use.
            tools: Available tools.
            accumulated_content: Text content accumulated from the stream.
            **kwargs: Additional API parameters.

        Yields:
            ChimericStreamChunk objects from the continuation.
        """
        # Build the assistant message with content and tool calls
        assistant_message = {
            "role": "assistant",
            "content": accumulated_content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ],
        }
        messages.append(assistant_message)

        # Execute tools and add results
        for call in tool_calls:
            tool_call_info = self._process_function_call(call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": tool_call_info["result"],
                }
            )

        # Continue conversation
        filtered_kwargs = self._filter_kwargs(self._async_client.chat.completions.create, kwargs)
        continuation = await self._async_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            stream=True,
            **filtered_kwargs,
        )

        async for chunk in self._process_stream_with_tools_async(
            cast("AsyncGenerator[ChatChunkResponse, None]", continuation),
            messages,
            model,
            tools,
            **kwargs,
        ):
            yield chunk

    def _handle_tool_execution_loop(
        self,
        response: ChatCompletionResponse,
        current_messages: list[dict[str, Any]],
        model: str,
        tools: Tools,
        **kwargs: Any,
    ) -> tuple[ChatCompletionResponse, list[dict[str, Any]]]:
        """Handles the tool execution loop for non-streaming responses.

        Args:
            response: The initial response from the API.
            current_messages: The current message history.
            model: The model being used.
            tools: The list of available tools.
            **kwargs: Additional API parameters.

        Returns:
            A tuple containing the final API response and all tool call metadata.
        """
        all_tool_calls = []

        while True:
            # Check for tool calls in the response
            choice = response.choices[0] if response.choices else None
            if not choice or not choice.message.tool_calls:
                # No tool calls, we're done
                break

            # Add the assistant's message with tool calls to the conversation
            assistant_message = {
                "role": "assistant",
                "content": choice.message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in choice.message.tool_calls
                ],
            }
            current_messages.append(assistant_message)

            # Process each tool call
            for call in choice.message.tool_calls:
                tool_call_info = self._process_function_call(call)
                all_tool_calls.append(tool_call_info)

                # Add the tool result to the conversation
                current_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": tool_call_info["result"],
                    }
                )

            # Continue the conversation
            filtered_kwargs = self._filter_kwargs(self._client.chat.completions.create, kwargs)
            response = cast(
                "ChatCompletionResponse",
                self._client.chat.completions.create(
                    model=model,
                    messages=current_messages,
                    tools=tools,
                    **filtered_kwargs,
                ),
            )

        return response, all_tool_calls

    async def _handle_async_tool_execution_loop(
        self,
        response: ChatCompletionResponse,
        current_messages: list[dict[str, Any]],
        model: str,
        tools: Tools,
        **kwargs: Any,
    ) -> tuple[ChatCompletionResponse, list[dict[str, Any]]]:
        """Handles the asynchronous tool execution loop for non-streaming responses.

        Args:
            response: The initial response from the API.
            current_messages: The current message history.
            model: The model being used.
            tools: The list of available tools.
            **kwargs: Additional API parameters.

        Returns:
            A tuple containing the final API response and all tool call metadata.
        """
        all_tool_calls = []

        while True:
            # Check for tool calls in the response
            choice = response.choices[0] if response.choices else None
            if not choice or not choice.message.tool_calls:
                # No tool calls, we're done
                break

            # Add the assistant's message with tool calls to the conversation
            assistant_message = {
                "role": "assistant",
                "content": choice.message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in choice.message.tool_calls
                ],
            }
            current_messages.append(assistant_message)

            # Process each tool call
            for call in choice.message.tool_calls:
                tool_call_info = self._process_function_call(call)
                all_tool_calls.append(tool_call_info)

                # Add the tool result to the conversation
                current_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": tool_call_info["result"],
                    }
                )

            # Continue the conversation
            filtered_kwargs = self._filter_kwargs(
                self._async_client.chat.completions.create, kwargs
            )
            response = cast(
                "ChatCompletionResponse",
                await self._async_client.chat.completions.create(
                    model=model,
                    messages=current_messages,
                    tools=tools,
                    **filtered_kwargs,
                ),
            )

        return response, all_tool_calls

    def _chat_completion_impl(
        self,
        messages: Input,
        model: str,
        stream: bool = False,
        tools: Tools = None,
        **kwargs: Any,
    ) -> (
        ChimericCompletionResponse[ChatCompletionResponse]
        | Generator[ChimericStreamChunk[ChatChunkResponse], None, None]
    ):
        """Sends a synchronous chat completion request to the Cerebras API.

        Args:
            messages: The input messages for the chat completion.
            model: The model to use for the completion.
            stream: Whether to return a streaming response.
            tools: A list of tools (already encoded for the API).
            **kwargs: Additional arguments for the `client.chat.completions.create` method.

        Returns:
            A ChimericCompletionResponse for a single response, or a generator
            of ChimericStreamChunk for a streaming response.
        """
        filtered_kwargs = self._filter_kwargs(self._client.chat.completions.create, kwargs)

        # Convert messages to mutable list of dicts
        if isinstance(messages, list):
            current_messages = [
                cast("dict[str, object]", msg)
                if isinstance(msg, dict)
                else cast("dict[str, object]", {"role": "user", "content": str(msg)})
                for msg in messages
            ]
        else:
            current_messages = [
                cast("dict[str, object]", {"role": "user", "content": str(messages)})
            ]

        # Make initial API call
        response = self._client.chat.completions.create(
            model=model,
            messages=current_messages,
            stream=stream,
            tools=tools,
            **filtered_kwargs,
        )

        if stream:
            # Handle streaming with tools if needed
            if tools:
                return self._process_stream_with_tools_sync(
                    cast("Generator[ChatChunkResponse, None, None]", response),
                    current_messages,
                    model,
                    tools,
                    **kwargs,
                )
            return self._stream(cast("Generator[ChatChunkResponse, None, None]", response))

        # Handle tool execution if tools are available
        if tools:
            final_response, all_tool_calls = self._handle_tool_execution_loop(
                cast("ChatCompletionResponse", response), current_messages, model, tools, **kwargs
            )
            return self._create_chimeric_response(final_response, all_tool_calls)

        return self._create_chimeric_response(cast("ChatCompletionResponse", response), [])

    async def _achat_completion_impl(
        self,
        messages: Input,
        model: str,
        stream: bool = False,
        tools: Tools = None,
        **kwargs: Any,
    ) -> (
        ChimericCompletionResponse[ChatCompletionResponse]
        | AsyncGenerator[ChimericStreamChunk[ChatChunkResponse], None]
    ):
        """Sends an asynchronous chat completion request to the Cerebras API.

        Args:
            messages: The input messages for the chat completion.
            model: The model to use for the completion.
            stream: Whether to return a streaming response.
            tools: A list of tools (already encoded for the API).
            **kwargs: Additional arguments for the `async_client.chat.completions.create` method.

        Returns:
            A ChimericCompletionResponse for a single response, or an async
            generator of ChimericStreamChunk for a streaming response.
        """
        filtered_kwargs = self._filter_kwargs(self._async_client.chat.completions.create, kwargs)

        # Convert messages to mutable list of dicts
        if isinstance(messages, list):
            current_messages = [
                cast("dict[str, object]", msg)
                if isinstance(msg, dict)
                else cast("dict[str, object]", {"role": "user", "content": str(msg)})
                for msg in messages
            ]
        else:
            current_messages = [
                cast("dict[str, object]", {"role": "user", "content": str(messages)})
            ]

        # Make initial API call
        response = await self._async_client.chat.completions.create(
            model=model,
            messages=current_messages,
            stream=stream,
            tools=tools,
            **filtered_kwargs,
        )

        if stream:
            # Handle streaming with tools if needed
            if tools:
                return self._process_stream_with_tools_async(
                    cast("AsyncGenerator[ChatChunkResponse, None]", response),
                    current_messages,
                    model,
                    tools,
                    **kwargs,
                )
            return self._astream(cast("AsyncGenerator[ChatChunkResponse, None]", response))

        # Handle tool execution if tools are available
        if tools:
            final_response, all_tool_calls = await self._handle_async_tool_execution_loop(
                cast("ChatCompletionResponse", response), current_messages, model, tools, **kwargs
            )
            return self._create_chimeric_response(final_response, all_tool_calls)

        return self._create_chimeric_response(cast("ChatCompletionResponse", response), [])

    def _upload_file(self, **kwargs: Any) -> ChimericFileUploadResponse[None]:
        """Cerebras does not support file uploads.

        Args:
            **kwargs: Not used.

        Raises:
            NotImplementedError: Cerebras does not support file uploads.
        """
        raise NotImplementedError("Cerebras does not support file uploads")
