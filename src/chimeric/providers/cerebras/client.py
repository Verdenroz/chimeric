from collections.abc import AsyncGenerator, Generator
import json
from typing import Any

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
    ChimericStreamChunk,
    CompletionResponse,
    Input,
    ModelSummary,
    StreamChunk,
    Tool,
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
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                    total_tokens=response.usage.total_tokens if response.usage else 0,
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
                parameters = tool.parameters.model_dump() if tool.parameters else {}
                parameters.pop('strict', None)  # Remove strict from parameters
                
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

        args = json.loads(call.function.arguments)
        result = tool.function(**args)
        tool_call_info["result"] = str(result)
        return tool_call_info

    def _handle_function_tool_calls(
        self, response: ChatCompletionResponse, messages: Input
    ) -> tuple[list[dict[str, Any]], Input]:
        """Processes tool calls from a response and updates the message history.

        Args:
            response: The Cerebras response containing potential tool calls.
            messages: The current list of messages to append to.

        Returns:
            A tuple containing (tool_calls_metadata, updated_messages).
        """
        choice = response.choices[0] if response.choices else None
        if not choice or not choice.message.tool_calls:
            return [], messages

        # Ensure messages is a mutable list of dictionaries for appending
        messages_list: list[Any] = list(messages) if isinstance(messages, list) else [messages]
        tool_calls_metadata = []

        # Add the assistant's message with tool calls
        messages_list.append(
            {
                "role": "assistant",
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
        )

        for call in choice.message.tool_calls:
            tool_call_info = self._process_function_call(call)
            tool_calls_metadata.append(tool_call_info)

            # Append the tool result to the message history
            messages_list.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": tool_call_info["result"],
                }
            )

        return tool_calls_metadata, messages_list

    def _stream(
        self, stream: Generator[ChatChunkResponse, None, None]
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

    async def _astream(
        self, stream: AsyncGenerator[ChatChunkResponse, None]
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

        # Handle parallel_tool_calls requirement for certain models
        if tools and "parallel_tool_calls" not in filtered_kwargs and "scout" in model.lower():
            filtered_kwargs["parallel_tool_calls"] = False

        # Send the initial request
        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            tools=tools,
            **filtered_kwargs,
        )

        if stream:
            return self._stream(response)

        # Check for and handle any tool calls requested by the model
        tool_calls_metadata, updated_messages = self._handle_function_tool_calls(response, messages)

        # If tool calls were made, send their results back to the model
        if tool_calls_metadata:
            response = self._client.chat.completions.create(
                model=model,
                messages=updated_messages,
                tools=tools,
                **filtered_kwargs,
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

        # Handle parallel_tool_calls requirement for certain models
        if tools and "parallel_tool_calls" not in filtered_kwargs and "scout" in model.lower():
            filtered_kwargs["parallel_tool_calls"] = False

        # Send the initial request
        response = await self._async_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            tools=tools,
            **filtered_kwargs,
        )

        if stream:
            return self._astream(response)

        # Check for and handle any tool calls requested by the model
        tool_calls_metadata, updated_messages = self._handle_function_tool_calls(response, messages)

        # If tool calls were made, send their results back to the model
        if tool_calls_metadata:
            response = await self._async_client.chat.completions.create(
                model=model,
                messages=updated_messages,
                tools=tools,
                **filtered_kwargs,
            )

        return self._create_chimeric_response(response, tool_calls_metadata)

    def _upload_file(self, **kwargs: Any) -> None:
        """Cerebras does not support file uploads.

        Args:
            **kwargs: Not used.

        Raises:
            NotImplementedError: Cerebras does not support file uploads.
        """
        raise NotImplementedError("Cerebras does not support file uploads")
