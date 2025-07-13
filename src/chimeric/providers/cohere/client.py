from collections.abc import AsyncGenerator, Generator
import json
from typing import Any

from cohere import AsyncClientV2 as AsyncCohere
from cohere import ClientV2 as Cohere

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
    Tools,
    Usage,
)


class CohereClient(BaseClient[Cohere, AsyncCohere, Any, Any, Any]):
    """Cohere Client for interacting with Cohere models via the Cohere API.

    This client provides a unified interface for synchronous and asynchronous
    interactions with Cohere's API via the `cohere` library. It returns `chimeric`
    response objects that wrap the native Cohere responses.
    """

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        """Initializes the Cohere client.

        Args:
            api_key: The Cohere API key for authentication.
            **kwargs: Additional keyword arguments to pass to the Cohere client constructor.
        """
        self._provider_name = "Cohere"
        super().__init__(api_key=api_key, **kwargs)

    def _get_generic_types(self) -> dict[str, type]:
        """Returns the concrete Cohere client types for `kwargs` filtering.

        Returns:
            A dictionary mapping client type names to their respective classes.
        """
        return {
            "sync": Cohere,
            "async": AsyncCohere,
        }

    def _init_client(self, client_type: type, **kwargs: Any) -> Cohere:
        """Initializes the synchronous Cohere client.

        Args:
            client_type: The Cohere client class.
            **kwargs: Additional arguments for the client constructor.

        Returns:
            An instance of the synchronous Cohere client.
        """
        return Cohere(api_key=self.api_key, **kwargs)

    def _init_async_client(self, async_client_type: type, **kwargs: Any) -> AsyncCohere:
        """Initializes the asynchronous Cohere client.

        Args:
            async_client_type: The AsyncCohere client class.
            **kwargs: Additional arguments for the client constructor.

        Returns:
            An instance of the asynchronous Cohere client.
        """
        return AsyncCohere(api_key=self.api_key, **kwargs)

    def _get_capabilities(self) -> Capability:
        """Gets Cohere provider capabilities.

        Returns:
            A Capability object indicating all supported features.
        """
        return Capability(multimodal=False, streaming=True, tools=True, agents=False, files=False)

    def _list_models_impl(self) -> list[ModelSummary]:
        """Lists available models from the Cohere API.

        Returns:
            A list of ModelSummary objects for all available models from the API.
        """
        models_response = self.client.models.list()
        models = []
        for model in models_response.models:
            model_id = str(getattr(model, "id", getattr(model, "name", "unknown")))
            model_name = str(getattr(model, "name", "unknown"))
            models.append(
                ModelSummary(
                    id=model_id,
                    name=model_name,
                    metadata=model.model_dump() if hasattr(model, "model_dump") else {},
                    owned_by="cohere",
                )
            )
        return models

    @staticmethod
    def _process_event(
        event: Any,
        accumulated: str,
    ) -> tuple[str, ChimericStreamChunk[Any] | None]:
        """Processes a single event from a Cohere response stream.

        Args:
            event: The response stream event from the Cohere API.
            accumulated: The accumulated content from previous events.

        Returns:
            A tuple containing the updated accumulated content and an optional
            ChimericStreamChunk to be yielded.
        """
        event_type = event.type

        if event_type == "content-delta":
            delta = (
                event.delta.message.content.text
                if hasattr(event, "delta") and hasattr(event.delta, "message")
                else ""
            )
            accumulated += delta
            return accumulated, ChimericStreamChunk(
                native=event,
                common=StreamChunk(
                    content=accumulated,
                    delta=delta,
                    metadata=event.model_dump() if hasattr(event, "model_dump") else {},
                ),
            )

        if event_type == "message-end":
            return accumulated, ChimericStreamChunk(
                native=event,
                common=StreamChunk(
                    content=accumulated,
                    finish_reason="end_turn",
                    metadata=event.model_dump() if hasattr(event, "model_dump") else {},
                ),
            )

        return accumulated, None

    def _stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: Tools = None,
        tools_enabled: bool = False,
        **kwargs: Any,
    ) -> Generator[ChimericStreamChunk[Any], None, None]:
        """Processes a streaming chat response with optional tool call handling.

        Args:
            messages: The current conversation messages.
            model: The model to use.
            tools: The tools available for use.
            tools_enabled: Whether tools are enabled for this chat.
            **kwargs: Additional API parameters.

        Yields:
            ChimericStreamChunk containing the processed stream data.
        """
        current_messages = list(messages)  # Make a copy to avoid modifying original

        while True:
            if not tools_enabled:
                # Simple streaming without tool handling
                stream = self._client.chat_stream(
                    model=model,
                    messages=current_messages,  # type: ignore[arg-type]
                    tools=tools,  # type: ignore[arg-type]
                    **kwargs,
                )

                accumulated_content = ""
                for event in stream:
                    accumulated_content, chunk = self._process_event(event, accumulated_content)
                    if chunk:
                        yield chunk
                break
            else:
                # Streaming with tool call handling
                stream = self._client.chat_stream(
                    model=model,
                    messages=current_messages,  # type: ignore[arg-type]
                    tools=tools,  # type: ignore[arg-type]
                    **kwargs,
                )

                accumulated_content = ""

                # Stream all events for the current response
                for event in stream:
                    accumulated_content, chunk = self._process_event(event, accumulated_content)
                    if chunk:
                        yield chunk

                # After streaming is complete, get the full response to check for tool calls
                response = self._client.chat(
                    model=model,
                    messages=current_messages,  # type: ignore[arg-type]
                    tools=tools,  # type: ignore[arg-type]
                    **kwargs,
                )

                # Check for tool calls and process them
                if (
                    hasattr(response, "message")
                    and hasattr(response.message, "tool_calls")
                    and response.message.tool_calls
                ):
                    # Add the assistant's tool call message
                    current_messages.append(
                        {
                            "role": "assistant",
                            "tool_plan": getattr(response.message, "tool_plan", ""),
                            "tool_calls": response.message.tool_calls,
                        }
                    )

                    # Process each tool call
                    for call in response.message.tool_calls:
                        tool_call_info = self._process_function_call(call)

                        # Add the tool result message
                        current_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": [
                                    {
                                        "type": "document",
                                        "document": {"data": tool_call_info["result"]},
                                    }
                                ],
                            }
                        )

                    # Continue to get the next response which will be streamed
                    continue

                # No more tool calls, we're done
                break

    async def _astream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: Tools = None,
        tools_enabled: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[ChimericStreamChunk[Any], None]:
        """Processes an async streaming chat response with optional tool call handling.

        Args:
            messages: The current conversation messages.
            model: The model to use.
            tools: The tools available for use.
            tools_enabled: Whether tools are enabled for this chat.
            **kwargs: Additional API parameters.

        Yields:
            ChimericStreamChunk containing the processed stream data.
        """
        current_messages = list(messages)  # Make a copy to avoid modifying original

        while True:
            if not tools_enabled:
                # Simple streaming without tool handling
                stream = self._async_client.chat_stream(
                    model=model,
                    messages=current_messages,  # type: ignore[arg-type]
                    tools=tools,  # type: ignore[arg-type]
                    **kwargs,
                )

                accumulated_content = ""
                async for event in stream:
                    accumulated_content, chunk = self._process_event(event, accumulated_content)
                    if chunk:
                        yield chunk
                break
            else:
                # Streaming with tool call handling
                stream = self._async_client.chat_stream(
                    model=model,
                    messages=current_messages,  # type: ignore[arg-type]
                    tools=tools,  # type: ignore[arg-type]
                    **kwargs,
                )

                accumulated_content = ""

                # Stream all events for the current response
                async for event in stream:
                    accumulated_content, chunk = self._process_event(event, accumulated_content)
                    if chunk:
                        yield chunk

                # After streaming is complete, get the full response to check for tool calls
                response = await self._async_client.chat(
                    model=model,
                    messages=current_messages,  # type: ignore[arg-type]
                    tools=tools,  # type: ignore[arg-type]
                    **kwargs,
                )

                # Check for tool calls and process them
                if (
                    hasattr(response, "message")
                    and hasattr(response.message, "tool_calls")
                    and response.message.tool_calls
                ):
                    # Add the assistant's tool call message
                    current_messages.append(
                        {
                            "role": "assistant",
                            "tool_plan": getattr(response.message, "tool_plan", ""),
                            "tool_calls": response.message.tool_calls,
                        }
                    )

                    # Process each tool call
                    for call in response.message.tool_calls:
                        tool_call_info = self._process_function_call(call)

                        # Add the tool result message
                        current_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": [
                                    {
                                        "type": "document",
                                        "document": {"data": tool_call_info["result"]},
                                    }
                                ],
                            }
                        )

                    # Continue to get the next response which will be streamed
                    continue
                else:
                    # No more tool calls, we're done
                    break

    @staticmethod
    def _create_chimeric_response(
        response: Any, tool_calls: list[dict[str, Any]]
    ) -> ChimericCompletionResponse[Any]:
        """Creates a ChimericCompletionResponse from a native Cohere ChatResponse.

        Args:
            response: The Cohere response object from `client.chat`.
            tool_calls: A list of tool calls made and executed during the request.

        Returns:
            A ChimericCompletionResponse wrapping the native response.
        """
        metadata = response.model_dump() if hasattr(response, "model_dump") else {}
        if tool_calls:
            metadata["tool_calls"] = tool_calls

        # Extract content from the response
        content = ""
        if hasattr(response, "message") and hasattr(response.message, "content"):
            if isinstance(response.message.content, list) and len(response.message.content) > 0:
                content = response.message.content[0].text
            else:
                content = str(response.message.content)
        elif hasattr(response, "text"):
            content = response.text

        # Extract usage information - Cohere API uses response.usage.tokens structure
        usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        if hasattr(response, "usage") and hasattr(response.usage, "tokens"):
            input_tokens = getattr(response.usage.tokens, "input_tokens", 0)
            output_tokens = getattr(response.usage.tokens, "output_tokens", 0)
            usage = Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )

        return ChimericCompletionResponse(
            native=response,
            common=CompletionResponse(
                content=content,
                usage=usage,
                model=getattr(response, "model", ""),
                metadata=metadata,
            ),
        )

    def _encode_tools(self, tools: Tools = None) -> Tools:
        """Encodes a list of Tool objects into the format expected by the Cohere API.

        Args:
            tools: A list of Tool objects or dictionaries.

        Returns:
            A list of tool dictionaries formatted for the Cohere API, or None.
        """
        if not tools:
            return None

        encoded_tools = []
        for tool in tools:
            if isinstance(tool, Tool):
                encoded_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters.model_dump() if tool.parameters else {},
                        },
                    }
                )
            else:
                encoded_tools.append(tool)  # Assumes tool is already a dict
        return encoded_tools

    def _process_function_call(self, call: Any) -> dict[str, Any]:
        """Executes a function call from the model and returns metadata.

        Args:
            call: The tool call object from the Cohere response.

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
            error_msg = f"Error executing tool '{tool_name}': {e!s}"
            tool_call_info["result"] = error_msg
            tool_call_info["error"] = True

        return tool_call_info

    def _handle_function_tool_calls(
        self, model: str, response: Any, messages: list[dict[str, Any]], tools: Tools = None
    ) -> tuple[list[dict[str, Any]], Any]:
        """Processes tool calls from a response and updates the message history iteratively.

        Args:
            model: The model to use for the chat completion.
            response: The Cohere response containing potential tool calls.
            messages: The current list of messages to append to.
            tools: The tools available for use.

        Returns:
            A tuple containing (all_tool_calls_metadata, final_response).
        """
        all_tool_calls_metadata = []
        current_response = response
        current_messages = list(messages)  # Make a copy

        # Continue processing until no more tool calls
        while (
            hasattr(current_response, "message")
            and hasattr(current_response.message, "tool_calls")
            and current_response.message.tool_calls
        ):
            # Add the assistant's tool call message
            current_messages.append(
                {
                    "role": "assistant",
                    "tool_plan": getattr(current_response.message, "tool_plan", ""),
                    "tool_calls": current_response.message.tool_calls,
                }
            )

            # Process each tool call
            for call in current_response.message.tool_calls:
                tool_call_info = self._process_function_call(call)
                all_tool_calls_metadata.append(tool_call_info)

                # Add the tool result message
                current_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": [
                            {
                                "type": "document",
                                "document": {"data": tool_call_info["result"]},
                            }
                        ],
                    }
                )

            # Get the next response to check for more tool calls
            current_response = self._client.chat(
                model=model,
                messages=current_messages,  # type: ignore[arg-type]
                tools=tools,  # type: ignore[arg-type]
            )

        return all_tool_calls_metadata, current_response

    async def _handle_function_tool_calls_async(
        self, model: str, response: Any, messages: list[dict[str, Any]], tools: Tools = None
    ) -> tuple[list[dict[str, Any]], Any]:
        """Processes tool calls from a response and updates the message history iteratively.

        Args:
            model: The model to use for the chat completion.
            response: The Cohere response containing potential tool calls.
            messages: The current list of messages to append to.
            tools: The tools available for use.

        Returns:
            A tuple containing (all_tool_calls_metadata, final_response).
        """
        all_tool_calls_metadata = []
        current_response = response
        current_messages = list(messages)  # Make a copy

        # Continue processing until no more tool calls
        while (
            hasattr(current_response, "message")
            and hasattr(current_response.message, "tool_calls")
            and current_response.message.tool_calls
        ):
            # Add the assistant's tool call message
            current_messages.append(
                {
                    "role": "assistant",
                    "tool_plan": getattr(current_response.message, "tool_plan", ""),
                    "tool_calls": current_response.message.tool_calls,
                }
            )

            # Process each tool call
            for call in current_response.message.tool_calls:
                tool_call_info = self._process_function_call(call)
                all_tool_calls_metadata.append(tool_call_info)

                # Add the tool result message
                current_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": [
                            {
                                "type": "document",
                                "document": {"data": tool_call_info["result"]},
                            }
                        ],
                    }
                )

            # Get the next response to check for more tool calls
            current_response = await self._async_client.chat(
                model=model,
                messages=current_messages,  # type: ignore[arg-type]
                tools=tools,  # type: ignore[arg-type]
            )

        return all_tool_calls_metadata, current_response

    @staticmethod
    def _convert_messages_to_cohere_format(messages: Input) -> list[dict[str, Any]]:
        """Converts input messages to Cohere API format.

        Args:
            messages: The input messages in various formats.

        Returns:
            A list of messages formatted for the Cohere API.
        """
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if isinstance(messages, dict):
            return [messages]
        if isinstance(messages, list):  # type: ignore[reportUnnecessaryIsInstance]
            return messages
        return [{"role": "user", "content": str(messages)}]

    def _chat_completion_impl(
        self,
        messages: Input,
        model: str,
        stream: bool = False,
        tools: Tools = None,
        **kwargs: Any,
    ) -> ChimericCompletionResponse[Any] | Generator[ChimericStreamChunk[Any], None, None]:
        """Sends a synchronous chat completion request to the Cohere API.

        Args:
            messages: The input messages for the chat completion.
            model: The model to use for the completion.
            stream: Whether to return a streaming response.
            tools: A list of tools (already encoded for the API).
            **kwargs: Additional arguments for the `client.chat` method.

        Returns:
            A ChimericCompletionResponse for a single response, or a generator
            of ChimericStreamChunk for a streaming response.
        """
        filtered_kwargs = self._filter_kwargs(self._client.chat, kwargs)
        cohere_messages = self._convert_messages_to_cohere_format(messages)

        if stream:
            return self._stream(
                cohere_messages, model, tools=tools, tools_enabled=bool(tools), **filtered_kwargs
            )

        # Send non-streaming request
        response = self._client.chat(
            model=model,
            messages=cohere_messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            **filtered_kwargs,
        )

        # Handle tool calls iteratively until complete
        tool_calls_metadata, final_response = self._handle_function_tool_calls(
            model=model,
            response=response,
            messages=cohere_messages,
            tools=tools,
        )

        return self._create_chimeric_response(final_response, tool_calls_metadata)

    async def _achat_completion_impl(
        self,
        messages: Input,
        model: str,
        stream: bool = False,
        tools: Tools = None,
        **kwargs: Any,
    ) -> ChimericCompletionResponse[Any] | AsyncGenerator[ChimericStreamChunk[Any], None]:
        """Sends an asynchronous chat completion request to the Cohere API.

        Args:
            messages: The input messages for the chat completion.
            model: The model to use for the completion.
            stream: Whether to return a streaming response.
            tools: A list of tools (already encoded for the API).
            **kwargs: Additional arguments for the `async_client.chat` method.

        Returns:
            A ChimericCompletionResponse for a single response, or an async
            generator of ChimericStreamChunk for a streaming response.
        """
        filtered_kwargs = self._filter_kwargs(self._async_client.chat, kwargs)
        cohere_messages = self._convert_messages_to_cohere_format(messages)

        if stream:
            return self._astream(
                cohere_messages, model, tools=tools, tools_enabled=bool(tools), **filtered_kwargs
            )

        # Send non-streaming request
        response = await self._async_client.chat(
            model=model,
            messages=cohere_messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            **filtered_kwargs,
        )

        # Handle tool calls iteratively until complete
        tool_calls_metadata, final_response = await self._handle_function_tool_calls_async(
            model=model,
            response=response,
            messages=cohere_messages,
            tools=tools,
        )

        return self._create_chimeric_response(final_response, tool_calls_metadata)

    def _upload_file(self, **kwargs: Any) -> ChimericFileUploadResponse[Any]:
        """Uploads a file to Cohere (not supported).

        Args:
            **kwargs: Provider-specific arguments for file upload.

        Returns:
            A ChimericFileUploadResponse containing the native response.

        Raises:
            NotImplementedError: Cohere does not support file uploads.
        """
        raise NotImplementedError("Cohere does not support file uploads")
