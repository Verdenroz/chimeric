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
        stream: Any,
    ) -> Generator[ChimericStreamChunk[Any], None, None]:
        """Yields processed chunks from a synchronous Cohere stream.

        Args:
            stream: The synchronous stream of response events from `client.chat_stream`.

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
        stream: Any,
    ) -> AsyncGenerator[ChimericStreamChunk[Any], None]:
        """Yields processed chunks from an asynchronous Cohere stream.

        Args:
            stream: The asynchronous stream of response events from `async_client.chat_stream`.

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

        args = json.loads(call.function.arguments)
        result = tool.function(**args)
        tool_call_info["result"] = str(result)
        return tool_call_info

    def _handle_function_tool_calls(
        self, response: Any, messages: Input
    ) -> tuple[list[dict[str, Any]], Input]:
        """Processes tool calls from a response and updates the message history.

        Args:
            response: The Cohere response containing potential tool calls.
            messages: The current list of messages to append to.

        Returns:
            A tuple containing (tool_calls_metadata, updated_messages).
        """
        if not hasattr(response, "message") or not hasattr(response.message, "tool_calls"):
            return [], messages

        calls = response.message.tool_calls or []
        if not calls:
            return [], messages

        # Ensure messages is a mutable list of dictionaries for appending
        messages_list: list[Any] = list(messages) if isinstance(messages, list) else [messages]
        tool_calls_metadata = []

        # Add the assistant's tool call message
        messages_list.append(
            {
                "role": "assistant",
                "tool_plan": getattr(response.message, "tool_plan", ""),
                "tool_calls": calls,
            }
        )

        # Process each tool call
        for call in calls:
            tool_call_info = self._process_function_call(call)
            tool_calls_metadata.append(tool_call_info)

            # Add tool result message
            messages_list.append(
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

        return tool_calls_metadata, messages_list

    def _convert_messages_to_cohere_format(self, messages: Input) -> list[dict[str, Any]]:
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
            # Send streaming request
            response = self._client.chat_stream(
                model=model,
                messages=cohere_messages,  # type: ignore[arg-type]
                tools=tools,  # type: ignore[arg-type]
                **filtered_kwargs,
            )
            return self._stream(response)
        # Send non-streaming request
        response = self._client.chat(
            model=model,
            messages=cohere_messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            **filtered_kwargs,
        )

        # Check for and handle any tool calls requested by the model
        tool_calls_metadata, updated_messages = self._handle_function_tool_calls(response, messages)

        # If tool calls were made, send their results back to the model
        if tool_calls_metadata:
            updated_cohere_messages = self._convert_messages_to_cohere_format(updated_messages)
            response = self._client.chat(
                model=model,
                messages=updated_cohere_messages,  # type: ignore[arg-type]
                tools=tools,  # type: ignore[arg-type]
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
            # Send streaming request
            response = self._async_client.chat_stream(
                model=model,
                messages=cohere_messages,  # type: ignore[arg-type]
                tools=tools,  # type: ignore[arg-type]
                **filtered_kwargs,
            )
            return self._astream(response)
        # Send non-streaming request
        response = await self._async_client.chat(
            model=model,
            messages=cohere_messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            **filtered_kwargs,
        )

        # Check for and handle any tool calls requested by the model
        tool_calls_metadata, updated_messages = self._handle_function_tool_calls(response, messages)

        # If tool calls were made, send their results back to the model
        if tool_calls_metadata:
            updated_cohere_messages = self._convert_messages_to_cohere_format(updated_messages)
            response = await self._async_client.chat(
                model=model,
                messages=updated_cohere_messages,  # type: ignore[arg-type]
                tools=tools,  # type: ignore[arg-type]
                **filtered_kwargs,
            )

        return self._create_chimeric_response(response, tool_calls_metadata)

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
