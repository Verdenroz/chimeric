from collections.abc import AsyncGenerator, Generator
import json
from typing import Any

from xai_sdk import AsyncClient, Client
from xai_sdk.chat import Chunk, Response, assistant, system, tool, tool_result, user

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


class GrokClient(BaseClient[Client, AsyncClient, Response, Chunk, Any]):
    """Grok Client for interacting with Grok's API using the xai-sdk."""

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        """Initializes the Grok client.

        Args:
            api_key: The Grok API key for authentication.
            **kwargs: Additional keyword arguments to pass to the xai-sdk client constructor.
        """
        self._provider_name = "Grok"
        super().__init__(api_key=api_key, **kwargs)

    def _get_generic_types(self) -> dict[str, type]:
        """Returns the concrete xai-sdk client types for kwargs filtering.

        Returns:
            A dictionary mapping client type names to their respective classes.
        """
        return {
            "sync": Client,
            "async": AsyncClient,
        }

    def _init_client(self, client_type: type, **kwargs: Any) -> Client:
        """Initializes the synchronous Grok client.

        Args:
            client_type: The xai-sdk Client class.
            **kwargs: Additional arguments for the client constructor.

        Returns:
            An instance of the synchronous xai-sdk Client.
        """
        return Client(api_key=self.api_key, **kwargs)

    def _init_async_client(self, async_client_type: type, **kwargs: Any) -> AsyncClient:
        """Initializes the asynchronous Grok client.

        Args:
            async_client_type: The xai-sdk AsyncClient class.
            **kwargs: Additional arguments for the client constructor.

        Returns:
            An instance of the asynchronous xai-sdk AsyncClient.
        """
        return AsyncClient(api_key=self.api_key, **kwargs)

    def _get_capabilities(self) -> Capability:
        """Gets Grok provider capabilities.

        Returns:
            A Capability object indicating supported features.
        """
        return Capability(
            multimodal=True,
            streaming=True,
            tools=True,
            agents=False,
            files=False,
        )

    def _list_models_impl(self) -> list[ModelSummary]:
        """Lists available models from the Grok API.

        Returns:
            A list of ModelSummary objects for available Grok models.
        """
        models = self._client.models.list_language_models()
        model_summaries = []

        for model in models:
            # Create metadata dictionary with additional model information
            metadata: dict[str, Any] = {
                "version": getattr(model, "version", None),
                "input_modalities": getattr(model, "input_modalities", []),
                "output_modalities": getattr(model, "output_modalities", []),
                "max_prompt_length": getattr(model, "max_prompt_length", None),
                "system_fingerprint": getattr(model, "system_fingerprint", None),
                "prompt_text_token_price": getattr(model, "prompt_text_token_price", None),
                "completion_text_token_price": getattr(model, "completion_text_token_price", None),
                "prompt_image_token_price": getattr(model, "prompt_image_token_price", None),
                "cached_prompt_token_price": getattr(model, "cached_prompt_token_price", None),
                "search_price": getattr(model, "search_price", None),
            }

            # Filter out None values from metadata
            metadata = {k: v for k, v in metadata.items() if v is not None}

            model_summary = ModelSummary(
                name=model.name,
                id=model.name,
                created_at=getattr(model.created, "seconds", None)
                if hasattr(model, "created")
                else None,
                metadata=metadata,
                provider="grok",
            )
            model_summaries.append(model_summary)

            # Add aliases as separate model entries
            if hasattr(model, "aliases") and model.aliases:
                for alias in model.aliases:
                    alias_summary = ModelSummary(
                        name=alias,
                        id=alias,
                        created_at=getattr(model.created, "seconds", None)
                        if hasattr(model, "created")
                        else None,
                        metadata={**metadata, "canonical_name": model.name},
                        provider="grok",
                    )
                    model_summaries.append(alias_summary)

        return model_summaries

    @staticmethod
    def _stream(chat: Any) -> Generator[ChimericStreamChunk[Chunk], None, None]:
        """Processes a streaming chat response.

        Args:
            chat: The chat instance from xai-sdk.

        Yields:
            ChimericStreamChunk containing the processed stream data.
        """
        accumulated_content = ""

        for _response, chunk in chat.stream():
            delta = chunk.content if hasattr(chunk, "content") else ""
            accumulated_content += delta

            yield ChimericStreamChunk(
                native=chunk,
                common=StreamChunk(
                    content=accumulated_content,
                    delta=delta,
                    metadata=getattr(chunk, "metadata", {}),
                ),
            )

    @staticmethod
    async def _astream(chat: Any) -> AsyncGenerator[ChimericStreamChunk[Chunk], None]:
        """Processes an async streaming chat response.

        Args:
            chat: The chat instance from xai-sdk.

        Yields:
            ChimericStreamChunk containing the processed stream data.
        """
        accumulated_content = ""

        async for _response, chunk in chat.stream():
            delta = chunk.content if hasattr(chunk, "content") else ""
            accumulated_content += delta

            yield ChimericStreamChunk(
                native=chunk,
                common=StreamChunk(
                    content=accumulated_content,
                    delta=delta,
                    metadata=getattr(chunk, "metadata", {}),
                ),
            )

    @staticmethod
    def _create_chimeric_response(
        response: Response, tool_calls: list[dict[str, Any]]
    ) -> ChimericCompletionResponse[Response]:
        """Creates a ChimericCompletionResponse from a native xai-sdk response.

        Args:
            response: The xai-sdk response object.
            tool_calls: A list of tool calls made and executed during the request.

        Returns:
            A ChimericCompletionResponse wrapping the native response.
        """
        content = response.content or ""

        # Extract usage information if available
        usage_data = getattr(response, "usage", None)
        if usage_data:
            if isinstance(usage_data, dict):
                usage = Usage(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                )
            else:
                usage = Usage(
                    prompt_tokens=getattr(usage_data, "prompt_tokens", 0),
                    completion_tokens=getattr(usage_data, "completion_tokens", 0),
                    total_tokens=getattr(usage_data, "total_tokens", 0),
                )
        else:
            usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        metadata = {"tool_calls": tool_calls} if tool_calls else {}

        return ChimericCompletionResponse(
            native=response,
            common=CompletionResponse(
                content=content,
                usage=usage,
                model=getattr(response, "model", None),
                metadata=metadata,
            ),
        )

    def _encode_tools(self, tools: Tools = None) -> Tools:
        """Encodes tools into a format suitable for the Grok API.

        Args:
            tools: A list of Tool objects or dictionaries.

        Returns:
            A list of tool dictionaries formatted for the Grok API, or None.
        """
        if not tools:
            return None

        encoded_tools = []
        for tool_obj in tools:
            if isinstance(tool_obj, Tool):
                # Convert our internal Tool object to dictionary format
                encoded_tools.append(
                    {
                        "name": tool_obj.name,
                        "description": tool_obj.description,
                        "parameters": tool_obj.parameters.model_dump()
                        if tool_obj.parameters
                        else {},
                    }
                )
            else:
                # Assume it's already properly formatted
                encoded_tools.append(tool_obj)
        return encoded_tools

    def _process_function_call(self, call: Any) -> dict[str, Any]:
        """Executes a function call from the model and returns metadata.

        Args:
            call: The tool call object from the xai-sdk response.

        Returns:
            A dictionary containing the tool call metadata and result.

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

    @staticmethod
    def _convert_messages(messages: Input) -> list[Any]:
        """Converts input messages to the format expected by xai-sdk.

        Args:
            messages: The input messages to convert.

        Returns:
            A list of message objects formatted for xai-sdk.
        """
        if isinstance(messages, str):
            return [user(messages)]

        if not hasattr(messages, "__iter__") or isinstance(messages, str):
            return [user(str(messages))]

        converted = []
        for msg in messages:
            if isinstance(msg, str):
                converted.append(user(msg))
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                role_mapping = {
                    "user": user,
                    "system": system,
                    "assistant": assistant,
                }
                message_func = role_mapping.get(role, user)
                converted.append(message_func(content))
            else:
                converted.append(user(str(msg)))
        return converted

    def _handle_function_tool_calls(
        self, response: Response, chat: Any
    ) -> tuple[list[dict[str, Any]], Response]:
        """Processes tool calls from a response and executes them.

        Args:
            response: The xai-sdk response containing potential tool calls.
            chat: The chat instance to append tool results to.

        Returns:
            A tuple containing (tool_calls_metadata, final_response).
        """
        if not hasattr(response, "tool_calls") or not response.tool_calls:
            return [], response

        # Append the assistant response with tool calls
        chat.append(response)
        tool_calls_metadata = []

        for tool_call in response.tool_calls:
            tool_call_info = self._process_function_call(tool_call)
            tool_calls_metadata.append(tool_call_info)

            # Add the tool result to chat using tool_result
            chat.append(tool_result(tool_call_info["result"]))

        # Get final response after tool calls
        final_response = chat.sample()
        return tool_calls_metadata, final_response

    async def _handle_function_tool_calls_async(
        self, response: Response, chat: Any
    ) -> tuple[list[dict[str, Any]], Response]:
        """Processes tool calls from a response and executes them asynchronously.

        Args:
            response: The xai-sdk response containing potential tool calls.
            chat: The async chat instance to append tool results to.

        Returns:
            A tuple containing (tool_calls_metadata, final_response).
        """
        if not hasattr(response, "tool_calls") or not response.tool_calls:
            return [], response

        # Append the assistant response with tool calls
        chat.append(response)
        tool_calls_metadata = []

        for tool_call in response.tool_calls:
            tool_call_info = self._process_function_call(tool_call)
            tool_calls_metadata.append(tool_call_info)

            # Add the tool result to chat using tool_result
            chat.append(tool_result(tool_call_info["result"]))

        # Get final response after tool calls
        final_response = await chat.sample()
        return tool_calls_metadata, final_response

    def _create_chat_params(
        self, messages: Input, model: str, tools: Tools, **kwargs: Any
    ) -> tuple[Any, list[Any]]:
        """Creates chat parameters for the Grok API call.

        Args:
            messages: Input messages.
            model: Model to use.
            tools: Tools to include (already encoded by base class).
            **kwargs: Additional API parameters.

        Returns:
            A tuple of (chat_instance, converted_messages).
        """
        converted_messages = self._convert_messages(messages)

        # Convert dictionary tools to xai-sdk tool objects
        xai_tools = None
        if tools:
            xai_tools = []
            for tool_dict in tools:
                xai_tools.append(
                    tool(
                        name=tool_dict["name"],
                        description=tool_dict["description"],
                        parameters=tool_dict.get("parameters", {}),
                    )
                )

        chat = self._client.chat.create(
            model=model,
            tools=xai_tools,
            tool_choice="auto" if xai_tools else None,
            **kwargs,
        )

        return chat, converted_messages

    def _create_async_chat_params(
        self, messages: Input, model: str, tools: Tools, **kwargs: Any
    ) -> tuple[Any, list[Any]]:
        """Creates async chat parameters for the Grok API call.

        Args:
            messages: Input messages.
            model: Model to use.
            tools: Tools to include (already encoded by base class).
            **kwargs: Additional API parameters.

        Returns:
            A tuple of (async_chat_instance, converted_messages).
        """
        converted_messages = self._convert_messages(messages)

        # Convert dictionary tools to xai-sdk tool objects
        xai_tools = None
        if tools:
            xai_tools = []
            for tool_dict in tools:
                xai_tools.append(
                    tool(
                        name=tool_dict["name"],
                        description=tool_dict["description"],
                        parameters=tool_dict.get("parameters", {}),
                    )
                )

        chat = self._async_client.chat.create(
            model=model,
            tools=xai_tools,
            tool_choice="auto" if xai_tools else None,
            **kwargs,
        )

        return chat, converted_messages

    def _chat_completion_impl(
        self,
        messages: Input,
        model: str,
        stream: bool = False,
        tools: Tools = None,
        **kwargs: Any,
    ) -> ChimericCompletionResponse[Response] | Generator[ChimericStreamChunk[Chunk], None, None]:
        """Sends a synchronous chat completion request to the Grok API.

        Args:
            messages: The input messages for the chat completion.
            model: The model to use for the completion.
            stream: Whether to return a streaming response.
            tools: A list of tools (already encoded by base class).
            **kwargs: Additional arguments for the chat completion.

        Returns:
            A ChimericCompletionResponse or a generator of ChimericStreamChunk.
        """
        chat, converted_messages = self._create_chat_params(messages, model, tools, **kwargs)

        # Append messages to the chat
        for message in converted_messages:
            chat.append(message)

        if stream:
            return self._stream(chat)

        response = chat.sample()
        tool_calls_metadata, final_response = self._handle_function_tool_calls(response, chat)

        return self._create_chimeric_response(final_response, tool_calls_metadata)

    async def _achat_completion_impl(
        self,
        messages: Input,
        model: str,
        stream: bool = False,
        tools: Tools = None,
        **kwargs: Any,
    ) -> ChimericCompletionResponse[Response] | AsyncGenerator[ChimericStreamChunk[Chunk], None]:
        """Sends an asynchronous chat completion request to the Grok API.

        Args:
            messages: The input messages for the chat completion.
            model: The model to use for the completion.
            stream: Whether to return a streaming response.
            tools: A list of tools (already encoded by base class).
            **kwargs: Additional arguments for the chat completion.

        Returns:
            A ChimericCompletionResponse or an async generator of ChimericStreamChunk.
        """
        chat, converted_messages = self._create_async_chat_params(messages, model, tools, **kwargs)

        # Append messages to the chat
        for message in converted_messages:
            chat.append(message)

        if stream:
            return self._astream(chat)

        response = await chat.sample()
        tool_calls_metadata, final_response = await self._handle_function_tool_calls_async(
            response, chat
        )

        return self._create_chimeric_response(final_response, tool_calls_metadata)

    def _upload_file(self, **kwargs: Any) -> ChimericFileUploadResponse[Any]:
        """Grok does not support file uploads.

        Args:
            **kwargs: Not used.

        Raises:
            NotImplementedError: Grok does not support file uploads.
        """
        raise NotImplementedError("Grok does not support file uploads")
