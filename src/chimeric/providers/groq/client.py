from collections.abc import AsyncGenerator, Generator
import json
import os
from typing import Any, cast

from groq import NOT_GIVEN, AsyncGroq, AsyncStream, Groq, Stream
from groq.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)
import httpx

from chimeric.base import BaseClient
from chimeric.exceptions import ProviderError, ToolRegistrationError
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


class GroqClient(BaseClient[Groq, AsyncGroq, ChatCompletion, ChatCompletionChunk, Any]):
    """Groq Client for interacting with Groq services.

    This client provides a unified interface for synchronous and asynchronous
    interactions with Groq's API via the `groq` library. It returns `chimeric`
    response objects that wrap the native Groq responses.
    """

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        """Initializes the Groq client.

        Args:
            api_key: The Groq API key for authentication.
            **kwargs: Additional keyword arguments to pass to the Groq client constructor.
        """
        self._provider_name = "Groq"
        super().__init__(api_key=api_key, **kwargs)

    def _get_generic_types(self) -> dict[str, type]:
        """Returns the concrete Groq client types for `kwargs` filtering.

        Returns:
            A dictionary mapping client type names to their respective classes.
        """
        return {
            "sync": Groq,
            "async": AsyncGroq,
        }

    def _init_client(self, client_type: type, **kwargs: Any) -> Groq:
        """Initializes the synchronous Groq client.

        Args:
            client_type: The Groq client class.
            **kwargs: Additional arguments for the client constructor.

        Returns:
            An instance of the synchronous Groq client.
        """
        return Groq(api_key=self.api_key, **kwargs)

    def _init_async_client(self, async_client_type: type, **kwargs: Any) -> AsyncGroq:
        """Initializes the asynchronous Groq client.

        Args:
            async_client_type: The AsyncGroq client class.
            **kwargs: Additional arguments for the client constructor.

        Returns:
            An instance of the asynchronous Groq client.
        """
        return AsyncGroq(api_key=self.api_key, **kwargs)

    def _get_capabilities(self) -> Capability:
        """Gets Groq provider capabilities.

        Returns:
            A Capability object indicating supported features.
            Groq supports tools, streaming, multimodal, and files but not agents.
        """
        return Capability(multimodal=True, streaming=True, tools=True, agents=False, files=True)

    def _list_models_impl(self) -> list[ModelSummary]:
        """Lists available models from the Groq API.

        Returns:
            A list of ModelSummary objects for all available models from the API.
        """
        return [
            ModelSummary(
                id=model.id, name=model.id, owned_by=model.owned_by, created_at=model.created
            )
            for model in self.client.models.list().data
        ]

    @staticmethod
    def _process_stream_chunk(
        chunk: ChatCompletionChunk,
        accumulated: str,
    ) -> tuple[str, ChimericStreamChunk[ChatCompletionChunk] | None]:
        """Processes a single chunk from a Groq response stream.

        Args:
            chunk: The chat completion chunk from the Groq API.
            accumulated: The accumulated content from previous chunks.

        Returns:
            A tuple containing the updated accumulated content and an optional
            ChimericStreamChunk to be yielded.
        """
        if not chunk.choices:
            return accumulated, None

        choice = chunk.choices[0]
        delta = choice.delta
        content = getattr(delta, "content", "") or ""
        finish_reason = choice.finish_reason

        if content:
            accumulated += content

        if not content and not finish_reason:
            return accumulated, None

        return accumulated, ChimericStreamChunk(
            native=chunk,
            common=StreamChunk(
                content=accumulated,
                delta=content,
                finish_reason=finish_reason,
                metadata=chunk.model_dump(),
            ),
        )

    def _stream(
        self,
        stream: Stream[ChatCompletionChunk],
    ) -> Generator[ChimericStreamChunk[ChatCompletionChunk], None, None]:
        """Yields processed chunks from a synchronous Groq stream.

        Args:
            stream: The synchronous stream of chat completion chunks.

        Yields:
            ChimericStreamChunk containing the processed chunk data.
        """
        accumulated = ""
        for chunk in stream:
            accumulated, processed_chunk = self._process_stream_chunk(chunk, accumulated)
            if processed_chunk:
                yield processed_chunk

    async def _astream(
        self,
        stream: AsyncStream[ChatCompletionChunk],
    ) -> AsyncGenerator[ChimericStreamChunk[ChatCompletionChunk], None]:
        """Yields processed chunks from an asynchronous Groq stream.

        Args:
            stream: The asynchronous stream of chat completion chunks.

        Yields:
            ChimericStreamChunk containing the processed chunk data.
        """
        accumulated = ""
        async for chunk in stream:
            accumulated, processed_chunk = self._process_stream_chunk(chunk, accumulated)
            if processed_chunk:
                yield processed_chunk

    @staticmethod
    def _create_chimeric_response(
        response: ChatCompletion, tool_calls: list[dict[str, Any]]
    ) -> ChimericCompletionResponse[ChatCompletion]:
        """Creates a ChimericCompletionResponse from a native Groq ChatCompletion.

        Args:
            response: The Groq chat completion response.
            tool_calls: A list of tool calls made and executed during the request.

        Returns:
            A ChimericCompletionResponse wrapping the native response.
        """
        metadata = response.model_dump()
        if tool_calls:
            metadata["tool_calls"] = tool_calls

        # Extract content from the first choice
        content = ""
        if response.choices:
            message = response.choices[0].message
            content = message.content or ""

        return ChimericCompletionResponse(
            native=response,
            common=CompletionResponse(
                content=content,
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
        """Encodes a list of Tool objects into the format expected by the Groq API.

        Args:
            tools: A list of Tool objects or dictionaries.

        Returns:
            A list of tool dictionaries formatted for the Groq API, or None.
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
            call: The tool call object from the Groq response.

        Returns:
            A dictionary containing the tool call ID, name, arguments, and result.

        Raises:
            ToolRegistrationError: If the requested tool is not registered or not callable.
        """
        function = call.function
        tool_name = function.name
        tool_call_info = {
            "call_id": call.id,
            "name": tool_name,
            "arguments": function.arguments,
        }

        tool = self.tool_manager.get_tool(tool_name)
        if not callable(tool.function):
            raise ToolRegistrationError(f"Tool '{tool_name}' is not callable.")

        args = json.loads(function.arguments)
        result = tool.function(**args)
        tool_call_info["result"] = str(result)
        return tool_call_info

    def _handle_function_tool_calls(
        self, response: ChatCompletion, messages: list[Any]
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        """Processes tool calls from a response and updates the message history.

        If the response contains tool calls, this method executes them, appends the
        call and result to the message list, and returns metadata about the calls.

        Args:
            response: The Groq response containing potential tool calls.
            messages: The current list of messages to append to.

        Returns:
            A tuple containing (tool_calls_metadata, updated_messages).
        """
        if not response.choices:
            return [], messages

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            return [], messages

        # Ensure messages is a mutable list of dictionaries for appending
        messages_list = list(messages)
        tool_calls_metadata = []

        # Add the assistant message with tool calls
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
                    for call in tool_calls
                ],
            }
        )

        # Process each tool call and add tool result messages
        for call in tool_calls:
            tool_call_info = self._process_function_call(call)
            tool_calls_metadata.append(tool_call_info)

            # Add the tool result message for the next turn in the conversation
            messages_list.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": tool_call_info["result"],
                }
            )

        return tool_calls_metadata, messages_list

    @staticmethod
    def _normalize_messages(messages: Input) -> list[ChatCompletionMessageParam]:
        """Normalizes the input messages to the format expected by the Groq API.

        Args:
            messages: The input messages, which can be a string or list of message dicts.

        Returns:
            A list of message dictionaries formatted for the Groq API.
        """
        # If messages is a string, convert it to a single user message
        if isinstance(messages, str):
            return [cast("ChatCompletionUserMessageParam", {"role": "user", "content": messages})]

        # If messages is already a list, ensure each message is a dict
        return cast("list[ChatCompletionMessageParam]", list(messages))

    def _chat_completion_impl(
        self,
        messages: Input,
        model: str,
        stream: bool = False,
        tools: Tools = None,
        **kwargs: Any,
    ) -> (
        ChimericCompletionResponse[ChatCompletion]
        | Generator[ChimericStreamChunk[ChatCompletionChunk], None, None]
    ):
        """Sends a synchronous chat completion request to the Groq API.

        This method supports a two-pass mechanism for tool calls. If the model
        responds with a tool call, the tool is executed and a second request
        is made with the tool's result.

        Args:
            messages: The input messages for the chat completion.
            model: The model to use for the completion.
            stream: Whether to return a streaming response.
            tools: A list of tools (already encoded for the API).
            **kwargs: Additional arguments for the completion request.

        Returns:
            A ChimericCompletionResponse for a single response, or a generator
            of ChimericStreamChunk for a streaming response.
        """
        filtered_kwargs = self._filter_kwargs(self._client.chat.completions.create, kwargs)
        tools_param = NOT_GIVEN if tools is None else tools

        # Normalize messages to the format expected by the API
        normalized_messages = self._normalize_messages(messages)

        # Send the initial user request
        response = self._client.chat.completions.create(
            model=model,
            messages=normalized_messages,
            stream=stream,
            tools=tools_param,
            **filtered_kwargs,
        )

        if isinstance(response, Stream):
            # If streaming is requested, return a generator that processes the stream
            return self._stream(response)

        # Check for and handle any tool calls requested by the model
        tool_calls_metadata, updated_messages = self._handle_function_tool_calls(
            response, normalized_messages
        )

        # If tool calls were made, send their results back to the model
        if tool_calls_metadata:
            response = self._client.chat.completions.create(
                model=model, messages=updated_messages, tools=tools_param, **filtered_kwargs
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
        ChimericCompletionResponse[ChatCompletion]
        | AsyncGenerator[ChimericStreamChunk[ChatCompletionChunk], None]
    ):
        """Sends an asynchronous chat completion request to the Groq API.

        This method supports a two-pass mechanism for tool calls. If the model
        responds with a tool call, the tool is executed and a second request
        is made with the tool's result.

        Args:
            messages: The input messages for the chat completion.
            model: The model to use for the completion.
            stream: Whether to return a streaming response.
            tools: A list of tools (already encoded for the API).
            **kwargs: Additional arguments for the completion request.

        Returns:
            A ChimericCompletionResponse for a single response, or an async
            generator of ChimericStreamChunk for a streaming response.
        """
        filtered_kwargs = self._filter_kwargs(self._async_client.chat.completions.create, kwargs)
        tools_param = NOT_GIVEN if tools is None else tools

        # Normalize messages to the format expected by the API
        normalized_messages = self._normalize_messages(messages)

        # Send the initial user request
        response = await self._async_client.chat.completions.create(
            model=model,
            messages=normalized_messages,
            stream=stream,
            tools=tools_param,
            **filtered_kwargs,
        )

        if isinstance(response, AsyncStream):
            # If streaming is requested, return an async generator that processes the stream
            return self._astream(response)

        # Check for and handle any tool calls requested by the model
        tool_calls_metadata, updated_messages = self._handle_function_tool_calls(
            response, normalized_messages
        )

        # If tool calls were made, send their results back to the model
        if tool_calls_metadata:
            response = await self._async_client.chat.completions.create(
                model=model, messages=updated_messages, tools=tools_param, **filtered_kwargs
            )

        return self._create_chimeric_response(response, tool_calls_metadata)

    def _upload_file(self, **kwargs: Any) -> ChimericFileUploadResponse[Any]:
        """Uploads a file to Groq for batch processing using httpx.

        Args:
            **kwargs: Provider-specific arguments for file upload.
                - file_path (str): Path to the file to upload
                - file_object: File-like object to upload (alternative to file_path)
                - filename (str): Name for the uploaded file (optional, defaults to basename of file_path)
                - purpose (str): Purpose of the file upload (defaults to "batch")

        Returns:
            A ChimericFileUploadResponse containing the native response and
            common file upload information.

        Raises:
            ValueError: If neither file_path nor file_object is provided.
            ProviderError: If the file upload fails.
        """
        # Extract file upload parameters from kwargs
        file_path = kwargs.get("file_path")
        file_object = kwargs.get("file_object")
        filename = kwargs.get("filename")
        purpose = kwargs.get("purpose", "batch")

        if not file_path and not file_object:
            raise ValueError("Either 'file_path' or 'file_object' must be provided")

        # Prepare the API request
        url = "https://api.groq.com/openai/v1/files"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"purpose": purpose}

        with httpx.Client() as client:
            if file_path:
                if not filename:
                    filename = os.path.basename(file_path)
                with open(file_path, "rb") as f:
                    file_content = f.read()
            else:
                if not filename:
                    filename = getattr(file_object, "name", "uploaded_file")

                # Get content from the file-like object
                if hasattr(file_object, "read"):
                    content = file_object.read()
                    # Reset the cursor if possible
                    if hasattr(file_object, "seek"):
                        file_object.seek(0)
                    # Ensure content is bytes
                    file_content = (
                        content if isinstance(content, bytes) else content.encode("utf-8")
                    )
                else:
                    # file_object is the content itself
                    file_content = (
                        file_object
                        if isinstance(file_object, bytes)
                        else str(file_object).encode("utf-8")
                    )

            files = {"file": (filename, file_content, "application/octet-stream")}
            response = client.post(url, headers=headers, files=files, data=data)

            # Check response status
            if not response.is_success:
                raise ProviderError(
                    provider=self._provider_name,
                    response_text=response.text,
                    endpoint="file_upload",
                    status_code=response.status_code,
                )

            native_response = response.json()
            common_response = FileUploadResponse(
                file_id=native_response.get("id", ""),
                filename=native_response.get("filename", filename),
                bytes=native_response.get("bytes", 0),
                purpose=native_response.get("purpose", purpose),
                status=native_response.get("status", "uploaded"),
                created_at=native_response.get("created_at", 0),
                metadata=native_response,
            )

            return ChimericFileUploadResponse(native=native_response, common=common_response)
