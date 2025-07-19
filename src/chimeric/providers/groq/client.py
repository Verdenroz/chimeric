import os
from typing import Any

from groq import NOT_GIVEN, AsyncGroq, Groq
from groq.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
)
import httpx

from chimeric.base import ChimericAsyncClient, ChimericClient
from chimeric.exceptions import ProviderError
from chimeric.types import (
    Capability,
    ChimericFileUploadResponse,
    ChimericStreamChunk,
    FileUploadResponse,
    Message,
    ModelSummary,
    Tool,
    ToolCall,
    ToolExecutionResult,
    Usage,
)
from chimeric.utils import StreamProcessor, create_stream_chunk


class GroqClient(ChimericClient[Groq, ChatCompletion, ChatCompletionChunk, Any]):
    """Synchronous Groq Client for interacting with Groq services.

    This client provides a unified interface for synchronous interactions with
    Groq's API via the `groq` library. It returns `chimeric` response objects
    that wrap the native Groq responses and provides comprehensive tool calling
    support for both streaming and non-streaming operations.

    The client supports:
        - Text generation with various Groq models
        - Function/tool calling with automatic execution
        - Streaming responses with real-time tool call handling
        - File uploads for batch processing
        - Model listing and metadata retrieval

    Example:
        ```python
        from chimeric.providers.groq import GroqClient
        from chimeric.tools import ToolManager

        tool_manager = ToolManager()
        client = GroqClient(api_key="your-api-key", tool_manager=tool_manager)

        response = client.chat_completion(
            messages="Hello, how are you?",
            model="llama3-8b-8192"
        )
        print(response.common.content)
        ```

    Attributes:
        api_key (str): The Groq API key for authentication.
        tool_manager (ToolManager): Manager for handling tool registration and execution.
    """

    def __init__(self, api_key: str, tool_manager, **kwargs: Any) -> None:
        """Initialize the synchronous Groq client.

        Args:
            api_key: The Groq API key for authentication.
            tool_manager: The tool manager instance for handling function calls.
            **kwargs: Additional keyword arguments to pass to the Groq client
                constructor, such as base_url, timeout, etc.

        Raises:
            ValueError: If api_key is None or empty.
            ProviderError: If client initialization fails.
        """
        self._provider_name = "Groq"
        super().__init__(api_key=api_key, tool_manager=tool_manager, **kwargs)

    def _get_client_type(self) -> type:
        """Get the synchronous Groq client class type.

        Returns:
            The Groq client class from the groq library.
        """
        return Groq

    def _init_client(self, client_type: type, **kwargs: Any) -> Groq:
        """Initialize the synchronous Groq client instance.

        Args:
            client_type: The Groq client class to instantiate.
            **kwargs: Additional keyword arguments for client initialization,
                such as base_url, timeout, max_retries, etc.

        Returns:
            Configured synchronous Groq client instance.

        Raises:
            ProviderError: If client initialization fails due to invalid
                credentials or configuration.
        """
        return Groq(api_key=self.api_key, **kwargs)

    def _get_capabilities(self) -> Capability:
        """Get supported features for the Groq provider.

        Returns:
            Capability object indicating which features are supported:
                - multimodal: True (supports text and image inputs)
                - streaming: True (supports real-time streaming responses)
                - tools: True (supports function calling)
                - agents: False (agent workflows not currently supported)
                - files: True (supports file uploads for batch processing)
        """
        return Capability(multimodal=True, streaming=True, tools=True, agents=False, files=True)

    def _list_models_impl(self) -> list[ModelSummary]:
        """List available models from the Groq API.

        Returns:
            List of ModelSummary objects containing model metadata from the API.
            Each summary includes id, name, owner, and creation timestamp.

        Raises:
            ProviderError: If the API request fails or returns invalid data.
        """
        return [
            ModelSummary(
                id=model.id, name=model.id, owned_by=model.owned_by, created_at=model.created
            )
            for model in self.client.models.list().data
        ]

    def _messages_to_provider_format(self, messages: list[Message]) -> Any:
        """Convert standardized messages to Groq's format.

        Args:
            messages: List of standardized Message objects with role and content.

        Returns:
            List of message dictionaries formatted for the Groq API. Each message
            contains 'role' and 'content' fields compatible with Groq's chat format.

        Note:
            Groq uses the same message format as OpenAI, so this is a straightforward
            conversion from Message objects to dictionaries.
        """
        return [msg.model_dump(exclude_none=True) for msg in messages]

    def _tools_to_provider_format(self, tools: list[Tool]) -> Any:
        """Convert standardized tools to Groq's format.

        Args:
            tools: List of standardized Tool objects containing function definitions.

        Returns:
            List of tool dictionaries formatted for the Groq API. Each tool follows
            the OpenAI function calling format with 'type' and 'function' fields.

        Example:
            Input Tool with name="get_weather" becomes:
            ```json
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {"type": "object", "properties": {...}}
                }
            }
            ```
        """
        encoded_tools = []
        for tool in tools:
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
        return encoded_tools

    def _make_provider_request(
        self,
        messages: Any,
        model: str,
        stream: bool,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Make the actual API request to Groq.

        Args:
            messages: Messages in Groq's format (list of message dictionaries).
            model: Model identifier (e.g., "llama3-8b-8192", "mixtral-8x7b-32768").
            stream: Whether to stream the response token by token.
            tools: Tools in Groq's format, or None to disable function calling.
            **kwargs: Additional parameters passed to the API request, such as
                temperature, max_tokens, top_p, etc.

        Returns:
            Raw response from Groq's API. Either a ChatCompletion object for
            non-streaming requests or a Stream object for streaming requests.

        Raises:
            ProviderError: If the API request fails due to authentication,
                rate limiting, model unavailability, or other API errors.
        """
        tools_param = NOT_GIVEN if tools is None else tools

        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            tools=tools_param,
            **kwargs,
        )

    def _process_provider_stream_event(
        self, event: ChatCompletionChunk, processor: StreamProcessor
    ) -> ChimericStreamChunk[Any] | None:
        """Processes a Groq stream event using the standardized processor."""
        if not event.choices:
            return None

        choice = event.choices[0]
        delta = choice.delta
        content = getattr(delta, "content", "") or ""
        finish_reason = choice.finish_reason

        # Handle content delta
        if content:
            return create_stream_chunk(
                native_event=event, processor=processor, content_delta=content
            )

        # Handle tool calls in delta
        if hasattr(delta, "tool_calls") and delta.tool_calls:
            for tool_call_delta in delta.tool_calls:
                tool_call_id = (
                    tool_call_delta.id or f"tool_call_{getattr(tool_call_delta, 'index', 0)}"
                )

                if tool_call_delta.function and tool_call_delta.function.name:
                    processor.process_tool_call_start(tool_call_id, tool_call_delta.function.name)

                if tool_call_delta.function and tool_call_delta.function.arguments:
                    processor.process_tool_call_delta(
                        tool_call_id, tool_call_delta.function.arguments
                    )

        # Handle completion
        if finish_reason:
            # Mark any streaming tool calls as complete
            for call_id in processor.state.tool_calls:
                processor.process_tool_call_complete(call_id)

            return create_stream_chunk(
                native_event=event, processor=processor, finish_reason=finish_reason
            )

        return None

    def _extract_usage_from_response(self, response: ChatCompletion) -> Usage:
        """Extracts usage information from Groq response."""
        if not response.usage:
            return Usage()

        return Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

    def _extract_content_from_response(self, response: ChatCompletion) -> str | list[Any]:
        """Extracts content from Groq response."""
        if response.choices:
            message = response.choices[0].message
            return message.content or ""
        return ""

    def _extract_tool_calls_from_response(self, response: ChatCompletion) -> list[ToolCall] | None:
        """Extracts tool calls from Groq response."""
        if not response.choices:
            return None

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            return None

        return [
            ToolCall(
                call_id=call.id,
                name=call.function.name,
                arguments=call.function.arguments,
            )
            for call in tool_calls
        ]

    def _update_messages_with_tool_calls(
        self,
        messages: list[Any],
        assistant_response: Any,
        tool_calls: list[ToolCall],
        tool_results: list[ToolExecutionResult],
    ) -> list[Any]:
        """Updates message history with assistant response and tool results.

        For Groq, we need to:
        1. Add the assistant message with tool_calls
        2. Add tool result messages for each tool call
        """
        updated_messages = list(messages)

        # Build assistant message with tool calls
        assistant_tool_calls = []
        for tool_call in tool_calls:
            assistant_tool_calls.append(
                {
                    "id": tool_call.call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                }
            )

        # Add assistant message with tool calls
        updated_messages.append(
            {
                "role": "assistant",
                "tool_calls": assistant_tool_calls,
            }
        )

        # Add tool result messages
        for result in tool_results:
            updated_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": result.result if not result.is_error else f"Error: {result.error}",
                }
            )

        return updated_messages

    def _upload_file(self, **kwargs: Any) -> ChimericFileUploadResponse[Any]:
        """Upload a file to Groq for batch processing using httpx.

        Args:
            **kwargs: Provider-specific arguments for file upload including:
                - file_path (str): Path to the file to upload
                - file_object: File-like object to upload (alternative to file_path)
                - filename (str): Name for the uploaded file (optional, defaults
                    to basename of file_path)
                - purpose (str): Purpose of the file upload (defaults to "batch")

        Returns:
            ChimericFileUploadResponse containing both the native response from
            Groq's API and standardized file upload information.

        Raises:
            ValueError: If neither file_path nor file_object is provided.
            ProviderError: If the file upload fails due to network issues,
                authentication problems, or API errors.

        Note:
            This method uses httpx directly to upload files to Groq's file API
            since the groq library may not expose file upload functionality.
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


class GroqAsyncClient(ChimericAsyncClient[AsyncGroq, ChatCompletion, ChatCompletionChunk, Any]):
    """Asynchronous Groq Client for interacting with Groq services.

    This client provides a unified interface for asynchronous interactions with
    Groq's API via the `groq` library. It returns `chimeric` response objects
    that wrap the native Groq responses and provides comprehensive tool calling
    support for both streaming and non-streaming operations.

    The async client supports all the same features as the synchronous client:
        - Asynchronous text generation with various Groq models
        - Asynchronous function/tool calling with automatic execution
        - Asynchronous streaming responses with real-time tool call handling
        - Asynchronous file uploads for batch processing
        - Model listing and metadata retrieval

    Example:
        ```python
        import asyncio
        from chimeric.providers.groq import GroqAsyncClient
        from chimeric.tools import ToolManager

        async def main():
            tool_manager = ToolManager()
            client = GroqAsyncClient(api_key="your-api-key", tool_manager=tool_manager)

            response = await client.chat_completion(
                messages="Hello, how are you?",
                model="llama3-8b-8192"
            )
            print(response.common.content)

        asyncio.run(main())
        ```

    Attributes:
        api_key (str): The Groq API key for authentication.
        tool_manager (ToolManager): Manager for handling tool registration and execution.
    """

    def __init__(self, api_key: str, tool_manager, **kwargs: Any) -> None:
        """Initialize the asynchronous Groq client.

        Args:
            api_key: The Groq API key for authentication.
            tool_manager: The tool manager instance for handling function calls.
            **kwargs: Additional keyword arguments to pass to the AsyncGroq client
                constructor, such as base_url, timeout, etc.

        Raises:
            ValueError: If api_key is None or empty.
            ProviderError: If client initialization fails.
        """
        self._provider_name = "Groq"
        super().__init__(api_key=api_key, tool_manager=tool_manager, **kwargs)

    def _get_async_client_type(self) -> type:
        """Get the asynchronous Groq client class type.

        Returns:
            The AsyncGroq client class from the groq library.
        """
        return AsyncGroq

    def _init_async_client(self, async_client_type: type, **kwargs: Any) -> AsyncGroq:
        """Initialize the asynchronous Groq client instance.

        Args:
            async_client_type: The AsyncGroq client class to instantiate.
            **kwargs: Additional keyword arguments for client initialization,
                such as base_url, timeout, max_retries, etc.

        Returns:
            Configured asynchronous Groq client instance.

        Raises:
            ProviderError: If client initialization fails due to invalid
                credentials or configuration.
        """
        return AsyncGroq(api_key=self.api_key, **kwargs)

    def _get_capabilities(self) -> Capability:
        """Get supported features for the Groq provider.

        Returns:
            Capability object indicating which features are supported:
                - multimodal: True (supports text and image inputs)
                - streaming: True (supports real-time streaming responses)
                - tools: True (supports function calling)
                - agents: False (agent workflows not currently supported)
                - files: True (supports file uploads for batch processing)
        """
        return Capability(multimodal=True, streaming=True, tools=True, agents=False, files=True)

    async def _list_models_impl(self) -> list[ModelSummary]:
        """List available models from the Groq API asynchronously.

        Returns:
            List of ModelSummary objects containing model metadata from the API.
            Each summary includes id, name, owner, and creation timestamp.

        Raises:
            ProviderError: If the API request fails or returns invalid data.
        """
        models = await self.async_client.models.list()
        return [
            ModelSummary(
                id=model.id, name=model.id, owned_by=model.owned_by, created_at=model.created
            )
            for model in models.data
        ]

    def _messages_to_provider_format(self, messages: list[Message]) -> Any:
        """Converts standardized messages to Groq format."""
        return [msg.model_dump(exclude_none=True) for msg in messages]

    def _tools_to_provider_format(self, tools: list[Tool]) -> Any:
        """Converts standardized tools to Groq format."""
        encoded_tools = []
        for tool in tools:
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
        return encoded_tools

    async def _make_async_provider_request(
        self,
        messages: Any,
        model: str,
        stream: bool,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Make the actual async API request to Groq.

        Args:
            messages: Messages in Groq's format (list of message dictionaries).
            model: Model identifier (e.g., "llama3-8b-8192", "mixtral-8x7b-32768").
            stream: Whether to stream the response token by token.
            tools: Tools in Groq's format, or None to disable function calling.
            **kwargs: Additional parameters passed to the API request, such as
                temperature, max_tokens, top_p, etc.

        Returns:
            Raw response from Groq's API. Either a ChatCompletion object for
            non-streaming requests or an AsyncStream object for streaming requests.

        Raises:
            ProviderError: If the API request fails due to authentication,
                rate limiting, model unavailability, or other API errors.
        """
        tools_param = NOT_GIVEN if tools is None else tools

        return await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            tools=tools_param,
            **kwargs,
        )

    def _process_provider_stream_event(
        self, event: ChatCompletionChunk, processor: StreamProcessor
    ) -> ChimericStreamChunk[Any] | None:
        """Processes a Groq stream event using the standardized processor."""
        if not event.choices:
            return None

        choice = event.choices[0]
        delta = choice.delta
        content = getattr(delta, "content", "") or ""
        finish_reason = choice.finish_reason

        # Handle content delta
        if content:
            return create_stream_chunk(
                native_event=event, processor=processor, content_delta=content
            )

        # Handle tool calls in delta
        if hasattr(delta, "tool_calls") and delta.tool_calls:
            for tool_call_delta in delta.tool_calls:
                tool_call_id = (
                    tool_call_delta.id or f"tool_call_{getattr(tool_call_delta, 'index', 0)}"
                )

                if tool_call_delta.function and tool_call_delta.function.name:
                    processor.process_tool_call_start(tool_call_id, tool_call_delta.function.name)

                if tool_call_delta.function and tool_call_delta.function.arguments:
                    processor.process_tool_call_delta(
                        tool_call_id, tool_call_delta.function.arguments
                    )

        # Handle completion
        if finish_reason:
            # Mark any streaming tool calls as complete
            for call_id in processor.state.tool_calls:
                processor.process_tool_call_complete(call_id)

            return create_stream_chunk(
                native_event=event, processor=processor, finish_reason=finish_reason
            )

        return None

    def _extract_usage_from_response(self, response: ChatCompletion) -> Usage:
        """Extracts usage information from Groq response."""
        if not response.usage:
            return Usage()

        return Usage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

    def _extract_content_from_response(self, response: ChatCompletion) -> str | list[Any]:
        """Extracts content from Groq response."""
        if response.choices:
            message = response.choices[0].message
            return message.content or ""
        return ""

    def _extract_tool_calls_from_response(self, response: ChatCompletion) -> list[ToolCall] | None:
        """Extracts tool calls from Groq response."""
        if not response.choices:
            return None

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            return None

        return [
            ToolCall(
                call_id=call.id,
                name=call.function.name,
                arguments=call.function.arguments,
            )
            for call in tool_calls
        ]

    def _update_messages_with_tool_calls(
        self,
        messages: list[Any],
        assistant_response: Any,
        tool_calls: list[ToolCall],
        tool_results: list[ToolExecutionResult],
    ) -> list[Any]:
        """Updates message history with assistant response and tool results.

        For Groq, we need to:
        1. Add the assistant message with tool_calls
        2. Add tool result messages for each tool call
        """
        updated_messages = list(messages)

        # Build assistant message with tool calls
        assistant_tool_calls = []
        for tool_call in tool_calls:
            assistant_tool_calls.append(
                {
                    "id": tool_call.call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                }
            )

        # Add assistant message with tool calls
        updated_messages.append(
            {
                "role": "assistant",
                "tool_calls": assistant_tool_calls,
            }
        )

        # Add tool result messages
        for result in tool_results:
            updated_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": result.result if not result.is_error else f"Error: {result.error}",
                }
            )

        return updated_messages

    async def _upload_file(self, **kwargs: Any) -> ChimericFileUploadResponse[Any]:
        """Upload a file to Groq for batch processing using httpx asynchronously.

        Args:
            **kwargs: Provider-specific arguments for file upload including:
                - file_path (str): Path to the file to upload
                - file_object: File-like object to upload (alternative to file_path)
                - filename (str): Name for the uploaded file (optional, defaults
                    to basename of file_path)
                - purpose (str): Purpose of the file upload (defaults to "batch")

        Returns:
            ChimericFileUploadResponse containing both the native response from
            Groq's API and standardized file upload information.

        Raises:
            ValueError: If neither file_path nor file_object is provided.
            ProviderError: If the file upload fails due to network issues,
                authentication problems, or API errors.

        Note:
            This method uses httpx AsyncClient to upload files to Groq's file API
            since the groq library may not expose async file upload functionality.
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

        async with httpx.AsyncClient() as client:
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
            response = await client.post(url, headers=headers, files=files, data=data)

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
