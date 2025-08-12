from typing import Any

from cerebras.cloud.sdk import AsyncCerebras, Cerebras
from cerebras.cloud.sdk.types.chat.chat_completion import (
    ChatChunkResponse,
    ChatCompletionResponse,
)

from chimeric.base import ChimericAsyncClient, ChimericClient
from chimeric.types import (
    Capability,
    ChimericStreamChunk,
    Message,
    ModelSummary,
    Tool,
    ToolCall,
    ToolExecutionResult,
    ToolParameters,
    Usage,
)
from chimeric.utils import StreamProcessor, create_stream_chunk


class CerebrasClient(ChimericClient[Cerebras, ChatCompletionResponse, ChatChunkResponse]):
    """Synchronous Cerebras Client for interacting with Cerebras Cloud API.

    This client provides a unified interface for synchronous interactions with
    Cerebras's high-performance inference API via the `cerebras-cloud-sdk` library.
    It returns `chimeric` response objects that wrap the native Cerebras responses
    and provides comprehensive tool calling support for both streaming and
    non-streaming operations.

    The client supports:
        - Ultra-fast text generation with Cerebras's optimized models
        - Function/tool calling with automatic execution
        - Streaming responses with real-time tool call handling
        - High-performance inference with specialized hardware acceleration
        - Model listing and metadata retrieval

    Note:
        Cerebras specializes in high-speed inference and currently supports text-only
        models for fast text processing.

    Example:
        ```python
        from chimeric.providers.cerebras import CerebrasClient
        from chimeric.tools import ToolManager

        tool_manager = ToolManager()
        client = CerebrasClient(api_key="your-api-key", tool_manager=tool_manager)

        response = client.chat_completion(
            messages="What's the fastest way to process this data?",
            model="llama3.1-8b"
        )
        print(response.common.content)
        ```

    Attributes:
        api_key (str): The Cerebras API key for authentication.
        tool_manager (ToolManager): Manager for handling tool registration and execution.
    """

    def __init__(self, api_key: str, tool_manager, **kwargs: Any) -> None:
        """Initialize the synchronous Cerebras client.

        Args:
            api_key: The Cerebras API key for authentication.
            tool_manager: The tool manager instance for handling function calls.
            **kwargs: Additional keyword arguments to pass to the Cerebras client
                constructor, such as base_url, timeout, etc.

        Raises:
            ValueError: If api_key is None or empty.
            ProviderError: If client initialization fails.
        """
        self._provider_name = "Cerebras"
        super().__init__(api_key=api_key, tool_manager=tool_manager, **kwargs)

    def _get_client_type(self) -> type:
        """Get the synchronous Cerebras client class type.

        Returns:
            The Cerebras client class from the cerebras-cloud-sdk library.
        """
        return Cerebras

    def _init_client(self, client_type: type, **kwargs: Any) -> Cerebras:
        """Initialize the synchronous Cerebras client instance.

        Args:
            client_type: The Cerebras client class to instantiate.
            **kwargs: Additional keyword arguments for client initialization,
                such as base_url, timeout, max_retries, etc.

        Returns:
            Configured synchronous Cerebras client instance.

        Raises:
            ProviderError: If client initialization fails due to invalid
                credentials or configuration.
        """
        return Cerebras(api_key=self.api_key, **kwargs)

    def _get_capabilities(self) -> Capability:
        """Get supported features for the Cerebras provider.

        Returns:
            Capability object indicating which features are supported:
                - streaming: True (supports real-time streaming responses)
                - tools: True (supports function calling)

        Note:
            Cerebras focuses on high-performance text inference for fast
            text-only processing.
        """
        return Capability(streaming=True, tools=True)

    def _list_models_impl(self) -> list[ModelSummary]:
        """List available models from the Cerebras API.

        Returns:
            List of ModelSummary objects containing model metadata from the API.
            Each summary includes id, name, owner, and creation timestamp.

        Raises:
            ProviderError: If the API request fails or returns invalid data.

        Note:
            Cerebras provides high-performance versions of popular open-source
            models optimized for their specialized hardware.
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

    def _messages_to_provider_format(self, messages: list[Message]) -> Any:
        """Convert standardized messages to Cerebras's format.

        Args:
            messages: List of standardized Message objects with role and content.

        Returns:
            List of message dictionaries formatted for the Cerebras API. Each message
            contains 'role' and 'content' fields compatible with Cerebras's chat format.

        Note:
            Cerebras uses the OpenAI-compatible message format, so this is a
            straightforward conversion from Message objects to dictionaries.
        """
        return [msg.model_dump(exclude_none=True) for msg in messages]

    def _tools_to_provider_format(self, tools: list[Tool]) -> Any:
        """Convert standardized tools to Cerebras's format.

        Args:
            tools: List of standardized Tool objects containing function definitions.

        Returns:
            List of tool dictionaries formatted for the Cerebras API. Each tool follows
            the OpenAI function calling format with Cerebras-specific optimizations.

        Note:
            Cerebras requires the 'strict' parameter at the function level rather than
            in the parameters schema, and expects strict mode to be enabled for
            optimal performance with their optimized inference.

        Example:
            Input Tool with name="calculate" becomes:
            ```json
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "strict": true,
                    "description": "Perform calculations",
                    "parameters": {"type": "object", "properties": {...}}
                }
            }
            ```
        """
        encoded_tools = []
        for tool in tools:
            # Get parameters and remove 'strict' from the parameters schema since
            # Cerebras expects it only in the function object
            parameters = (
                tool.parameters.model_dump() if isinstance(tool.parameters, ToolParameters) else {}
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
        return encoded_tools

    def _make_provider_request(
        self,
        messages: Any,
        model: str,
        stream: bool,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Make the actual API request to Cerebras.

        Args:
            messages: Messages in Cerebras's format (list of message dictionaries).
            model: Model identifier (e.g., "llama3.1-8b", "llama3.1-70b").
            stream: Whether to stream the response token by token.
            tools: Tools in Cerebras's format, or None to disable function calling.
            **kwargs: Additional parameters passed to the API request, such as
                temperature, max_tokens, top_p, etc.

        Returns:
            Raw response from Cerebras's API. Either a ChatCompletionResponse object
            for non-streaming requests or a stream object for streaming requests.

        Raises:
            ProviderError: If the API request fails due to authentication,
                rate limiting, model unavailability, or other API errors.

        Note:
            Cerebras provides ultra-fast inference with their specialized hardware,
            significantly reducing response times compared to traditional GPU inference.
        """
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            tools=tools,
            **kwargs,
        )

    def _process_provider_stream_event(
        self, event: ChatChunkResponse, processor: StreamProcessor
    ) -> ChimericStreamChunk[Any] | None:
        """Process a streaming event from Cerebras API into standardized format.

        Args:
            event: Raw streaming event from the Cerebras API containing delta content,
                tool calls, or completion signals.
            processor: StreamProcessor instance that manages streaming state and
                accumulates content across multiple events.

        Returns:
            ChimericStreamChunk object containing processed content delta, tool call
            information, or completion status. Returns None if the event contains
            no processable content.

        Note:
            Cerebras streaming events follow the OpenAI format with delta content
            and tool_calls arrays. This method handles content accumulation and
            tool call state management automatically.
        """
        if event.choices and event.choices[0].delta.content:
            delta = event.choices[0].delta.content
            return create_stream_chunk(native_event=event, processor=processor, content_delta=delta)

        # Handle tool calls in streaming
        if event.choices and event.choices[0].delta.tool_calls:
            for tool_call_delta in event.choices[0].delta.tool_calls:
                call_id = tool_call_delta.id or f"tool_call_{getattr(tool_call_delta, 'index', 0)}"
                if tool_call_delta.function and tool_call_delta.function.name:
                    processor.process_tool_call_start(call_id, tool_call_delta.function.name)
                if tool_call_delta.function and tool_call_delta.function.arguments:
                    processor.process_tool_call_delta(call_id, tool_call_delta.function.arguments)

        # Handle completion
        if event.choices and event.choices[0].finish_reason:
            # Mark any streaming tool calls as complete
            for call_id in processor.state.tool_calls:
                processor.process_tool_call_complete(call_id)

            return create_stream_chunk(
                native_event=event,
                processor=processor,
                finish_reason=event.choices[0].finish_reason,
            )

        return None

    def _extract_usage_from_response(self, response: ChatCompletionResponse) -> Usage:
        """Extract token usage statistics from Cerebras API response.

        Args:
            response: ChatCompletionResponse from Cerebras API containing usage metadata.

        Returns:
            Usage object with prompt_tokens, completion_tokens, and total_tokens fields.
            Returns empty Usage object if no usage information is available.

        Note:
            Cerebras provides detailed token usage information similar to OpenAI,
            enabling accurate cost tracking and usage monitoring.
        """
        if not response.usage:
            return Usage()

        return Usage(
            prompt_tokens=response.usage.prompt_tokens or 0,
            completion_tokens=response.usage.completion_tokens or 0,
            total_tokens=response.usage.total_tokens or 0,
        )

    def _extract_content_from_response(self, response: ChatCompletionResponse) -> str | list[Any]:
        """Extract text content from Cerebras API response.

        Args:
            response: ChatCompletionResponse from Cerebras API containing message content.

        Returns:
            String containing the generated text content from the first choice.
            Returns empty string if no content is available.

        Note:
            Cerebras responses follow the OpenAI format with choices[0].message.content
            containing the generated text.
        """
        choice = response.choices[0] if response.choices else None
        return (
            choice.message.content if choice and choice.message and choice.message.content else ""
        )

    def _extract_tool_calls_from_response(
        self, response: ChatCompletionResponse
    ) -> list[ToolCall] | None:
        """Extract function tool calls from Cerebras API response.

        Args:
            response: ChatCompletionResponse from Cerebras API that may contain tool calls.

        Returns:
            List of ToolCall objects representing functions the model wants to execute,
            or None if no tool calls are present. Each ToolCall includes call_id,
            function name, and JSON-encoded arguments.

        Note:
            Cerebras uses the OpenAI tool calling format with tool_calls arrays
            containing function specifications and arguments.
        """
        choice = response.choices[0] if response.choices else None
        if not choice or not choice.message.tool_calls:
            return None

        return [
            ToolCall(
                call_id=call.id,
                name=call.function.name,
                arguments=call.function.arguments,
            )
            for call in choice.message.tool_calls
        ]

    def _update_messages_with_tool_calls(
        self,
        messages: list[Any],
        assistant_response: Any,
        tool_calls: list[ToolCall],
        tool_results: list[ToolExecutionResult],
    ) -> list[Any]:
        """Update message history with assistant response and tool results.

        This method formats the conversation history to include the assistant's
        tool calls and their results, following Cerebras's message format requirements.

        Args:
            messages: Current list of messages in the conversation.
            assistant_response: The assistant's response that contained tool calls
                (not directly used but maintained for interface compatibility).
            tool_calls: List of ToolCall objects representing functions called
                by the assistant.
            tool_results: List of ToolExecutionResult objects containing the
                results of executing the tool calls.

        Returns:
            Updated list of messages including:
                1. Original conversation messages
                2. Assistant message with tool_calls array
                3. Tool result messages for each executed function

        Note:
            Cerebras uses the OpenAI message format where tool calls are represented
            as an assistant message with a 'tool_calls' field, followed by separate
            'tool' role messages containing the results.
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


class CerebrasAsyncClient(
    ChimericAsyncClient[AsyncCerebras, ChatCompletionResponse, ChatChunkResponse]
):
    """Asynchronous Cerebras Client for interacting with Cerebras Cloud API.

    This client provides a unified interface for asynchronous interactions with
    Cerebras's high-performance inference API via the `cerebras-cloud-sdk` library.
    It returns `chimeric` response objects that wrap the native Cerebras responses
    and provides comprehensive tool calling support for both streaming and
    non-streaming operations.

    The async client supports all the same features as the synchronous client:
        - Asynchronous ultra-fast text generation with Cerebras's optimized models
        - Asynchronous function/tool calling with automatic execution
        - Asynchronous streaming responses with real-time tool call handling
        - High-performance inference with specialized hardware acceleration
        - Model listing and metadata retrieval

    Note:
        Cerebras specializes in high-speed inference and currently supports text-only
        models for fast text processing.

    Example:
        ```python
        import asyncio
        from chimeric.providers.cerebras import CerebrasAsyncClient
        from chimeric.tools import ToolManager

        async def main():
            tool_manager = ToolManager()
            client = CerebrasAsyncClient(api_key="your-api-key", tool_manager=tool_manager)

            response = await client.chat_completion(
                messages="What's the fastest way to process this data?",
                model="llama3.1-8b"
            )
            print(response.common.content)

        asyncio.run(main())
        ```

    Attributes:
        api_key (str): The Cerebras API key for authentication.
        tool_manager (ToolManager): Manager for handling tool registration and execution.
    """

    def __init__(self, api_key: str, tool_manager, **kwargs: Any) -> None:
        """Initialize the asynchronous Cerebras client.

        Args:
            api_key: The Cerebras API key for authentication.
            tool_manager: The tool manager instance for handling function calls.
            **kwargs: Additional keyword arguments to pass to the AsyncCerebras client
                constructor, such as base_url, timeout, etc.

        Raises:
            ValueError: If api_key is None or empty.
            ProviderError: If client initialization fails.
        """
        self._provider_name = "Cerebras"
        super().__init__(api_key=api_key, tool_manager=tool_manager, **kwargs)

    def _get_async_client_type(self) -> type:
        """Get the asynchronous Cerebras client class type.

        Returns:
            The AsyncCerebras client class from the cerebras-cloud-sdk library.
        """
        return AsyncCerebras

    def _init_async_client(self, async_client_type: type, **kwargs: Any) -> AsyncCerebras:
        """Initialize the asynchronous Cerebras client instance.

        Args:
            async_client_type: The AsyncCerebras client class to instantiate.
            **kwargs: Additional keyword arguments for client initialization,
                such as base_url, timeout, max_retries, etc.

        Returns:
            Configured asynchronous Cerebras client instance.

        Raises:
            ProviderError: If client initialization fails due to invalid
                credentials or configuration.
        """
        return AsyncCerebras(api_key=self.api_key, **kwargs)

    def _get_capabilities(self) -> Capability:
        """Get supported features for the Cerebras provider.

        Returns:
            Capability object indicating which features are supported:
                - streaming: True (supports real-time streaming responses)
                - tools: True (supports function calling)

        Note:
            Cerebras focuses on high-performance text inference for fast
            text-only processing.
        """
        return Capability(streaming=True, tools=True)

    async def _list_models_impl(self) -> list[ModelSummary]:
        """List available models from the Cerebras API asynchronously.

        Returns:
            List of ModelSummary objects containing model metadata from the API.
            Each summary includes id, name, owner, and creation timestamp.

        Raises:
            ProviderError: If the API request fails or returns invalid data.

        Note:
            Cerebras provides high-performance versions of popular open-source
            models optimized for their specialized hardware.
        """
        models = await self.async_client.models.list()
        return [
            ModelSummary(
                id=model.id,
                name=model.id,
                owned_by=getattr(model, "owned_by", "cerebras"),
                created_at=getattr(model, "created", None),
            )
            for model in models.data
        ]

    def _messages_to_provider_format(self, messages: list[Message]) -> Any:
        """Convert standardized messages to Cerebras's format.

        Args:
            messages: List of standardized Message objects with role and content.

        Returns:
            List of message dictionaries formatted for the Cerebras API. Each message
            contains 'role' and 'content' fields compatible with Cerebras's chat format.

        Note:
            Cerebras uses the OpenAI-compatible message format, so this is a
            straightforward conversion from Message objects to dictionaries.
        """
        return [msg.model_dump(exclude_none=True) for msg in messages]

    def _tools_to_provider_format(self, tools: list[Tool]) -> Any:
        """Convert standardized tools to Cerebras's format.

        Args:
            tools: List of standardized Tool objects containing function definitions.

        Returns:
            List of tool dictionaries formatted for the Cerebras API. Each tool follows
            the OpenAI function calling format with Cerebras-specific optimizations.

        Note:
            Cerebras requires the 'strict' parameter at the function level rather than
            in the parameters schema, and expects strict mode to be enabled for
            optimal performance with their optimized inference.

        Example:
            Input Tool with name="calculate" becomes:
            ```json
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "strict": true,
                    "description": "Perform calculations",
                    "parameters": {"type": "object", "properties": {...}}
                }
            }
            ```
        """
        encoded_tools = []
        for tool in tools:
            # Get parameters and remove 'strict' from the parameters schema since
            # Cerebras expects it only in the function object
            parameters = (
                tool.parameters.model_dump() if isinstance(tool.parameters, ToolParameters) else {}
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
        return encoded_tools

    async def _make_async_provider_request(
        self,
        messages: Any,
        model: str,
        stream: bool,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Make the actual asynchronous API request to Cerebras.

        Args:
            messages: Messages in Cerebras's format (list of message dictionaries).
            model: Model identifier (e.g., "llama3.1-8b", "llama3.1-70b").
            stream: Whether to stream the response token by token.
            tools: Tools in Cerebras's format, or None to disable function calling.
            **kwargs: Additional parameters passed to the API request, such as
                temperature, max_tokens, top_p, etc.

        Returns:
            Raw response from Cerebras's async API. Either a ChatCompletionResponse object
            for non-streaming requests or an async stream object for streaming requests.

        Raises:
            ProviderError: If the API request fails due to authentication,
                rate limiting, model unavailability, or other API errors.

        Note:
            Cerebras provides ultra-fast inference with their specialized hardware,
            significantly reducing response times compared to traditional GPU inference.
        """
        return await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            tools=tools,
            **kwargs,
        )

    def _process_provider_stream_event(
        self, event: ChatChunkResponse, processor: StreamProcessor
    ) -> ChimericStreamChunk[Any] | None:
        """Processes a Cerebras stream event using the standardized processor."""
        if event.choices and event.choices[0].delta.content:
            delta = event.choices[0].delta.content
            return create_stream_chunk(native_event=event, processor=processor, content_delta=delta)

        # Handle tool calls in streaming
        if event.choices and event.choices[0].delta.tool_calls:
            for tool_call_delta in event.choices[0].delta.tool_calls:
                call_id = tool_call_delta.id or f"tool_call_{getattr(tool_call_delta, 'index', 0)}"
                if tool_call_delta.function and tool_call_delta.function.name:
                    processor.process_tool_call_start(call_id, tool_call_delta.function.name)
                if tool_call_delta.function and tool_call_delta.function.arguments:
                    processor.process_tool_call_delta(call_id, tool_call_delta.function.arguments)

        # Handle completion
        if event.choices and event.choices[0].finish_reason:
            # Mark any streaming tool calls as complete
            for call_id in processor.state.tool_calls:
                processor.process_tool_call_complete(call_id)

            return create_stream_chunk(
                native_event=event,
                processor=processor,
                finish_reason=event.choices[0].finish_reason,
            )

        return None

    def _extract_usage_from_response(self, response: ChatCompletionResponse) -> Usage:
        """Extract token usage statistics from Cerebras API response.

        Args:
            response: ChatCompletionResponse from Cerebras API containing usage metadata.

        Returns:
            Usage object with prompt_tokens, completion_tokens, and total_tokens fields.
            Returns empty Usage object if no usage information is available.

        Note:
            Cerebras provides detailed token usage information similar to OpenAI,
            enabling accurate cost tracking and usage monitoring.
        """
        if not response.usage:
            return Usage()

        return Usage(
            prompt_tokens=response.usage.prompt_tokens or 0,
            completion_tokens=response.usage.completion_tokens or 0,
            total_tokens=response.usage.total_tokens or 0,
        )

    def _extract_content_from_response(self, response: ChatCompletionResponse) -> str | list[Any]:
        """Extract text content from Cerebras API response.

        Args:
            response: ChatCompletionResponse from Cerebras API containing message content.

        Returns:
            String containing the generated text content from the first choice.
            Returns empty string if no content is available.

        Note:
            Cerebras responses follow the OpenAI format with choices[0].message.content
            containing the generated text.
        """
        choice = response.choices[0] if response.choices else None
        return (
            choice.message.content if choice and choice.message and choice.message.content else ""
        )

    def _extract_tool_calls_from_response(
        self, response: ChatCompletionResponse
    ) -> list[ToolCall] | None:
        """Extract function tool calls from Cerebras API response.

        Args:
            response: ChatCompletionResponse from Cerebras API that may contain tool calls.

        Returns:
            List of ToolCall objects representing functions the model wants to execute,
            or None if no tool calls are present. Each ToolCall includes call_id,
            function name, and JSON-encoded arguments.

        Note:
            Cerebras uses the OpenAI tool calling format with tool_calls arrays
            containing function specifications and arguments.
        """
        choice = response.choices[0] if response.choices else None
        if not choice or not choice.message.tool_calls:
            return None

        return [
            ToolCall(
                call_id=call.id,
                name=call.function.name,
                arguments=call.function.arguments,
            )
            for call in choice.message.tool_calls
        ]

    def _update_messages_with_tool_calls(
        self,
        messages: list[Any],
        assistant_response: Any,
        tool_calls: list[ToolCall],
        tool_results: list[ToolExecutionResult],
    ) -> list[Any]:
        """Update message history with assistant response and tool results.

        This method formats the conversation history to include the assistant's
        tool calls and their results, following Cerebras's message format requirements.

        Args:
            messages: Current list of messages in the conversation.
            assistant_response: The assistant's response that contained tool calls
                (not directly used but maintained for interface compatibility).
            tool_calls: List of ToolCall objects representing functions called
                by the assistant.
            tool_results: List of ToolExecutionResult objects containing the
                results of executing the tool calls.

        Returns:
            Updated list of messages including:
                1. Original conversation messages
                2. Assistant message with tool_calls array
                3. Tool result messages for each executed function

        Note:
            Cerebras uses the OpenAI message format where tool calls are represented
            as an assistant message with a 'tool_calls' field, followed by separate
            'tool' role messages containing the results.
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
