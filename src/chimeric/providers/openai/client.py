from typing import Any

from openai import NOT_GIVEN, AsyncOpenAI, OpenAI
from openai.types.responses import Response, ResponseFunctionToolCall, ResponseStreamEvent

from chimeric.base import ChimericAsyncClient, ChimericClient
from chimeric.types import (
    Capability,
    ChimericStreamChunk,
    Message,
    ModelSummary,
    Tool,
    ToolCall,
    ToolExecutionResult,
    Usage,
)
from chimeric.utils import StreamProcessor, create_stream_chunk


class OpenAIClient(ChimericClient[OpenAI, Response, ResponseStreamEvent]):
    """Synchronous OpenAI Client for interacting with GPT models via OpenAI API.

    This client provides a unified interface for synchronous interactions with
    OpenAI's GPT models via the `openai` library. It returns `chimeric` response
    objects that wrap the native OpenAI responses and provides comprehensive
    tool calling support for both streaming and non-streaming operations.

    The client supports:
        - Advanced text generation with GPT-4 and other models
        - Function/tool calling with automatic execution
        - Streaming responses with real-time tool call handling
        - Model listing and metadata retrieval

    Note:
        OpenAI provides industry-leading language models including GPT-4 series,
        with strong capabilities across text generation, reasoning, and coding.

    Example:
        ```python
        from chimeric.providers.openai import OpenAIClient
        from chimeric.tools import ToolManager

        tool_manager = ToolManager()
        client = OpenAIClient(api_key="your-api-key", tool_manager=tool_manager)

        response = client.chat_completion(
            messages="What's the capital of France?",
            model="gpt-4"
        )
        print(response.common.content)
        ```

    Attributes:
        api_key (str): The OpenAI API key for authentication.
        tool_manager (ToolManager): Manager for handling tool registration and execution.
    """

    def __init__(self, api_key: str, tool_manager, **kwargs: Any) -> None:
        """Initialize the synchronous OpenAI client.

        Args:
            api_key: The OpenAI API key for authentication.
            tool_manager: The tool manager instance for handling function calls.
            **kwargs: Additional keyword arguments to pass to the OpenAI client
                constructor, such as base_url, timeout, organization, etc.

        Raises:
            ValueError: If api_key is None or empty.
            ProviderError: If client initialization fails.
        """
        self._provider_name = "OpenAI"
        super().__init__(api_key=api_key, tool_manager=tool_manager, **kwargs)

    # ====================================================================
    # Required abstract method implementations
    # ====================================================================

    def _get_client_type(self) -> type:
        """Get the synchronous OpenAI client class type.

        Returns:
            The OpenAI client class from the openai library.
        """
        return OpenAI

    def _init_client(self, client_type: type, **kwargs: Any) -> OpenAI:
        """Initializes the synchronous OpenAI client."""
        return OpenAI(api_key=self.api_key, **kwargs)

    def _get_capabilities(self) -> Capability:
        """Get supported features for the OpenAI provider.

        Returns:
            Capability object indicating which features are supported:
                - streaming: True (supports real-time streaming responses)
                - tools: True (supports function calling and tool use)

        Note:
            OpenAI offers comprehensive text generation capabilities with
            advanced tool use.
        """
        return Capability(streaming=True, tools=True)

    def _list_models_impl(self) -> list[ModelSummary]:
        """Lists available models from the OpenAI API."""
        return [
            ModelSummary(
                id=model.id, name=model.id, owned_by=model.owned_by, created_at=model.created
            )
            for model in self.client.models.list()
        ]

    def _messages_to_provider_format(self, messages: list[Message]) -> Any:
        """Converts standardized messages to OpenAI Responses API format.

        OpenAI Responses API expects messages in a specific format for function calling.
        """
        formatted_messages = []
        for msg in messages:
            if msg.role == "tool":
                # Tool result message - format for OpenAI Responses API
                formatted_messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id,
                        "output": str(msg.content),
                    }
                )
            elif msg.role == "assistant" and msg.tool_calls:
                # Assistant message with tool calls - format for OpenAI Responses API
                for tool_call in msg.tool_calls:
                    formatted_messages.append(
                        {
                            "type": "function_call",
                            "id": tool_call.call_id,
                            "call_id": tool_call.call_id,
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        }
                    )
            else:
                # Regular message - convert to standard format
                formatted_messages.append(msg.model_dump(exclude_none=True))

        return formatted_messages

    def _tools_to_provider_format(self, tools: list[Tool]) -> Any:
        """Converts standardized tools to OpenAI format."""
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters.model_dump() if tool.parameters else None,
            }
            for tool in tools
        ]

    def _make_provider_request(
        self,
        messages: Any,
        model: str,
        stream: bool,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Makes the actual API request to OpenAI."""
        tools_param = NOT_GIVEN if tools is None else tools

        return self.client.responses.create(
            model=model, input=messages, stream=stream, tools=tools_param, **kwargs
        )

    def _process_provider_stream_event(
        self, event: ResponseStreamEvent, processor: StreamProcessor
    ) -> ChimericStreamChunk[ResponseStreamEvent] | None:
        """Processes an OpenAI stream event using the standardized processor."""
        event_type = getattr(event, "type", None)

        # Handle text content deltas
        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "") or ""
            return create_stream_chunk(native_event=event, processor=processor, content_delta=delta)

        # Handle tool call events
        if event_type == "response.output_item.added":
            item = getattr(event, "item", None)
            if item and hasattr(item, "type") and item.type == "function_call":
                tool_call_id = getattr(item, "id", None)  # fc_xxx ID
                call_id = getattr(item, "call_id", None)  # call_xxx ID for outputs
                if tool_call_id:
                    processor.process_tool_call_start(
                        tool_call_id, getattr(item, "name", ""), call_id
                    )
            return None

        if event_type == "response.function_call_arguments.delta":
            tool_call_id = getattr(event, "item_id", None)
            delta = getattr(event, "delta", "") or ""
            if tool_call_id:
                processor.process_tool_call_delta(tool_call_id, delta)
            return None

        if event_type == "response.function_call_arguments.done":
            tool_call_id = getattr(event, "item_id", None)
            if tool_call_id:
                processor.process_tool_call_complete(tool_call_id)
            return None

        # Handle response completion
        if event_type == "response.completed":
            response = getattr(event, "response", None)
            finish_reason = getattr(response, "status", None) if response else None

            return create_stream_chunk(
                native_event=event, processor=processor, finish_reason=finish_reason
            )

        return None

    def _extract_usage_from_response(self, response: Response) -> Usage:
        """Extracts usage information from OpenAI response."""
        if not response.usage:
            return Usage()

        return Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.total_tokens,
        )

    def _extract_content_from_response(self, response: Response) -> str | list[Any]:
        """Extracts content from OpenAI response."""
        return response.output_text or response.output or getattr(response.response, "text", "")

    def _extract_tool_calls_from_response(self, response: Response) -> list[ToolCall] | None:
        """Extracts tool calls from OpenAI response."""
        calls = [
            output
            for output in getattr(response, "output", [])
            if isinstance(output, ResponseFunctionToolCall)
        ]

        if not calls:
            return None

        return [
            ToolCall(call_id=call.call_id, name=call.name, arguments=call.arguments)
            for call in calls
        ]

    def _update_messages_with_tool_calls(
        self,
        messages: list[Any],
        assistant_response: Any,
        tool_calls: list[ToolCall],
        tool_results: list[ToolExecutionResult],
    ) -> list[Any]:
        """Updates message history with assistant response and tool results.

        For OpenAI, this follows the Responses API format where:
        1. Function calls are added as individual entries with type "function_call"
        2. Function results are added as entries with type "function_call_output"
        """
        updated_messages = list(messages)

        # Add the original function calls from the assistant response
        if hasattr(assistant_response, "output") and assistant_response.output:
            # Non-streaming response - add function calls from response output
            for output_item in assistant_response.output:
                if hasattr(output_item, "type") and output_item.type == "function_call":
                    updated_messages.append(output_item)
        else:
            # Streaming response - reconstruct function call objects with correct IDs
            for tool_call in tool_calls:
                # Get the original fc_xxx ID from metadata (this is what we use as the primary tracking ID)
                original_id = tool_call.metadata.get("original_id") if tool_call.metadata else None
                if original_id:
                    function_call_obj = {
                        "type": "function_call",
                        "id": original_id,  # Original fc_xxx ID from streaming
                        "call_id": tool_call.call_id,  # call_xxx ID for output matching
                        "name": tool_call.name,  # Function name
                        "arguments": tool_call.arguments,  # Function arguments
                    }
                    updated_messages.append(function_call_obj)

        # Add the function call results to the message history
        for result in tool_results:
            updated_messages.append(
                {
                    "type": "function_call_output",
                    "call_id": result.call_id,
                    "output": result.result if not result.is_error else f"Error: {result.error}",
                }
            )

        return updated_messages


class OpenAIAsyncClient(ChimericAsyncClient[AsyncOpenAI, Response, ResponseStreamEvent]):
    """Asynchronous OpenAI Client for interacting with GPT models via OpenAI API.

    This client provides a unified interface for asynchronous interactions with
    OpenAI's GPT models via the `openai` library. It returns `chimeric` response
    objects that wrap the native OpenAI responses and provides comprehensive
    tool calling support for both streaming and non-streaming operations.

    The async client supports all the same features as the synchronous client:
        - Asynchronous advanced text generation with GPT-4 and other models
        - Asynchronous function/tool calling with automatic execution
        - Asynchronous streaming responses with real-time tool call handling
        - Model listing and metadata retrieval

    Note:
        OpenAI provides industry-leading language models including GPT-4 series,
        with strong capabilities across text generation, reasoning, and coding.
        The async client is ideal for high-throughput applications and concurrent
        request processing.

    Example:
        ```python
        import asyncio
        from chimeric.providers.openai import OpenAIAsyncClient
        from chimeric.tools import ToolManager

        async def main():
            tool_manager = ToolManager()
            client = OpenAIAsyncClient(api_key="your-api-key", tool_manager=tool_manager)

            response = await client.chat_completion(
                messages="What's the capital of France?",
                model="gpt-4"
            )
            print(response.common.content)

        asyncio.run(main())
        ```

    Attributes:
        api_key (str): The OpenAI API key for authentication.
        tool_manager (ToolManager): Manager for handling tool registration and execution.
    """

    def __init__(self, api_key: str, tool_manager, **kwargs: Any) -> None:
        """Initialize the asynchronous OpenAI client.

        Args:
            api_key: The OpenAI API key for authentication.
            tool_manager: The tool manager instance for handling function calls.
            **kwargs: Additional keyword arguments to pass to the AsyncOpenAI client
                constructor, such as base_url, timeout, organization, etc.

        Raises:
            ValueError: If api_key is None or empty.
            ProviderError: If client initialization fails.
        """
        self._provider_name = "OpenAI"
        super().__init__(api_key=api_key, tool_manager=tool_manager, **kwargs)

    # ====================================================================
    # Required abstract method implementations
    # ====================================================================

    def _get_async_client_type(self) -> type:
        """Get the asynchronous OpenAI client class type.

        Returns:
            The AsyncOpenAI client class from the openai library.
        """
        return AsyncOpenAI

    def _init_async_client(self, async_client_type: type, **kwargs: Any) -> AsyncOpenAI:
        """Initialize the asynchronous OpenAI client instance.

        Args:
            async_client_type: The AsyncOpenAI client class to instantiate.
            **kwargs: Additional keyword arguments for client initialization,
                such as base_url, timeout, max_retries, organization, etc.

        Returns:
            Configured asynchronous OpenAI client instance.

        Raises:
            ProviderError: If client initialization fails due to invalid
                credentials or configuration.
        """
        return AsyncOpenAI(api_key=self.api_key, **kwargs)

    def _get_capabilities(self) -> Capability:
        """Get supported features for the OpenAI provider.

        Returns:
            Capability object indicating which features are supported:
                - streaming: True (supports real-time streaming responses)
                - tools: True (supports function calling and tool use)

        Note:
            OpenAI offers comprehensive text generation capabilities with
            advanced tool use.
        """
        return Capability(streaming=True, tools=True)

    async def _list_models_impl(self) -> list[ModelSummary]:
        """List available models from the OpenAI API asynchronously.

        Returns:
            List of ModelSummary objects containing model metadata from the API.
            Each summary includes id, name, owner, and creation timestamp.

        Raises:
            ProviderError: If the API request fails or returns invalid data.

        Note:
            OpenAI provides access to various GPT model versions, including
            GPT-4, GPT-3.5, and specialized models for different use cases.
        """
        models = await self.async_client.models.list()
        return [
            ModelSummary(
                id=model.id, name=model.id, owned_by=model.owned_by, created_at=model.created
            )
            for model in models.data
        ]

    def _messages_to_provider_format(self, messages: list[Message]) -> Any:
        """Converts standardized messages to OpenAI Responses API format.

        OpenAI Responses API expects messages in a specific format for function calling.
        """
        formatted_messages = []
        for msg in messages:
            if msg.role == "tool":
                # Tool result message - format for OpenAI Responses API
                formatted_messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id,
                        "output": str(msg.content),
                    }
                )
            elif msg.role == "assistant" and msg.tool_calls:
                # Assistant message with tool calls - format for OpenAI Responses API
                for tool_call in msg.tool_calls:
                    formatted_messages.append(
                        {
                            "type": "function_call",
                            "id": tool_call.call_id,
                            "call_id": tool_call.call_id,
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        }
                    )
            else:
                # Regular message - convert to standard format
                formatted_messages.append(msg.model_dump(exclude_none=True))

        return formatted_messages

    def _tools_to_provider_format(self, tools: list[Tool]) -> Any:
        """Converts standardized tools to OpenAI format."""
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters.model_dump() if tool.parameters else None,
            }
            for tool in tools
        ]

    async def _make_async_provider_request(
        self,
        messages: Any,
        model: str,
        stream: bool,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Makes the actual async API request to OpenAI."""
        tools_param = NOT_GIVEN if tools is None else tools

        return await self.async_client.responses.create(
            model=model, input=messages, stream=stream, tools=tools_param, **kwargs
        )

    def _process_provider_stream_event(
        self, event: ResponseStreamEvent, processor: StreamProcessor
    ) -> ChimericStreamChunk[ResponseStreamEvent] | None:
        """Processes an OpenAI stream event using the standardized processor.

        This is the same implementation as the sync client since event processing
        is identical.
        """
        event_type = getattr(event, "type", None)

        # Handle text content deltas
        if event_type == "response.output_text.delta":
            delta = getattr(event, "delta", "") or ""
            return create_stream_chunk(native_event=event, processor=processor, content_delta=delta)

        # Handle tool call events
        if event_type == "response.output_item.added":
            item = getattr(event, "item", None)
            if item and hasattr(item, "type") and item.type == "function_call":
                tool_call_id = getattr(item, "id", None)  # fc_xxx ID
                call_id = getattr(item, "call_id", None)  # call_xxx ID for outputs
                if tool_call_id:
                    processor.process_tool_call_start(
                        tool_call_id, getattr(item, "name", ""), call_id
                    )
            return None

        if event_type == "response.function_call_arguments.delta":
            tool_call_id = getattr(event, "item_id", None)
            delta = getattr(event, "delta", "") or ""
            if tool_call_id:
                processor.process_tool_call_delta(tool_call_id, delta)
            return None

        if event_type == "response.function_call_arguments.done":
            tool_call_id = getattr(event, "item_id", None)
            if tool_call_id:
                processor.process_tool_call_complete(tool_call_id)
            return None

        # Handle response completion
        if event_type == "response.completed":
            response = getattr(event, "response", None)
            finish_reason = getattr(response, "status", None) if response else None

            return create_stream_chunk(
                native_event=event, processor=processor, finish_reason=finish_reason
            )

        return None

    def _extract_usage_from_response(self, response: Response) -> Usage:
        """Extracts usage information from OpenAI response."""
        if not response.usage:
            return Usage()

        return Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.total_tokens,
        )

    def _extract_content_from_response(self, response: Response) -> str | list[Any]:
        """Extracts content from OpenAI response."""
        return response.output_text or response.output or getattr(response.response, "text", "")

    def _extract_tool_calls_from_response(self, response: Response) -> list[ToolCall] | None:
        """Extracts tool calls from OpenAI response."""
        calls = [
            output for output in response.output if isinstance(output, ResponseFunctionToolCall)
        ]
        if not calls:
            return None

        return [
            ToolCall(call_id=call.call_id, name=call.name, arguments=call.arguments)
            for call in calls
        ]

    def _update_messages_with_tool_calls(
        self,
        messages: list[Any],
        assistant_response: Any,
        tool_calls: list[ToolCall],
        tool_results: list[ToolExecutionResult],
    ) -> list[Any]:
        """Updates message history with assistant response and tool results.

        For OpenAI, this follows the Responses API format where:
        1. Function calls are added as individual entries with type "function_call"
        2. Function results are added as entries with type "function_call_output"
        """
        updated_messages = list(messages)

        # Add the original function calls from the assistant response
        if hasattr(assistant_response, "output") and assistant_response.output:
            # Non-streaming response - add function calls from response output
            for output_item in assistant_response.output:
                if hasattr(output_item, "type") and output_item.type == "function_call":
                    updated_messages.append(output_item)
        else:
            # Streaming response - reconstruct function call objects with correct IDs
            for tool_call in tool_calls:
                # Get the original fc_xxx ID from metadata (this is what we use as the primary tracking ID)
                original_id = tool_call.metadata.get("original_id") if tool_call.metadata else None
                if original_id:
                    function_call_obj = {
                        "type": "function_call",
                        "id": original_id,  # Original fc_xxx ID from streaming
                        "call_id": tool_call.call_id,  # call_xxx ID for output matching
                        "name": tool_call.name,  # Function name
                        "arguments": tool_call.arguments,  # Function arguments
                    }
                    updated_messages.append(function_call_obj)

        # Add the function call results to the message history
        for result in tool_results:
            updated_messages.append(
                {
                    "type": "function_call_output",
                    "call_id": result.call_id,
                    "output": result.result if not result.is_error else f"Error: {result.error}",
                }
            )

        return updated_messages
