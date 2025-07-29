import json
from typing import Any

from anthropic import NOT_GIVEN, Anthropic, AsyncAnthropic
from anthropic.types import Message, MessageStreamEvent

from chimeric.base import ChimericAsyncClient, ChimericClient
from chimeric.types import (
    Capability,
    ChimericStreamChunk,
    ModelSummary,
    Tool,
    ToolCall,
    ToolExecutionResult,
    Usage,
)
from chimeric.types import (
    Message as ChimericMessage,
)
from chimeric.utils import StreamProcessor, create_stream_chunk


class AnthropicClient(ChimericClient[Anthropic, Message, MessageStreamEvent]):
    """Synchronous Anthropic Client for interacting with Claude models via Anthropic API.

    This client provides a unified interface for synchronous interactions with
    Anthropic's Claude models via the `anthropic` library. It returns `chimeric`
    response objects that wrap the native Anthropic responses and provides
    comprehensive tool calling support for both streaming and non-streaming operations.

    The client supports:
        - Advanced text generation with Claude's reasoning capabilities
        - Multimodal inputs including images and documents
        - Function/tool calling with automatic execution
        - Streaming responses with real-time tool call handling
        - Extended context lengths for complex tasks
        - Model listing and metadata retrieval

    Note:
        Anthropic provides some of the most capable language models with strong
        reasoning abilities. Claude models excel at complex analysis, coding,
        and creative tasks while maintaining helpful, harmless, and honest outputs.

    Example:
        ```python
        from chimeric.providers.anthropic import AnthropicClient
        from chimeric.tools import ToolManager

        tool_manager = ToolManager()
        client = AnthropicClient(api_key="your-api-key", tool_manager=tool_manager)

        response = client.chat_completion(
            messages="Analyze this complex problem and provide a solution.",
            model="claude-3-5-sonnet-20241022"
        )
        print(response.common.content)
        ```

    Attributes:
        api_key (str): The Anthropic API key for authentication.
        tool_manager (ToolManager): Manager for handling tool registration and execution.
    """

    def __init__(self, api_key: str, tool_manager, **kwargs: Any) -> None:
        """Initialize the synchronous Anthropic client.

        Args:
            api_key: The Anthropic API key for authentication.
            tool_manager: The tool manager instance for handling function calls.
            **kwargs: Additional keyword arguments to pass to the Anthropic client
                constructor, such as base_url, timeout, default_headers, etc.

        Raises:
            ValueError: If api_key is None or empty.
            ProviderError: If client initialization fails.
        """
        self._provider_name = "Anthropic"
        super().__init__(api_key=api_key, tool_manager=tool_manager, **kwargs)

    # ====================================================================
    # Required abstract method implementations
    # ====================================================================

    def _get_client_type(self) -> type:
        """Get the synchronous Anthropic client class type.

        Returns:
            The Anthropic client class from the anthropic library.
        """
        return Anthropic

    def _init_client(self, client_type: type, **kwargs: Any) -> Anthropic:
        """Initialize the synchronous Anthropic client instance.

        Args:
            client_type: The Anthropic client class to instantiate.
            **kwargs: Additional keyword arguments for client initialization,
                such as base_url, timeout, max_retries, default_headers, etc.

        Returns:
            Configured synchronous Anthropic client instance.

        Raises:
            ProviderError: If client initialization fails due to invalid
                credentials or configuration.
        """
        return Anthropic(api_key=self.api_key, **kwargs)

    def _get_capabilities(self) -> Capability:
        """Gets Anthropic provider capabilities.

        Returns:
            Capability object indicating which features are supported:
            - streaming: True (supports real-time streaming responses)
            - tools: True (supports function calling)
        """
        return Capability(streaming=True, tools=True)

    def _get_model_aliases(self) -> list[str]:
        """Return model aliases to include in model listings.

        Returns:
            List of string aliases for popular Anthropic Claude models,
            including latest versions and commonly used model names.

        Note:
            These aliases help users access Claude models with simpler names
            like 'claude-opus-4-0' instead of full version identifiers.
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
        """List available models from the Anthropic API.

        Returns:
            List of ModelSummary objects containing model metadata from the API.
            Each summary includes id, display name, creation timestamp, and
            additional model metadata.

        Raises:
            ProviderError: If the API request fails or returns invalid data.

        Note:
            Anthropic provides access to various Claude model versions with
            different capabilities and performance characteristics.
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

    def _messages_to_provider_format(self, messages: list[ChimericMessage]) -> Any:
        """Converts standardized messages to Anthropic format."""
        formatted_messages = []

        for message in messages:
            msg_dict = message.model_dump(exclude_none=True)
            if msg_dict.get("role") != "system":
                formatted_messages.append(msg_dict)

        return formatted_messages

    def _tools_to_provider_format(self, tools: list[Tool]) -> Any:
        """Converts standardized tools to Anthropic format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters.model_dump() if tool.parameters else {},
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
        """Make the actual API request to Anthropic.

        Args:
            messages: Messages in Anthropic's format (list of message dictionaries).
            model: Model identifier (e.g., "claude-3-5-sonnet-20241022", "claude-3-opus-20240229").
            stream: Whether to stream the response token by token.
            tools: Tools in Anthropic's format, or None to disable function calling.
            **kwargs: Additional parameters passed to the API request, such as
                temperature, max_tokens, top_p, system, etc.

        Returns:
            Raw response from Anthropic's API. Either a Message object for non-streaming
            requests or a stream object for streaming requests.

        Raises:
            ProviderError: If the API request fails due to authentication,
                rate limiting, model unavailability, or other API errors.

        Note:
            Anthropic's Claude models support advanced reasoning and tool use,
            with excellent performance on complex analysis and coding tasks.
        """
        # Build params with only valid Anthropic parameters
        max_tokens = kwargs.pop("max_tokens", 4096)  # Default max tokens if not provided (needs to be set)
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
            "tools": tools if tools else NOT_GIVEN,
            **kwargs,  # Include all other parameters directly
        }

        return self.client.messages.create(**params)  # type: ignore[reportArgumentType]

    def _process_provider_stream_event(
        self, event: MessageStreamEvent, processor: StreamProcessor
    ) -> ChimericStreamChunk[Any] | None:
        """Process a streaming event from Anthropic API into standardized format.

        Args:
            event: Raw streaming event from the Anthropic API containing delta content,
                tool calls, or completion signals.
            processor: StreamProcessor instance that manages streaming state and
                accumulates content across multiple events.

        Returns:
            ChimericStreamChunk object containing processed content delta, tool call
            information, or completion status. Returns None if the event contains
            no processable content.

        Note:
            Anthropic streaming events have a unique format with content_block_delta
            for text and tool_use events for function calls. This method handles
            the complex event structure and state management automatically.
        """
        event_type = event.type

        # Handle text content deltas
        if event_type == "content_block_delta" and hasattr(event.delta, "text"):
            delta = event.delta.text
            return create_stream_chunk(native_event=event, processor=processor, content_delta=delta)

        # Handle tool call events
        if event_type == "content_block_start":
            if (
                hasattr(event, "content_block")
                and getattr(event.content_block, "type", None) == "tool_use"
            ):
                block_index = getattr(event, "index", 0)
                tool_call_id = f"tool_call_{block_index}"
                processor.process_tool_call_start(
                    tool_call_id, getattr(event.content_block, "name", "")
                )
            return None

        if event_type == "content_block_delta" and hasattr(event.delta, "partial_json"):
            block_index = getattr(event, "index", 0)
            tool_call_id = f"tool_call_{block_index}"
            delta = event.delta.partial_json
            processor.process_tool_call_delta(tool_call_id, delta)
            return None

        if event_type == "content_block_stop":
            block_index = getattr(event, "index", 0)
            tool_call_id = f"tool_call_{block_index}"
            processor.process_tool_call_complete(tool_call_id)
            return None

        # Handle response completion
        if event_type == "message_stop":
            return create_stream_chunk(
                native_event=event, processor=processor, finish_reason="end_turn"
            )

        return None

    def _extract_usage_from_response(self, response: Message) -> Usage:
        """Extracts usage information from Anthropic response."""
        if not response.usage:
            return Usage()

        return Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

    def _extract_content_from_response(self, response: Message) -> str | list[Any]:
        """Extracts content from Anthropic response."""
        return "".join(block.text for block in response.content if hasattr(block, "text"))

    def _extract_tool_calls_from_response(self, response: Message) -> list[ToolCall] | None:
        """Extracts tool calls from Anthropic response."""
        tool_use_blocks = [
            block for block in response.content if getattr(block, "type", None) == "tool_use"
        ]

        if not tool_use_blocks:
            return None

        return [
            ToolCall(
                call_id=block.id,
                name=block.name,
                arguments=json.dumps(block.input)
                if isinstance(block.input, dict)
                else str(block.input),
            )
            for block in tool_use_blocks
        ]

    def _update_messages_with_tool_calls(
        self,
        messages: list[Any],
        assistant_response: Any,
        tool_calls: list[ToolCall],
        tool_results: list[ToolExecutionResult],
    ) -> list[Any]:
        """Updates message history with assistant response and tool results.

        For Anthropic, we need to:
        1. Add the assistant message with tool_use blocks
        2. Add the user message with tool_result blocks
        """
        # Messages is now just a list of formatted messages
        current_messages = messages

        updated_messages = list(current_messages)

        # Build assistant message with tool uses
        assistant_content = []

        # Check if we have a streaming response event (no content attribute)
        # or a full Message object (has content attribute)
        if hasattr(assistant_response, "content") and assistant_response.content:
            # Non-streaming response - extract from Message object
            for block in assistant_response.content:
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
        else:
            # Streaming response - reconstruct from tool_calls
            # Add any accumulated text content (if available)
            if hasattr(assistant_response, "accumulated_content"):
                text_content = assistant_response.accumulated_content
                if text_content:
                    assistant_content.append({"type": "text", "text": text_content})

            # Add tool use blocks from tool_calls
            for tool_call in tool_calls:
                # Parse arguments back to dict for Anthropic format
                try:
                    input_data = json.loads(tool_call.arguments) if tool_call.arguments else {}
                except (json.JSONDecodeError, TypeError):
                    input_data = {}

                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.call_id,
                        "name": tool_call.name,
                        "input": input_data,
                    }
                )

        updated_messages.append({"role": "assistant", "content": assistant_content})

        # Add tool results as a user message
        tool_results_content = []
        for result in tool_results:
            tool_results_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": result.call_id,
                    "content": result.result if not result.is_error else f"Error: {result.error}",
                    "is_error": result.is_error,
                }
            )

        updated_messages.append({"role": "user", "content": tool_results_content})

        return updated_messages


class AnthropicAsyncClient(ChimericAsyncClient[AsyncAnthropic, Message, MessageStreamEvent]):
    """Asynchronous Anthropic Client for interacting with Claude models via Anthropic API.

    This client provides a unified interface for asynchronous interactions with
    Anthropic's Claude models via the `anthropic` library. It returns `chimeric`
    response objects that wrap the native Anthropic responses and provides
    comprehensive tool calling support for both streaming and non-streaming operations.

    The async client supports all the same features as the synchronous client:
        - Asynchronous advanced text generation with Claude's reasoning capabilities
        - Asynchronous function/tool calling with automatic execution
        - Asynchronous streaming responses with real-time tool call handling
        - Extended context lengths for complex tasks
        - Model listing and metadata retrieval

    Note:
        Anthropic provides some of the most capable language models with strong
        reasoning abilities. Claude models excel at complex analysis, coding,
        and creative tasks while maintaining helpful, harmless, and honest outputs.

    Example:
        ```python
        import asyncio
        from chimeric.providers.anthropic import AnthropicAsyncClient
        from chimeric.tools import ToolManager

        async def main():
            tool_manager = ToolManager()
            client = AnthropicAsyncClient(api_key="your-api-key", tool_manager=tool_manager)

            response = await client.chat_completion(
                messages="Analyze this complex problem and provide a solution.",
                model="claude-3-5-sonnet-20241022"
            )
            print(response.common.content)

        asyncio.run(main())
        ```

    Attributes:
        api_key (str): The Anthropic API key for authentication.
        tool_manager (ToolManager): Manager for handling tool registration and execution.
    """

    def __init__(self, api_key: str, tool_manager, **kwargs: Any) -> None:
        """Initializes the asynchronous Anthropic client."""
        self._provider_name = "Anthropic"
        super().__init__(api_key=api_key, tool_manager=tool_manager, **kwargs)

    # ====================================================================
    # Required abstract method implementations
    # ====================================================================

    def _get_async_client_type(self) -> type:
        """Returns the AsyncAnthropic client type."""
        return AsyncAnthropic

    def _init_async_client(self, async_client_type: type, **kwargs: Any) -> AsyncAnthropic:
        """Initializes the asynchronous Anthropic client."""
        return AsyncAnthropic(api_key=self.api_key, **kwargs)

    def _get_capabilities(self) -> Capability:
        """Gets Anthropic provider capabilities.

        Returns:
            Capability object indicating which features are supported:
            - streaming: True (supports real-time streaming responses)
            - tools: True (supports function calling)
        """
        return Capability(streaming=True, tools=True)

    def _get_model_aliases(self) -> list[str]:
        """Gets a list of Anthropic model aliases."""
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

    async def _list_models_impl(self) -> list[ModelSummary]:
        """Lists available models from the Anthropic API."""
        models_response = await self.async_client.models.list()
        return [
            ModelSummary(
                id=model.id,
                name=model.display_name,
                created_at=int(model.created_at.timestamp()),
                metadata=model.model_dump() if hasattr(model, "model_dump") else {},
            )
            for model in models_response.data
        ]

    def _messages_to_provider_format(self, messages: list[ChimericMessage]) -> Any:
        """Converts standardized messages to Anthropic format."""
        formatted_messages = []

        for message in messages:
            msg_dict = message.model_dump(exclude_none=True)
            if msg_dict.get("role") != "system":
                formatted_messages.append(msg_dict)

        return formatted_messages

    def _tools_to_provider_format(self, tools: list[Tool]) -> Any:
        """Converts standardized tools to Anthropic format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters.model_dump() if tool.parameters else {},
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
        """Makes the actual async API request to Anthropic."""
        # Build params with only valid Anthropic parameters
        max_tokens = kwargs.pop("max_tokens", 4096)  # Default max tokens if not provided (needs to be set)
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
            "tools": tools if tools else NOT_GIVEN,
            **kwargs,  # Include all other parameters directly
        }

        return await self.async_client.messages.create(**params)  # type: ignore[reportArgumentType]

    def _process_provider_stream_event(
        self, event: MessageStreamEvent, processor: StreamProcessor
    ) -> ChimericStreamChunk[Any] | None:
        """Processes an Anthropic stream event using the standardized processor."""
        event_type = event.type

        # Handle text content deltas
        if event_type == "content_block_delta" and hasattr(event.delta, "text"):
            delta = event.delta.text
            return create_stream_chunk(native_event=event, processor=processor, content_delta=delta)

        # Handle tool call events
        if event_type == "content_block_start":
            if (
                hasattr(event, "content_block")
                and getattr(event.content_block, "type", None) == "tool_use"
            ):
                block_index = getattr(event, "index", 0)
                tool_call_id = f"tool_call_{block_index}"
                processor.process_tool_call_start(
                    tool_call_id, getattr(event.content_block, "name", "")
                )
            return None

        if event_type == "content_block_delta" and hasattr(event.delta, "partial_json"):
            block_index = getattr(event, "index", 0)
            tool_call_id = f"tool_call_{block_index}"
            delta = event.delta.partial_json
            processor.process_tool_call_delta(tool_call_id, delta)
            return None

        if event_type == "content_block_stop":
            block_index = getattr(event, "index", 0)
            tool_call_id = f"tool_call_{block_index}"
            processor.process_tool_call_complete(tool_call_id)
            return None

        # Handle response completion
        if event_type == "message_stop":
            return create_stream_chunk(
                native_event=event, processor=processor, finish_reason="end_turn"
            )

        return None

    def _extract_usage_from_response(self, response: Message) -> Usage:
        """Extracts usage information from Anthropic response."""
        if not response.usage:
            return Usage()

        return Usage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

    def _extract_content_from_response(self, response: Message) -> str | list[Any]:
        """Extracts content from Anthropic response."""
        return "".join(block.text for block in response.content if hasattr(block, "text"))

    def _extract_tool_calls_from_response(self, response: Message) -> list[ToolCall] | None:
        """Extracts tool calls from Anthropic response."""
        tool_use_blocks = [
            block for block in response.content if getattr(block, "type", None) == "tool_use"
        ]

        if not tool_use_blocks:
            return None

        return [
            ToolCall(
                call_id=block.id,
                name=block.name,
                arguments=json.dumps(block.input)
                if isinstance(block.input, dict)
                else str(block.input),
            )
            for block in tool_use_blocks
        ]

    def _update_messages_with_tool_calls(
        self,
        messages: list[Any],
        assistant_response: Any,
        tool_calls: list[ToolCall],
        tool_results: list[ToolExecutionResult],
    ) -> list[Any]:
        """Updates message history with assistant response and tool results.

        For Anthropic, we need to:
        1. Add the assistant message with tool_use blocks
        2. Add the user message with tool_result blocks
        """
        # Messages is now just a list of formatted messages
        current_messages = messages

        updated_messages = list(current_messages)

        # Build assistant message with tool uses
        assistant_content = []

        # Check if we have a streaming response event (no content attribute)
        # or a full Message object (has content attribute)
        if hasattr(assistant_response, "content") and assistant_response.content:
            # Non-streaming response - extract from Message object
            for block in assistant_response.content:
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
        else:
            # Streaming response - reconstruct from tool_calls
            # Add any accumulated text content (if available)
            if hasattr(assistant_response, "accumulated_content"):
                text_content = assistant_response.accumulated_content
                if text_content:
                    assistant_content.append({"type": "text", "text": text_content})

            # Add tool use blocks from tool_calls
            for tool_call in tool_calls:
                # Parse arguments back to dict for Anthropic format
                try:
                    input_data = json.loads(tool_call.arguments) if tool_call.arguments else {}
                except (json.JSONDecodeError, TypeError):
                    input_data = {}

                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.call_id,
                        "name": tool_call.name,
                        "input": input_data,
                    }
                )

        updated_messages.append({"role": "assistant", "content": assistant_content})

        # Add tool results as a user message
        tool_results_content = []
        for result in tool_results:
            tool_results_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": result.call_id,
                    "content": result.result if not result.is_error else f"Error: {result.error}",
                    "is_error": result.is_error,
                }
            )

        updated_messages.append({"role": "user", "content": tool_results_content})

        return updated_messages
