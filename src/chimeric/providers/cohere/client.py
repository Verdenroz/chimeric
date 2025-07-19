from collections.abc import AsyncGenerator, Generator
from typing import Any

from cohere import AsyncClientV2 as AsyncCohere
from cohere import (
    ChatResponse,
    CitationEndStreamedChatResponseV2,
    CitationStartStreamedChatResponseV2,
    ContentDeltaStreamedChatResponseV2,
    ContentEndStreamedChatResponseV2,
    ContentStartStreamedChatResponseV2,
    DebugStreamedChatResponseV2,
    MessageEndStreamedChatResponseV2,
    MessageStartStreamedChatResponseV2,
    ToolCallDeltaStreamedChatResponseV2,
    ToolCallEndStreamedChatResponseV2,
    ToolCallStartStreamedChatResponseV2,
    ToolPlanDeltaStreamedChatResponseV2,
)
from cohere import ClientV2 as Cohere

from chimeric.base import ChimericAsyncClient, ChimericClient
from chimeric.types import (
    Capability,
    ChimericStreamChunk,
    Message,
    ModelSummary,
    StreamChunk,
    Tool,
    ToolCall,
    ToolExecutionResult,
    Usage,
)
from chimeric.utils import StreamProcessor, create_stream_chunk

CohereChunk = (
    MessageStartStreamedChatResponseV2
    | ContentStartStreamedChatResponseV2
    | ContentDeltaStreamedChatResponseV2
    | ContentEndStreamedChatResponseV2
    | ToolPlanDeltaStreamedChatResponseV2
    | ToolCallStartStreamedChatResponseV2
    | ToolCallDeltaStreamedChatResponseV2
    | ToolCallEndStreamedChatResponseV2
    | CitationStartStreamedChatResponseV2
    | CitationEndStreamedChatResponseV2
    | MessageEndStreamedChatResponseV2
    | DebugStreamedChatResponseV2
)


class CohereClient(ChimericClient[Cohere, ChatResponse, CohereChunk]):
    """Synchronous Cohere Client for interacting with Cohere models."""

    def __init__(self, api_key: str, tool_manager, **kwargs: Any) -> None:
        """Initializes the synchronous Cohere client.

        Args:
            api_key: The Cohere API key.
            tool_manager: The tool manager instance.
            **kwargs: Additional keyword arguments for the Cohere client.
        """
        self._provider_name = "Cohere"
        super().__init__(api_key=api_key, tool_manager=tool_manager, **kwargs)

    def _get_client_type(self) -> type:
        """Returns the Cohere client type.

        Returns:
            The type of the synchronous Cohere client.
        """
        return Cohere

    def _init_client(self, client_type: type, **kwargs: Any) -> Cohere:
        """Initializes the synchronous Cohere client.

        Args:
            client_type: The type of the client to initialize.
            **kwargs: Additional keyword arguments for the client.

        Returns:
            An instance of the synchronous Cohere client.
        """
        return Cohere(api_key=self.api_key, **kwargs)

    def _get_capabilities(self) -> Capability:
        """Gets Cohere provider capabilities.

        Returns:
            Capability object indicating which features are supported:
            - streaming: True (supports real-time streaming responses)
            - tools: True (supports function calling)
        """
        return Capability(streaming=True, tools=True)

    def _list_models_impl(self) -> list[ModelSummary]:
        """Lists available models from the Cohere API.

        Returns:
            A list of ModelSummary objects for available models.
        """
        models_response = self.client.models.list()
        return [
            ModelSummary(
                id=str(getattr(model, "id", getattr(model, "name", "unknown"))),
                name=str(getattr(model, "name", "unknown")),
                metadata=model.model_dump() if hasattr(model, "model_dump") else {},
                owned_by="cohere",
            )
            for model in models_response.models
        ]

    def _messages_to_provider_format(self, messages: list[Message]) -> Any:
        """Converts standardized messages to Cohere format.

        Args:
            messages: A list of standardized Message objects.

        Returns:
            A list of messages in the format expected by the Cohere API.
        """
        return [msg.model_dump(exclude_none=True) for msg in messages]

    def _tools_to_provider_format(self, tools: list[Tool]) -> Any:
        """Converts standardized tools to Cohere format.

        Args:
            tools: A list of standardized Tool objects.

        Returns:
            A list of tools in the format expected by the Cohere API.
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
        """Makes the actual API request to Cohere.

        Args:
            messages: The provider-formatted messages.
            model: The model to use for the request.
            stream: Whether to stream the response.
            tools: The provider-formatted tools, if any.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            The response from the Cohere API.
        """
        if stream:
            return self.client.chat_stream(
                model=model,
                messages=messages,
                tools=tools,
                **kwargs,
            )
        return self.client.chat(
            model=model,
            messages=messages,
            tools=tools,
            **kwargs,
        )

    def _process_provider_stream_event(
        self, event: Any, processor: StreamProcessor
    ) -> ChimericStreamChunk[CohereChunk] | None:
        """Processes a Cohere stream event using the standardized processor.

        Args:
            event: The event from the Cohere stream.
            processor: The stream processor to manage state.

        Returns:
            A ChimericStreamChunk if the event is processed, otherwise None.
        """
        event_type = event.type

        # Handle tool plan deltas - these are part of the content stream for Cohere
        if event_type == "tool-plan-delta":
            delta = ""
            if hasattr(event, "delta") and hasattr(event.delta, "message"):
                delta = getattr(event.delta.message, "tool_plan", "")
            return create_stream_chunk(native_event=event, processor=processor, content_delta=delta)

        # Handle tool call start - register the tool call with the processor
        if event_type == "tool-call-start":
            if (
                hasattr(event, "delta")
                and hasattr(event.delta, "message")
                and hasattr(event.delta.message, "tool_calls")
                and event.delta.message.tool_calls
            ):
                tool_call = event.delta.message.tool_calls
                call_id = tool_call.id
                function_name = tool_call.function.name
                processor.process_tool_call_start(call_id, function_name)
            return create_stream_chunk(native_event=event, processor=processor)

        # Handle tool call argument deltas - accumulate arguments
        if event_type == "tool-call-delta":
            if (
                hasattr(event, "delta")
                and hasattr(event.delta, "message")
                and hasattr(event.delta.message, "tool_calls")
            ):
                tool_calls_delta = event.delta.message.tool_calls
                if hasattr(tool_calls_delta, "function") and hasattr(
                    tool_calls_delta.function, "arguments"
                ):
                    # We need to find which tool call this delta belongs to - use index
                    call_index = getattr(event, "index", 0)
                    # Since we don't have the call_id directly in delta events, we need to track by index
                    # For now, assume single tool call or track state differently
                    arguments_delta = tool_calls_delta.function.arguments

                    # Get the active tool call ID from processor state
                    tool_calls = list(processor.state.tool_calls.keys())
                    if tool_calls:
                        # Use the most recent tool call or match by index
                        call_id = (
                            tool_calls[call_index]
                            if call_index < len(tool_calls)
                            else tool_calls[-1]
                        )
                        processor.process_tool_call_delta(call_id, arguments_delta)
            return create_stream_chunk(native_event=event, processor=processor)

        # Handle tool call end - mark as complete
        if event_type == "tool-call-end":
            call_index = getattr(event, "index", 0)
            # Get the tool call ID to mark complete
            tool_calls = list(processor.state.tool_calls.keys())
            if tool_calls:
                call_id = tool_calls[call_index] if call_index < len(tool_calls) else tool_calls[-1]
                processor.process_tool_call_complete(call_id)
            return create_stream_chunk(native_event=event, processor=processor)

        # Handle content deltas (main response text after tools)
        if event_type == "content-delta":
            delta = ""
            if (
                hasattr(event, "delta")
                and hasattr(event.delta, "message")
                and hasattr(event.delta.message, "content")
                and hasattr(event.delta.message.content, "text")
            ):
                delta = event.delta.message.content.text
            return create_stream_chunk(native_event=event, processor=processor, content_delta=delta)

        # Handle message end
        if event_type == "message-end":
            finish_reason = "stop"
            if hasattr(event, "delta") and hasattr(event.delta, "finish_reason"):
                reason = event.delta.finish_reason
                if reason == "TOOL_CALL":
                    finish_reason = "tool_calls"
                elif reason == "COMPLETE":
                    finish_reason = "stop"
            return create_stream_chunk(
                native_event=event, processor=processor, finish_reason=finish_reason
            )

        # Handle other events like content-start, content-end, citation events, etc.
        if event_type in [
            "message-start",
            "content-start",
            "content-end",
            "citation-start",
            "citation-end",
        ]:
            return create_stream_chunk(native_event=event, processor=processor)

        return None

    def _extract_usage_from_response(self, response: Any) -> Usage:
        """Extracts usage information from Cohere response.

        Args:
            response: The response from the Cohere API.

        Returns:
            A Usage object containing token counts.
        """
        usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        if hasattr(response, "usage") and hasattr(response.usage, "tokens"):
            input_tokens = getattr(response.usage.tokens, "input_tokens", 0)
            output_tokens = getattr(response.usage.tokens, "output_tokens", 0)
            usage = Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )
        return usage

    def _extract_content_from_response(self, response: Any) -> str | list[Any]:
        """Extracts content from Cohere response.

        Args:
            response: The response from the Cohere API.

        Returns:
            The extracted content as a string or list.
        """
        content = ""
        if hasattr(response, "message") and hasattr(response.message, "content"):
            if isinstance(response.message.content, list) and len(response.message.content) > 0:
                content = response.message.content[0].text
            else:
                content = str(response.message.content)
        elif hasattr(response, "text"):
            content = response.text
        return content

    def _extract_tool_calls_from_response(self, response: Any) -> list[ToolCall] | None:
        """Extracts tool calls from Cohere response.

        Args:
            response: The response from the Cohere API.

        Returns:
            A list of ToolCall objects if tool calls are present, otherwise None.
        """
        if (
            hasattr(response, "message")
            and hasattr(response.message, "tool_calls")
            and response.message.tool_calls
        ):
            return [
                ToolCall(
                    call_id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                )
                for call in response.message.tool_calls
            ]
        return None

    def _update_messages_with_tool_calls(
        self,
        messages: list[Any],
        assistant_response: Any,
        tool_calls: list[ToolCall],
        tool_results: list[ToolExecutionResult],
    ) -> list[Any]:
        """Updates message history with assistant response and tool results for Cohere.

        Args:
            messages: The current list of provider-formatted messages.
            assistant_response: The native response from the assistant that contained the tool calls.
            tool_calls: The list of tool calls made by the assistant.
            tool_results: The results of executing the tool calls.

        Returns:
            An updated list of provider-formatted messages.
        """
        updated_messages = list(messages)

        # Add the assistant's response with tool calls
        if hasattr(assistant_response, "message") and hasattr(
            assistant_response.message, "tool_calls"
        ):
            updated_messages.append(
                {
                    "role": "assistant",
                    "tool_plan": getattr(assistant_response.message, "tool_plan", ""),
                    "tool_calls": assistant_response.message.tool_calls,
                }
            )

        # Add tool results
        for result in tool_results:
            updated_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": [
                        {
                            "type": "document",
                            "document": {"data": result.result or result.error or "No result"},
                        }
                    ],
                }
            )

        return updated_messages

    def _handle_streaming_tool_calls(
        self,
        stream: Any,
        processor: StreamProcessor,
        messages: Any,
        model: str,
        tools: Any,
        **kwargs: Any,
    ) -> Generator[ChimericStreamChunk[CohereChunk], None, None]:
        """Handles streaming with tool calls for Cohere's specific pattern.

        This method manages the conversation loop for tool calls in streaming mode,
        which involves making non-streaming requests to get tool calls, executing them,
        and then making a final streaming request for the user-facing response.

        Args:
            stream: The initial stream from the provider.
            processor: The stream processor.
            messages: The provider-formatted messages.
            model: The model to use.
            tools: The provider-formatted tools.
            **kwargs: Additional arguments for the API call.

        Yields:
            ChimericStreamChunk objects from the final response stream.
        """
        current_messages = list(messages)

        # Collect all events from the initial stream but don't yield content to user yet
        # Only process to detect if tools are being called
        stream_events = list(stream)

        # Check if the initial response contains tool calls by looking at events
        has_initial_tool_calls = any(event.type == "tool-call-start" for event in stream_events)

        # If no tool calls in initial response, stream it normally
        if not has_initial_tool_calls:
            accumulated_content = ""
            for event in stream_events:
                accumulated_content, chunk = self._process_event(event, accumulated_content)
                if chunk:
                    yield chunk
            return

        # Continue the conversation loop until no more tool calls are needed
        while True:
            # Get a fresh non-streaming response to check for tool calls
            non_stream_response = self._make_provider_request(
                current_messages, model, False, tools, **kwargs
            )

            # Check for tool calls and process them
            if (
                hasattr(non_stream_response, "message")
                and hasattr(non_stream_response.message, "tool_calls")
                and non_stream_response.message.tool_calls
            ):
                # Add the assistant's tool call message with properly formatted tool calls
                current_messages.append(
                    {
                        "role": "assistant",
                        "tool_plan": getattr(non_stream_response.message, "tool_plan", ""),
                        "tool_calls": non_stream_response.message.tool_calls,
                    }
                )

                # Convert to ToolCall objects
                tool_calls = [
                    ToolCall(
                        call_id=call.id,
                        name=call.function.name,
                        arguments=call.function.arguments,
                    )
                    for call in non_stream_response.message.tool_calls
                ]

                # Execute tool calls
                tool_results = self._execute_tool_calls(tool_calls)

                # Add tool result messages
                for result in tool_results:
                    current_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": result.call_id,
                            "content": [
                                {
                                    "type": "document",
                                    "document": {
                                        "data": result.result
                                        if not result.is_error
                                        else f"Error: {result.error}"
                                    },
                                }
                            ],
                        }
                    )

                # Continue to next iteration to check for more tool calls
                continue
            else:
                # No tool calls, make final streaming response
                final_response = self._make_provider_request(
                    current_messages, model, True, tools, **kwargs
                )

                # Stream the final response
                final_processor = StreamProcessor()
                for event in final_response:
                    chunk = self._process_provider_stream_event(event, final_processor)
                    if chunk:
                        yield chunk
                break

    def _process_event(
        self,
        event: Any,
        accumulated: str,
    ) -> tuple[str, ChimericStreamChunk[CohereChunk] | None]:
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


class CohereAsyncClient(ChimericAsyncClient[AsyncCohere, ChatResponse, CohereChunk]):
    """Asynchronous Cohere Client for interacting with Cohere models."""

    def __init__(self, api_key: str, tool_manager, **kwargs: Any) -> None:
        """Initializes the asynchronous Cohere client.

        Args:
            api_key: The Cohere API key.
            tool_manager: The tool manager instance.
            **kwargs: Additional keyword arguments for the Cohere client.
        """
        self._provider_name = "Cohere"
        super().__init__(api_key=api_key, tool_manager=tool_manager, **kwargs)

    def _get_async_client_type(self) -> type:
        """Returns the AsyncCohere client type.

        Returns:
            The type of the asynchronous Cohere client.
        """
        return AsyncCohere

    def _init_async_client(self, async_client_type: type, **kwargs: Any) -> AsyncCohere:
        """Initializes the asynchronous Cohere client.

        Args:
            async_client_type: The type of the async client to initialize.
            **kwargs: Additional keyword arguments for the client.

        Returns:
            An instance of the asynchronous Cohere client.
        """
        return AsyncCohere(api_key=self.api_key, **kwargs)

    def _get_capabilities(self) -> Capability:
        """Gets Cohere provider capabilities.

        Returns:
            Capability object indicating which features are supported:
            - streaming: True (supports real-time streaming responses)
            - tools: True (supports function calling)
        """
        return Capability(streaming=True, tools=True)

    async def _list_models_impl(self) -> list[ModelSummary]:
        """Lists available models from the Cohere API.

        Returns:
            A list of ModelSummary objects for available models.
        """
        models_response = await self.async_client.models.list()
        return [
            ModelSummary(
                id=str(getattr(model, "id", getattr(model, "name", "unknown"))),
                name=str(getattr(model, "name", "unknown")),
                metadata=model.model_dump() if hasattr(model, "model_dump") else {},
                owned_by="cohere",
            )
            for model in models_response.models
        ]

    def _messages_to_provider_format(self, messages: list[Message]) -> Any:
        """Converts standardized messages to Cohere format.

        Args:
            messages: A list of standardized Message objects.

        Returns:
            A list of messages in the format expected by the Cohere API.
        """
        return [msg.model_dump(exclude_none=True) for msg in messages]

    def _tools_to_provider_format(self, tools: list[Tool]) -> Any:
        """Converts standardized tools to Cohere format.

        Args:
            tools: A list of standardized Tool objects.

        Returns:
            A list of tools in the format expected by the Cohere API.
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

    async def _make_async_provider_request(
        self,
        messages: Any,
        model: str,
        stream: bool,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Makes the actual async API request to Cohere.

        Args:
            messages: The provider-formatted messages.
            model: The model to use for the request.
            stream: Whether to stream the response.
            tools: The provider-formatted tools, if any.
            **kwargs: Additional keyword arguments for the API call.

        Returns:
            The response from the Cohere API.
        """
        if stream:
            return self.async_client.chat_stream(
                model=model,
                messages=messages,
                tools=tools,
                **kwargs,
            )
        return await self.async_client.chat(
            model=model,
            messages=messages,
            tools=tools,
            **kwargs,
        )

    def _process_provider_stream_event(
        self, event: Any, processor: StreamProcessor
    ) -> ChimericStreamChunk[CohereChunk] | None:
        """Processes a Cohere stream event using the standardized processor.

        Args:
            event: The event from the Cohere stream.
            processor: The stream processor to manage state.

        Returns:
            A ChimericStreamChunk if the event is processed, otherwise None.
        """
        event_type = event.type

        # Handle tool plan deltas - these are part of the content stream for Cohere
        if event_type == "tool-plan-delta":
            delta = ""
            if hasattr(event, "delta") and hasattr(event.delta, "message"):
                delta = getattr(event.delta.message, "tool_plan", "")
            return create_stream_chunk(native_event=event, processor=processor, content_delta=delta)

        # Handle tool call start - register the tool call with the processor
        if event_type == "tool-call-start":
            if (
                hasattr(event, "delta")
                and hasattr(event.delta, "message")
                and hasattr(event.delta.message, "tool_calls")
                and event.delta.message.tool_calls
            ):
                tool_call = event.delta.message.tool_calls
                call_id = tool_call.id
                function_name = tool_call.function.name
                processor.process_tool_call_start(call_id, function_name)
            return create_stream_chunk(native_event=event, processor=processor)

        # Handle tool call argument deltas - accumulate arguments
        if event_type == "tool-call-delta":
            if (
                hasattr(event, "delta")
                and hasattr(event.delta, "message")
                and hasattr(event.delta.message, "tool_calls")
            ):
                tool_calls_delta = event.delta.message.tool_calls
                if hasattr(tool_calls_delta, "function") and hasattr(
                    tool_calls_delta.function, "arguments"
                ):
                    # We need to find which tool call this delta belongs to - use index
                    call_index = getattr(event, "index", 0)
                    # Since we don't have the call_id directly in delta events, we need to track by index
                    # For now, assume single tool call or track state differently
                    arguments_delta = tool_calls_delta.function.arguments

                    # Get the active tool call ID from processor state
                    tool_calls = list(processor.state.tool_calls.keys())
                    if tool_calls:
                        # Use the most recent tool call or match by index
                        call_id = (
                            tool_calls[call_index]
                            if call_index < len(tool_calls)
                            else tool_calls[-1]
                        )
                        processor.process_tool_call_delta(call_id, arguments_delta)
            return create_stream_chunk(native_event=event, processor=processor)

        # Handle tool call end - mark as complete
        if event_type == "tool-call-end":
            call_index = getattr(event, "index", 0)
            # Get the tool call ID to mark complete
            tool_calls = list(processor.state.tool_calls.keys())
            if tool_calls:
                call_id = tool_calls[call_index] if call_index < len(tool_calls) else tool_calls[-1]
                processor.process_tool_call_complete(call_id)
            return create_stream_chunk(native_event=event, processor=processor)

        # Handle content deltas (main response text after tools)
        if event_type == "content-delta":
            delta = ""
            if (
                hasattr(event, "delta")
                and hasattr(event.delta, "message")
                and hasattr(event.delta.message, "content")
                and hasattr(event.delta.message.content, "text")
            ):
                delta = event.delta.message.content.text
            return create_stream_chunk(native_event=event, processor=processor, content_delta=delta)

        # Handle message end
        if event_type == "message-end":
            finish_reason = "stop"
            if hasattr(event, "delta") and hasattr(event.delta, "finish_reason"):
                reason = event.delta.finish_reason
                if reason == "TOOL_CALL":
                    finish_reason = "tool_calls"
                elif reason == "COMPLETE":
                    finish_reason = "stop"
            return create_stream_chunk(
                native_event=event, processor=processor, finish_reason=finish_reason
            )

        # Handle other events like content-start, content-end, citation events, etc.
        if event_type in [
            "message-start",
            "content-start",
            "content-end",
            "citation-start",
            "citation-end",
        ]:
            return create_stream_chunk(native_event=event, processor=processor)

        return None

    def _extract_usage_from_response(self, response: Any) -> Usage:
        """Extracts usage information from Cohere response.

        Args:
            response: The response from the Cohere API.

        Returns:
            A Usage object containing token counts.
        """
        usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        if hasattr(response, "usage") and hasattr(response.usage, "tokens"):
            input_tokens = getattr(response.usage.tokens, "input_tokens", 0)
            output_tokens = getattr(response.usage.tokens, "output_tokens", 0)
            usage = Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )
        return usage

    def _extract_content_from_response(self, response: Any) -> str | list[Any]:
        """Extracts content from Cohere response.

        Args:
            response: The response from the Cohere API.

        Returns:
            The extracted content as a string or list.
        """
        content = ""
        if hasattr(response, "message") and hasattr(response.message, "content"):
            if isinstance(response.message.content, list) and len(response.message.content) > 0:
                content = response.message.content[0].text
            else:
                content = str(response.message.content)
        elif hasattr(response, "text"):
            content = response.text
        return content

    def _extract_tool_calls_from_response(self, response: Any) -> list[ToolCall] | None:
        """Extracts tool calls from Cohere response.

        Args:
            response: The response from the Cohere API.

        Returns:
            A list of ToolCall objects if tool calls are present, otherwise None.
        """
        if (
            hasattr(response, "message")
            and hasattr(response.message, "tool_calls")
            and response.message.tool_calls
        ):
            return [
                ToolCall(
                    call_id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                )
                for call in response.message.tool_calls
            ]
        return None

    def _update_messages_with_tool_calls(
        self,
        messages: list[Any],
        assistant_response: Any,
        tool_calls: list[ToolCall],
        tool_results: list[ToolExecutionResult],
    ) -> list[Any]:
        """Updates message history with assistant response and tool results for Cohere.

        Args:
            messages: The current list of provider-formatted messages.
            assistant_response: The native response from the assistant that contained the tool calls.
            tool_calls: The list of tool calls made by the assistant.
            tool_results: The results of executing the tool calls.

        Returns:
            An updated list of provider-formatted messages.
        """
        updated_messages = list(messages)

        # Add the assistant's response with tool calls
        if hasattr(assistant_response, "message") and hasattr(
            assistant_response.message, "tool_calls"
        ):
            updated_messages.append(
                {
                    "role": "assistant",
                    "tool_plan": getattr(assistant_response.message, "tool_plan", ""),
                    "tool_calls": assistant_response.message.tool_calls,
                }
            )

        # Add tool results
        for result in tool_results:
            updated_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": [
                        {
                            "type": "document",
                            "document": {"data": result.result or result.error or "No result"},
                        }
                    ],
                }
            )

        return updated_messages

    async def _handle_streaming_tool_calls(
        self,
        stream: Any,
        processor: StreamProcessor,
        messages: Any,
        model: str,
        tools: Any,
        **kwargs: Any,
    ) -> AsyncGenerator[ChimericStreamChunk[CohereChunk], None]:
        """Handles streaming with tool calls for Cohere's specific pattern.

        This method manages the conversation loop for tool calls in streaming mode,
        which involves making non-streaming requests to get tool calls, executing them,
        and then making a final streaming request for the user-facing response.

        Args:
            stream: The initial stream from the provider.
            processor: The stream processor.
            messages: The provider-formatted messages.
            model: The model to use.
            tools: The provider-formatted tools.
            **kwargs: Additional arguments for the API call.

        Yields:
            ChimericStreamChunk objects from the final response stream.
        """
        current_messages = list(messages)

        # Collect all events from the initial stream but don't yield content to user yet
        # Only process to detect if tools are being called
        stream_events = []
        async for event in stream:
            stream_events.append(event)

        # Check if the initial response contains tool calls by looking at events
        has_initial_tool_calls = any(event.type == "tool-call-start" for event in stream_events)

        # If no tool calls in initial response, stream it normally
        if not has_initial_tool_calls:
            accumulated_content = ""
            for event in stream_events:
                accumulated_content, chunk = self._process_event(event, accumulated_content)
                if chunk:
                    yield chunk
            return

        # Continue the conversation loop until no more tool calls are needed
        while True:
            # Get a fresh non-streaming response to check for tool calls
            non_stream_response = await self._make_async_provider_request(
                current_messages, model, False, tools, **kwargs
            )

            # Check for tool calls and process them
            if (
                hasattr(non_stream_response, "message")
                and hasattr(non_stream_response.message, "tool_calls")
                and non_stream_response.message.tool_calls
            ):
                # Add the assistant's tool call message with properly formatted tool calls
                current_messages.append(
                    {
                        "role": "assistant",
                        "tool_plan": getattr(non_stream_response.message, "tool_plan", ""),
                        "tool_calls": non_stream_response.message.tool_calls,
                    }
                )

                # Convert to ToolCall objects and execute using framework
                tool_calls = [
                    ToolCall(
                        call_id=call.id,
                        name=call.function.name,
                        arguments=call.function.arguments,
                    )
                    for call in non_stream_response.message.tool_calls
                ]

                # Execute tool calls using framework method
                tool_results = await self._execute_tool_calls(tool_calls)

                # Add tool result messages
                for result in tool_results:
                    current_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": result.call_id,
                            "content": [
                                {
                                    "type": "document",
                                    "document": {
                                        "data": result.result
                                        if not result.is_error
                                        else f"Error: {result.error}"
                                    },
                                }
                            ],
                        }
                    )

                # Continue to next iteration to check for more tool calls
                continue
            else:
                # No tool calls, make final streaming response
                final_response = await self._make_async_provider_request(
                    current_messages, model, True, tools, **kwargs
                )

                # Stream the final response
                final_processor = StreamProcessor()
                async for event in final_response:
                    chunk = self._process_provider_stream_event(event, final_processor)
                    if chunk:
                        yield chunk
                break

    def _process_event(
        self,
        event: Any,
        accumulated: str,
    ) -> tuple[str, ChimericStreamChunk[CohereChunk] | None]:
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
