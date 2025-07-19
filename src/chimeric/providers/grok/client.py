from typing import Any

from xai_sdk import AsyncClient, Client
from xai_sdk.chat import Chunk, Response, assistant, system, tool, tool_result, user

from chimeric.base import ChimericAsyncClient, ChimericClient
from chimeric.types import (
    Capability,
    ChimericCompletionResponse,
    ChimericFileUploadResponse,
    ChimericStreamChunk,
    Message,
    ModelSummary,
    Tool,
    ToolCall,
    ToolExecutionResult,
    Usage,
)
from chimeric.utils import StreamProcessor, create_completion_response, create_stream_chunk


class GrokClient(ChimericClient[Client, Response, Chunk, Any]):
    """Synchronous Grok Client for interacting with Grok's API using the xai-sdk.

    This client provides a unified interface for synchronous interactions with
    Grok's (xAI) API via the `xai-sdk` library. It returns `chimeric` response
    objects that wrap the native Grok responses and provides comprehensive tool
    calling support for both streaming and non-streaming operations.

    The client supports:
        - Text generation with Grok's conversational AI models
        - Multimodal inputs (text and images)
        - Function/tool calling with automatic execution
        - Streaming responses with real-time output
        - Model listing and metadata retrieval

    Note:
        Grok uses a unique chat-based API where conversations are managed
        through chat objects rather than stateless request/response patterns.
        File uploads are not currently supported.

    Example:
        ```python
        from chimeric.providers.grok import GrokClient
        from chimeric.tools import ToolManager

        tool_manager = ToolManager()
        client = GrokClient(api_key="your-xai-api-key", tool_manager=tool_manager)

        response = client.chat_completion(
            messages="What's the latest in AI research?",
            model="grok-beta"
        )
        print(response.common.content)
        ```

    Attributes:
        api_key (str): The xAI API key for authentication.
        tool_manager (ToolManager): Manager for handling tool registration and execution.
    """

    def __init__(self, api_key: str, tool_manager, **kwargs: Any) -> None:
        """Initialize the synchronous Grok client.

        Args:
            api_key: The xAI API key for authentication.
            tool_manager: The tool manager instance for handling function calls.
            **kwargs: Additional keyword arguments to pass to the xAI Client
                constructor, such as base_url, timeout, etc.

        Raises:
            ValueError: If api_key is None or empty.
            ProviderError: If client initialization fails.
        """
        self._provider_name = "Grok"
        super().__init__(api_key=api_key, tool_manager=tool_manager, **kwargs)

    def _get_client_type(self) -> type:
        """Get the synchronous Grok client class type.

        Returns:
            The Client class from the xai-sdk library.
        """
        return Client

    def _init_client(self, client_type: type, **kwargs: Any) -> Client:
        """Initialize the synchronous Grok client instance.

        Args:
            client_type: The xAI Client class to instantiate.
            **kwargs: Additional keyword arguments for client initialization,
                such as base_url, timeout, max_retries, etc.

        Returns:
            Configured synchronous xAI Client instance.

        Raises:
            ProviderError: If client initialization fails due to invalid
                credentials or configuration.
        """
        return Client(api_key=self.api_key, **kwargs)

    def _get_capabilities(self) -> Capability:
        """Get supported features for the Grok provider.

        Returns:
            Capability object indicating which features are supported:
                - multimodal: True (supports text and image inputs)
                - streaming: True (supports real-time streaming responses)
                - tools: True (supports function calling)
                - agents: False (agent workflows not currently supported)
                - files: False (no file upload support)

        Note:
            Grok provides advanced conversational AI capabilities with support
            for multimodal inputs and sophisticated tool calling.
        """
        return Capability(
            multimodal=True,
            streaming=True,
            tools=True,
            agents=False,
            files=False,
        )

    def _list_models_impl(self) -> list[ModelSummary]:
        """Lists available models from the Grok API."""
        models = self.client.models.list_language_models()
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

    def _messages_to_provider_format(self, messages: list[Message]) -> Any:
        """Converts standardized messages to Grok format."""
        converted = []
        for msg in messages:
            role = msg.role
            content = msg.content

            # Convert content to string if it's a list
            if isinstance(content, list):
                content = str(content)

            role_mapping = {
                "user": user,
                "system": system,
                "assistant": assistant,
            }
            message_func = role_mapping.get(role, user)
            converted.append(message_func(content))
        return converted

    def _tools_to_provider_format(self, tools: list[Tool]) -> Any:
        """Converts standardized tools to Grok format."""
        xai_tools = []
        for tool_obj in tools:
            xai_tools.append(
                tool(
                    name=tool_obj.name,
                    description=tool_obj.description,
                    parameters=tool_obj.parameters.model_dump() if tool_obj.parameters else {},
                )
            )
        return xai_tools

    def _make_provider_request(
        self,
        messages: Any,
        model: str,
        stream: bool,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Makes the actual API request to Grok."""
        chat = self.client.chat.create(
            model=model,
            tools=tools,
            tool_choice=kwargs.get("tool_choice", "auto") if tools else None,
            max_tokens=kwargs.get("max_tokens", 4096),
            **kwargs,
        )

        # Append messages to the chat
        for message in messages:
            chat.append(message)

        # For non-streaming, return the response directly
        # For streaming, return the chat.stream() iterator
        if stream:
            return chat.stream()
        return chat.sample()

    def _process_provider_stream_event(
        self, event: tuple[Response, Chunk], processor: StreamProcessor
    ) -> ChimericStreamChunk[Chunk] | None:
        """Processes a Grok stream event using the standardized processor.

        Args:
            event: A tuple of (response, chunk) from chat.stream()
            processor: The stream processor instance

        Returns:
            A ChimericStreamChunk or None if no content to process
        """
        _, chunk = event
        delta = chunk.content
        if delta:
            return create_stream_chunk(native_event=chunk, processor=processor, content_delta=delta)
        return None

    def _extract_usage_from_response(self, response: Response) -> Usage:
        """Extracts usage information from Grok response."""
        usage_data = getattr(response, "usage", None)
        if usage_data:
            if isinstance(usage_data, dict):
                return Usage(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                )
            return Usage(
                prompt_tokens=getattr(usage_data, "prompt_tokens", 0),
                completion_tokens=getattr(usage_data, "completion_tokens", 0),
                total_tokens=getattr(usage_data, "total_tokens", 0),
            )
        return Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    def _extract_content_from_response(self, response: Response) -> str | list[Any]:
        """Extracts content from Grok response."""
        return response.content or ""

    def _extract_tool_calls_from_response(self, response: Response) -> list[ToolCall] | None:
        """Extracts tool calls from Grok response."""
        if response.tool_calls:
            return [
                ToolCall(
                    call_id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                )
                for call in response.tool_calls
            ]
        return None

    def _handle_tool_calling_completion(
        self,
        messages: Any,
        model: str,
        tools: Any = None,
        **kwargs: Any,
    ) -> ChimericCompletionResponse[Response]:
        """Override tool calling for Grok's stateful chat pattern.

        Grok uses a stateful chat object approach where we:
        1. Create a chat object once
        2. Append messages and sample()
        3. If tool calls exist, append response and tool results to same chat
        4. Call sample() again on the same chat object
        """
        # Create the chat object
        chat = self.client.chat.create(
            model=model,
            tools=tools,
            tool_choice="auto" if tools else None,
            max_tokens=kwargs.get("max_tokens", 4096),
            **kwargs,
        )

        # Append initial messages to the chat
        for message in messages:
            chat.append(message)

        # Start iterative tool calling process
        all_tool_calls_metadata = []
        current_response = chat.sample()

        # Continue processing until no more tool calls
        while True:
            tool_calls = self._extract_tool_calls_from_response(current_response)
            if not tool_calls:
                # No more tools needed, we're done
                break

            # Append the assistant response with tool calls to the chat
            chat.append(current_response)

            # Execute tools and append results to the same chat object
            if self.tool_manager:
                tool_results = self._execute_tool_calls(tool_calls)

                # Append tool results to the chat object in Grok format
                for result in tool_results:
                    chat.append(
                        tool_result(
                            result=result.result or "No result"
                            if not result.is_error
                            else f"Error: {result.error or 'Unknown error'}"
                        )
                    )

            # Track all tool calls for metadata
            all_tool_calls_metadata.extend(tool_calls)

            # Get next response to check for more tool calls
            current_response = chat.sample()

        return create_completion_response(
            native_response=current_response,
            content=self._extract_content_from_response(current_response),
            usage=self._extract_usage_from_response(current_response),
            tool_calls=all_tool_calls_metadata,
        )

    def _handle_streaming_tool_calls(
        self,
        stream: Any,
        processor,
        messages: Any,
        model: str,
        tools: Any,
        **kwargs: Any,
    ):
        """Override streaming tool calling for Grok's stateful chat pattern.

        For Grok, we need to handle streaming with tools differently because
        of the stateful chat object pattern. We'll execute tools iteratively
        and stream the final response.

        Args:
            stream: The initial native stream (chat.stream() result)
            processor: StreamProcessor instance
            messages: Provider-formatted messages
            model: Model name
            tools: Provider-formatted tools
            **kwargs: Additional arguments
        """
        # For Grok, we ignore the initial stream and recreate the chat
        # because we need to handle tool calling iteratively

        # Create the chat object
        chat = self.client.chat.create(
            model=model,
            tools=tools,
            tool_choice=kwargs.get("tool_choice", "auto") if tools else None,
            max_tokens=kwargs.get("max_tokens", 4096),
            **kwargs,
        )

        # Append initial messages to the chat
        for message in messages:
            chat.append(message)

        # Start iterative tool calling process
        all_tool_calls_metadata = []
        current_response = chat.sample()

        # Continue processing until no more tool calls
        while True:
            tool_calls = self._extract_tool_calls_from_response(current_response)
            if not tool_calls:
                # No more tools needed, now stream the final response
                break

            # Append the assistant response with tool calls to the chat
            chat.append(current_response)

            # Execute tools and append results to the same chat object
            if self.tool_manager:
                tool_results = self._execute_tool_calls(tool_calls)

                # Append tool results to the chat object in Grok format
                for result in tool_results:
                    chat.append(
                        tool_result(
                            result=result.result or "No result"
                            if not result.is_error
                            else f"Error: {result.error or 'Unknown error'}"
                        )
                    )

            # Track all tool calls for metadata
            all_tool_calls_metadata.extend(tool_calls)

            # Get next response to check for more tool calls
            current_response = chat.sample()

        # Now stream the final response
        for response, chunk in chat.stream():
            chunk_obj = self._process_provider_stream_event((response, chunk), processor)
            if chunk_obj:
                yield chunk_obj

    def _update_messages_with_tool_calls(
        self,
        messages: list[Any],
        assistant_response: Any,
        tool_calls: list[ToolCall],
        tool_results: list[ToolExecutionResult],
    ) -> list[Any]:
        """Update message history with assistant response and tool calls.

        For Grok, this method is not used in our overridden tool calling flow
        since we override _handle_tool_calling_completion to work with Grok's
        stateful chat object pattern.

        Args:
            messages: Current list of messages in the conversation.
            assistant_response: The assistant's response that contained tool calls.
            tool_calls: List of ToolCall objects representing functions called
                by the assistant.
            tool_results: List of ToolExecutionResult objects containing the
                results of executing the tool calls.

        Returns:
            Updated list of messages. For Grok, this returns the original
            messages since chat state is handled internally by the chat object.
        """
        # Since we override _handle_tool_calling_completion, this method
        # won't be used in the tool calling flow, but we need to implement
        # it for interface compatibility
        return messages

    def _upload_file(self, **kwargs: Any) -> ChimericFileUploadResponse[Any]:
        """Grok does not support file uploads.

        Args:
            **kwargs: Not used.

        Raises:
            NotImplementedError: Grok does not support file uploads.
        """
        raise NotImplementedError("Grok does not support file uploads")


class GrokAsyncClient(ChimericAsyncClient[AsyncClient, Response, Chunk, Any]):
    """Asynchronous Grok Client for interacting with Grok's API using the xai-sdk."""

    def __init__(self, api_key: str, tool_manager, **kwargs: Any) -> None:
        """Initializes the asynchronous Grok client."""
        self._provider_name = "Grok"
        super().__init__(api_key=api_key, tool_manager=tool_manager, **kwargs)

    def _get_async_client_type(self) -> type:
        """Returns the xai-sdk AsyncClient type."""
        return AsyncClient

    def _init_async_client(self, async_client_type: type, **kwargs: Any) -> AsyncClient:
        """Initializes the asynchronous Grok client."""
        return AsyncClient(api_key=self.api_key, **kwargs)

    def _get_capabilities(self) -> Capability:
        """Gets Grok provider capabilities."""
        return Capability(
            multimodal=True,
            streaming=True,
            tools=True,
            agents=False,
            files=False,
        )

    async def _list_models_impl(self) -> list[ModelSummary]:
        """Lists available models from the Grok API."""
        models = await self.async_client.models.list_language_models()
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

    def _messages_to_provider_format(self, messages: list[Message]) -> Any:
        """Converts standardized messages to Grok format."""
        converted = []
        for msg in messages:
            role = msg.role
            content = msg.content

            # Convert content to string if it's a list
            if isinstance(content, list):
                content = str(content)

            role_mapping = {
                "user": user,
                "system": system,
                "assistant": assistant,
            }
            message_func = role_mapping.get(role, user)
            converted.append(message_func(content))
        return converted

    def _tools_to_provider_format(self, tools: list[Tool]) -> Any:
        """Converts standardized tools to Grok format."""
        xai_tools = []
        for tool_obj in tools:
            xai_tools.append(
                tool(
                    name=tool_obj.name,
                    description=tool_obj.description,
                    parameters=tool_obj.parameters.model_dump() if tool_obj.parameters else {},
                )
            )
        return xai_tools

    async def _make_async_provider_request(
        self,
        messages: Any,
        model: str,
        stream: bool,
        tools: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Makes the actual async API request to Grok."""
        chat = self.async_client.chat.create(
            model=model,
            tools=tools,
            tool_choice="auto" if tools else None,
            **kwargs,
        )

        # Append messages to the chat
        for message in messages:
            chat.append(message)

        # For non-streaming, return the response directly
        # For streaming, return the chat.stream() iterator
        if stream:
            return chat.stream()
        return await chat.sample()

    def _process_provider_stream_event(
        self, event: tuple[Response, Chunk], processor: StreamProcessor
    ) -> ChimericStreamChunk[Chunk] | None:
        """Processes a Grok stream event using the standardized processor.

        Args:
            event: A tuple of (response, chunk) from chat.stream()
            processor: The stream processor instance

        Returns:
            A ChimericStreamChunk or None if no content to process
        """
        _, chunk = event
        delta = chunk.content
        if delta:
            return create_stream_chunk(native_event=chunk, processor=processor, content_delta=delta)
        return None

    def _extract_usage_from_response(self, response: Response) -> Usage:
        """Extracts usage information from Grok response."""
        usage_data = getattr(response, "usage", None)
        if usage_data:
            if isinstance(usage_data, dict):
                return Usage(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                )
            return Usage(
                prompt_tokens=getattr(usage_data, "prompt_tokens", 0),
                completion_tokens=getattr(usage_data, "completion_tokens", 0),
                total_tokens=getattr(usage_data, "total_tokens", 0),
            )
        return Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    def _extract_content_from_response(self, response: Response) -> str | list[Any]:
        """Extracts content from Grok response."""
        return response.content or ""

    def _extract_tool_calls_from_response(self, response: Response) -> list[ToolCall] | None:
        """Extracts tool calls from Grok response."""
        if response.tool_calls:
            return [
                ToolCall(
                    call_id=call.id,
                    name=call.function.name,
                    arguments=call.function.arguments,
                )
                for call in response.tool_calls
            ]
        return None

    async def _handle_tool_calling_completion(
        self,
        messages: Any,
        model: str,
        tools: Any = None,
        **kwargs: Any,
    ) -> ChimericCompletionResponse[Response]:
        """Override async tool calling for Grok's stateful chat pattern.

        Grok uses a stateful chat object approach where we:
        1. Create a chat object once
        2. Append messages and sample()
        3. If tool calls exist, append response and tool results to same chat
        4. Call sample() again on the same chat object
        """
        # Create the chat object
        chat = self.async_client.chat.create(
            model=model,
            tools=tools,
            tool_choice=kwargs.get("tool_choice", "auto") if tools else None,
            max_tokens=kwargs.get("max_tokens", 4096),
            **kwargs,
        )

        # Append initial messages to the chat
        for message in messages:
            chat.append(message)

        # Start iterative tool calling process
        all_tool_calls_metadata = []
        current_response = await chat.sample()

        # Continue processing until no more tool calls
        while True:
            tool_calls = self._extract_tool_calls_from_response(current_response)
            if not tool_calls:
                # No more tools needed, we're done
                break

            # Append the assistant response with tool calls to the chat
            chat.append(current_response)

            # Execute tools and append results to the same chat object
            if self.tool_manager:
                tool_results = await self._execute_tool_calls(tool_calls)

                # Append tool results to the chat object in Grok format
                for result in tool_results:
                    chat.append(
                        tool_result(
                            result=result.result or "No result"
                            if not result.is_error
                            else f"Error: {result.error or 'Unknown error'}"
                        )
                    )

            # Track all tool calls for metadata
            all_tool_calls_metadata.extend(tool_calls)

            # Get next response to check for more tool calls
            current_response = await chat.sample()

        return create_completion_response(
            native_response=current_response,
            content=self._extract_content_from_response(current_response),
            usage=self._extract_usage_from_response(current_response),
            tool_calls=all_tool_calls_metadata,
        )

    async def _handle_streaming_tool_calls(
        self,
        stream: Any,
        processor,
        messages: Any,
        model: str,
        tools: Any,
        **kwargs: Any,
    ):
        """Override async streaming tool calling for Grok's stateful chat pattern.

        For Grok, we need to handle streaming with tools differently because
        of the stateful chat object pattern. We'll execute tools iteratively
        and stream the final response.

        Args:
            stream: The initial native stream (chat.stream() result)
            processor: StreamProcessor instance
            messages: Provider-formatted messages
            model: Model name
            tools: Provider-formatted tools
            **kwargs: Additional arguments
        """
        # For Grok, we ignore the initial stream and recreate the chat
        # because we need to handle tool calling iteratively

        # Create the chat object
        chat = self.async_client.chat.create(
            model=model,
            tools=tools,
            tool_choice=kwargs.get("tool_choice", "auto") if tools else None,
            max_tokens=kwargs.get("max_tokens", 4096),
            **kwargs,
        )

        # Append initial messages to the chat
        for message in messages:
            chat.append(message)

        # Start iterative tool calling process
        all_tool_calls_metadata = []
        current_response = await chat.sample()

        # Continue processing until no more tool calls
        while True:
            tool_calls = self._extract_tool_calls_from_response(current_response)
            if not tool_calls:
                # No more tools needed, now stream the final response
                break

            # Append the assistant response with tool calls to the chat
            chat.append(current_response)

            # Execute tools and append results to the same chat object
            if self.tool_manager:
                tool_results = await self._execute_tool_calls(tool_calls)

                # Append tool results to the chat object in Grok format
                for result in tool_results:
                    chat.append(
                        tool_result(
                            result=result.result or "No result"
                            if not result.is_error
                            else f"Error: {result.error or 'Unknown error'}"
                        )
                    )

            # Track all tool calls for metadata
            all_tool_calls_metadata.extend(tool_calls)

            # Get next response to check for more tool calls
            current_response = await chat.sample()

        # Now stream the final response
        async for response, chunk in chat.stream():
            chunk_obj = self._process_provider_stream_event((response, chunk), processor)
            if chunk_obj:
                yield chunk_obj

    def _update_messages_with_tool_calls(
        self,
        messages: list[Any],
        assistant_response: Any,
        tool_calls: list[ToolCall],
        tool_results: list[ToolExecutionResult],
    ) -> list[Any]:
        """Update message history with assistant response and tool calls.

        For Grok, this method is not used in our overridden tool calling flow
        since we override _handle_tool_calling_completion to work with
        Grok's stateful chat object pattern.

        Args:
            messages: Current list of messages in the conversation.
            assistant_response: The assistant's response that contained tool calls.
            tool_calls: List of ToolCall objects representing functions called
                by the assistant.
            tool_results: List of ToolExecutionResult objects containing the
                results of executing the tool calls.

        Returns:
            Updated list of messages. For Grok, this returns the original
            messages since chat state is handled internally by the chat object.
        """
        # Since we override _handle_async_tool_calling_completion, this method
        # won't be used in the tool calling flow, but we need to implement
        # it for interface compatibility
        return messages

    async def _upload_file(self, **kwargs: Any) -> ChimericFileUploadResponse[Any]:
        """Grok does not support file uploads.

        Args:
            **kwargs: Not used.

        Raises:
            NotImplementedError: Grok does not support file uploads.
        """
        raise NotImplementedError("Grok does not support file uploads")
