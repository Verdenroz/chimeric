"""HTTP transport layer for all provider communication.

HttpClient owns the httpx session and orchestrates the request/response cycle.
Adapters (pure functions) do the provider-specific serialization; this module
handles transport, tool-execution loops, SSE parsing, and error mapping.

Both sync and async paths share the same adapter calls
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from typing import TYPE_CHECKING, Any

import httpx

from .adapters import ADAPTER_REGISTRY
from .adapters.base import Adapter, EmbeddingAdapter, StreamState
from .adapters.google import GoogleAdapter
from .exceptions import AuthenticationError, ProviderError, RateLimitError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator

    from .config import ProviderConfig
    from .types import (
        CompletionResponse,
        EmbeddingResponse,
        Message,
        ModelSummary,
        StreamChunk,
        Tool,
        ToolCall,
    )

__all__ = ["HttpClient"]

# Maximum parallel tool executions per turn
_MAX_TOOL_WORKERS = 8


class HttpClient:
    """Shared sync + async HTTP client for all providers.

    Create one instance per Chimeric session; it holds connection pools for
    both sync and async transports.
    """

    def __init__(
        self,
        timeout: float = 60.0,
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self._timeout = timeout
        self._max_retries = max_retries
        self._default_headers: dict[str, str] = default_headers or {}
        # Lazy-initialised; created on first use to avoid overhead when unused
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Sync public API
    # ------------------------------------------------------------------

    def complete(
        self,
        config: ProviderConfig,
        api_key: str,
        messages: list[Message],
        model: str,
        stream: bool,
        tools: list[Tool] | None,
        **kwargs: Any,
    ) -> CompletionResponse | Generator[StreamChunk, None, None]:
        """Synchronous chat completion with optional tool-calling loop."""
        adapter = ADAPTER_REGISTRY[config.adapter]
        # Merge provider defaults with caller kwargs (caller wins)
        merged_kwargs = {**config.default_kwargs, **kwargs}

        if stream:
            return self._stream_sync(
                config, api_key, messages, model, tools, adapter, merged_kwargs
            )
        return self._complete_sync(config, api_key, messages, model, tools, adapter, merged_kwargs)

    def list_models(self, config: ProviderConfig, api_key: str) -> list[ModelSummary]:
        """Fetch and parse the provider's model list."""
        if config.models_path is None:
            return []

        adapter = ADAPTER_REGISTRY[config.adapter]
        url = _build_url(config, config.models_path, api_key)
        headers = _build_headers(config, api_key)

        response = self._sync().get(url, headers=headers)
        _raise_for_status(response, config.name)
        return adapter.parse_models_response(response.json())

    def embed(
        self,
        config: ProviderConfig,
        api_key: str,
        input: str | list[str],
        model: str,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Synchronous embedding request."""
        adapter = _get_embedding_adapter(config)
        batch = isinstance(input, list)
        body = adapter.build_embedding_request(input, model, **kwargs)
        url = _get_embedding_url(config, adapter, model, batch, api_key)
        headers = _build_headers(config, api_key)

        response = self._sync().post(url, json=body, headers=headers)
        _raise_for_status(response, config.name)
        return adapter.parse_embedding_response(response.json(), model, batch=batch)

    # ------------------------------------------------------------------
    # Async public API
    # ------------------------------------------------------------------

    async def acomplete(
        self,
        config: ProviderConfig,
        api_key: str,
        messages: list[Message],
        model: str,
        stream: bool,
        tools: list[Tool] | None,
        **kwargs: Any,
    ) -> CompletionResponse | AsyncGenerator[StreamChunk, None]:
        """Async chat completion with optional tool-calling loop."""
        adapter = ADAPTER_REGISTRY[config.adapter]
        merged_kwargs = {**config.default_kwargs, **kwargs}

        if stream:
            return self._stream_async(
                config, api_key, messages, model, tools, adapter, merged_kwargs
            )
        return await self._complete_async(
            config, api_key, messages, model, tools, adapter, merged_kwargs
        )

    async def alist_models(self, config: ProviderConfig, api_key: str) -> list[ModelSummary]:
        """Async fetch and parse the provider's model list."""
        if config.models_path is None:
            return []

        adapter = ADAPTER_REGISTRY[config.adapter]
        url = _build_url(config, config.models_path, api_key)
        headers = _build_headers(config, api_key)

        response = await self._async().get(url, headers=headers)
        _raise_for_status(response, config.name)
        return adapter.parse_models_response(response.json())

    async def aembed(
        self,
        config: ProviderConfig,
        api_key: str,
        input: str | list[str],
        model: str,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Async embedding request."""
        adapter = _get_embedding_adapter(config)
        batch = isinstance(input, list)
        body = adapter.build_embedding_request(input, model, **kwargs)
        url = _get_embedding_url(config, adapter, model, batch, api_key)
        headers = _build_headers(config, api_key)

        response = await self._async().post(url, json=body, headers=headers)
        _raise_for_status(response, config.name)
        return adapter.parse_embedding_response(response.json(), model, batch=batch)

    # ------------------------------------------------------------------
    # Sync internals
    # ------------------------------------------------------------------

    def _complete_sync(
        self,
        config: ProviderConfig,
        api_key: str,
        messages: list[Message],
        model: str,
        tools: list[Tool] | None,
        adapter: Adapter,
        kwargs: dict[str, Any],
    ) -> CompletionResponse:
        """Non-streaming request with automatic tool-calling loop."""
        current_messages = messages

        while True:
            body = adapter.build_request_body(current_messages, model, False, tools, **kwargs)
            url = _get_completion_url(config, adapter, model, stream=False, api_key=api_key)
            headers = _build_headers(config, api_key)

            response = self._sync().post(url, json=body, headers=headers)
            _raise_for_status(response, config.name)
            data = response.json()

            tool_calls = adapter.parse_tool_calls(data)
            if not tool_calls or not tools:
                return adapter.parse_response(data, model)

            # Execute tools in parallel, then loop back for the model's final answer
            results = _execute_tools_sync(tool_calls, tools)
            current_messages = adapter.build_tool_result_messages(
                current_messages, tool_calls, results
            )

    def _stream_sync(
        self,
        config: ProviderConfig,
        api_key: str,
        messages: list[Message],
        model: str,
        tools: list[Tool] | None,
        adapter: Adapter,
        kwargs: dict[str, Any],
    ) -> Generator[StreamChunk, None, None]:
        """Streaming request generator with tool-calling loop support."""
        current_messages = messages

        while True:
            body = adapter.build_request_body(current_messages, model, True, tools, **kwargs)
            url = _get_completion_url(config, adapter, model, stream=True, api_key=api_key)
            headers = _build_headers(config, api_key)
            state = StreamState()

            with self._sync().stream("POST", url, json=body, headers=headers) as response:
                _raise_for_status(response, config.name)
                for line in response.iter_lines():
                    chunk = _parse_sse_line(line, adapter, state)
                    if chunk is not None:
                        yield chunk

            tool_calls = adapter.finalize_stream(state)
            if not tool_calls or not tools:
                return

            results = _execute_tools_sync(tool_calls, tools)
            current_messages = adapter.build_tool_result_messages(
                current_messages, tool_calls, results
            )

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------

    async def _complete_async(
        self,
        config: ProviderConfig,
        api_key: str,
        messages: list[Message],
        model: str,
        tools: list[Tool] | None,
        adapter: Adapter,
        kwargs: dict[str, Any],
    ) -> CompletionResponse:
        """Async non-streaming request with tool-calling loop."""
        current_messages = messages

        while True:
            body = adapter.build_request_body(current_messages, model, False, tools, **kwargs)
            url = _get_completion_url(config, adapter, model, stream=False, api_key=api_key)
            headers = _build_headers(config, api_key)

            response = await self._async().post(url, json=body, headers=headers)
            _raise_for_status(response, config.name)
            data = response.json()

            tool_calls = adapter.parse_tool_calls(data)
            if not tool_calls or not tools:
                return adapter.parse_response(data, model)

            results = await _execute_tools_async(tool_calls, tools)
            current_messages = adapter.build_tool_result_messages(
                current_messages, tool_calls, results
            )

    async def _stream_async(
        self,
        config: ProviderConfig,
        api_key: str,
        messages: list[Message],
        model: str,
        tools: list[Tool] | None,
        adapter: Adapter,
        kwargs: dict[str, Any],
    ) -> AsyncGenerator[StreamChunk, None]:
        """Async streaming generator with tool-calling loop support."""
        current_messages = messages

        while True:
            body = adapter.build_request_body(current_messages, model, True, tools, **kwargs)
            url = _get_completion_url(config, adapter, model, stream=True, api_key=api_key)
            headers = _build_headers(config, api_key)
            state = StreamState()

            async with self._async().stream("POST", url, json=body, headers=headers) as response:
                _raise_for_status(response, config.name)
                async for line in response.aiter_lines():
                    chunk = _parse_sse_line(line, adapter, state)
                    if chunk is not None:
                        yield chunk

            tool_calls = adapter.finalize_stream(state)
            if not tool_calls or not tools:
                return

            results = await _execute_tools_async(tool_calls, tools)
            current_messages = adapter.build_tool_result_messages(
                current_messages, tool_calls, results
            )

    # ------------------------------------------------------------------
    # Client accessors (lazy init)
    # ------------------------------------------------------------------

    def _sync(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                timeout=self._timeout,
                transport=httpx.HTTPTransport(retries=self._max_retries),
                headers=self._default_headers,
            )
        return self._sync_client

    def _async(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                timeout=self._timeout,
                transport=httpx.AsyncHTTPTransport(retries=self._max_retries),
                headers=self._default_headers,
            )
        return self._async_client

    def close(self) -> None:
        """Close HTTP connections."""
        if self._sync_client:
            self._sync_client.close()

    async def aclose(self) -> None:
        """Async close HTTP connections."""
        if self._async_client:
            await self._async_client.aclose()

    def __enter__(self) -> HttpClient:
        """Enter sync context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Exit sync context manager and close connections."""
        self.close()

    async def __aenter__(self) -> HttpClient:
        """Enter async context manager."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Exit async context manager and close connections."""
        await self.aclose()


# ---------------------------------------------------------------------------
# Module-level helpers — pure functions, no I/O
# ---------------------------------------------------------------------------


def _get_embedding_adapter(config: ProviderConfig) -> EmbeddingAdapter:
    """Look up the adapter and verify it supports embeddings."""
    adapter = ADAPTER_REGISTRY[config.adapter]
    if not isinstance(adapter, EmbeddingAdapter):
        msg = f"Provider '{config.name}' does not support embeddings"
        raise ProviderError(error=None, provider=config.name, message=msg)
    if config.embedding_path is None:
        msg = f"Provider '{config.name}' has no embedding endpoint configured"
        raise ProviderError(error=None, provider=config.name, message=msg)
    return adapter


def _get_embedding_url(
    config: ProviderConfig,
    adapter: EmbeddingAdapter,
    model: str,
    batch: bool,
    api_key: str,
) -> str:
    """Resolve the embedding URL, letting the Google adapter substitute the model name."""
    if isinstance(adapter, GoogleAdapter):
        path = adapter.get_embedding_path(model, batch)
    else:
        if config.embedding_path is None:
            raise ValueError(f"Provider '{config.name}' has no embedding_path configured")
        path = config.embedding_path
    return _build_url(config, path, api_key)


def _build_headers(config: ProviderConfig, api_key: str) -> dict[str, str]:
    """Build request headers including auth (when auth_style == "header")."""
    headers: dict[str, str] = {"Content-Type": "application/json", **config.extra_headers}
    if config.auth_style == "header":
        headers[config.auth_header] = f"{config.auth_prefix}{api_key}"
    return headers


def _build_url(config: ProviderConfig, path: str, api_key: str) -> str:
    """Construct the full URL, appending the API key as a query param for Google."""
    base = config.base_url.rstrip("/")
    url = f"{base}{path}"
    if config.auth_style == "query_param":
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{config.auth_query_param}={api_key}"
    return url


def _get_completion_url(
    config: ProviderConfig,
    adapter: Adapter,
    model: str,
    stream: bool,
    api_key: str,
) -> str:
    """Resolve the completion URL, letting the Google adapter substitute the model name."""
    if isinstance(adapter, GoogleAdapter):
        path = adapter.get_completion_path(model, stream)
    elif stream and config.stream_path:
        path = config.stream_path
    else:
        path = config.completion_path
    return _build_url(config, path, api_key)


def _raise_for_status(response: httpx.Response, provider_name: str) -> None:
    """Map HTTP error codes to typed Chimeric exceptions."""
    if response.is_success:
        return

    status = response.status_code
    try:
        body = response.json()
        error_detail = body.get("error", {})
        message = error_detail.get("message") or body.get("message") or response.text
    except Exception:
        message = response.text or f"HTTP {status}"

    if status == 401:
        raise AuthenticationError(provider=provider_name, message=message)
    if status == 429:
        raise RateLimitError(provider=provider_name, message=message)
    raise ProviderError(
        error=None,
        provider=provider_name,
        message=message,
        status_code=status,
    )


def _parse_sse_line(line: str, adapter: Adapter, state: StreamState) -> StreamChunk | None:
    """Strip the "data: " prefix and delegate to the adapter for parsing."""
    line = line.strip()
    if not line:
        return None
    # Standard SSE format: "data: <payload>"
    if line.startswith("data: "):
        line = line[6:]
    elif line.startswith("data:"):
        line = line[5:]
    # Skip SSE comment lines and event/id/retry lines
    if (
        not line
        or line.startswith(":")
        or line.startswith("event:")
        or line.startswith("id:")
        or line.startswith("retry:")
    ):
        return None
    return adapter.parse_sse_event(line, state)


def _execute_tools_sync(
    tool_calls: list[ToolCall],
    tools: list[Tool],
) -> list[str]:
    """Execute tool calls in parallel using a thread pool, return results as strings."""
    tool_map = {t.name: t for t in tools}

    def _run(tc: ToolCall) -> str:
        tool = tool_map.get(tc.name)
        if tool is None or tool.function is None:
            return f"Error: tool '{tc.name}' not found"
        try:
            args = json.loads(tc.arguments) if tc.arguments else {}
            result = tool.function(**args)
            # Coroutines in sync context: run them in a new event loop
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)
            return str(result)
        except Exception as e:
            return f"Error executing {tc.name}: {e}"

    with ThreadPoolExecutor(max_workers=min(_MAX_TOOL_WORKERS, len(tool_calls))) as executor:
        return list(executor.map(_run, tool_calls))


async def _execute_tools_async(
    tool_calls: list[ToolCall],
    tools: list[Tool],
) -> list[str]:
    """Execute tool calls concurrently using asyncio.gather()."""
    tool_map = {t.name: t for t in tools}

    async def _run(tc: ToolCall) -> str:
        tool = tool_map.get(tc.name)
        if tool is None or tool.function is None:
            return f"Error: tool '{tc.name}' not found"
        try:
            args = json.loads(tc.arguments) if tc.arguments else {}
            result = tool.function(**args)
            if asyncio.iscoroutine(result):
                result = await result
            return str(result)
        except Exception as e:
            return f"Error executing {tc.name}: {e}"

    return list(await asyncio.gather(*[_run(tc) for tc in tool_calls]))
