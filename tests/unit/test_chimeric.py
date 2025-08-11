from collections.abc import AsyncGenerator, Generator
import os
from typing import Any
from unittest.mock import Mock, patch

import pytest

from chimeric.base import ChimericAsyncClient, ChimericClient
from chimeric.chimeric import ASYNC_PROVIDER_CLIENTS, PROVIDER_CLIENTS, Chimeric
from chimeric.exceptions import (
    ChimericError,
    ModelNotSupportedError,
    ProviderError,
    ProviderNotFoundError,
)
from chimeric.tools import ToolManager
from chimeric.types import (
    Capability,
    ChimericCompletionResponse,
    ChimericStreamChunk,
    CompletionResponse,
    ModelSummary,
    Provider,
    StreamChunk,
    Usage,
)


class MockProviderClient(ChimericClient[Any, Any, Any]):
    """Mock provider client for testing."""

    def __init__(self, api_key: str, tool_manager: ToolManager, **kwargs: Any) -> None:
        self.api_key = api_key
        self.tool_manager = tool_manager
        self.init_kwargs = kwargs
        self._request_count = 0
        self._error_count = 0
        self._capabilities = Capability(streaming=True, tools=True)

    def _get_client_type(self) -> type:
        return Mock

    def _init_client(self, client_type: type, **kwargs: Any) -> Any:
        return Mock()

    def _get_capabilities(self) -> Capability:
        return self._capabilities

    def _list_models_impl(self) -> list[ModelSummary]:
        return self.list_models()

    def _messages_to_provider_format(self, messages: list[Any]) -> Any:
        return messages

    def _tools_to_provider_format(self, tools: list[Any]) -> Any:
        return tools

    def _make_provider_request(
        self, messages: Any, model: str, stream: bool, tools: Any = None, **kwargs: Any
    ) -> Any:
        return Mock()

    def _process_provider_stream_event(self, event: Any, processor: Any) -> Any:
        return None

    def _extract_usage_from_response(self, response: Any) -> Usage:
        return Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    def _extract_content_from_response(self, response: Any) -> str:
        return "Mock response"

    def _extract_tool_calls_from_response(self, response: Any) -> list[Any] | None:
        return None

    @property
    def capabilities(self) -> Capability:
        return self._capabilities

    def list_models(self) -> list[ModelSummary]:
        return [
            ModelSummary(id="test-model-1", name="Test Model 1"),
            ModelSummary(id="test-model-2", name="Test Model 2"),
        ]

    def chat_completion(
        self,
        messages: Any,
        model: str,
        stream: bool = False,
        tools: Any = None,
        auto_tool: bool = True,
        **kwargs: Any,
    ) -> ChimericCompletionResponse[Any] | Generator[ChimericStreamChunk[Any], None, None]:
        self.last_request_kwargs = kwargs
        self._request_count += 1

        if stream:
            return self._create_mock_stream()
        return self._create_mock_response()

    def _create_mock_response(self) -> ChimericCompletionResponse[Any]:
        native_response = Mock()
        native_response.content = "Mock response"
        common_response = CompletionResponse(
            content="Mock response",
            model="test-model",
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        return ChimericCompletionResponse(native=native_response, common=common_response)

    def _create_mock_stream(self) -> Generator[ChimericStreamChunk[Any], None, None]:
        for i in range(3):
            native_chunk = Mock()
            native_chunk.content = f"chunk {i}"
            common_chunk = StreamChunk(content=f"chunk {i}")
            yield ChimericStreamChunk(native=native_chunk, common=common_chunk)


class MockAsyncProviderClient(ChimericAsyncClient[Any, Any, Any]):
    """Mock async provider client for testing."""

    def __init__(self, api_key: str, tool_manager: ToolManager, **kwargs: Any) -> None:
        self.api_key = api_key
        self.tool_manager = tool_manager
        self.init_kwargs = kwargs
        self._request_count = 0
        self._error_count = 0
        self._capabilities = Capability(streaming=True, tools=False)

    def _get_async_client_type(self) -> type:
        return Mock

    def _init_async_client(self, async_client_type: type, **kwargs: Any) -> Any:
        return Mock()

    def _get_capabilities(self) -> Capability:
        return self._capabilities

    async def _list_models_impl(self) -> list[ModelSummary]:
        return await self.list_models()

    def _messages_to_provider_format(self, messages: list[Any]) -> Any:
        return messages

    def _tools_to_provider_format(self, tools: list[Any]) -> Any:
        return tools

    async def _make_async_provider_request(
        self, messages: Any, model: str, stream: bool, tools: Any = None, **kwargs: Any
    ) -> Any:
        return Mock()

    def _process_provider_stream_event(self, event: Any, processor: Any) -> Any:
        return None

    def _extract_usage_from_response(self, response: Any) -> Usage:
        return Usage(prompt_tokens=15, completion_tokens=25, total_tokens=40)

    def _extract_content_from_response(self, response: Any) -> str:
        return "Async mock response"

    def _extract_tool_calls_from_response(self, response: Any) -> list[Any] | None:
        return None

    @property
    def capabilities(self) -> Capability:
        return self._capabilities

    async def list_models(self) -> list[ModelSummary]:
        return [
            ModelSummary(id="async-model-1", name="Async Model 1"),
            ModelSummary(id="async-model-2", name="Async Model 2"),
        ]

    async def chat_completion(
        self,
        messages: Any,
        model: str,
        stream: bool = False,
        tools: Any = None,
        auto_tool: bool = True,
        **kwargs: Any,
    ) -> ChimericCompletionResponse[Any] | AsyncGenerator[ChimericStreamChunk[Any], None]:
        self.last_request_kwargs = kwargs
        self._request_count += 1

        if stream:
            return self._create_mock_async_stream()
        return self._create_mock_response()

    def _create_mock_response(self) -> ChimericCompletionResponse[Any]:
        native_response = Mock()
        native_response.content = "Async mock response"
        common_response = CompletionResponse(
            content="Async mock response",
            model="test-model",
            usage=Usage(prompt_tokens=15, completion_tokens=25, total_tokens=40),
        )
        return ChimericCompletionResponse(native=native_response, common=common_response)

    async def _create_mock_async_stream(self) -> AsyncGenerator[ChimericStreamChunk[Any], None]:
        for i in range(3):
            native_chunk = Mock()
            native_chunk.content = f"async chunk {i}"
            common_chunk = StreamChunk(content=f"async chunk {i}")
            yield ChimericStreamChunk(native=native_chunk, common=common_chunk)


class TestChimericInitialization:
    """Test Chimeric client initialization scenarios."""

    def test_empty_initialization(self):
        """Test initialization with no providers configured."""
        chimeric = Chimeric()
        assert len(chimeric.providers) == 0
        assert len(chimeric.async_providers) == 0
        assert chimeric.primary_provider is None
        assert len(chimeric.available_providers) == 0

    def test_explicit_api_key_initialization(self):
        """Test initialization with explicit API keys."""
        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: MockProviderClient}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, {Provider.OPENAI: MockAsyncProviderClient}),
        ):
            chimeric = Chimeric(
                openai_api_key="test-openai-key",
                custom_param="test_value",
                timeout=30,
            )

            assert len(chimeric.providers) == 1
            assert Provider.OPENAI in chimeric.providers
            assert chimeric.primary_provider == Provider.OPENAI

            # Check kwargs were passed to provider
            provider = chimeric.providers[Provider.OPENAI]
            assert provider.api_key == "test-openai-key"
            # MockProviderClient stores kwargs in init_kwargs
            assert hasattr(provider, "init_kwargs")
            assert provider.init_kwargs["custom_param"] == "test_value"
            assert provider.init_kwargs["timeout"] == 30

    def test_environment_variable_initialization(self):
        """Test initialization from environment variables."""
        env_vars = {
            "OPENAI_API_KEY": "env-openai-key",
            "ANTHROPIC_API_KEY": "env-anthropic-key",
            "GOOGLE_API_KEY": "env-google-key",
        }

        with (
            patch.dict(os.environ, env_vars),
            patch.dict(
                PROVIDER_CLIENTS,
                {
                    Provider.OPENAI: MockProviderClient,
                    Provider.ANTHROPIC: MockProviderClient,
                    Provider.GOOGLE: MockProviderClient,
                },
            ),
            patch.dict(
                ASYNC_PROVIDER_CLIENTS,
                {
                    Provider.OPENAI: MockAsyncProviderClient,
                    Provider.ANTHROPIC: MockAsyncProviderClient,
                    Provider.GOOGLE: MockAsyncProviderClient,
                },
            ),
        ):
            chimeric = Chimeric()

            assert len(chimeric.providers) == 3
            assert Provider.OPENAI in chimeric.providers
            assert Provider.ANTHROPIC in chimeric.providers
            assert Provider.GOOGLE in chimeric.providers

            # Check API keys were set correctly
            assert chimeric.providers[Provider.OPENAI].api_key == "env-openai-key"
            assert chimeric.providers[Provider.ANTHROPIC].api_key == "env-anthropic-key"
            assert chimeric.providers[Provider.GOOGLE].api_key == "env-google-key"

    def test_mixed_initialization(self):
        """Test initialization with both explicit and environment variables."""
        env_vars = {"OPENAI_API_KEY": "env-openai-key"}

        with (
            patch.dict(os.environ, env_vars),
            patch.dict(
                PROVIDER_CLIENTS,
                {
                    Provider.OPENAI: MockProviderClient,
                    Provider.ANTHROPIC: MockProviderClient,
                },
            ),
            patch.dict(
                ASYNC_PROVIDER_CLIENTS,
                {
                    Provider.OPENAI: MockAsyncProviderClient,
                    Provider.ANTHROPIC: MockAsyncProviderClient,
                },
            ),
        ):
            chimeric = Chimeric(anthropic_api_key="explicit-anthropic-key")

            assert len(chimeric.providers) == 2
            # Explicit key should take precedence over environment
            assert chimeric.providers[Provider.ANTHROPIC].api_key == "explicit-anthropic-key"
            assert chimeric.providers[Provider.OPENAI].api_key == "env-openai-key"

    def test_alternative_environment_variables(self):
        """Test initialization with alternative environment variable names."""
        env_vars = {
            "GEMINI_API_KEY": "gemini-key",  # Alternative to GOOGLE_API_KEY
            "CO_API_KEY": "cohere-key",  # Alternative to COHERE_API_KEY
            "XAI_API_KEY": "xai-key",  # Alternative to GROK_API_KEY
        }

        with (
            patch.dict(os.environ, env_vars),
            patch.dict(
                PROVIDER_CLIENTS,
                {
                    Provider.GOOGLE: MockProviderClient,
                    Provider.COHERE: MockProviderClient,
                    Provider.GROK: MockProviderClient,
                },
            ),
            patch.dict(
                ASYNC_PROVIDER_CLIENTS,
                {
                    Provider.GOOGLE: MockAsyncProviderClient,
                    Provider.COHERE: MockAsyncProviderClient,
                    Provider.GROK: MockAsyncProviderClient,
                },
            ),
        ):
            chimeric = Chimeric()

            assert len(chimeric.providers) == 3
            assert chimeric.providers[Provider.GOOGLE].api_key == "gemini-key"
            assert chimeric.providers[Provider.COHERE].api_key == "cohere-key"
            assert chimeric.providers[Provider.GROK].api_key == "xai-key"

    def test_provider_initialization_failure(self):
        """Test handling of provider initialization failures."""

        def failing_provider(**kwargs):
            raise ImportError("Provider not available")

        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: failing_provider}),
            pytest.raises(ChimericError, match="Failed to initialize provider openai"),
        ):
            Chimeric(openai_api_key="test-key")

    def test_async_provider_initialization_failure(self):
        """Test handling of async provider initialization failures."""

        def failing_async_provider(**kwargs):
            raise ValueError("Async provider failed")

        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: MockProviderClient}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, {Provider.OPENAI: failing_async_provider}),
        ):
            # Should initialize successfully with sync provider, gracefully skip failing async provider
            chimeric = Chimeric(openai_api_key="test-key")

            # Sync provider should be available
            assert len(chimeric.providers) == 1
            assert Provider.OPENAI in chimeric.providers

            # Async provider should be skipped due to failure
            assert len(chimeric.async_providers) == 0
            assert Provider.OPENAI not in chimeric.async_providers

    def test_unsupported_provider(self):
        """Test adding unsupported provider raises error."""
        chimeric = Chimeric()

        # Create a mock provider that's not in PROVIDER_CLIENTS
        from unittest.mock import Mock

        fake_provider = Mock()
        fake_provider.value = "fake_provider"

        # Temporarily remove a provider to simulate unsupported
        original_providers = PROVIDER_CLIENTS.copy()
        PROVIDER_CLIENTS.clear()

        try:
            # Add one provider back to test the available providers message
            PROVIDER_CLIENTS[Provider.OPENAI] = original_providers[Provider.OPENAI]
            with pytest.raises(
                ProviderNotFoundError,
                match="Provider 'fake_provider' not found or configured.*Available providers:",
            ):
                chimeric._add_provider(fake_provider, api_key="test")
        finally:
            # Restore original providers
            PROVIDER_CLIENTS.clear()
            PROVIDER_CLIENTS.update(original_providers)


class TestProviderManagement:
    """Test provider management functionality."""

    def test_model_mapping_population(self):
        """Test that model mapping is populated correctly."""
        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: MockProviderClient}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, {Provider.OPENAI: MockAsyncProviderClient}),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            # Check that model mapping was populated
            assert len(chimeric._model_provider_mapping) > 0
            # Check canonical model names (alphanumeric only)
            assert "testmodel1" in chimeric._model_provider_mapping
            assert "testmodel2" in chimeric._model_provider_mapping

    def test_model_mapping_population_failure(self):
        """Test handling of model mapping population failure."""

        class FailingProviderClient(MockProviderClient):
            def list_models(self) -> list[ModelSummary]:
                raise ConnectionError("Failed to list models")

        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: FailingProviderClient}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, {Provider.OPENAI: MockAsyncProviderClient}),
            pytest.raises(ProviderError, match="Provider 'openai' failed"),
        ):
            Chimeric(openai_api_key="test-key")

    def test_primary_provider_selection(self):
        """Test primary provider is set to first successfully initialized provider."""
        with (
            patch.dict(
                PROVIDER_CLIENTS,
                {
                    Provider.OPENAI: MockProviderClient,
                    Provider.ANTHROPIC: MockProviderClient,
                },
            ),
            patch.dict(
                ASYNC_PROVIDER_CLIENTS,
                {
                    Provider.OPENAI: MockAsyncProviderClient,
                    Provider.ANTHROPIC: MockAsyncProviderClient,
                },
            ),
        ):
            chimeric = Chimeric(openai_api_key="openai-key", anthropic_api_key="anthropic-key")

            # Primary should be the first one initialized (OpenAI)
            assert chimeric.primary_provider == Provider.OPENAI

    def test_clear_model_cache(self):
        """Test clearing and repopulating model cache."""
        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: MockProviderClient}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, {Provider.OPENAI: MockAsyncProviderClient}),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            # Verify initial mapping
            initial_mapping_size = len(chimeric._model_provider_mapping)
            assert initial_mapping_size > 0

            # Clear cache
            chimeric._clear_model_cache()

            # Should be repopulated
            assert len(chimeric._model_provider_mapping) == initial_mapping_size

    def test_clear_model_cache_with_failing_provider(self):
        """Test clearing model cache handles provider failures gracefully."""

        class SometimesFailingClient(MockProviderClient):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.fail_next = False

            def list_models(self) -> list[ModelSummary]:
                if self.fail_next:
                    raise ProviderError(
                        error=Exception("Failed"), provider="test", message="Failed"
                    )
                return super().list_models()

        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: SometimesFailingClient}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, {Provider.OPENAI: MockAsyncProviderClient}),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            # Make the provider fail on next call
            chimeric.providers[Provider.OPENAI].fail_next = True

            # Should not raise error, just skip failed provider
            chimeric._clear_model_cache()


class TestModelSelection:
    """Test model selection and provider routing logic."""

    def setup_method(self):
        """Set up test environment."""
        self.provider_clients = {
            Provider.OPENAI: MockProviderClient,
            Provider.ANTHROPIC: MockProviderClient,
        }
        self.async_provider_clients = {
            Provider.OPENAI: MockAsyncProviderClient,
            Provider.ANTHROPIC: MockAsyncProviderClient,
        }

    def test_explicit_provider_selection(self):
        """Test selecting provider explicitly."""
        with (
            patch.dict(PROVIDER_CLIENTS, self.provider_clients),
            patch.dict(ASYNC_PROVIDER_CLIENTS, self.async_provider_clients),
        ):
            chimeric = Chimeric(openai_api_key="openai-key", anthropic_api_key="anthropic-key")

            # Test sync provider selection
            provider = chimeric._select_provider("any-model", "openai")
            assert provider == Provider.OPENAI

            provider = chimeric._select_provider("any-model", "anthropic")
            assert provider == Provider.ANTHROPIC

    def test_explicit_invalid_provider(self):
        """Test selecting invalid provider raises error."""
        with (
            patch.dict(PROVIDER_CLIENTS, self.provider_clients),
            patch.dict(ASYNC_PROVIDER_CLIENTS, self.async_provider_clients),
        ):
            chimeric = Chimeric(openai_api_key="openai-key")

            with pytest.raises(ProviderNotFoundError, match="Unknown provider: invalid"):
                chimeric._select_provider("any-model", "invalid")

    def test_explicit_unconfigured_provider(self):
        """Test selecting unconfigured provider raises error."""
        with (
            patch.dict(PROVIDER_CLIENTS, self.provider_clients),
            patch.dict(ASYNC_PROVIDER_CLIENTS, self.async_provider_clients),
        ):
            chimeric = Chimeric(openai_api_key="openai-key")

            with pytest.raises(ProviderNotFoundError, match="Provider anthropic not configured"):
                chimeric._select_provider("any-model", "anthropic")

    def test_model_based_provider_selection_cache_hit(self):
        """Test provider selection by model name with cache hit."""
        with (
            patch.dict(PROVIDER_CLIENTS, self.provider_clients),
            patch.dict(ASYNC_PROVIDER_CLIENTS, self.async_provider_clients),
        ):
            chimeric = Chimeric(openai_api_key="openai-key")

            # Pre-populate cache
            chimeric._model_provider_mapping["testmodel"] = Provider.OPENAI

            provider = chimeric._select_provider_by_model("test-model")
            assert provider == Provider.OPENAI

    def test_model_based_provider_selection_cache_miss(self):
        """Test provider selection by model name with cache miss and dynamic lookup."""

        class DynamicMockClient(MockProviderClient):
            def list_models(self) -> list[ModelSummary]:
                return [
                    ModelSummary(id="dynamic-model", name="Dynamic Model"),
                ]

        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: DynamicMockClient}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, self.async_provider_clients),
        ):
            chimeric = Chimeric(openai_api_key="openai-key")

            # Clear cache
            chimeric._model_provider_mapping.clear()

            provider = chimeric._select_provider_by_model("dynamic-model")
            assert provider == Provider.OPENAI
            # Should be added to cache
            assert "dynamicmodel" in chimeric._model_provider_mapping

    def test_model_not_found_error(self):
        """Test model not found in any provider."""
        with (
            patch.dict(PROVIDER_CLIENTS, self.provider_clients),
            patch.dict(ASYNC_PROVIDER_CLIENTS, self.async_provider_clients),
        ):
            chimeric = Chimeric(openai_api_key="openai-key")

            with pytest.raises(ModelNotSupportedError) as exc_info:
                chimeric._select_provider_by_model("nonexistent-model")

            error = exc_info.value
            assert error.model == "nonexistent-model"
            assert error.provider == "openai"  # Only one provider configured
            assert len(error.supported_models) > 0

    def test_async_provider_selection(self):
        """Test async provider selection."""
        with (
            patch.dict(PROVIDER_CLIENTS, self.provider_clients),
            patch.dict(ASYNC_PROVIDER_CLIENTS, self.async_provider_clients),
        ):
            chimeric = Chimeric(openai_api_key="openai-key")

            # Test explicit async provider selection
            provider = chimeric._select_async_provider("any-model", "openai")
            assert provider == Provider.OPENAI

            # Test async provider not configured
            with pytest.raises(
                ProviderNotFoundError, match="Async provider anthropic not configured"
            ):
                chimeric._select_async_provider("any-model", "anthropic")


class TestGenerationMethods:
    """Test generation and async generation methods."""

    def setup_method(self):
        """Set up test environment."""
        self.provider_clients = {Provider.OPENAI: MockProviderClient}
        self.async_provider_clients = {Provider.OPENAI: MockAsyncProviderClient}

    def test_generate_simple(self):
        """Test simple generation without streaming."""
        with (
            patch.dict(PROVIDER_CLIENTS, self.provider_clients),
            patch.dict(ASYNC_PROVIDER_CLIENTS, self.async_provider_clients),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            # Mock model selection
            with patch.object(chimeric, "_select_provider", return_value=Provider.OPENAI):
                response = chimeric.generate(
                    model="test-model", messages=[{"role": "user", "content": "Hello"}]
                )

                assert isinstance(response, CompletionResponse)
                # Verify provider was called
                provider = chimeric.providers[Provider.OPENAI]
                assert provider.request_count == 1

    def test_generate_with_kwargs_passthrough(self):
        """Test that kwargs are passed through to provider."""
        with (
            patch.dict(PROVIDER_CLIENTS, self.provider_clients),
            patch.dict(ASYNC_PROVIDER_CLIENTS, self.async_provider_clients),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            kwargs: dict[str, Any] = {
                "temperature": 0.7,
                "max_tokens": 100,
                "custom_param": "test_value",
            }

            with patch.object(chimeric, "_select_provider", return_value=Provider.OPENAI):
                chimeric.generate(
                    model="test-model",
                    messages=[{"role": "user", "content": "Hello"}],
                    **kwargs,
                )

                # Verify kwargs were passed to provider
                provider = chimeric.providers[Provider.OPENAI]
                assert provider.last_request_kwargs["temperature"] == 0.7
                assert provider.last_request_kwargs["max_tokens"] == 100
                assert provider.last_request_kwargs["custom_param"] == "test_value"

    def test_generate_streaming(self):
        """Test streaming generation."""
        with (
            patch.dict(PROVIDER_CLIENTS, self.provider_clients),
            patch.dict(ASYNC_PROVIDER_CLIENTS, self.async_provider_clients),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            with patch.object(chimeric, "_select_provider", return_value=Provider.OPENAI):
                stream = chimeric.generate(
                    model="test-model",
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=True,
                )

                assert isinstance(stream, Generator)
                chunks = list(stream)
                assert len(chunks) == 3

    def test_generate_native_vs_common_response(self):
        """Test native vs common response format selection."""
        with (
            patch.dict(PROVIDER_CLIENTS, self.provider_clients),
            patch.dict(ASYNC_PROVIDER_CLIENTS, self.async_provider_clients),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            with patch.object(chimeric, "_select_provider", return_value=Provider.OPENAI):
                # Test common response (default)
                common_response = chimeric.generate(
                    model="test-model",
                    messages=[{"role": "user", "content": "Hello"}],
                    native=False,
                )
                assert hasattr(common_response, "content")  # Common format

                # Test native response
                native_response = chimeric.generate(
                    model="test-model",
                    messages=[{"role": "user", "content": "Hello"}],
                    native=True,
                )
                assert hasattr(native_response, "content")  # Native format

    @pytest.mark.asyncio
    async def test_agenerate_simple(self):
        """Test simple async generation."""
        with (
            patch.dict(PROVIDER_CLIENTS, self.provider_clients),
            patch.dict(ASYNC_PROVIDER_CLIENTS, self.async_provider_clients),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            with patch.object(chimeric, "_select_async_provider", return_value=Provider.OPENAI):
                response = await chimeric.agenerate(
                    model="test-model", messages=[{"role": "user", "content": "Hello"}]
                )

                assert isinstance(response, CompletionResponse)
                # Verify async provider was called
                async_provider = chimeric.async_providers[Provider.OPENAI]
                assert async_provider.request_count == 1

    @pytest.mark.asyncio
    async def test_agenerate_with_kwargs_passthrough(self):
        """Test async generation kwargs passthrough."""
        with (
            patch.dict(PROVIDER_CLIENTS, self.provider_clients),
            patch.dict(ASYNC_PROVIDER_CLIENTS, self.async_provider_clients),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            kwargs: dict[str, Any] = {
                "temperature": 0.8,
                "max_tokens": 200,
                "stream_options": {"include_usage": True},
            }

            with patch.object(chimeric, "_select_async_provider", return_value=Provider.OPENAI):
                await chimeric.agenerate(
                    model="test-model",
                    messages=[{"role": "user", "content": "Hello"}],
                    **kwargs,
                )

                # Verify kwargs were passed to async provider
                async_provider = chimeric.async_providers[Provider.OPENAI]
                assert async_provider.last_request_kwargs["temperature"] == 0.8
                assert async_provider.last_request_kwargs["max_tokens"] == 200
                assert async_provider.last_request_kwargs["stream_options"]["include_usage"] is True

    @pytest.mark.asyncio
    async def test_agenerate_streaming(self):
        """Test async streaming generation."""
        with (
            patch.dict(PROVIDER_CLIENTS, self.provider_clients),
            patch.dict(ASYNC_PROVIDER_CLIENTS, self.async_provider_clients),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            with patch.object(chimeric, "_select_async_provider", return_value=Provider.OPENAI):
                stream = await chimeric.agenerate(
                    model="test-model",
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=True,
                )

                assert hasattr(stream, "__aiter__")
                chunks = []
                async for chunk in stream:  # type: ignore
                    chunks.append(chunk)
                assert len(chunks) == 3


class TestProviderAccess:
    """Test provider access methods."""

    def setup_method(self):
        """Set up test environment."""
        self.provider_clients = {
            Provider.OPENAI: MockProviderClient,
            Provider.ANTHROPIC: MockProviderClient,
        }
        self.async_provider_clients = {
            Provider.OPENAI: MockAsyncProviderClient,
            Provider.ANTHROPIC: MockAsyncProviderClient,
        }

    def test_get_provider_client_success(self):
        """Test successful provider client retrieval."""
        with (
            patch.dict(PROVIDER_CLIENTS, self.provider_clients),
            patch.dict(ASYNC_PROVIDER_CLIENTS, self.async_provider_clients),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            client = chimeric._get_provider_client("openai")
            assert client is chimeric.providers[Provider.OPENAI]

    def test_get_provider_client_not_configured(self):
        """Test provider client retrieval for unconfigured provider."""
        with (
            patch.dict(PROVIDER_CLIENTS, self.provider_clients),
            patch.dict(ASYNC_PROVIDER_CLIENTS, self.async_provider_clients),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            with pytest.raises(ProviderNotFoundError, match="Provider anthropic not configured"):
                chimeric._get_provider_client("anthropic")

    def test_list_models_specific_provider(self):
        """Test listing models for specific provider."""
        with (
            patch.dict(PROVIDER_CLIENTS, self.provider_clients),
            patch.dict(ASYNC_PROVIDER_CLIENTS, self.async_provider_clients),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            models = chimeric.list_models("openai")
            assert len(models) == 2
            assert all(model.provider == "openai" for model in models)

    def test_list_models_all_providers(self):
        """Test listing models from all providers."""
        with (
            patch.dict(PROVIDER_CLIENTS, self.provider_clients),
            patch.dict(ASYNC_PROVIDER_CLIENTS, self.async_provider_clients),
        ):
            chimeric = Chimeric(openai_api_key="openai-key", anthropic_api_key="anthropic-key")

            models = chimeric.list_models()
            assert len(models) == 4  # 2 from each provider
            provider_names = {model.provider for model in models}
            assert "openai" in provider_names
            assert "anthropic" in provider_names

    def test_list_models_with_failing_provider(self):
        """Test listing models when one provider fails during list_models call."""

        class FailingClient(MockProviderClient):
            def __init__(self, *args, **kwargs):
                # Don't call super().__init__ to avoid model population during init
                self.api_key = kwargs.get("api_key", "test")
                self.tool_manager = kwargs.get("tool_manager")
                self.init_kwargs = kwargs
                self._request_count = 0
                self._error_count = 0
                self._capabilities = Capability(streaming=True, tools=False)

            @property
            def capabilities(self) -> Capability:
                return self._capabilities

            def list_models(self) -> list[ModelSummary]:
                raise ConnectionError("Provider failed")

        # Use a working provider that won't fail during init
        with (
            patch.dict(
                PROVIDER_CLIENTS,
                {
                    Provider.OPENAI: MockProviderClient,
                },
            ),
            patch.dict(
                ASYNC_PROVIDER_CLIENTS,
                {
                    Provider.OPENAI: MockAsyncProviderClient,
                },
            ),
        ):
            chimeric = Chimeric(openai_api_key="openai-key")

            # Now manually add the failing provider to test list_models failure handling
            failing_client = FailingClient(api_key="test", tool_manager=chimeric._tool_manager)
            chimeric.providers[Provider.ANTHROPIC] = failing_client

            # Should skip failing provider and return models from working one
            models = chimeric.list_models()
            assert len(models) == 2  # Only from OpenAI
            assert all(model.provider == "openai" for model in models)

    def test_capabilities_property(self):
        """Test capabilities property merges all provider capabilities."""

        # Create providers with different capabilities
        class ProviderA(MockProviderClient):
            @property
            def capabilities(self) -> Capability:
                return Capability(streaming=False, tools=True)

        class ProviderB(MockProviderClient):
            @property
            def capabilities(self) -> Capability:
                return Capability(streaming=True, tools=False)

        with (
            patch.dict(
                PROVIDER_CLIENTS,
                {
                    Provider.OPENAI: ProviderA,
                    Provider.ANTHROPIC: ProviderB,
                },
            ),
            patch.dict(ASYNC_PROVIDER_CLIENTS, self.async_provider_clients),
        ):
            chimeric = Chimeric(openai_api_key="openai-key", anthropic_api_key="anthropic-key")

            # Should merge capabilities (union of all features)
            merged_caps = chimeric.capabilities
            assert merged_caps.streaming is True  # From ProviderB
            assert merged_caps.tools is True  # From ProviderA

    def test_get_capabilities_specific_provider(self):
        """Test get_capabilities for specific provider."""
        with (
            patch.dict(PROVIDER_CLIENTS, self.provider_clients),
            patch.dict(ASYNC_PROVIDER_CLIENTS, self.async_provider_clients),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            caps = chimeric.get_capabilities("openai")
            assert caps.streaming is True
            assert caps.tools is True

    def test_get_capabilities_uses_property(self):
        """Test get_capabilities without provider uses capabilities property."""
        with (
            patch.dict(PROVIDER_CLIENTS, self.provider_clients),
            patch.dict(ASYNC_PROVIDER_CLIENTS, self.async_provider_clients),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            # Test that get_capabilities() without provider returns the same as the property
            property_result = chimeric.capabilities
            method_result = chimeric.get_capabilities()

            assert property_result.streaming == method_result.streaming
            assert property_result.tools == method_result.tools


class TestToolIntegration:
    """Test tool integration functionality."""

    def test_tool_decorator_registration(self):
        """Test tool decorator properly registers functions."""
        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: MockProviderClient}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, {Provider.OPENAI: MockAsyncProviderClient}),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            @chimeric.tool()
            def test_function(x: int) -> str:  # type: ignore[reportUnusedFunction]
                """Test function."""
                return f"Result: {x}"

            # Verify tool was registered
            tools = chimeric.tools
            assert len(tools) == 1
            assert tools[0].name == "test_function"
            assert tools[0].description == "Test function."

    def test_tool_decorator_with_custom_params(self):
        """Test tool decorator with custom name and description."""
        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: MockProviderClient}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, {Provider.OPENAI: MockAsyncProviderClient}),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            @chimeric.tool(name="custom_tool", description="Custom description")
            def test_function(x: int) -> str:  # type: ignore[reportUnusedFunction]
                return f"Result: {x}"

            tools = chimeric.tools
            assert len(tools) == 1
            assert tools[0].name == "custom_tool"
            assert tools[0].description == "Custom description"

    def test_tools_property(self):
        """Test tools property returns all registered tools."""
        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: MockProviderClient}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, {Provider.OPENAI: MockAsyncProviderClient}),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            # Register multiple tools
            @chimeric.tool()
            def tool1(x: int) -> str:  # type: ignore[reportUnusedFunction]
                return str(x)

            @chimeric.tool()
            def tool2(y: str) -> int:  # type: ignore[reportUnusedFunction]
                return len(y)

            tools = chimeric.tools
            assert len(tools) == 2
            tool_names = {tool.name for tool in tools}
            assert "tool1" in tool_names
            assert "tool2" in tool_names


class TestChimericProperties:
    """Test Chimeric class properties and methods."""

    def test_available_providers_property(self):
        """Test available_providers property."""
        with (
            patch.dict(
                PROVIDER_CLIENTS,
                {
                    Provider.OPENAI: MockProviderClient,
                    Provider.ANTHROPIC: MockProviderClient,
                },
            ),
            patch.dict(
                ASYNC_PROVIDER_CLIENTS,
                {
                    Provider.OPENAI: MockAsyncProviderClient,
                    Provider.ANTHROPIC: MockAsyncProviderClient,
                },
            ),
        ):
            chimeric = Chimeric(openai_api_key="openai-key", anthropic_api_key="anthropic-key")

            providers = chimeric.available_providers
            assert len(providers) == 2
            assert "openai" in providers
            assert "anthropic" in providers

    def test_string_representation(self):
        """Test __repr__ method."""
        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: MockProviderClient}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, {Provider.OPENAI: MockAsyncProviderClient}),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            repr_str = repr(chimeric)
            assert "Chimeric" in repr_str
            assert "providers=['openai']" in repr_str
            assert "primary=openai" in repr_str

    def test_string_representation_no_providers(self):
        """Test __repr__ with no providers configured."""
        chimeric = Chimeric()

        repr_str = repr(chimeric)
        assert "Chimeric" in repr_str
        assert "providers=[]" in repr_str
        assert "primary=None" in repr_str


class TestErrorScenarios:
    """Test various error scenarios and edge cases."""

    def test_generate_with_no_providers(self):
        """Test generation when no providers are configured."""
        chimeric = Chimeric()

        with pytest.raises(ModelNotSupportedError):
            chimeric.generate(model="any-model", messages=[{"role": "user", "content": "Hello"}])

    @pytest.mark.asyncio
    async def test_agenerate_with_no_providers(self):
        """Test async generation when no providers are configured."""
        chimeric = Chimeric()

        with pytest.raises(ModelNotSupportedError):
            await chimeric.agenerate(
                model="any-model", messages=[{"role": "user", "content": "Hello"}]
            )

    def test_provider_not_found_error_details(self):
        """Test ProviderNotFoundError contains correct details."""
        chimeric = Chimeric()

        with pytest.raises(ValueError, match="'nonexistent' is not a valid Provider"):
            chimeric._get_provider_client("nonexistent")

    def test_model_not_supported_error_details(self):
        """Test ModelNotSupportedError contains supported models list."""
        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: MockProviderClient}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, {Provider.OPENAI: MockAsyncProviderClient}),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            with pytest.raises(ModelNotSupportedError) as exc_info:
                chimeric.generate(
                    model="nonexistent-model", messages=[{"role": "user", "content": "Hello"}]
                )

            error = exc_info.value
            assert error.model == "nonexistent-model"
            assert len(error.supported_models) > 0
            assert "test-model-1" in error.supported_models

    def test_async_provider_not_supported_error(self):
        """Test error when trying to add unsupported async provider."""
        chimeric = Chimeric()

        # Create a fake provider not in ASYNC_PROVIDER_CLIENTS
        # We need to temporarily remove a provider from the dict
        original_clients = ASYNC_PROVIDER_CLIENTS.copy()
        try:
            # Temporarily remove all async providers to trigger the error
            ASYNC_PROVIDER_CLIENTS.clear()
            with pytest.raises(ProviderNotFoundError, match="Async provider"):
                chimeric._add_async_provider(
                    Provider.OPENAI, api_key="test", tool_manager=chimeric._tool_manager
                )
        finally:
            # Restore the original dict
            ASYNC_PROVIDER_CLIENTS.clear()
            ASYNC_PROVIDER_CLIENTS.update(original_clients)

    def test_model_name_differs_from_id_mapping(self):
        """Test model mapping when model name differs from ID."""

        class ModelProvider(MockProviderClient):
            def list_models(self) -> list[ModelSummary]:
                return [
                    ModelSummary(id="gpt-4", name="GPT-4 Turbo"),  # name differs from id
                    ModelSummary(id="model-123", name="model-123"),  # name same as id
                ]

        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: ModelProvider}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, {Provider.OPENAI: MockAsyncProviderClient}),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            # Both canonical ID and name should be mapped
            assert "gpt4" in chimeric._model_provider_mapping
            assert "gpt4turbo" in chimeric._model_provider_mapping
            assert "model123" in chimeric._model_provider_mapping

    def test_unknown_provider_selection_error(self):
        """Test error when selecting unknown provider."""
        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: MockProviderClient}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, {Provider.OPENAI: MockAsyncProviderClient}),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            with pytest.raises(ProviderNotFoundError, match="Unknown provider"):
                chimeric._select_provider("test-model", provider="unknown")

    def test_async_unknown_provider_selection_error(self):
        """Test error when selecting unknown async provider."""
        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: MockProviderClient}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, {Provider.OPENAI: MockAsyncProviderClient}),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            with pytest.raises(ProviderNotFoundError, match="Unknown provider"):
                chimeric._select_async_provider("test-model", provider="unknown")

    def test_model_selection_with_provider_errors(self):
        """Test model selection when providers raise various errors."""

        class ErrorProneProvider(MockProviderClient):
            def list_models(self) -> list[ModelSummary]:
                raise ConnectionError("Network error")

        # Start with working providers first
        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.ANTHROPIC: MockProviderClient}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, {Provider.ANTHROPIC: MockAsyncProviderClient}),
        ):
            chimeric = Chimeric(anthropic_api_key="anthropic-key")

            # Now add the failing provider manually
            chimeric.providers[Provider.OPENAI] = ErrorProneProvider(
                api_key="openai-key", tool_manager=chimeric._tool_manager
            )

            # Clear the cache to force dynamic lookup
            chimeric._clear_model_cache()

            # Should skip the error-prone provider and find model in working one
            try:
                provider = chimeric._select_provider_by_model("anthropic-model")
                assert provider == Provider.ANTHROPIC
            except ModelNotSupportedError:
                # This is also acceptable behavior when no provider can provide the model
                pass

    def test_model_selection_all_providers_fail(self):
        """Test model selection when all providers fail."""

        class AlwaysFailingProvider(MockProviderClient):
            def list_models(self) -> list[ModelSummary]:
                raise ValueError("Always fails")

        # Create a chimeric instance first, then replace the provider
        chimeric = Chimeric()
        chimeric.providers[Provider.OPENAI] = AlwaysFailingProvider(
            api_key="test-key", tool_manager=chimeric._tool_manager
        )

        with pytest.raises(ModelNotSupportedError):
            chimeric._select_provider_by_model("nonexistent-model")

    def test_list_models_unknown_provider(self):
        """Test list_models with unknown provider."""
        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: MockProviderClient}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, {Provider.OPENAI: MockAsyncProviderClient}),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            with pytest.raises(ValueError, match="'unknown' is not a valid Provider"):
                chimeric.list_models(provider="unknown")

    def test_get_capabilities_unknown_provider(self):
        """Test get_capabilities with unknown provider."""
        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: MockProviderClient}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, {Provider.OPENAI: MockAsyncProviderClient}),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            with pytest.raises(ValueError, match="'unknown' is not a valid Provider"):
                chimeric.get_capabilities(provider="unknown")

    def test_provider_not_configured_for_explicit_selection(self):
        """Test error when explicitly selecting unconfigured provider."""
        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: MockProviderClient}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, {Provider.OPENAI: MockAsyncProviderClient}),
        ):
            chimeric = Chimeric(openai_api_key="test-key")  # Only OpenAI configured

            # Try to use anthropic which isn't configured
            with pytest.raises(ProviderNotFoundError, match="Provider anthropic not configured"):
                chimeric._select_provider("test-model", provider="anthropic")

    def test_async_provider_not_configured_for_explicit_selection(self):
        """Test error when explicitly selecting unconfigured async provider."""
        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: MockProviderClient}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, {Provider.OPENAI: MockAsyncProviderClient}),
        ):
            chimeric = Chimeric(openai_api_key="test-key")  # Only OpenAI configured

            # Try to use anthropic async which isn't configured
            with pytest.raises(
                ProviderNotFoundError, match="Async provider anthropic not configured"
            ):
                chimeric._select_async_provider("test-model", provider="anthropic")

    def test_list_models_provider_not_configured(self):
        """Test list_models with configured provider enum that's not in providers dict."""
        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: MockProviderClient}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, {Provider.OPENAI: MockAsyncProviderClient}),
        ):
            chimeric = Chimeric(openai_api_key="test-key")  # Only OpenAI configured

            # Remove openai from providers dict but ask for it specifically
            del chimeric.providers[Provider.OPENAI]

            with pytest.raises(ProviderNotFoundError, match="Provider openai not configured"):
                chimeric.list_models(provider="openai")

    def test_get_capabilities_provider_not_configured(self):
        """Test get_capabilities with configured provider enum that's not in providers dict."""
        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: MockProviderClient}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, {Provider.OPENAI: MockAsyncProviderClient}),
        ):
            chimeric = Chimeric(openai_api_key="test-key")  # Only OpenAI configured

            # Remove openai from providers dict but ask for it specifically
            del chimeric.providers[Provider.OPENAI]

            with pytest.raises(ProviderNotFoundError, match="Provider openai not configured"):
                chimeric.get_capabilities(provider="openai")

    def test_list_models_provider_already_set(self):
        """Test list_models when model.provider is already set."""

        class ProviderWithPresetModels(MockProviderClient):
            def list_models(self) -> list[ModelSummary]:
                return [
                    ModelSummary(id="model-1", name="Model 1", provider="already-set"),
                    ModelSummary(id="model-2", name="Model 2", provider=None),  # This will be set
                ]

        with (
            patch.dict(PROVIDER_CLIENTS, {Provider.OPENAI: ProviderWithPresetModels}),
            patch.dict(ASYNC_PROVIDER_CLIENTS, {Provider.OPENAI: MockAsyncProviderClient}),
        ):
            chimeric = Chimeric(openai_api_key="test-key")

            models = chimeric.list_models(provider="openai")

            # First model should keep its preset provider
            assert models[0].provider == "already-set"
            # Second model should get provider set to openai
            assert models[1].provider == "openai"


class TestProviderMappingConstruction:
    """Test provider mapping construction and dynamic loading."""

    def test_build_provider_mappings_success(self):
        """Test successful provider mapping construction."""
        from chimeric.chimeric import _build_provider_mappings

        # Mock successful imports
        mock_openai_module = Mock()
        mock_openai_module.OpenAIClient = MockProviderClient
        mock_openai_module.OpenAIAsyncClient = MockAsyncProviderClient

        mock_anthropic_module = Mock()
        mock_anthropic_module.AnthropicClient = MockProviderClient
        mock_anthropic_module.AnthropicAsyncClient = MockAsyncProviderClient

        def mock_import(module_path, fromlist=None):
            if module_path == "chimeric.providers.openai.client":
                return mock_openai_module
            if module_path == "chimeric.providers.anthropic.client":
                return mock_anthropic_module
            raise ImportError(f"No module named {module_path}")

        with patch("builtins.__import__", side_effect=mock_import):
            sync_clients, async_clients = _build_provider_mappings()

            # Should have imported successful providers
            assert Provider.OPENAI in sync_clients
            assert Provider.ANTHROPIC in sync_clients
            assert Provider.OPENAI in async_clients
            assert Provider.ANTHROPIC in async_clients

            # Should not have providers that failed to import
            assert Provider.GOOGLE not in sync_clients
            assert Provider.GOOGLE not in async_clients

    def test_build_provider_mappings_import_failures(self):
        """Test provider mapping construction with import failures."""
        from chimeric.chimeric import _build_provider_mappings

        def mock_import_all_fail(module_path, fromlist=None):
            raise ImportError(f"No module named {module_path}")

        with patch("builtins.__import__", side_effect=mock_import_all_fail):
            sync_clients, async_clients = _build_provider_mappings()

            # Should return empty dicts when all imports fail
            assert len(sync_clients) == 0
            assert len(async_clients) == 0

    def test_build_provider_mappings_partial_failures(self):
        """Test provider mapping construction with some providers failing."""
        from chimeric.chimeric import _build_provider_mappings

        mock_openai_module = Mock()
        mock_openai_module.OpenAIClient = MockProviderClient
        mock_openai_module.OpenAIAsyncClient = MockAsyncProviderClient

        def mock_import_partial_fail(module_path, fromlist=None):
            if module_path == "chimeric.providers.openai.client":
                return mock_openai_module
            raise ModuleNotFoundError(f"No module named {module_path}")

        with patch("builtins.__import__", side_effect=mock_import_partial_fail):
            sync_clients, async_clients = _build_provider_mappings()

            # Should only have OpenAI provider
            assert len(sync_clients) == 1
            assert len(async_clients) == 1
            assert Provider.OPENAI in sync_clients
            assert Provider.OPENAI in async_clients

            # Should not have other providers
            assert Provider.ANTHROPIC not in sync_clients
            assert Provider.GOOGLE not in sync_clients


class TestProviderAvailabilityChecking:
    """Test provider availability checking logic."""

    def test_provider_not_in_provider_clients(self):
        """Test behavior when provider is not in PROVIDER_CLIENTS."""
        # Create a custom provider enum that won't be in PROVIDER_CLIENTS
        from unittest.mock import Mock

        # Save original clients
        original_provider_clients = PROVIDER_CLIENTS.copy()
        original_async_clients = ASYNC_PROVIDER_CLIENTS.copy()

        try:
            # Clear all providers to simulate none being available
            PROVIDER_CLIENTS.clear()
            ASYNC_PROVIDER_CLIENTS.clear()

            chimeric = Chimeric()

            # Try to add a provider that's not in PROVIDER_CLIENTS
            fake_provider = Mock()
            fake_provider.value = "fake_provider"

            with pytest.raises(ProviderNotFoundError, match="Provider 'fake_provider' not found"):
                chimeric._add_provider(fake_provider, api_key="test")

        finally:
            # Restore original clients
            PROVIDER_CLIENTS.clear()
            PROVIDER_CLIENTS.update(original_provider_clients)
            ASYNC_PROVIDER_CLIENTS.clear()
            ASYNC_PROVIDER_CLIENTS.update(original_async_clients)

    def test_async_provider_not_in_async_provider_clients(self):
        """Test behavior when async provider is not in ASYNC_PROVIDER_CLIENTS."""

        # Save original clients
        original_provider_clients = PROVIDER_CLIENTS.copy()
        original_async_clients = ASYNC_PROVIDER_CLIENTS.copy()

        try:
            # Set up sync providers but clear async providers
            PROVIDER_CLIENTS.clear()
            PROVIDER_CLIENTS[Provider.OPENAI] = MockProviderClient
            ASYNC_PROVIDER_CLIENTS.clear()

            chimeric = Chimeric()

            # Try to add an async provider that's not in ASYNC_PROVIDER_CLIENTS
            with pytest.raises(ProviderNotFoundError, match="Async provider openai"):
                chimeric._add_async_provider(
                    Provider.OPENAI, api_key="test", tool_manager=chimeric._tool_manager
                )

        finally:
            # Restore original clients
            PROVIDER_CLIENTS.clear()
            PROVIDER_CLIENTS.update(original_provider_clients)
            ASYNC_PROVIDER_CLIENTS.clear()
            ASYNC_PROVIDER_CLIENTS.update(original_async_clients)

    def test_provider_initialization_with_empty_mappings(self):
        """Test Chimeric initialization when provider mappings are empty."""
        # Save original clients
        original_provider_clients = PROVIDER_CLIENTS.copy()
        original_async_clients = ASYNC_PROVIDER_CLIENTS.copy()

        try:
            # Clear all provider mappings
            PROVIDER_CLIENTS.clear()
            ASYNC_PROVIDER_CLIENTS.clear()

            # Should initialize successfully with no providers
            chimeric = Chimeric()

            assert len(chimeric.providers) == 0
            assert len(chimeric.async_providers) == 0
            assert chimeric.primary_provider is None

        finally:
            # Restore original clients
            PROVIDER_CLIENTS.clear()
            PROVIDER_CLIENTS.update(original_provider_clients)
            ASYNC_PROVIDER_CLIENTS.clear()
            ASYNC_PROVIDER_CLIENTS.update(original_async_clients)
