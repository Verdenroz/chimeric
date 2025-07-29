from collections.abc import AsyncGenerator

import pytest

from chimeric import Chimeric
from chimeric.exceptions import ProviderError

from .vcr_config import get_cassette_path, get_vcr


@pytest.mark.anthropic
def test_anthropic_model_listing(api_keys):
    """Test Anthropic model listing functionality."""
    if "anthropic_api_key" not in api_keys:
        pytest.skip("Anthropic API key not found")

    cassette_path = get_cassette_path("anthropic", "test_model_listing")

    with get_vcr().use_cassette(cassette_path):
        chimeric = Chimeric(anthropic_api_key=api_keys["anthropic_api_key"])

        models = chimeric.list_models()
        assert len(models) > 0

        # Should have Claude models
        model_ids = [model.id.lower() for model in models]
        assert any("claude" in model_id for model_id in model_ids)

        print(f"Found {len(models)} Anthropic models")


@pytest.mark.anthropic
def test_anthropic_sync_generation(api_keys):
    """Test Anthropic synchronous generation functionality."""
    if "anthropic_api_key" not in api_keys:
        pytest.skip("Anthropic API key not found")

    cassette_path = get_cassette_path("anthropic", "test_sync_generation")

    with get_vcr().use_cassette(cassette_path):
        chimeric = Chimeric(anthropic_api_key=api_keys["anthropic_api_key"])

        response = chimeric.generate(
            model="claude-3-5-haiku-latest",
            messages=[{"role": "user", "content": "Hello, respond briefly."}],
            stream=False,
            max_tokens=20,
        )

        assert response is not None
        assert response.content


@pytest.mark.anthropic
@pytest.mark.asyncio
async def test_anthropic_async_generation(api_keys):
    """Test Anthropic asynchronous generation functionality."""
    if "anthropic_api_key" not in api_keys:
        pytest.skip("Anthropic API key not found")

    cassette_path = get_cassette_path("anthropic", "test_async_generation")

    with get_vcr().use_cassette(cassette_path):
        chimeric = Chimeric(anthropic_api_key=api_keys["anthropic_api_key"])

        response = await chimeric.agenerate(
            model="claude-3-5-haiku-latest",
            messages=[{"role": "user", "content": "Hello, respond briefly."}],
            stream=False,
            max_tokens=20,
        )

        assert response is not None
        assert response.content


@pytest.mark.anthropic
def test_anthropic_sync_tools_non_streaming(api_keys):
    """Test Anthropic sync generation with tools (non-streaming)."""
    if "anthropic_api_key" not in api_keys:
        pytest.skip("Anthropic API key not found")

    cassette_path = get_cassette_path("anthropic", "test_sync_tools_non_streaming")

    with get_vcr().use_cassette(cassette_path):
        chimeric = Chimeric(anthropic_api_key=api_keys["anthropic_api_key"])

        # Track tool calls
        tool_calls = {"add": 0, "subtract": 0, "joke": 0}

        @chimeric.tool()
        def add(x: int, y: int) -> int:  # type: ignore[reportUnusedFunction]
            """
            Adds two numbers together.
            Args:
                x: the first number
                y: the second number

            Returns:
                The sum of x and y.
            """
            print("Adding numbers:", x, y)
            tool_calls["add"] += 1
            return x + y

        @chimeric.tool()
        def subtract(x: int, y: int) -> int:  # type: ignore[reportUnusedFunction]
            """
            Subtracts the second number from the first.
            Args:
                x: the first number
                y: the second number

            Returns:
                The result of x - y.
            """
            print("Subtracting numbers:", x, y)
            tool_calls["subtract"] += 1
            return x - y

        @chimeric.tool()
        def joke() -> str:  # type: ignore[reportUnusedFunction]
            """
            Returns a joke.
            """
            print("Telling a joke...")
            tool_calls["joke"] += 1
            return "Why did the chicken cross the road? To get to the other side!"

        response = chimeric.generate(
            model="claude-3-5-haiku-latest",
            messages=[{"role": "user", "content": "What is 2+2-4-10+50? Tell me a joke."}],
            stream=False,
        )

        assert response is not None
        assert response.content

        # Verify tools were called
        assert tool_calls["add"] > 0, "Add function should have been called"
        assert tool_calls["subtract"] > 0, "Subtract function should have been called"
        assert tool_calls["joke"] > 0, "Joke function should have been called"

        # Print summary for debugging
        print(f"Tool call counts: {tool_calls}")


@pytest.mark.anthropic
def test_anthropic_sync_tools_streaming(api_keys):
    """Test Anthropic sync generation with tools (streaming)."""
    if "anthropic_api_key" not in api_keys:
        pytest.skip("Anthropic API key not found")

    cassette_path = get_cassette_path("anthropic", "test_sync_tools_streaming")

    with get_vcr().use_cassette(cassette_path):
        chimeric = Chimeric(anthropic_api_key=api_keys["anthropic_api_key"])

        # Track tool calls
        tool_calls = {"add": 0, "subtract": 0, "joke": 0}

        @chimeric.tool()
        def add(x: int, y: int) -> int:  # type: ignore[reportUnusedFunction]
            """
            Adds two numbers together.
            Args:
                x: the first number
                y: the second number

            Returns:
                The sum of x and y.
            """
            print("Adding numbers:", x, y)
            tool_calls["add"] += 1
            return x + y

        @chimeric.tool()
        def subtract(x: int, y: int) -> int:  # type: ignore[reportUnusedFunction]
            """
            Subtracts the second number from the first.
            Args:
                x: the first number
                y: the second number

            Returns:
                The result of x - y.
            """
            print("Subtracting numbers:", x, y)
            tool_calls["subtract"] += 1
            return x - y

        @chimeric.tool()
        def joke() -> str:  # type: ignore[reportUnusedFunction]
            """
            Returns a joke.
            """
            print("Telling a joke...")
            tool_calls["joke"] += 1
            return "Why did the chicken cross the road? To get to the other side!"

        response = chimeric.generate(
            model="claude-3-5-haiku-latest",
            messages=[{"role": "user", "content": "What is 2+2-4-10+50? Tell me a joke."}],
            stream=True,
        )

        # Collect all chunks and verify at least some have content
        assert response is not None
        chunks = list(response)
        assert len(chunks) > 0
        content_chunks = [chunk for chunk in chunks if hasattr(chunk, "content") and chunk.content]
        assert len(content_chunks) > 0, "At least some chunks should have content"

        # Verify tools were actually called
        assert tool_calls["add"] > 0, "Add function should have been called"
        assert tool_calls["subtract"] > 0, "Subtract function should have been called"
        assert tool_calls["joke"] > 0, "Joke function should have been called"

        # Print summary for debugging
        print(f"Tool call counts: {tool_calls}")


@pytest.mark.anthropic
@pytest.mark.asyncio
async def test_anthropic_async_tools_streaming(api_keys):
    """Test Anthropic async generation with tools (streaming)."""
    if "anthropic_api_key" not in api_keys:
        pytest.skip("Anthropic API key not found")

    cassette_path = get_cassette_path("anthropic", "test_async_tools_streaming")

    with get_vcr().use_cassette(cassette_path):
        chimeric = Chimeric(anthropic_api_key=api_keys["anthropic_api_key"])

        # Track tool calls
        tool_calls = {"add": 0, "subtract": 0, "joke": 0}

        @chimeric.tool()
        def add(x: int, y: int) -> int:  # type: ignore[reportUnusedFunction]
            """
            Adds two numbers together.
            Args:
                x: the first number
                y: the second number

            Returns:
                The sum of x and y.
            """
            print("Adding numbers:", x, y)
            tool_calls["add"] += 1
            return x + y

        @chimeric.tool()
        def subtract(x: int, y: int) -> int:  # type: ignore[reportUnusedFunction]
            """
            Subtracts the second number from the first.
            Args:
                x: the first number
                y: the second number

            Returns:
                The result of x - y.
            """
            print("Subtracting numbers:", x, y)
            tool_calls["subtract"] += 1
            return x - y

        @chimeric.tool()
        def joke() -> str:  # type: ignore[reportUnusedFunction]
            """
            Returns a joke.
            """
            print("Telling a joke...")
            tool_calls["joke"] += 1
            return "Why did the chicken cross the road? To get to the other side!"

        response = await chimeric.agenerate(
            model="claude-3-5-haiku-latest",
            messages=[{"role": "user", "content": "What is 2+2-4-10+50? Tell me a joke."}],
            stream=True,
        )

        # Collect all chunks and verify at least some have content
        assert response is not None
        assert isinstance(response, AsyncGenerator)
        chunks = [chunk async for chunk in response]
        assert len(chunks) > 0
        content_chunks = [chunk for chunk in chunks if chunk.content]
        assert len(content_chunks) > 0, "At least some chunks should have content"

        # Verify tools were actually called
        assert tool_calls["add"] > 0, "Add function should have been called"
        assert tool_calls["subtract"] > 0, "Subtract function should have been called"
        assert tool_calls["joke"] > 0, "Joke function should have been called"

        # Print summary for debugging
        print(f"Tool call counts: {tool_calls}")


@pytest.mark.anthropic
@pytest.mark.asyncio
async def test_anthropic_async_tools_non_streaming(api_keys):
    """Test Anthropic async generation with tools (non-streaming)."""
    if "anthropic_api_key" not in api_keys:
        pytest.skip("Anthropic API key not found")

    cassette_path = get_cassette_path("anthropic", "test_async_tools_non_streaming")

    with get_vcr().use_cassette(cassette_path):
        chimeric = Chimeric(anthropic_api_key=api_keys["anthropic_api_key"])

        # Track tool calls
        tool_calls = {"add": 0, "subtract": 0, "joke": 0}

        @chimeric.tool()
        def add(x: int, y: int) -> int:  # type: ignore[reportUnusedFunction]
            """
            Adds two numbers together.
            Args:
                x: the first number
                y: the second number

            Returns:
                The sum of x and y.
            """
            print("Adding numbers:", x, y)
            tool_calls["add"] += 1
            return x + y

        @chimeric.tool()
        def subtract(x: int, y: int) -> int:  # type: ignore[reportUnusedFunction]
            """
            Subtracts the second number from the first.
            Args:
                x: the first number
                y: the second number

            Returns:
                The result of x - y.
            """
            print("Subtracting numbers:", x, y)
            tool_calls["subtract"] += 1
            return x - y

        @chimeric.tool()
        def joke() -> str:  # type: ignore[reportUnusedFunction]
            """
            Returns a joke.
            """
            print("Telling a joke...")
            tool_calls["joke"] += 1
            return "Why did the chicken cross the road? To get to the other side!"

        response = await chimeric.agenerate(
            model="claude-3-5-haiku-latest",
            messages=[{"role": "user", "content": "What is 2+2-4-10+50? Tell me a joke."}],
            stream=False,
        )

        assert response is not None
        assert response.content

        # Verify tools were actually called
        assert tool_calls["add"] > 0, "Add function should have been called"
        assert tool_calls["subtract"] > 0, "Subtract function should have been called"
        assert tool_calls["joke"] > 0, "Joke function should have been called"

        # Print summary for debugging
        print(f"Tool call counts: {tool_calls}")


@pytest.mark.anthropic
def test_anthropic_init_kwargs_propagation(api_keys):
    """Test Anthropic kwargs propagation through the stack with fake cross-provider params."""
    if "anthropic_api_key" not in api_keys:
        pytest.skip("Anthropic API key not found")

    cassette_path = get_cassette_path("anthropic", "test_kwargs_propagation")

    with get_vcr().use_cassette(cassette_path):
        # Test with custom initialization kwargs including fake params from other providers
        chimeric = Chimeric(
            anthropic_api_key=api_keys["anthropic_api_key"],
            timeout=60,
            max_retries=3,
            # Fake params that other providers might use
            openai_fake_param="should_be_ignored",
            google_vertex_project="fake_project",
            cohere_fake_setting=True,
        )

        # Test with generation kwargs
        response = chimeric.generate(
            model="claude-3-5-haiku-latest",
            messages=[{"role": "user", "content": "Hello, respond briefly."}],
            temperature=0.1,
            max_tokens=20,
            stream=False,
        )

        assert response is not None
        assert response.content


@pytest.mark.anthropic
def test_anthropic_invalid_generate_kwargs_raises_provider_error(api_keys):
    """Test that invalid kwargs in generate raise ProviderError."""
    if "anthropic_api_key" not in api_keys:
        pytest.skip("Anthropic API key not found")

    cassette_path = get_cassette_path("anthropic", "test_invalid_kwargs_raises_provider_error")

    with get_vcr().use_cassette(cassette_path):
        chimeric = Chimeric(anthropic_api_key=api_keys["anthropic_api_key"])

        # Test with an invalid parameter that doesn't exist in Anthropic API
        with pytest.raises(ProviderError) as exc_info:
            chimeric.generate(
                model="claude-3-5-haiku-latest",
                messages=[{"role": "user", "content": "Hello"}],
                invalid_anthropic_parameter="this_should_fail",
                stream=False,
            )

        # Verify the error contains provider information
        assert "Anthropic" in str(exc_info.value) or "anthropic" in str(exc_info.value).lower()
        print(f"ProviderError raised as expected: {exc_info.value}")
