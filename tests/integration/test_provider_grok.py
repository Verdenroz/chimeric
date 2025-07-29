import pytest

from chimeric import Chimeric
from chimeric.exceptions import ProviderError

# NOTE: VCR is not used for Grok tests because the xai-sdk uses gRPC instead of HTTP.
# VCR can only record HTTP requests (requests, httpx, aiohttp) but cannot intercept
# gRPC calls. Therefore, Grok tests make real API calls and require valid API keys.


@pytest.mark.grok
def test_grok_model_listing(api_keys):
    """Test Grok model listing functionality.

    Note: This test makes real API calls as VCR cannot record gRPC interactions.
    """
    if "grok_api_key" not in api_keys:
        pytest.skip("Grok API key not found")

    # No VCR - xai-sdk uses gRPC which VCR cannot record
    chimeric = Chimeric(grok_api_key=api_keys["grok_api_key"])

    models = chimeric.list_models()
    assert len(models) > 0

    # Should have Grok models
    model_ids = [model.id.lower() for model in models]
    assert any("grok" in model_id for model_id in model_ids)

    print(f"Found {len(models)} Grok models")


@pytest.mark.grok
def test_grok_sync_generation(api_keys):
    """Test Grok synchronous generation functionality.

    Note: This test makes real API calls as VCR cannot record gRPC interactions.
    """
    if "grok_api_key" not in api_keys:
        pytest.skip("Grok API key not found")

    # No VCR - xai-sdk uses gRPC which VCR cannot record
    chimeric = Chimeric(grok_api_key=api_keys["grok_api_key"])

    response = chimeric.generate(
        model="grok-3-mini",
        messages=[{"role": "user", "content": "Hello, respond briefly."}],
        stream=False,
    )

    assert response is not None
    assert response.content


@pytest.mark.grok
@pytest.mark.asyncio
async def test_grok_async_generation(api_keys):
    """Test Grok asynchronous generation functionality.

    Note: This test makes real API calls as VCR cannot record gRPC interactions.
    """
    if "grok_api_key" not in api_keys:
        pytest.skip("Grok API key not found")

    # No VCR - xai-sdk uses gRPC which VCR cannot record
    chimeric = Chimeric(grok_api_key=api_keys["grok_api_key"])

    response = await chimeric.agenerate(
        model="grok-3-mini",
        messages=[{"role": "user", "content": "Hello, respond briefly."}],
        stream=False,
    )

    assert response is not None
    assert response.content


@pytest.mark.grok
def test_grok_sync_tools_streaming(api_keys):
    """Test Grok sync generation with tools (streaming).

    Note: This test makes real API calls as VCR cannot record gRPC interactions.
    """
    if "grok_api_key" not in api_keys:
        pytest.skip("Grok API key not found")

    # No VCR - xai-sdk uses gRPC which VCR cannot record
    chimeric = Chimeric(grok_api_key=api_keys["grok_api_key"])

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
        model="grok-3-latest",
        messages=[{"role": "user", "content": "What is 2+2-4? Tell me a joke."}],
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


@pytest.mark.grok
@pytest.mark.asyncio
async def test_grok_async_tools_non_streaming(api_keys):
    """Test Grok async generation with tools (non-streaming).

    Note: This test makes real API calls as VCR cannot record gRPC interactions.
    """
    if "grok_api_key" not in api_keys:
        pytest.skip("Grok API key not found")

    # No VCR - xai-sdk uses gRPC which VCR cannot record
    chimeric = Chimeric(grok_api_key=api_keys["grok_api_key"])

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
        model="grok-3-latest",
        messages=[{"role": "user", "content": "What is 2+2-4? Tell me a joke."}],
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


@pytest.mark.grok
def test_grok_init_kwargs_propagation(api_keys):
    """Test Grok kwargs propagation through the stack with fake cross-provider params.

    Note: This test makes real API calls as VCR cannot record gRPC interactions.
    """
    if "grok_api_key" not in api_keys:
        pytest.skip("Grok API key not found")

    # No VCR - xai-sdk uses gRPC which VCR cannot record
    # Test with custom initialization kwargs including fake params from other providers
    chimeric = Chimeric(
        grok_api_key=api_keys["grok_api_key"],
        timeout=60,
        max_retries=3,
        # Fake params that other providers might use
        openai_fake_param="should_be_ignored",
        anthropic_fake_param="should_be_ignored",
        google_vertex_project="fake_project",
        cohere_fake_setting=True,
    )

    # Test with generation kwargs
    response = chimeric.generate(
        model="grok-3-mini",
        messages=[{"role": "user", "content": "Hello, respond briefly."}],
        stream=False,
    )

    assert response is not None
    assert response.content


@pytest.mark.grok
def test_grok_invalid_generate_kwargs_raises_provider_error(api_keys):
    """Test that invalid kwargs in generate raise ProviderError.

    Note: This test makes real API calls as VCR cannot record gRPC interactions.
    """
    if "grok_api_key" not in api_keys:
        pytest.skip("Grok API key not found")

    # No VCR - xai-sdk uses gRPC which VCR cannot record
    chimeric = Chimeric(grok_api_key=api_keys["grok_api_key"])

    # Test with an invalid parameter that doesn't exist in Grok API
    with pytest.raises(ProviderError) as exc_info:
        chimeric.generate(
            model="grok-3-mini",
            messages=[{"role": "user", "content": "Hello"}],
            invalid_grok_parameter="this_should_fail",
            stream=False,
        )

    # Verify the error contains provider information
    assert "Grok" in str(exc_info.value) or "grok" in str(exc_info.value).lower()
    print(f"ProviderError raised as expected: {exc_info.value}")
