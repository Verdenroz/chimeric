"""Integration tests for structured output (response_model) across all providers."""

from pydantic import BaseModel
import pytest

from chimeric import Chimeric

from .vcr_config import get_cassette_path, get_vcr

# ---------------------------------------------------------------------------
# Shared response models
# ---------------------------------------------------------------------------


class MathAnswer(BaseModel):
    """Simple arithmetic answer used across all provider tests."""

    result: int
    explanation: str


class PersonInfo(BaseModel):
    """Extraction target for data-extraction style tests."""

    name: str
    age: int
    city: str


_MATH_PROMPT = (
    "What is 6 multiplied by 7? Return the numeric result and a one-sentence explanation."
)
_EXTRACT_PROMPT = (
    "Extract the following information: Alice is 30 years old and lives in Paris. "
    "Return her name, age, and city."
)

# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


@pytest.mark.openai
def test_openai_structured_output_sync(api_keys):
    """OpenAI: response_model populates parsed on a non-streaming response."""
    if "openai_api_key" not in api_keys:
        pytest.skip("OpenAI API key not found")

    chimeric = Chimeric(openai_api_key=api_keys["openai_api_key"])
    cassette_path = get_cassette_path("openai", "test_structured_output_sync")

    with get_vcr().use_cassette(cassette_path):
        response = chimeric.generate(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": _MATH_PROMPT}],
            response_model=MathAnswer,
            max_tokens=150,
        )

    assert isinstance(response.parsed, MathAnswer)
    assert response.parsed.result == 42
    assert response.parsed.explanation


@pytest.mark.openai
def test_openai_structured_output_streaming(api_keys):
    """OpenAI: response_model sets parsed on the final streaming chunk."""
    if "openai_api_key" not in api_keys:
        pytest.skip("OpenAI API key not found")

    chimeric = Chimeric(openai_api_key=api_keys["openai_api_key"])
    cassette_path = get_cassette_path("openai", "test_structured_output_streaming")

    with get_vcr().use_cassette(cassette_path):
        chunks = list(
            chimeric.generate(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": _MATH_PROMPT}],
                response_model=MathAnswer,
                stream=True,
                max_tokens=150,
            )
        )

    final = next(c for c in reversed(chunks) if c.finish_reason is not None)
    assert isinstance(final.parsed, MathAnswer)
    assert final.parsed.result == 42
    intermediate = [c for c in chunks if c.finish_reason is None]
    assert all(c.parsed is None for c in intermediate)


@pytest.mark.openai
@pytest.mark.asyncio
async def test_openai_structured_output_async(api_keys):
    """OpenAI: agenerate with response_model populates parsed."""
    if "openai_api_key" not in api_keys:
        pytest.skip("OpenAI API key not found")

    chimeric = Chimeric(openai_api_key=api_keys["openai_api_key"])
    cassette_path = get_cassette_path("openai", "test_structured_output_async")

    with get_vcr().use_cassette(cassette_path):
        response = await chimeric.agenerate(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": _MATH_PROMPT}],
            response_model=MathAnswer,
            max_tokens=150,
        )

    assert isinstance(response.parsed, MathAnswer)
    assert response.parsed.result == 42


@pytest.mark.openai
def test_openai_structured_output_extraction(api_keys):
    """OpenAI: extraction of nested fields into a Pydantic model."""
    if "openai_api_key" not in api_keys:
        pytest.skip("OpenAI API key not found")

    chimeric = Chimeric(openai_api_key=api_keys["openai_api_key"])
    cassette_path = get_cassette_path("openai", "test_structured_output_extraction")

    with get_vcr().use_cassette(cassette_path):
        response = chimeric.generate(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": _EXTRACT_PROMPT}],
            response_model=PersonInfo,
            max_tokens=150,
        )

    assert isinstance(response.parsed, PersonInfo)
    assert response.parsed.name == "Alice"
    assert response.parsed.age == 30
    assert response.parsed.city == "Paris"


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


@pytest.mark.anthropic
def test_anthropic_structured_output_sync(api_keys):
    """Anthropic: response_model populates parsed on a non-streaming response."""
    if "anthropic_api_key" not in api_keys:
        pytest.skip("Anthropic API key not found")

    chimeric = Chimeric(anthropic_api_key=api_keys["anthropic_api_key"])
    cassette_path = get_cassette_path("anthropic", "test_structured_output_sync")

    with get_vcr().use_cassette(cassette_path):
        response = chimeric.generate(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": _MATH_PROMPT}],
            response_model=MathAnswer,
            max_tokens=150,
        )

    assert isinstance(response.parsed, MathAnswer)
    assert response.parsed.result == 42
    assert response.parsed.explanation


@pytest.mark.anthropic
def test_anthropic_structured_output_streaming(api_keys):
    """Anthropic: response_model sets parsed on the final streaming chunk."""
    if "anthropic_api_key" not in api_keys:
        pytest.skip("Anthropic API key not found")

    chimeric = Chimeric(anthropic_api_key=api_keys["anthropic_api_key"])
    cassette_path = get_cassette_path("anthropic", "test_structured_output_streaming")

    with get_vcr().use_cassette(cassette_path):
        chunks = list(
            chimeric.generate(
                model="claude-haiku-4-5-20251001",
                messages=[{"role": "user", "content": _MATH_PROMPT}],
                response_model=MathAnswer,
                stream=True,
                max_tokens=150,
            )
        )

    final = next(c for c in reversed(chunks) if c.finish_reason is not None)
    assert isinstance(final.parsed, MathAnswer)
    assert final.parsed.result == 42


@pytest.mark.anthropic
@pytest.mark.asyncio
async def test_anthropic_structured_output_async(api_keys):
    """Anthropic: agenerate with response_model populates parsed."""
    if "anthropic_api_key" not in api_keys:
        pytest.skip("Anthropic API key not found")

    chimeric = Chimeric(anthropic_api_key=api_keys["anthropic_api_key"])
    cassette_path = get_cassette_path("anthropic", "test_structured_output_async")

    with get_vcr().use_cassette(cassette_path):
        response = await chimeric.agenerate(
            model="claude-haiku-4-5-20251001",
            messages=[{"role": "user", "content": _MATH_PROMPT}],
            response_model=MathAnswer,
            max_tokens=150,
        )

    assert isinstance(response.parsed, MathAnswer)
    assert response.parsed.result == 42


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------


@pytest.mark.google
def test_google_structured_output_sync(api_keys):
    """Google Gemini: response_model populates parsed on a non-streaming response."""
    if "google_api_key" not in api_keys:
        pytest.skip("Google API key not found")

    chimeric = Chimeric(google_api_key=api_keys["google_api_key"])
    cassette_path = get_cassette_path("google", "test_structured_output_sync")

    with get_vcr().use_cassette(cassette_path):
        response = chimeric.generate(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": _MATH_PROMPT}],
            response_model=MathAnswer,
            max_output_tokens=150,
        )

    assert isinstance(response.parsed, MathAnswer)
    assert response.parsed.result == 42
    assert response.parsed.explanation


@pytest.mark.google
def test_google_structured_output_streaming(api_keys):
    """Google Gemini: response_model sets parsed on the final streaming chunk."""
    if "google_api_key" not in api_keys:
        pytest.skip("Google API key not found")

    chimeric = Chimeric(google_api_key=api_keys["google_api_key"])
    cassette_path = get_cassette_path("google", "test_structured_output_streaming")

    with get_vcr().use_cassette(cassette_path):
        chunks = list(
            chimeric.generate(
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": _MATH_PROMPT}],
                response_model=MathAnswer,
                stream=True,
                max_output_tokens=150,
            )
        )

    final = next(c for c in reversed(chunks) if c.finish_reason is not None)
    assert isinstance(final.parsed, MathAnswer)
    assert final.parsed.result == 42


# ---------------------------------------------------------------------------
# Cohere
# ---------------------------------------------------------------------------


@pytest.mark.cohere
def test_cohere_structured_output_sync(api_keys):
    """Cohere: response_model populates parsed on a non-streaming response."""
    if "cohere_api_key" not in api_keys:
        pytest.skip("Cohere API key not found")

    chimeric = Chimeric(cohere_api_key=api_keys["cohere_api_key"])
    cassette_path = get_cassette_path("cohere", "test_structured_output_sync")

    with get_vcr().use_cassette(cassette_path):
        response = chimeric.generate(
            model="command-r-08-2024",
            messages=[{"role": "user", "content": _MATH_PROMPT}],
            response_model=MathAnswer,
            max_tokens=150,
        )

    assert isinstance(response.parsed, MathAnswer)
    assert response.parsed.result == 42
    assert response.parsed.explanation


@pytest.mark.cohere
def test_cohere_structured_output_streaming(api_keys):
    """Cohere: response_model sets parsed on the final streaming chunk."""
    if "cohere_api_key" not in api_keys:
        pytest.skip("Cohere API key not found")

    chimeric = Chimeric(cohere_api_key=api_keys["cohere_api_key"])
    cassette_path = get_cassette_path("cohere", "test_structured_output_streaming")

    with get_vcr().use_cassette(cassette_path):
        chunks = list(
            chimeric.generate(
                model="command-r-08-2024",
                messages=[{"role": "user", "content": _MATH_PROMPT}],
                response_model=MathAnswer,
                stream=True,
                max_tokens=150,
            )
        )

    final = next(c for c in reversed(chunks) if c.finish_reason is not None)
    assert isinstance(final.parsed, MathAnswer)
    assert final.parsed.result == 42


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------


@pytest.mark.groq
def test_groq_structured_output_sync(api_keys):
    """Groq: response_model populates parsed on a non-streaming response."""
    if "groq_api_key" not in api_keys:
        pytest.skip("Groq API key not found")

    chimeric = Chimeric(groq_api_key=api_keys["groq_api_key"])
    cassette_path = get_cassette_path("groq", "test_structured_output_sync")

    with get_vcr().use_cassette(cassette_path):
        response = chimeric.generate(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": _MATH_PROMPT}],
            response_model=MathAnswer,
            max_tokens=150,
        )

    assert isinstance(response.parsed, MathAnswer)
    assert response.parsed.result == 42
    assert response.parsed.explanation


@pytest.mark.groq
def test_groq_structured_output_streaming(api_keys):
    """Groq: streaming with response_model is not supported.

    Groq does not support streaming with Structured Outputs (json_schema mode).
    """
    pytest.skip("Groq does not support streaming with Structured Outputs (json_schema)")


# ---------------------------------------------------------------------------
# Cerebras
# ---------------------------------------------------------------------------


@pytest.mark.cerebras
def test_cerebras_structured_output_sync(api_keys):
    """Cerebras: response_model populates parsed on a non-streaming response."""
    if "cerebras_api_key" not in api_keys:
        pytest.skip("Cerebras API key not found")

    chimeric = Chimeric(cerebras_api_key=api_keys["cerebras_api_key"])
    cassette_path = get_cassette_path("cerebras", "test_structured_output_sync")

    with get_vcr().use_cassette(cassette_path):
        response = chimeric.generate(
            model="llama3.1-8b",
            messages=[{"role": "user", "content": _MATH_PROMPT}],
            response_model=MathAnswer,
            max_tokens=150,
        )

    assert isinstance(response.parsed, MathAnswer)
    assert response.parsed.result == 42
    assert response.parsed.explanation


@pytest.mark.cerebras
def test_cerebras_structured_output_streaming(api_keys):
    """Cerebras: response_model sets parsed on the final streaming chunk."""
    if "cerebras_api_key" not in api_keys:
        pytest.skip("Cerebras API key not found")

    chimeric = Chimeric(cerebras_api_key=api_keys["cerebras_api_key"])
    cassette_path = get_cassette_path("cerebras", "test_structured_output_streaming")

    with get_vcr().use_cassette(cassette_path):
        chunks = list(
            chimeric.generate(
                model="llama3.1-8b",
                messages=[{"role": "user", "content": _MATH_PROMPT}],
                response_model=MathAnswer,
                stream=True,
                max_tokens=150,
            )
        )

    final = next(c for c in reversed(chunks) if c.finish_reason is not None)
    assert isinstance(final.parsed, MathAnswer)
    assert final.parsed.result == 42


# ---------------------------------------------------------------------------
# Grok  (no VCR)
# ---------------------------------------------------------------------------


@pytest.mark.grok
def test_grok_structured_output_sync(api_keys):
    """Grok: response_model populates parsed on a non-streaming response."""
    if "grok_api_key" not in api_keys:
        pytest.skip("Grok API key not found")

    chimeric = Chimeric(grok_api_key=api_keys["grok_api_key"])

    response = chimeric.generate(
        model="grok-3-mini",
        messages=[{"role": "user", "content": _MATH_PROMPT}],
        response_model=MathAnswer,
        max_tokens=150,
    )

    assert isinstance(response.parsed, MathAnswer)
    assert response.parsed.result == 42
    assert response.parsed.explanation


@pytest.mark.grok
def test_grok_structured_output_streaming(api_keys):
    """Grok: response_model sets parsed on the final streaming chunk."""
    if "grok_api_key" not in api_keys:
        pytest.skip("Grok API key not found")

    chimeric = Chimeric(grok_api_key=api_keys["grok_api_key"])

    chunks = list(
        chimeric.generate(
            model="grok-3-mini",
            messages=[{"role": "user", "content": _MATH_PROMPT}],
            response_model=MathAnswer,
            stream=True,
            max_tokens=150,
        )
    )

    final = next(c for c in reversed(chunks) if c.finish_reason is not None)
    assert isinstance(final.parsed, MathAnswer)
    assert final.parsed.result == 42
