"""Integration tests for the embedding API."""

import pytest

from chimeric import Chimeric
from chimeric.types import EmbeddingResponse

from .vcr_config import get_cassette_path, get_vcr

# ---------------------------------------------------------------------------
# OpenAI embeddings
# ---------------------------------------------------------------------------


@pytest.mark.openai
def test_openai_embed_single(api_keys):
    """Test single-text embedding via OpenAI."""
    if "openai_api_key" not in api_keys:
        pytest.skip("OpenAI API key not found")

    chimeric = Chimeric(openai_api_key=api_keys["openai_api_key"])
    cassette_path = get_cassette_path("openai", "test_embed_single")

    with get_vcr().use_cassette(cassette_path):
        result = chimeric.embed(
            model="text-embedding-3-small",
            input="Python developer with 5 years experience",
        )

        assert isinstance(result, EmbeddingResponse)
        assert result.embedding is not None
        assert len(result.embedding) > 0
        assert all(isinstance(v, float) for v in result.embedding)
        assert result.model is not None
        assert result.usage is not None
        assert result.usage.prompt_tokens > 0
        assert result.usage.total_tokens > 0


@pytest.mark.openai
def test_openai_embed_batch(api_keys):
    """Test batch embedding via OpenAI."""
    if "openai_api_key" not in api_keys:
        pytest.skip("OpenAI API key not found")

    chimeric = Chimeric(openai_api_key=api_keys["openai_api_key"])
    cassette_path = get_cassette_path("openai", "test_embed_batch")

    with get_vcr().use_cassette(cassette_path):
        texts = [
            "Python developer with 5 years experience",
            "Looking for a backend engineer with Go expertise",
            "Frontend React developer needed",
        ]
        result = chimeric.embed(
            model="text-embedding-3-small",
            input=texts,
        )

        assert isinstance(result, EmbeddingResponse)
        assert result.embedding is None
        assert len(result.embeddings) == 3
        for emb in result.embeddings:
            assert len(emb) > 0
            assert all(isinstance(v, float) for v in emb)
        assert result.usage is not None
        assert result.usage.prompt_tokens > 0


@pytest.mark.openai
def test_openai_embed_with_dimensions(api_keys):
    """Test embedding with custom dimensions (text-embedding-3 feature)."""
    if "openai_api_key" not in api_keys:
        pytest.skip("OpenAI API key not found")

    chimeric = Chimeric(openai_api_key=api_keys["openai_api_key"])
    cassette_path = get_cassette_path("openai", "test_embed_with_dimensions")

    with get_vcr().use_cassette(cassette_path):
        result = chimeric.embed(
            model="text-embedding-3-small",
            input="Hello world",
            dimensions=256,
        )

        assert isinstance(result, EmbeddingResponse)
        assert result.embedding is not None
        assert len(result.embedding) == 256


@pytest.mark.openai
@pytest.mark.asyncio
async def test_openai_aembed_single(api_keys):
    """Test async single-text embedding via OpenAI."""
    if "openai_api_key" not in api_keys:
        pytest.skip("OpenAI API key not found")

    chimeric = Chimeric(openai_api_key=api_keys["openai_api_key"])
    cassette_path = get_cassette_path("openai", "test_aembed_single")

    with get_vcr().use_cassette(cassette_path):
        result = await chimeric.aembed(
            model="text-embedding-3-small",
            input="Python developer with 5 years experience",
        )

        assert isinstance(result, EmbeddingResponse)
        assert result.embedding is not None
        assert len(result.embedding) > 0
        assert result.usage is not None


@pytest.mark.openai
@pytest.mark.asyncio
async def test_openai_aembed_batch(api_keys):
    """Test async batch embedding via OpenAI."""
    if "openai_api_key" not in api_keys:
        pytest.skip("OpenAI API key not found")

    chimeric = Chimeric(openai_api_key=api_keys["openai_api_key"])
    cassette_path = get_cassette_path("openai", "test_aembed_batch")

    with get_vcr().use_cassette(cassette_path):
        result = await chimeric.aembed(
            model="text-embedding-3-small",
            input=["text one", "text two"],
        )

        assert isinstance(result, EmbeddingResponse)
        assert result.embedding is None
        assert len(result.embeddings) == 2


# ---------------------------------------------------------------------------
# Cosine similarity smoke test (validates vectors are usable)
# ---------------------------------------------------------------------------


@pytest.mark.openai
def test_openai_embed_cosine_similarity(api_keys):
    """Verify that similar texts produce closer embeddings than dissimilar ones."""
    if "openai_api_key" not in api_keys:
        pytest.skip("OpenAI API key not found")

    chimeric = Chimeric(openai_api_key=api_keys["openai_api_key"])
    cassette_path = get_cassette_path("openai", "test_embed_cosine_similarity")

    with get_vcr().use_cassette(cassette_path):
        result = chimeric.embed(
            model="text-embedding-3-small",
            input=[
                "Python backend developer",
                "Python backend engineer",
                "Chocolate cake recipe",
            ],
        )

        a, b, c = result.embeddings
        sim_ab = _cosine_similarity(a, b)
        sim_ac = _cosine_similarity(a, c)

        # "Python backend developer" should be closer to "Python backend engineer"
        # than to "Chocolate cake recipe"
        assert sim_ab > sim_ac, (
            f"Expected similar texts to have higher similarity: {sim_ab:.4f} vs {sim_ac:.4f}"
        )


# ---------------------------------------------------------------------------
# Google embeddings
# ---------------------------------------------------------------------------


@pytest.mark.google
def test_google_embed_single(api_keys):
    """Test single-text embedding via Google Gemini."""
    if "google_api_key" not in api_keys:
        pytest.skip("Google API key not found")

    chimeric = Chimeric(google_api_key=api_keys["google_api_key"])
    cassette_path = get_cassette_path("google", "test_embed_single")

    with get_vcr().use_cassette(cassette_path):
        result = chimeric.embed(
            model="gemini-embedding-001",
            input="Python developer with 5 years experience",
        )

        assert isinstance(result, EmbeddingResponse)
        assert result.embedding is not None
        assert len(result.embedding) > 0
        assert all(isinstance(v, float) for v in result.embedding)
        assert result.model is not None


@pytest.mark.google
def test_google_embed_batch(api_keys):
    """Test batch embedding via Google Gemini."""
    if "google_api_key" not in api_keys:
        pytest.skip("Google API key not found")

    chimeric = Chimeric(google_api_key=api_keys["google_api_key"])
    cassette_path = get_cassette_path("google", "test_embed_batch")

    with get_vcr().use_cassette(cassette_path):
        texts = [
            "Python developer with 5 years experience",
            "Looking for a backend engineer with Go expertise",
            "Frontend React developer needed",
        ]
        result = chimeric.embed(
            model="gemini-embedding-001",
            input=texts,
        )

        assert isinstance(result, EmbeddingResponse)
        assert result.embedding is None
        assert len(result.embeddings) == 3
        for emb in result.embeddings:
            assert len(emb) > 0
            assert all(isinstance(v, float) for v in emb)


@pytest.mark.google
@pytest.mark.asyncio
async def test_google_aembed_single(api_keys):
    """Test async single-text embedding via Google Gemini."""
    if "google_api_key" not in api_keys:
        pytest.skip("Google API key not found")

    chimeric = Chimeric(google_api_key=api_keys["google_api_key"])
    cassette_path = get_cassette_path("google", "test_aembed_single")

    with get_vcr().use_cassette(cassette_path):
        result = await chimeric.aembed(
            model="gemini-embedding-001",
            input="Python developer with 5 years experience",
        )

        assert isinstance(result, EmbeddingResponse)
        assert result.embedding is not None
        assert len(result.embedding) > 0


@pytest.mark.google
@pytest.mark.asyncio
async def test_google_aembed_batch(api_keys):
    """Test async batch embedding via Google Gemini."""
    if "google_api_key" not in api_keys:
        pytest.skip("Google API key not found")

    chimeric = Chimeric(google_api_key=api_keys["google_api_key"])
    cassette_path = get_cassette_path("google", "test_aembed_batch")

    with get_vcr().use_cassette(cassette_path):
        result = await chimeric.aembed(
            model="gemini-embedding-001",
            input=["text one", "text two"],
        )

        assert isinstance(result, EmbeddingResponse)
        assert result.embedding is None
        assert len(result.embeddings) == 2


@pytest.mark.google
def test_google_embed_cosine_similarity(api_keys):
    """Verify that similar texts produce closer embeddings than dissimilar ones via Google."""
    if "google_api_key" not in api_keys:
        pytest.skip("Google API key not found")

    chimeric = Chimeric(google_api_key=api_keys["google_api_key"])
    cassette_path = get_cassette_path("google", "test_embed_cosine_similarity")

    with get_vcr().use_cassette(cassette_path):
        result = chimeric.embed(
            model="gemini-embedding-001",
            input=[
                "Python backend developer",
                "Python backend engineer",
                "Chocolate cake recipe",
            ],
        )

        a, b, c = result.embeddings
        sim_ab = _cosine_similarity(a, b)
        sim_ac = _cosine_similarity(a, c)

        assert sim_ab > sim_ac, (
            f"Expected similar texts to have higher similarity: {sim_ab:.4f} vs {sim_ac:.4f}"
        )


# ---------------------------------------------------------------------------
# Cohere embeddings
# ---------------------------------------------------------------------------


@pytest.mark.cohere
def test_cohere_embed_single(api_keys):
    """Test single-text embedding via Cohere."""
    if "cohere_api_key" not in api_keys:
        pytest.skip("Cohere API key not found")

    chimeric = Chimeric(cohere_api_key=api_keys["cohere_api_key"])
    cassette_path = get_cassette_path("cohere", "test_embed_single")

    with get_vcr().use_cassette(cassette_path):
        result = chimeric.embed(
            model="embed-english-light-v3.0",
            input="Python developer with 5 years experience",
        )

        assert isinstance(result, EmbeddingResponse)
        assert result.embedding is not None
        assert len(result.embedding) > 0
        assert all(isinstance(v, float) for v in result.embedding)
        assert result.model is not None
        assert result.usage is not None
        assert result.usage.prompt_tokens > 0


@pytest.mark.cohere
def test_cohere_embed_batch(api_keys):
    """Test batch embedding via Cohere."""
    if "cohere_api_key" not in api_keys:
        pytest.skip("Cohere API key not found")

    chimeric = Chimeric(cohere_api_key=api_keys["cohere_api_key"])
    cassette_path = get_cassette_path("cohere", "test_embed_batch")

    with get_vcr().use_cassette(cassette_path):
        texts = [
            "Python developer with 5 years experience",
            "Looking for a backend engineer with Go expertise",
            "Frontend React developer needed",
        ]
        result = chimeric.embed(
            model="embed-english-light-v3.0",
            input=texts,
        )

        assert isinstance(result, EmbeddingResponse)
        assert result.embedding is None
        assert len(result.embeddings) == 3
        for emb in result.embeddings:
            assert len(emb) > 0
            assert all(isinstance(v, float) for v in emb)
        assert result.usage is not None
        assert result.usage.prompt_tokens > 0


@pytest.mark.cohere
@pytest.mark.asyncio
async def test_cohere_aembed_single(api_keys):
    """Test async single-text embedding via Cohere."""
    if "cohere_api_key" not in api_keys:
        pytest.skip("Cohere API key not found")

    chimeric = Chimeric(cohere_api_key=api_keys["cohere_api_key"])
    cassette_path = get_cassette_path("cohere", "test_aembed_single")

    with get_vcr().use_cassette(cassette_path):
        result = await chimeric.aembed(
            model="embed-english-light-v3.0",
            input="Python developer with 5 years experience",
        )

        assert isinstance(result, EmbeddingResponse)
        assert result.embedding is not None
        assert len(result.embedding) > 0
        assert result.usage is not None


@pytest.mark.cohere
@pytest.mark.asyncio
async def test_cohere_aembed_batch(api_keys):
    """Test async batch embedding via Cohere."""
    if "cohere_api_key" not in api_keys:
        pytest.skip("Cohere API key not found")

    chimeric = Chimeric(cohere_api_key=api_keys["cohere_api_key"])
    cassette_path = get_cassette_path("cohere", "test_aembed_batch")

    with get_vcr().use_cassette(cassette_path):
        result = await chimeric.aembed(
            model="embed-english-light-v3.0",
            input=["text one", "text two"],
        )

        assert isinstance(result, EmbeddingResponse)
        assert result.embedding is None
        assert len(result.embeddings) == 2


@pytest.mark.cohere
def test_cohere_embed_cosine_similarity(api_keys):
    """Verify that similar texts produce closer embeddings than dissimilar ones via Cohere."""
    if "cohere_api_key" not in api_keys:
        pytest.skip("Cohere API key not found")

    chimeric = Chimeric(cohere_api_key=api_keys["cohere_api_key"])
    cassette_path = get_cassette_path("cohere", "test_embed_cosine_similarity")

    with get_vcr().use_cassette(cassette_path):
        result = chimeric.embed(
            model="embed-english-light-v3.0",
            input=[
                "Python backend developer",
                "Python backend engineer",
                "Chocolate cake recipe",
            ],
        )

        a, b, c = result.embeddings
        sim_ab = _cosine_similarity(a, b)
        sim_ac = _cosine_similarity(a, c)

        assert sim_ab > sim_ac, (
            f"Expected similar texts to have higher similarity: {sim_ab:.4f} vs {sim_ac:.4f}"
        )


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
