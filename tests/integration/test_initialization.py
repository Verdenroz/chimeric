import os

import pytest

from chimeric import Chimeric
from chimeric.exceptions import ChimericError
from chimeric.types import Provider


@pytest.mark.openai
def test_openai_only_initialization(api_keys):
    """Test functionality when only chimeric[openai] is initialized."""
    if not api_keys.get("openai_api_key"):
        pytest.skip("OpenAI API key not available")

    # Initialize Chimeric with OpenAI API key
    chimeric = Chimeric(openai_api_key=api_keys.get("openai_api_key"))
    assert len(chimeric.providers) == 1
    assert Provider.OPENAI in chimeric.providers


@pytest.mark.openai
def test_openai_only_env_initialization(api_keys):
    """Test functionality when only chimeric[openai] is initialized via environment variables."""
    if not api_keys.get("openai_api_key"):
        pytest.skip("OpenAI API key not available")

    # Set the environment variable for OpenAI API key
    os.environ["OPENAI_API_KEY"] = api_keys["openai_api_key"]

    # Only OpenAI should be initialized
    chimeric = Chimeric()
    assert len(chimeric.providers) == 1
    assert Provider.OPENAI in chimeric.providers


@pytest.mark.anthropic
def test_anthropic_only_initialization(api_keys):
    """Test functionality when only chimeric[anthropic] is initialized."""
    if not api_keys.get("anthropic_api_key"):
        pytest.skip("Anthropic API key not available")

    # Initialize Chimeric with Anthropic API key
    chimeric = Chimeric(anthropic_api_key=api_keys.get("anthropic_api_key"))
    assert len(chimeric.providers) == 1
    assert Provider.ANTHROPIC in chimeric.providers


@pytest.mark.anthropic
def test_anthropic_only_env_initialization(api_keys):
    """Test functionality when only chimeric[anthropic] is initialized via environment variables."""
    if not api_keys.get("anthropic_api_key"):
        pytest.skip("Anthropic API key not available")

    # Set the environment variable for Anthropic API key
    os.environ["ANTHROPIC_API_KEY"] = api_keys["anthropic_api_key"]

    # Only Anthropic should be initialized
    chimeric = Chimeric()
    assert len(chimeric.providers) == 1
    assert Provider.ANTHROPIC in chimeric.providers


@pytest.mark.google
def test_google_only_initialization(api_keys):
    """Test functionality when only chimeric[google] is initialized."""
    if not api_keys.get("google_api_key"):
        pytest.skip("Google API key not available")

    # Initialize Chimeric with Google API key
    chimeric = Chimeric(google_api_key=api_keys.get("google_api_key"))
    assert len(chimeric.providers) == 1
    assert Provider.GOOGLE in chimeric.providers


@pytest.mark.google
def test_google_only_env_initialization(api_keys):
    """Test functionality when only chimeric[google] is initialized via environment variables."""
    if not api_keys.get("google_api_key"):
        pytest.skip("Google API key not available")

    # Set the environment variable for Google API key
    os.environ["GOOGLE_API_KEY"] = api_keys["google_api_key"]

    # Only Google should be initialized
    chimeric = Chimeric()
    assert len(chimeric.providers) == 1
    assert Provider.GOOGLE in chimeric.providers


# Cerebras tests
@pytest.mark.cerebras
def test_cerebras_only_initialization(api_keys):
    """Test functionality when only chimeric[cerebras] is initialized."""
    if not api_keys.get("cerebras_api_key"):
        pytest.skip("Cerebras API key not available")

    # Initialize Chimeric with Cerebras API key
    chimeric = Chimeric(cerebras_api_key=api_keys.get("cerebras_api_key"))
    assert len(chimeric.providers) == 1
    assert Provider.CEREBRAS in chimeric.providers


@pytest.mark.cerebras
def test_cerebras_only_env_initialization(api_keys):
    """Test functionality when only chimeric[cerebras] is initialized via environment variables."""
    if not api_keys.get("cerebras_api_key"):
        pytest.skip("Cerebras API key not available")

    # Set the environment variable for Cerebras API key
    os.environ["CEREBRAS_API_KEY"] = api_keys["cerebras_api_key"]

    # Only Cerebras should be initialized
    chimeric = Chimeric()
    assert len(chimeric.providers) == 1
    assert Provider.CEREBRAS in chimeric.providers


# Cohere tests
@pytest.mark.cohere
def test_cohere_only_initialization(api_keys):
    """Test functionality when only chimeric[cohere] is initialized."""
    if not api_keys.get("cohere_api_key"):
        pytest.skip("Cohere API key not available")

    # Initialize Chimeric with Cohere API key
    chimeric = Chimeric(cohere_api_key=api_keys.get("cohere_api_key"))
    assert len(chimeric.providers) == 1
    assert Provider.COHERE in chimeric.providers


@pytest.mark.cohere
def test_cohere_only_env_initialization(api_keys):
    """Test functionality when only chimeric[cohere] is initialized via environment variables."""
    if not api_keys.get("cohere_api_key"):
        pytest.skip("Cohere API key not available")

    # Set the environment variable for Cohere API key
    os.environ["COHERE_API_KEY"] = api_keys["cohere_api_key"]

    # Only Cohere should be initialized
    chimeric = Chimeric()
    assert len(chimeric.providers) == 1
    assert Provider.COHERE in chimeric.providers


# Grok tests
@pytest.mark.grok
def test_grok_only_initialization(api_keys):
    """Test functionality when only chimeric[grok] is initialized."""
    if not api_keys.get("grok_api_key"):
        pytest.skip("Grok API key not available")

    # Initialize Chimeric with Grok API key
    chimeric = Chimeric(grok_api_key=api_keys.get("grok_api_key"))
    assert len(chimeric.providers) == 1
    assert Provider.GROK in chimeric.providers


@pytest.mark.grok
def test_grok_only_env_initialization(api_keys):
    """Test functionality when only chimeric[grok] is initialized via environment variables."""
    if not api_keys.get("grok_api_key"):
        pytest.skip("Grok API key not available")

    # Set the environment variable for Grok API key
    os.environ["GROK_API_KEY"] = api_keys["grok_api_key"]

    # Only Grok should be initialized
    chimeric = Chimeric()
    assert len(chimeric.providers) == 1
    assert Provider.GROK in chimeric.providers


# Groq tests
@pytest.mark.groq
def test_groq_only_initialization(api_keys):
    """Test functionality when only chimeric[groq] is initialized."""
    if not api_keys.get("groq_api_key"):
        pytest.skip("Groq API key not available")

    # Initialize Chimeric with Groq API key
    chimeric = Chimeric(groq_api_key=api_keys.get("groq_api_key"))
    assert len(chimeric.providers) == 1
    assert Provider.GROQ in chimeric.providers


@pytest.mark.groq
def test_groq_only_env_initialization(api_keys):
    """Test functionality when only chimeric[groq] is initialized via environment variables."""
    if not api_keys.get("groq_api_key"):
        pytest.skip("Groq API key not available")

    # Set the environment variable for Groq API key
    os.environ["GROQ_API_KEY"] = api_keys["groq_api_key"]

    # Only Groq should be initialized
    chimeric = Chimeric()
    assert len(chimeric.providers) == 1
    assert Provider.GROQ in chimeric.providers


@pytest.mark.all_extras
def test_all_providers_initialization(api_keys):
    """Test functionality when all providers are initialized."""
    # Filter out None API keys
    available_keys = {k: v for k, v in api_keys.items() if v is not None}
    chimeric = Chimeric(**available_keys)

    # Should have multiple providers
    assert len(chimeric.providers) == len(available_keys)


@pytest.mark.bare_install
def test_bare_initialization_graceful_failure():
    """Test that bare chimeric initialization fails gracefully."""
    # When no API keys are provided, Chimeric should handle it gracefully
    chimeric = Chimeric()

    # Should have no providers
    assert len(chimeric.providers) == 0

    # Should raise appropriate error when trying to generate
    with pytest.raises(ChimericError):
        chimeric.generate(
            model="gpt-4o-mini",
            messages="Hello",
        )

    # Should return empty list for models
    models = chimeric.list_models()
    assert len(models) == 0
