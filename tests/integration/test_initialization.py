import pytest

from chimeric import Chimeric
from chimeric.exceptions import ChimericError
from chimeric.types import Provider

# All env-var names that Chimeric checks during auto-detection (see config.py)
_ALL_API_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "COHERE_API_KEY",
    "CO_API_KEY",
    "GROQ_API_KEY",
    "CEREBRAS_API_KEY",
    "GROK_API_KEY",
    "XAI_API_KEY",
    "OPENROUTER_API_KEY",
)


@pytest.fixture()
def clean_provider_env(monkeypatch):
    """Remove all provider API key env vars so Chimeric starts from a blank slate."""
    for var in _ALL_API_KEY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.mark.openai
def test_openai_only_initialization(api_keys, clean_provider_env):
    """Test functionality when only chimeric[openai] is initialized."""
    if not api_keys.get("openai_api_key"):
        pytest.skip("OpenAI API key not available")

    chimeric = Chimeric(openai_api_key=api_keys.get("openai_api_key"))
    assert len(chimeric.providers) == 1
    assert Provider.OPENAI in chimeric.providers


@pytest.mark.openai
def test_openai_only_env_initialization(api_keys, monkeypatch, clean_provider_env):
    """Test functionality when only chimeric[openai] is initialized via environment variables."""
    if not api_keys.get("openai_api_key"):
        pytest.skip("OpenAI API key not available")

    monkeypatch.setenv("OPENAI_API_KEY", api_keys["openai_api_key"])

    chimeric = Chimeric()
    assert len(chimeric.providers) == 1
    assert Provider.OPENAI in chimeric.providers


@pytest.mark.anthropic
def test_anthropic_only_initialization(api_keys, clean_provider_env):
    """Test functionality when only chimeric[anthropic] is initialized."""
    if not api_keys.get("anthropic_api_key"):
        pytest.skip("Anthropic API key not available")

    chimeric = Chimeric(anthropic_api_key=api_keys.get("anthropic_api_key"))
    assert len(chimeric.providers) == 1
    assert Provider.ANTHROPIC in chimeric.providers


@pytest.mark.anthropic
def test_anthropic_only_env_initialization(api_keys, monkeypatch, clean_provider_env):
    """Test functionality when only chimeric[anthropic] is initialized via environment variables."""
    if not api_keys.get("anthropic_api_key"):
        pytest.skip("Anthropic API key not available")

    monkeypatch.setenv("ANTHROPIC_API_KEY", api_keys["anthropic_api_key"])

    chimeric = Chimeric()
    assert len(chimeric.providers) == 1
    assert Provider.ANTHROPIC in chimeric.providers


@pytest.mark.google
def test_google_only_initialization(api_keys, clean_provider_env):
    """Test functionality when only chimeric[google] is initialized."""
    if not api_keys.get("google_api_key"):
        pytest.skip("Google API key not available")

    chimeric = Chimeric(google_api_key=api_keys.get("google_api_key"))
    assert len(chimeric.providers) == 1
    assert Provider.GOOGLE in chimeric.providers


@pytest.mark.google
def test_google_only_env_initialization(api_keys, monkeypatch, clean_provider_env):
    """Test functionality when only chimeric[google] is initialized via environment variables."""
    if not api_keys.get("google_api_key"):
        pytest.skip("Google API key not available")

    monkeypatch.setenv("GOOGLE_API_KEY", api_keys["google_api_key"])

    chimeric = Chimeric()
    assert len(chimeric.providers) == 1
    assert Provider.GOOGLE in chimeric.providers


@pytest.mark.cerebras
def test_cerebras_only_initialization(api_keys, clean_provider_env):
    """Test functionality when only chimeric[cerebras] is initialized."""
    if not api_keys.get("cerebras_api_key"):
        pytest.skip("Cerebras API key not available")

    chimeric = Chimeric(cerebras_api_key=api_keys.get("cerebras_api_key"))
    assert len(chimeric.providers) == 1
    assert Provider.CEREBRAS in chimeric.providers


@pytest.mark.cerebras
def test_cerebras_only_env_initialization(api_keys, monkeypatch, clean_provider_env):
    """Test functionality when only chimeric[cerebras] is initialized via environment variables."""
    if not api_keys.get("cerebras_api_key"):
        pytest.skip("Cerebras API key not available")

    monkeypatch.setenv("CEREBRAS_API_KEY", api_keys["cerebras_api_key"])

    chimeric = Chimeric()
    assert len(chimeric.providers) == 1
    assert Provider.CEREBRAS in chimeric.providers


@pytest.mark.cohere
def test_cohere_only_initialization(api_keys, clean_provider_env):
    """Test functionality when only chimeric[cohere] is initialized."""
    if not api_keys.get("cohere_api_key"):
        pytest.skip("Cohere API key not available")

    chimeric = Chimeric(cohere_api_key=api_keys.get("cohere_api_key"))
    assert len(chimeric.providers) == 1
    assert Provider.COHERE in chimeric.providers


@pytest.mark.cohere
def test_cohere_only_env_initialization(api_keys, monkeypatch, clean_provider_env):
    """Test functionality when only chimeric[cohere] is initialized via environment variables."""
    if not api_keys.get("cohere_api_key"):
        pytest.skip("Cohere API key not available")

    monkeypatch.setenv("COHERE_API_KEY", api_keys["cohere_api_key"])

    chimeric = Chimeric()
    assert len(chimeric.providers) == 1
    assert Provider.COHERE in chimeric.providers


@pytest.mark.grok
def test_grok_only_initialization(api_keys, clean_provider_env):
    """Test functionality when only chimeric[grok] is initialized."""
    if not api_keys.get("grok_api_key"):
        pytest.skip("Grok API key not available")

    chimeric = Chimeric(grok_api_key=api_keys.get("grok_api_key"))
    assert len(chimeric.providers) == 1
    assert Provider.GROK in chimeric.providers


@pytest.mark.grok
def test_grok_only_env_initialization(api_keys, monkeypatch, clean_provider_env):
    """Test functionality when only chimeric[grok] is initialized via environment variables."""
    if not api_keys.get("grok_api_key"):
        pytest.skip("Grok API key not available")

    monkeypatch.setenv("GROK_API_KEY", api_keys["grok_api_key"])

    chimeric = Chimeric()
    assert len(chimeric.providers) == 1
    assert Provider.GROK in chimeric.providers


@pytest.mark.groq
def test_groq_only_initialization(api_keys, clean_provider_env):
    """Test functionality when only chimeric[groq] is initialized."""
    if not api_keys.get("groq_api_key"):
        pytest.skip("Groq API key not available")

    chimeric = Chimeric(groq_api_key=api_keys.get("groq_api_key"))
    assert len(chimeric.providers) == 1
    assert Provider.GROQ in chimeric.providers


@pytest.mark.groq
def test_groq_only_env_initialization(api_keys, monkeypatch, clean_provider_env):
    """Test functionality when only chimeric[groq] is initialized via environment variables."""
    if not api_keys.get("groq_api_key"):
        pytest.skip("Groq API key not available")

    monkeypatch.setenv("GROQ_API_KEY", api_keys["groq_api_key"])

    chimeric = Chimeric()
    assert len(chimeric.providers) == 1
    assert Provider.GROQ in chimeric.providers


@pytest.mark.all_extras
def test_all_providers_initialization(api_keys):
    """Test functionality when all providers are initialized."""
    available_keys = {k: v for k, v in api_keys.items() if v is not None}
    chimeric = Chimeric(**available_keys)

    assert len(chimeric.providers) == len(available_keys)


@pytest.mark.bare_install
def test_bare_initialization_graceful_failure(clean_provider_env):
    """Test that bare chimeric initialization fails gracefully."""
    chimeric = Chimeric()

    assert len(chimeric.providers) == 0

    with pytest.raises(ChimericError):
        chimeric.generate(
            model="gpt-4o-mini",
            messages="Hello",
        )

    models = chimeric.list_models()
    assert len(models) == 0
