import os
from pathlib import Path

from dotenv import load_dotenv
import pytest

from chimeric import Chimeric, ToolManager
from chimeric.types import Message

from .vcr_config import get_vcr

# Load environment variables from .env file for integration tests
load_dotenv()


@pytest.fixture(scope="session")
def vcr_cassette_dir():
    """Get the VCR cassette directory."""
    return Path(__file__).parent / "cassettes"


@pytest.fixture
def real_api_keys():
    """Get real API keys from environment variables for recording."""
    return {
        "openai_api_key": os.environ.get("OPENAI_API_KEY"),
        "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY"),
        "google_api_key": os.environ.get("GOOGLE_API_KEY", os.environ.get("GEMINI_API_KEY")),
        "cerebras_api_key": os.environ.get("CEREBRAS_API_KEY"),
        "cohere_api_key": os.environ.get("COHERE_API_KEY", os.environ.get("CO_API_KEY")),
        "grok_api_key": os.environ.get("GROK_API_KEY", os.environ.get("GROK_API_TOKEN")),
        "groq_api_key": os.environ.get("GROQ_API_KEY"),
    }


@pytest.fixture
def api_keys(real_api_keys):
    """Get appropriate API keys based on VCR recording mode."""
    # Always use real keys for recording
    # Filter out None values
    return {k: v for k, v in real_api_keys.items() if v is not None}


@pytest.fixture
def chimeric_client(api_keys):
    """Create a Chimeric client with appropriate API keys."""
    return Chimeric(**api_keys)


@pytest.fixture
def test_messages():
    """Standard test messages for integration testing."""
    return [
        Message(role="system", content="You are a helpful assistant that responds concisely."),
        Message(role="user", content="Say hello and tell me 2+2."),
    ]


@pytest.fixture
def simple_test_message():
    """Simple test message for basic functionality testing."""
    return "What is 2+2? Answer in one word."


@pytest.fixture
def integration_tool_manager():
    """Tool manager with tools for integration testing."""
    tool_manager = ToolManager()

    def calculate_sum(x: int, y: int) -> int:
        """Calculate the sum of two numbers."""
        return x + y

    def get_weather(location: str) -> str:
        """Get weather information for a location."""
        return f"The weather in {location} is sunny with a temperature of 72°F."

    async def async_multiply(x: int, y: int) -> int:
        """Multiply two numbers asynchronously."""
        return x * y

    tool_manager.register(calculate_sum)
    tool_manager.register(get_weather)
    tool_manager.register(async_multiply)

    return tool_manager


@pytest.fixture
def provider_specific_kwargs():
    """Provider-specific kwargs for testing."""
    return {
        "openai": {
            "organization": "test-org",
            "base_url": "https://api.openai.com/v1",
        },
        "anthropic": {
            "base_url": "https://api.anthropic.com",
            "max_retries": 3,
        },
        "google": {
            "timeout": 30,
        },
        "cerebras": {
            "base_url": "https://api.cerebras.ai/v1",
        },
        "cohere": {
            "timeout": 45,
        },
        "grok": {
            "base_url": "https://api.x.ai/v1",
        },
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
        },
    }


@pytest.fixture(scope="session")
def vcr_config():
    """Get VCR configuration for pytest-vcr."""

    # Get the VCR instance and extract its configuration
    vcr_instance = get_vcr()
    return {
        "record_mode": "once",
        "match_on": vcr_instance.match_on,
        "filter_headers": vcr_instance.filter_headers,
        "filter_query_parameters": getattr(vcr_instance, "filter_query_parameters", []),
    }
