import pytest

from chimeric import ToolManager
from chimeric.types import Message


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    """Clear all API key environment variables for unit tests."""
    api_key_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "CEREBRAS_API_KEY",
        "COHERE_API_KEY",
        "CO_API_KEY",
        "GROK_API_KEY",
        "GROK_API_TOKEN",
        "GROQ_API_KEY",
    ]

    for var in api_key_vars:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def mock_api_keys():
    """Provide mock API keys for unit testing."""
    return {
        "openai_api_key": "test-openai-key",
        "anthropic_api_key": "test-anthropic-key",
        "google_api_key": "test-google-key",
        "cerebras_api_key": "test-cerebras-key",
        "cohere_api_key": "test-cohere-key",
        "grok_api_key": "test-grok-key",
        "groq_api_key": "test-groq-key",
    }


@pytest.fixture
def sample_messages():
    """Provide sample messages for testing."""
    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello, how are you?"),
    ]


@pytest.fixture
def test_tool_manager():
    """Create a tool manager with test tools for unit testing."""
    tool_manager = ToolManager()

    def add_numbers(x: int, y: int) -> int:
        """Add two numbers together."""
        return x + y

    async def async_add_numbers(x: int, y: int) -> int:
        """Async add two numbers together."""
        return x + y

    def error_tool() -> str:
        """A tool that always raises an error."""
        raise ValueError("Test error")

    tool_manager.register(add_numbers)
    tool_manager.register(async_add_numbers)
    tool_manager.register(error_tool)

    return tool_manager


@pytest.fixture
def sample_kwargs():
    """Provide sample kwargs for testing parameter propagation."""
    return {
        "temperature": 0.7,
        "max_tokens": 100,
        "timeout": 30,
        "base_url": "https://custom-endpoint.example.com",
    }
