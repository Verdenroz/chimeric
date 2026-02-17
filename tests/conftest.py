"""Shared test fixtures and base utilities."""

from chimeric import ToolManager
from chimeric.types import Message


class BaseProviderTestSuite:
    """Reusable base class for provider-specific test suites.

    Subclasses should override the abstract properties to supply
    provider-specific sample data.
    """

    @property
    def sample_model(self) -> str:
        return "test-model"

    @property
    def sample_messages(self) -> list[Message]:
        return [Message(role="user", content="Hello")]

    @property
    def sample_response(self) -> object:
        """Override in subclass to return a provider-specific response dict."""
        raise NotImplementedError

    @property
    def sample_stream_events(self) -> list[object]:
        """Override in subclass to return provider-specific SSE payloads."""
        raise NotImplementedError

    def create_tool_manager(self) -> ToolManager:
        """Create a ToolManager pre-populated with test tools."""
        tool_manager = ToolManager()

        def test_tool(x: int) -> str:
            """Test tool."""
            return f"Result: {x}"

        async def async_test_tool(x: int) -> str:
            """Async test tool."""
            return f"Async result: {x}"

        def error_tool(x: int) -> str:
            """Tool that raises an error."""
            raise ValueError("Tool error")

        tool_manager.register(test_tool)
        tool_manager.register(async_test_tool)
        tool_manager.register(error_tool)
        return tool_manager
