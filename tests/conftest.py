from chimeric import ToolManager
from chimeric.types import Message


class BaseProviderTestSuite:
    """Base test suite that can be reused for all provider clients.

    Subclasses should implement the abstract properties to customize for each provider.
    """

    # Provider-specific configuration - override in subclasses
    client_class = None
    provider_name = None
    mock_client_path = None

    # Sample test data
    @property
    def sample_model(self):
        return "test-model"

    @property
    def sample_messages(self):
        return [Message(role="user", content="Hello")]

    @property
    def sample_response(self):
        """Override in subclass to return provider-specific response."""
        raise NotImplementedError

    @property
    def sample_stream_events(self):
        """Override in subclass to return provider-specific stream events."""
        raise NotImplementedError

    def create_tool_manager(self):
        """Create a tool manager with test tools."""
        tool_manager = ToolManager()

        # Register sync tool
        def test_tool(x: int) -> str:
            """Test tool."""
            return f"Result: {x}"

        # Register async tool
        async def async_test_tool(x: int) -> str:
            """Async test tool."""
            return f"Async result: {x}"

        # Register error tool
        def error_tool(x: int) -> str:
            """Tool that raises an error."""
            raise ValueError("Tool error")

        tool_manager.register(test_tool)
        tool_manager.register(async_test_tool)
        tool_manager.register(error_tool)

        return tool_manager
