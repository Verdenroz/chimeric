from collections.abc import Callable, Iterable
import inspect
from typing import Any, Union, cast, get_type_hints

from .exceptions import ToolRegistrationError
from .types import (
    JSONSchemaArray,
    JSONSchemaBoolean,
    JSONSchemaInteger,
    JSONSchemaNumber,
    JSONSchemaObject,
    JSONSchemaString,
    JSONSchemaType,
    Tool,
    ToolParameterMetadata,
    ToolParameters,
    ToolType,
)

__all__ = [
    "ToolManager",
    "tool_parameter",
]


def tool_parameter(
    description: str,
    *,
    required: bool = True,
    enum: Iterable[Any] | None = None,
    format: str | None = None,
) -> ToolParameterMetadata:
    """Create a parameter definition for tool functions.

    This function can be used to annotate function parameters with additional
    metadata for tool registration.

    Args:
        description: Description of the parameter
        required: Whether the parameter is required
        enum: List of possible values for the parameter
        format: Format specifier for the parameter (e.g., 'date-time', 'uri')

    Returns:
        ToolParameterMetadata dictionary with parameter metadata
    """
    metadata: ToolParameterMetadata = {"description": description}

    if not required:
        metadata["required"] = False

    if enum:
        # Convert callable to list to handle the case where a function is passed instead of an iterable
        if callable(enum):
            enum_result = enum()
            # Ensure the callable actually returns an iterable
            if hasattr(enum_result, "__iter__") or isinstance(enum_result, Iterable):
                metadata["enum"] = list(enum_result)
            else:
                # Handle the error case or use a default empty list
                metadata["enum"] = []
        else:
            metadata["enum"] = list(enum)

    if format:
        metadata["format"] = format

    return metadata


class ToolManager:
    """Tool registration and management system.

    This class handles the registration and management of tool functions that can
    be used with LLM providers that support function calling/tools.
    """

    def __init__(self) -> None:
        """Initialize the tool manager."""
        self.tools: dict[str, Tool] = {}

    def register(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[..., Any]:
        """Register a function as a tool.

        This method registers a function as a tool, making it available for use
        with LLM providers that support function calling. The function's parameters
        and their types are automatically extracted and converted to a compatible
        JSON schema.

        Args:
            func: The function to register
            name: Optional name for the tool. If None, the function name is used.
            description: Optional description for the tool. If None, the function's docstring is used.

        Returns:
            The original function, unchanged

        Raises:
            ToolRegistrationError: If the function cannot be registered as a tool
        """
        tool_name = name or func.__name__
        tool_description = description or (func.__doc__ or "").strip()

        # Check for name conflicts
        if tool_name in self.tools:
            raise ToolRegistrationError(tool_name=tool_name, existing_tool=True)

        # Get parameter info from function signature and type annotations
        params = self._extract_parameters(func)

        # Create the function parameters schema
        function_params: ToolParameters = {
            "type": "object",
            "properties": dict(params.items()),
            "required": [name for name, schema in params.items() if schema.get("required", True)],
        }

        # Create the tool and register it
        tool = Tool(
            type=ToolType.FUNCTION,
            name=tool_name,
            description=tool_description if tool_description else f"Call {tool_name}",
            parameters=function_params,
            function=func,
        )

        self.tools[tool_name] = tool
        return func

    def _extract_parameters(self, func: Callable[..., Any]) -> dict[str, JSONSchemaType]:
        """Extract parameter information from function signature and annotations.

        Args:
            func: The function to extract parameters from

        Returns:
            Dictionary mapping parameter names to their JSON schema
        """
        signature = inspect.signature(func)
        hints = get_type_hints(func)
        parameters: dict[str, JSONSchemaType] = {}

        for name, param in signature.parameters.items():
            # Skip self parameter for methods
            if name == "self":
                continue

            param_type = hints.get(name, Any)
            metadata = None

            # Check if the default value is a ToolParameterMetadata
            if (
                param.default is not inspect.Parameter.empty
                and isinstance(param.default, dict)
                and "description" in param.default
            ):
                metadata = cast("ToolParameterMetadata", param.default)

            # Create parameter schema
            param_schema = self._create_parameter_schema(
                param_type,
                name,
                param.default if param.default is not inspect.Parameter.empty else None,
                metadata,
            )

            # Mark as required unless it has a default value
            if param.default is inspect.Parameter.empty and not param_schema.get("required", False):
                parameters[name] = param_schema

        return parameters

    def _create_parameter_schema(
        self,
        param_type: Any,
        name: str,
        default: Any = None,
        metadata: ToolParameterMetadata | None = None,
    ) -> JSONSchemaType:
        """Create JSON Schema for a parameter based on its Python type.

        Args:
            param_type: Python type annotation
            name: Parameter name
            default: Default value, if any
            metadata: Additional parameter metadata

        Returns:
            JSONSchemaType with JSON Schema for the parameter
        """
        # Determine the appropriate JSON Schema type
        json_type = self._get_json_type(param_type)
        description = (
            metadata.get("description", f"Parameter: {name}") if metadata else f"Parameter: {name}"
        )

        # Create the appropriate schema based on the type
        if json_type == "string":
            string_schema: JSONSchemaString = {
                "type": "string",
                "description": description,
            }
            if metadata and "enum" in metadata:
                string_schema["enum"] = metadata["enum"]
            if metadata and "format" in metadata:
                string_schema["format"] = metadata["format"]
            if default is not None:
                string_schema["default"] = str(default)
            schema = cast("JSONSchemaType", string_schema)

        elif json_type == "number":
            number_schema: JSONSchemaNumber = {
                "type": "number",
                "description": description,
            }
            if default is not None:
                number_schema["default"] = float(default)
            schema = cast("JSONSchemaType", number_schema)

        elif json_type == "integer":
            integer_schema: JSONSchemaInteger = {
                "type": "integer",
                "description": description,
            }
            if default is not None:
                integer_schema["default"] = int(default)
            schema = cast("JSONSchemaType", integer_schema)

        elif json_type == "boolean":
            boolean_schema: JSONSchemaBoolean = {
                "type": "boolean",
                "description": description,
            }
            if default is not None:
                boolean_schema["default"] = bool(default)
            schema = cast("JSONSchemaType", boolean_schema)

        elif json_type == "array":
            array_schema: JSONSchemaArray = {
                "type": "array",
                "description": description,
            }

            # Get item type for the array
            item_type = getattr(param_type, "__args__", (Any,))[0]
            array_schema["items"] = {"type": self._get_json_type(item_type)}

            if default is not None:
                array_schema["default"] = list(default)
            schema = cast("JSONSchemaType", array_schema)

        elif json_type == "object":
            object_schema: JSONSchemaObject = {
                "type": "object",
                "description": description,
                "properties": {},
            }
            if default is not None:
                object_schema["default"] = dict(default)
            schema = cast("JSONSchemaType", object_schema)

        else:
            # Default to string for unknown types
            unknown_schema: JSONSchemaString = {
                "type": "string",
                "description": description,
            }
            if default is not None:
                unknown_schema["default"] = str(default)
            schema = cast("JSONSchemaType", unknown_schema)

        # Handle Union types (Optional)
        origin = getattr(param_type, "__origin__", None)
        if origin is Union and getattr(param_type, "__args__", None):
            args = param_type.__args__
            if type(None) in args and "required" in schema:  # Optional type
                # For Optional types, we'll mark it as not required
                schema["required"] = False

        # Handle the required flag if present in metadata
        if metadata and "required" in metadata:
            schema["required"] = metadata["required"]

        return schema

    def _get_json_type(self, python_type: Any) -> str:
        """Convert Python type to JSON schema type.

        Args:
            python_type: Python type to convert

        Returns:
            String with JSON schema type
        """
        # Handle common Python types
        origin = getattr(python_type, "__origin__", None)
        if origin is list or origin is list:
            return "array"
        if origin is dict or origin is dict:
            return "object"
        if origin is Union:
            # For Union types, check if it's Optional (Union with None)
            args = python_type.__args__
            if type(None) in args and len(args) == 2:
                # Get the non-None type
                non_none_type = next(arg for arg in args if arg is not type(None))
                return self._get_json_type(non_none_type)
            # For other Unions, default to string as it's most flexible
            return "string"

        # Map Python built-in types to JSON schema types
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            # Add more types as needed
        }

        # Check for exact type match
        for py_type, json_type in type_mapping.items():
            if python_type is py_type:
                return json_type

        # Default to string for unknown types
        return "string"

    def get_tool(self, name: str) -> Tool:
        """Get a registered tool by name.

        Args:
            name: Name of the tool to retrieve

        Returns:
            The Tool object for the specified name

        Raises:
            KeyError: If no tool exists with the specified name
        """
        if name not in self.tools:
            raise KeyError(f"No tool registered with name '{name}'")
        return self.tools[name]

    def get_all_tools(self) -> list[Tool]:
        """Get all registered tools.

        Returns:
            List of all registered Tool objects
        """
        return list(self.tools.values())

    def clear(self) -> None:
        """Clear all registered tools."""
        self.tools.clear()
