from collections.abc import Callable
import inspect
import re
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from .exceptions import ToolRegistrationError
from .types import Tool, ToolParameters

__all__ = [
    "ToolManager",
]


class ToolManager:
    """Tool registration and management system with multiple docstring style support.

    This class provides functionality to register Python functions as tools with
    automatic parameter schema generation from type hints and docstring parsing.
    Supports Google, NumPy, and Sphinx docstring styles.
    """

    def __init__(self) -> None:
        """Initialize the ToolManager with empty tool registry."""
        self.tools: dict[str, Tool] = {}

    def register(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        strict: bool = True,
    ) -> Callable[..., Any]:
        """Register a function as a tool with automatic docstring parsing.

        Args:
            func: The function to register as a tool.
            name: Optional custom name for the tool. Defaults to function name.
            description: Optional custom description. If None, parses from docstring.
            strict: Whether to use strict parameter validation.

        Returns:
            The original function (for use as a decorator).

        Raises:
            ToolRegistrationError: If a tool with the same name already exists.
        """
        tool_name = name or func.__name__
        if tool_name in self.tools:
            raise ToolRegistrationError(tool_name=tool_name, existing_tool=True)

        # Extract description and parameter descriptions
        if description is None:
            parsed = self._parse_docstring(func.__doc__ or "")
            tool_description = parsed["description"]
            param_descriptions = parsed["args"]
        else:
            tool_description = description
            param_descriptions = {}

        # Generate parameter schema
        properties, required = self._extract_parameters(func, param_descriptions)
        function_params = ToolParameters(
            type="object",
            strict=strict,
            properties=properties,
            required=required or None,
            additionalProperties=False,
        )

        # Create and register the tool
        tool = Tool(
            name=tool_name,
            description=tool_description or f"Call {tool_name}",
            parameters=function_params,
            function=func,
        )
        self.tools[tool_name] = tool
        return func

    def get_tool(self, name: str) -> Tool:
        """Get a registered tool by name.

        Args:
            name: The name of the tool to retrieve.

        Returns:
            The Tool instance.

        Raises:
            ToolRegistrationError: If no tool is registered with the given name.
        """
        # Check if the tool exists
        if name not in self.tools:
            raise ToolRegistrationError(f"No tool registered with name '{name}'")
        return self.tools[name]

    def get_all_tools(self) -> list[Tool]:
        """Get all registered tools.

        Returns:
            A list of all registered Tool instances.
        """
        return list(self.tools.values())

    def clear(self) -> None:
        """Clear all registered tools from the registry."""
        self.tools.clear()

    def _parse_docstring(self, doc: str) -> dict[str, Any]:
        """Parse docstring using multiple format parsers.

        Args:
            doc: The docstring to parse.

        Returns:
            Dictionary with 'description' and 'args' keys containing parsed content.
        """
        if not doc:
            return {"description": "", "args": {}}

        # Try different parsers in order of preference
        parsers = [
            self._parse_google_docstring,
            self._parse_numpy_docstring,
            self._parse_sphinx_docstring,
        ]

        for parser in parsers:
            result = parser(doc)
            if result["description"] or result["args"]:
                return result

        return {"description": doc.strip(), "args": {}}

    def _parse_google_docstring(self, doc: str) -> dict[str, Any]:
        """Parse Google-style docstring.

        Args:
            doc: The docstring to parse.

        Returns:
            Dictionary with parsed description and arguments.
        """
        lines = [line.strip() for line in doc.strip().splitlines()]
        description_lines = []
        args_lines = []
        section = "desc"

        for line in lines:
            line_lower = line.lower()

            # Check for the args section
            if line_lower.startswith(("args:", "arguments:", "parameters:")):
                section = "args"
                continue

            # Check for other sections that end args parsing
            if line_lower.startswith(
                (
                    "returns:",
                    "yields:",
                    "raises:",
                    "note:",
                    "notes:",
                    "example:",
                    "examples:",
                    "see also:",
                    "attributes:",
                )
            ):
                section = "other"
                continue

            if section == "desc":
                description_lines.append(line)
            elif section == "args":
                args_lines.append(line)

        description = " ".join(description_lines).strip()
        args = self._parse_google_args_section(args_lines)

        return {"description": description, "args": args}

    def _parse_numpy_docstring(self, docstring: str) -> dict[str, Any]:
        """Parse NumPy-style docstring.

        Args:
            docstring: The docstring to parse.

        Returns:
            Dictionary with parsed description and arguments.
        """
        lines = docstring.split("\n")
        description_lines = []
        args_lines = []
        section = "description"

        for idx, line in enumerate(lines):
            stripped = line.strip()

            # Check for the Parameters section
            if (
                stripped in ("Parameters", "Arguments")
                and idx + 1 < len(lines)
                and set(lines[idx + 1].strip()) == {"-"}
            ):
                section = "args"
                continue

            # Check for other sections
            if (
                stripped in ("Returns", "Yields", "Raises", "Notes", "Examples", "See Also")
                and idx + 1 < len(lines)
                and set(lines[idx + 1].strip()) == {"-"}
            ):
                section = "other"
                continue

            if section == "description":
                description_lines.append(line)
            elif section == "args":
                # Skip blank lines and underlines
                if not stripped or set(stripped) == {"-"}:
                    continue
                args_lines.append(line)

        description = " ".join(line.strip() for line in description_lines).strip()
        args = self._parse_numpy_args_section(args_lines)

        return {"description": description, "args": args}

    @staticmethod
    def _parse_sphinx_docstring(doc: str) -> dict[str, Any]:
        """Parse Sphinx-style docstring.

        Args:
            doc: The docstring to parse.

        Returns:
            Dictionary with parsed description and arguments.
        """
        description_lines = []
        args = {}

        for line in doc.splitlines():
            stripped_line = line.strip()

            # Look for parameter definitions
            param_match = re.match(r":(?:param|parameter)\s+(\w+):\s*(.*)", stripped_line)
            if param_match:
                args[param_match.group(1)] = param_match.group(2)
            elif not stripped_line.startswith(":"):
                description_lines.append(stripped_line)

        description = " ".join(description_lines).strip()
        return {"description": description, "args": args}

    @staticmethod
    def _parse_google_args_section(lines: list[str]) -> dict[str, str]:
        """Parse Google-style arguments section.

        Args:
            lines: Lines from the arguments section of the docstring.

        Returns:
            Dictionary mapping parameter names to their descriptions.
        """
        args = {}
        current_param = None
        description_buffer = []

        for line in lines:
            if line == "":
                continue

            # Check for parameter definition line
            param_match = re.match(r"^(\w+)(?:\s*\([^)]+\))?\s*:\s*(.*)$", line)
            if param_match:
                # Save previous parameter if exists
                if current_param is not None:
                    args[current_param] = " ".join(description_buffer).strip()

                current_param = param_match.group(1)
                description_buffer = [param_match.group(2)]
            else:
                # Continuation line
                if current_param is not None:
                    description_buffer.append(line.strip())

        # Remember the last parameter
        if current_param is not None:
            args[current_param] = " ".join(description_buffer).strip()

        return args

    @staticmethod
    def _parse_numpy_args_section(lines: list[str]) -> dict[str, str]:
        """Parse NumPy-style arguments section.

        Args:
            lines: Lines from the parameters section of the docstring.

        Returns:
            Dictionary mapping parameter names to their descriptions.
        """
        args = {}
        current_param = None
        description_buffer = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check for parameter definition line
            param_match = re.match(r"^(\w+)\s*(?::.*)?$", stripped)
            if param_match:
                # Save previous parameter if exists
                if current_param is not None:
                    args[current_param] = " ".join(description_buffer).strip()
                    description_buffer = []

                current_param = param_match.group(1)

                # Extract description from the same line if present
                desc_match = re.search(r"^[^:]+:\s*(.*)$", stripped)
                if desc_match:
                    type_and_desc = desc_match.group(1).strip()

                    # If it's only type tokens (with optional/default), skip inline description
                    if re.fullmatch(
                        r"\w+(?:\s*,\s*\w+)*(?:\s*,\s*optional)?(?:\s*,\s*default\s+\S+)?",
                        type_and_desc,
                    ):
                        continue

                    # Extract description after default value if present
                    default_match = re.search(r"\bdefault\s+\S+\s+(.*)", type_and_desc)
                    if default_match:
                        description_buffer = [default_match.group(1).strip()]
                    else:
                        # Extract description after optional if present
                        optional_match = re.search(r"\boptional\s+(.*)", type_and_desc)
                        if optional_match:
                            description_buffer = [optional_match.group(1).strip()]
                        # Otherwise, split on the first space to separate type info from description
                        else:
                            description_buffer = [type_and_desc.split(" ", 1)[1].strip()]
            else:
                # Continuation line for current parameter
                if current_param is not None:
                    description_buffer.append(stripped)

        # Remember the last parameter
        if current_param is not None:
            args[current_param] = " ".join(description_buffer).strip()

        return args

    def _extract_parameters(
        self, func: Callable[..., Any], descriptions: dict[str, str]
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        """Extract parameter schema from function signature and type hints.

        Args:
            func: The function to analyze.
            descriptions: Parameter descriptions from docstring parsing.

        Returns:
            Tuple of (properties dict, required parameters list).
        """
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)
        properties = {}
        required = []

        for param_name, param in signature.parameters.items():
            param_type = type_hints.get(param_name, Any)
            description = descriptions.get(param_name, f"Parameter: {param_name}")
            has_default = param.default is not inspect.Parameter.empty

            schema = self._create_parameter_schema(param_type, description)
            properties[param_name] = schema

            # Determine if a parameter is required
            if self._is_parameter_required(param_type, has_default):
                required.append(param_name)

        return properties, required

    @staticmethod
    def _is_parameter_required(param_type: Any, has_default: bool) -> bool:
        """Determine if a parameter is required based on type and default value.

        Args:
            param_type: The parameter's type annotation.
            has_default: Whether the parameter has a default value.

        Returns:
            True if the parameter is required, False otherwise.
        """
        if has_default:
            return False

        # Check if type is Union/Optional and includes None
        origin = get_origin(param_type)
        if origin in (Union, UnionType):
            args = get_args(param_type)
            if type(None) in args:
                return False

        return True

    def _create_parameter_schema(self, param_type: Any, description: str) -> dict[str, Any]:
        """Create JSON schema for a parameter.

        Args:
            param_type: The parameter's type annotation.
            description: Parameter description.

        Returns:
            JSON schema dictionary for the parameter.
        """
        origin = get_origin(param_type)

        # Handle Union types (including Optional)
        if origin in (Union, UnionType):
            args = get_args(param_type)
            if type(None) in args:
                # Optional type
                non_none_types = [arg for arg in args if arg is not type(None)]
                chosen_type = non_none_types[0] if non_none_types else Any
                schema = self._create_schema_for_type(chosen_type, description, nullable=True)
            else:
                # Non-optional union - use first type
                chosen_type = args[0] if args else Any
                schema = self._create_schema_for_type(chosen_type, description, nullable=False)
        else:
            schema = self._create_schema_for_type(param_type, description)

        # Remove 'required' key if present (handled at tool level)
        schema.pop("required", None)
        return schema

    def _create_schema_for_type(
        self, param_type: Any, description: str, nullable: bool = False
    ) -> dict[str, Any]:
        """Create JSON schema for a specific type.

        Args:
            param_type: The type to create schema for.
            description: Description for the schema.
            nullable: Whether the type can be null.

        Returns:
            JSON schema dictionary.
        """
        origin = get_origin(param_type)

        schema: dict[str, Any]
        # Handle generic types
        if origin is list or param_type is list:
            args = get_args(param_type)
            item_schema = self._get_simple_type_schema(args[0] if args else Any)
            schema = {"type": "array", "description": description, "items": item_schema}
        elif origin is dict or param_type is dict:
            schema = {
                "type": "object",
                "description": description,
                "properties": {},
                "additionalProperties": True,
            }
        else:
            # Handle basic types
            schema = self._get_basic_type_schema(param_type, description)

        # Add null type if nullable
        if nullable:
            current_type = schema["type"]
            if isinstance(current_type, str):
                schema["type"] = [current_type, "null"]
            elif isinstance(current_type, list) and "null" not in current_type:
                schema["type"] = [*current_type, "null"]

        return schema

    @staticmethod
    def _get_basic_type_schema(param_type: Any, description: str) -> dict[str, Any]:
        """Get schema for basic Python types.

        Args:
            param_type: The type to get schema for.
            description: Description for the schema.

        Returns:
            JSON schema dictionary.
        """
        # Primitives
        if param_type in (str, type(str)):
            return {"type": "string", "description": description}
        if param_type in (int, type(int)):
            return {"type": "integer", "description": description}
        if param_type in (float, type(float)):
            return {"type": "number", "description": description}
        if param_type in (bool, type(bool)):
            return {"type": "boolean", "description": description}

        # Fallback to string
        return {"type": "string", "description": description}

    @staticmethod
    def _get_simple_type_schema(param_type: Any) -> dict[str, Any]:
        """Get a simple schema for array item types.

        Args:
            param_type: The type to get schema for.

        Returns:
            Simple JSON schema dictionary without description.
        """
        if param_type in (str, type(str)):
            return {"type": "string"}
        if param_type in (int, type(int)):
            return {"type": "integer"}
        if param_type in (float, type(float)):
            return {"type": "number"}
        if param_type in (bool, type(bool)):
            return {"type": "boolean"}
        return {"type": "string"}
