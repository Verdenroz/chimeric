from typing import Any

import pytest

from chimeric import ToolManager
from chimeric.exceptions import ToolRegistrationError


# noinspection PyUnusedLocal
class TestToolManager:
    @pytest.fixture
    def tool_manager(self):
        """Create a fresh ToolManager instance for each test."""
        return ToolManager()

    def test_init(self):
        """Test ToolManager initialization."""
        tm = ToolManager()
        assert tm.tools == {}

    def test_register_basic_function(self, tool_manager):
        """Test registering basic function without parameters."""

        def simple_func():
            """Simple function."""
            return "result"

        result = tool_manager.register(simple_func)
        assert result is simple_func  # Should return original function
        assert "simple_func" in tool_manager.tools

        tool = tool_manager.get_tool("simple_func")
        assert tool.name == "simple_func"
        assert tool.description == "Simple function."
        assert tool.function is simple_func

    def test_register_with_custom_name(self, tool_manager):
        """Test registering function with custom name."""

        def func():
            pass

        tool_manager.register(func, name="custom_name")
        assert "custom_name" in tool_manager.tools
        assert "func" not in tool_manager.tools

    def test_register_with_custom_description(self, tool_manager):
        """Test registering function with custom description."""

        def func():
            """Original description."""
            pass

        tool_manager.register(func, description="Custom description")
        tool = tool_manager.get_tool("func")
        assert tool.description == "Custom description"

    def test_register_duplicate_name_raises_error(self, tool_manager):
        """Test that registering duplicate tool name raises error."""

        def func1():
            pass

        def func2():
            pass

        tool_manager.register(func1, name="duplicate")

        with pytest.raises(ToolRegistrationError) as exc_info:
            tool_manager.register(func2, name="duplicate")
        assert "duplicate" in str(exc_info.value)

    def test_register_with_parameters(self, tool_manager):
        """Test registering function with typed parameters."""

        def func_with_params(name: str, age: int, active: bool = True):
            """Function with parameters.

            Args:
                name: Person's name
                age: Person's age
                active: Whether person is active
            """
            return f"{name} is {age} years old"

        tool_manager.register(func_with_params)
        tool = tool_manager.get_tool("func_with_params")

        params = tool.parameters
        assert params.type == "object"
        assert params.additionalProperties is False
        assert set(params.required) == {"name", "age"}  # active has default

        # Check parameter schemas
        assert params.properties["name"]["type"] == "string"
        assert params.properties["name"]["description"] == "Person's name"
        assert params.properties["age"]["type"] == "integer"
        assert params.properties["age"]["description"] == "Person's age"
        assert params.properties["active"]["type"] == "boolean"
        assert params.properties["active"]["description"] == "Whether person is active"

    def test_extract_parameters_with_self(self, tool_manager):
        """Test that 'self' parameter is ignored."""

        class TestClass:
            def method(self, param: str):
                pass

        instance = TestClass()
        tool_manager.register(instance.method)
        tool = tool_manager.get_tool("method")

        assert "self" not in tool.parameters.properties
        assert "param" in tool.parameters.properties

    def test_optional_parameters(self, tool_manager):
        """Test handling of Optional parameters."""

        def func_with_optional(required: str, optional: str | None = None):
            """Function with optional parameter.

            Args:
                required: Required parameter
                optional: Optional parameter
            """
            pass

        tool_manager.register(func_with_optional)
        tool = tool_manager.get_tool("func_with_optional")

        assert tool.parameters.required == ["required"]
        assert "optional" not in tool.parameters.required
        assert tool.parameters.properties["optional"]["type"] == ["string", "null"]

    def test_union_parameters(self, tool_manager):
        """Test handling of Union parameters."""

        def func_with_union(param: str | int, param2: float = 0.0):
            """Function with union parameter.

            Args:
                param: Can be string or int
                param2: Optional float parameter
            """
            pass

        tool_manager.register(func_with_union)
        tool = tool_manager.get_tool("func_with_union")

        # Should use first type in union
        assert tool.parameters.properties["param"]["type"] == "string"
        assert tool.parameters.properties["param2"]["type"] == "number"

    def test_union_with_none_parameters(self, tool_manager):
        """Test handling of Union with None parameters."""

        def func_with_union_none(param: str | int | None):
            """Function with union parameter including None.

            Args:
                param: Can be string, int, or None
            """
            pass

        tool_manager.register(func_with_union_none)
        tool = tool_manager.get_tool("func_with_union_none")

        # Should use first non-None type and mark as nullable
        assert tool.parameters.properties["param"]["type"] == ["string", "null"]

    def test_list_parameters(self, tool_manager):
        """Test handling of List parameters."""

        def func_with_list(items: list[str]):
            """Function with list parameter.

            Args:
                items: List of strings
            """
            pass

        tool_manager.register(func_with_list)
        tool = tool_manager.get_tool("func_with_list")

        param_schema = tool.parameters.properties["items"]
        assert param_schema["type"] == "array"
        assert param_schema["items"]["type"] == "string"

    def test_dict_parameters(self, tool_manager):
        """Test handling of Dict parameters."""

        def func_with_dict(data: dict[str, Any]):
            """Function with dict parameter.

            Args:
                data: Dictionary data
            """
            pass

        tool_manager.register(func_with_dict)
        tool = tool_manager.get_tool("func_with_dict")

        param_schema = tool.parameters.properties["data"]
        assert param_schema["type"] == "object"
        assert param_schema["additionalProperties"] is True

    def test_untyped_parameters(self, tool_manager):
        """Test handling of parameters without type hints."""

        def func_without_types(param):
            """Function without type hints.

            Args:
                param: Some parameter
            """
            pass

        tool_manager.register(func_without_types)
        tool = tool_manager.get_tool("func_without_types")

        # Should default to string type
        assert tool.parameters.properties["param"]["type"] == "string"

    def test_unknown_type_parameters(self, tool_manager):
        """Test handling of unknown type parameters."""

        class CustomType:
            pass

        def func_with_custom_type(param: CustomType):
            """Function with custom type.

            Args:
                param: Custom type parameter
            """
            pass

        tool_manager.register(func_with_custom_type)
        tool = tool_manager.get_tool("func_with_custom_type")

        # Should default to string type for unknown types
        assert tool.parameters.properties["param"]["type"] == "string"

    def test_parse_google_docstring_full(self, tool_manager):
        """Test parsing complete Google-style docstring."""

        def func(param1, param2: str, param3):
            """This is a function description.

            Args:
                param1: First parameter description
                param2 (str): Second parameter with type
                param3: Multi-line parameter
                    description that continues

            Returns:
                str: Return value description

            Raises:
                ValueError: When something goes wrong

            Note:
                This is a note section
            """
            pass

        result = tool_manager._parse_google_docstring(func.__doc__)
        assert result["description"] == "This is a function description."
        assert result["args"]["param1"] == "First parameter description"
        assert result["args"]["param2"] == "Second parameter with type"
        assert "Multi-line parameter description that continues" in result["args"]["param3"]

    def test_parse_google_docstring_variations(self, tool_manager):
        """Test parsing Google docstring with different section names."""
        docstring = """Description here.

        Arguments:
            This line should be ignored.
            arg1: Argument description

        Parameters:
            param1: Parameter description
        """

        result = tool_manager._parse_google_docstring(docstring)
        assert result["description"] == "Description here."
        assert "arg1" in result["args"]

    def test_parse_numpy_docstring_full(self, tool_manager):
        """Test parsing complete NumPy-style docstring with full coverage of edge cases."""

        def func(
            param1: str,
            param2: int = 42,
            param3: float = 3.14,
            param4: bool = False,
            param5: str = "test",
            param6: str = "x",
        ):
            """This is a function description.

            Parameters
            ----------
            Continuation line before any parameter is defined
            param1 : str
                First parameter description
            param2 : int, optional
                Second parameter description
                with multiple lines
            param3 : ndarray shape (n,) The array parameter with shape info
            param4 : bool
            param5 : str, default "test" Fourth parameter with default and description
            param6 : customtype

            Returns
            -------
            str
                Return description

            Notes
            -----
            Some notes here
            """
            pass

        result = tool_manager._parse_numpy_docstring(func.__doc__)

        assert result["description"].startswith("This is a function description.")
        assert result["args"]["param1"] == "First parameter description"
        assert "Second parameter description with multiple lines" in result["args"]["param2"]
        assert result["args"]["param3"] == "shape (n,) The array parameter with shape info"
        assert result["args"]["param4"] == ""
        assert result["args"]["param5"] == "Fourth parameter with default and description"
        assert result["args"]["param6"] == ""
        assert len(result["args"]) == 6

    def test_parse_numpy_docstring_alternative_sections(self, tool_manager):
        """Test NumPy docstring with Arguments instead of Parameters."""
        docstring = """Description.

        Arguments
        ---------
        arg1 : str
            Argument description
        """

        result = tool_manager._parse_numpy_docstring(docstring)
        assert result["args"]["arg1"] == "Argument description"

    def test_parse_sphinx_docstring_full(self, tool_manager):
        """Test parsing complete Sphinx-style docstring."""

        def func(param1: str, param2: str):
            """This is a function description.

            :param param1: First parameter description
            :parameter param2: Second parameter description
            :type param1: str
            :returns: Return description
            :rtype: str
            """
            pass

        result = tool_manager._parse_sphinx_docstring(func.__doc__)
        assert result["description"] == "This is a function description."
        assert result["args"]["param1"] == "First parameter description"
        assert result["args"]["param2"] == "Second parameter description"

    def test_parse_unknown_docstring_format(self, tool_manager):
        """Test parsing docstring with unknown format."""
        docstring = "Just a simple description without any special formatting."
        result = tool_manager._parse_docstring(docstring)
        assert result["description"] == docstring
        assert result["args"] == {}

    def test_parse_docstring_fallback_order(self, tool_manager):
        """Test that docstring parsing tries different formats in order."""
        # Create a docstring that would fail Google parsing but work with plain text
        docstring = "Simple description"
        result = tool_manager._parse_docstring(docstring)
        assert result["description"] == "Simple description"

    def test_parse_docstring_fallback(self, tool_manager):
        """Test that _parse_docstring returns the raw docstring when no parser succeeds."""
        # A docstring with no recognizable format
        docstring = "Just a plain description with no structured format."

        # Create properly typed mock function
        def mock_parser(doc: str) -> dict[str, Any]:
            return {"description": None, "args": {}}

        # Mock the parser methods to always return empty results
        original_google_parser = tool_manager._parse_google_docstring
        original_numpy_parser = tool_manager._parse_numpy_docstring
        original_sphinx_parser = tool_manager._parse_sphinx_docstring

        tool_manager._parse_google_docstring = mock_parser
        tool_manager._parse_numpy_docstring = mock_parser
        tool_manager._parse_sphinx_docstring = mock_parser

        try:
            result = tool_manager._parse_docstring(docstring)
            # The function should use the plain docstring as fallback
            assert result["description"] == docstring.strip()
            assert result["args"] == {}
        finally:
            # Restore original parser methods
            tool_manager._parse_google_docstring = original_google_parser
            tool_manager._parse_numpy_docstring = original_numpy_parser
            tool_manager._parse_sphinx_docstring = original_sphinx_parser

    def test_numpy_docstring_same_line_description(self, tool_manager):
        """Test NumPy docstring with description on the same line as parameter."""
        docstring = """Description.

        Parameters
        ----------
        param1 : str Same line description here
        param2 : int, optional Another same line description
        param3 : float, optional, default 1.0 Description with default value
        """

        result = tool_manager._parse_numpy_docstring(docstring)
        assert result["args"]["param1"] == "Same line description here"
        assert result["args"]["param2"] == "Another same line description"
        assert result["args"]["param3"] == "Description with default value"

    def test_parse_numpy_args_section_complex(self, tool_manager):
        """Test parsing complex NumPy arguments section."""
        args_section = [
            "param1 : str",
            "    First parameter description",
            "param2 : int, optional",
            "    Second parameter description",
            "    with continuation",
            "param3",
            "    Third parameter without type",
        ]

        result = tool_manager._parse_numpy_args_section(args_section)
        assert result["param1"] == "First parameter description"
        assert result["param2"] == "Second parameter description with continuation"
        assert result["param3"] == "Third parameter without type"

    def test_get_simple_type_schema_all_types(self, tool_manager):
        """Test getting simple type schemas for all supported types."""
        assert tool_manager._get_simple_type_schema(str) == {"type": "string"}
        assert tool_manager._get_simple_type_schema(int) == {"type": "integer"}
        assert tool_manager._get_simple_type_schema(float) == {"type": "number"}
        assert tool_manager._get_simple_type_schema(bool) == {"type": "boolean"}

        # Unknown type should default to string
        class Unknown:
            pass

        assert tool_manager._get_simple_type_schema(Unknown) == {"type": "string"}

    def test_get_tool_existing(self, tool_manager):
        """Test getting existing tool."""

        def test_func():
            pass

        tool_manager.register(test_func)
        tool = tool_manager.get_tool("test_func")
        assert tool.name == "test_func"

    def test_get_tool_nonexistent(self, tool_manager):
        """Test getting non-existent tool raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            tool_manager.get_tool("nonexistent")
        assert "No tool registered with name 'nonexistent'" in str(exc_info.value)

    def test_get_all_tools_empty(self, tool_manager):
        """Test getting all tools when none registered."""
        assert tool_manager.get_all_tools() == []

    def test_get_all_tools_multiple(self, tool_manager):
        """Test getting all tools with multiple registered."""

        def func1():
            pass

        def func2():
            pass

        tool_manager.register(func1)
        tool_manager.register(func2)

        tools = tool_manager.get_all_tools()
        assert len(tools) == 2
        tool_names = [tool.name for tool in tools]
        assert "func1" in tool_names
        assert "func2" in tool_names

    def test_clear(self, tool_manager):
        """Test clearing all tools."""

        def func1():
            pass

        def func2():
            pass

        tool_manager.register(func1)
        tool_manager.register(func2, strict=False)

        assert len(tool_manager.tools) == 2
        tool_manager.clear()
        assert len(tool_manager.tools) == 0

    def test_no_required_parameters(self, tool_manager):
        """Test function with no required parameters."""

        def func_all_optional(a: int = 1, b: str = "default"):
            """Function with all optional parameters.

            Args:
                a: First optional parameter
                b: Second optional parameter
            """
            pass

        tool_manager.register(func_all_optional)
        tool = tool_manager.get_tool("func_all_optional")

        assert tool.parameters.required is None

    def test_no_description_fallback(self, tool_manager):
        """Test function with no docstring uses default description."""

        def func_no_doc():
            pass

        tool_manager.register(func_no_doc)
        tool = tool_manager.get_tool("func_no_doc")

        assert tool.description == "Call func_no_doc"

    def test_parameter_without_description(self, tool_manager):
        """Test parameter without description in docstring gets default."""

        def func_no_param_desc(param: str):
            """Function description."""
            pass

        tool_manager.register(func_no_param_desc)
        tool = tool_manager.get_tool("func_no_param_desc")

        assert tool.parameters.properties["param"]["description"] == "Parameter: param"

    def test_list_without_type_args(self, tool_manager):
        """Test List without type arguments."""

        def func_raw_list(items: list[Any]):
            """Function with raw list.

            Args:
                items: List of items
            """
            pass

        tool_manager.register(func_raw_list)
        tool = tool_manager.get_tool("func_raw_list")

        param_schema = tool.parameters.properties["items"]
        assert param_schema["type"] == "array"
        assert param_schema["items"]["type"] == "string"  # Should default to string

    def test_dict_without_type_args(self, tool_manager):
        """Test Dict without type arguments."""

        def func_raw_dict(data: dict[str, Any]):
            """Function with raw dict.

            Args:
                data: Dictionary data
            """
            pass

        tool_manager.register(func_raw_dict)
        tool = tool_manager.get_tool("func_raw_dict")

        param_schema = tool.parameters.properties["data"]
        assert param_schema["type"] == "object"
        assert param_schema["additionalProperties"] is True

    def test_numpy_docstring_without_underline(self, tool_manager):
        """Test NumPy docstring detection without proper underline."""
        docstring = """Description.

        Parameters
        param1 : str
            Parameter description
        """

        result = tool_manager._parse_numpy_docstring(docstring)
        # Should not parse as NumPy format without proper underline
        assert result["args"] == {}

    def test_numpy_docstring_short_lines(self, tool_manager):
        """Test NumPy docstring with edge case line handling."""
        docstring = """Description.

        Parameters
        ----------
        """

        result = tool_manager._parse_numpy_docstring(docstring)
        assert result["description"].startswith("Description.")
        assert result["args"] == {}

    def test_parse_numpy_args_section_edge_cases(self, tool_manager):
        """Test NumPy args section with edge cases."""
        args_section = [
            "",  # Empty line
            "param1 : str",
            "",  # Empty line between param and description
            "    Description for param1",
            "param2",  # No type
            "    Description for param2",
        ]

        result = tool_manager._parse_numpy_args_section(args_section)
        assert result["param1"] == "Description for param1"
        assert result["param2"] == "Description for param2"

    def test_nullable_type_with_existing_list_type(self, tool_manager):
        """Test nullable type handling when current type is already a list."""

        original_method = tool_manager._get_basic_type_schema

        def mock_basic_type_schema(param_type, description):
            # Return a schema with type as a list (simulating some complex type handling)
            return {"type": ["string", "integer"], "description": description}

        # Temporarily replace the method
        tool_manager._get_basic_type_schema = mock_basic_type_schema

        try:
            # Now call _create_schema_for_type with nullable=True
            schema = tool_manager._create_schema_for_type(str, "test description", nullable=True)

            # Should have added "null" to the existing list
            assert schema["type"] == ["string", "integer", "null"]

        finally:
            # Restore the original method
            tool_manager._get_basic_type_schema = original_method

    def test_nullable_type_with_existing_list_type_containing_null(self, tool_manager):
        """Test nullable type handling when current type is already a list containing null."""

        original_method = tool_manager._get_basic_type_schema

        def mock_basic_type_schema(param_type, description):
            # Return a schema with type as a list that already contains "null"
            return {"type": ["string", "null"], "description": description}

        # Temporarily replace the method
        tool_manager._get_basic_type_schema = mock_basic_type_schema

        try:
            # Now call _create_schema_for_type with nullable=True
            schema = tool_manager._create_schema_for_type(str, "test description", nullable=True)

            # Should NOT add another "null" to the existing list since it already contains one
            assert schema["type"] == ["string", "null"]
            assert schema["type"].count("null") == 1  # Ensure only one "null" exists

        finally:
            # Restore the original method
            tool_manager._get_basic_type_schema = original_method

    def test_comprehensive_integration(self, tool_manager):
        """Test comprehensive integration with complex function."""

        def complex_function(
            name: str,
            age: int,
            scores: list[float],
            integers: list[int],
            metadata: dict[str, Any],
            is_active: bool = True,
            nickname: str | None = None,
            tags: str | list[str] = "default",
        ):
            """A complex function for testing.

            This function demonstrates various parameter types and optional parameters.

            Args:
                name: The person's full name
                age: The person's age in years
                scores: List of test scores
                integers: List of integers
                metadata: Additional metadata dictionary
                is_active: Whether the person is currently active
                nickname: Optional nickname for the person
                tags: Either a single tag string or list of tags

            Returns:
                str: Formatted person information

            Raises:
                ValueError: If age is negative
            """
            return f"Person: {name}"

        tool_manager.register(complex_function)
        tool = tool_manager.get_tool("complex_function")

        # Verify tool properties
        assert tool.name == "complex_function"
        assert "A complex function for testing." in tool.description
        assert tool.function is complex_function

        # Verify parameters
        params = tool.parameters
        assert params.type == "object"
        assert params.additionalProperties is False
        assert set(params.required) == {"name", "age", "scores", "integers", "metadata"}

        # Verify individual parameter schemas
        props = params.properties
        assert props["name"]["type"] == "string"
        assert props["name"]["description"] == "The person's full name"

        assert props["age"]["type"] == "integer"
        assert props["age"]["description"] == "The person's age in years"

        assert props["scores"]["type"] == "array"
        assert props["scores"]["items"]["type"] == "number"
        assert props["scores"]["description"] == "List of test scores"

        assert props["integers"]["type"] == "array"
        assert props["integers"]["items"]["type"] == "integer"
        assert props["integers"]["description"] == "List of integers"

        assert props["metadata"]["type"] == "object"
        assert props["metadata"]["additionalProperties"] is True

        assert props["is_active"]["type"] == "boolean"
        assert "is_active" not in params.required

        assert props["nickname"]["type"] == ["string", "null"]
        assert "nickname" not in params.required

        # Union type should use first type
        assert props["tags"]["type"] == "string"
