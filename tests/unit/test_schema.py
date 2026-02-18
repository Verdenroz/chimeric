"""Tests for the schema utility module."""

import json
from typing import ClassVar

from pydantic import BaseModel, ValidationError
import pytest

from chimeric.schema import build_response_format_kwargs, extract_json_schema, parse_json_response
from chimeric.types import Provider

# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class Simple(BaseModel):
    name: str
    age: int


class Nested(BaseModel):
    class Address(BaseModel):
        street: str
        city: str

    full_name: str
    address: Address


class WithTitle(BaseModel):
    model_config = {"title": "CustomTitle"}
    value: str


# ---------------------------------------------------------------------------
# extract_json_schema
# ---------------------------------------------------------------------------


class TestExtractJsonSchema:
    def test_returns_class_name_when_no_title(self):
        name, _ = extract_json_schema(Simple)
        assert name == "Simple"

    def test_title_stripped_from_schema(self):
        _, schema = extract_json_schema(Simple)
        assert "title" not in schema

    def test_schema_contains_properties(self):
        _, schema = extract_json_schema(Simple)
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]

    def test_custom_title_used_as_name(self):
        name, schema = extract_json_schema(WithTitle)
        assert name == "CustomTitle"
        assert "title" not in schema

    def test_nested_model_schema_preserved(self):
        _, schema = extract_json_schema(Nested)
        assert "properties" in schema
        assert "full_name" in schema["properties"]
        assert "address" in schema["properties"]

    def test_original_model_unaffected(self):
        extract_json_schema(Simple)
        # Calling twice should work (schema is re-generated each time)
        name, schema = extract_json_schema(Simple)
        assert name == "Simple"
        assert "properties" in schema


# ---------------------------------------------------------------------------
# parse_json_response
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    def test_valid_json_returns_model_instance(self):
        result = parse_json_response('{"name": "Alice", "age": 30}', Simple)
        assert isinstance(result, Simple)
        assert result.name == "Alice"
        assert result.age == 30

    def test_invalid_json_raises_decode_error(self):
        with pytest.raises(json.JSONDecodeError):
            parse_json_response("not valid json", Simple)

    def test_schema_mismatch_raises_validation_error(self):
        with pytest.raises(ValidationError):
            parse_json_response('{"name": "Alice", "age": "not-an-int"}', Simple)

    def test_missing_required_field_raises_validation_error(self):
        with pytest.raises(ValidationError):
            parse_json_response('{"name": "Alice"}', Simple)

    def test_empty_string_raises_decode_error(self):
        with pytest.raises(json.JSONDecodeError):
            parse_json_response("", Simple)


# ---------------------------------------------------------------------------
# build_response_format_kwargs
# ---------------------------------------------------------------------------


class TestBuildResponseFormatKwargs:
    SCHEMA: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
    }
    # _enforce_additional_properties adds additionalProperties: False on every object
    ENRICHED_SCHEMA: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "additionalProperties": False,
    }

    def test_openai_format(self):
        result = build_response_format_kwargs(Provider.OPENAI, "MyModel", self.SCHEMA)
        assert result == {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "MyModel", "schema": self.ENRICHED_SCHEMA, "strict": True},
            }
        }

    def test_groq_uses_openai_format(self):
        result = build_response_format_kwargs(Provider.GROQ, "MyModel", self.SCHEMA)
        assert "response_format" in result
        assert result["response_format"]["type"] == "json_schema"

    def test_cerebras_uses_openai_format(self):
        result = build_response_format_kwargs(Provider.CEREBRAS, "MyModel", self.SCHEMA)
        assert "response_format" in result

    def test_grok_uses_openai_format(self):
        result = build_response_format_kwargs(Provider.GROK, "MyModel", self.SCHEMA)
        assert "response_format" in result

    def test_openrouter_uses_openai_format(self):
        result = build_response_format_kwargs(Provider.OPENROUTER, "MyModel", self.SCHEMA)
        assert "response_format" in result

    def test_anthropic_format(self):
        result = build_response_format_kwargs(Provider.ANTHROPIC, "MyModel", self.SCHEMA)
        assert result == {
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": self.ENRICHED_SCHEMA,
                }
            }
        }

    def test_google_format(self):
        result = build_response_format_kwargs(Provider.GOOGLE, "MyModel", self.SCHEMA)
        assert result == {
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": self.SCHEMA,
            }
        }

    def test_cohere_format(self):
        result = build_response_format_kwargs(Provider.COHERE, "MyModel", self.SCHEMA)
        assert result == {
            "response_format": {
                "type": "json_object",
                "schema": self.SCHEMA,
            }
        }
