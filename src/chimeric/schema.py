"""Structured output schema utilities.

Pure functions for extracting JSON schemas from Pydantic models and building
the provider-specific request body fragments needed to enable constrained
JSON output.  No I/O — safe to call from any layer.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import TYPE_CHECKING, Any

from .types import Provider

if TYPE_CHECKING:
    from pydantic import BaseModel

__all__ = [
    "build_response_format_kwargs",
    "extract_json_schema",
    "parse_json_response",
]

# ---------------------------------------------------------------------------
# Private per-format builders
# ---------------------------------------------------------------------------
# Each function takes (schema_name, schema) and returns the provider-specific
# request body fragment.  Adding a new format means adding one function here
# and one entry in _FORMAT_BUILDERS below — nothing else changes.

_Builder = Callable[[str, dict[str, Any]], dict[str, Any]]


def _enforce_additional_properties(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively add additionalProperties: false to all object sub-schemas.

    OpenAI's strict structured output mode requires this on every object.
    """
    schema = dict(schema)
    if schema.get("type") == "object" or "properties" in schema:
        schema["additionalProperties"] = False
        if "properties" in schema:
            schema["properties"] = {
                k: _enforce_additional_properties(v) for k, v in schema["properties"].items()
            }
    if "$defs" in schema:
        schema["$defs"] = {k: _enforce_additional_properties(v) for k, v in schema["$defs"].items()}
    return schema


def _openai_format(schema_name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": _enforce_additional_properties(schema),
                "strict": True,
            },
        }
    }


def _groq_format(schema_name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Groq best-effort structured output format.

    Groq only supports strict=true on openai/gpt-oss-* models.  All other
    supported models (llama-4-scout, llama-4-maverick, kimi-k2) require
    strict=false.  Note: streaming is not supported with structured outputs
    on Groq regardless of the mode.
    """
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                "strict": False,
            },
        }
    }


def _anthropic_format(_schema_name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_config": {
            "format": {"type": "json_schema", "schema": _enforce_additional_properties(schema)}
        }
    }


def _google_format(_schema_name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {"generationConfig": {"responseMimeType": "application/json", "responseSchema": schema}}


def _cohere_format(_schema_name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {"response_format": {"type": "json_object", "schema": schema}}


# Dispatch table: Provider → format builder.
# OpenAI-compatible providers share _openai_format; only their base_url differs.
_FORMAT_BUILDERS: dict[Provider, _Builder] = {
    Provider.OPENAI: _openai_format,
    Provider.GROQ: _groq_format,
    Provider.CEREBRAS: _openai_format,
    Provider.GROK: _openai_format,
    Provider.OPENROUTER: _openai_format,
    Provider.ANTHROPIC: _anthropic_format,
    Provider.GOOGLE: _google_format,
    Provider.COHERE: _cohere_format,
    Provider.CUSTOM: _openai_format,
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_json_schema(model: type[BaseModel]) -> tuple[str, dict[str, Any]]:
    """Extract the name and JSON schema from a Pydantic model class.

    Args:
        model: A Pydantic ``BaseModel`` subclass.

    Returns:
        A ``(name, schema)`` tuple.  *name* is the model's ``title`` (or the
        class name if no title is set).  *schema* is the raw JSON Schema dict
        with the top-level ``title`` key removed — providers set their own
        display name via the enclosing request field.
    """
    schema = model.model_json_schema()
    name: str = schema.pop("title", model.__name__)
    return name, schema


def parse_json_response(content: str, model: type[BaseModel]) -> BaseModel:
    """Deserialise a JSON string into a Pydantic model instance.

    Args:
        content: Raw JSON string from the provider response.
        model: Target Pydantic ``BaseModel`` subclass.

    Returns:
        A validated model instance.

    Raises:
        json.JSONDecodeError: If *content* is not valid JSON.
        pydantic.ValidationError: If the parsed JSON doesn't match the schema.
    """
    data = json.loads(content)
    return model.model_validate(data)


def build_response_format_kwargs(
    provider: Provider,
    schema_name: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Build the provider-specific request body kwargs for structured output.

    Args:
        provider: The target ``Provider`` enum member.
        schema_name: Display name for the schema (derived from the Pydantic
            model class name).
        schema: JSON Schema dict describing the expected output structure.

    Returns:
        A ``dict`` of kwargs to merge into the HTTP request body.  The format
        kwargs always take precedence over any user-supplied kwargs that share
        the same top-level keys.

    Raises:
        ValueError: If *provider* has no registered format builder.
    """
    builder = _FORMAT_BUILDERS.get(provider)
    if builder is None:
        raise ValueError(
            f"No structured output format registered for provider '{provider.value}'. "
            f"Registered providers: {', '.join(p.value for p in _FORMAT_BUILDERS)}."
        )
    return builder(schema_name, schema)
