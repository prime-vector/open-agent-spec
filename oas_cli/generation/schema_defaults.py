# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""Shared schema → default literal helpers (parser codegen + JSON examples). Reduces drift between modules."""

from typing import Any


def string_literal_from_schema(
    field_schema: dict[str, Any], field_name: str, suffix: str
) -> str:
    """Quoted string default/example: first word of description + suffix, or field_name + suffix."""
    description = field_schema.get("description", "")
    if description:
        token = description.split()[0].lower() + suffix
    else:
        token = field_name + suffix
    return f'"{token}"'


def parser_default_rhs(field_schema: dict[str, Any], field_name: str) -> str:
    """Python RHS expression as string for parse_with_fallback defaults dict (nested values use JSON-like keys)."""
    field_type = field_schema.get("type", "string")
    if field_type == "string":
        return string_literal_from_schema(field_schema, field_name, "_default")
    if field_type == "boolean":
        return "False"
    if field_type in ("integer", "number"):
        return "0"
    if field_type == "array":
        return "[]"
    if field_type == "object":
        nested_props = field_schema.get("properties", {})
        if not nested_props:
            return "{}"
        pairs = []
        for nested_name, nested_schema in nested_props.items():
            nested_type = nested_schema.get("type", "string")
            # Match prior single_step_codegen: one level only; nested object -> {}
            if nested_type == "object":
                rhs = "{}"
            elif nested_type == "string":
                rhs = f'"default_{nested_name}"'
            elif nested_type == "boolean":
                rhs = "False"
            elif nested_type in ("integer", "number"):
                rhs = "0"
            elif nested_type == "array":
                rhs = "[]"
            else:
                rhs = '""'
            pairs.append(f'"{nested_name}": {rhs}')
        inner = ",\n".join(f"                {p}" for p in pairs)
        return "{\n" + inner + "\n            }"
    return '""'


def json_scalar_value(field_schema: dict[str, Any], field_name: str) -> str:
    """JSON value fragment (no key) for primitives; used when building example JSON lines."""
    field_type = field_schema.get("type", "string")
    if field_type == "string":
        return string_literal_from_schema(field_schema, field_name, "_example")
    if field_type == "integer":
        return "123"
    if field_type == "number":
        return "123.45"
    if field_type == "boolean":
        return "true"
    return f'"{field_name}_value"'


def json_array_primitive_example(field_name: str, item_type: str) -> str:
    """JSON array literal for non-object items."""
    if item_type == "string":
        return f'["{field_name}_item1", "{field_name}_item2"]'
    if item_type == "integer":
        return "[1, 2, 3]"
    if item_type == "number":
        return "[1.1, 2.2, 3.3]"
    if item_type == "boolean":
        return "[true, false]"
    return "[]"
