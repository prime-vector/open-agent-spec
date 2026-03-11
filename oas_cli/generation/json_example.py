# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""Build JSON example strings from output schemas (used for prompt templates)."""

from typing import Any

from .schema_defaults import json_array_primitive_example, json_scalar_value


def generate_json_example_lines(
    field_name: str,
    field_schema: dict[str, Any],
    indent: int = 0,
    comma: str = "",
) -> list[str]:
    """Generate lines for a structured JSON example from a JSON-schema-like field."""
    lines: list[str] = []
    field_type = field_schema.get("type", "string")
    field_line = f'{" " * indent}"{field_name}": '

    if field_type in ("string", "integer", "number", "boolean"):
        lines.append(field_line + json_scalar_value(field_schema, field_name) + comma)
    elif field_type == "array":
        items = field_schema.get("items", {})
        if items.get("type") == "object":
            lines.append(field_line + "[")
            lines.append(f"{' ' * (indent + 2)}{{")
            nested_props = items.get("properties", {})
            for j, (nested_name, nested_schema) in enumerate(nested_props.items()):
                nested_comma = "," if j < len(nested_props) - 1 else ""
                nested_lines = generate_json_example_lines(
                    nested_name, nested_schema, indent + 4, nested_comma
                )
                lines.extend(nested_lines)
            lines.append(f"{' ' * (indent + 2)}}}")
            lines.append(f"{' ' * indent}]" + comma)
        else:
            item_type = items.get("type", "string")
            lines.append(
                field_line + json_array_primitive_example(field_name, item_type) + comma
            )
    elif field_type == "object":
        lines.append(field_line + "{")
        nested_props = field_schema.get("properties", {})
        for j, (nested_name, nested_schema) in enumerate(nested_props.items()):
            nested_comma = "," if j < len(nested_props) - 1 else ""
            nested_lines = generate_json_example_lines(
                nested_name, nested_schema, indent + 2, nested_comma
            )
            lines.extend(nested_lines)
        lines.append(f"{' ' * indent}}}" + comma)
    else:
        lines.append(field_line + json_scalar_value(field_schema, field_name) + comma)
    return lines
