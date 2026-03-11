# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""LLM output parser and human-readable output schema helpers for single-step tasks."""

from typing import Any

from .schema_defaults import parser_default_rhs


def _generate_llm_output_parser(task_name: str, output_schema: dict[str, Any]) -> str:
    """Generate a function for parsing LLM output using DACP's parse_with_fallback."""
    model_name = f"{task_name.replace('-', '_').title()}Output"
    parser_name = f"parse_{task_name.replace('-', '_')}_output"

    properties = output_schema.get("properties", {})
    default_values = []
    conflicting_params = {"response", "model_class"}

    for field_name, field_schema in properties.items():
        if field_name in conflicting_params:
            continue
        default_value = parser_default_rhs(field_schema, field_name)
        default_values.append(f'            "{field_name}": {default_value}')

    if default_values:
        defaults_dict = "{\n" + ",\n".join(default_values) + "\n        }"
    else:
        defaults_dict = "{}"

    return f"""def {parser_name}(response) -> {model_name}:
    \"\"\"Parse LLM response into {model_name} using DACP's enhanced parser.

    Args:
        response: Raw response from the LLM (str or dict)

    Returns:
        Parsed and validated {model_name} instance

    Raises:
        ValueError: If the response cannot be parsed
    \"\"\"
    if isinstance(response, {model_name}):
        return response

    # Parse JSON string if needed
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f'Failed to parse JSON response: {{e}}')

    # Use DACP's enhanced JSON parser with fallback support
    try:
        defaults = {defaults_dict}
        result = parse_with_fallback(
            response=response,
            model_class={model_name},
            **defaults
        )
        return result
    except Exception as e:
        raise ValueError(f'Error parsing response with DACP parser: {{e}}')
"""


def _generate_human_readable_output(schema: dict[str, Any], indent: int = 0) -> str:
    """Generate a human-readable description of the output schema.

    Args:
        schema: The JSON schema to convert
        indent: Current indentation level

    Returns:
        String containing a human-readable description of the output format
    """
    if not schema.get("properties"):
        return ""

    lines = []
    for field_name, field_schema in schema.get("properties", {}).items():
        field_type = _get_human_readable_type(field_schema)
        description = field_schema.get("description", "")

        is_required = field_name in schema.get("required", [])
        required_str = " (required)" if is_required else " (optional)"

        if description:
            lines.append(f"{' ' * indent}- {field_name}{required_str}: {field_type}")
            lines.append(f"{' ' * (indent + 2)}{description}")
        else:
            lines.append(f"{' ' * indent}- {field_name}{required_str}: {field_type}")

        if field_schema.get("type") == "object" and field_schema.get("properties"):
            nested_desc = _generate_human_readable_output(field_schema, indent + 2)
            if nested_desc:
                lines.append(nested_desc)

        elif (
            field_schema.get("type") == "array"
            and field_schema.get("items", {}).get("type") == "object"
        ):
            lines.append(f"{' ' * (indent + 2)}Each item contains:")
            nested_desc = _generate_human_readable_output(
                field_schema["items"], indent + 4
            )
            if nested_desc:
                lines.append(nested_desc)

    return "\n".join(lines)


def _get_human_readable_type(schema: dict[str, Any]) -> str:
    """Convert JSON schema type to human-readable type."""
    schema_type = schema.get("type")

    if schema_type == "string":
        return "string"
    elif schema_type == "integer":
        return "integer"
    elif schema_type == "number":
        return "number"
    elif schema_type == "boolean":
        return "boolean"
    elif schema_type == "array":
        items = schema.get("items", {})
        if items.get("type") == "object":
            return "array of objects"
        else:
            item_type = _get_human_readable_type(items)
            return f"array of {item_type}s"
    elif schema_type == "object":
        return "object"
    else:
        return "any"
