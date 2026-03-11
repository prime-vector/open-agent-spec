# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""Pydantic model string generation from task output schemas."""

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("oas")


def _generate_pydantic_model(
    name: str, schema: dict[str, Any], is_root: bool = True
) -> str:
    """Generate a Pydantic model from a JSON schema."""
    if not schema.get("properties"):
        return ""

    model_code = []
    nested_models = []

    for field_name, field_schema in schema.get("properties", {}).items():
        if field_schema.get("type") == "object" and field_schema.get("properties"):
            nested_name = f"{name}{field_name.title()}"
            nested_model = _generate_pydantic_model(nested_name, field_schema, False)
            if nested_model:
                nested_models.append(nested_model)
        elif (
            field_schema.get("type") == "array"
            and field_schema.get("items", {}).get("type") == "object"
        ):
            nested_name = f"{name}{field_name.title()}Item"
            nested_model = _generate_pydantic_model(
                nested_name, field_schema["items"], False
            )
            if nested_model:
                nested_models.append(nested_model)

    if is_root:
        model_code.append(f"class {name}(BaseModel):")
    else:
        model_code.append(f"class {name}(BaseModel):")

    for field_name, field_schema in schema.get("properties", {}).items():
        field_type = _get_pydantic_type(field_schema, name, field_name)
        description = field_schema.get("description", "")
        is_required = field_name in schema.get("required", [])
        if not is_required:
            field_type = f"Optional[{field_type}] = None"
        if description:
            model_code.append(f'    """{description}"""')
        model_code.append(f"    {field_name}: {field_type}")

    return "\n".join(nested_models + model_code)


def _get_pydantic_type(
    schema: dict[str, Any], parent_name: str, field_name: str
) -> str:
    """Convert JSON schema type to Pydantic type."""
    schema_type = schema.get("type")

    if schema_type == "string":
        return "str"
    if schema_type == "integer":
        return "int"
    if schema_type == "number":
        return "float"
    if schema_type == "boolean":
        return "bool"
    if schema_type == "array":
        items = schema.get("items", {})
        if items.get("type") == "object":
            return f"List[{parent_name}{field_name.title()}Item]"
        item_type = _get_pydantic_type(items, parent_name, field_name)
        return f"List[{item_type}]"
    if schema_type == "object":
        return f"{parent_name}{field_name.title()}"
    return "Any"


def generate_models(output: Path, spec_data: dict[str, Any]) -> None:
    """Generate models.py file with Pydantic models for task outputs."""
    if (output / "models.py").exists():
        log.warning("models.py already exists and will be overwritten")

    tasks = spec_data.get("tasks", {})
    if not tasks:
        log.warning("No tasks defined in spec file")
        return

    model_code = [
        "from typing import Any, Dict, List, Optional",
        "from pydantic import BaseModel",
        "",
    ]

    for task_name, task_def in tasks.items():
        if "output" in task_def:
            model_name = f"{task_name.replace('-', '_').title()}Output"
            model_code.append(_generate_pydantic_model(model_name, task_def["output"]))
            model_code.append("")

    (output / "models.py").write_text("\n".join(model_code))
    log.info("models.py created")
