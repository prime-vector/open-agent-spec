# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under AGPL-3.0 with Additional Terms
# See LICENSE for details on attribution, naming, and branding restrictions.

"""File generation functions for Open Agent Spec."""

import logging
from pathlib import Path
from typing import Any, Dict, List


log = logging.getLogger("oas")


def get_agent_info(spec_data: Dict[str, Any]) -> Dict[str, str]:
    """Get agent info from either old or new spec format."""
    # Try new format first
    agent = spec_data.get("agent", {})
    if agent:
        return {
            "name": agent.get("name", ""),
            "description": agent.get("description", ""),
        }

    # Fall back to old format
    info = spec_data.get("info", {})
    return {"name": info.get("name", ""), "description": info.get("description", "")}


def get_memory_config(spec_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get memory configuration from spec."""
    memory = spec_data.get("memory", {})
    return {
        "enabled": memory.get("enabled", False),
        "format": memory.get("format", "string"),
        "usage": memory.get("usage", "prompt-append"),
        "required": memory.get("required", False),
        "description": memory.get("description", ""),
    }


def get_logging_config(spec_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get logging configuration from spec."""
    logging_config = spec_data.get("logging", {})
    return {
        "enabled": logging_config.get("enabled", True),
        "level": logging_config.get("level", "INFO"),
        "format_style": logging_config.get("format_style", "emoji"),
        "include_timestamp": logging_config.get("include_timestamp", True),
        "log_file": logging_config.get("log_file"),
        "env_overrides": logging_config.get("env_overrides", {}),
    }


def to_pascal_case(name: str) -> str:
    """Convert snake_case or kebab-case to PascalCase."""
    # Handle both underscores and hyphens
    name = name.replace("-", "_")
    return "".join(word.capitalize() for word in name.split("_"))


def _generate_input_params(task_def: Dict[str, Any]) -> List[str]:
    """Generate input parameters for a task function."""
    input_params = []

    # Check if this is a multi-step task
    is_multi_step = task_def.get("multi_step", False)

    if is_multi_step:
        # For multi-step tasks, infer input parameters from step input mappings
        steps = task_def.get("steps", [])
        inferred_params = set()

        for step in steps:
            input_map = step.get("input_map", {})
            for param, value in input_map.items():
                # Handle Jinja2-style templating {{variable}}
                if isinstance(value, str) and "{{" in value and "}}" in value:
                    # Extract variable name from {{variable}}
                    var_name = value.replace("{{", "").replace("}}", "").strip()
                    # Handle nested references like input.name -> extract just 'name'
                    if "." in var_name:
                        parts = var_name.split(".")
                        if parts[0] == "input":
                            # This is an input parameter
                            var_name = parts[-1]
                            inferred_params.add(var_name)
                        # Skip steps.* references as they are previous step results, not input parameters
                    else:
                        # Simple variable name without dots
                        inferred_params.add(var_name)

        # Add inferred parameters
        for param_name in sorted(inferred_params):
            input_params.append(f"{param_name}: str")
    else:
        # For regular tasks, use the input schema
        for param_name, param_def in (
            task_def.get("input", {}).get("properties", {}).items()
        ):
            param_type = map_type_to_python(param_def.get("type", "string"))
            input_params.append(f"{param_name}: {param_type}")

    input_params.append("memory_summary: str = ''")
    return input_params


def _generate_function_docstring(
    task_name: str, task_def: Dict[str, Any], output_type: str
) -> str:
    """Generate docstring for a task function."""
    return f'''"""Process {task_name} task.

    Args:
{chr(10).join(f"        {param_name}: {param_type}" for param_name, param_type in task_def.get("input", {}).get("properties", {}).items())}
        memory_summary: Optional memory context for the task

    Returns:
        {output_type}
    """'''


def _generate_contract_data(
    spec_data: Dict[str, Any],
    task_def: Dict[str, Any],
    agent_name: str,
    memory_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate behavioural contract data from spec."""
    behavioural_section = spec_data.get("behavioural_contract", {})

    # Use the task's description for the contract
    contract_data = {
        "version": behavioural_section.get("version", "0.1.2"),
        "description": task_def.get(
            "description", behavioural_section.get("description", "")
        ),
    }

    # Add role from agent section (not from behavioural_contract)
    agent_role = spec_data.get("agent", {}).get("role")
    if agent_role:
        contract_data["role"] = agent_role

    # Only add behavioural_flags if specified
    if behavioural_section.get("behavioural_flags"):
        contract_data["behavioural_flags"] = behavioural_section["behavioural_flags"]

    # Add function-specific response_contract based on the task's output schema
    output_schema = task_def.get("output", {})
    required_fields = output_schema.get("required", [])
    if required_fields:
        contract_data["response_contract"] = {
            "output_format": {"required_fields": required_fields}
        }

    return contract_data


def _generate_pydantic_model(
    name: str, schema: Dict[str, Any], is_root: bool = True
) -> str:
    """Generate a Pydantic model from a JSON schema.

    Args:
        name: The name of the model
        schema: The JSON schema to convert
        is_root: Whether this is the root model (affects class inheritance)

    Returns:
        String containing the generated Pydantic model code
    """
    if not schema.get("properties"):
        return ""

    model_code = []
    nested_models = []

    # First, generate nested models
    for field_name, field_schema in schema.get("properties", {}).items():
        # Handle nested objects
        if field_schema.get("type") == "object" and field_schema.get("properties"):
            nested_name = f"{name}{field_name.title()}"
            nested_model = _generate_pydantic_model(nested_name, field_schema, False)
            if nested_model:
                nested_models.append(nested_model)

        # Handle arrays of objects
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

    # Then generate the main model
    if is_root:
        model_code.append(f"class {name}(BaseModel):")
    else:
        model_code.append(
            f"class {name}(BaseModel):"
        )  # Always use BaseModel for nested models

    # Add field definitions
    for field_name, field_schema in schema.get("properties", {}).items():
        field_type = _get_pydantic_type(field_schema, name, field_name)
        description = field_schema.get("description", "")

        # Handle required fields
        is_required = field_name in schema.get("required", [])
        if not is_required:
            field_type = f"Optional[{field_type}] = None"

        # Add field with description
        if description:
            model_code.append(f'    """{description}"""')
        model_code.append(f"    {field_name}: {field_type}")

    # Combine nested models and main model
    return "\n".join(nested_models + model_code)


def _get_pydantic_type(
    schema: Dict[str, Any], parent_name: str, field_name: str
) -> str:
    """Convert JSON schema type to Pydantic type."""
    schema_type = schema.get("type")

    if schema_type == "string":
        return "str"
    elif schema_type == "integer":
        return "int"
    elif schema_type == "number":
        return "float"
    elif schema_type == "boolean":
        return "bool"
    elif schema_type == "array":
        items = schema.get("items", {})
        if items.get("type") == "object":
            # For array of objects, use the nested model type
            return f"List[{parent_name}{field_name.title()}Item]"
        else:
            item_type = _get_pydantic_type(items, parent_name, field_name)
            return f"List[{item_type}]"
    elif schema_type == "object":
        # For nested objects, use the nested model type
        return f"{parent_name}{field_name.title()}"
    else:
        return "Any"


def generate_models(output: Path, spec_data: Dict[str, Any]) -> None:
    """Generate models.py file with Pydantic models for task outputs."""
    if (output / "models.py").exists():
        log.warning("models.py already exists and will be overwritten")

    tasks = spec_data.get("tasks", {})
    if not tasks:
        log.warning("No tasks defined in spec file")
        return

    # Generate imports
    model_code = [
        "from typing import Any, Dict, List, Optional",
        "from pydantic import BaseModel",
        "",
    ]

    # Generate models for each task
    for task_name, task_def in tasks.items():
        if "output" in task_def:
            model_name = f"{task_name.replace('-', '_').title()}Output"
            model_code.append(_generate_pydantic_model(model_name, task_def["output"]))
            model_code.append("")  # Add blank line between models

    # Write the file
    (output / "models.py").write_text("\n".join(model_code))
    log.info("models.py created")


def _generate_llm_output_parser(task_name: str, output_schema: Dict[str, Any]) -> str:
    """Generate a function for parsing LLM output using DACP's parse_with_fallback."""
    model_name = f"{task_name.replace('-', '_').title()}Output"
    parser_name = f"parse_{task_name.replace('-', '_')}_output"

    # Generate default values for all required fields from the schema
    properties = output_schema.get("properties", {})
    default_values = []

    # Parameters that conflict with parse_with_fallback function parameters
    conflicting_params = {"response", "model_class"}

    for field_name, field_schema in properties.items():
        # Skip fields that would conflict with function parameters
        if field_name in conflicting_params:
            continue

        field_type = field_schema.get("type", "string")
        if field_type == "string":
            # Use description or field name for meaningful defaults
            description = field_schema.get("description", "")
            if description:
                default_value = f'"{description.split()[0].lower()}_default"'
            else:
                default_value = f'"{field_name}_default"'
        elif field_type == "boolean":
            default_value = "False"
        elif field_type in ["integer", "number"]:
            default_value = "0"
        elif field_type == "array":
            default_value = "[]"
        elif field_type == "object":
            # Generate structured defaults for objects
            nested_props = field_schema.get("properties", {})
            if nested_props:
                nested_defaults = []
                for nested_name, nested_schema in nested_props.items():
                    nested_type = nested_schema.get("type", "string")
                    if nested_type == "string":
                        nested_defaults.append(
                            f'"{nested_name}": "default_{nested_name}"'
                        )
                    elif nested_type == "boolean":
                        nested_defaults.append(f'"{nested_name}": False')
                    elif nested_type in ["integer", "number"]:
                        nested_defaults.append(f'"{nested_name}": 0')
                    elif nested_type == "array":
                        nested_defaults.append(f'"{nested_name}": []')
                    elif nested_type == "object":
                        nested_defaults.append(f'"{nested_name}": {{}}')
                    else:
                        nested_defaults.append(f'"{nested_name}": ""')
                default_value = (
                    "{\n"
                    + ",\n".join(f"                {d}" for d in nested_defaults)
                    + "\n            }"
                )
            else:
                default_value = "{}"
        else:
            default_value = '""'

        default_values.append(f'            "{field_name}": {default_value}')

    # Build defaults as a properly formatted dictionary
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


def _generate_human_readable_output(schema: Dict[str, Any], indent: int = 0) -> str:
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

        # Handle required fields
        is_required = field_name in schema.get("required", [])
        required_str = " (required)" if is_required else " (optional)"

        # Add field with description
        if description:
            lines.append(f"{' ' * indent}- {field_name}{required_str}: {field_type}")
            lines.append(f"{' ' * (indent + 2)}{description}")
        else:
            lines.append(f"{' ' * indent}- {field_name}{required_str}: {field_type}")

        # Handle nested objects
        if field_schema.get("type") == "object" and field_schema.get("properties"):
            nested_desc = _generate_human_readable_output(field_schema, indent + 2)
            if nested_desc:
                lines.append(nested_desc)

        # Handle arrays of objects
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


def _get_human_readable_type(schema: Dict[str, Any]) -> str:
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


def _generate_json_example(
    field_name: str, field_schema: Dict[str, Any], indent: int = 0, comma: str = ""
) -> List[str]:
    """Generate a structured JSON example for a field based on its schema.

    Args:
        field_name: Name of the field
        field_schema: JSON schema for the field
        indent: Indentation level
        comma: Comma to append (for formatting)

    Returns:
        List of lines for the JSON example
    """
    lines = []
    field_type = field_schema.get("type", "string")
    description = field_schema.get("description", "")

    # Add field name with proper indentation
    field_line = f'{" " * indent}"{field_name}": '

    if field_type == "string":
        # Use description or field name for meaningful examples
        if description:
            example_value = f'"{description.split()[0].lower()}_example"'
        else:
            example_value = f'"{field_name}_example"'
        lines.append(field_line + example_value + comma)

    elif field_type == "integer":
        lines.append(field_line + "123" + comma)

    elif field_type == "number":
        lines.append(field_line + "123.45" + comma)

    elif field_type == "boolean":
        lines.append(field_line + "true" + comma)

    elif field_type == "array":
        items = field_schema.get("items", {})
        if items.get("type") == "object":
            # Array of objects
            lines.append(field_line + "[")
            lines.append(f'{" " * (indent + 2)}{{')
            nested_props = items.get("properties", {})
            for j, (nested_name, nested_schema) in enumerate(nested_props.items()):
                nested_comma = "," if j < len(nested_props) - 1 else ""
                nested_lines = _generate_json_example(
                    nested_name, nested_schema, indent + 4, nested_comma
                )
                lines.extend(nested_lines)
            lines.append(f'{" " * (indent + 2)}}}')
            lines.append(f'{" " * indent}]' + comma)
        else:
            # Array of primitives
            item_type = items.get("type", "string")
            if item_type == "string":
                lines.append(
                    field_line + f'["{field_name}_item1", "{field_name}_item2"]' + comma
                )
            elif item_type == "integer":
                lines.append(field_line + "[1, 2, 3]" + comma)
            elif item_type == "number":
                lines.append(field_line + "[1.1, 2.2, 3.3]" + comma)
            elif item_type == "boolean":
                lines.append(field_line + "[true, false]" + comma)
            else:
                lines.append(field_line + "[]" + comma)

    elif field_type == "object":
        # Object with nested properties
        lines.append(field_line + "{")
        nested_props = field_schema.get("properties", {})
        for j, (nested_name, nested_schema) in enumerate(nested_props.items()):
            nested_comma = "," if j < len(nested_props) - 1 else ""
            nested_lines = _generate_json_example(
                nested_name, nested_schema, indent + 2, nested_comma
            )
            lines.extend(nested_lines)
        lines.append(f'{" " * indent}}}' + comma)

    else:
        # Fallback
        lines.append(field_line + f'"{field_name}_value"' + comma)

    return lines


def _generate_multi_step_task_function(
    task_name: str,
    task_def: Dict[str, Any],
    spec_data: Dict[str, Any],
    agent_name: str,
    memory_config: Dict[str, Any],
) -> str:
    """Generate a multi-step task function that orchestrates other tasks."""
    func_name = task_name.replace("-", "_")
    input_params = _generate_input_params(task_def)
    output_type = f"{task_name.replace('-', '_').title()}Output"
    docstring = _generate_function_docstring(task_name, task_def, output_type)
    contract_data = _generate_contract_data(
        spec_data, task_def, agent_name, memory_config
    )

    # Get the steps from the task definition
    steps = task_def.get("steps", [])

    # Generate step execution code
    step_code = []
    step_results: List[str] = []

    for i, step in enumerate(steps):
        step_task = step["task"]
        input_map = step.get("input_map", {})

        # Convert input mapping to Python code
        step_inputs = []
        for param, value in input_map.items():
            # Handle Jinja2-style templating {{variable}}
            if isinstance(value, str) and "{{" in value and "}}" in value:
                # Extract variable name from {{variable}}
                var_name = value.replace("{{", "").replace("}}", "").strip()
                # Handle nested references like input.name -> extract just 'name'
                if "." in var_name:
                    parts = var_name.split(".")
                    if parts[0] == "input":
                        # This is an input parameter
                        var_name = parts[-1]
                        step_inputs.append(f"{param}={var_name}")
                    elif parts[0] == "steps" and len(parts) >= 3:
                        # This is a reference to a previous step result
                        step_index = int(parts[1])
                        field_name = parts[2]
                        if step_index < i:  # Only reference previous steps
                            step_var = step_results[step_index]
                            step_inputs.append(
                                f"{param}={step_var}.{field_name} if hasattr({step_var}, '{field_name}') else {step_var}.get('{field_name}', '')"
                            )
                        else:
                            # Invalid reference to future step
                            step_inputs.append(f'{param}=""')
                else:
                    # Simple variable name without dots
                    step_inputs.append(f"{param}={var_name}")
            else:
                # Literal value
                step_inputs.append(f'{param}="{value}"')

        step_input_str = ", ".join(step_inputs)
        step_var = f"step_{i}_result"
        step_results.append(step_var)

        step_code.append(
            f"""    # Execute step {i + 1}: {step_task}
    {step_var} = {step_task.replace("-", "_")}({step_input_str})"""
        )

    # Generate output construction with better mapping
    output_schema = task_def.get("output", {})
    output_properties = output_schema.get("properties", {})

    output_construction = []

    # Special handling for save_greeting task
    if task_name == "save_greeting" and len(step_results) >= 2:
        # step_0_result is from greet task (has greeting)
        # step_1_result is from write_file task (has success, file_path)
        output_construction = [
            f"        success={step_results[1]}.success if hasattr({step_results[1]}, 'success') else {step_results[1]}.get('success', False)",
            f"        file_path={step_results[1]}.file_path if hasattr({step_results[1]}, 'file_path') else {step_results[1]}.get('file_path', '')",
            f"        greeting={step_results[0]}.greeting if hasattr({step_results[0]}, 'greeting') else {step_results[0]}.get('greeting', '')",
        ]
    else:
        # Generic mapping for other multi-step tasks
        for i, prop_name in enumerate(output_properties):
            if i < len(step_results):
                step_result = step_results[i]
                output_construction.append(
                    f"        {prop_name}={step_result}.{prop_name} if hasattr({step_result}, '{prop_name}') else {step_result}.get('{prop_name}', '')"
                )

    output_construction_str = ",\n".join(output_construction)

    # Format the contract data for the decorator
    def format_value(v):
        if isinstance(v, bool):
            return str(v)
        elif isinstance(v, (list, tuple)):
            return f"[{', '.join(format_value(x) for x in v)}]"
        elif isinstance(v, dict):
            items = [f'"{k}": {format_value(v)}' for k, v in v.items()]
            return f"{{{', '.join(items)}}}"
        elif isinstance(v, str):
            return f'"{v}"'
        return str(v)

    contract_str = ",\n    ".join(
        f"{k}={format_value(v)}" for k, v in contract_data.items()
    )

    return f"""
@behavioural_contract(
    {contract_str}
)
def {func_name}({", ".join(input_params)}) -> {output_type}:
    {docstring}
    # Execute multi-step task: {task_name}
{chr(10).join(step_code)}

    # Construct output from step results
    return {output_type}(
{output_construction_str}
    )
"""


# Legacy functions (deprecated - use template-based generation instead)
def _generate_intelligence_config(
    spec_data: Dict[str, Any], config: Dict[str, Any]
) -> str:
    """Generate intelligence configuration for DACP invoke_intelligence.

    Deprecated: Use PythonCodeSerializer.dict_to_python_code() instead.
    """
    from .code_generation import PythonCodeSerializer

    intelligence = spec_data.get("intelligence", {})
    intelligence_config = {
        "engine": intelligence.get("engine", "openai"),
        "model": intelligence.get("model", config.get("model", "gpt-4")),
        "endpoint": intelligence.get(
            "endpoint", config.get("endpoint", "https://api.openai.com/v1")
        ),
    }

    # Add additional config if present
    intelligence_cfg = intelligence.get("config", {})
    if intelligence_cfg:
        intelligence_config.update(intelligence_cfg)

    # Use proper serialization
    serializer = PythonCodeSerializer()
    return serializer.dict_to_python_code(intelligence_config)


def _generate_embedded_config(spec_data: Dict[str, Any]) -> str:
    """Generate embedded YAML configuration as Python dict.

    Deprecated: Use AgentDataPreparator._prepare_embedded_config() instead.
    """
    from .data_preparation import AgentDataPreparator

    preparator = AgentDataPreparator()
    return preparator._prepare_embedded_config(spec_data)


def _generate_setup_logging_method() -> str:
    """Generate setup_logging method for DACP logging integration.

    Deprecated: Use AgentDataPreparator._prepare_setup_logging_method() instead.
    """
    from .data_preparation import AgentDataPreparator

    preparator = AgentDataPreparator()
    return preparator._prepare_setup_logging_method()


def _generate_tool_task_function(
    task_name: str,
    task_def: Dict[str, Any],
    spec_data: Dict[str, Any],
    agent_name: str,
    memory_config: Dict[str, Any],
    config: Dict[str, Any],
) -> str:
    """Generate a task function that uses a DACP tool."""
    func_name = task_name.replace("-", "_")
    input_params = _generate_input_params(task_def)
    output_type = f"{task_name.replace('-', '_').title()}Output"
    docstring = _generate_function_docstring(task_name, task_def, output_type)
    contract_data = _generate_contract_data(
        spec_data, task_def, agent_name, memory_config
    )

    # Get tool information
    tool_id = task_def["tool"]
    tool_params = task_def.get("tool_params", {})

    # Map tool parameters to DACP parameter names
    tool_param_mapping = {}
    tool_args_lines = []

    if tool_params:
        # Use explicit tool_params mapping if provided
        for param_name, param_info in tool_params.items():
            if isinstance(param_info, dict):
                # Parameter mapping specified
                dacp_param = param_info.get("dacp_param", param_name)
                tool_param_mapping[param_name] = dacp_param
                tool_args_lines.append(f'        "{dacp_param}": {param_name}')
            else:
                # Direct mapping
                tool_param_mapping[param_name] = param_name
                tool_args_lines.append(f'        "{param_name}": {param_name}')
    else:
        # Auto-map input parameters to tool parameters based on common patterns
        input_props = task_def.get("input", {}).get("properties", {})
        for param_name in input_props.keys():
            if tool_id == "file_writer":
                # Map file_writer specific parameters
                if param_name == "file_path":
                    tool_param_mapping[param_name] = "path"
                    tool_args_lines.append(f'        "path": {param_name}')
                elif param_name == "content":
                    tool_param_mapping[param_name] = "content"
                    tool_args_lines.append(f'        "content": {param_name}')
                else:
                    # Default mapping
                    tool_param_mapping[param_name] = param_name
                    tool_args_lines.append(f'        "{param_name}": {param_name}')
            else:
                # Default mapping for other tools
                tool_param_mapping[param_name] = param_name
                tool_args_lines.append(f'        "{param_name}": {param_name}')

    # Create tool description
    tool_description = f"Tool: {tool_id}"
    if tool_params:
        param_descriptions = []
        for param_name, param_info in tool_params.items():
            if isinstance(param_info, dict):
                desc = param_info.get("description", param_name)
                param_descriptions.append(f"- {param_name}: {desc}")
            else:
                param_descriptions.append(f"- {param_name}")
        tool_description += "\nParameters:\n" + "\n".join(param_descriptions)

    tool_description_with_params = tool_description

    # Format the contract data for the decorator
    def format_value(v):
        if isinstance(v, bool):
            return str(v)
        elif isinstance(v, (list, tuple)):
            return f"[{', '.join(format_value(x) for x in v)}]"
        elif isinstance(v, dict):
            items = [f'"{k}": {format_value(v)}' for k, v in v.items()]
            return f"{{{', '.join(items)}}}"
        elif isinstance(v, str):
            return f'"{v}"'
        return str(v)

    contract_str = ",\n    ".join(
        f"{k}={format_value(v)}" for k, v in contract_data.items()
    )

    return f"""
from dacp import invoke_intelligence, execute_tool
from dacp.protocol import parse_agent_response, is_tool_request, get_tool_request, wrap_tool_result, get_final_response, is_final_response

@behavioural_contract(
    {contract_str}
)
def {func_name}({", ".join(input_params)}) -> {output_type}:
    {docstring}
    # Prepare tool arguments
    tool_args = {{
{", ".join(tool_args_lines)}
    }}

    # Create prompt with tool description
    json_example1 = '{{"tool_request": {{"name": "{tool_id}", "args": {{"path": "file_path_here", "content": "content_here"}}}}}}'
    json_example2 = '{{"final_response": {{"result": "your final result here"}}}}'

    tool_prompt = f'''You have access to the following tool:

{tool_description_with_params}

Your task is to use this tool appropriately. You can use the tool by responding with a tool request, or provide a final response.

Available parameters: {list(tool_param_mapping.values())}
Current input values: {{tool_args}}

Respond with JSON in one of these formats:

1. For tool requests:
{{json_example1}}

2. For final responses:
{{json_example2}}

Remember: Only use the tool if it's necessary for your task.'''

    # Configure intelligence for DACP
    intelligence_config = {_generate_intelligence_config(spec_data, config)}

    # Call the LLM with tool context
    response = invoke_intelligence(tool_prompt, intelligence_config)

    # Parse the response
    parsed_response = parse_agent_response(response)

    # Check if LLM wants to use a tool
    if is_tool_request(parsed_response):
        tool_name, tool_params = get_tool_request(parsed_response)

        # Execute the tool
        tool_result = execute_tool(tool_name, tool_params)

        # Wrap the tool result for the LLM
        wrapped_result = wrap_tool_result(tool_name, tool_result)

        # Continue conversation with tool result
        follow_up_prompt = f'''The tool execution result: {{wrapped_result}}

Based on this result, provide your final response in JSON format:

{{{{"final_response": {{{{"result": "your final result here"}}}}}}}}

Remember to respond with valid JSON.'''

        final_response = invoke_intelligence(follow_up_prompt, intelligence_config)
        final_parsed = parse_agent_response(final_response)

        if is_final_response(final_parsed):
            result = get_final_response(final_parsed)
        else:
            result = {{"error": "LLM did not provide final response after tool execution"}}
    else:
        # LLM provided final response directly
        if is_final_response(parsed_response):
            result = get_final_response(parsed_response)
        else:
            result = {{"error": "LLM response format not recognized"}}

    # Map result to expected output format
    if "{tool_id}" == "file_writer":
        # Handle file_writer specific mapping for new DACP response format
        if isinstance(result, dict) and result.get("success") is True:
            # New DACP format: {{'success': True, 'path': '...', 'message': '...'}}
            mapped_result = {{
                "success": True,
                "file_path": result.get("path", tool_args.get("path", "")),
                "bytes_written": len(tool_args.get("content", ""))
            }}
        elif isinstance(result, dict) and "result" in result and ("Written to " in result["result"] or "Successfully wrote" in result["result"]):
            # Legacy format: {{'result': 'Written to path'}} or {{'result': 'Successfully wrote X characters to path'}}
            result_text = result["result"]
            if "Written to " in result_text:
                file_path = result_text.replace("Written to ", "")
            elif "Successfully wrote" in result_text:
                # Extract path from "Successfully wrote X characters to path"
                file_path = result_text.split(" to ")[-1]
            else:
                file_path = tool_args.get("path", "")
            mapped_result = {{
                "success": True,
                "file_path": file_path,
                "bytes_written": len(tool_args.get("content", ""))
            }}
        else:
            mapped_result = {{
                "success": False,
                "file_path": tool_args.get("path", ""),
                "bytes_written": 0
            }}
    else:
        mapped_result = result

    # Return the result in the expected output format
    return {output_type}(**mapped_result)
"""


def _generate_task_function(
    task_name: str,
    task_def: Dict[str, Any],
    spec_data: Dict[str, Any],
    agent_name: str,
    memory_config: Dict[str, Any],
    config: Dict[str, Any],
) -> str:
    """Generate a single task function."""
    # Check if this task uses a tool
    if "tool" in task_def:
        return _generate_tool_task_function(
            task_name, task_def, spec_data, agent_name, memory_config, config
        )

    # Check if this is a multi-step task
    if task_def.get("multi_step", False):
        return _generate_multi_step_task_function(
            task_name, task_def, spec_data, agent_name, memory_config
        )

    # Regular single-step task generation (existing logic)
    func_name = task_name.replace("-", "_")
    input_params = _generate_input_params(task_def)
    output_type = f"{task_name.replace('-', '_').title()}Output"
    docstring = _generate_function_docstring(task_name, task_def, output_type)
    contract_data = _generate_contract_data(
        spec_data, task_def, agent_name, memory_config
    )

    # Create input dict with actual parameter values
    input_dict = {}
    for param in input_params:
        if param != "memory_summary: str = ''":
            param_name = param.split(":")[0]
            input_dict[param_name] = param_name

    # Add LLM output parser if this is an LLM-based agent
    llm_parser = ""
    parser_function_name = ""
    if config.get("model"):  # If model is specified, this is an LLM agent
        llm_parser = _generate_llm_output_parser(task_name, task_def.get("output", {}))
        parser_function_name = f"parse_{task_name.replace('-', '_')}_output"

    # Determine client usage based on engine
    engine = spec_data.get("intelligence", {}).get("engine", "openai")
    custom_module = spec_data.get("intelligence", {}).get("module", None)

    # Use DACP for LLM communication or custom router
    if engine == "custom" and custom_module:
        client_code = f"""# Create and use custom LLM router
    router = load_custom_llm_router("{config["endpoint"]}", "{config["model"]}", {{}})
    result = router.run(prompt, **input_dict)"""
    else:
        intelligence_config_str = _generate_intelligence_config(spec_data, config)
        client_code = f"""# Configure intelligence for DACP
    intelligence_config = {intelligence_config_str}

    # Call the LLM using DACP
    result = invoke_intelligence(prompt, intelligence_config)"""

    # Generate prompt rendering with actual parameter values
    prompt_render_params = []
    for param in input_params:
        if param != "memory_summary: str = ''":
            param_name = param.split(":")[0]
            prompt_render_params.append(f"{param_name}={param_name}")

    # Define memory configuration with proper Python boolean values
    memory_config_str = f"""{{
        "enabled": {repr(memory_config["enabled"])},
        "format": "{memory_config["format"]}",
        "usage": "{memory_config["usage"]}",
        "required": {repr(memory_config["required"])},
        "description": "{memory_config["description"]}"
    }}"""
    memory_summary_str = "memory_summary if memory_config['enabled'] else ''"

    # Generate human-readable output description
    output_description = _generate_human_readable_output(task_def.get("output", {}))
    output_description_str = f'"""\n{output_description}\n"""'

    # Format the contract data for the decorator with proper Python values
    def format_value(v):
        if isinstance(v, bool):
            return str(v)
        elif isinstance(v, (list, tuple)):
            return f"[{', '.join(format_value(x) for x in v)}]"
        elif isinstance(v, dict):
            items = [f'"{k}": {format_value(v)}' for k, v in v.items()]
            return f"{{{', '.join(items)}}}"
        elif isinstance(v, str):
            return f'"{v}"'
        return str(v)

    contract_str = ",\n    ".join(
        f"{k}={format_value(v)}" for k, v in contract_data.items()
    )

    return f"""
{llm_parser}

@behavioural_contract(
    {contract_str}
)
def {func_name}({", ".join(input_params)}) -> {output_type}:
    {docstring}
    # Define memory configuration
    memory_config = {memory_config_str}

    # Define output format description
    output_format = {output_description_str}

    # Load and render the prompt template
    prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
    env = Environment(loader=FileSystemLoader([".", prompts_dir]))
    try:
        template = env.get_template(f"{func_name}.jinja2")
    except FileNotFoundError:
        log.warning(f"Task-specific prompt template not found, using default template")
        template = env.get_template("agent_prompt.jinja2")

    # Create input dictionary for template
    input_dict = {{
        {", ".join(f'"{param.split(":")[0]}": {param.split(":")[0]}' for param in input_params if param != "memory_summary: str = ''")}
    }}

    # Render the prompt with all necessary context - pass variables directly for template access
    prompt = template.render(
        input=input_dict,
        memory_summary={memory_summary_str},
        output_format=output_format,
        memory_config=memory_config,
        **input_dict  # Also pass variables directly for template access
    )

    {client_code}
    return {parser_function_name}(result)
"""


def generate_agent_code(
    output: Path, spec_data: Dict[str, Any], agent_name: str, class_name: str
) -> None:
    """Generate the agent.py file using template-based approach."""
    if (output / "agent.py").exists():
        log.warning("agent.py already exists and will be overwritten")

    tasks = spec_data.get("tasks", {})
    if not tasks:
        log.warning("No tasks defined in spec file")
        return

    # Use the new data preparation and template-based generation
    from .data_preparation import AgentDataPreparator
    from .code_generation import CodeGenerator

    try:
        # Prepare all data using the structured approach
        preparator = AgentDataPreparator()
        template_data = preparator.prepare_all_data(spec_data, agent_name, class_name)

        # Generate code using templates
        generator = CodeGenerator()

        # Ensure the agent template exists with a default
        default_agent_template = """{{ imports | join('\\n') }}

load_dotenv()

log = logging.getLogger(__name__)

ROLE = "{{ agent_name.title() }}"

# Generate output models
{% for model in models %}
{{ model }}

{% endfor %}

# Task functions
{% for task_function in task_functions %}
{{ task_function }}
{% endfor %}

{% if custom_router_loader %}
{{ custom_router_loader }}
{% endif %}

class {{ class_name }}(dacp.Agent):
    def __init__(self, agent_id: str, orchestrator: Orchestrator):
        super().__init__()
        self.agent_id = agent_id
        orchestrator.register_agent(agent_id, self)
        self.model = "{{ config.model }}"

        # Embed YAML config as dict during generation
        {{ embedded_config }}

        # Setup DACP logging FIRST
        self.setup_logging()
{% if custom_router_init %}
        {{ custom_router_init }}
{% endif %}

{{ handle_message_method }}

{{ setup_logging_method }}

{% for class_method in class_methods %}
{{ class_method }}
{% endfor %}

{% for memory_method in memory_methods %}
{{ memory_method }}
{% endfor %}

def main():
    # Example usage - in production, you would get these from your orchestrator setup
    from dacp.orchestrator import Orchestrator

    orchestrator = Orchestrator()
    agent = {{ class_name }}("example-agent-id", orchestrator)
{{ example_task_code }}

if __name__ == "__main__":
    main()"""

        generator.ensure_template_exists("agent.py.j2", default_agent_template)

        # Generate the agent code
        agent_code = generator.generate_from_template("agent.py.j2", **template_data)

        # Write the generated code
        (output / "agent.py").write_text(agent_code)
        log.info("agent.py created using template-based generation")
        log.debug(f"Agent class name generated: {class_name}")

    except Exception as e:
        log.error(f"Error during template-based generation: {e}")
        log.warning("Falling back to legacy generation method")
        # Fallback to legacy method if template generation fails
        _generate_agent_code_legacy(output, spec_data, agent_name, class_name)


def map_type_to_python(t):
    return {
        "string": "str",
        "number": "float",
        "integer": "int",
        "boolean": "bool",
        "object": "Dict[str, Any]",
        "array": "List[Any]",
    }.get(t, "Any")


def _generate_task_docs(tasks: Dict[str, Any]) -> List[str]:
    """Generate documentation for tasks."""
    task_docs = []
    for task_name in tasks.keys():
        task_def = tasks[task_name]
        task_docs.append(f"### {task_name.title()}\n")
        task_docs.append(f"{task_def.get('description', '')}\n")

        if task_def.get("input"):
            task_docs.append("#### Input:")
            for param_name, param_type in task_def.get("input", {}).items():
                task_docs.append(f"- {param_name}: {param_type}")
            task_docs.append("")

        if task_def.get("output"):
            task_docs.append("#### Output:")
            for param_name, param_type in task_def.get("output", {}).items():
                task_docs.append(f"- {param_name}: {param_type}")
            task_docs.append("")
    return task_docs


def _generate_memory_docs(memory_config: Dict[str, Any]) -> List[str]:
    """Generate documentation for memory configuration."""
    memory_docs = []
    if memory_config["enabled"]:
        memory_docs.append("## Memory Support\n")
        memory_docs.append(f"{memory_config['description']}\n")
        memory_docs.append("### Configuration\n")
        memory_docs.append(f"- Format: {memory_config['format']}\n")
        memory_docs.append(f"- Usage: {memory_config['usage']}\n")
        memory_docs.append(f"- Required: {memory_config['required']}\n")
        memory_docs.append(
            "\nTo implement memory support, override the `get_memory()` method in the agent class.\n"
        )
    return memory_docs


def _generate_behavioural_docs(behavioural_contract: Dict[str, Any]) -> List[str]:
    """Generate documentation for behavioural contract."""
    behavioural_docs = []
    behavioural_docs.append("## Behavioural Contract\n\n")
    behavioural_docs.append(
        "This agent is governed by the following behavioural contract policy:\n\n"
    )

    if "pii" in behavioural_contract:
        behavioural_docs.append(f"- PII: {behavioural_contract['pii']}\n")

    if "compliance_tags" in behavioural_contract:
        behavioural_docs.append(
            f"- Compliance Tags: {', '.join(behavioural_contract['compliance_tags'])}\n"
        )

    if "allowed_tools" in behavioural_contract:
        behavioural_docs.append(
            f"- Allowed Tools: {', '.join(behavioural_contract['allowed_tools'])}\n"
        )

    behavioural_docs.append(
        "\nRefer to `behavioural_contracts` for enforcement logic.\n"
    )
    return behavioural_docs


def _generate_example_usage(agent_info: Dict[str, str], tasks: Dict[str, Any]) -> str:
    """Generate example usage code."""
    first_task_name = next(iter(tasks.keys()), "")
    if not first_task_name:
        return ""

    return f"""```python
from agent import {to_pascal_case(agent_info["name"])}

agent = {to_pascal_case(agent_info["name"])}()
# Example usage
task_name = "{first_task_name}"
if task_name:
    result = getattr(agent, task_name.replace("-", "_"))(
        {", ".join(f'{k}="example_{k}"' for k in tasks[first_task_name].get("input", {}))}
    )
    print(result)
```"""


def generate_readme(output: Path, spec_data: Dict[str, Any]) -> None:
    """Generate the README.md file."""
    if (output / "README.md").exists():
        log.warning("README.md already exists and will be overwritten")

    agent_info = get_agent_info(spec_data)
    memory_config = get_memory_config(spec_data)
    tasks = spec_data.get("tasks", {})

    task_docs = _generate_task_docs(tasks)
    memory_docs = _generate_memory_docs(memory_config)
    behavioural_docs = (
        _generate_behavioural_docs(spec_data["behavioural_contract"])
        if "behavioural_contract" in spec_data
        else []
    )
    example_usage = _generate_example_usage(agent_info, tasks)

    readme_content = f"""# {agent_info["name"].title().replace("-", " ")}

{agent_info["description"]}

## Usage

```bash
pip install -r requirements.txt
cp .env.example .env
python agent.py
```

## Tasks

{chr(10).join(task_docs)}
{chr(10).join(memory_docs)}
{chr(10).join(behavioural_docs)}

## Example Usage

{example_usage}
"""
    (output / "README.md").write_text(readme_content)
    log.info("README.md created")


def generate_requirements(output: Path, spec_data: Dict[str, Any]) -> None:
    """Generate the requirements.txt file."""
    if (output / "requirements.txt").exists():
        log.warning("requirements.txt already exists and will be overwritten")

    engine = spec_data.get("intelligence", {}).get("engine", "openai")

    requirements = []
    if engine == "openai":
        requirements.append("openai>=1.0.0")
    elif engine == "anthropic":
        requirements.append("anthropic>=0.18.0")
    elif engine == "local":
        requirements.append("# Add your local engine dependencies here")
    elif engine == "custom":
        requirements.append("# Add your custom engine dependencies here")
    else:
        requirements.append("openai>=1.0.0")  # Default fallback

    requirements.extend(
        [
            "# Note: During development, install with: pip install -r requirements.txt --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/",
            "behavioural-contracts>=0.1.0",
            "python-dotenv>=0.19.0",
            "pydantic>=2.0.0",
            "jinja2>=3.0.0",
            "dacp>=0.1.0",
        ]
    )

    (output / "requirements.txt").write_text("\n".join(requirements) + "\n")
    log.info("requirements.txt created")


def generate_env_example(output: Path, spec_data: Dict[str, Any]) -> None:
    """Generate the .env.example file."""
    if (output / ".env.example").exists():
        log.warning(".env.example already exists and will be overwritten")

    engine = spec_data.get("intelligence", {}).get("engine", "openai")

    if engine == "anthropic":
        env_content = "ANTHROPIC_API_KEY=your-api-key-here\n"
    elif engine == "openai":
        env_content = "OPENAI_API_KEY=your-api-key-here\n"
    elif engine == "local":
        env_content = "# Add your local engine environment variables here\n"
    elif engine == "custom":
        env_content = "# Add your custom engine environment variables here\n"
    else:
        env_content = "OPENAI_API_KEY=your-api-key-here\n"

    (output / ".env.example").write_text(env_content)
    log.info(".env.example created")


def generate_prompt_template(output: Path, spec_data: Dict[str, Any]) -> None:
    """Generate the prompt template file."""
    prompts_dir = output / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    # Generate task-specific templates
    for task_name, task_def in spec_data.get("tasks", {}).items():
        template_name = f"{task_name}.jinja2"
        if (prompts_dir / template_name).exists():
            log.warning(f"{template_name} already exists and will be overwritten")

        # Get the output schema
        output_schema = task_def.get("output", {})

        # Prepare example JSON for the output
        # Use the output schema's properties to generate a structured example
        example_json_lines = []
        if output_schema.get("properties"):
            example_json_lines.append("{")
            for i, (k, v) in enumerate(output_schema["properties"].items()):
                comma = "," if i < len(output_schema["properties"]) - 1 else ""
                example_json_lines.extend(_generate_json_example(k, v, 2, comma))
            example_json_lines.append("}")
        else:
            example_json_lines.append("{}")
        example_json = "\n".join(example_json_lines)

        # Handle both old and new prompt formats
        if "prompt" in spec_data and "template" in spec_data["prompt"]:
            # Old format - use the template directly
            prompt_content = spec_data["prompt"]["template"]
        elif "prompts" in spec_data:
            prompts = spec_data.get("prompts", {})

            # Check for task-specific prompts first
            task_system_prompt = prompts.get(task_name, {}).get("system")
            task_user_prompt = prompts.get(task_name, {}).get("user")

            if task_system_prompt or task_user_prompt:
                # Use task-specific prompts
                prompt_content = task_system_prompt or ""
                if task_user_prompt:
                    if prompt_content:
                        prompt_content += "\n\n"
                    prompt_content += task_user_prompt
            elif prompts.get("system") or prompts.get("user"):
                # Fall back to global prompts
                system_prompt = prompts.get(
                    "system",
                    "You are a professional AI agent designed to process tasks according to the Open Agent Spec.\n\n",
                )
                user_prompt = prompts.get("user", "")

                # Combine system and user prompts with proper spacing
                prompt_content = system_prompt
                if user_prompt:
                    # Ensure proper spacing between system and user prompts
                    if not prompt_content.endswith(
                        "\n"
                    ) and not prompt_content.endswith(" "):
                        prompt_content += " "
                    prompt_content += user_prompt
            else:
                # No prompts defined, use default
                prompt_content = ""

            # Add memory context if not already present
            if "{% if memory_summary %}" not in prompt_content:
                prompt_content = (
                    "{% if memory_summary %}\n"
                    "--- MEMORY CONTEXT ---\n"
                    "{{ memory_summary }}\n"
                    "------------------------\n"
                    "{% endif %}\n\n"
                ) + prompt_content
        else:
            # Default format - use the full default template content
            default_content = """You are a professional AI agent designed to process tasks according to the Open Agent Spec.

{% if memory_summary %}
--- MEMORY CONTEXT ---
{{ memory_summary }}
------------------------
{% endif %}

TASK:
Process the following task:

{% for key, value in input.items() %}
{{ key }}: {{ value }}
{% endfor %}

INSTRUCTIONS:
1. Review the input data carefully
2. Consider all relevant factors
{% if memory_summary %}
3. Take into account the provided memory context
{% endif %}
4. Provide a clear, actionable response
5. Explain your reasoning in detail

OUTPUT FORMAT:
Your response should include the following fields:
{{ output_format }}

CONSTRAINTS:
- Be clear and specific
- Focus on actionable insights
- Maintain professional objectivity
{% if memory_summary and memory_config.required %}
- Must reference and incorporate memory context
{% endif %}"""

            prompt_content = default_content

        # Always append the JSON schema instruction and example
        prompt_content += (
            f"\nRespond ONLY with a JSON object in this exact format:\n{example_json}\n"
        )

        (prompts_dir / template_name).write_text(prompt_content)
        log.info(f"Created prompt template: {template_name}")

    # Generate default template
    default_template = prompts_dir / "agent_prompt.jinja2"
    if default_template.exists():
        log.warning("agent_prompt.jinja2 already exists and will be overwritten")

    default_content = """You are a professional AI agent designed to process tasks according to the Open Agent Spec.

{% if memory_summary %}
--- MEMORY CONTEXT ---
{{ memory_summary }}
------------------------
{% endif %}

TASK:
Process the following task:

{% for key, value in input.items() %}
{{ key }}: {{ value }}
{% endfor %}

INSTRUCTIONS:
1. Review the input data carefully
2. Consider all relevant factors
{% if memory_summary %}
3. Take into account the provided memory context
{% endif %}
4. Provide a clear, actionable response
5. Explain your reasoning in detail

OUTPUT FORMAT:
Your response should include the following fields:
{{ output_format }}

CONSTRAINTS:
- Be clear and specific
- Focus on actionable insights
- Maintain professional objectivity
{% if memory_summary and memory_config.required %}
- Must reference and incorporate memory context
{% endif %}"""

    default_template.write_text(default_content)
    log.info("Created default prompt template: agent_prompt.jinja2")


def _generate_agent_code_legacy(
    output: Path, spec_data: Dict[str, Any], agent_name: str, class_name: str
) -> None:
    """Legacy agent code generation method (fallback only).

    This is kept as a fallback in case template-based generation fails.
    Should not be used directly - use generate_agent_code() instead.
    """
    log.warning("Using legacy agent code generation method")

    tasks = spec_data.get("tasks", {})
    config = {
        "endpoint": spec_data.get("intelligence", {}).get(
            "endpoint", "https://api.openai.com/v1"
        ),
        "model": spec_data.get("intelligence", {}).get("model", "gpt-3.5-turbo"),
        "temperature": spec_data.get("intelligence", {})
        .get("config", {})
        .get("temperature", 0.7),
        "max_tokens": spec_data.get("intelligence", {})
        .get("config", {})
        .get("max_tokens", 1000),
    }
    memory_config = get_memory_config(spec_data)

    # Generate task functions and class methods using legacy methods
    task_functions = []
    class_methods = []
    model_definitions = []

    for task_name, task_def in tasks.items():
        # Generate model definition
        model_name = f"{task_name.replace('-', '_').title()}Output"
        model_def = _generate_pydantic_model(model_name, task_def.get("output", {}))
        if model_def:
            model_definitions.append(model_def)

        # Generate task function
        task_functions.append(
            _generate_task_function(
                task_name, task_def, spec_data, agent_name, memory_config, config
            )
        )

        # Generate corresponding class method
        input_params_without_memory = [
            param.split(":")[0]
            for param in _generate_input_params(task_def)
            if param != "memory_summary: str = ''"
        ]

        if input_params_without_memory:
            method_signature = f"def {task_name.replace('-', '_')}(self, {', '.join(input_params_without_memory)}) -> {model_name}:"
            method_call = f"return {task_name.replace('-', '_')}({', '.join(input_params_without_memory)}, memory_summary=memory_summary)"
        else:
            method_signature = (
                f"def {task_name.replace('-', '_')}(self) -> {model_name}:"
            )
            method_call = (
                f"return {task_name.replace('-', '_')}(memory_summary=memory_summary)"
            )

        class_methods.append(
            f'''
    {method_signature}
        """Process {task_name} task."""
        memory_summary = self.get_memory() if hasattr(self, 'get_memory') else ""
        {method_call}
'''
        )

    # Generate memory-related methods if memory is enabled
    memory_methods = []
    if memory_config["enabled"]:
        memory_methods.append(
            '''
    def get_memory(self) -> str:
        """Get memory for the current context.

        This is a stub method that should be implemented by the developer.
        The memory format and retrieval mechanism are not prescribed by OAS.

        Returns:
            str: Memory string in the format specified by the spec
        """
        return ""  # Implement your memory retrieval logic here
'''
        )

    # Generate basic example task code
    example_task_code = ""
    if tasks:
        first_task_name = next(iter(tasks))
        example_task_code = f"""
    # Example usage with {first_task_name} task
    result = agent.{first_task_name.replace("-", "_")}()
    print(result)"""

    # Generate handle_message method
    handle_message_method = '''
    def handle_message(self, message: dict) -> dict:
        """Handle incoming messages from the orchestrator."""
        task = message.get("task")
        if not task:
            return {"error": "Missing required field: task"}
        method_name = task.replace("-", "_")
        if not hasattr(self, method_name):
            return {"error": f"Unknown task: {task}"}
        try:
            method = getattr(self, method_name)
            method_params = {k: v for k, v in message.items() if k != "task"}
            result = method(**method_params)
            if hasattr(result, 'model_dump'):
                return result.model_dump()
            else:
                return result
        except Exception as e:
            return {"error": f"Error executing task {task}: {str(e)}"}
'''

    # Generate imports
    imports = [
        "from typing import Dict, Any, List, Optional",
        "import json",
        "import logging",
        "import os",
        "from dotenv import load_dotenv",
        "from behavioural_contracts import behavioural_contract",
        "from jinja2 import Environment, FileSystemLoader",
        "from pydantic import BaseModel",
        "from dacp.orchestrator import Orchestrator",
        "import dacp",
    ]

    # Generate the complete agent code using legacy f-string approach
    agent_code = f"""{chr(10).join(imports)}

load_dotenv()

log = logging.getLogger(__name__)

ROLE = "{agent_name.title()}"

# Generate output models
{chr(10).join(model_definitions)}

{chr(10).join(task_functions)}

class {class_name}(dacp.Agent):
    def __init__(self, agent_id: str, orchestrator: Orchestrator):
        super().__init__()
        self.agent_id = agent_id
        orchestrator.register_agent(agent_id, self)
        self.model = "{config["model"]}"

        # Embed YAML config as dict during generation
        {_generate_embedded_config(spec_data)}

        # Setup DACP logging FIRST
        self.setup_logging()

{handle_message_method}

{_generate_setup_logging_method()}

{chr(10).join(class_methods)}

{chr(10).join(memory_methods)}

def main():
    from dacp.orchestrator import Orchestrator
    orchestrator = Orchestrator()
    agent = {class_name}("example-agent-id", orchestrator)
{example_task_code}

if __name__ == "__main__":
    main()
"""
    (output / "agent.py").write_text(agent_code)
    log.info("agent.py created using legacy generation method")
