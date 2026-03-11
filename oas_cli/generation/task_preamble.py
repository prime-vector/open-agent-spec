# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""Shared preamble for task function generation (input params, docstring, contract data)."""

from typing import Any

from .contracts_codegen import _generate_contract_data
from .types_mapping import map_type_to_python


def _generate_input_params(task_def: dict[str, Any]) -> list[str]:
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
    task_name: str,
    task_def: dict[str, Any],
    output_type: str,
    input_params: list[str] | None = None,
) -> str:
    """Generate docstring for a task function.

    Args section is built from ``input_params`` when provided so multi-step
    inferred params match the signature; otherwise falls back to input.properties.
    """
    if input_params:
        # Skip memory_summary default in Args (documented separately).
        arg_lines = []
        for p in input_params:
            if p.strip().startswith("memory_summary"):
                continue
            # Already "name: type" form
            arg_lines.append(f"        {p}")
        args_block = "\n".join(arg_lines) if arg_lines else "        (see signature)"
    else:
        props = task_def.get("input", {}).get("properties", {})
        if props:
            args_block = "\n".join(
                f"        {param_name}: {param_def.get('type', 'any') if isinstance(param_def, dict) else param_def}"
                for param_name, param_def in props.items()
            )
        else:
            args_block = "        (see signature)"
    return f'''"""Process {task_name} task.

    Args:
{args_block}
        memory_summary: Optional memory context for the task

    Returns:
        {output_type}
    """'''


def _get_task_function_preamble(
    task_name: str,
    task_def: dict[str, Any],
    spec_data: dict[str, Any],
    agent_name: str,
    memory_config: dict[str, Any],
) -> tuple[str, list[str], str, str, dict[str, Any]]:
    """Return (func_name, input_params, output_type, docstring, contract_data) for task generators."""
    func_name = task_name.replace("-", "_")
    input_params = _generate_input_params(task_def)
    output_type = f"{task_name.replace('-', '_').title()}Output"
    docstring = _generate_function_docstring(
        task_name, task_def, output_type, input_params=input_params
    )
    contract_data = _generate_contract_data(
        spec_data, task_def, agent_name, memory_config
    )
    return func_name, input_params, output_type, docstring, contract_data
