# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""Multi-step task function generation (step execution + output construction)."""

import json
from typing import Any

from .contracts_codegen import _format_contract_for_decorator
from .task_preamble import _get_task_function_preamble


def _step_arg_from_input_map_value(
    param: str,
    value: Any,
    current_step_index: int,
    step_result_vars: list[str],
) -> str:
    """Turn one input_map entry into a single call arg fragment, e.g. param=foo or param=step_0_result.field.

    step_result_vars[k] is the variable name for step k's result (only steps k < current_step_index exist).
    Invalid or forward references yield param=\"\" to avoid generating broken code.
    Non-string literals (int/float/bool) are emitted without quotes; dict/list are
    serialized as JSON so the generated call stays valid Python.
    """
    if not (isinstance(value, str) and "{{" in value and "}}" in value):
        if value is True:
            return f"{param}=True"
        if value is False:
            return f"{param}=False"
        if value is None:
            return f"{param}=None"
        if isinstance(value, (int, float)):
            return f"{param}={value}"
        if isinstance(value, (dict, list)):
            try:
                return f"{param}={json.dumps(value)}"
            except (TypeError, ValueError):
                return f'{param}=""'
        return f'{param}="{value}"'

    var_name = value.replace("{{", "").replace("}}", "").strip()
    if "." not in var_name:
        return f"{param}={var_name}"

    parts = var_name.split(".")
    if parts[0] == "input":
        return f"{param}={parts[-1]}"

    if parts[0] == "steps" and len(parts) >= 3:
        index_part, field_name = parts[1], parts[2]
        if not index_part.isdigit():
            return f'{param}=""'
        step_index = int(index_part)
        # Only previous steps are available; step_result_vars has length == current_step_index
        if step_index < 0 or step_index >= len(step_result_vars):
            return f'{param}=""'
        if step_index >= current_step_index:
            return f'{param}=""'
        step_var = step_result_vars[step_index]
        return (
            f"{param}={step_var}.{field_name} if hasattr({step_var}, '{field_name}') "
            f"else {step_var}.get('{field_name}', '')"
        )

    return f"{param}={var_name}"


def _build_multi_step_execution_code(steps: list[dict[str, Any]]) -> str:
    """Build the step execution code block for a multi-step task (e.g. step_0_result = ...)."""
    step_code: list[str] = []
    step_result_vars: list[str] = []

    for i, step in enumerate(steps):
        step_task = step["task"]
        input_map = step.get("input_map", {})
        step_inputs = [
            _step_arg_from_input_map_value(param, v, i, step_result_vars)
            for param, v in input_map.items()
        ]
        step_input_str = ", ".join(step_inputs)
        step_var = f"step_{i}_result"
        step_result_vars.append(step_var)
        step_code.append(
            f"""    # Execute step {i + 1}: {step_task}
    {step_var} = {step_task.replace("-", "_")}({step_input_str})"""
        )
    return "\n".join(step_code)


def _build_multi_step_output_construction(
    steps: list[dict[str, Any]], output_schema: dict[str, Any]
) -> str:
    """Build the output construction from step results (prop=next((v for v in [...]), None))."""
    step_results = [f"step_{i}_result" for i in range(len(steps))]
    output_properties = output_schema.get("properties", {})
    output_construction = []
    for prop_name in output_properties:
        candidates = []
        for sv in step_results:
            candidates.append(f"getattr({sv}, '{prop_name}', None)")
            candidates.append(
                f"({sv}.get('{prop_name}', None) if isinstance({sv}, dict) else None)"
            )
        candidates_str = ", ".join(candidates)
        output_construction.append(
            f"        {prop_name}=next((v for v in [{candidates_str}] if v is not None), None)"
        )
    return ",\n".join(output_construction)


def _generate_multi_step_task_function(
    task_name: str,
    task_def: dict[str, Any],
    spec_data: dict[str, Any],
    agent_name: str,
    memory_config: dict[str, Any],
) -> str:
    """Generate a multi-step task function that orchestrates other tasks."""
    func_name, input_params, output_type, docstring, contract_data = (
        _get_task_function_preamble(
            task_name, task_def, spec_data, agent_name, memory_config
        )
    )
    steps = task_def.get("steps", [])

    step_code = _build_multi_step_execution_code(steps)
    output_construction_str = _build_multi_step_output_construction(
        steps, task_def.get("output", {})
    )
    decorator = ""
    if contract_data:
        contract_str = _format_contract_for_decorator(contract_data)
        decorator = f"@behavioural_contract(\n    {contract_str}\n)\n"

    return f"""
{decorator}def {func_name}({", ".join(input_params)}) -> {output_type}:
    {docstring}
    # Execute multi-step task: {task_name}
{step_code}

    # Construct output from step results
    return {output_type}(
{output_construction_str}
    )
"""
