# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""Task function string generation: dispatcher + re-exports from split submodules."""

import logging
from typing import Any

from .contracts_codegen import _format_contract_for_decorator
from .intelligence_codegen import (
    _build_memory_config_python_code,
    _build_single_step_llm_client_code,
    _generate_embedded_config,
    _generate_intelligence_config,
    _generate_setup_logging_method,
)
from .multi_step_codegen import (
    _build_multi_step_execution_code,
    _build_multi_step_output_construction,
    _generate_multi_step_task_function,
)
from .single_step_codegen import (
    _generate_human_readable_output,
    _generate_llm_output_parser,
    _get_human_readable_type,
)
from .task_preamble import (
    _generate_function_docstring,
    _generate_input_params,
    _get_task_function_preamble,
)
from .tool_task_codegen import (
    _build_tool_args_and_description,
    _generate_tool_task_function,
)

log = logging.getLogger("oas")

__all__ = [
    "_build_memory_config_python_code",
    "_build_multi_step_execution_code",
    "_build_multi_step_output_construction",
    "_build_single_step_llm_client_code",
    "_build_tool_args_and_description",
    "_generate_embedded_config",
    "_generate_function_docstring",
    "_generate_input_params",
    "_generate_intelligence_config",
    "_generate_llm_output_parser",
    "_generate_multi_step_task_function",
    "_generate_setup_logging_method",
    "_generate_task_function",
    "_generate_tool_task_function",
    "_get_human_readable_type",
    "_get_task_function_preamble",
]


def _generate_task_function(
    task_name: str,
    task_def: dict[str, Any],
    spec_data: dict[str, Any],
    agent_name: str,
    memory_config: dict[str, Any],
    config: dict[str, Any],
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
    func_name, input_params, output_type, docstring, contract_data = (
        _get_task_function_preamble(
            task_name, task_def, spec_data, agent_name, memory_config
        )
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
    if config.get("model"):
        llm_parser = _generate_llm_output_parser(task_name, task_def.get("output", {}))
        parser_function_name = f"parse_{task_name.replace('-', '_')}_output"

    client_code = _build_single_step_llm_client_code(spec_data, config, input_dict)

    memory_config_str = _build_memory_config_python_code(memory_config)
    memory_summary_str = "memory_summary if memory_config['enabled'] else ''"
    output_description = _generate_human_readable_output(task_def.get("output", {}))
    output_description_str = f'"""\n{output_description}\n"""'
    decorator = ""
    if contract_data:
        contract_str = _format_contract_for_decorator(contract_data)
        decorator = f"@behavioural_contract(\n    {contract_str}\n)\n"

    return f"""
{llm_parser}

{decorator}def {func_name}({", ".join(input_params)}) -> {output_type}:
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
