# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""Facade re-exports for task/model codegen (split into submodules)."""

from .contracts_codegen import _format_contract_for_decorator, _generate_contract_data
from .pydantic_codegen import _generate_pydantic_model, _get_pydantic_type, generate_models
from .task_functions import (
    _build_memory_config_python_code,
    _build_multi_step_execution_code,
    _build_multi_step_output_construction,
    _build_single_step_llm_client_code,
    _build_tool_args_and_description,
    _generate_embedded_config,
    _generate_function_docstring,
    _generate_input_params,
    _generate_intelligence_config,
    _generate_llm_output_parser,
    _generate_multi_step_task_function,
    _generate_setup_logging_method,
    _generate_task_function,
    _generate_tool_task_function,
    _get_human_readable_type,
    _get_task_function_preamble,
)
from .types_mapping import map_type_to_python

__all__ = [
    "_format_contract_for_decorator",
    "_generate_contract_data",
    "_generate_embedded_config",
    "_generate_function_docstring",
    "_generate_input_params",
    "_generate_intelligence_config",
    "_generate_llm_output_parser",
    "_generate_multi_step_task_function",
    "_generate_pydantic_model",
    "_generate_setup_logging_method",
    "_generate_task_function",
    "_generate_tool_task_function",
    "_get_human_readable_type",
    "_get_pydantic_type",
    "_get_task_function_preamble",
    "generate_models",
    "map_type_to_python",
]
