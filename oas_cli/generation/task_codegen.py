# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""Facade re-exports for task/model codegen (split into submodules).

Import from here for a stable aggregate API, or import from the specific
submodules (task_preamble, single_step_codegen, etc.) directly.
"""

from .contracts_codegen import _format_contract_for_decorator, _generate_contract_data
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
from .pydantic_codegen import (
    _generate_pydantic_model,
    _get_pydantic_type,
    generate_models,
)
from .single_step_codegen import (
    _generate_llm_output_parser,
    _get_human_readable_type,
)
from .task_functions import _generate_task_function
from .task_preamble import (
    _generate_function_docstring,
    _generate_input_params,
    _get_task_function_preamble,
)
from .tool_task_codegen import (
    _build_tool_args_and_description,
    _generate_tool_task_function,
)
from .types_mapping import map_type_to_python

__all__ = [
    "_build_memory_config_python_code",
    "_build_multi_step_execution_code",
    "_build_multi_step_output_construction",
    "_build_single_step_llm_client_code",
    "_build_tool_args_and_description",
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
