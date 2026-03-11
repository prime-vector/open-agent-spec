# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""Modular code generation. Prefer importing from submodules; generators.py re-exports for compatibility."""

from .artifacts import (
    generate_env_example,
    generate_prompt_template,
    generate_readme,
    generate_requirements,
)
from .constants import DEFAULT_AGENT_PROMPT_TEMPLATE
from .pydantic_codegen import generate_models, _generate_pydantic_model
from .spec_config import (
    get_agent_info,
    get_logging_config,
    get_memory_config,
    to_pascal_case,
)
from .task_functions import _generate_input_params, _generate_task_function
from .types_mapping import map_type_to_python

__all__ = [
    "DEFAULT_AGENT_PROMPT_TEMPLATE",
    "generate_env_example",
    "generate_models",
    "generate_prompt_template",
    "generate_readme",
    "generate_requirements",
    "get_agent_info",
    "get_logging_config",
    "get_memory_config",
    "map_type_to_python",
    "to_pascal_case",
    "_generate_input_params",
    "_generate_pydantic_model",
    "_generate_task_function",
]
