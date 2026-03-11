# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""File generation façade: re-exports generation package; agent.py writer only here."""

import logging
from pathlib import Path
from typing import Any

from .generation.artifacts import (
    generate_env_example,
    generate_prompt_template,
    generate_readme,
    generate_requirements,
)
from .generation.constants import DEFAULT_AGENT_PROMPT_TEMPLATE
from .generation.spec_config import (
    get_agent_info,
    get_logging_config,
    get_memory_config,
    to_pascal_case,
)
from .generation.pydantic_codegen import _generate_pydantic_model, generate_models
from .generation.task_functions import _generate_input_params, _generate_task_function
from .generation.types_mapping import map_type_to_python

log = logging.getLogger("oas")

__all__ = [
    "DEFAULT_AGENT_PROMPT_TEMPLATE",
    "generate_agent_code",
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


def generate_agent_code(
    output: Path, spec_data: dict[str, Any], agent_name: str, class_name: str
) -> None:
    """Generate agent.py via the template path only (no legacy fallback)."""
    if (output / "agent.py").exists():
        log.warning("agent.py already exists and will be overwritten")

    tasks = spec_data.get("tasks", {})
    if not tasks:
        log.warning("No tasks defined in spec file")
        return

    from jinja2.exceptions import TemplateError as JinjaTemplateError

    from .code_generation import CodeGenerator
    from .data_preparation import AgentDataPreparator

    _GEN_AGENT_CODE_ERRORS = (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        JinjaTemplateError,
    )

    try:
        preparator = AgentDataPreparator()
        template_data = preparator.prepare_all_data(spec_data, agent_name, class_name)

        generator = CodeGenerator()
        _agent_template_path = Path(__file__).parent / "templates" / "agent.py.j2"
        if not _agent_template_path.is_file():
            raise RuntimeError(
                f"Missing packaged template: {_agent_template_path}. "
                "Reinstall open-agent-spec or restore oas_cli/templates/agent.py.j2."
            )
        generator.ensure_template_exists(
            "agent.py.j2",
            _agent_template_path.read_text(encoding="utf-8"),
        )

        agent_code = generator.generate_from_template("agent.py.j2", **template_data)
        (output / "agent.py").write_text(agent_code)
    except _GEN_AGENT_CODE_ERRORS as e:
        log.error("Agent code generation failed: %s", e)
        raise RuntimeError(
            "Agent code generation failed (template path only; legacy codegen "
            "has been removed). Check the spec and template data, or reinstall "
            "the package if templates are missing."
        ) from e

    log.info("agent.py created using template-based generation")
    log.debug("Agent class name generated: %s", class_name)
