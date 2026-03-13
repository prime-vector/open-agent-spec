# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""Intelligence config, embedded config, logging setup, memory literal, and LLM client code blocks."""

from typing import Any


def _generate_intelligence_config(
    spec_data: dict[str, Any], config: dict[str, Any]
) -> str:
    """Generate intelligence configuration for DACP invoke_intelligence.

    Deprecated: Use PythonCodeSerializer.dict_to_python_code() instead.
    """
    from ..code_generation import PythonCodeSerializer

    intelligence = spec_data.get("intelligence", {})
    intelligence_config = {
        "engine": intelligence.get("engine", "openai"),
        # Prefer a widely available default model when none is specified.
        "model": intelligence.get("model", config.get("model", "gpt-4o")),
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


def _generate_embedded_config(spec_data: dict[str, Any]) -> str:
    """Generate embedded YAML configuration as Python dict.

    Deprecated: Use AgentDataPreparator._prepare_embedded_config() instead.
    """
    from ..data_preparation import AgentDataPreparator

    preparator = AgentDataPreparator()
    return preparator._prepare_embedded_config(spec_data)


def _generate_setup_logging_method() -> str:
    """Generate setup_logging method for DACP logging integration.

    Deprecated: Use AgentDataPreparator._prepare_setup_logging_method() instead.
    """
    from ..data_preparation import AgentDataPreparator

    preparator = AgentDataPreparator()
    return preparator._prepare_setup_logging_method()


def _build_memory_config_python_code(memory_config: dict[str, Any]) -> str:
    """Build the memory_config dict literal as Python code for generated task functions."""
    return f"""{{
        "enabled": {memory_config["enabled"]!r},
        "format": "{memory_config["format"]}",
        "usage": "{memory_config["usage"]}",
        "required": {memory_config["required"]!r},
        "description": "{memory_config["description"]}"
    }}"""


def _build_single_step_llm_client_code(
    spec_data: dict[str, Any], config: dict[str, Any], input_dict: dict[str, str]
) -> str:
    """Build the LLM client code block (DACP or custom router) for a single-step task."""
    engine = spec_data.get("intelligence", {}).get("engine", "openai")
    custom_module = spec_data.get("intelligence", {}).get("module", None)
    if engine == "custom" and custom_module:
        return f"""# Create and use custom LLM router
    router = load_custom_llm_router("{config["endpoint"]}", "{config["model"]}", {{}})
    result = router.run(prompt, **input_dict)"""
    intelligence_config_str = _generate_intelligence_config(spec_data, config)
    return f"""# Configure intelligence for DACP
    intelligence_config = {intelligence_config_str}

    # Call the LLM using DACP
    result = invoke_intelligence(prompt, intelligence_config)"""
