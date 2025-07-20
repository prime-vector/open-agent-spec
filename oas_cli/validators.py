# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under AGPL-3.0 with Additional Terms
# See LICENSE for details on attribution, naming, and branding restrictions.

"""Validation functions for Open Agent Spec."""

from typing import Tuple
import json
import logging

from jsonschema import validate
from jsonschema.exceptions import SchemaError, ValidationError

log = logging.getLogger(__name__)


def _validate_version(spec_data: dict) -> None:
    """Validate the spec version."""
    version = spec_data.get("open_agent_spec")
    if not isinstance(version, str):
        raise ValueError("open_agent_spec version must be specified as a string")
    if not version:
        raise ValueError("open_agent_spec version cannot be empty")


def _validate_agent(spec_data: dict) -> None:
    """Validate the agent section."""
    agent = spec_data.get("agent", {})
    if not isinstance(agent.get("name"), str):
        raise ValueError("agent.name must be a string")
    if not isinstance(agent.get("role"), str):
        raise ValueError("agent.role must be a string")


def _validate_behavioural_contract(spec_data: dict) -> None:
    """Validate the behavioural contract section."""
    contract = spec_data.get("behavioural_contract", {})
    if not isinstance(contract.get("version"), str):
        raise ValueError("behavioural_contract.version must be a string")
    if not isinstance(contract.get("description"), str):
        raise ValueError("behavioural_contract.description must be a string")

    # Optional fields - only validate if present
    if "behavioural_flags" in contract and not isinstance(
        contract["behavioural_flags"], dict
    ):
        raise ValueError("behavioural_contract.behavioural_flags must be a dictionary")
    if "response_contract" in contract and not isinstance(
        contract["response_contract"], dict
    ):
        raise ValueError("behavioural_contract.response_contract must be a dictionary")
    if "policy" in contract and not isinstance(contract["policy"], dict):
        raise ValueError("behavioural_contract.policy must be a dictionary")
    if "teardown_policy" in contract and not isinstance(
        contract["teardown_policy"], dict
    ):
        raise ValueError("behavioural_contract.teardown_policy must be a dictionary")


def _validate_tools(spec_data: dict) -> None:
    """Validate the tools section."""
    tools = spec_data.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")

    for i, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"tool {i} must be a dictionary")

        if not isinstance(tool.get("id"), str):
            raise ValueError(f"tool {i}.id must be a string")

        if not isinstance(tool.get("description"), str):
            raise ValueError(f"tool {i}.description must be a string")

        if not isinstance(tool.get("type"), str):
            raise ValueError(f"tool {i}.type must be a string")

        # Validate allowed_paths if present (for file operations)
        if "allowed_paths" in tool:
            if not isinstance(tool["allowed_paths"], list):
                raise ValueError(f"tool {i}.allowed_paths must be a list")
            for j, path in enumerate(tool["allowed_paths"]):
                if not isinstance(path, str):
                    raise ValueError(f"tool {i}.allowed_paths[{j}] must be a string")


def _validate_tasks(spec_data: dict) -> None:
    """Validate the tasks section."""
    tasks = spec_data.get("tasks", {})
    tools = spec_data.get("tools", [])
    tool_ids = [tool["id"] for tool in tools]

    if not isinstance(tasks, dict):
        raise ValueError("tasks must be a dictionary")
    for task_name, task_def in tasks.items():
        # Check if this task uses a tool
        if "tool" in task_def:
            tool_id = task_def["tool"]
            if not isinstance(tool_id, str):
                raise ValueError(f"task {task_name}.tool must be a string")
            if tool_id not in tool_ids:
                raise ValueError(
                    f"task {task_name} references non-existent tool '{tool_id}'"
                )

        # Check if this is a multi-step task
        is_multi_step = task_def.get("multi_step", False)

        # For multi-step tasks, input and output are optional since they orchestrate other tasks
        if not is_multi_step:
            if not isinstance(task_def.get("input"), dict):
                raise ValueError(f"task {task_name}.input must be a dictionary")
            if not isinstance(task_def.get("output"), dict):
                raise ValueError(f"task {task_name}.output must be a dictionary")
        else:
            # For multi-step tasks, validate that steps are defined
            if not isinstance(task_def.get("steps"), list):
                raise ValueError(f"multi-step task {task_name}.steps must be a list")
            if not task_def.get("steps"):
                raise ValueError(f"multi-step task {task_name}.steps cannot be empty")

            # Validate output schema for multi-step tasks
            if not isinstance(task_def.get("output"), dict):
                raise ValueError(
                    f"multi-step task {task_name}.output must be a dictionary"
                )
            if not task_def.get("output"):
                raise ValueError(f"multi-step task {task_name}.output cannot be empty")

            # Validate each step
            for i, step in enumerate(task_def["steps"]):
                if not isinstance(step, dict):
                    raise ValueError(
                        f"step {i} in task {task_name} must be a dictionary"
                    )
                if "task" not in step:
                    raise ValueError(
                        f"step {i} in task {task_name} must have a 'task' field"
                    )
                if not isinstance(step["task"], str):
                    raise ValueError(
                        f"step {i} in task {task_name}.task must be a string"
                    )

                # Check that the referenced task exists
                referenced_task = step["task"]
                if referenced_task not in tasks:
                    raise ValueError(
                        f"step {i} in task {task_name} references non-existent task '{referenced_task}'"
                    )

                # Validate input_map if present
                if "input_map" in step:
                    if not isinstance(step["input_map"], dict):
                        raise ValueError(
                            f"step {i} in task {task_name}.input_map must be a dictionary"
                        )


def _validate_integration(spec_data: dict) -> None:
    """Validate the integration section."""
    integration = spec_data.get("integration", {})
    if integration:
        if not isinstance(integration.get("memory"), dict):
            raise ValueError("integration.memory must be a dictionary")
        if not isinstance(integration.get("task_queue"), dict):
            raise ValueError("integration.task_queue must be a dictionary")


def _validate_prompts(spec_data: dict) -> None:
    """Validate the prompts section."""
    prompts = spec_data.get("prompts", {})
    if not isinstance(prompts.get("system"), str):
        raise ValueError("prompts.system must be a string")
    if not isinstance(prompts.get("user"), str):
        raise ValueError("prompts.user must be a string")


def _generate_names(agent: dict) -> Tuple[str, str]:
    """Generate agent name and class name from agent info."""
    agent_name = agent["name"].replace("-", "_")
    base_class_name = agent_name.title().replace("_", "")
    class_name = (
        base_class_name
        if base_class_name.endswith("Agent")
        else base_class_name + "Agent"
    )
    return agent_name, class_name


def validate_with_json_schema(spec_data: dict, schema_path: str) -> None:
    """Validate spec data against a JSON schema."""
    try:
        with open(schema_path) as f:
            schema = json.load(f)
    except FileNotFoundError:
        log.warning(
            f"Schema file not found at {schema_path}, skipping schema validation."
        )
        return
    except json.JSONDecodeError:
        log.warning(
            f"Invalid JSON in schema file at {schema_path}, skipping schema validation."
        )
        return

    try:
        validate(instance=spec_data, schema=schema)
    except (ValidationError, SchemaError) as e:
        raise ValueError(f"Spec validation failed: {e.message}")


def validate_spec(spec_data: dict) -> Tuple[str, str]:
    """Validate the Open Agent Spec structure and return agent name and class name.

    Args:
        spec_data: The parsed YAML spec data

    Returns:
        Tuple of (agent_name, class_name)

    Raises:
        KeyError: If required fields are missing
        ValueError: If field types are invalid
    """
    try:
        _validate_version(spec_data)
        _validate_agent(spec_data)
        _validate_behavioural_contract(spec_data)
        _validate_tools(spec_data)
        _validate_tasks(spec_data)
        _validate_integration(spec_data)
        _validate_prompts(spec_data)

        return _generate_names(spec_data["agent"])

    except KeyError as e:
        raise KeyError(f"Missing required field in spec: {e}") from e
    except Exception as e:
        raise ValueError(f"Invalid spec format: {e}") from e
