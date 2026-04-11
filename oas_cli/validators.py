# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""Validation functions for Open Agent Spec."""

import json
import logging

from jsonschema import validate
from jsonschema.exceptions import SchemaError, ValidationError

log = logging.getLogger(__name__)


def _validate_version(spec_data: dict) -> None:
    """Validate the spec version."""
    version = spec_data.get("open_agent_spec")
    if not isinstance(version, str):
        actual_type = type(version).__name__ if version is not None else "missing"
        raise ValueError(
            f"Field 'open_agent_spec' must be a string, got {actual_type}. "
            f'Example: open_agent_spec: "1.0"'
        )
    if not version:
        raise ValueError(
            "Field 'open_agent_spec' cannot be empty. "
            'Provide a valid version string, e.g., "1.0"'
        )


def _validate_agent(spec_data: dict) -> None:
    """Validate the agent section."""
    if "agent" not in spec_data:
        raise ValueError(
            "Missing required section 'agent'. "
            'Add:\n  agent:\n    name: "your-agent-name"\n    role: "agent description"'
        )

    agent = spec_data.get("agent", {})

    if not isinstance(agent.get("name"), str):
        actual_type = type(agent.get("name")).__name__ if "name" in agent else "missing"
        raise ValueError(
            f"Field 'agent.name' must be a string, got {actual_type}. "
            f'Example: agent:\n  name: "my-agent"'
        )
    if not isinstance(agent.get("role"), str):
        actual_type = type(agent.get("role")).__name__ if "role" in agent else "missing"
        raise ValueError(
            f"Field 'agent.role' must be a string, got {actual_type}. "
            f'Example: agent:\n  role: "A helpful assistant"'
        )


def _validate_single_contract(contract: dict, location: str) -> None:
    """Validate one behavioural_contract block at the given spec location.

    ``location`` is used in error messages, e.g. ``'behavioural_contract'`` or
    ``"tasks.summarize.behavioural_contract"``.
    """
    if not isinstance(contract, dict):
        raise ValueError(
            f"Field '{location}' must be a dictionary (object), "
            f"got {type(contract).__name__}."
        )

    if not isinstance(contract.get("version"), str):
        actual = (
            type(contract["version"]).__name__ if "version" in contract else "missing"
        )
        raise ValueError(
            f"Field '{location}.version' must be a string, got {actual}. "
            f'Example: version: "1.0"'
        )
    if not isinstance(contract.get("description"), str):
        actual = (
            type(contract["description"]).__name__
            if "description" in contract
            else "missing"
        )
        raise ValueError(
            f"Field '{location}.description' must be a string, got {actual}. "
            f"Provide a description of the task's behaviour."
        )

    for opt_field in (
        "behavioural_flags",
        "response_contract",
        "policy",
        "teardown_policy",
    ):
        if opt_field in contract and not isinstance(contract[opt_field], dict):
            raise ValueError(
                f"Field '{location}.{opt_field}' must be a dictionary (object), "
                f"got {type(contract[opt_field]).__name__}."
            )


def _validate_behavioural_contract(spec_data: dict) -> None:
    """Validate all behavioural_contract blocks in the spec.

    Checks the optional top-level block and any per-task blocks declared under
    ``tasks.<name>.behavioural_contract``.  Missing contracts are fine — they
    are optional at every level.
    """
    # Top-level contract
    global_contract = spec_data.get("behavioural_contract")
    if global_contract is not None:
        _validate_single_contract(global_contract, "behavioural_contract")

    # Per-task contracts
    tasks = spec_data.get("tasks") or {}
    if not isinstance(tasks, dict):
        return
    for task_name, task_def in tasks.items():
        if not isinstance(task_def, dict):
            continue
        task_contract = task_def.get("behavioural_contract")
        if task_contract is not None:
            _validate_single_contract(
                task_contract, f"tasks.{task_name}.behavioural_contract"
            )


_VALID_TOOL_TYPES = {"native", "mcp", "custom"}
_VALID_NATIVE_IDS = {"file.read", "file.write", "http.get", "http.post", "env.read"}


def _validate_tools(spec_data: dict) -> None:
    """Validate the top-level tools: block (dict of named tool declarations)."""
    tools = spec_data.get("tools")
    if tools is None:
        return

    if not isinstance(tools, dict):
        actual_type = type(tools).__name__
        raise ValueError(
            f"Field 'tools' must be a dictionary (object), got {actual_type}. "
            "Example:\n  tools:\n    read_file:\n      type: native\n"
            "      native: file.read"
        )

    for tool_name, tool_cfg in tools.items():
        if not isinstance(tool_cfg, dict):
            raise ValueError(
                f"tools.{tool_name} must be a dictionary (object), "
                f"got {type(tool_cfg).__name__}."
            )

        tool_type = tool_cfg.get("type")
        if tool_type not in _VALID_TOOL_TYPES:
            raise ValueError(
                f"tools.{tool_name}.type must be one of "
                f"{sorted(_VALID_TOOL_TYPES)}, got '{tool_type}'."
            )

        if tool_type == "native":
            native_id = tool_cfg.get("native")
            if not native_id:
                raise ValueError(
                    f"tools.{tool_name} has type 'native' but is missing the "
                    f"'native' field. Available: {sorted(_VALID_NATIVE_IDS)}"
                )
            if native_id not in _VALID_NATIVE_IDS:
                raise ValueError(
                    f"tools.{tool_name}.native '{native_id}' is not a recognised "
                    f"native tool. Available: {sorted(_VALID_NATIVE_IDS)}"
                )

        if tool_type == "mcp" and not tool_cfg.get("endpoint"):
            raise ValueError(
                f"tools.{tool_name} has type 'mcp' but is missing the "
                "'endpoint' field (e.g. endpoint: http://localhost:3000)."
            )

        if tool_type == "custom" and not tool_cfg.get("module"):
            raise ValueError(
                f"tools.{tool_name} has type 'custom' but is missing the "
                "'module' field (e.g. module: my_package.MyTool)."
            )


def _validate_tasks(spec_data: dict) -> None:
    """Validate the tasks section."""
    tasks = spec_data.get("tasks", {})
    tools = spec_data.get("tools") or {}
    # Support both the new dict format and the legacy list format gracefully.
    if isinstance(tools, dict):
        tool_ids = list(tools.keys())
    else:
        tool_ids = [t["id"] for t in tools if isinstance(t, dict) and "id" in t]

    if not isinstance(tasks, dict):
        actual_type = type(tasks).__name__
        raise ValueError(
            f"Field 'tasks' must be a dictionary (object), "
            f"got {actual_type}. "
            "Example:\n  tasks:\n    task1:\n"
            "      input: {}\n      output: {}"
        )

    for task_name, task_def in tasks.items():
        if not isinstance(task_def, dict):
            actual_type = type(task_def).__name__
            raise ValueError(
                f"tasks.{task_name} must be a dictionary (object), "
                f"got {actual_type}. "
                "Each task should have 'input' and 'output' "
                "definitions."
            )

        # Check if this task uses a tool (legacy single string)
        if "tool" in task_def:
            tool_id = task_def["tool"]
            if not isinstance(tool_id, str):
                actual_type = type(tool_id).__name__
                raise ValueError(
                    f"tasks.{task_name}.tool must be a string, got {actual_type}."
                )
            if tool_ids and tool_id not in tool_ids:
                available = ", ".join(f"'{tid}'" for tid in tool_ids)
                raise ValueError(
                    f"tasks.{task_name} references non-existent "
                    f"tool '{tool_id}'. "
                    f"Available tools: {available}. "
                    "Check your 'tools' section."
                )

        # Check if this task uses tools (new list format)
        if "tools" in task_def:
            task_tools = task_def["tools"]
            if not isinstance(task_tools, list):
                raise ValueError(
                    f"tasks.{task_name}.tools must be a list of tool names, "
                    f"got {type(task_tools).__name__}."
                )
            for ref in task_tools:
                if not isinstance(ref, str):
                    raise ValueError(
                        f"tasks.{task_name}.tools entries must be strings, "
                        f"got {type(ref).__name__}."
                    )
                if tool_ids and ref not in tool_ids:
                    available = ", ".join(f"'{tid}'" for tid in tool_ids)
                    raise ValueError(
                        f"tasks.{task_name}.tools references undeclared "
                        f"tool '{ref}'. "
                        f"Available tools: {available}. "
                        "Check your top-level 'tools:' section."
                    )

        # Check if this is a multi-step task
        is_multi_step = task_def.get("multi_step", False)

        # For multi-step tasks, input and output are optional
        if not is_multi_step:
            if not isinstance(task_def.get("input"), dict):
                actual_type = (
                    type(task_def.get("input")).__name__
                    if "input" in task_def
                    else "missing"
                )
                raise ValueError(
                    f"tasks.{task_name}.input must be a dictionary "
                    f"(object), got {actual_type}. "
                    "Define input schema, e.g., "
                    "input: {query: {type: string}}"
                )
            if not isinstance(task_def.get("output"), dict):
                actual_type = (
                    type(task_def.get("output")).__name__
                    if "output" in task_def
                    else "missing"
                )
                raise ValueError(
                    f"tasks.{task_name}.output must be a dictionary "
                    f"(object), got {actual_type}. "
                    "Define output schema, e.g., "
                    "output: {result: {type: string}}"
                )
            # Require output to have non-empty 'required' when it has 'properties'
            output_schema = task_def["output"]
            props = output_schema.get("properties")
            if isinstance(props, dict) and props:
                req = output_schema.get("required")
                if not isinstance(req, list) or len(req) == 0:
                    raise ValueError(
                        f"tasks.{task_name}.output has 'properties' but missing or "
                        "empty 'required'. Specify which output properties are "
                        "required, e.g. required: [response]"
                    )
                for r in req:
                    if r not in props:
                        raise ValueError(
                            f"tasks.{task_name}.output.required references '{r}' "
                            "which is not in output.properties"
                        )
        else:
            # For multi-step tasks, validate that steps are defined
            if not isinstance(task_def.get("steps"), list):
                actual_type = (
                    type(task_def.get("steps")).__name__
                    if "steps" in task_def
                    else "missing"
                )
                raise ValueError(
                    f"Multi-step tasks.{task_name}.steps must be "
                    f"a list (array), got {actual_type}. "
                    "Example:\n  steps:\n"
                    "    - task: step1\n    - task: step2"
                )
            if not task_def.get("steps"):
                raise ValueError(f"Multi-step task {task_name}.steps cannot be empty")

            # Validate output schema for multi-step tasks
            if not isinstance(task_def.get("output"), dict):
                actual_type = (
                    type(task_def.get("output")).__name__
                    if "output" in task_def
                    else "missing"
                )
                raise ValueError(
                    f"Multi-step task {task_name}.output must be "
                    f"a dictionary (object), got {actual_type}."
                )
            if not task_def.get("output"):
                raise ValueError(f"Multi-step task {task_name}.output cannot be empty")

            # Validate each step
            for i, step in enumerate(task_def["steps"]):
                if not isinstance(step, dict):
                    raise ValueError(
                        f"steps[{i}] in task {task_name} must be a dictionary (object)."
                    )
                if "task" not in step:
                    raise ValueError(
                        f"steps[{i}] in task {task_name} must have a 'task' field."
                    )
                if not isinstance(step["task"], str):
                    actual_type = type(step["task"]).__name__
                    raise ValueError(
                        f"steps[{i}] in task {task_name}.task "
                        f"must be a string, got {actual_type}."
                    )

                # Check that the referenced task exists
                referenced_task = step["task"]
                if referenced_task not in tasks:
                    available_tasks = ", ".join(f"'{t}'" for t in tasks)
                    raise ValueError(
                        f"steps[{i}] in task {task_name} "
                        f"references non-existent task "
                        f"'{referenced_task}'. "
                        f"Available tasks: {available_tasks}."
                    )

                # Validate input_map if present
                if "input_map" in step:
                    if not isinstance(step["input_map"], dict):
                        actual_type = type(step["input_map"]).__name__
                        raise ValueError(
                            f"steps[{i}] in task "
                            f"{task_name}.input_map must be "
                            f"a dictionary (object), "
                            f"got {actual_type}."
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
    """Validate the prompts section (global fallback — optional)."""
    prompts = spec_data.get("prompts")
    if not prompts:
        return
    if not isinstance(prompts, dict):
        raise ValueError(
            f"Field 'prompts' must be a dictionary, got {type(prompts).__name__}."
        )
    if "system" in prompts and not isinstance(prompts["system"], str):
        raise ValueError("prompts.system must be a string")
    if "user" in prompts and not isinstance(prompts["user"], str):
        raise ValueError("prompts.user must be a string")


def _generate_names(agent: dict) -> tuple[str, str]:
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
        raise ValueError(f"Spec validation failed: {e!s}")


def validate_spec(spec_data: dict) -> tuple[str, str]:
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
