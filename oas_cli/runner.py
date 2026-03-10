"""High-level runner for executing Open Agent Spec directly from YAML.

This allows using a spec as "infra as code": load the spec, choose a task,
build a prompt from the configured prompts and inputs, and invoke the
underlying intelligence provider via the runtime abstraction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from .runtime import invoke_intelligence


def _load_spec(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Spec YAML must decode to a mapping/object")
    return data


def _choose_task(spec_data: Dict[str, Any], task_name: str | None) -> Tuple[str, Dict[str, Any]]:
    tasks = spec_data.get("tasks") or {}
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("Spec has no tasks defined")

    if task_name:
        task = tasks.get(task_name)
        if not task:
            raise ValueError(f"Task '{task_name}' not found in spec")
        return task_name, task

    # Default: first non-multi-step task, falling back to the first task.
    for name, task_def in tasks.items():
        if not isinstance(task_def, dict):
            continue
        if not task_def.get("multi_step", False):
            return name, task_def

    # If all tasks are multi-step, just pick the first one.
    first_name = next(iter(tasks))
    return first_name, tasks[first_name]


def _build_prompt(spec_data: Dict[str, Any], task_name: str, input_data: Dict[str, Any]) -> str:
    prompts = spec_data.get("prompts") or {}
    task_prompts = prompts.get(task_name) if isinstance(prompts.get(task_name), dict) else {}

    system = ""
    if isinstance(task_prompts, dict):
        system = task_prompts.get("system") or ""
    if not system:
        system = prompts.get("system") or ""

    user_template = ""
    if isinstance(task_prompts, dict):
        user_template = task_prompts.get("user") or ""
    if not user_template:
        user_template = prompts.get("user") or "{{ name }}"

    user = user_template
    for key, value in input_data.items():
        placeholder = "{{ " + key + " }}"
        user = user.replace(placeholder, str(value))
        placeholder_input = "{{ input." + key + " }}"
        user = user.replace(placeholder_input, str(value))

    return f"{system}\n\n{user}".strip()


def _build_intelligence_config(spec_data: Dict[str, Any]) -> Dict[str, Any]:
    intelligence = spec_data.get("intelligence") or {}
    if not isinstance(intelligence, dict):
        raise ValueError("intelligence section must be an object")

    endpoint = intelligence.get("endpoint", "https://api.openai.com/v1")
    model = intelligence.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("intelligence.model must be a non-empty string")

    cfg = intelligence.get("config") or {}
    if not isinstance(cfg, dict):
        cfg = {}

    return {
        "engine": intelligence.get("engine", "openai"),
        "model": model,
        "endpoint": endpoint,
        "temperature": cfg.get("temperature", 0.7),
        "max_tokens": cfg.get("max_tokens", 1000),
    }


def run_task_from_spec(
    spec_data: Dict[str, Any],
    task_name: str | None = None,
    input_data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run a single task defined in the spec and return a simple result dict.

    This makes no assumptions about the surrounding orchestration; it just
    builds a prompt and invokes the model via the runtime abstraction.
    """
    input_payload: Dict[str, Any] = dict(input_data or {})
    chosen_task, _ = _choose_task(spec_data, task_name)
    prompt = _build_prompt(spec_data, chosen_task, input_payload)
    intelligence_config = _build_intelligence_config(spec_data)

    raw_output = invoke_intelligence(prompt, intelligence_config)

    # Best-effort attempt to normalise JSON responses if they come back as strings.
    parsed_output: Any
    if isinstance(raw_output, str):
        try:
            parsed_output = json.loads(raw_output)
        except json.JSONDecodeError:
            parsed_output = raw_output
    else:
        parsed_output = raw_output

    return {
        "task": chosen_task,
        "input": input_payload,
        "prompt": prompt,
        "engine": intelligence_config.get("engine"),
        "model": intelligence_config.get("model"),
        "raw_output": raw_output,
        "output": parsed_output,
    }


def run_task_from_file(
    spec_path: Path,
    task_name: str | None = None,
    input_data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Convenience wrapper to load a spec from disk and run a task."""
    spec = _load_spec(spec_path)
    return run_task_from_spec(spec, task_name=task_name, input_data=input_data)


