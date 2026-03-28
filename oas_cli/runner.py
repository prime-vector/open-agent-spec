"""High-level runner for executing Open Agent Spec directly from YAML.

This allows using a spec as "infra as code": load the spec, choose a task,
build a prompt from the configured prompts and inputs, and invoke the
underlying intelligence provider via the runtime abstraction.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from .runtime import invoke_intelligence

logger = logging.getLogger(__name__)

# Optional BCE integration — degrades gracefully when library is not installed.
try:
    from behavioural_contracts import validate_task_output  # type: ignore[import]

    CONTRACTS_ENABLED = True
except ImportError:
    CONTRACTS_ENABLED = False


class OARunError(Exception):
    """Structured run-time error that carries machine-readable metadata.

    Attributes:
        message: Human-readable description.
        code:    Machine-readable error code (e.g. TASK_NOT_FOUND).
        stage:   Pipeline stage where the error occurred.
        task:    Task name involved, if known.
    """

    def __init__(
        self,
        message: str,
        code: str,
        stage: str,
        task: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.task = task

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "error": str(self),
            "code": self.code,
            "stage": self.stage,
        }
        if self.task is not None:
            d["task"] = self.task
        return d


def _load_spec(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError as exc:
        raise OARunError(
            f"Spec file not found: {path}",
            code="SPEC_LOAD_ERROR",
            stage="load",
        ) from exc
    except yaml.YAMLError as exc:
        raise OARunError(
            f"Invalid YAML in spec: {exc}",
            code="SPEC_LOAD_ERROR",
            stage="load",
        ) from exc
    if not isinstance(data, dict):
        raise OARunError(
            "Spec YAML must decode to a mapping/object",
            code="SPEC_LOAD_ERROR",
            stage="load",
        )
    return data


def _choose_task(
    spec_data: dict[str, Any], task_name: str | None
) -> tuple[str, dict[str, Any]]:
    tasks = spec_data.get("tasks") or {}
    if not isinstance(tasks, dict) or not tasks:
        raise OARunError(
            "Spec has no tasks defined",
            code="SPEC_LOAD_ERROR",
            stage="routing",
        )

    if task_name:
        task = tasks.get(task_name)
        if not task:
            raise OARunError(
                f"Task '{task_name}' not found in spec",
                code="TASK_NOT_FOUND",
                stage="routing",
                task=task_name,
            )
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


def _build_prompt(
    spec_data: dict[str, Any],
    task_name: str,
    input_data: dict[str, Any],
    override_system: str | None = None,
    override_user: str | None = None,
) -> str:
    """Build the prompt for a task using a layered resolution order.

    Priority (highest → lowest):
      1. CLI overrides  (override_system / override_user)
      2. Per-task inline prompts  tasks.<name>.prompts  (Style A — preferred)
      3. Per-task map prompts     prompts.<name>.system/user  (Style B — legacy)
      4. Global fallback          prompts.system / prompts.user
    """
    # Style A — prompts block co-located inside the task definition
    tasks = spec_data.get("tasks") or {}
    task_def = tasks.get(task_name) or {}
    inline: dict[str, Any] = task_def.get("prompts") or {}

    # Style B — keyed sub-object under the global prompts map (existing behaviour)
    prompts = spec_data.get("prompts") or {}
    mapped: dict[str, Any] = {}
    if isinstance(prompts.get(task_name), dict):
        mapped = prompts[task_name]

    # Resolve system prompt
    if override_system is not None:
        system = override_system
    elif inline.get("system"):
        system = str(inline["system"])
    elif mapped.get("system"):
        system = str(mapped["system"])
    else:
        system = prompts.get("system") or ""

    # Resolve user template
    if override_user is not None:
        user_template = override_user
    elif inline.get("user"):
        user_template = str(inline["user"])
    elif mapped.get("user"):
        user_template = str(mapped["user"])
    else:
        user_template = prompts.get("user") or "{{ input }}"

    user = user_template
    for key, value in input_data.items():
        placeholder = "{{ " + key + " }}"
        user = user.replace(placeholder, str(value))
        placeholder_input = "{{ input." + key + " }}"
        user = user.replace(placeholder_input, str(value))

    return f"{system}\n\n{user}".strip()


def _build_intelligence_config(spec_data: dict[str, Any]) -> dict[str, Any]:
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

    # Base keys for OpenAI-style calls; merge cfg so engines like codex get
    # sandbox, cwd, extra_args, etc. from intelligence.config.
    out: dict[str, Any] = {
        "engine": intelligence.get("engine", "openai"),
        "model": model,
        "endpoint": endpoint,
        "temperature": cfg.get("temperature", 0.7),
        "max_tokens": cfg.get("max_tokens", 1000),
    }
    for k, v in cfg.items():
        if k not in out:
            out[k] = v
    return out


def _resolve_chain(
    spec_data: dict[str, Any],
    task_name: str,
    base_input: dict[str, Any],
    override_system: str | None,
    override_user: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve depends_on chain and return (merged_input, chain_results).

    Executes prerequisite tasks in the order listed, merging their outputs into
    the input for the dependent task.  Previous task output wins on key collision:
        merged = {**base_input, **dep1_output, **dep2_output, ...}
    """
    tasks = spec_data.get("tasks") or {}
    task_def = tasks.get(task_name) or {}
    deps: list[str] = task_def.get("depends_on") or []

    if not deps:
        return dict(base_input), {}

    # Cycle detection: walk the full dependency graph breadth-first.
    seen: set[str] = {task_name}

    def _check_cycles(name: str) -> None:
        for dep in (tasks.get(name) or {}).get("depends_on") or []:
            if dep in seen:
                raise OARunError(
                    f"Circular dependency detected: '{dep}' is already in the chain",
                    code="CHAIN_CYCLE_ERROR",
                    stage="routing",
                    task=name,
                )
            seen.add(dep)
            _check_cycles(dep)

    _check_cycles(task_name)

    merged: dict[str, Any] = dict(base_input)
    chain: dict[str, Any] = {}

    for dep_name in deps:
        if dep_name not in tasks:
            raise OARunError(
                f"depends_on references unknown task '{dep_name}'",
                code="TASK_NOT_FOUND",
                stage="routing",
                task=dep_name,
            )
        dep_result = _run_single_task(
            spec_data,
            dep_name,
            dict(merged),
            override_system=None,
            override_user=None,
        )
        chain[dep_name] = dep_result
        dep_output = dep_result.get("output") or {}
        if isinstance(dep_output, dict):
            merged.update(dep_output)

    return merged, chain


def _merge_contracts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge two behavioural contract dicts.

    Merge rules:
    - Arrays: unioned (order preserved, duplicates removed)
    - Dicts:  merged recursively using the same rules
    - Scalars: override wins

    This ensures global constraints (e.g. required_fields: [confidence]) are
    preserved when a task adds its own fields, rather than silently dropped.
    """
    merged = dict(base)
    for key, val in override.items():
        if key in merged:
            base_val = merged[key]
            if isinstance(base_val, list) and isinstance(val, list):
                merged[key] = list(dict.fromkeys(base_val + val))
            elif isinstance(base_val, dict) and isinstance(val, dict):
                merged[key] = _merge_contracts(base_val, val)
            else:
                merged[key] = val
        else:
            merged[key] = val
    return merged


def _resolve_contract(
    spec_data: dict[str, Any], task_name: str
) -> dict[str, Any] | None:
    """Resolve the effective behavioural contract for a task.

    Priority:
      global behavioural_contract  ← base
      tasks.<name>.behavioural_contract  ← merged on top (arrays unioned)

    Returns None when no contract is declared at either level.
    """
    global_contract: dict[str, Any] = spec_data.get("behavioural_contract") or {}
    tasks = spec_data.get("tasks") or {}
    task_contract: dict[str, Any] = (tasks.get(task_name) or {}).get(
        "behavioural_contract"
    ) or {}
    if not global_contract and not task_contract:
        return None
    return _merge_contracts(global_contract, task_contract)


def _run_single_task(
    spec_data: dict[str, Any],
    task_name: str,
    input_data: dict[str, Any],
    override_system: str | None,
    override_user: str | None,
) -> dict[str, Any]:
    """Execute one task (no chain resolution) and return the result envelope."""
    tasks = spec_data.get("tasks") or {}
    task_def = tasks.get(task_name) or {}

    # Validate required inputs before touching the model.
    inp_schema = task_def.get("input") or {}
    required_fields: list[str] = inp_schema.get("required") or []
    missing = [f for f in required_fields if f not in input_data]
    if missing:
        raise OARunError(
            f"Missing required input field(s) for task '{task_name}': {', '.join(missing)}",
            code="CHAIN_INPUT_MISSING",
            stage="input_validation",
            task=task_name,
        )

    prompt = _build_prompt(
        spec_data,
        task_name,
        input_data,
        override_system=override_system,
        override_user=override_user,
    )
    intelligence_config = _build_intelligence_config(spec_data)

    try:
        raw_output = invoke_intelligence(prompt, intelligence_config)
    except Exception as exc:
        raise OARunError(
            str(exc),
            code="RUN_ERROR",
            stage="run",
            task=task_name,
        ) from exc

    # response_format: text → skip JSON parsing entirely.
    response_format = task_def.get("response_format", "json")
    parsed_output: Any
    if response_format == "text":
        parsed_output = raw_output
    elif isinstance(raw_output, str):
        text = raw_output.strip()
        # Models often wrap JSON in ```json ... ``` — strip fences before parsing.
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
            text = re.sub(r"\s*```\s*$", "", text, count=1)
            text = text.strip()
        try:
            parsed_output = json.loads(text)
        except json.JSONDecodeError:
            parsed_output = raw_output
    else:
        parsed_output = raw_output

    # Behavioural contract validation — AFTER parsing, BEFORE returning.
    # Runs for every task including chain dependencies, so a bad dep output is
    # caught before it can be merged into the next task's input.
    contract = _resolve_contract(spec_data, task_name)
    if contract is not None:
        if response_format == "text":
            logger.warning(
                "[warning] Contract validation skipped for task '%s': "
                "response_format is 'text' — field validation is meaningless on raw strings.",
                task_name,
            )
        elif not isinstance(parsed_output, dict):
            logger.warning(
                "[warning] Contract validation skipped for task '%s': "
                "output could not be parsed as a dict.",
                task_name,
            )
        elif not CONTRACTS_ENABLED:
            logger.warning(
                "[warning] behavioural-contracts not installed — "
                "contract validation for task '%s' will be skipped. "
                "Install with: pip install 'open-agent-spec[contracts]'",
                task_name,
            )
        else:
            try:
                validate_task_output(parsed_output, contract)
            except Exception as exc:
                raise OARunError(
                    str(exc),
                    code="CONTRACT_VIOLATION",
                    stage="contract",
                    task=task_name,
                ) from exc

    return {
        "task": task_name,
        "input": input_data,
        "prompt": prompt,
        "engine": intelligence_config.get("engine"),
        "model": intelligence_config.get("model"),
        "raw_output": raw_output,
        "output": parsed_output,
    }


def run_task_from_spec(
    spec_data: dict[str, Any],
    task_name: str | None = None,
    input_data: dict[str, Any] | None = None,
    override_system: str | None = None,
    override_user: str | None = None,
) -> dict[str, Any]:
    """Run a task defined in the spec, resolving any depends_on chain first.

    Returns a result envelope.  When the task has dependencies the envelope
    includes a ``chain`` key with intermediate results keyed by task name.

    override_system / override_user replace the resolved spec prompt for this
    invocation only — useful for targeted one-off instructions from the CLI.
    """
    base_input: dict[str, Any] = dict(input_data or {})
    chosen_task, _ = _choose_task(spec_data, task_name)

    merged_input, chain = _resolve_chain(
        spec_data, chosen_task, base_input, override_system, override_user
    )

    result = _run_single_task(
        spec_data,
        chosen_task,
        merged_input,
        override_system=override_system,
        override_user=override_user,
    )

    if chain:
        result["chain"] = chain

    return result


def run_task_from_file(
    spec_path: Path,
    task_name: str | None = None,
    input_data: dict[str, Any] | None = None,
    override_system: str | None = None,
    override_user: str | None = None,
) -> dict[str, Any]:
    """Convenience wrapper to load a spec from disk and run a task."""
    spec = _load_spec(spec_path)
    return run_task_from_spec(
        spec,
        task_name=task_name,
        input_data=input_data,
        override_system=override_system,
        override_user=override_user,
    )
