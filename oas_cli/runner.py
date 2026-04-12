"""High-level runner for executing Open Agent Spec directly from YAML.

This allows using a spec as "infra as code": load the spec, choose a task,
build a prompt from the configured prompts and inputs, and invoke the
underlying intelligence provider via the runtime abstraction.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from .providers import ProviderError, invoke_intelligence
from .providers.registry import get_provider
from .tool_providers import (
    ToolError,
    dispatch_tool_call,
    resolve_task_tools,
)
from .tool_providers.base import InvokeResult

# ── Registry constants ────────────────────────────────────────────────────────
_REGISTRY_BASE = "https://openagentspec.dev/registry"
_OA_SCHEME = "oa://"

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


def _resolve_spec_url(ref: str) -> str:
    """Expand an ``oa://`` shorthand into a full HTTPS registry URL.

    Supported formats:
        oa://namespace/name            → registry/.../latest/spec.yaml
        oa://namespace/name@1.0.0      → registry/.../1.0.0/spec.yaml
        https://...                    → returned as-is
        http://...                     → returned as-is

    Local paths are not handled here — caller checks ``_is_remote_ref`` first.
    """
    if not ref.startswith(_OA_SCHEME):
        return ref  # already a plain HTTP(S) URL

    rest = ref[len(_OA_SCHEME) :]
    version = "latest"
    if "@" in rest:
        rest, version = rest.rsplit("@", 1)

    parts = rest.strip("/").split("/")
    if len(parts) != 2:
        raise OARunError(
            f"Invalid oa:// reference '{ref}'. "
            "Expected format: oa://namespace/name or oa://namespace/name@version",
            code="SPEC_LOAD_ERROR",
            stage="delegation",
        )

    namespace, name = parts
    url = f"{_REGISTRY_BASE}/{namespace}/{name}/{version}/spec.yaml"
    logger.debug("[registry] resolved %s → %s", ref, url)
    return url


def _is_remote_ref(ref: str) -> bool:
    """Return True when *ref* should be fetched over HTTP rather than disk."""
    return (
        ref.startswith(_OA_SCHEME)
        or ref.startswith("http://")
        or ref.startswith("https://")
    )


def _fetch_remote_spec(url: str) -> dict[str, Any]:
    """Download a spec YAML from a URL and return the parsed dict."""
    logger.debug("[registry] fetching %s", url)
    try:
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/yaml, text/yaml, text/plain, */*"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise OARunError(
            f"Registry fetch failed for '{url}': HTTP {exc.code} {exc.reason}",
            code="SPEC_LOAD_ERROR",
            stage="delegation",
        ) from exc
    except urllib.error.URLError as exc:
        raise OARunError(
            f"Registry fetch failed for '{url}': {exc.reason}",
            code="SPEC_LOAD_ERROR",
            stage="delegation",
        ) from exc

    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise OARunError(
            f"Invalid YAML from registry '{url}': {exc}",
            code="SPEC_LOAD_ERROR",
            stage="delegation",
        ) from exc

    if not isinstance(data, dict):
        raise OARunError(
            f"Registry spec at '{url}' must decode to a mapping/object",
            code="SPEC_LOAD_ERROR",
            stage="delegation",
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


def _resolve_prompts(
    spec_data: dict[str, Any],
    task_name: str,
    input_data: dict[str, Any],
    override_system: str | None = None,
    override_user: str | None = None,
) -> tuple[str, str]:
    """Resolve the system and user prompts for a task (layered resolution).

    Priority (highest → lowest):
      1. CLI overrides  (override_system / override_user)
      2. Per-task inline prompts  tasks.<name>.prompts  (Style A — preferred)
      3. Per-task map prompts     prompts.<name>.system/user  (Style B — legacy)
      4. Global fallback          prompts.system / prompts.user

    Returns:
        (system, user) — both are plain strings, user already rendered.
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
        str_val = str(value)
        # Single-brace Python style: {key}
        user = user.replace(f"{{{key}}}", str_val)
        # Double-brace Jinja style: {{ key }} and {{ input.key }}
        user = user.replace("{{ " + key + " }}", str_val)
        user = user.replace("{{ input." + key + " }}", str_val)

    return system, user


def _build_prompt(
    spec_data: dict[str, Any],
    task_name: str,
    input_data: dict[str, Any],
    override_system: str | None = None,
    override_user: str | None = None,
) -> str:
    """Return system and user merged into a single string (for backward compat).

    Prefer ``_resolve_prompts`` for new code — it keeps the roles separate so
    providers can pass them to the model API individually.
    """
    system, user = _resolve_prompts(
        spec_data, task_name, input_data, override_system, override_user
    )
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
    *,
    spec_path: Path | None = None,
    _visited_specs: frozenset[Path] = frozenset(),
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
            spec_path=spec_path,
            _visited_specs=_visited_specs,
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


_MAX_TOOL_ITERATIONS = 10


def _invoke_with_tools(
    system: str,
    user: str,
    tools: list,
    intelligence_config: dict[str, Any],
    task_name: str,
) -> str:
    """Multi-turn tool-call loop.

    Builds the initial message list, asks the provider to invoke (potentially
    returning tool calls), executes those calls via the registered tool providers,
    feeds results back, and repeats until the model returns a final answer or the
    iteration cap is hit.

    Falls back to a single ``invoke_intelligence`` call when the selected provider
    does not support tool use natively (e.g. Codex adapter).
    """
    provider = get_provider(intelligence_config)

    tool_defs = [defn.to_openai_schema() for _, defn in tools]

    if not provider.supports_tools():
        # Inject tool descriptions into the system prompt for text-only providers.
        tool_descriptions = "\n".join(
            f"- {defn.name}: {defn.description}" for _, defn in tools
        )
        augmented_system = (
            f"{system}\n\nYou have access to the following tools:\n{tool_descriptions}"
        )
        return invoke_intelligence(augmented_system, user, intelligence_config)

    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]

    for iteration in range(_MAX_TOOL_ITERATIONS):
        result: InvokeResult = provider.invoke_with_tools(
            system=system,
            messages=messages,
            tools=tool_defs,
            config=intelligence_config,
        )

        if result.is_final:
            return result.text

        if not result.tool_calls:
            logger.warning(
                "[tools] Provider returned is_final=False with no tool_calls on task '%s' "
                "(iteration %d). Treating as final with empty response.",
                task_name,
                iteration,
            )
            return ""

        # Append the assistant's tool-call request to the history.
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in result.tool_calls
                ],
            }
        )

        # Execute every tool call and append results.
        for tc in result.tool_calls:
            try:
                tool_result = dispatch_tool_call(tc.name, tc.arguments, tools)
            except ToolError as exc:
                tool_result = f"[tool error] {exc}"
                logger.warning("[tools] Tool '%s' raised: %s", tc.name, exc)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                }
            )

    raise OARunError(
        f"Tool-call loop for task '{task_name}' exceeded {_MAX_TOOL_ITERATIONS} iterations.",
        code="RUN_ERROR",
        stage="run",
        task=task_name,
    )


def _run_single_task(
    spec_data: dict[str, Any],
    task_name: str,
    input_data: dict[str, Any],
    override_system: str | None,
    override_user: str | None,
    *,
    spec_path: Path | None = None,
    _visited_specs: frozenset[Path] = frozenset(),
) -> dict[str, Any]:
    """Execute one task (no chain resolution) and return the result envelope.

    If the task declares ``spec:`` + ``task:`` it is a *delegated* task — the
    runner loads the referenced spec and executes that task transparently,
    returning the result as if it had been defined inline.
    """
    tasks = spec_data.get("tasks") or {}
    task_def = tasks.get(task_name) or {}

    # ── Spec delegation ───────────────────────────────────────────────────
    delegation_spec_ref: str | None = task_def.get("spec")
    if delegation_spec_ref is not None:
        if not isinstance(delegation_spec_ref, str) or not delegation_spec_ref.strip():
            raise OARunError(
                f"Task '{task_name}': 'spec' must be a non-empty string path",
                code="SPEC_LOAD_ERROR",
                stage="delegation",
                task=task_name,
            )

        raw_ref = delegation_spec_ref.strip()

        # ── Remote spec (oa:// or https://) ──────────────────────────────
        if _is_remote_ref(raw_ref):
            url = _resolve_spec_url(raw_ref)
            # Use URL string as the cycle-detection key.
            canonical_key: Any = url
            if canonical_key in _visited_specs:
                raise OARunError(
                    f"Circular spec delegation detected: '{url}' is already in "
                    "the delegation stack",
                    code="DELEGATION_CYCLE_ERROR",
                    stage="delegation",
                    task=task_name,
                )
            delegated_spec = _fetch_remote_spec(url)
            new_visited = _visited_specs | {canonical_key}
            canonical: Any = url  # used for logging + result envelope below

        # ── Local path (relative or absolute) ────────────────────────────
        else:
            ref = Path(raw_ref)
            if not ref.is_absolute() and spec_path is not None:
                ref = (spec_path.parent / ref).resolve()
            else:
                ref = ref.resolve()

            canonical = ref
            if canonical in _visited_specs:
                raise OARunError(
                    f"Circular spec delegation detected: '{canonical}' is already in "
                    "the delegation stack",
                    code="DELEGATION_CYCLE_ERROR",
                    stage="delegation",
                    task=task_name,
                )
            delegated_spec = _load_spec(canonical)
            new_visited = _visited_specs | {canonical}

        # Use the explicitly named task, or fall back to same name as the caller.
        delegated_task: str = task_def.get("task") or task_name

        # Validate the task exists in the referenced spec before going further.
        delegated_tasks = delegated_spec.get("tasks") or {}
        if delegated_task not in delegated_tasks:
            available = ", ".join(f"'{t}'" for t in delegated_tasks) or "none"
            raise OARunError(
                f"Task '{delegated_task}' not found in delegated spec '{canonical}'. "
                f"Available tasks: {available}",
                code="TASK_NOT_FOUND",
                stage="delegation",
                task=task_name,
            )

        logger.debug(
            "[delegation] task '%s' → %s#%s",
            task_name,
            canonical,
            delegated_task,
        )

        # For remote specs there is no local path, so pass None for spec_path.
        # Relative references inside a remote spec will also resolve remotely.
        next_spec_path = canonical if isinstance(canonical, Path) else None

        result = _run_single_task(
            delegated_spec,
            delegated_task,
            input_data,
            override_system,
            override_user,
            spec_path=next_spec_path,
            _visited_specs=new_visited,
        )
        # Surface the coordinator's task name so the envelope is consistent
        # from the caller's perspective.
        result["task"] = task_name
        result["delegated_to"] = f"{canonical}#{delegated_task}"
        return result

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

    system, user = _resolve_prompts(
        spec_data,
        task_name,
        input_data,
        override_system=override_system,
        override_user=override_user,
    )
    intelligence_config = _build_intelligence_config(spec_data)

    try:
        tools = resolve_task_tools(spec_data, task_name)
        if tools:
            raw_output = _invoke_with_tools(
                system, user, tools, intelligence_config, task_name
            )
        else:
            raw_output = invoke_intelligence(system, user, intelligence_config)
    except (ProviderError, ToolError) as exc:
        raise OARunError(
            str(exc),
            code="PROVIDER_ERROR",
            stage="run",
            task=task_name,
        ) from exc
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
        "prompt": f"{system}\n\n{user}".strip(),
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
    *,
    spec_path: Path | None = None,
) -> dict[str, Any]:
    """Run a task defined in the spec, resolving any depends_on chain first.

    Returns a result envelope.  When the task has dependencies the envelope
    includes a ``chain`` key with intermediate results keyed by task name.

    override_system / override_user replace the resolved spec prompt for this
    invocation only — useful for targeted one-off instructions from the CLI.

    spec_path is used to resolve relative ``spec:`` delegation references.
    """
    base_input: dict[str, Any] = dict(input_data or {})
    chosen_task, _ = _choose_task(spec_data, task_name)

    # Seed the visited set with the calling spec so direct self-delegation is caught.
    visited: frozenset[Path] = frozenset()
    if spec_path is not None:
        visited = frozenset({spec_path.resolve()})

    merged_input, chain = _resolve_chain(
        spec_data,
        chosen_task,
        base_input,
        override_system,
        override_user,
        spec_path=spec_path,
        _visited_specs=visited,
    )

    result = _run_single_task(
        spec_data,
        chosen_task,
        merged_input,
        override_system=override_system,
        override_user=override_user,
        spec_path=spec_path,
        _visited_specs=visited,
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
    resolved = spec_path.resolve()
    spec = _load_spec(resolved)
    return run_task_from_spec(
        spec,
        task_name=task_name,
        input_data=input_data,
        override_system=override_system,
        override_user=override_user,
        spec_path=resolved,
    )
