"""Tests for spec delegation / composition.

A task with ``spec: ./other.yaml`` + ``task: some_task`` delegates execution
to that spec's task.  The delegation is transparent to depends_on chains and
returns an envelope consistent with the coordinator's task name.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from oas_cli.runner import OARunError, run_task_from_file

# ── Minimal spec helpers ──────────────────────────────────────────────────────

_INTELLIGENCE = {
    "type": "llm",
    "engine": "openai",
    "model": "gpt-4o-mini",
}


def _make_spec(tasks: dict, prompts: dict | None = None) -> dict:
    return {
        "open_agent_spec": "1.3.0",
        "agent": {"name": "test", "description": "test", "role": "analyst"},
        "intelligence": _INTELLIGENCE,
        "tasks": tasks,
        "prompts": prompts or {"system": "You are helpful.", "user": "Hello"},
    }


def _write_spec(tmp_path: Path, name: str, spec: dict) -> Path:
    import yaml

    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.dump(spec))
    return p


# ── Basic delegation ──────────────────────────────────────────────────────────


class TestBasicDelegation:
    def test_delegated_task_executes_referenced_spec(self, tmp_path):
        """Coordinator delegates 'greet' to shared/greeter.yaml#greet."""
        shared_spec = _make_spec(
            tasks={
                "greet": {
                    "description": "Say hello",
                    "input": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
                    "output": {"type": "object", "properties": {"reply": {"type": "string"}}, "required": ["reply"]},
                    "prompts": {"system": "Greet the user.", "user": "Hello {name}"},
                }
            }
        )
        coordinator_spec = _make_spec(
            tasks={
                "greet": {
                    "description": "Delegated greeter",
                    "spec": "shared/greeter.yaml",
                    "task": "greet",
                }
            }
        )

        _write_spec(tmp_path / "shared", "greeter.yaml", shared_spec)
        coordinator_path = _write_spec(tmp_path, "coordinator.yaml", coordinator_spec)

        def fake_invoke(system, user, config):
            return json.dumps({"reply": "Hello there!"})

        with patch("oas_cli.runner.invoke_intelligence", fake_invoke):
            result = run_task_from_file(coordinator_path, task_name="greet", input_data={"name": "Alice"})

        assert result["task"] == "greet"
        assert result["output"]["reply"] == "Hello there!"
        assert "delegated_to" in result

    def test_delegation_task_name_defaults_to_caller(self, tmp_path):
        """When 'task:' is omitted, uses the same name as the coordinator task."""
        shared_spec = _make_spec(
            tasks={
                "analyse": {
                    "description": "Analyse",
                    "input": {"type": "object", "properties": {}, "required": []},
                    "output": {"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"]},
                    "prompts": {"system": "Analyse.", "user": "Go"},
                }
            }
        )
        coordinator_spec = _make_spec(
            tasks={
                "analyse": {
                    "description": "Delegated — no 'task:' key",
                    "spec": "shared.yaml",
                    # no 'task:' — should default to "analyse"
                }
            }
        )

        _write_spec(tmp_path, "shared.yaml", shared_spec)
        coordinator_path = _write_spec(tmp_path, "coordinator.yaml", coordinator_spec)

        def fake_invoke(system, user, config):
            return json.dumps({"result": "done"})

        with patch("oas_cli.runner.invoke_intelligence", fake_invoke):
            result = run_task_from_file(coordinator_path, task_name="analyse")

        assert result["output"]["result"] == "done"
        assert "shared.yaml#analyse" in result["delegated_to"]

    def test_result_task_field_reflects_coordinator_name(self, tmp_path):
        """Even when delegated task has a different internal name, envelope shows coordinator name."""
        shared_spec = _make_spec(
            tasks={
                "internal_summarise": {
                    "description": "Internal summariser",
                    "input": {"type": "object", "properties": {}, "required": []},
                    "output": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
                    "prompts": {"system": "Summarise.", "user": "Go"},
                }
            }
        )
        coordinator_spec = _make_spec(
            tasks={
                "summarise": {
                    "description": "Public summarise task",
                    "spec": "shared.yaml",
                    "task": "internal_summarise",
                }
            }
        )

        _write_spec(tmp_path, "shared.yaml", shared_spec)
        coordinator_path = _write_spec(tmp_path, "coordinator.yaml", coordinator_spec)

        with patch("oas_cli.runner.invoke_intelligence", lambda s, u, c: json.dumps({"summary": "brief"})):
            result = run_task_from_file(coordinator_path, task_name="summarise")

        assert result["task"] == "summarise"


# ── Delegation + depends_on chain ─────────────────────────────────────────────


class TestDelegationWithChain:
    def test_delegated_task_feeds_downstream_via_depends_on(self, tmp_path):
        """A delegated task's output merges into the next task's input."""
        shared_spec = _make_spec(
            tasks={
                "extract": {
                    "description": "Extract keywords",
                    "input": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
                    "output": {"type": "object", "properties": {"keywords": {"type": "array", "items": {"type": "string"}}}, "required": ["keywords"]},
                    "prompts": {"system": "Extract.", "user": "Extract from: {text}"},
                }
            }
        )

        coordinator_spec = _make_spec(
            tasks={
                "extract": {
                    "description": "Delegated extractor",
                    "spec": "shared.yaml",
                    "task": "extract",
                },
                "report": {
                    "description": "Build report from keywords",
                    "depends_on": ["extract"],
                    "input": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
                    "output": {"type": "object", "properties": {"report": {"type": "string"}}, "required": ["report"]},
                    "prompts": {"system": "Report.", "user": "Report on keywords"},
                },
            }
        )

        _write_spec(tmp_path, "shared.yaml", shared_spec)
        coordinator_path = _write_spec(tmp_path, "coordinator.yaml", coordinator_spec)

        call_count = {"n": 0}

        def fake_invoke(system, user, config):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return json.dumps({"keywords": ["foo", "bar"]})
            return json.dumps({"report": "Report about foo and bar"})

        with patch("oas_cli.runner.invoke_intelligence", fake_invoke):
            result = run_task_from_file(
                coordinator_path,
                task_name="report",
                input_data={"text": "Some text"},
            )

        assert result["output"]["report"] == "Report about foo and bar"
        assert "extract" in result["chain"]
        assert result["chain"]["extract"]["output"]["keywords"] == ["foo", "bar"]


# ── Cycle detection ───────────────────────────────────────────────────────────


class TestCycleDetection:
    def test_self_delegation_raises(self, tmp_path):
        """A spec that delegates a task to itself raises DELEGATION_CYCLE_ERROR."""
        import yaml

        spec_path = tmp_path / "self.yaml"
        # Write a placeholder first so the path resolves
        spec_path.write_text("")
        spec = _make_spec(
            tasks={
                "do_thing": {
                    "description": "Self-delegating",
                    "spec": str(spec_path),   # absolute path to itself
                    "task": "do_thing",
                }
            }
        )
        spec_path.write_text(yaml.dump(spec))

        with pytest.raises(OARunError) as exc_info:
            run_task_from_file(spec_path, task_name="do_thing")

        assert exc_info.value.code == "DELEGATION_CYCLE_ERROR"

    def test_mutual_delegation_raises(self, tmp_path):
        """A → B → A cycle raises DELEGATION_CYCLE_ERROR."""
        import yaml

        path_a = tmp_path / "a.yaml"
        path_b = tmp_path / "b.yaml"

        # Write placeholders first
        path_a.write_text("")
        path_b.write_text("")

        spec_a = _make_spec(
            tasks={
                "run": {
                    "description": "Delegates to B",
                    "spec": str(path_b),
                    "task": "run",
                }
            }
        )
        spec_b = _make_spec(
            tasks={
                "run": {
                    "description": "Delegates back to A",
                    "spec": str(path_a),
                    "task": "run",
                }
            }
        )
        path_a.write_text(yaml.dump(spec_a))
        path_b.write_text(yaml.dump(spec_b))

        with pytest.raises(OARunError) as exc_info:
            run_task_from_file(path_a, task_name="run")

        assert exc_info.value.code == "DELEGATION_CYCLE_ERROR"


# ── Error cases ───────────────────────────────────────────────────────────────


class TestDelegationErrors:
    def test_missing_referenced_spec_raises(self, tmp_path):
        """Referencing a spec file that doesn't exist raises SPEC_LOAD_ERROR."""
        coordinator_spec = _make_spec(
            tasks={
                "run": {
                    "description": "Points at nothing",
                    "spec": "./nonexistent/spec.yaml",
                    "task": "run",
                }
            }
        )
        coordinator_path = _write_spec(tmp_path, "coordinator.yaml", coordinator_spec)

        with pytest.raises(OARunError) as exc_info:
            run_task_from_file(coordinator_path, task_name="run")

        assert exc_info.value.code == "SPEC_LOAD_ERROR"

    def test_missing_task_in_referenced_spec_raises(self, tmp_path):
        """Referencing a task that doesn't exist in the shared spec raises TASK_NOT_FOUND."""
        shared_spec = _make_spec(
            tasks={
                "real_task": {
                    "description": "exists",
                    "input": {"type": "object", "properties": {}, "required": []},
                    "output": {"type": "object", "properties": {"r": {"type": "string"}}, "required": ["r"]},
                    "prompts": {"system": "s", "user": "u"},
                }
            }
        )
        coordinator_spec = _make_spec(
            tasks={
                "run": {
                    "description": "Wrong task name",
                    "spec": "shared.yaml",
                    "task": "ghost_task",
                }
            }
        )

        _write_spec(tmp_path, "shared.yaml", shared_spec)
        coordinator_path = _write_spec(tmp_path, "coordinator.yaml", coordinator_spec)

        with pytest.raises(OARunError) as exc_info:
            run_task_from_file(coordinator_path, task_name="run")

        assert exc_info.value.code == "TASK_NOT_FOUND"

    def test_relative_path_resolved_from_spec_directory(self, tmp_path):
        """Relative spec: paths are resolved from the calling spec's directory, not cwd."""
        sub = tmp_path / "agents"
        sub.mkdir()

        shared_spec = _make_spec(
            tasks={
                "ping": {
                    "description": "ping",
                    "input": {"type": "object", "properties": {}, "required": []},
                    "output": {"type": "object", "properties": {"pong": {"type": "string"}}, "required": ["pong"]},
                    "prompts": {"system": "s", "user": "u"},
                }
            }
        )
        coordinator_spec = _make_spec(
            tasks={
                "ping": {
                    "description": "Delegated ping",
                    "spec": "../shared/pinger.yaml",  # relative to agents/
                    "task": "ping",
                }
            }
        )

        _write_spec(tmp_path / "shared", "pinger.yaml", shared_spec)
        coordinator_path = _write_spec(sub, "coordinator.yaml", coordinator_spec)

        with patch("oas_cli.runner.invoke_intelligence", lambda s, u, c: json.dumps({"pong": "ok"})):
            result = run_task_from_file(coordinator_path, task_name="ping")

        assert result["output"]["pong"] == "ok"


# ── Validator ─────────────────────────────────────────────────────────────────


class TestDelegationValidator:
    def test_validator_accepts_delegated_task(self):
        from oas_cli.validators import validate_spec

        spec = _make_spec(
            tasks={
                "run": {
                    "description": "Delegated",
                    "spec": "./shared/worker.yaml",
                    "task": "work",
                }
            }
        )
        validate_spec(spec)  # must not raise

    def test_validator_rejects_empty_spec_field(self):
        from oas_cli.validators import validate_spec

        spec = _make_spec(
            tasks={
                "run": {
                    "description": "Bad delegation",
                    "spec": "",
                }
            }
        )
        with pytest.raises(ValueError, match="non-empty string"):
            validate_spec(spec)

    def test_validator_rejects_non_string_task_field(self):
        from oas_cli.validators import validate_spec

        spec = _make_spec(
            tasks={
                "run": {
                    "description": "Bad task field",
                    "spec": "./shared.yaml",
                    "task": 123,
                }
            }
        )
        with pytest.raises(ValueError, match=r"task.*string"):
            validate_spec(spec)

    def test_delegated_task_exempt_from_input_output_requirement(self):
        """Delegated tasks don't need inline input/output schemas."""
        from oas_cli.validators import validate_spec

        spec = _make_spec(
            tasks={
                "run": {
                    "description": "Delegated — no input/output",
                    "spec": "./shared.yaml",
                    # deliberately no input: or output:
                }
            }
        )
        validate_spec(spec)  # must not raise
