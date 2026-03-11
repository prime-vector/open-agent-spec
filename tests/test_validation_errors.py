"""Tests for improved validation error messages.

Based on the contribution by Chu Julung (PR #27), updated to work
with the current validator structure including optional behavioural
contracts and multi-step task validation.
"""

import pytest

from oas_cli.validators import validate_spec

# -- Minimal valid spec helper ----------------------------------------


def _valid_spec(**overrides):
    """Return a minimal valid spec, with optional overrides."""
    spec = {
        "open_agent_spec": "1.0",
        "agent": {"name": "test", "role": "test"},
        "tools": [],
        "tasks": {
            "task1": {
                "input": {"query": {"type": "string"}},
                "output": {"result": {"type": "string"}},
            }
        },
        "prompts": {"system": "test", "user": "test"},
    }
    spec.update(overrides)
    return spec


# -- Version validation -----------------------------------------------


class TestVersionErrors:
    def test_missing_version_shows_type_and_example(self):
        spec = _valid_spec()
        del spec["open_agent_spec"]
        with pytest.raises(ValueError, match="missing"):
            validate_spec(spec)

    def test_wrong_type_shows_actual(self):
        spec = _valid_spec(open_agent_spec=123)
        with pytest.raises(ValueError, match="int"):
            validate_spec(spec)

    def test_empty_version_shows_example(self):
        spec = _valid_spec(open_agent_spec="")
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_spec(spec)


# -- Agent validation --------------------------------------------------


class TestAgentErrors:
    def test_missing_agent_section(self):
        spec = _valid_spec()
        del spec["agent"]
        with pytest.raises(ValueError, match="Missing required section"):
            validate_spec(spec)

    def test_wrong_type_for_name(self):
        spec = _valid_spec(agent={"name": 123, "role": "test"})
        with pytest.raises(ValueError, match=r"agent\.name"):
            validate_spec(spec)

    def test_missing_role_shows_example(self):
        spec = _valid_spec(agent={"name": "test"})
        with pytest.raises(ValueError, match=r"agent\.role"):
            validate_spec(spec)


# -- Behavioural contract validation -----------------------------------


class TestBehaviouralContractErrors:
    def test_no_contract_is_valid(self):
        """Specs without behavioural_contract should pass."""
        spec = _valid_spec()
        # Should not raise
        validate_spec(spec)

    def test_contract_wrong_type(self):
        spec = _valid_spec(behavioural_contract="not a dict")
        with pytest.raises(ValueError, match="dictionary"):
            validate_spec(spec)

    def test_contract_missing_version(self):
        spec = _valid_spec(behavioural_contract={"description": "test"})
        with pytest.raises(ValueError, match=r"behavioural_contract\.version"):
            validate_spec(spec)

    def test_contract_wrong_type_for_flags(self):
        spec = _valid_spec(
            behavioural_contract={
                "version": "1.0",
                "description": "test",
                "behavioural_flags": "not a dict",
            }
        )
        with pytest.raises(ValueError, match="got str"):
            validate_spec(spec)


# -- Tools validation --------------------------------------------------


class TestToolErrors:
    def test_tools_wrong_type(self):
        spec = _valid_spec(tools="not a list")
        with pytest.raises(ValueError, match=r"list.*got str"):
            validate_spec(spec)

    def test_tool_item_wrong_type(self):
        spec = _valid_spec(tools=["not a dict"])
        with pytest.raises(ValueError, match=r"tools\[0\]"):
            validate_spec(spec)

    def test_tool_id_wrong_type_shows_index(self):
        spec = _valid_spec(
            tools=[
                {
                    "id": "good",
                    "type": "function",
                    "description": "ok",
                },
                {
                    "id": 123,
                    "type": "function",
                    "description": "bad",
                },
            ]
        )
        with pytest.raises(ValueError, match=r"tools\[1\]\.id.*int"):
            validate_spec(spec)

    def test_tool_missing_description(self):
        spec = _valid_spec(tools=[{"id": "t1", "type": "function"}])
        with pytest.raises(ValueError, match=r"tools\[0\]\.description.*missing"):
            validate_spec(spec)

    def test_tool_missing_type(self):
        spec = _valid_spec(tools=[{"id": "t1", "description": "desc"}])
        with pytest.raises(ValueError, match=r"tools\[0\]\.type.*missing"):
            validate_spec(spec)


# -- Task validation ---------------------------------------------------


class TestTaskErrors:
    def test_tasks_wrong_type(self):
        spec = _valid_spec(tasks="not a dict")
        with pytest.raises(ValueError, match=r"tasks.*got str"):
            validate_spec(spec)

    def test_task_def_wrong_type(self):
        spec = _valid_spec(tasks={"bad": "not a dict"})
        with pytest.raises(ValueError, match=r"tasks\.bad.*got str"):
            validate_spec(spec)

    def test_nonexistent_tool_lists_available(self):
        spec = _valid_spec(
            tools=[
                {
                    "id": "tool1",
                    "type": "function",
                    "description": "T1",
                },
                {
                    "id": "tool2",
                    "type": "function",
                    "description": "T2",
                },
            ],
            tasks={
                "bad_task": {
                    "tool": "nonexistent",
                    "input": {},
                    "output": {},
                }
            },
        )
        with pytest.raises(ValueError) as exc_info:
            validate_spec(spec)

        msg = str(exc_info.value)
        assert "nonexistent" in msg
        assert "tool1" in msg
        assert "tool2" in msg
        assert "Available" in msg or "available" in msg

    def test_missing_input_shows_type(self):
        spec = _valid_spec(tasks={"t1": {"output": {"r": {"type": "string"}}}})
        with pytest.raises(ValueError, match=r"tasks\.t1\.input.*missing"):
            validate_spec(spec)

    def test_missing_output_shows_type(self):
        spec = _valid_spec(tasks={"t1": {"input": {"q": {"type": "string"}}}})
        with pytest.raises(ValueError, match=r"tasks\.t1\.output.*missing"):
            validate_spec(spec)


# -- Multi-step task validation ----------------------------------------


class TestMultiStepTaskErrors:
    def test_missing_steps(self):
        spec = _valid_spec(
            tasks={
                "pipeline": {
                    "multi_step": True,
                    "output": {"r": {"type": "string"}},
                }
            }
        )
        with pytest.raises(ValueError, match=r"steps.*missing"):
            validate_spec(spec)

    def test_empty_steps(self):
        spec = _valid_spec(
            tasks={
                "pipeline": {
                    "multi_step": True,
                    "steps": [],
                    "output": {"r": {"type": "string"}},
                }
            }
        )
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_spec(spec)

    def test_step_references_nonexistent_task(self):
        spec = _valid_spec(
            tasks={
                "step1": {
                    "input": {"q": {"type": "string"}},
                    "output": {"r": {"type": "string"}},
                },
                "pipeline": {
                    "multi_step": True,
                    "steps": [
                        {"task": "step1"},
                        {"task": "ghost"},
                    ],
                    "output": {"r": {"type": "string"}},
                },
            }
        )
        with pytest.raises(ValueError) as exc_info:
            validate_spec(spec)

        msg = str(exc_info.value)
        assert "ghost" in msg
        assert "step1" in msg or "pipeline" in msg


# -- Error format consistency -----------------------------------------


class TestErrorFormat:
    def test_field_path_included(self):
        spec = _valid_spec(agent={"name": 123, "role": "test"})
        with pytest.raises(ValueError) as exc_info:
            validate_spec(spec)
        assert "agent.name" in str(exc_info.value)

    def test_actual_type_shown(self):
        spec = _valid_spec(open_agent_spec=123)
        with pytest.raises(ValueError) as exc_info:
            validate_spec(spec)
        assert "int" in str(exc_info.value)
