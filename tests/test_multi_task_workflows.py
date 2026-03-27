"""Exhaustive multi-task / multi-prompt workflow tests for Open Agent Spec.

Covers:
  A. Prompt resolution — all four layers × system/user independently
  B. Multi-task routing — default, explicit, invalid, all-multi-step
  C. Template interpolation — {{ field }}, {{ input.field }}, edge cases
  D. Output normalisation — JSON, fenced JSON, non-JSON, dict passthrough
  E. Intelligence config building — defaults, custom, extra keys
  F. Spec shape permutations — no global prompts, per-task only, mixed, etc.
  G. Real-world scenario specs — code-assistant (edit/ask/explain/review)
  H. CLI integration — all flag combinations via CliRunner + monkeypatched invoke
  I. Error cases — missing model, empty tasks, unknown task, bad JSON input
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from oas_cli.main import app
from oas_cli.runner import (
    OARunError,
    _build_intelligence_config,
    _build_prompt,
    _choose_task,
    run_task_from_spec,
)

cli = CliRunner()


# ===========================================================================
# Shared fixtures / spec builders
# ===========================================================================


def _base_intelligence() -> dict:
    return {"type": "llm", "engine": "openai", "model": "gpt-4o"}


def _task(
    description: str = "a task",
    *,
    system: str | None = None,
    user: str | None = None,
    input_fields: dict | None = None,
    output_fields: dict | None = None,
    multi_step: bool = False,
) -> dict:
    """Build a task definition dict."""
    t: dict = {"description": description}
    if multi_step:
        t["multi_step"] = True

    prompts: dict = {}
    if system is not None:
        prompts["system"] = system
    if user is not None:
        prompts["user"] = user
    if prompts:
        t["prompts"] = prompts

    props = output_fields or {"result": {"type": "string"}}
    t["output"] = {
        "type": "object",
        "properties": props,
        "required": list(props.keys()),
    }

    if input_fields:
        t["input"] = {
            "type": "object",
            "properties": input_fields,
            "required": list(input_fields.keys()),
        }
    return t


def _spec(
    tasks: dict,
    *,
    global_system: str | None = None,
    global_user: str | None = None,
    style_b: dict | None = None,  # {task_name: {system:.., user:..}}
    intelligence: dict | None = None,
) -> dict:
    """Build a complete spec dict."""
    prompts: dict = {}
    if global_system is not None:
        prompts["system"] = global_system
    if global_user is not None:
        prompts["user"] = global_user
    if style_b:
        prompts.update(style_b)

    s: dict = {
        "open_agent_spec": "1.2.9",
        "agent": {"name": "test-agent", "description": "test agent"},
        "intelligence": intelligence or _base_intelligence(),
        "tasks": tasks,
    }
    if prompts:
        s["prompts"] = prompts
    return s


def _run(
    spec: dict,
    task_name: str | None = None,
    input_data: dict | None = None,
    override_system: str | None = None,
    override_user: str | None = None,
    fake_response: str = '{"result": "ok"}',
) -> dict:
    """Run run_task_from_spec with a monkeypatched invoke_intelligence."""
    captured: dict = {}

    def fake_invoke(prompt: str, config: dict) -> str:
        captured["prompt"] = prompt
        captured["config"] = config
        return fake_response

    with patch("oas_cli.runner.invoke_intelligence", fake_invoke):
        result = run_task_from_spec(
            spec,
            task_name=task_name,
            input_data=input_data or {},
            override_system=override_system,
            override_user=override_user,
        )
    result["_captured_prompt"] = captured.get("prompt", "")
    result["_captured_config"] = captured.get("config", {})
    return result


# ===========================================================================
# A. Prompt resolution — all four layers
# ===========================================================================


class TestPromptResolutionLayers:
    """All four priority layers, both dimensions, independent resolution."""

    # --- Layer 4: global fallback ---

    def test_global_system_used_when_nothing_else(self):
        spec = _spec({"t": _task()}, global_system="global-sys", global_user="{{ x }}")
        result = _run(spec, "t", {"x": "val"})
        assert "global-sys" in result["_captured_prompt"]
        assert "val" in result["_captured_prompt"]

    def test_global_user_template_interpolated(self):
        spec = _spec({"t": _task()}, global_system="sys", global_user="hi {{ name }}")
        result = _run(spec, "t", {"name": "Bob"})
        assert "hi Bob" in result["_captured_prompt"]

    def test_empty_global_prompts_section_no_crash(self):
        """Spec with prompts: {} should not crash; produces minimal prompt."""
        spec = _spec({"t": _task()})
        # No prompts section at all → system="" and user_template="{{ input }}"
        result = _run(spec, "t", {})
        assert result["task"] == "t"

    def test_no_prompts_section_at_all(self):
        """Spec without any prompts key (per-task or global) runs without error."""
        spec = {
            "open_agent_spec": "1.2.9",
            "agent": {"name": "a", "description": "b"},
            "intelligence": _base_intelligence(),
            "tasks": {"t": _task(system="task-sys", user="{{ q }}")},
        }
        result = _run(spec, "t", {"q": "hello"})
        assert "task-sys" in result["_captured_prompt"]
        assert "hello" in result["_captured_prompt"]

    # --- Layer 3: Style B (prompts.<task_name>) ---

    def test_style_b_system_overrides_global(self):
        spec = _spec(
            {"t": _task()},
            global_system="global-sys",
            global_user="{{ q }}",
            style_b={"t": {"system": "style-b-sys"}},
        )
        result = _run(spec, "t", {"q": "hi"})
        assert "style-b-sys" in result["_captured_prompt"]
        assert "global-sys" not in result["_captured_prompt"]

    def test_style_b_user_overrides_global(self):
        spec = _spec(
            {"t": _task()},
            global_system="sys",
            global_user="global {{ q }}",
            style_b={"t": {"user": "style-b {{ q }}"}},
        )
        result = _run(spec, "t", {"q": "x"})
        assert "style-b x" in result["_captured_prompt"]
        assert "global x" not in result["_captured_prompt"]

    def test_style_b_partial_system_only(self):
        """Style B has system; user falls through to global."""
        spec = _spec(
            {"t": _task()},
            global_system="global-sys",
            global_user="global {{ q }}",
            style_b={"t": {"system": "style-b-sys"}},
        )
        result = _run(spec, "t", {"q": "y"})
        assert "style-b-sys" in result["_captured_prompt"]
        assert "global y" in result["_captured_prompt"]

    # --- Layer 2: Style A (tasks.<name>.prompts) ---

    def test_style_a_system_overrides_global(self):
        spec = _spec(
            {"t": _task(system="inline-sys")},
            global_system="global-sys",
            global_user="{{ q }}",
        )
        result = _run(spec, "t", {"q": "z"})
        assert "inline-sys" in result["_captured_prompt"]
        assert "global-sys" not in result["_captured_prompt"]

    def test_style_a_user_overrides_global(self):
        spec = _spec(
            {"t": _task(user="inline {{ q }}")},
            global_system="sys",
            global_user="global {{ q }}",
        )
        result = _run(spec, "t", {"q": "z"})
        assert "inline z" in result["_captured_prompt"]
        assert "global z" not in result["_captured_prompt"]

    def test_style_a_overrides_style_b_system(self):
        spec = _spec(
            {"t": _task(system="inline-sys")},
            global_system="global-sys",
            global_user="{{ q }}",
            style_b={"t": {"system": "style-b-sys"}},
        )
        result = _run(spec, "t", {"q": "hi"})
        assert "inline-sys" in result["_captured_prompt"]
        assert "style-b-sys" not in result["_captured_prompt"]
        assert "global-sys" not in result["_captured_prompt"]

    def test_style_a_overrides_style_b_user(self):
        spec = _spec(
            {"t": _task(user="inline {{ q }}")},
            global_system="sys",
            global_user="global {{ q }}",
            style_b={"t": {"user": "style-b {{ q }}"}},
        )
        result = _run(spec, "t", {"q": "v"})
        assert "inline v" in result["_captured_prompt"]
        assert "style-b v" not in result["_captured_prompt"]
        assert "global v" not in result["_captured_prompt"]

    def test_style_a_partial_system_user_falls_to_global(self):
        """Style A has system only; user template falls through to global."""
        spec = _spec(
            {"t": _task(system="inline-sys")},
            global_system="global-sys",
            global_user="global {{ q }}",
        )
        result = _run(spec, "t", {"q": "fallthrough"})
        assert "inline-sys" in result["_captured_prompt"]
        assert "global fallthrough" in result["_captured_prompt"]

    def test_style_a_partial_user_system_falls_to_global(self):
        """Style A has user only; system falls through to global."""
        spec = _spec(
            {"t": _task(user="inline {{ q }}")},
            global_system="global-sys",
            global_user="global {{ q }}",
        )
        result = _run(spec, "t", {"q": "ft"})
        assert "global-sys" in result["_captured_prompt"]
        assert "inline ft" in result["_captured_prompt"]

    # --- Layer 1: CLI overrides ---

    def test_override_system_beats_everything(self):
        spec = _spec(
            {"t": _task(system="inline-sys")},
            global_system="global-sys",
            style_b={"t": {"system": "style-b-sys"}},
            global_user="{{ q }}",
        )
        result = _run(spec, "t", {"q": "hi"}, override_system="cli-sys")
        assert "cli-sys" in result["_captured_prompt"]
        assert "inline-sys" not in result["_captured_prompt"]
        assert "style-b-sys" not in result["_captured_prompt"]
        assert "global-sys" not in result["_captured_prompt"]

    def test_override_user_beats_everything(self):
        spec = _spec(
            {"t": _task(system="sys", user="inline {{ q }}")},
            global_user="global {{ q }}",
            style_b={"t": {"user": "style-b {{ q }}"}},
        )
        result = _run(spec, "t", {"q": "x"}, override_user="cli {{ q }}")
        assert "cli x" in result["_captured_prompt"]
        assert "inline x" not in result["_captured_prompt"]
        assert "style-b x" not in result["_captured_prompt"]
        assert "global x" not in result["_captured_prompt"]

    def test_both_overrides_fully_replace(self):
        spec = _spec(
            {"t": _task(system="inline-sys", user="inline {{ q }}")},
            global_system="global-sys",
            global_user="global {{ q }}",
        )
        result = _run(
            spec,
            "t",
            {"q": "ignored"},
            override_system="cli-sys",
            override_user="cli-user",
        )
        assert result["_captured_prompt"] == "cli-sys\n\ncli-user"

    def test_override_system_none_does_not_suppress_inline(self):
        spec = _spec({"t": _task(system="inline-sys", user="{{ q }}")})
        result = _run(spec, "t", {"q": "v"}, override_system=None)
        assert "inline-sys" in result["_captured_prompt"]

    def test_override_empty_string_replaces_system(self):
        """Explicit empty string override replaces (not suppresses) system prompt."""
        spec = _spec({"t": _task(system="inline-sys", user="{{ q }}")})
        result = _run(spec, "t", {"q": "v"}, override_system="")
        # override_system="" is not None, so it replaces → system is empty
        assert "inline-sys" not in result["_captured_prompt"]


# ===========================================================================
# B. Multi-task routing
# ===========================================================================


class TestMultiTaskRouting:
    def test_default_selects_first_non_multistep(self):
        spec = _spec(
            {
                "alpha": _task("first", system="alpha-sys", user="{{ q }}"),
                "beta": _task("second", system="beta-sys", user="{{ q }}"),
            },
            global_user="{{ q }}",
        )
        result = _run(spec, None, {"q": "hi"})
        assert result["task"] == "alpha"
        assert "alpha-sys" in result["_captured_prompt"]

    def test_explicit_task_second(self):
        spec = _spec(
            {
                "alpha": _task("first", system="alpha-sys", user="{{ q }}"),
                "beta": _task("second", system="beta-sys", user="{{ q }}"),
            }
        )
        result = _run(spec, "beta", {"q": "hi"})
        assert result["task"] == "beta"
        assert "beta-sys" in result["_captured_prompt"]

    def test_unknown_task_raises(self):
        spec = _spec({"t": _task()})
        with pytest.raises(OARunError, match="not found"):
            _run(spec, "nonexistent", {})

    def test_empty_tasks_raises(self):
        spec = _spec({})
        with pytest.raises(OARunError, match="no tasks"):
            _run(spec, None, {})

    def test_all_multistep_falls_back_to_first(self):
        spec = _spec(
            {
                "a": _task("a", multi_step=True, system="a-sys", user="{{ q }}"),
                "b": _task("b", multi_step=True, system="b-sys", user="{{ q }}"),
            }
        )
        result = _run(spec, None, {"q": "hi"})
        assert result["task"] == "a"

    def test_three_tasks_each_with_own_system(self):
        spec = _spec(
            {
                "edit": _task("edit", system="edit-sys", user="{{ instructions }}"),
                "ask": _task("ask", system="ask-sys", user="{{ question }}"),
                "explain": _task("explain", system="explain-sys", user="{{ code }}"),
            }
        )
        for task_name, field, sys_str in [
            ("edit", "instructions", "edit-sys"),
            ("ask", "question", "ask-sys"),
            ("explain", "code", "explain-sys"),
        ]:
            result = _run(spec, task_name, {field: "value"})
            assert result["task"] == task_name
            assert sys_str in result["_captured_prompt"]

    def test_mixed_prompts_tasks(self):
        """Task A has per-task prompt; task B falls through to global."""
        spec = _spec(
            {
                "a": _task("a", system="a-sys", user="{{ x }}"),
                "b": _task("b"),  # no per-task prompts
            },
            global_system="global-sys",
            global_user="{{ x }}",
        )
        a = _run(spec, "a", {"x": "val"})
        b = _run(spec, "b", {"x": "val"})
        assert "a-sys" in a["_captured_prompt"]
        assert "global-sys" not in a["_captured_prompt"]
        assert "global-sys" in b["_captured_prompt"]


# ===========================================================================
# C. Template interpolation
# ===========================================================================


class TestTemplateInterpolation:
    def test_single_placeholder(self):
        spec = _spec({"t": _task(system="s", user="{{ name }}")})
        result = _run(spec, "t", {"name": "Alice"})
        assert "Alice" in result["_captured_prompt"]

    def test_input_dot_prefix(self):
        spec = _spec({"t": _task(system="s", user="{{ input.name }}")})
        result = _run(spec, "t", {"name": "Bob"})
        assert "Bob" in result["_captured_prompt"]

    def test_multiple_fields(self):
        spec = _spec({"t": _task(system="s", user="{{ a }} and {{ b }}")})
        result = _run(spec, "t", {"a": "foo", "b": "bar"})
        assert "foo and bar" in result["_captured_prompt"]

    def test_numeric_value_stringified(self):
        spec = _spec({"t": _task(system="s", user="{{ n }}")})
        result = _run(spec, "t", {"n": 42})
        assert "42" in result["_captured_prompt"]

    def test_boolean_value_stringified(self):
        spec = _spec({"t": _task(system="s", user="{{ flag }}")})
        result = _run(spec, "t", {"flag": True})
        assert "True" in result["_captured_prompt"]

    def test_unmatched_placeholder_left_intact(self):
        spec = _spec({"t": _task(system="s", user="{{ missing }}")})
        result = _run(spec, "t", {})
        assert "{{ missing }}" in result["_captured_prompt"]

    def test_both_notations_in_same_template(self):
        spec = _spec({"t": _task(system="s", user="{{ x }} / {{ input.x }}")})
        result = _run(spec, "t", {"x": "val"})
        assert "val / val" in result["_captured_prompt"]

    def test_no_input_template_unchanged(self):
        spec = _spec({"t": _task(system="s", user="static user text")})
        result = _run(spec, "t", {})
        assert "static user text" in result["_captured_prompt"]

    def test_multiline_system_prompt_preserved(self):
        system = "Line 1\nLine 2\nLine 3"
        spec = _spec({"t": _task(system=system, user="{{ q }}")})
        result = _run(spec, "t", {"q": "hi"})
        assert "Line 1" in result["_captured_prompt"]
        assert "Line 2" in result["_captured_prompt"]
        assert "Line 3" in result["_captured_prompt"]

    def test_large_input_value(self):
        big = "x" * 5000
        spec = _spec({"t": _task(system="s", user="{{ content }}")})
        result = _run(spec, "t", {"content": big})
        assert big in result["_captured_prompt"]

    def test_system_prompt_with_placeholder(self):
        """Unusual but valid: placeholder in system prompt should not be substituted
        (only user template is interpolated currently)."""
        spec = _spec({"t": _task(system="sys {{ name }}", user="{{ name }}")})
        result = _run(spec, "t", {"name": "Alice"})
        # System prompt is NOT interpolated — placeholder stays literal
        assert "sys {{ name }}" in result["_captured_prompt"]
        # User prompt IS interpolated
        assert "Alice" in result["_captured_prompt"]


# ===========================================================================
# D. Output normalisation
# ===========================================================================


class TestOutputNormalisation:
    def test_plain_json_string_parsed(self):
        spec = _spec({"t": _task(system="s", user="{{ q }}")})
        result = _run(spec, "t", {"q": "hi"}, fake_response='{"answer": "42"}')
        assert result["output"] == {"answer": "42"}

    def test_json_fenced_with_language_tag(self):
        fenced = '```json\n{"answer": "ok"}\n```'
        spec = _spec({"t": _task(system="s", user="{{ q }}")})
        result = _run(spec, "t", {"q": "hi"}, fake_response=fenced)
        assert result["output"] == {"answer": "ok"}

    def test_json_fenced_without_language_tag(self):
        fenced = '```\n{"answer": "plain"}\n```'
        spec = _spec({"t": _task(system="s", user="{{ q }}")})
        result = _run(spec, "t", {"q": "hi"}, fake_response=fenced)
        assert result["output"] == {"answer": "plain"}

    def test_non_json_string_returned_as_string(self):
        spec = _spec({"t": _task(system="s", user="{{ q }}")})
        result = _run(spec, "t", {"q": "hi"}, fake_response="just a plain string")
        assert result["output"] == "just a plain string"
        assert result["raw_output"] == "just a plain string"

    def test_dict_returned_directly_passthrough(self):
        """If invoke_intelligence returns a dict, pass it straight through."""
        spec = _spec({"t": _task(system="s", user="{{ q }}")})

        def fake_invoke(prompt: str, config: dict) -> dict:
            return {"result": "direct"}

        with patch("oas_cli.runner.invoke_intelligence", fake_invoke):
            result = run_task_from_spec(spec, "t", {"q": "hi"})

        assert result["output"] == {"result": "direct"}

    def test_whitespace_stripped_before_json_parse(self):
        spec = _spec({"t": _task(system="s", user="{{ q }}")})
        result = _run(spec, "t", {"q": "hi"}, fake_response='  {"v": 1}  ')
        assert result["output"] == {"v": 1}

    def test_result_envelope_fields_present(self):
        spec = _spec({"t": _task(system="s", user="{{ q }}")})
        result = _run(spec, "t", {"q": "hi"})
        for key in (
            "task",
            "input",
            "prompt",
            "engine",
            "model",
            "raw_output",
            "output",
        ):
            assert key in result

    def test_result_task_name_matches(self):
        spec = _spec(
            {
                "alpha": _task("a", system="s", user="{{ q }}"),
                "beta": _task("b", system="s", user="{{ q }}"),
            }
        )
        result = _run(spec, "beta", {"q": "hi"})
        assert result["task"] == "beta"

    def test_result_input_echoed(self):
        spec = _spec({"t": _task(system="s", user="{{ q }}")})
        result = _run(spec, "t", {"q": "myvalue"})
        assert result["input"] == {"q": "myvalue"}

    def test_result_engine_and_model(self):
        spec = _spec(
            {"t": _task(system="s", user="{{ q }}")},
            intelligence={"type": "llm", "engine": "openai", "model": "gpt-4o-mini"},
        )
        result = _run(spec, "t", {"q": "hi"})
        assert result["engine"] == "openai"
        assert result["model"] == "gpt-4o-mini"


# ===========================================================================
# E. Intelligence config building
# ===========================================================================


class TestIntelligenceConfigBuilding:
    def test_defaults_when_not_specified(self):
        spec = _spec({"t": _task()})
        cfg = _build_intelligence_config(spec)
        assert cfg["endpoint"] == "https://api.openai.com/v1"
        assert cfg["temperature"] == 0.7
        assert cfg["max_tokens"] == 1000

    def test_custom_endpoint(self):
        spec = _spec(
            {"t": _task()},
            intelligence={
                "type": "llm",
                "engine": "openai",
                "model": "gpt-4o",
                "endpoint": "https://custom.example.com/v1",
            },
        )
        cfg = _build_intelligence_config(spec)
        assert cfg["endpoint"] == "https://custom.example.com/v1"

    def test_config_overrides_defaults(self):
        spec = _spec(
            {"t": _task()},
            intelligence={
                "type": "llm",
                "engine": "openai",
                "model": "gpt-4o",
                "config": {"temperature": 0.1, "max_tokens": 256},
            },
        )
        cfg = _build_intelligence_config(spec)
        assert cfg["temperature"] == 0.1
        assert cfg["max_tokens"] == 256

    def test_extra_config_keys_passed_through(self):
        spec = _spec(
            {"t": _task()},
            intelligence={
                "type": "llm",
                "engine": "openai",
                "model": "gpt-4o",
                "config": {"top_p": 0.9, "frequency_penalty": 0.5},
            },
        )
        cfg = _build_intelligence_config(spec)
        assert cfg["top_p"] == 0.9
        assert cfg["frequency_penalty"] == 0.5

    def test_missing_model_raises(self):
        spec = _spec({"t": _task()}, intelligence={"type": "llm", "engine": "openai"})
        with pytest.raises(ValueError, match="model"):
            _build_intelligence_config(spec)

    def test_engine_field_passed_through(self):
        spec = _spec(
            {"t": _task()},
            intelligence={
                "type": "llm",
                "engine": "anthropic",
                "model": "claude-3-sonnet",
            },
        )
        cfg = _build_intelligence_config(spec)
        assert cfg["engine"] == "anthropic"


# ===========================================================================
# F. Spec shape permutations
# ===========================================================================


class TestSpecShapePermutations:
    """Integration-level: run_task_from_spec with various spec structures."""

    def test_per_task_only_no_global_prompts(self):
        """Spec with no top-level prompts section; each task has its own."""
        spec = {
            "open_agent_spec": "1.2.9",
            "agent": {"name": "a", "description": "b"},
            "intelligence": _base_intelligence(),
            "tasks": {
                "t1": _task("t1", system="t1-sys", user="{{ x }}"),
                "t2": _task("t2", system="t2-sys", user="{{ x }}"),
            },
        }
        t1 = _run(spec, "t1", {"x": "v"})
        t2 = _run(spec, "t2", {"x": "v"})
        assert "t1-sys" in t1["_captured_prompt"]
        assert "t2-sys" in t2["_captured_prompt"]

    def test_global_only_no_per_task_prompts(self):
        """Classic pre-1.2.9 style: all tasks share the global prompt."""
        spec = _spec(
            {
                "greet": _task("greet"),
                "bye": _task("bye"),
            },
            global_system="you are an agent",
            global_user="{{ msg }}",
        )
        for task in ("greet", "bye"):
            result = _run(spec, task, {"msg": "hello"})
            assert "you are an agent" in result["_captured_prompt"]

    def test_five_tasks_mixed_prompts(self):
        """Five tasks: 2 per-task, 1 Style B, 2 global fallback."""
        spec = _spec(
            {
                "edit": _task("edit", system="edit-sys", user="{{ code }}"),
                "ask": _task("ask", system="ask-sys", user="{{ q }}"),
                "format": _task("format"),  # global fallback
                "explain": _task("explain"),  # global fallback
                "review": _task("review"),  # Style B
            },
            global_system="global-sys",
            global_user="{{ text }}",
            style_b={"review": {"system": "review-sys", "user": "{{ diff }}"}},
        )

        edit_r = _run(spec, "edit", {"code": "x"})
        ask_r = _run(spec, "ask", {"q": "y"})
        format_r = _run(spec, "format", {"text": "z"})
        explain_r = _run(spec, "explain", {"text": "w"})
        review_r = _run(spec, "review", {"diff": "d"})

        assert "edit-sys" in edit_r["_captured_prompt"]
        assert "ask-sys" in ask_r["_captured_prompt"]
        assert "global-sys" in format_r["_captured_prompt"]
        assert "global-sys" in explain_r["_captured_prompt"]
        assert "review-sys" in review_r["_captured_prompt"]
        assert "global-sys" not in review_r["_captured_prompt"]

    def test_single_task_spec_runs_without_task_name(self):
        spec = _spec({"only": _task("only", system="sys", user="{{ q }}")})
        result = _run(spec, None, {"q": "hi"})
        assert result["task"] == "only"

    def test_task_with_no_input_defined(self):
        """Task with no input schema accepts empty input_data."""
        spec = _spec({"t": _task(system="sys", user="static prompt")})
        result = _run(spec, "t", {})
        assert result["task"] == "t"

    def test_spec_with_optional_behavioural_contract(self):
        """Extra spec fields don't break the runner."""
        spec = _spec({"t": _task(system="sys", user="{{ q }}")})
        spec["behavioural_contract"] = {"version": "1.0", "description": "test"}
        result = _run(spec, "t", {"q": "hi"})
        assert result["task"] == "t"


# ===========================================================================
# G. Real-world scenario: code-assistant (edit / ask / explain / review)
# ===========================================================================

CODE_ASSISTANT_SPEC = _spec(
    {
        "edit": _task(
            "Apply targeted edits to code",
            system=(
                "You are a precise code editor. Apply only the requested changes. "
                "Output a unified diff. Output valid JSON only."
            ),
            user="{{ instructions }}",
            input_fields={"instructions": {"type": "string"}},
            output_fields={"diff": {"type": "string"}},
        ),
        "ask": _task(
            "Answer a question about the codebase",
            system=(
                "You are a helpful code assistant. Be concise and accurate. "
                "Output valid JSON only."
            ),
            user="{{ question }}",
            input_fields={"question": {"type": "string"}},
            output_fields={"answer": {"type": "string"}},
        ),
        "explain": _task(
            "Explain a code snippet",
            system=(
                "You are a code explainer. Explain the intent and behavior of the "
                "given code clearly. Output valid JSON only."
            ),
            user="{{ code }}",
            input_fields={"code": {"type": "string"}},
            output_fields={"explanation": {"type": "string"}},
        ),
        "review": _task(
            "Review a git diff",
            system=(
                "You are a code reviewer. Decision: approve, comment, or request_changes. "
                "Output valid JSON only."
            ),
            user="{{ diff }}",
            input_fields={"diff": {"type": "string"}},
            output_fields={
                "decision": {"type": "string"},
                "summary": {"type": "string"},
            },
        ),
    },
    global_system="You are a general-purpose code agent. Output valid JSON only.",
    global_user="{{ input }}",
)


class TestCodeAssistantScenario:
    def test_each_task_uses_its_own_system_prompt(self):
        for task, field, sys_kw in [
            ("edit", "instructions", "precise code editor"),
            ("ask", "question", "helpful code assistant"),
            ("explain", "code", "code explainer"),
            ("review", "diff", "code reviewer"),
        ]:
            result = _run(CODE_ASSISTANT_SPEC, task, {field: "sample"})
            assert sys_kw in result["_captured_prompt"], (
                f"Task '{task}' did not use its own system prompt"
            )
            assert "general-purpose" not in result["_captured_prompt"], (
                f"Task '{task}' fell through to global prompt unexpectedly"
            )

    def test_edit_task_prompt_contains_instructions(self):
        result = _run(
            CODE_ASSISTANT_SPEC, "edit", {"instructions": "remove all print statements"}
        )
        assert "remove all print statements" in result["_captured_prompt"]

    def test_ask_task_prompt_contains_question(self):
        result = _run(CODE_ASSISTANT_SPEC, "ask", {"question": "what does auth.py do?"})
        assert "what does auth.py do?" in result["_captured_prompt"]

    def test_explain_task_prompt_contains_code(self):
        code = "def foo(x): return x * 2"
        result = _run(CODE_ASSISTANT_SPEC, "explain", {"code": code})
        assert code in result["_captured_prompt"]

    def test_review_task_prompt_contains_diff(self):
        diff = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new"
        result = _run(CODE_ASSISTANT_SPEC, "review", {"diff": diff})
        assert "--- a/foo.py" in result["_captured_prompt"]

    def test_cli_override_on_ask_task(self):
        result = _run(
            CODE_ASSISTANT_SPEC,
            "ask",
            {"question": "what is this?"},
            override_system="You are a senior Go engineer. Be terse.",
        )
        assert "senior Go engineer" in result["_captured_prompt"]
        assert "helpful code assistant" not in result["_captured_prompt"]

    def test_default_task_is_edit(self):
        """With no --task flag, the first task (edit) is selected."""
        result = _run(CODE_ASSISTANT_SPEC, None, {"instructions": "fix it"})
        assert result["task"] == "edit"

    def test_sequential_tasks_independent(self):
        """Running multiple tasks sequentially doesn't bleed prompts between them."""
        r1 = _run(CODE_ASSISTANT_SPEC, "edit", {"instructions": "fix x"})
        r2 = _run(CODE_ASSISTANT_SPEC, "ask", {"question": "why?"})
        r3 = _run(CODE_ASSISTANT_SPEC, "explain", {"code": "y = 1"})

        assert "precise code editor" in r1["_captured_prompt"]
        assert "precise code editor" not in r2["_captured_prompt"]
        assert "helpful code assistant" in r2["_captured_prompt"]
        assert "code explainer" in r3["_captured_prompt"]

    def test_review_task_output_parsed(self):
        fake = '{"decision": "approve", "summary": "Looks good."}'
        result = _run(CODE_ASSISTANT_SPEC, "review", {"diff": "x"}, fake_response=fake)
        assert result["output"]["decision"] == "approve"
        assert result["output"]["summary"] == "Looks good."


# ===========================================================================
# H. CLI integration — all flag combinations
# ===========================================================================

_CLI_SPEC_YAML = """\
open_agent_spec: "1.2.9"

agent:
  name: cli-test-agent
  description: test agent for CLI flag coverage

intelligence:
  type: llm
  engine: openai
  model: gpt-4o

tasks:
  greet:
    description: say hello
    prompts:
      system: "greet-sys"
      user: "Greet {{ name }}"
    input:
      type: object
      properties:
        name: { type: string }
      required: [name]
    output:
      type: object
      properties:
        response: { type: string }
      required: [response]

  bye:
    description: say goodbye
    prompts:
      system: "bye-sys"
      user: "Say bye to {{ name }}"
    input:
      type: object
      properties:
        name: { type: string }
      required: [name]
    output:
      type: object
      properties:
        response: { type: string }
      required: [response]

prompts:
  system: "global-sys"
  user: "{{ name }}"
"""


@pytest.fixture()
def cli_spec_file(tmp_path: Path) -> Path:
    f = tmp_path / "agent.yaml"
    f.write_text(_CLI_SPEC_YAML)
    return f


def _cli_run(
    cli_spec_file: Path, *extra_args: str, fake_response: str = '{"response": "ok"}'
) -> object:
    captured: dict = {}

    def fake_invoke(prompt: str, config: dict) -> str:
        captured["prompt"] = prompt
        return fake_response

    with patch("oas_cli.runner.invoke_intelligence", fake_invoke):
        result = cli.invoke(
            app,
            [
                "run",
                "--spec",
                str(cli_spec_file),
                "--quiet",
                *extra_args,
            ],
        )

    result._captured_prompt = captured.get("prompt", "")  # type: ignore[attr-defined]
    return result


class TestCLIIntegration:
    def test_default_task_uses_first_task_sys_prompt(self, cli_spec_file):
        result = _cli_run(cli_spec_file, "--input", '{"name":"Alice"}')
        assert result.exit_code == 0, result.output
        assert "greet-sys" in result._captured_prompt

    def test_explicit_task_greet(self, cli_spec_file):
        result = _cli_run(cli_spec_file, "--task", "greet", "--input", '{"name":"A"}')
        assert result.exit_code == 0
        assert "greet-sys" in result._captured_prompt

    def test_explicit_task_bye(self, cli_spec_file):
        result = _cli_run(cli_spec_file, "--task", "bye", "--input", '{"name":"A"}')
        assert result.exit_code == 0
        assert "bye-sys" in result._captured_prompt

    def test_system_prompt_flag_overrides_per_task(self, cli_spec_file):
        result = _cli_run(
            cli_spec_file,
            "--task",
            "greet",
            "--input",
            '{"name":"Alice"}',
            "--system-prompt",
            "cli-override-sys",
        )
        assert result.exit_code == 0
        assert "cli-override-sys" in result._captured_prompt
        assert "greet-sys" not in result._captured_prompt

    def test_user_prompt_flag_overrides_per_task(self, cli_spec_file):
        result = _cli_run(
            cli_spec_file,
            "--task",
            "greet",
            "--input",
            '{"name":"Alice"}',
            "--user-prompt",
            "custom user {{ name }}",
        )
        assert result.exit_code == 0
        assert "custom user Alice" in result._captured_prompt
        assert "Greet Alice" not in result._captured_prompt

    def test_both_prompt_flags(self, cli_spec_file):
        result = _cli_run(
            cli_spec_file,
            "--task",
            "greet",
            "--input",
            '{"name":"Alice"}',
            "--system-prompt",
            "cli-sys",
            "--user-prompt",
            "cli-user",
        )
        assert result.exit_code == 0
        assert "cli-sys" in result._captured_prompt
        assert "cli-user" in result._captured_prompt

    def test_quiet_mode_outputs_valid_json(self, cli_spec_file):
        result = _cli_run(
            cli_spec_file,
            "--task",
            "greet",
            "--input",
            '{"name":"Alice"}',
            fake_response='{"response": "Hello Alice!"}',
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed.get("response") == "Hello Alice!"

    def test_invalid_json_input_exits_nonzero(self, cli_spec_file):
        result = _cli_run(cli_spec_file, "--input", "not-json-and-not-a-file")
        assert result.exit_code != 0

    def test_unknown_task_exits_nonzero(self, cli_spec_file):
        result = _cli_run(
            cli_spec_file, "--task", "nonexistent", "--input", '{"name":"A"}'
        )
        assert result.exit_code != 0

    def test_file_input_single_required_string_field(self, tmp_path, cli_spec_file):
        """Passing a file path as --input works for single-string-field tasks."""
        input_file = tmp_path / "input.txt"
        input_file.write_text("Alice")

        # Greet task has a single required string field — file should be accepted
        captured: dict = {}

        def fake_invoke(prompt: str, config: dict) -> str:
            captured["prompt"] = prompt
            return '{"response": "hi"}'

        with patch("oas_cli.runner.invoke_intelligence", fake_invoke):
            result = cli.invoke(
                app,
                [
                    "run",
                    "--spec",
                    str(cli_spec_file),
                    "--task",
                    "greet",
                    "--input",
                    str(input_file),
                    "--quiet",
                ],
            )

        assert result.exit_code == 0, result.output
        assert "Alice" in captured["prompt"]

    def test_system_prompt_on_bye_task(self, cli_spec_file):
        result = _cli_run(
            cli_spec_file,
            "--task",
            "bye",
            "--input",
            '{"name":"Bob"}',
            "--system-prompt",
            "override for bye",
        )
        assert result.exit_code == 0
        assert "override for bye" in result._captured_prompt
        assert "bye-sys" not in result._captured_prompt


# ===========================================================================
# I. Error cases
# ===========================================================================


class TestErrorCases:
    def test_no_tasks_in_spec_raises(self):
        spec = _spec({})
        with pytest.raises((ValueError, KeyError, Exception)):
            _run(spec, None, {})

    def test_intelligence_missing_model_raises(self):
        spec = _spec(
            {"t": _task(system="s", user="{{ q }}")},
            intelligence={"type": "llm", "engine": "openai"},
        )
        with pytest.raises(ValueError, match="model"):
            _run(spec, "t", {"q": "hi"})

    def test_task_not_found_raises_with_name(self):
        spec = _spec({"real": _task()})
        with pytest.raises(OARunError, match="fake"):
            _run(spec, "fake", {})

    def test_intelligence_not_dict_raises(self):
        spec = _spec({"t": _task()})
        spec["intelligence"] = "not-a-dict"
        with pytest.raises(ValueError):
            _run(spec, "t", {})

    def test_spec_empty_tasks_dict(self):
        spec = _spec({})
        with pytest.raises(OARunError):
            run_task_from_spec(spec, task_name=None, input_data={})

    def test_choose_task_returns_correct_tuple(self):
        tasks = {
            "first": {"description": "f"},
            "second": {"description": "s"},
        }
        name, defn = _choose_task({"tasks": tasks}, "second")
        assert name == "second"
        assert defn == {"description": "s"}

    def test_build_prompt_returns_string(self):
        spec = _spec({"t": _task(system="sys", user="{{ x }}")})
        result = _build_prompt(spec, "t", {"x": "val"})
        assert isinstance(result, str)
        assert "sys" in result
        assert "val" in result
