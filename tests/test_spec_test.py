"""Tests for oas_cli.spec_test path navigation and expectations."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from oas_cli.main import app
from oas_cli.spec_test import (
    SpecTestError,
    check_expectations,
    load_test_definition,
    navigate_value,
    resolve_spec_path,
    run_cases_from_file,
)

runner = CliRunner()


def test_navigate_value_nested_dict_and_index():
    data = {"questions": ["first Q", "second"], "meta": {"n": 1}}
    assert navigate_value(data, "questions[0]") == "first Q"
    assert navigate_value(data, "meta.n") == 1


def test_navigate_value_empty_path_returns_root():
    assert navigate_value({"a": 1}, "") == {"a": 1}


def test_check_expectations_min_length_and_contains():
    out = {"questions": ["What are the main climate risks?", "How do we adapt?"]}
    errs = check_expectations(
        out,
        {
            "output.questions": {"min_length": 2},
            "output.questions[0]": {"contains": "climate"},
        },
    )
    assert errs == []


def test_check_expectations_failure_messages():
    out = {"questions": ["only one"]}
    errs = check_expectations(
        out,
        {"output.questions": {"min_length": 2}},
    )
    assert len(errs) == 1
    assert "min_length" in errs[0]


def test_check_expectations_bad_path_key():
    errs = check_expectations({}, {"foo.bar": {"equals": 1}})
    assert any("output." in e for e in errs)


def test_resolve_spec_path_relative(tmp_path: Path):
    tf = tmp_path / "dir" / "t.test.yaml"
    tf.parent.mkdir(parents=True)
    tf.write_text("x: 1")
    spec = resolve_spec_path(tf, "./agent.yaml")
    assert spec == (tmp_path / "dir" / "agent.yaml").resolve()


def test_load_test_definition_roundtrip(tmp_path: Path):
    p = tmp_path / "x.test.yaml"
    p.write_text("spec: a.yaml\ncases: []\n")
    d = load_test_definition(p)
    assert d["spec"] == "a.yaml"


_MINIMAL_RUNNABLE_SPEC = """\
open_agent_spec: "1.5.0"

agent:
  name: test-agent
  description: test agent
  role: chat

intelligence:
  type: llm
  engine: openai
  model: gpt-4o

tasks:
  greet:
    description: say hello
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
      system: "You are a greeter."
      user: "Hello {{ name }}"

prompts:
  system: "Global system fallback."
  user: "{{ name }}"
"""


def test_run_cases_from_file_with_mock(tmp_path: Path):
    spec = tmp_path / "agent.yaml"
    spec.write_text(_MINIMAL_RUNNABLE_SPEC)
    testf = tmp_path / "agent.test.yaml"
    testf.write_text(
        """\
spec: agent.yaml
cases:
  - name: smoke
    task: greet
    input:
      name: "Bob"
    expect:
      output.response:
        contains: "Bob"
        min_length: 1
"""
    )

    def fake_invoke(system: str, user: str, config: dict) -> str:
        return json.dumps({"response": "Hello Bob from mock"})

    with patch("oas_cli.runner.invoke_intelligence", fake_invoke):
        results, resolved = run_cases_from_file(testf)
    assert resolved == spec.resolve()
    assert len(results) == 1
    assert results[0].passed
    assert results[0].errors == []


def test_run_cases_expect_failure(tmp_path: Path):
    spec = tmp_path / "agent.yaml"
    spec.write_text(_MINIMAL_RUNNABLE_SPEC)
    testf = tmp_path / "agent.test.yaml"
    testf.write_text(
        """\
spec: agent.yaml
cases:
  - task: greet
    input: { name: "x" }
    expect:
      output.response:
        contains: "ZZZNOTFOUND"
"""
    )

    def fake_invoke(system: str, user: str, config: dict) -> str:
        return json.dumps({"response": "Hello x"})

    with patch("oas_cli.runner.invoke_intelligence", fake_invoke):
        results, _ = run_cases_from_file(testf)
    assert not results[0].passed
    assert any("contains" in e for e in results[0].errors)


def test_run_cases_invalid_test_file_raises():
    with pytest.raises(SpecTestError):
        run_cases_from_file(Path("/nonexistent/nope.test.yaml"))


def test_oa_test_cli_quiet_json(tmp_path: Path):
    spec = tmp_path / "agent.yaml"
    spec.write_text(_MINIMAL_RUNNABLE_SPEC)
    testf = tmp_path / "agent.test.yaml"
    testf.write_text(
        """\
spec: agent.yaml
cases:
  - task: greet
    input: { name: "Bob" }
    expect:
      output.response: { contains: "Bob" }
"""
    )

    def fake_invoke(system: str, user: str, config: dict) -> str:
        return json.dumps({"response": "Hi Bob"})

    with patch("oas_cli.runner.invoke_intelligence", fake_invoke):
        result = runner.invoke(app, ["test", str(testf), "--quiet"])
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["passed"] == 1
    assert summary["failed"] == 0
    assert summary["cases"][0]["passed"] is True
