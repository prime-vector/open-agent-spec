"""Tests for the Open Agent Spec CLI commands."""

import json
import re
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from oas_cli.main import app

runner = CliRunner()


def test_version_command():
    """Test that the version command returns a valid version string."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "Open Agent Spec CLI version" in result.output
    # CLI reports installed package version; assert it looks like a version (non-empty, contains digits)
    version_part = result.output.split("version")[-1].strip()
    assert version_part and re.search(r"\d", version_part), (
        "Version output should contain a version-like string"
    )


def test_version_flag():
    """Test that the --version flag works correctly."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    # New banner shows "Open Agent Spec" text and a version digit (e.g. 1.5.0)
    assert "Open Agent Spec" in result.output
    assert re.search(r"\d+\.\d+", result.output), (
        "Version output should contain a version-like string"
    )


def test_init_aac_creates_agents_example(tmp_path):
    """init aac creates .agents/example.yaml."""
    result = runner.invoke(app, ["init", "aac", "--directory", str(tmp_path), "-q"])
    assert result.exit_code == 0
    example = tmp_path / ".agents" / "example.yaml"
    review = tmp_path / ".agents" / "review.yaml"
    change_diff = tmp_path / ".agents" / "change.diff"
    assert example.is_file()
    assert review.is_file()
    assert change_diff.is_file()
    text = example.read_text()
    diff_text = change_diff.read_text()
    assert "open_agent_spec" in text
    assert "hello-world-agent" in text
    assert "diff --git" in diff_text


def test_init_without_output_shows_helpful_message():
    """init with no subcommand and no --output exits 1 with hint."""
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 1
    assert "init aac" in result.output or "aac" in result.output


def test_init_with_directory_spec_returns_clean_error(tmp_path):
    """init --spec <directory> exits with error and no traceback (ValueError normalized)."""
    result = runner.invoke(
        app, ["init", "--spec", str(tmp_path), "--output", str(tmp_path / "out")]
    )
    assert result.exit_code != 0
    # Should show a user-facing error, not a Python traceback
    assert "Traceback" not in result.output
    assert (
        "ValueError" in result.output
        or "Invalid spec" in result.output
        or "Invalid YAML" in result.output
    )


# ---------------------------------------------------------------------------
# oa run --system-prompt / --user-prompt CLI flag tests
# ---------------------------------------------------------------------------

_MINIMAL_SPEC = """\
open_agent_spec: "1.5.0"

agent:
  name: test-agent
  description: test agent

intelligence:
  type: llm
  engine: openai
  model: gpt-4o

tasks:
  greet:
    description: say hello
    output:
      type: object
      properties:
        response: { type: string }
      required: [response]
    prompts:
      system: "spec system prompt"
      user: "Hello {{ name }}"

prompts:
  system: "global system prompt"
  user: "{{ name }}"
"""


def _write_spec(tmp_path: Path) -> Path:
    spec_file = tmp_path / "agent.yaml"
    spec_file.write_text(_MINIMAL_SPEC)
    return spec_file


def test_run_system_prompt_override_reaches_runner(tmp_path):
    """--system-prompt passed to oa run replaces the spec system prompt."""
    spec_file = _write_spec(tmp_path)
    captured: dict = {}

    def fake_invoke(system: str, user: str, config: dict, history=None) -> str:
        captured["prompt"] = f"{system}\n\n{user}"
        return '{"response": "hi"}'

    with patch("oas_cli.runner.invoke_intelligence", fake_invoke):
        result = runner.invoke(
            app,
            [
                "run",
                "--spec",
                str(spec_file),
                "--task",
                "greet",
                "--input",
                '{"name": "Alice"}',
                "--system-prompt",
                "override sys",
                "--quiet",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "override sys" in captured["prompt"]
    assert "spec system prompt" not in captured["prompt"]
    assert "global system prompt" not in captured["prompt"]


def test_run_user_prompt_override_reaches_runner(tmp_path):
    """--user-prompt passed to oa run replaces the spec user template."""
    spec_file = _write_spec(tmp_path)
    captured: dict = {}

    def fake_invoke(system: str, user: str, config: dict, history=None) -> str:
        captured["prompt"] = f"{system}\n\n{user}"
        return '{"response": "hi"}'

    with patch("oas_cli.runner.invoke_intelligence", fake_invoke):
        result = runner.invoke(
            app,
            [
                "run",
                "--spec",
                str(spec_file),
                "--task",
                "greet",
                "--input",
                '{"name": "Alice"}',
                "--user-prompt",
                "custom user {{ name }}",
                "--quiet",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "custom user Alice" in captured["prompt"]
    assert "Hello Alice" not in captured["prompt"]


def test_run_no_overrides_uses_per_task_prompt(tmp_path):
    """Without overrides, the per-task prompts.system is used (Style A)."""
    spec_file = _write_spec(tmp_path)
    captured: dict = {}

    def fake_invoke(system: str, user: str, config: dict, history=None) -> str:
        captured["prompt"] = f"{system}\n\n{user}"
        return '{"response": "hi"}'

    with patch("oas_cli.runner.invoke_intelligence", fake_invoke):
        result = runner.invoke(
            app,
            [
                "run",
                "--spec",
                str(spec_file),
                "--task",
                "greet",
                "--input",
                '{"name": "Alice"}',
                "--quiet",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "spec system prompt" in captured["prompt"]
    assert "global system prompt" not in captured["prompt"]


def test_run_output_is_valid_json_in_quiet_mode(tmp_path):
    """oa run --quiet outputs valid JSON."""
    spec_file = _write_spec(tmp_path)

    def fake_invoke(system: str, user: str, config: dict, history=None) -> str:
        return '{"response": "Hello Alice!"}'

    with patch("oas_cli.runner.invoke_intelligence", fake_invoke):
        result = runner.invoke(
            app,
            [
                "run",
                "--spec",
                str(spec_file),
                "--task",
                "greet",
                "--input",
                '{"name": "Alice"}',
                "--quiet",
            ],
        )

    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert parsed.get("response") == "Hello Alice!"


# ---------------------------------------------------------------------------
# oa run --usage PATH
# ---------------------------------------------------------------------------

_CHAIN_SPEC = """\
open_agent_spec: "1.5.0"

agent:
  name: test-agent
  description: test agent

intelligence:
  type: llm
  engine: openai
  model: gpt-4o

tasks:
  extract:
    description: extract facts
    output:
      type: object
      properties:
        facts: { type: string }
      required: [facts]
    prompts:
      user: "extract facts"
  summarize:
    description: summarize facts
    depends_on: [extract]
    output:
      type: object
      properties:
        summary: { type: string }
      required: [summary]

prompts:
  system: "you summarize"
  user: "{{ facts }}"
"""


def test_run_usage_flag_writes_file_and_leaves_quiet_stdout_clean(tmp_path):
    """--usage writes the report to a file; --quiet stdout stays output-only."""
    spec_file = _write_spec(tmp_path)
    usage_file = tmp_path / "usage.json"

    def fake_invoke(system: str, user: str, config: dict, history=None) -> str:
        from oas_cli.providers.registry import record_usage

        record_usage(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "gpt-4o",
        )
        return '{"response": "Hello Alice!"}'

    with patch("oas_cli.runner.invoke_intelligence", fake_invoke):
        result = runner.invoke(
            app,
            [
                "run",
                "--spec",
                str(spec_file),
                "--task",
                "greet",
                "--input",
                '{"name": "Alice"}',
                "--quiet",
                "--usage",
                str(usage_file),
            ],
        )

    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert parsed == {"response": "Hello Alice!"}
    assert "usage" not in parsed
    report = json.loads(usage_file.read_text(encoding="utf-8"))
    assert report["task"] == "greet"
    assert report["usage"]["total_tokens"] == 15
    assert report["total"]["total_tokens"] == 15
    assert "chain" not in report


def test_run_usage_flag_rolls_up_depends_on_chain(tmp_path):
    """--usage total includes dependency spend, not just the leaf task."""
    spec_file = tmp_path / "agent.yaml"
    spec_file.write_text(_CHAIN_SPEC)
    usage_file = tmp_path / "usage.json"

    def fake_invoke(system: str, user: str, config: dict, history=None) -> str:
        from oas_cli.providers.registry import record_usage

        if "extract facts" in user:
            record_usage(
                {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                "gpt-4o",
            )
            return '{"facts": "the sky is blue"}'
        record_usage(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "gpt-4o",
        )
        return '{"summary": "sky=blue"}'

    with patch("oas_cli.runner.invoke_intelligence", fake_invoke):
        result = runner.invoke(
            app,
            [
                "run",
                "--spec",
                str(spec_file),
                "--task",
                "summarize",
                "--quiet",
                "--usage",
                str(usage_file),
            ],
        )

    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert parsed == {"summary": "sky=blue"}
    report = json.loads(usage_file.read_text(encoding="utf-8"))
    assert report["task"] == "summarize"
    assert report["usage"]["total_tokens"] == 15
    assert report["chain"]["extract"]["usage"]["total_tokens"] == 30
    assert report["total"]["total_tokens"] == 45
    assert report["total"]["total_tokens"] != report["usage"]["total_tokens"]
    assert "estimated_cost_usd" in report["total"]


def test_run_usage_flag_works_without_quiet(tmp_path):
    """--usage writes the same report in non-quiet mode."""
    spec_file = _write_spec(tmp_path)
    usage_file = tmp_path / "usage.json"

    def fake_invoke(system: str, user: str, config: dict, history=None) -> str:
        from oas_cli.providers.registry import record_usage

        record_usage(
            {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
            "gpt-4o",
        )
        return '{"response": "hi"}'

    with patch("oas_cli.runner.invoke_intelligence", fake_invoke):
        result = runner.invoke(
            app,
            [
                "run",
                "--spec",
                str(spec_file),
                "--task",
                "greet",
                "--input",
                '{"name": "Alice"}',
                "--usage",
                str(usage_file),
            ],
        )

    assert result.exit_code == 0, result.output
    report = json.loads(usage_file.read_text(encoding="utf-8"))
    assert report["usage"]["total_tokens"] == 10
    assert report["total"]["total_tokens"] == 10
