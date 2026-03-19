"""Tests for the Open Agent Spec CLI commands."""

import re

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
    assert "Open Agent Spec CLI version" in result.output
    version_part = result.output.split("version")[-1].strip()
    assert version_part and re.search(r"\d", version_part), (
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
