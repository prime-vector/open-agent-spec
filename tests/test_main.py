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


def test_init_with_directory_spec_returns_clean_error(tmp_path):
    """init --spec <directory> exits with error and no traceback (ValueError normalized)."""
    result = runner.invoke(app, ["init", "--spec", str(tmp_path), "--output", str(tmp_path / "out")])
    assert result.exit_code != 0
    # Should show a user-facing error, not a Python traceback
    assert "Traceback" not in result.output
    assert "ValueError" in result.output or "Invalid spec" in result.output or "Invalid YAML" in result.output
