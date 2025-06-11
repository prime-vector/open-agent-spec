"""Tests for the Open Agent Spec CLI commands."""
import pytest
from typer.testing import CliRunner
from oas_cli.main import app

runner = CliRunner()

def test_version_command():
    """Test that the version command returns the correct version."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.8" in result.stdout  # This should match the version in pyproject.toml 