"""Tests for the Open Agent Spec CLI commands."""

import os

import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        import toml as tomllib  # type: ignore
from typer.testing import CliRunner

from oas_cli.main import app

runner = CliRunner()


def get_version_from_pyproject():
    pyproject_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "pyproject.toml"
    )
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    return pyproject_data["project"]["version"]


def test_version_command():
    """Test that the version command returns the correct version."""
    version = get_version_from_pyproject()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert version in result.output


def test_version_flag():
    """Test that the --version flag works correctly."""
    version = get_version_from_pyproject()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "Open Agent Spec CLI version" in result.output
    assert version in result.output


def test_init_with_minimal_template(tmp_path):
    """Test that oas init --template minimal works correctly."""
    output_dir = tmp_path / "minimal_agent"
    result = runner.invoke(
        app, ["init", "--template", "minimal", "--output", str(output_dir)]
    )

    assert result.exit_code == 0
    assert "Agent project initialized!" in result.output

    # Verify that the key files were created
    agent_py = output_dir / "agent.py"
    prompt_file = output_dir / "prompts" / "greet.jinja2"

    assert agent_py.exists()
    assert prompt_file.exists()

    # Verify the content of the prompt template
    prompt_content = prompt_file.read_text()
    assert "{{ input.name }}" in prompt_content
    assert "Hello {{ input.name }}!" in prompt_content
