"""Tests for the public library API (oas_cli.generate, oas_cli.validate_spec)."""

from pathlib import Path

import pytest

from oas_cli import generate, validate_spec


def test_validate_spec_returns_spec_data_and_names():
    """validate_spec(spec_path) returns (spec_data, agent_name, class_name)."""
    spec_path = (
        Path(__file__).parent.parent / "oas_cli" / "templates" / "minimal-agent.yaml"
    )
    spec_data, agent_name, class_name = validate_spec(spec_path)
    assert isinstance(spec_data, dict)
    assert "agent" in spec_data
    assert agent_name == "hello_world_agent"
    assert class_name == "HelloWorldAgent"


def test_validate_spec_raises_on_missing_file():
    """validate_spec raises ValueError for missing file."""
    with pytest.raises(ValueError, match="Invalid spec path|Invalid YAML"):
        validate_spec(Path("/nonexistent/spec.yaml"))


def test_validate_spec_raises_on_directory():
    """validate_spec raises ValueError when path is a directory (not a file)."""
    with pytest.raises(ValueError, match="Invalid spec path|Invalid YAML"):
        validate_spec(Path(__file__).parent)  # tests/ is a directory


def test_generate_raises_value_error_on_directory_spec(tmp_path):
    """generate() raises ValueError when spec_path is a directory."""
    with pytest.raises(ValueError, match="Invalid spec path|Invalid YAML"):
        generate(Path(__file__).parent, tmp_path, dry_run=False)


def test_generate_dry_run_does_not_write(tmp_path):
    """generate(..., dry_run=True) validates but writes no files."""
    spec_path = (
        Path(__file__).parent.parent / "oas_cli" / "templates" / "minimal-agent.yaml"
    )
    generate(spec_path, tmp_path, dry_run=True)
    assert not list(tmp_path.iterdir()), "dry_run should create no files"


def test_generate_writes_files(tmp_path):
    """generate(spec_path, output_dir) produces agent files."""
    spec_path = (
        Path(__file__).parent.parent / "oas_cli" / "templates" / "minimal-agent.yaml"
    )
    generate(spec_path, tmp_path, dry_run=False)
    assert (tmp_path / "agent.py").exists()
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "requirements.txt").exists()
    assert (tmp_path / ".env.example").exists()
    assert (tmp_path / "prompts" / "agent_prompt.jinja2").exists()
