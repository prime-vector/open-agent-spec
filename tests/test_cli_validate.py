"""CLI tests for `oa validate` argument handling (#94).

`validate` is a command group (for `oa validate aac`), so a bare positional
used to be resolved as a subcommand name and fail with "No such command
'<path>'". These tests pin the ergonomics fix: a bare *.yaml/*.yml argument is
treated as an implicit --spec, while subcommand routing and error messages for
genuine typos are preserved.
"""

import shutil
from pathlib import Path

from typer.testing import CliRunner

from oas_cli.main import app

runner = CliRunner()

MINIMAL_SPEC = (
    Path(__file__).parent.parent / "oas_cli" / "templates" / "minimal-agent.yaml"
)


def _flat(output: str) -> str:
    """Collapse rich panel borders/wrapping so substrings match reliably."""
    return " ".join(output.split())


def test_bare_yaml_path_validates():
    result = runner.invoke(app, ["validate", str(MINIMAL_SPEC)])
    assert result.exit_code == 0
    assert "Spec is valid" in result.output


def test_bare_yml_suffix_also_accepted(tmp_path):
    spec = tmp_path / "agent.yml"
    shutil.copy(MINIMAL_SPEC, spec)
    result = runner.invoke(app, ["validate", str(spec)])
    assert result.exit_code == 0
    assert "Spec is valid" in result.output


def test_spec_option_still_works():
    result = runner.invoke(app, ["validate", "--spec", str(MINIMAL_SPEC)])
    assert result.exit_code == 0
    assert "Spec is valid" in result.output


def test_bare_path_and_spec_option_together_is_an_error():
    result = runner.invoke(
        app, ["validate", str(MINIMAL_SPEC), "--spec", str(MINIMAL_SPEC)]
    )
    assert result.exit_code != 0
    assert "not both" in _flat(result.output)


def test_bare_path_to_missing_yaml_is_a_validation_error_not_no_such_command():
    result = runner.invoke(app, ["validate", "missing.yaml"])
    assert result.exit_code == 1
    assert "No such command" not in result.output


def test_aac_subcommand_still_routes(tmp_path):
    result = runner.invoke(app, ["validate", "aac", "--directory", str(tmp_path)])
    assert result.exit_code == 1
    assert ".agents/ not found" in _flat(result.output)


def test_typo_still_errors_and_names_the_valid_forms():
    result = runner.invoke(app, ["validate", "acc"])
    assert result.exit_code != 0
    flat = _flat(result.output)
    assert "No such command 'acc'" in flat
    assert "Valid forms" in flat


def test_non_yaml_argument_names_the_valid_forms():
    result = runner.invoke(app, ["validate", "README.md"])
    assert result.exit_code != 0
    flat = _flat(result.output)
    assert "No such command 'README.md'" in flat
    assert "oa validate aac" in flat
