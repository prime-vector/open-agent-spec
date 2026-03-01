# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""
Core library API for Open Agent Spec: load spec → validate → generate.

Use this module (or the public `oas_cli` API) when driving generation from
scripts or other tools instead of the CLI.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from .generators import (
    generate_agent_code,
    generate_env_example,
    generate_prompt_template,
    generate_readme,
    generate_requirements,
)
from .validators import validate_spec as validate_spec_data
from .validators import validate_with_json_schema


def validate_spec_file(spec_path: Path) -> Tuple[Dict[str, Any], str, str]:
    """
    Load and validate a spec file.

    Returns (spec_data, agent_name, class_name). Raises ValueError on
    invalid path, invalid YAML, or validation failure (e.g. missing file,
    directory passed as spec, permission error).
    """
    try:
        with open(spec_path) as f:
            spec_data = yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as err:
        raise ValueError(f"Invalid spec path or YAML: {err}") from err

    schema_path = Path(__file__).parent / "schemas" / "oas-schema.json"
    validate_with_json_schema(spec_data, str(schema_path))
    agent_name, class_name = validate_spec_data(spec_data)
    return spec_data, agent_name, class_name


def generate_files(
    output: Path,
    spec_data: Dict[str, Any],
    agent_name: str,
    class_name: str,
    log: logging.Logger,
    console: Optional[Any] = None,
) -> None:
    """
    Generate all agent files into output directory.

    If console is provided (e.g. rich.console.Console), a success message
    is printed. Use console=None when calling from scripts to avoid Rich output.
    """
    try:
        output.mkdir(parents=True, exist_ok=True)

        generate_agent_code(output, spec_data, agent_name, class_name)
        generate_readme(output, spec_data)
        generate_requirements(output, spec_data)
        generate_env_example(output, spec_data)
        generate_prompt_template(output, spec_data)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as err:
        raise RuntimeError(f"Failed to generate agent code: {err}") from err

    log.info("Project initialized")
    log.info("\nNext steps:")
    log.info("1. cd into the output directory")
    log.info("2. Copy .env.example to .env and set your OpenAI key")
    log.info("3. Run: pip install -r requirements.txt")
    log.info("4. Run: python agent.py")

    if console is not None:
        console.print("\n[bold green]✅ Agent project initialized![/] ✨")


def generate(
    spec_path: Path,
    output_dir: Path,
    dry_run: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Load spec, validate, and generate agent files into output_dir.

    Use this from scripts or other tools to drive generation without the CLI.
    If dry_run is True, validation is performed but no files are written.
    Raises ValueError if the spec is invalid.
    """
    spec_data, agent_name, class_name = validate_spec_file(spec_path)
    if dry_run:
        return

    log = logger or logging.getLogger("oas")
    try:
        generate_files(output_dir, spec_data, agent_name, class_name, log, console=None)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as err:
        log.error(f"Error during file generation: {err}")
        raise RuntimeError(f"Failed to generate agent code: {err}") from err
