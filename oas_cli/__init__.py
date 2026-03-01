# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""
Open Agent Spec CLI — scaffold AI agent projects from YAML specs.

Public API for use from scripts and other tools:

  from oas_cli import generate, validate_spec

  # Validate a spec file (returns spec_data, agent_name, class_name)
  spec_data, agent_name, class_name = validate_spec(Path("my-spec.yaml"))

  # Generate agent project (raises on invalid spec or write error)
  generate(Path("my-spec.yaml"), Path("./output"), dry_run=False)
"""

from pathlib import Path
from typing import Any

from .core import generate, validate_spec_file


def validate_spec(spec_path: Path) -> tuple[dict[str, Any], str, str]:
    """
    Load and validate a spec file.

    Returns (spec_data, agent_name, class_name). Raises ValueError on
    invalid path or YAML (e.g. missing file, directory, permission error)
    or validation failure.
    """
    return validate_spec_file(spec_path)


__all__ = ["generate", "validate_spec"]
