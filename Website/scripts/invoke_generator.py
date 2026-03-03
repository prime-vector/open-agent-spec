#!/usr/bin/env python3
"""
Invoke the real Open Agent Spec generator for the Website playground.
Usage: python invoke_generator.py <spec_path> <output_dir>
Reads spec from spec_path, runs oas_cli generation, then prints JSON to stdout with
keys: agentPy, readme, requirementsTxt, envExample, prompts (object of filename -> content).
Exits 0 on success, 1 on validation/generation error (stderr has message; stdout is empty or error JSON).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Repo root is parent of Website (parent of this script's parent)
_SCRIPT_DIR = Path(__file__).resolve().parent
_WEBSITE_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _WEBSITE_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import after path setup so oas_cli from repo root is found (E402: intentional)
from oas_cli.core import validate_spec_file  # noqa: E402
from oas_cli.generators import (  # noqa: E402
    generate_agent_code,
    generate_env_example,
    generate_prompt_template,
    generate_readme,
    generate_requirements,
)


def main() -> int:
    logging.basicConfig(level=logging.WARNING)

    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: invoke_generator.py <spec_path> <output_dir>"}), file=sys.stderr)
        return 1

    spec_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not spec_path.is_file():
        print(json.dumps({"error": f"Spec file not found: {spec_path}"}), file=sys.stderr)
        return 1

    try:
        spec_data, agent_name, class_name = validate_spec_file(spec_path)
    except ValueError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        generate_agent_code(output_dir, spec_data, agent_name, class_name)
        generate_readme(output_dir, spec_data)
        generate_requirements(output_dir, spec_data)
        generate_env_example(output_dir, spec_data)
        generate_prompt_template(output_dir, spec_data)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as e:
        print(json.dumps({"error": f"Generation failed: {e}"}), file=sys.stderr)
        return 1

    # Collect generated file contents for JSON response
    result = {}
    agent_py = output_dir / "agent.py"
    if agent_py.exists():
        result["agentPy"] = agent_py.read_text(encoding="utf-8")
    readme = output_dir / "README.md"
    if readme.exists():
        result["readme"] = readme.read_text(encoding="utf-8")
    req = output_dir / "requirements.txt"
    if req.exists():
        result["requirementsTxt"] = req.read_text(encoding="utf-8")
    env_example = output_dir / ".env.example"
    if env_example.exists():
        result["envExample"] = env_example.read_text(encoding="utf-8")

    prompts_dir = output_dir / "prompts"
    if prompts_dir.is_dir():
        result["prompts"] = {}
        for f in prompts_dir.iterdir():
            if f.suffix in (".jinja2", ".j2") and f.is_file():
                result["prompts"][f.name] = f.read_text(encoding="utf-8")

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
