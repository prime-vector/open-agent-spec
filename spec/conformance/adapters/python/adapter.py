#!/usr/bin/env python3
"""Conformance adapter for the reference Python runtime (oas_cli).

Speaks the OA Conformance Adapter Protocol v1 (see ../../PROTOCOL.md):

    adapter.py --capabilities      → capability manifest on stdout
    adapter.py  (case on stdin)    → result JSON on stdout

Mocking strategy: patches oas_cli.runner.invoke_intelligence with a canned
responder driven by the case's mock_responses (matched by task name appearing
in the rendered prompt, falling back to declaration order).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

# Make the repo root importable when invoked as a script.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

PROTOCOL_VERSION = 1


def _capabilities() -> dict:
    try:
        from importlib.metadata import version as _pkg_version

        runtime_version = _pkg_version("open-agent-spec")
    except Exception:
        runtime_version = "dev"

    caps = [
        "core",
        "depends-on",
        "delegation",
        "registry",
        "response-format-text",
        "output-schema-validation",
        "history",
        "tools",
        "sandbox",
    ]
    try:
        import behavioural_contracts  # noqa: F401

        caps.append("contracts")
    except ImportError:
        pass
    return {
        "protocol": PROTOCOL_VERSION,
        "runtime": "python-reference",
        "version": runtime_version,
        "capabilities": caps,
    }


def _build_mock(mock_responses: dict[str, str]):
    remaining = dict(mock_responses)
    delivery_order = list(mock_responses.keys())
    delivery_idx = [0]

    def fake_invoke(system: str, user: str, config: dict, history=None) -> str:
        for task_name in list(remaining.keys()):
            if task_name in user or task_name in system:
                return remaining.pop(task_name)
        while delivery_idx[0] < len(delivery_order):
            key = delivery_order[delivery_idx[0]]
            delivery_idx[0] += 1
            if key in remaining:
                return remaining.pop(key)
        return "{}"

    return fake_invoke


def _run(case: dict) -> dict:
    from jsonschema import validate as _schema_validate
    from jsonschema.exceptions import ValidationError as _SchemaValidationError

    from oas_cli.runner import OARunError, run_task_from_spec

    spec_data = yaml.safe_load(case["spec"])
    invoke = case.get("invoke") or {}
    files_dir = case.get("files_dir")
    spec_path = Path(files_dir) / "main.yaml" if files_dir else None

    # Validate against the canonical JSON Schema — the normative artifact.
    # (The CLI's hand-written validator is a UX layer and may be stricter
    # than the standard; conformance is judged against the schema.)
    schema_path = _REPO_ROOT / "oas_cli" / "schemas" / "oas-schema.json"
    schema = json.loads(schema_path.read_text())
    try:
        _schema_validate(instance=spec_data, schema=schema)
    except _SchemaValidationError as exc:
        return {
            "protocol": PROTOCOL_VERSION,
            "status": "error",
            "error": {
                "error": exc.message,
                "code": "SPEC_LOAD_ERROR",
                "stage": "load",
            },
        }

    fake_invoke = _build_mock(case.get("mock_responses") or {})

    try:
        with patch("oas_cli.runner.invoke_intelligence", fake_invoke):
            result = run_task_from_spec(
                spec_data,
                task_name=invoke.get("task"),
                input_data=invoke.get("input") or {},
                override_system=invoke.get("override_system"),
                override_user=invoke.get("override_user"),
                spec_path=spec_path,
            )
        return {"protocol": PROTOCOL_VERSION, "status": "ok", "result": result}
    except OARunError as exc:
        return {
            "protocol": PROTOCOL_VERSION,
            "status": "error",
            "error": exc.to_dict(),
        }
    except Exception as exc:  # Defensive: surface anything else as a RUN_ERROR.
        return {
            "protocol": PROTOCOL_VERSION,
            "status": "error",
            "error": {"error": str(exc), "code": "RUN_ERROR", "stage": "run"},
        }


def main() -> None:
    if "--capabilities" in sys.argv:
        print(json.dumps(_capabilities()))
        return
    case = json.loads(sys.stdin.read())
    print(json.dumps(_run(case)))


if __name__ == "__main__":
    main()
