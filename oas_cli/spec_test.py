# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""Load YAML eval cases for a spec, run tasks, and assert on structured output."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .core import validate_spec_file
from .runner import OARunError, run_task_from_file


class SpecTestError(Exception):
    """Invalid test file or configuration."""


def load_test_definition(path: Path) -> dict[str, Any]:
    """Parse a ``*.test.yaml`` (or any YAML) test definition file."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SpecTestError(f"Cannot read test file: {path}") from exc
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise SpecTestError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SpecTestError("Test file must decode to a YAML mapping at the top level")
    return data


def resolve_spec_path(test_file: Path, spec_ref: Any) -> Path:
    """Resolve ``spec:`` relative to the test file's directory."""
    if not isinstance(spec_ref, str) or not spec_ref.strip():
        raise SpecTestError("Test file must set a non-empty string 'spec:' path")
    ref = spec_ref.strip()
    p = Path(ref)
    if not p.is_absolute():
        p = (test_file.parent / p).resolve()
    return p


_SEGMENT_RE = re.compile(r"^([^[\]]+)(\[(\d+)\])?$")


def _parse_segment(seg: str) -> tuple[str, int | None]:
    seg = seg.strip()
    if not seg:
        raise SpecTestError(f"Invalid empty path segment in {seg!r}")
    m = _SEGMENT_RE.match(seg)
    if not m:
        raise SpecTestError(f"Invalid path segment: {seg!r}")
    name, bracket, idx_s = m.group(1), m.group(2), m.group(3)
    if not name:
        raise SpecTestError(f"Invalid path segment: {seg!r}")
    idx = int(idx_s) if bracket else None
    return name, idx


def navigate_value(root: Any, dotted_path: str) -> Any:
    """Follow a path like ``questions[0].title`` starting from *root* (a dict or list)."""
    path = dotted_path.strip()
    if not path:
        return root
    current: Any = root
    for seg in path.split("."):
        key, idx = _parse_segment(seg)
        if isinstance(current, dict):
            if key not in current:
                raise KeyError(key)
            current = current[key]
        else:
            raise TypeError(f"cannot index non-mapping with {key!r}")
        if idx is not None:
            if not isinstance(current, (list, tuple)):
                raise TypeError(
                    f"segment [{idx}] requires a list, got {type(current).__name__}"
                )
            if idx < 0 or idx >= len(current):
                raise IndexError(
                    f"index {idx} out of range for path (len={len(current)})"
                )
            current = current[idx]
    return current


def _output_path(expect_key: str) -> str:
    """Strip optional ``output.`` prefix; remainder is navigated from task output dict."""
    key = expect_key.strip()
    if key.startswith("output."):
        return key[len("output.") :]
    if key == "output":
        return ""
    raise SpecTestError(
        f"Expectation key must start with 'output.' or be 'output', got {expect_key!r}"
    )


def _check_rule(
    value: Any,
    rule_name: str,
    rule_val: Any,
    *,
    case_sensitive: bool,
) -> str | None:
    """Return an error message if the rule fails, else None."""
    if rule_name == "min_length":
        if not isinstance(rule_val, int) or rule_val < 0:
            return f"min_length must be a non-negative int, got {rule_val!r}"
        try:
            ln = len(value)  # type: ignore[arg-type]
        except TypeError:
            return f"min_length: value is not sized (got {type(value).__name__})"
        if ln < rule_val:
            return f"min_length: expected len >= {rule_val}, got {ln}"
        return None

    if rule_name == "max_length":
        if not isinstance(rule_val, int) or rule_val < 0:
            return f"max_length must be a non-negative int, got {rule_val!r}"
        try:
            ln = len(value)  # type: ignore[arg-type]
        except TypeError:
            return f"max_length: value is not sized (got {type(value).__name__})"
        if ln > rule_val:
            return f"max_length: expected len <= {rule_val}, got {ln}"
        return None

    if rule_name == "contains":
        if not isinstance(rule_val, str):
            return f"contains: expected string needle, got {type(rule_val).__name__}"
        hay = value if isinstance(value, str) else json.dumps(value, default=str)
        if not case_sensitive:
            ok = rule_val.lower() in hay.lower()
        else:
            ok = rule_val in hay
        if not ok:
            return f"contains: expected substring {rule_val!r} not found in value"
        return None

    if rule_name == "equals":
        if value != rule_val:
            return f"equals: expected {rule_val!r}, got {value!r}"
        return None

    if rule_name == "type":
        if rule_val == "string":
            if not isinstance(value, str):
                return f"type: expected string, got {type(value).__name__}"
        elif rule_val == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return f"type: expected number, got {type(value).__name__}"
        elif rule_val == "boolean":
            if not isinstance(value, bool):
                return f"type: expected boolean, got {type(value).__name__}"
        elif rule_val in ("object", "dict"):
            if not isinstance(value, dict):
                return f"type: expected object, got {type(value).__name__}"
        elif rule_val in ("array", "list"):
            if not isinstance(value, list):
                return f"type: expected array, got {type(value).__name__}"
        else:
            return f"type: unsupported type label {rule_val!r}"
        return None

    if rule_name == "case_sensitive":
        return None  # handled as meta-flag

    return f"unknown expectation rule {rule_name!r}"


def check_expectations(
    output_obj: Any,
    expect_block: dict[str, Any],
) -> list[str]:
    """Evaluate all paths in *expect_block* against *output_obj* (task output). Return errors."""
    errors: list[str] = []
    if not isinstance(expect_block, dict):
        return [f"expect must be a mapping, got {type(expect_block).__name__}"]

    for path_key, rules in expect_block.items():
        if not isinstance(path_key, str):
            errors.append(f"expect keys must be strings, got {path_key!r}")
            continue
        try:
            subpath = _output_path(path_key)
        except SpecTestError as exc:
            errors.append(str(exc))
            continue

        if not isinstance(rules, dict):
            errors.append(
                f"{path_key}: rules must be a mapping, got {type(rules).__name__}"
            )
            continue

        case_sensitive = bool(rules.get("case_sensitive", False))

        try:
            if subpath == "":
                at = output_obj
            else:
                if not isinstance(output_obj, dict):
                    errors.append(
                        f"{path_key}: output is not an object, cannot navigate path"
                    )
                    continue
                at = navigate_value(output_obj, subpath)
        except KeyError as exc:
            errors.append(f"{path_key}: missing key {exc!s}")
            continue
        except (TypeError, IndexError) as exc:
            errors.append(f"{path_key}: {exc}")
            continue

        for rule_name, rule_val in rules.items():
            msg = _check_rule(at, rule_name, rule_val, case_sensitive=case_sensitive)
            if msg:
                errors.append(f"{path_key}: {msg}")

    return errors


@dataclass
class CaseResult:
    name: str
    task: str
    passed: bool
    errors: list[str] = field(default_factory=list)
    envelope: dict[str, Any] | None = None


def run_cases_from_file(test_file: Path) -> tuple[list[CaseResult], Path]:
    """Validate spec, run each case, apply expectations. Returns results and spec path."""
    definition = load_test_definition(test_file)
    spec_path = resolve_spec_path(test_file, definition.get("spec"))
    validate_spec_file(spec_path)

    cases_raw = definition.get("cases")
    if not isinstance(cases_raw, list):
        raise SpecTestError("'cases' must be a list")

    results: list[CaseResult] = []
    for i, case in enumerate(cases_raw):
        if not isinstance(case, dict):
            raise SpecTestError(f"cases[{i}] must be a mapping")
        name = case.get("name")
        if name is None:
            name = f"case_{i}"
        elif not isinstance(name, str):
            raise SpecTestError(f"cases[{i}].name must be a string")

        task = case.get("task")
        if task is not None and not isinstance(task, str):
            raise SpecTestError(f"cases[{i}].task must be a string")

        inp = case.get("input")
        if inp is None:
            input_data: dict[str, Any] = {}
        elif isinstance(inp, dict):
            input_data = dict(inp)
        else:
            raise SpecTestError(f"cases[{i}].input must be a mapping or omitted")

        expect = case.get("expect")
        if expect is None:
            expect = {}
        if not isinstance(expect, dict):
            raise SpecTestError(f"cases[{i}].expect must be a mapping or omitted")

        try:
            envelope = run_task_from_file(
                spec_path,
                task_name=task,
                input_data=input_data,
            )
        except OARunError as exc:
            results.append(
                CaseResult(
                    name=name,
                    task=task or "(default)",
                    passed=False,
                    errors=[f"run error [{exc.code}]: {exc}"],
                )
            )
            continue
        except Exception as exc:  # pragma: no cover — defensive
            results.append(
                CaseResult(
                    name=name,
                    task=task or "(default)",
                    passed=False,
                    errors=[f"unexpected error: {exc}"],
                )
            )
            continue

        out = envelope.get("output")
        errs = check_expectations(out, expect) if expect else []
        results.append(
            CaseResult(
                name=name,
                task=str(envelope.get("task", task or "")),
                passed=len(errs) == 0,
                errors=errs,
                envelope=envelope,
            )
        )

    return results, spec_path
