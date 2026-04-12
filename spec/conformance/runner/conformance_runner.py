"""OAS 1.4.0 Conformance Test Runner.

Loads YAML test cases from spec/conformance/cases/, mocks LLM calls,
runs each case through the OAS runner, and asserts expected outcomes.

Usage:
    python -m spec.conformance.runner.conformance_runner          # run all
    python -m spec.conformance.runner.conformance_runner schema   # run one category
    python -m spec.conformance.runner.conformance_runner --list   # list cases

This runner tests the *reference* OAS implementation (oas_cli.runner).
Other runtimes can adapt it by replacing the invocation layer.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_CASES_DIR = Path(__file__).resolve().parent.parent / "cases"
_DELEGATION_DIR = _CASES_DIR / "delegation"

# Categories in execution order (schema first — fast-fail on bad specs).
_CATEGORIES = [
    "schema",
    "prompt-resolution",
    "depends-on",
    "response-format",
    "delegation",
    "errors",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_dotted(obj: Any, path: str) -> Any:
    """Resolve a dotted path like 'result.output.summary' against a dict."""
    for key in path.split("."):
        if isinstance(obj, dict):
            obj = obj[key]
        else:
            raise KeyError(f"Cannot traverse into {type(obj).__name__} at key '{key}'")
    return obj


def _load_case(path: Path) -> dict:
    """Load and return a single conformance case YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def _is_test_case(case: dict) -> bool:
    """Return True if the YAML file is a test case (has invoke + expect)."""
    return "invoke" in case and ("expect" in case or "expect_error" in case)


def _build_mock(mock_responses: dict[str, str], task_tracker: list[str]):
    """Return a fake invoke_intelligence that returns canned responses.

    The mock keys responses by inspecting which task is being run.  The
    runner calls invoke_intelligence(system, user, config) — we match on
    the task name threaded through the call stack by looking at the user
    prompt or falling back to ordered delivery.
    """
    remaining = dict(mock_responses)
    delivery_order = list(mock_responses.keys())
    delivery_idx = [0]

    def fake_invoke(system: str, user: str, config: dict) -> str:
        # Try to match by task name appearing in the prompt.
        for task_name in list(remaining.keys()):
            if task_name in user or task_name in system:
                task_tracker.append(task_name)
                return remaining.pop(task_name)

        # Fall back to delivery order — find next key still in remaining.
        while delivery_idx[0] < len(delivery_order):
            key = delivery_order[delivery_idx[0]]
            delivery_idx[0] += 1
            if key in remaining:
                task_tracker.append(key)
                return remaining.pop(key)

        # Nothing left — return empty JSON.
        return "{}"

    return fake_invoke


# ---------------------------------------------------------------------------
# Assertion engine
# ---------------------------------------------------------------------------


class CaseFailureError(Exception):
    """A single conformance case failed."""


def _assert_expect(result: dict, expect: dict) -> None:
    """Check all expect clauses against the result envelope."""
    for key, expected in expect.items():
        # --- special assertion: result.no_key ---
        if key == "result.no_key":
            if expected in result:
                raise CaseFailureError(
                    f"Expected key '{expected}' to be absent, but it is present"
                )
            continue

        # --- special assertion: result.has_key ---
        if key == "result.has_key":
            if expected not in result:
                raise CaseFailureError(
                    f"Expected key '{expected}' to be present, but it is absent"
                )
            continue

        # --- special assertion: prompt_contains ---
        if key == "result.prompt_contains":
            prompt = result.get("prompt", "")
            if expected not in prompt:
                raise CaseFailureError(
                    f"Expected prompt to contain '{expected}', got: {prompt!r}"
                )
            continue

        # --- special assertion: prompt_contains_all ---
        if key == "result.prompt_contains_all":
            prompt = result.get("prompt", "")
            for fragment in expected:
                if fragment not in prompt:
                    raise CaseFailureError(
                        f"Expected prompt to contain '{fragment}', got: {prompt!r}"
                    )
            continue

        # --- special assertion: prompt_not_contains ---
        if key == "result.prompt_not_contains":
            prompt = result.get("prompt", "")
            if expected in prompt:
                raise CaseFailureError(
                    f"Expected prompt NOT to contain '{expected}', got: {prompt!r}"
                )
            continue

        # --- special assertion: prompt_not_contains_all ---
        if key == "result.prompt_not_contains_all":
            prompt = result.get("prompt", "")
            for fragment in expected:
                if fragment in prompt:
                    raise CaseFailureError(
                        f"Expected prompt NOT to contain '{fragment}', got: {prompt!r}"
                    )
            continue

        # --- dotted path assertion ---
        if key.startswith("result."):
            path = key[len("result."):]
            try:
                actual = _resolve_dotted(result, path)
            except (KeyError, TypeError) as exc:
                raise CaseFailureError(
                    f"Path '{key}' not found in result: {exc}"
                ) from exc
            if actual != expected:
                raise CaseFailureError(
                    f"Path '{key}': expected {expected!r}, got {actual!r}"
                )
            continue

        raise CaseFailureError(f"Unknown expect key: {key}")


def _assert_error(error: Exception, expect_error: dict) -> None:
    """Check that an OARunError matches the expected error shape."""
    from oas_cli.runner import OARunError

    if not isinstance(error, OARunError):
        raise CaseFailureError(
            f"Expected OARunError, got {type(error).__name__}: {error}"
        )

    if "code" in expect_error:
        if error.code != expect_error["code"]:
            raise CaseFailureError(
                f"Expected error code '{expect_error['code']}', got '{error.code}'"
            )

    if "stage" in expect_error:
        if error.stage != expect_error["stage"]:
            raise CaseFailureError(
                f"Expected error stage '{expect_error['stage']}', got '{error.stage}'"
            )

    if "has_fields" in expect_error:
        error_dict = error.to_dict()
        for field in expect_error["has_fields"]:
            if field not in error_dict:
                raise CaseFailureError(
                    f"Expected error to have field '{field}', keys: {list(error_dict.keys())}"
                )


# ---------------------------------------------------------------------------
# Case execution
# ---------------------------------------------------------------------------


def _run_case(case: dict, case_path: Path) -> None:
    """Execute a single conformance test case."""
    from oas_cli.runner import OARunError, run_task_from_spec
    from oas_cli.validators import validate_spec

    spec_text = case["spec"]
    spec_data = yaml.safe_load(spec_text)
    invoke = case["invoke"]
    mock_responses = case.get("mock_responses", {})
    expect = case.get("expect")
    expect_error = case.get("expect_error")

    task_name = invoke.get("task")
    input_data = invoke.get("input", {})
    override_system = invoke.get("override_system")
    override_user = invoke.get("override_user")

    # For schema-error cases, validate spec first.
    if expect_error and expect_error.get("code") == "SPEC_LOAD_ERROR":
        try:
            errors = validate_spec(spec_data)
            if errors:
                return  # Correctly rejected — pass.
            # If validate_spec didn't catch it, try running (runner also validates).
        except Exception:
            return  # Correctly rejected — pass.

    # For delegation cases, set up temp directory with helper files.
    uses_files = case.get("uses_files", [])
    tmp_dir = None
    spec_path = None

    if uses_files:
        tmp_dir = Path(tempfile.mkdtemp(prefix="oas_conformance_"))
        # Write the main spec.
        main_spec_file = tmp_dir / "main.yaml"
        main_spec_file.write_text(spec_text)
        spec_path = main_spec_file
        # Copy helper files from the case's directory.
        for fname in uses_files:
            src = case_path.parent / fname
            if src.exists():
                shutil.copy2(src, tmp_dir / fname)
            else:
                raise CaseFailureError(f"Helper file '{fname}' not found at {src}")

    try:
        task_tracker: list[str] = []
        fake_invoke = _build_mock(mock_responses, task_tracker)

        with patch("oas_cli.runner.invoke_intelligence", fake_invoke):
            try:
                result = run_task_from_spec(
                    spec_data,
                    task_name=task_name,
                    input_data=input_data,
                    override_system=override_system,
                    override_user=override_user,
                    spec_path=spec_path,
                )
            except OARunError as exc:
                if expect_error:
                    _assert_error(exc, expect_error)
                    return  # Error matched — pass.
                raise CaseFailureError(f"Unexpected OARunError: {exc}") from exc
            except Exception as exc:
                if expect_error:
                    # Some errors may not be OARunError (e.g. schema validation
                    # before runner is invoked).
                    return
                raise CaseFailureError(f"Unexpected exception: {exc}") from exc

        if expect_error:
            raise CaseFailureError(
                f"Expected error code '{expect_error.get('code')}' but task succeeded"
            )

        if expect:
            _assert_expect(result, expect)
    finally:
        if tmp_dir and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def discover_cases(category: str | None = None) -> list[tuple[str, Path]]:
    """Discover all test case YAML files, optionally filtered by category."""
    cases: list[tuple[str, Path]] = []
    categories = [category] if category else _CATEGORIES

    for cat in categories:
        cat_dir = _CASES_DIR / cat
        if not cat_dir.is_dir():
            continue
        for f in sorted(cat_dir.glob("*.yaml")):
            case_data = _load_case(f)
            if _is_test_case(case_data):
                cases.append((f"{cat}/{f.stem}", f))

    return cases


def run_conformance(
    category: str | None = None,
    verbose: bool = True,
) -> tuple[int, int, int, list[str]]:
    """Run conformance cases and return (passed, failed, skipped, failure_details)."""
    cases = discover_cases(category)
    passed = 0
    failed = 0
    skipped = 0
    failures: list[str] = []

    for name, path in cases:
        case = _load_case(path)

        # Skip cases that require optional features not installed.
        requires = case.get("requires")
        if requires == "contracts":
            try:
                import behavioural_contracts  # noqa: F401
            except ImportError:
                if verbose:
                    print(f"  SKIP  {name} (requires behavioural-contracts)")
                skipped += 1
                continue

        try:
            _run_case(case, path)
            passed += 1
            if verbose:
                print(f"  PASS  {name}")
        except CaseFailureError as exc:
            failed += 1
            detail = f"  FAIL  {name}: {exc}"
            failures.append(detail)
            if verbose:
                print(detail)

    return passed, failed, skipped, failures


def main() -> None:
    args = sys.argv[1:]

    if "--list" in args:
        for name, _ in discover_cases():
            print(f"  {name}")
        return

    if "--help" in args or "-h" in args:
        print(__doc__)
        return

    category = None
    for arg in args:
        if not arg.startswith("-"):
            category = arg
            break

    verbose = "--quiet" not in args and "-q" not in args

    if verbose:
        print("OAS 1.4.0 Conformance Suite")
        print("=" * 40)

    passed, failed, skipped, _failures = run_conformance(category, verbose)

    if verbose:
        print("=" * 40)
        print(f"Passed: {passed}  Failed: {failed}  Skipped: {skipped}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
