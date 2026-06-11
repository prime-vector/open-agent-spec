"""OA Conformance Harness — runtime-agnostic certification driver.

Discovers YAML cases in spec/conformance/cases/, executes them against one or
more runtime adapters (see PROTOCOL.md), asserts the results, and reports a
conformance matrix.

The harness never imports a runtime. Adapters are external executables that
speak JSON over stdin/stdout. The reference adapters live in
spec/conformance/adapters/.

Usage:
    python -m spec.conformance.harness.harness --adapter python
    python -m spec.conformance.harness.harness --adapter node
    python -m spec.conformance.harness.harness --adapter python --adapter node --matrix
    python -m spec.conformance.harness.harness --adapter "./my-runtime-adapter"
    python -m spec.conformance.harness.harness --list

Named adapters ("python", "node") resolve to the bundled reference adapters;
anything else is treated as a shell command.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROTOCOL_VERSION = 1

_CONFORMANCE_DIR = Path(__file__).resolve().parent.parent
_CASES_DIR = _CONFORMANCE_DIR / "cases"
_ADAPTERS_DIR = _CONFORMANCE_DIR / "adapters"

# Categories in execution order (schema first — fast-fail on bad specs).
_CATEGORIES = [
    "schema",
    "prompt-resolution",
    "depends-on",
    "response-format",
    "delegation",
    "errors",
]

# Built-in adapter name → command resolution.
_BUILTIN_ADAPTERS: dict[str, list[str]] = {
    "python": [sys.executable, str(_ADAPTERS_DIR / "python" / "adapter.py")],
    "node": ["node", str(_ADAPTERS_DIR / "node" / "adapter.mjs")],
}

_ADAPTER_TIMEOUT_S = 60


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------


def _load_case(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _is_test_case(case: dict) -> bool:
    return "invoke" in case and ("expect" in case or "expect_error" in case)


def _case_requirements(case: dict) -> set[str]:
    """Capabilities a case requires. Bare cases are implicitly 'core'."""
    req = case.get("requires")
    if req is None:
        return {"core"}
    if isinstance(req, str):
        return {req}
    return set(req)


def discover_cases(category: str | None = None) -> list[tuple[str, Path]]:
    cases: list[tuple[str, Path]] = []
    categories = [category] if category else _CATEGORIES
    # Include any category directories not in the canonical order list
    # (future categories shouldn't need a harness change).
    if category is None:
        known = set(_CATEGORIES)
        extra = sorted(
            d.name for d in _CASES_DIR.iterdir() if d.is_dir() and d.name not in known
        )
        categories = _CATEGORIES + extra

    for cat in categories:
        cat_dir = _CASES_DIR / cat
        if not cat_dir.is_dir():
            continue
        for f in sorted(cat_dir.glob("*.yaml")):
            if _is_test_case(_load_case(f)):
                cases.append((f"{cat}/{f.stem}", f))
    return cases


# ---------------------------------------------------------------------------
# Adapter invocation
# ---------------------------------------------------------------------------


class AdapterError(Exception):
    """The adapter itself failed (crash, bad JSON, protocol mismatch)."""


@dataclass
class Adapter:
    name: str
    command: list[str]
    runtime: str = ""
    version: str = ""
    capabilities: set[str] = field(default_factory=set)

    def query_capabilities(self) -> None:
        out = self._run([*self.command, "--capabilities"], stdin_data=None)
        try:
            manifest = json.loads(out)
        except json.JSONDecodeError as exc:
            raise AdapterError(
                f"Adapter '{self.name}' returned invalid capabilities JSON: {out[:200]!r}"
            ) from exc
        if manifest.get("protocol") != PROTOCOL_VERSION:
            raise AdapterError(
                f"Adapter '{self.name}' speaks protocol {manifest.get('protocol')}, "
                f"harness requires {PROTOCOL_VERSION}"
            )
        self.runtime = manifest.get("runtime", self.name)
        self.version = manifest.get("version", "?")
        self.capabilities = set(manifest.get("capabilities", []))

    def run_case(self, case_payload: dict) -> dict:
        out = self._run(self.command, stdin_data=json.dumps(case_payload))
        try:
            response = json.loads(out)
        except json.JSONDecodeError as exc:
            raise AdapterError(
                f"Adapter '{self.name}' returned invalid result JSON: {out[:500]!r}"
            ) from exc
        if response.get("status") not in ("ok", "error"):
            raise AdapterError(
                f"Adapter '{self.name}' returned unknown status: {response.get('status')!r}"
            )
        return response

    def _run(self, cmd: list[str], stdin_data: str | None) -> str:
        try:
            proc = subprocess.run(
                cmd,
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=_ADAPTER_TIMEOUT_S,
            )
        except FileNotFoundError as exc:
            raise AdapterError(f"Adapter command not found: {cmd[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise AdapterError(
                f"Adapter '{self.name}' timed out after {_ADAPTER_TIMEOUT_S}s"
            ) from exc
        if proc.returncode != 0:
            raise AdapterError(
                f"Adapter '{self.name}' exited {proc.returncode}. "
                f"stderr: {proc.stderr.strip()[:500]}"
            )
        return proc.stdout


def resolve_adapter(name: str) -> Adapter:
    if name in _BUILTIN_ADAPTERS:
        return Adapter(name=name, command=list(_BUILTIN_ADAPTERS[name]))
    # Treat as a shell command (possibly with arguments).
    return Adapter(name=name, command=name.split())


# ---------------------------------------------------------------------------
# Assertion engine (runtime-agnostic — operates on the result envelope JSON)
# ---------------------------------------------------------------------------


class CaseFailureError(Exception):
    """A single conformance case failed."""


def _resolve_dotted(obj: Any, path: str) -> Any:
    for key in path.split("."):
        if isinstance(obj, dict):
            obj = obj[key]
        else:
            raise KeyError(f"Cannot traverse into {type(obj).__name__} at key '{key}'")
    return obj


def _assert_expect(result: dict, expect: dict) -> None:
    for key, expected in expect.items():
        if key == "result.no_key":
            if expected in result:
                raise CaseFailureError(
                    f"Expected key '{expected}' to be absent, but it is present"
                )
            continue
        if key == "result.has_key":
            if expected not in result:
                raise CaseFailureError(
                    f"Expected key '{expected}' to be present, but it is absent"
                )
            continue
        if key == "result.prompt_contains":
            prompt = result.get("prompt", "")
            if expected not in prompt:
                raise CaseFailureError(
                    f"Expected prompt to contain '{expected}', got: {prompt!r}"
                )
            continue
        if key == "result.prompt_contains_all":
            prompt = result.get("prompt", "")
            for fragment in expected:
                if fragment not in prompt:
                    raise CaseFailureError(
                        f"Expected prompt to contain '{fragment}', got: {prompt!r}"
                    )
            continue
        if key == "result.prompt_not_contains":
            prompt = result.get("prompt", "")
            if expected in prompt:
                raise CaseFailureError(
                    f"Expected prompt NOT to contain '{expected}', got: {prompt!r}"
                )
            continue
        if key == "result.prompt_not_contains_all":
            prompt = result.get("prompt", "")
            for fragment in expected:
                if fragment in prompt:
                    raise CaseFailureError(
                        f"Expected prompt NOT to contain '{fragment}', got: {prompt!r}"
                    )
            continue
        if key.startswith("result."):
            path = key[len("result.") :]
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


def _assert_error(error_obj: dict, expect_error: dict) -> None:
    if "code" in expect_error and error_obj.get("code") != expect_error["code"]:
        raise CaseFailureError(
            f"Expected error code '{expect_error['code']}', got '{error_obj.get('code')}'"
        )
    if "stage" in expect_error and error_obj.get("stage") != expect_error["stage"]:
        raise CaseFailureError(
            f"Expected error stage '{expect_error['stage']}', got '{error_obj.get('stage')}'"
        )
    if "has_fields" in expect_error:
        for f in expect_error["has_fields"]:
            if f not in error_obj:
                raise CaseFailureError(
                    f"Expected error to have field '{f}', keys: {list(error_obj.keys())}"
                )


# ---------------------------------------------------------------------------
# Case execution
# ---------------------------------------------------------------------------

# Per-case outcome states.
PASS, FAIL, UNSUPPORTED, ADAPTER_FAIL = "PASS", "FAIL", "UNSUPPORTED", "ERROR"


def run_case_against(adapter: Adapter, case: dict, case_path: Path) -> tuple[str, str]:
    """Run one case; return (outcome, detail)."""
    requirements = _case_requirements(case)
    missing = requirements - adapter.capabilities
    if missing:
        return UNSUPPORTED, f"requires {sorted(missing)}"

    invoke = case["invoke"]
    payload: dict[str, Any] = {
        "protocol": PROTOCOL_VERSION,
        "spec": case["spec"],
        "invoke": {
            "task": invoke.get("task"),
            "input": invoke.get("input", {}),
            "override_system": invoke.get("override_system"),
            "override_user": invoke.get("override_user"),
        },
        "mock_responses": case.get("mock_responses", {}),
        "files_dir": None,
    }

    tmp_dir: Path | None = None
    uses_files = case.get("uses_files", [])
    if uses_files:
        tmp_dir = Path(tempfile.mkdtemp(prefix="oa_conformance_"))
        (tmp_dir / "main.yaml").write_text(case["spec"])
        for fname in uses_files:
            src = case_path.parent / fname
            if not src.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return FAIL, f"helper file '{fname}' not found at {src}"
            shutil.copy2(src, tmp_dir / fname)
        payload["files_dir"] = str(tmp_dir)

    try:
        try:
            response = adapter.run_case(payload)
        except AdapterError as exc:
            return ADAPTER_FAIL, str(exc)

        expect = case.get("expect")
        expect_error = case.get("expect_error")

        if response["status"] == "error":
            if expect_error:
                try:
                    _assert_error(response.get("error", {}), expect_error)
                    return PASS, ""
                except CaseFailureError as exc:
                    return FAIL, str(exc)
            err = response.get("error", {})
            return FAIL, f"unexpected error: [{err.get('code')}] {err.get('error')}"

        # status == ok
        if expect_error:
            return FAIL, (
                f"expected error '{expect_error.get('code')}' but task succeeded"
            )
        if expect:
            try:
                _assert_expect(response.get("result", {}), expect)
            except CaseFailureError as exc:
                return FAIL, str(exc)
        return PASS, ""
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass
class AdapterReport:
    adapter: Adapter
    outcomes: dict[str, tuple[str, str]] = field(default_factory=dict)

    @property
    def counts(self) -> dict[str, int]:
        c = {PASS: 0, FAIL: 0, UNSUPPORTED: 0, ADAPTER_FAIL: 0}
        for outcome, _ in self.outcomes.values():
            c[outcome] += 1
        return c


_OUTCOME_GLYPH = {PASS: "✅", FAIL: "❌", UNSUPPORTED: "⬜", ADAPTER_FAIL: "💥"}


def render_matrix_markdown(reports: list[AdapterReport]) -> str:
    """Render a conformance matrix as a markdown document."""
    all_cases: list[str] = []
    for r in reports:
        for name in r.outcomes:
            if name not in all_cases:
                all_cases.append(name)

    lines = ["# OA Conformance Matrix", ""]
    header = (
        "| Case | "
        + " | ".join(f"{r.adapter.runtime} {r.adapter.version}" for r in reports)
        + " |"
    )
    sep = "|---|" + "---|" * len(reports)
    lines += [header, sep]
    for case_name in all_cases:
        row = [case_name]
        for r in reports:
            outcome, _ = r.outcomes.get(case_name, (UNSUPPORTED, "not run"))
            row.append(_OUTCOME_GLYPH[outcome] + " " + outcome)
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Runtime | Pass | Fail | Unsupported | Adapter errors |")
    lines.append("|---|---|---|---|---|")
    for r in reports:
        c = r.counts
        lines.append(
            f"| {r.adapter.runtime} {r.adapter.version} "
            f"| {c[PASS]} | {c[FAIL]} | {c[UNSUPPORTED]} | {c[ADAPTER_FAIL]} |"
        )
    lines.append("")
    lines.append(
        "Legend: ✅ PASS · ❌ FAIL · ⬜ UNSUPPORTED (capability not declared) · 💥 adapter error"
    )
    lines.append("")
    return "\n".join(lines)


def render_matrix_json(reports: list[AdapterReport]) -> str:
    payload = {
        "protocol": PROTOCOL_VERSION,
        "runtimes": [
            {
                "runtime": r.adapter.runtime,
                "version": r.adapter.version,
                "capabilities": sorted(r.adapter.capabilities),
                "summary": r.counts,
                "cases": {
                    name: {"outcome": outcome, "detail": detail}
                    for name, (outcome, detail) in r.outcomes.items()
                },
            }
            for r in reports
        ],
    }
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_adapter_suite(
    adapter: Adapter,
    category: str | None,
    verbose: bool,
) -> AdapterReport:
    report = AdapterReport(adapter=adapter)
    for name, path in discover_cases(category):
        case = _load_case(path)
        outcome, detail = run_case_against(adapter, case, path)
        report.outcomes[name] = (outcome, detail)
        if verbose:
            suffix = f"  ({detail})" if detail and outcome != PASS else ""
            print(f"  {outcome:<11} {name}{suffix}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="OA conformance harness")
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        help="Adapter to certify: 'python', 'node', or a shell command. Repeatable.",
    )
    parser.add_argument("--category", default=None, help="Run one case category only")
    parser.add_argument("--list", action="store_true", help="List discovered cases")
    parser.add_argument("--matrix", action="store_true", help="Print markdown matrix")
    parser.add_argument(
        "--matrix-out", default=None, help="Write markdown matrix to this file"
    )
    parser.add_argument(
        "--json-out", default=None, help="Write JSON report to this file"
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name, _ in discover_cases(args.category):
            print(f"  {name}")
        return

    adapters = args.adapter or ["python"]
    verbose = not args.quiet
    reports: list[AdapterReport] = []
    any_failed = False

    for adapter_name in adapters:
        adapter = resolve_adapter(adapter_name)
        try:
            adapter.query_capabilities()
        except AdapterError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(2)

        if verbose:
            print(
                f"\n{adapter.runtime} {adapter.version} "
                f"(capabilities: {', '.join(sorted(adapter.capabilities))})"
            )
            print("=" * 60)
        report = run_adapter_suite(adapter, args.category, verbose)
        reports.append(report)
        c = report.counts
        if verbose:
            print("-" * 60)
            print(
                f"  Passed: {c[PASS]}  Failed: {c[FAIL]}  "
                f"Unsupported: {c[UNSUPPORTED]}  Adapter errors: {c[ADAPTER_FAIL]}"
            )
        if c[FAIL] or c[ADAPTER_FAIL]:
            any_failed = True

    if args.matrix or args.matrix_out:
        md = render_matrix_markdown(reports)
        if args.matrix:
            print("\n" + md)
        if args.matrix_out:
            Path(args.matrix_out).write_text(md)
    if args.json_out:
        Path(args.json_out).write_text(render_matrix_json(reports))

    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
