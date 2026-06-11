"""DEPRECATED shim — the conformance runner moved to the adapter architecture.

The suite is now driven by a runtime-agnostic harness that certifies any OA
runtime through subprocess adapters (see ../PROTOCOL.md):

    python -m spec.conformance.harness.harness --adapter python
    python -m spec.conformance.harness.harness --adapter node
    python -m spec.conformance.harness.harness --adapter python --adapter node --matrix

This module forwards to the harness with the Python reference adapter so
existing invocations of `python -m spec.conformance.runner.conformance_runner`
keep working. Positional category arguments are translated to --category.
"""

from __future__ import annotations

import sys


def main() -> None:
    from spec.conformance.harness import harness

    print(
        "NOTE: conformance_runner is deprecated — use "
        "`python -m spec.conformance.harness.harness --adapter python`\n",
        file=sys.stderr,
    )

    argv = ["--adapter", "python"]
    for arg in sys.argv[1:]:
        if arg == "--list":
            argv.append("--list")
        elif not arg.startswith("-"):
            argv += ["--category", arg]
        else:
            argv.append(arg)

    sys.argv = [sys.argv[0], *argv]
    harness.main()


if __name__ == "__main__":
    main()
