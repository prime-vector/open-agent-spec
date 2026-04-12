#!/usr/bin/env python3
"""Run a multi-agent orchestration workflow.

Usage:
    # Inline — prints results to stdout
    python run.py "Write a blog post about AI agent frameworks"

    # With dashboard UI
    python run.py --dashboard

    # Custom specs
    python run.py "Analyse our Q1 metrics" --manager personas/manager.yaml \
        --workers personas/researcher.yaml,personas/reviewer.yaml

    # Machine-readable JSON
    python run.py "Research quantum computing" --quiet
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv

# Ensure the repo root is on the path so `oas_cli` is importable.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load .env from the example directory.
load_dotenv(os.path.join(EXAMPLE_DIR, ".env"))

from loop import OrchestrationLoop  # noqa: E402


def default_path(filename: str) -> str:
    return os.path.join(EXAMPLE_DIR, "personas", filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent orchestration example")
    parser.add_argument(
        "objective", nargs="?", default="", help="Objective for the agent team"
    )
    parser.add_argument(
        "--manager", default=default_path("manager.yaml"), help="Manager spec path"
    )
    parser.add_argument(
        "--workers",
        default=",".join(
            [
                default_path("researcher.yaml"),
                default_path("writer.yaml"),
                default_path("reviewer.yaml"),
            ]
        ),
        help="Comma-separated worker spec paths",
    )
    parser.add_argument(
        "--concierge",
        default=default_path("concierge.yaml"),
        help="Concierge spec path (empty to disable)",
    )
    parser.add_argument("--dashboard", action="store_true", help="Launch web dashboard")
    parser.add_argument("--port", type=int, default=8420, help="Dashboard port")
    parser.add_argument("--quiet", "-q", action="store_true", help="JSON output only")

    args = parser.parse_args()

    worker_list = [w.strip() for w in args.workers.split(",") if w.strip()]
    concierge_path = args.concierge.strip() if args.concierge.strip() else None

    loop = OrchestrationLoop(
        manager_spec=args.manager,
        worker_specs=worker_list,
        concierge_spec=concierge_path,
    )

    if args.dashboard:
        from dashboard import serve

        print(f"Starting dashboard at http://localhost:{args.port}")
        print("Enter an objective in the web UI to begin.")
        serve(loop, port=args.port)
    else:
        if not args.objective:
            parser.error("objective is required when not using --dashboard")

        if not args.quiet:
            print(f"Objective: {args.objective}")
            print(f"Manager: {args.manager}")
            print(f"Workers: {', '.join(worker_list)}")
            if concierge_path:
                print(f"Concierge: {concierge_path}")
            print()

        result = loop.run(args.objective)

        if args.quiet:
            print(json.dumps(result, indent=2, default=str))
        else:
            board = result.get("board", {})
            by_status = board.get("by_status", {})
            print(f"\n{'=' * 50}")
            print(f"  Tasks: {board.get('total', 0)}")
            print(f"  Completed: {by_status.get('completed', 0)}")
            print(f"  Failed: {by_status.get('failed', 0)}")
            print(f"  Iterations: {result.get('iterations', 0)}")
            if result.get("summary"):
                print(f"\n  Summary: {result['summary'].get('summary', '')}")
            print(f"{'=' * 50}\n")
            print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
