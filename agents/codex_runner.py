#!/usr/bin/env python3
"""
Portable Codex CLI runner — no BCE dependency. Drop this file into any project.

Runs OpenAI Codex (https://github.com/openai/codex) in non-interactive mode:
  codex exec "<prompt>" with --skip-git-repo-check and optional sandbox/cwd.

Prerequisites:
  - Codex CLI installed:  npm install -g @openai/codex  (or brew install --cask codex)
  - Authenticated:  codex login

Usage:
  python codex_runner.py "your prompt"
  python codex_runner.py -C /path/to/repo "fix the bug in src/api.py"
  python codex_runner.py --sandbox danger-full-access "refactor auth"

From another agent or script:
  subprocess.run([sys.executable, "codex_runner.py", "add tests for X"], cwd="...")
  Or:  os.system('python codex_runner.py "add tests for X"')
"""

import argparse
import asyncio
import os
import sys

CODEX_EXEC = "codex"
DEFAULT_SANDBOX = "workspace-write"


async def _run_codex(
    prompt: str,
    cwd: str | None = None,
    sandbox: str = DEFAULT_SANDBOX,
    no_approval: bool = True,
) -> tuple[str, int]:
    cmd = [
        CODEX_EXEC,
        "exec",
        "--skip-git-repo-check",
        "--sandbox",
        sandbox,
        prompt,
    ]
    if no_approval:
        cmd.insert(-1, "--dangerously-bypass-approvals-and-sandbox")
    env = os.environ.copy()
    env["TERM"] = "dumb"
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd or os.getcwd(),
            env=env,
        )
        out_b, err_b = await proc.communicate()
        out = (out_b or b"").decode("utf-8", errors="replace").strip()
        err = (err_b or b"").decode("utf-8", errors="replace").strip()
        combined = f"{out}\n{err}".strip() if err else out
        if proc.returncode != 0:
            combined = f"[codex exit {proc.returncode}]\n{combined}"
        return (
            combined or "(no output)",
            proc.returncode if proc.returncode is not None else 0,
        )
    except FileNotFoundError:
        return (
            "Codex CLI not found. Install: npm install -g @openai/codex then codex login",
            127,
        )
    except Exception as e:
        return f"Codex exec error: {e}", 1


def main():
    parser = argparse.ArgumentParser(
        description="Run Codex CLI non-interactively (portable, no BCE)."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Instruction for Codex. Use - for stdin.",
    )
    parser.add_argument(
        "-C", "--cwd", default=None, help="Working directory for Codex."
    )
    parser.add_argument(
        "--sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        default=DEFAULT_SANDBOX,
        help="Sandbox policy (default: workspace-write).",
    )
    parser.add_argument(
        "--approval",
        action="store_true",
        help="Ask for approval before running commands.",
    )
    args = parser.parse_args()

    prompt = args.prompt
    if prompt is None or prompt == "-":
        prompt = sys.stdin.read().strip()
    if not prompt:
        print(
            "Usage: codex_runner.py PROMPT  or  echo 'prompt' | codex_runner.py -",
            file=sys.stderr,
        )
        sys.exit(1)

    cwd = os.path.abspath(args.cwd) if args.cwd else None
    if cwd and not os.path.isdir(cwd):
        print(f"Error: directory not found: {cwd}", file=sys.stderr)
        sys.exit(1)

    output, exit_code = asyncio.run(
        _run_codex(prompt, cwd=cwd, sandbox=args.sandbox, no_approval=not args.approval)
    )
    print(output)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
