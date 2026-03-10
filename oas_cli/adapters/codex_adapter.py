"""Codex engine adapter for the Open Agent Spec runtime.

This module provides a best-effort integration with the Codex CLI, treating it
as just another intelligence engine. It is intentionally thin and can be
evolved independently of the spec format.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from pathlib import Path
from typing import Any


def _build_codex_command(prompt: str, config: dict[str, Any]) -> tuple[list[str], str]:
    """Build the Codex CLI command and working directory from config."""
    sandbox = str(config.get("sandbox", "workspace-write"))
    cwd_raw = config.get("cwd") or "."
    cwd = os.fspath(Path(cwd_raw))

    cmd: list[str] = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--sandbox",
        sandbox,
        prompt,
    ]

    # Optionally allow callers to pass through extra CLI flags
    extra_args = config.get("extra_args")
    if isinstance(extra_args, str) and extra_args.strip():
        cmd[1:1] = shlex.split(extra_args)

    return cmd, cwd


async def _run_codex_async(prompt: str, config: dict[str, Any]) -> tuple[str, int]:
    cmd, cwd = _build_codex_command(prompt, config)
    env = os.environ.copy()
    env.setdefault("TERM", "dumb")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
    except FileNotFoundError:
        return (
            "Codex CLI not found. Install @openai/codex and run `codex login`.",
            127,
        )

    out_b, err_b = await proc.communicate()
    out = (out_b or b"").decode("utf-8", errors="replace").strip()
    err = (err_b or b"").decode("utf-8", errors="replace").strip()
    combined = f"{out}\n{err}".strip() if err else out

    return combined or "(no output)", proc.returncode or 0


def invoke(prompt: str, config: dict[str, Any]) -> Any:
    """Invoke Codex for the given prompt and config.

    The adapter attempts to parse Codex output as JSON first; if that fails,
    it returns a simple dict with `success` and `text` keys so that callers
    can still map it onto their task output schema.
    """
    text, exit_code = asyncio.run(_run_codex_async(prompt, config))

    # Try JSON first (for structured results)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {"success": exit_code == 0, "text": text}

    if isinstance(data, dict) and "success" not in data:
        data["success"] = exit_code == 0

    return data
