"""Terminal UI helpers for the OA CLI.

Provides a consistent visual identity:

  ┌──────────────────────────────────────────┐
  │  Banner     — compact bot-face on startup │
  │  Run header — agent/task/model before run │
  │  Spinner    — animated "thinking" state   │
  │  Result     — styled JSON output panel    │
  └──────────────────────────────────────────┘

All functions accept a ``console: Console`` argument so callers can swap
in a test-friendly console without affecting global state.

Nothing here should import from runner.py to avoid circular deps.
"""

from __future__ import annotations

import json
import re
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .banner import BANNER

# ── Colour palette ─────────────────────────────────────────────────────────

_C_BRAND = "bold cyan"
_C_DIM = "dim"
_C_OK = "bold green"
_C_WARN = "bold yellow"
_C_ERR = "bold red"
_C_KEY = "cyan"
_C_VAL = "white"
_C_SUBTLE = "grey50"

# ── Banner ──────────────────────────────────────────────────────────────────


def print_banner(console: Console, version: str = "") -> None:
    """Print the OA bot-face banner with an optional version string."""
    ver_str = f"  [dim]v{version}[/]" if version else ""
    lines = BANNER.rstrip("\n").split("\n")
    # Inject version on the last non-empty line
    if ver_str and lines:
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip():
                lines[i] = lines[i] + ver_str
                break
    body = "\n".join(lines)
    console.print(
        Panel(
            f"[{_C_BRAND}]{body}[/]",
            border_style="cyan",
            padding=(0, 2),
        )
    )


# ── Run header ──────────────────────────────────────────────────────────────


def print_run_header(
    console: Console,
    spec_data: dict[str, Any],
    task_name: str,
) -> None:
    """Print a compact metadata header before invoking the model.

    Example output:
        ◈  chat-agent  ·  assistant                       gpt-4o-mini  openai
           task: reply
    """
    agent = spec_data.get("agent") or {}
    intel = spec_data.get("intelligence") or {}

    agent_name: str = agent.get("name") or "agent"
    agent_role: str = agent.get("role") or ""
    model: str = intel.get("model") or "—"
    engine: str = intel.get("engine") or "openai"

    # Left: agent identity
    left = Text()
    left.append("◈  ", style=_C_BRAND)
    left.append(agent_name, style="bold white")
    if agent_role:
        left.append(f"  ·  {agent_role}", style=_C_DIM)

    # Right: intelligence config
    right = Text()
    right.append(model, style=_C_KEY)
    right.append(f"  {engine}", style=_C_SUBTLE)

    table = Table.grid(expand=True)
    table.add_column(justify="left")
    table.add_column(justify="right")
    table.add_row(left, right)

    console.print(table)
    console.print(f"   [dim]task:[/] [{_C_KEY}]{task_name}[/]")
    console.print(Rule(style="grey23"))


# ── Spinner ─────────────────────────────────────────────────────────────────


@contextmanager
def inference_spinner(
    console: Console,
    model: str,
    task_name: str,
) -> Generator[None, None, None]:
    """Context manager that shows a live spinner while the model is running.

    Usage::

        with inference_spinner(console, "gpt-4o-mini", "reply"):
            result = run_task_from_file(...)
    """
    label = (
        f"[{_C_SUBTLE}]calling[/] [{_C_KEY}]{model}[/] [dim]·[/] [dim]{task_name}[/]"
    )
    with console.status(label, spinner="dots", spinner_style=_C_BRAND):
        yield


# ── Result panel ─────────────────────────────────────────────────────────────

_FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", re.IGNORECASE
)


def _render_output(output: Any) -> Any:
    """Return a Rich renderable for the task output.

    • dict / list      → monokai JSON syntax block
    • str with fenced JSON  → strip prose, render the JSON block highlighted
    • plain str        → Markdown (handles headers, bullets, bold naturally)
    """
    if isinstance(output, (dict, list)):
        return Syntax(
            json.dumps(output, indent=2, ensure_ascii=False),
            "json",
            theme="monokai",
            word_wrap=True,
        )

    if isinstance(output, str):
        # Try to pull a JSON block out of a markdown-fenced response.
        match = _FENCED_JSON_RE.search(output)
        if match:
            try:
                extracted = json.loads(match.group(1))
                prose = output[: match.start()].strip()
                if prose:
                    # Show any introductory prose above the JSON block.
                    return _VStack([Markdown(prose), _json_syntax(extracted)])
                return _json_syntax(extracted)
            except json.JSONDecodeError:
                pass
        # Plain text — render via Markdown so headings/bullets/bold work.
        return Markdown(output)

    # Fallback: anything else (int, None, …)
    return Syntax(
        json.dumps(output, indent=2, ensure_ascii=False),
        "json",
        theme="monokai",
        word_wrap=True,
    )


def _json_syntax(obj: Any) -> Syntax:
    return Syntax(
        json.dumps(obj, indent=2, ensure_ascii=False),
        "json",
        theme="monokai",
        word_wrap=True,
    )


class _VStack:
    """Minimal vertical stack of Rich renderables (no extra deps)."""

    def __init__(self, items: list) -> None:
        self._items = items

    def __rich_console__(self, console: Console, options: Any):
        yield from self._items


def print_result_panel(
    console: Console,
    result: dict[str, Any],
    elapsed_s: float | None = None,
) -> None:
    """Print the task result in a styled panel.

    Shows:
    • A ✓ header with task name and optional elapsed time
    • Output rendered appropriately: JSON dicts highlighted, plain strings
      as Markdown, fenced-JSON-in-prose extracted and highlighted
    • Chain intermediate results collapsed in a dimmed sub-panel (when present)
    """
    task_name: str = result.get("task") or "output"
    model: str = result.get("model") or ""
    output: Any = result.get("output", result)
    chain: dict | None = result.get("chain")

    content = _render_output(output)

    elapsed_str = f"  [dim]{elapsed_s:.1f}s[/]" if elapsed_s is not None else ""
    model_str = f"  [{_C_SUBTLE}]{model}[/]" if model else ""

    console.print(
        Panel(
            content,
            title=f"[{_C_OK}]✓[/] [{_C_KEY}]{task_name}[/]",
            title_align="left",
            subtitle=f"{elapsed_str}{model_str}",
            subtitle_align="right",
            border_style="green",
            padding=(1, 2),
        )
    )

    # Show chain intermediates in a muted panel
    if chain:
        rows: list[str] = []
        for dep_task, dep_result in chain.items():
            dep_out = (
                dep_result.get("output") if isinstance(dep_result, dict) else dep_result
            )
            dep_json = json.dumps(dep_out, indent=2, ensure_ascii=False)
            rows.append(f"[dim]── {dep_task}[/]\n[grey39]{dep_json}[/]")
        console.print(
            Panel(
                "\n\n".join(rows),
                title="[dim]chain[/]",
                title_align="left",
                border_style="grey35",
                padding=(0, 2),
            )
        )


# ── Error panel ──────────────────────────────────────────────────────────────


def print_error_panel(console: Console, title: str, message: str) -> None:
    """Print a styled error panel."""
    console.print(
        Panel(
            f"[{_C_ERR}]{message}[/]",
            title=f"[{_C_ERR}]✗ {title}[/]",
            title_align="left",
            border_style="red",
            padding=(0, 2),
        )
    )


# ── Help panel ────────────────────────────────────────────────────────────────


def print_help_panel(console: Console, version: str = "") -> None:
    """Print the startup help panel shown when oa is run with no subcommand."""
    print_banner(console, version)
    console.print(
        Panel.fit(
            "  [bold magenta]oa run[/]    [white]--spec path.yaml --task name --input '\\{…\\}'[/]\n"
            "  [bold magenta]oa init[/]   [white]aac | --spec path.yaml --output dir/[/]\n"
            "  [bold magenta]oa test[/]   [white]agent.test.yaml[/]\n"
            "  [bold magenta]oa update[/] [white]--spec path.yaml --output dir/[/]\n\n"
            "  [dim]--quiet / -q[/]  [dim]JSON-only output for pipes and CI[/]",
            title=f"[{_C_BRAND}]oa[/]  [dim]Open Agent Spec CLI[/]",
            border_style="cyan",
            padding=(1, 3),
        )
    )
