# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

import json
import logging
import sys
import tempfile
from importlib.metadata import version as _get_version
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from .banner import ASCII_TITLE
from .core import generate_files as core_generate_files
from .core import validate_spec_file
from .exceptions import AgentGenerationError
from .runner import _choose_task, _load_spec, run_task_from_file

app = typer.Typer(help="Open Agent (OA) CLI")
console = Console()


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging with appropriate level and handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
    return logging.getLogger("oas")


def get_version_from_pyproject():
    """Get the version from package metadata (no setuptools dependency)."""
    try:
        return _get_version("open-agent-spec")
    except (ModuleNotFoundError, AttributeError):
        return "unknown"


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit"),
):
    """Main CLI entry point."""
    if version:
        cli_version = get_version_from_pyproject()
        console.print(
            f"[bold cyan]Open Agent Spec CLI[/] version [green]{cli_version}[/]"
        )
        raise typer.Exit()

    if ctx.invoked_subcommand is None or ctx.invoked_subcommand == "help":
        console.print(f"[bold cyan]{ASCII_TITLE}[/]\n")
        console.print(
            Panel.fit(
                "Use [bold magenta]oa init aac[/] for .agents/ agent-as-code layout\n"
                "Use [bold magenta]oa init --spec … --output …[/] to scaffold code\n"
                "Use [bold magenta]oa update[/] to update existing agent code\n"
                "Define it via Open Agent Spec YAML\n"
                "Use [bold yellow]--dry-run[/] to preview without writing files.",
                title="[bold green]OA CLI[/]",
                subtitle="Open Agent Spec Generator",
            )
        )


def load_and_validate_spec(
    spec_path: Path, log: logging.Logger
) -> tuple[dict[str, Any], str, str]:
    """Load and validate a spec file, returning the data and derived names."""
    log.info(f"Reading spec from: {spec_path}")
    try:
        spec_data, agent_name, class_name = validate_spec_file(spec_path)
        log.info("Spec file loaded successfully")
        return spec_data, agent_name, class_name
    except ValueError as err:
        log.error(str(err))
        typer.echo(str(err), err=True)
        raise


def resolve_spec_path(
    spec: Path | None, template: str | None, log: logging.Logger
) -> tuple[Path, Path | None]:
    """Return (spec_path, temp_file_to_delete).

    temp_file_to_delete is set only when using the minimal template.
    """
    if spec is not None:
        return spec, None
    elif template == "minimal":
        # Load the template from the package directory (no pkg_resources)
        try:
            template_path = Path(__file__).parent / "templates" / "minimal-agent.yaml"
            with open(template_path, "rb") as f:
                temp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
                temp.write(f.read())
                temp.close()
                return Path(temp.name), Path(temp.name)
        except (OSError, FileNotFoundError) as e:
            log.error(f"Failed to load minimal template from package: {e}")
            raise typer.Exit(1)
    else:
        log.error("You must provide either --spec or --template minimal.")
        raise typer.Exit(1)


def generate_files(
    output: Path,
    spec_data: dict[str, Any],
    agent_name: str,
    class_name: str,
    log: logging.Logger,
) -> None:
    """Generate all agent files (CLI wrapper with Rich console output)."""
    core_generate_files(output, spec_data, agent_name, class_name, log, console=console)


def _run_init_code_gen(
    spec: Path | None,
    output: Path,
    template: str | None,
    verbose: bool,
    dry_run: bool,
) -> None:
    """Legacy init: scaffold a full agent project into ``output``."""
    log = setup_logging(verbose)
    if verbose:
        log.setLevel(logging.DEBUG)

    console.print(
        Panel(
            ASCII_TITLE,
            title="[bold cyan]OA CLI[/]",
            subtitle="[green]Open Agent Spec Generator[/]",
        )
    )

    spec_path, temp_file_to_delete = resolve_spec_path(spec, template, log)
    try:
        spec_data, agent_name, class_name = load_and_validate_spec(spec_path, log)

        if dry_run:
            console.print(
                Panel.fit(
                    "🧪 [bold]Dry run mode[/]: No files will be written.",
                    style="yellow",
                )
            )
            log.info("Agent Name: %s", agent_name)
            log.info("Class Name: %s", class_name)
            log.info("Output directory would be: %s", output.resolve())
            log.info("Files that would be created:")
            log.info("- agent.py")
            log.info("- README.md")
            log.info("- requirements.txt")
            log.info("- .env.example")
            log.info("- prompts/agent_prompt.jinja2")
            return

        try:
            generate_files(output, spec_data, agent_name, class_name, log)
        except AgentGenerationError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1) from e
    finally:
        if temp_file_to_delete is not None and temp_file_to_delete.exists():
            try:
                temp_file_to_delete.unlink()
            except OSError:
                pass


# Init as a group so we can support `oa init aac` while keeping
# `oa init --spec ... --output ...` for code generation.
init_app = typer.Typer(
    help="Initialize agent projects (code gen or agent-as-code layout)"
)


@init_app.callback(invoke_without_command=True)
def init_callback(
    ctx: typer.Context,
    spec: Path | None = typer.Option(None, help="Path to Open Agent Spec YAML file"),
    output: Path | None = typer.Option(
        None, help="Directory to scaffold the agent into (required for code generation)"
    ),
    template: str | None = typer.Option(
        None, help="Template name to use (e.g., 'minimal')"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview what would be created without writing files"
    ),
):
    """Init code generation; use ``oa init aac`` for .agents/ layout only."""
    if ctx.invoked_subcommand is not None:
        return
    if output is None:
        console.print(
            "[yellow]Code generation requires --output.[/]\n"
            "  [bold]oa init --spec path/to/spec.yaml --output path/to/agent[/]\n"
            "  [bold]oa init --template minimal --output path/to/agent[/]\n"
            "For agent-as-code only (spec in .agents/), run: [bold]oa init aac[/]"
        )
        raise typer.Exit(1)
    _run_init_code_gen(spec, output, template, verbose, dry_run)


AAC_README = """# Agent as code (`.agents/`)

This folder holds **Open Agent Spec** YAML files. Treat them like infra-as-code.

## Validate all agents

```bash
oa validate aac
```

## Run the example agent

```bash
oa run --spec .agents/example.yaml --task greet --input '{"name": "CI"}' --quiet
```

## Review a diff (code reviewer agent)

After making a small change, have the reviewer agent decide and summarize:

```bash
git diff > change.diff
oa run .agents/review.yaml --task review --input change.diff --quiet
```

Example output:

```json
{"decision": "comment", "summary": "Debug print statement detected. Consider removing before production."}
```

## Generate full agent code

```bash
oa init --spec .agents/example.yaml --output ./generated-agent
```
"""


@init_app.command("aac")
def init_aac(
    directory: Path = typer.Option(
        Path("."),
        "--directory",
        "-C",
        help="Repo root to create .agents/ under (default: current directory)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing .agents/example.yaml",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Print only a short summary line",
    ),
):
    """Agent-as-code: create `.agents/` with example.yaml and review.yaml (no full code gen)."""
    agents_dir = directory.resolve() / ".agents"
    templates_dir = Path(__file__).parent / "templates"
    example_path = agents_dir / "example.yaml"
    review_path = agents_dir / "review.yaml"
    example_tpl = templates_dir / "aac-example.yaml"
    review_tpl = templates_dir / "aac-review.yaml"

    if not example_tpl.is_file():
        console.print(f"[red]Missing bundled template: {example_tpl}[/]")
        raise typer.Exit(1)
    if not review_tpl.is_file():
        console.print(f"[red]Missing bundled template: {review_tpl}[/]")
        raise typer.Exit(1)

    if example_path.exists() and not force:
        console.print(
            f"[yellow]{example_path} already exists.[/] Use --force to overwrite."
        )
        raise typer.Exit(1)

    agents_dir.mkdir(parents=True, exist_ok=True)
    example_path.write_text(example_tpl.read_text(encoding="utf-8"), encoding="utf-8")
    if not review_path.exists() or force:
        review_path.write_text(review_tpl.read_text(encoding="utf-8"), encoding="utf-8")

    readme_path = agents_dir / "README.md"
    if not readme_path.exists() or force:
        readme_path.write_text(AAC_README, encoding="utf-8")

    if quiet:
        typer.echo(str(example_path))
        typer.echo(str(review_path))
    else:
        console.print(
            Panel.fit(
                f"[green]Agent-as-code layout ready.[/]\n\n"
                f"  [bold]{example_path}[/]\n"
                f"  [bold]{review_path}[/]\n"
                f"  {readme_path}\n\n"
                "[dim]Validate:[/] [bold]oa validate aac[/]\n"
                "[dim]Run example:[/] [bold]oa run --spec .agents/example.yaml --quiet[/]\n"
                "[dim]Review a diff:[/] [bold]oa run --spec .agents/review.yaml --task review --input …[/]",
                title="[bold cyan]oa init aac[/]",
            )
        )


app.add_typer(init_app, name="init")

# Validate: single spec (oa validate --spec path) or all .agents/ (oa validate aac)
validate_app = typer.Typer(
    name="validate",
    help="Validate spec file(s) against the Open Agent Spec schema (no model calls).",
)


@validate_app.callback(invoke_without_command=True)
def validate_callback(
    ctx: typer.Context,
    spec: Path = typer.Option(None, "--spec", help="Path to a single Open Agent Spec YAML file"),
):
    """Validate one spec file. Use 'oa validate aac' to validate all specs in .agents/."""
    if ctx.invoked_subcommand is not None:
        return
    if spec is None:
        console.print("Specify [bold]--spec path[/] or use [bold]oa validate aac[/].")
        raise typer.Exit(1)
    log = setup_logging(verbose=False)
    try:
        validate_spec_file(spec)
    except ValueError as err:
        typer.echo(str(err), err=True)
        raise typer.Exit(1) from err
    log.info("Spec validation succeeded for %s", spec)
    typer.echo(f"Spec is valid: {spec}")


@validate_app.command("aac")
def validate_aac(
    directory: Path = typer.Option(
        Path("."),
        "--directory",
        "-C",
        help="Repo root containing .agents/ (default: current directory)",
    ),
):
    """Validate all agent spec files in .agents/ and print a summary per agent."""
    agents_dir = directory.resolve() / ".agents"
    if not agents_dir.is_dir():
        console.print(f"[red].agents/ not found[/] at [bold]{agents_dir}[/]")
        console.print("Run [bold]oa init aac[/] to create it.")
        raise typer.Exit(1)

    yaml_files = sorted(agents_dir.glob("*.yaml")) + sorted(agents_dir.glob("*.yml"))
    if not yaml_files:
        console.print(f"[yellow]No .yaml or .yml files[/] in [bold]{agents_dir}[/]")
        raise typer.Exit(1)

    failed = 0
    for spec_path in yaml_files:
        try:
            rel_path = spec_path.resolve().relative_to(directory.resolve())
        except ValueError:
            rel_path = spec_path.name
        try:
            spec_data, agent_name, class_name = validate_spec_file(spec_path)
        except (ValueError, KeyError) as err:
            failed += 1
            console.print(
                Panel(
                    f"[red]✘[/] [bold]{rel_path}[/]\n\n[red]{err!s}[/]",
                    title="[bold red]Validation failed[/]",
                    border_style="red",
                )
            )
            continue

        version = spec_data.get("open_agent_spec", "?")
        intelligence = spec_data.get("intelligence") or {}
        engine = intelligence.get("engine", "?")
        model = intelligence.get("model", "?")
        tasks = list((spec_data.get("tasks") or {}).keys())
        tasks_str = ", ".join(tasks) if tasks else "(none)"
        display_name = (spec_data.get("agent") or {}).get("name") or agent_name

        lines = [
            f"[green]✔[/] Spec version: [bold]{version}[/]",
            "[green]✔[/] Schema valid",
            "[green]✔[/] Required fields present",
            f"[green]✔[/] Intelligence engine configured: [bold]{engine}[/] ([bold]{model}[/])",
            f"[green]✔[/] Tasks validated: [bold]{tasks_str}[/]",
        ]
        console.print(
            Panel(
                "\n".join(lines) + "\n\n[dim]Spec is valid.[/]",
                title=f"[bold cyan]{rel_path}[/] — {display_name}",
                border_style="green",
            )
        )

    if failed:
        raise typer.Exit(1)


app.add_typer(validate_app, name="validate")


@app.command()
def version():
    """Show the Open Agent Spec CLI version."""
    cli_version = get_version_from_pyproject()
    console.print(f"[bold cyan]Open Agent Spec CLI[/] version [green]{cli_version}[/]")


@app.command()
def update(
    spec: Path = typer.Option(..., help="Path to updated Open Agent Spec YAML file"),
    output: Path = typer.Option(..., help="Directory containing the agent to update"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview what would be updated without writing files"
    ),
):
    """Update an existing agent project based on changes to the Open Agent Spec."""
    log = setup_logging(verbose)
    if verbose:
        log.setLevel(logging.DEBUG)

    console.print(
        Panel(
            ASCII_TITLE,
            title="[bold cyan]OA CLI[/]",
            subtitle="[green]Open Agent Spec Updater[/]",
        )
    )

    # Check if output directory exists
    if not output.exists():
        log.error(f"Output directory {output} does not exist. Run 'oa init' first.")
        raise typer.Exit(1)

    spec_data, agent_name, class_name = load_and_validate_spec(spec, log)

    if dry_run:
        console.print(
            Panel.fit(
                "🧪 [bold]Dry run mode[/]: No files will be updated.", style="yellow"
            )
        )
        log.info("Agent Name: %s", agent_name)
        log.info("Class Name: %s", class_name)
        log.info("Files that would be updated:")
        log.info("- agent.py")
        log.info("- README.md")
        log.info("- requirements.txt")
        log.info("- .env.example")
        log.info("- prompts/agent_prompt.jinja2")
        return

    # Generate updated files
    log.info("Generating updated files...")
    try:
        generate_files(output, spec_data, agent_name, class_name, log)
    except AgentGenerationError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e

    console.print("\n[bold green]✅ Agent project updated![/] ✨")
    log.info("Note: If you're using version control, make sure to commit your changes.")


@app.command()
def run(
    spec: Path = typer.Option(..., help="Path to Open Agent Spec YAML file"),
    task: str | None = typer.Option(
        None,
        "--task",
        "-t",
        help="Optional task name to run (defaults to first non-multi-step task)",
    ),
    input: str | None = typer.Option(
        None,
        "--input",
        "-i",
        help="Optional JSON object with input fields for the task",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="No banner; print only the task output JSON to stdout (clean for | jq)",
    ),
):
    """Run a single task directly from an Open Agent Spec file.

    This treats the spec as infra-as-code: no code generation step is required.
    """
    log = setup_logging(verbose)
    if verbose:
        log.setLevel(logging.DEBUG)
    if quiet:
        # Script/CI mode: no banner, plain JSON; keep log noise down unless -v
        if not verbose:
            logging.getLogger("oas").setLevel(logging.WARNING)

    if not quiet:
        console.print(
            Panel(
                ASCII_TITLE,
                title="[bold cyan]OA CLI[/]",
                subtitle="[green]Open Agent Spec Runner[/]",
            )
        )

    try:
        input_data: dict[str, Any] | None = None
        if input:
            input_stripped = input.strip()
            # Convenience: if --input is a path to an existing file and the task
            # has a single required string input, use file contents as that key.
            if not input_stripped.startswith("{") and not input_stripped.startswith("["):
                input_path = Path(input_stripped).resolve()
                if not input_path.is_file():
                    raise typer.BadParameter(f"File not found: {input_path}")
                spec_data = _load_spec(spec)
                _task_name, task_def = _choose_task(spec_data, task)
                inp_schema = task_def.get("input") or {}
                req = inp_schema.get("required") or []
                props = inp_schema.get("properties") or {}
                if len(req) != 1 or (props.get(req[0]) or {}).get("type") != "string":
                    raise typer.BadParameter(
                        "When --input is a file path, the task must have exactly "
                        "one required string input (e.g. diff). Use JSON otherwise."
                    )
                content = input_path.read_text(encoding="utf-8")
                if not content.strip():
                    raise typer.BadParameter(
                        f"File '{input_path.name}' is empty. "
                        "Generate a diff first (e.g. make a change, then run "
                        "'git diff > change.diff'), then run again."
                    )
                input_data = {req[0]: content}
            else:
                try:
                    parsed = json.loads(input)
                except json.JSONDecodeError as e:
                    raise typer.BadParameter(f"Invalid JSON for --input: {e}") from e
                if not isinstance(parsed, dict):
                    raise typer.BadParameter("--input JSON must be an object")
                input_data = parsed

        # When --quiet, redirect stdout to stderr during the run so logs from
        # our code or deps (dacp, httpx) don't pollute the pipe to jq.
        if quiet:
            real_stdout = sys.stdout
            sys.stdout = sys.stderr
        try:
            result = run_task_from_file(spec, task_name=task, input_data=input_data)
        finally:
            if quiet:
                sys.stdout = real_stdout
        if quiet:
            # Print only the agent's output (e.g. decision + summary) for clean piping
            out = result.get("output", result)
            typer.echo(json.dumps(out, indent=2))
        else:
            console.print_json(data=result)
    except Exception as err:
        if not quiet:
            log.error(str(err))
        typer.echo(str(err), err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
