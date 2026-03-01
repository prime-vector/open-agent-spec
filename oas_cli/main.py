# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

import logging
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

app = typer.Typer(help="Open Agent Spec (OA) CLI")
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
                "Use [bold magenta]oas init[/] to scaffold an agent project\n"
                "Use [bold magenta]oas update[/] to update existing agent code\n"
                "Define it via Open Agent Spec YAML\n"
                "Use [bold yellow]--dry-run[/] to preview actions without writing files.",
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
    """Return (spec_path, temp_file_to_delete). Second is non-None only when using minimal template."""
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


@app.command()
def version():
    """Show the Open Agent Spec CLI version."""
    cli_version = get_version_from_pyproject()
    console.print(f"[bold cyan]Open Agent Spec CLI[/] version [green]{cli_version}[/]")


@app.command()
def init(
    spec: Path | None = typer.Option(None, help="Path to Open Agent Spec YAML file"),
    output: Path = typer.Option(..., help="Directory to scaffold the agent into"),
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
    """Initialize an agent project based on Open Agent Spec."""
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

    # Determine which spec to use
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

        generate_files(output, spec_data, agent_name, class_name, log)
    finally:
        if temp_file_to_delete is not None and temp_file_to_delete.exists():
            try:
                temp_file_to_delete.unlink()
            except OSError:
                pass


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
        log.error(
            f"Output directory {output} does not exist. Use 'oas init' to create a new agent."
        )
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
    generate_files(output, spec_data, agent_name, class_name, log)

    console.print("\n[bold green]✅ Agent project updated![/] ✨")
    log.info("Note: If you're using version control, make sure to commit your changes.")


if __name__ == "__main__":
    app()
