# Changelog

All notable changes to **open-agent-spec** (Open Agent CLI) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- This changelog.

### Changed
- Standardised spec version to `1.0.9` across all examples, templates, and docs (CONTRIBUTING.md, README.md, `.agents/`, Website content).
- README quick-start YAML now includes the required `open_agent_spec` version field.

### Fixed
- Removed broken references to non-existent `security-threat-analyzer.yaml` template and `SECURITY_TEMPLATES.md` from REFERENCE.md.

---

## [1.2.3] - 2026-03-11

### Breaking
- **CLI command renamed `oas` → `oa`.** The console script installed by `pip install open-agent-spec` is now `oa`. Update scripts and docs accordingly. Equivalent: `python -m oas_cli …` (unchanged).

### Changed
- README and docs updated for Open Agent (OA) positioning and `oa` commands.
- `pyproject.toml` build backend pinned to `hatchling>=1.26.1` so wheels/sdists no longer emit `License-File` with Metadata-Version 2.3 (avoids PyPI/twine upload errors).
- PyPI project description and `[project.urls]` aligned with current README.

### Added
- `AgentGenerationError` for template/generation failures with actionable messages; CLI prints to stderr and exits cleanly on init/update.
- `oa run --quiet` / `-q` for JSON-only stdout (scripting/CI).
- Runtime dependency on `dacp` so `oa run` and `oas_cli.runtime` work without extra installs.

### Fixed
- Task README/docs generation now walks `input`/`output` JSON Schema `properties` instead of iterating schema top-level keys.
- Multi-step `input_map` handling: safer step indexing; non-string literals emitted without quoting in generated code.
- `task_codegen` façade re-exports aligned with split generation modules.

---

## Earlier releases

Versions **1.0.x–1.2.x** on PyPI predate this changelog. For historical behaviour, see git history and [releases](https://github.com/prime-vector/open-agent-spec/releases).

When backfilling, use sections **Added** / **Changed** / **Deprecated** / **Removed** / **Fixed** / **Security** as appropriate.
