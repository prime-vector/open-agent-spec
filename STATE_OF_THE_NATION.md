# Open Agent Spec (OA) — State of the Nation

**Project:** `open-agent-spec` (oas-new)  
**Location:** `Documents/src/oas-new/open-agent-spec`  
**Version (pyproject):** 1.0.9  
**Last reviewed:** Mar 3, 2025  

---

## 1. Executive summary

The Open Agent Spec CLI is a Python tool that scaffolds AI agent projects from YAML specs. It supports multiple LLM engines (OpenAI, Anthropic, Grok, Cortex, local, custom), behavioural contracts, multi-step tasks, and tool usage. The codebase is **functional**: build succeeds, 75 tests pass (2 skipped), and `oas init --template minimal` generates a working agent. Technical debt and quality opportunities remain: a large monolithic generator module, inconsistent docs (CONTRIBUTING examples use old `info` format), some broad exception handling, and no typed spec model. This document summarises current state, verification results, and a prioritised list of improvements.

---

## 2. Verification (Mar 3, 2025)

### Build and tests

- **Build:** `python -m build` succeeds; produces `open_agent_spec-1.0.9.tar.gz` and `.whl`.
- **Tests:** `pytest tests/` — **75 passed**, 2 skipped (integration scripts designed to run standalone). No failures.
- **Coverage:** ~60% overall; `generators.py` 63%, `validators.py` 39%, `main.py` 58%; `tools.py` and `utils.py` 0% (used by generated code / narrow paths).

### CLI and generation

- **Version:** `oas --version` and `oas version` print a version string (e.g. 1.0.9).
- **Minimal agent:** `oas init --template minimal --output /tmp/oas-review-agent` runs successfully and creates:
  - `agent.py` (template-based, DACP + behavioural_contracts)
  - `README.md`, `requirements.txt`, `.env.example`
  - `prompts/greet.jinja2`, `prompts/agent_prompt.jinja2`
- **Temp file cleanup:** The minimal template is written to a temp file; `main.py` deletes it in a `finally` block after use. No leak.

### Public API

- `from oas_cli import generate, validate_spec` works.
- `validate_spec(spec_path)` returns `(spec_data, agent_name, class_name)`; raises `ValueError` on invalid path/YAML/validation.
- `generate(spec_path, output_dir, dry_run=False)` runs the full pipeline; `dry_run=True` validates without writing files.

---

## 3. What’s in good shape

- **Clear purpose:** CLI (`oas init`, `oas update`) and generation pipeline are easy to follow.
- **Test coverage:** 77 tests total; 75 run and pass; markers (contract, cortex, multi_engine, generator, integration, slow) registered in `pytest.ini` and `pyproject.toml`.
- **Structured generation:** `AgentDataPreparator` + `CodeGenerator` + Jinja2 (`agent.py.j2`, `task_function.py.j2`) provide template-based generation; legacy f-string path exists as fallback.
- **Reusable pieces:** `PythonCodeSerializer`, `TemplateVariableParser`, schema validation, and DACP/BCE integration are separable.
- **CI:** Single consolidated workflow (`.github/workflows/ci.yml`) for test, Ruff, mypy, integration; `.github/workflows/README.md` documents workflows.
- **Linting/config:** Ruff and mypy in `pyproject.toml`; pre-commit available.
- **Version and validation:** Version tests assert a version-like string (not a literal from pyproject); `importlib.metadata` used for version; `ValidationError` in validators uses `str(e)` (`{e!s}`).
- **Docs:** README documents `open_agent_spec` as the canonical spec version field and role enum (analyst, reviewer, chat, retriever, planner, executor) aligned with schema.

---

## 4. Bugs and potential issues

### Resolved (since previous review)

| Issue | Resolution |
|-------|------------|
| Failing version tests | Tests now assert version-like string (non-empty, contains digits), not literal from pyproject. |
| Intentional failing test | `tests/bad_test.py::test_true_equals_false` is `@pytest.mark.skip(reason="...")`. |
| Temp file leak | `main.py` returns `temp_file_to_delete` from `resolve_spec_path` and unlinks it in `finally` after init. |
| Mutable default in `file_writer` | `tools.py` uses `allowed_paths: list[str] \| None = None` and `if allowed_paths is None: allowed_paths = []`. |
| ValidationError.message | `validators.py` uses `raise ValueError(f"Spec validation failed: {e!s}")` (no `.message`). |
| pkg_resources | Replaced with `importlib.metadata.version("open-agent-spec")` in `main.py`. |

### Remaining / low risk

| Issue | Location | Description |
|-------|----------|-------------|
| CONTRIBUTING examples | `CONTRIBUTING.md` | “Basic Structure” and “Required Fields” still show old `info` section; schema and README use `agent` + `open_agent_spec`. New contributors may copy-paste invalid examples. |
| typer extra | `pyproject.toml` | `typer[all]` triggers install warning (“does not provide the extra 'all'”). Can switch to `typer` only. |
| Pytest config | `pytest.ini` vs `pyproject.toml` | Pytest reports “ignoring pytest config in pyproject.toml” when both exist. Either consolidate in `pytest.ini` or migrate fully to pyproject and remove `pytest.ini`. |
| Schema role enum | README vs schema | README and schema are aligned (analyst, reviewer, chat, retriever, planner, executor). No change needed unless you add more roles. |

---

## 5. Code quality and opportunities

### 5.1 Size and complexity

- **`generators.py`** (~1,660 lines): Single large module mixing orchestration (`generate_agent_code`, `generate_readme`, …) with low-level snippet building (`_generate_pydantic_model`, `_generate_llm_output_parser`, …). Previous refactors added shared helpers (`_format_contract_for_decorator`, `_get_task_function_preamble`, `_build_*`), but the file remains hard to navigate. **Opportunity:** Split into an “orchestrator” (e.g. `generators.py` or `generate/runner.py`) and “snippet” modules (e.g. `generate/snippets.py`, `generate/models.py`).
- **`validators.py`**: Manual validation plus JSON Schema; coverage 39%. **Opportunity:** Add targeted tests for validation branches and schema validation errors.

### 5.2 Exception handling

- **Broad `except Exception`** remains in:
  - `oas_cli/generators.py` (template fallback)
  - `oas_cli/code_generation.py` (template load/render)
  - `oas_cli/data_preparation.py`, `oas_cli/tools.py`, `oas_cli/validators.py`, `oas_cli/utils.py`
- Some are intentional (e.g. file write failure, last-resort fallback); others could be narrowed to specific exceptions for clearer diagnostics. **Opportunity:** Audit each and replace with specific exceptions where feasible.

### 5.3 Abstraction and types

- **No typed spec model:** Spec is `Dict[str, Any]` everywhere. A small dataclass or Pydantic model for top-level spec (and optionally tasks/intelligence) would centralise validation, defaults, and version handling. **Opportunity:** Introduce `Spec` (and related) models used by validators and generators.
- **Tight coupling to DACP/BCE:** Generation is tied to DACP and behavioural_contracts. For reuse with other runtimes, a thin adapter layer would allow swapping without forking generators.

### 5.4 Package and config

- **Wheel contents:** `pyproject.toml` explicitly includes `oas_cli/templates/*.yaml` and `oas_cli/schemas/*.json`. Jinja2 templates (e.g. `agent.py.j2`, `task_function.py.j2`) live under `oas_cli/templates/` and are loaded via `Path(__file__).parent / "templates"`; they are included as part of the package. **Recommendation:** Confirm built wheel contains `oas_cli/templates/*.j2` (e.g. `unzip -l dist/*.whl`) after any build changes.

---

## 6. Structure and composability

- **Entrypoint:** `main.py` handles CLI, validation, and orchestration; core logic lives in `core.py` and `generators.py`. Public API in `oas_cli/__init__.py` exposes `generate` and `validate_spec` for script use.
- **Layering:** Generation is split across `generators.py`, `data_preparation.py`, and `code_generation.py`; boundaries are clear but `generators.py` still does a lot. Further splitting would improve composability and testability.

---

## 7. Suggested priority order

**Quick wins (no behaviour change)**

1. **CONTRIBUTING.md:** Update “Basic Structure” and “Required Fields” to use `open_agent_spec` and `agent` (name, description, role) so examples match the schema and README.
2. **typer dependency:** Change `typer[all]` to `typer` in `pyproject.toml` to remove the install warning.
3. **Pytest config:** Either remove duplicate markers from `pyproject.toml` and keep a single source in `pytest.ini`, or move all options to `pyproject.toml` and remove `pytest.ini` to clear the warning.

**Next (quality and maintainability)**

4. **Exception handling:** Replace broad `except Exception` with specific exceptions in `validators.py`, `utils.py`, and any generator fallbacks where it’s safe.
5. **Validators tests:** Add tests for validation branches and schema validation failure paths to improve coverage and regression safety.
6. **Split generators:** Extract snippet-building and model-building into separate modules; keep `generators.py` as a thin orchestrator.

**When you have time**

7. **Spec model:** Introduce a Spec (and related) dataclass or Pydantic model; use it in validators and pass it into generators.
8. **Adapter layer:** Consider a thin adapter for the runtime (DACP/BCE) so alternative runtimes can be plugged in without forking.

---

## 8. File and area reference

| Area | Path | Notes |
|------|------|--------|
| CLI entry | `oas_cli/main.py` | Typer app, version, init, update, temp file cleanup in `finally` |
| Core API | `oas_cli/core.py` | `validate_spec_file`, `generate_files`, `generate` |
| Public API | `oas_cli/__init__.py` | `generate`, `validate_spec` |
| Generation | `oas_cli/generators.py` | All file gen; template + legacy; large file |
| Validation | `oas_cli/validators.py` | Spec + JSON schema; version field `open_agent_spec`; `str(e)` for ValidationError |
| Data prep | `oas_cli/data_preparation.py` | AgentDataPreparator for template data |
| Code gen | `oas_cli/code_generation.py` | PythonCodeSerializer, CodeGenerator, TemplateVariableParser |
| Tools | `oas_cli/tools.py` | file_writer, TOOL_REGISTRY; safe default for `allowed_paths` |
| Utils | `oas_cli/utils.py` | parse_response |
| Schema | `oas_cli/schemas/oas-schema.json` | open_agent_spec, agent, intelligence, tasks, roles enum |
| Templates | `oas_cli/templates/*.yaml`, `*.j2` | minimal-agent, security-*, agent.py.j2, task_function.py.j2 |
| Tests | `tests/*.py`, `tests/integration/` | test_main, test_generators, test_public_api, test_contract_validation, etc. |
| Config | `pyproject.toml`, `pytest.ini`, `.pre-commit-config.yaml` | Version 1.0.9; Ruff in pyproject; pytest markers in both |

---

Once you decide which items to tackle first, we can go through them step by step (e.g. CONTRIBUTING + typer + pytest, then exception and validator tests, then generator split).
