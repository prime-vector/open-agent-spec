# Open Agent Spec (OAS) — State of the Nation

**Project:** `open-agent-spec` (oas-new)  
**Location:** `Documents/src/oas-new/open-agent-spec`  
**Version (pyproject):** 1.0.9  
**Last reviewed:** Feb 28, 2025  

---

## 1. Executive summary

The Open Agent Spec CLI is a Python tool that scaffolds AI agent projects from YAML specs. It supports multiple LLM engines (OpenAI, Anthropic, Grok, Cortex, local, custom), behavioural contracts, multi-step tasks, and tool usage. The codebase is functional and test-heavy, but has accumulated **technical debt**: duplicated logic, inconsistent naming, a few bugs, weak composability, and missing open-source hygiene (pytest marks, version handling, docs/schema alignment). This document summarises current state, issues, and a cleanup strategy so you can prioritise work.

---

## 2. What’s in good shape

- **Clear purpose:** CLI (`oas init`, `oas update`) and generation pipeline are easy to follow.
- **Test coverage:** 66+ tests (generators, main, enhanced spec, cortex, multi-engine, contract validation, integration). Most pass when run with the repo’s `pyproject` version.
- **Structured generation:** `AgentDataPreparator` + `CodeGenerator` + Jinja2 give a template-based path; legacy f-string generation exists as fallback.
- **Reusable pieces:** `PythonCodeSerializer`, `TemplateVariableParser`, schema validation, and DACP/BCE integration are separable.
- **CI:** Multiple workflows (feature-test, pr-test, integration-tests, publish, test-enhanced, ci-error-analysis).
- **Linting/config:** Ruff, mypy, pre-commit, and build metadata in `pyproject.toml`.

---

## 3. Bugs and potential bugs

### 3.1 Confirmed / high confidence

| Issue | Location | Description |
|-------|----------|-------------|
| **Failing version tests** | `tests/test_main.py` | Tests assert version from `pyproject.toml` (e.g. 1.0.9) but CLI reports installed package version (e.g. 1.2.0). Fails when running tests in an env where a different version is installed. |
| **Intentional failing test** | `tests/bad_test.py` | `test_true_equals_false` asserts `True == False`. Fails by design; should be removed or gated (e.g. `@pytest.mark.skip`) if kept for demo. |
| **Temp file leak** | `main.py` `resolve_spec_path()` | For `--template minimal`, a `NamedTemporaryFile(delete=False)` is created and never deleted. Leaves temp YAML files on disk. |
| **Mutable default argument** | `tools.py` `file_writer()` | `allowed_paths: List[str] = None` is fine, but if it were `= []` it would be a classic mutable-default bug. Worth making default explicit (e.g. `None` and `if allowed_paths is None: allowed_paths = []`) for clarity and safety. |

### 3.2 Possible / version-sensitive

| Issue | Location | Description |
|-------|----------|-------------|
| **ValidationError.message** | `validators.py` | Uses `e.message` on jsonschema’s `ValidationError`. Supported in current jsonschema 4.x; if you ever rely on a different error API (e.g. `str(e)`), this could break. Prefer `str(e)` for compatibility. |
| **Schema vs README version field** | README vs schema/code | README documents `spec_version`; schema and code use `open_agent_spec`. One test uses `spec_version` (`test_custom_llm_router.py`). Either align README to `open_agent_spec` or support both and document. |
| **Schema role enum vs README** | `oas-schema.json` vs README | Schema role enum: `analyst`, `reviewer`, `chat`, `retriever`, `planner`, `executor`. README lists: `assistant`, `analyst`, `specialist`, `coordinator`, `researcher`, `consultant`. Mismatch will confuse users and can cause validation failures. |

---

## 4. Code quality and patterns

### 4.1 Duplication

- **`format_value`-style logic:** **Resolved.** Now uses `PythonCodeSerializer.format_value` from `code_generation` everywhere; the three local copies were removed.
- **Class method / example task generation:** Similar blocks in `generators._generate_agent_code_legacy` and `data_preparation._prepare_class_methods` / `_prepare_example_task_code`. Could be a shared “class method snippet” generator used by both legacy and template path.
- **Long prompt strings:** Large default prompt and Jinja2 snippets duplicated (e.g. in `generate_prompt_template` and inside `generate_agent_code` default template). Better as single constants or small template files.

### 4.2 Sloppy or fragile patterns

- **Bare `except Exception`:** **Partly resolved.** Narrowed to specific exceptions in `main.py` and `generators.py` (template fallback). Remaining broad catches are in generated code or intentional fallbacks.
- **Deprecated API:** **Resolved.** Replaced `pkg_resources` with `importlib.metadata.version` in `main.py`.
- **Magic task name:** **Resolved.** Replaced `save_greeting` special-case with generic mapping from step results to output schema (first value that is not None per property).
- **Large functions:** `_generate_task_function`, `_generate_tool_task_function`, `_generate_multi_step_task_function`, and `_generate_agent_code_legacy` are very long (hundreds of lines) and mix concerns (contract formatting, code gen, tool wiring). Hard to test and refactor.

### 4.3 Missing or weak abstraction

- **No explicit “spec model”:** Spec is passed as `Dict[str, Any]` everywhere. A small dataclass/Pydantic model for the top-level spec (and maybe tasks/intelligence) would give one place for validation, defaults, and version handling.
- **Tight coupling to DACP/BCE:** Generation is tied to DACP and behavioural_contracts. For “enterprise / open-source” reuse, consider a thin adapter layer so that another runtime could be plugged in without forking generators.
- **Template discovery:** `CodeGenerator` uses a single directory; `ensure_template_exists` writes a default template to disk. No clear story for “bundled vs user overrides” or multiple template roots.

---

## 5. Structure and composability

- **Entrypoint:** `main.py` does CLI, validation, and orchestration. Fine for a CLI, but “load spec → validate → prepare data → generate” could be a small library API (e.g. `oas_cli.generate(spec_path, output_dir)`) for reuse from other tools or scripts.
- **Layering:** `generators.py` mixes high-level orchestration (`generate_agent_code`, `generate_readme`, …) with low-level string building (`_generate_pydantic_model`, `_generate_llm_output_parser`, …). Splitting into “orchestrator” vs “snippet generators” would improve composability.
- **Package layout:** Single package `oas_cli` with many modules. No clear split like `oas_cli.spec`, `oas_cli.generate`, `oas_cli.cli`; adding that would make boundaries clearer.

---

## 6. Open-source and project hygiene

- **Pytest marks:** Tests use `@pytest.mark.contract`, `@pytest.mark.cortex`, `@pytest.mark.multi_engine` but these marks are not registered in `pyproject.toml` or `pytest.ini`, so you get “Unknown pytest.mark” warnings. Register them under `[tool.pytest.ini_options]` (or equivalent).
- **Version in tests:** Version tests assume “version comes from repo’s pyproject.toml”. In CI you often install the package, so reported version is the built package. Either (a) test that version string is non-empty and matches a pattern, or (b) install in editable mode and keep asserting against pyproject, or (c) inject version in tests via env/mock.
- **Ruff config:** Ruff is configured in both `pyproject.toml` and `ruff.toml` (e.g. different `line-ending`: `lf` vs `auto`). One source of truth (prefer `pyproject.toml`) avoids confusion.
- **Duplicate/overlapping workflows:** Many workflow files (e.g. `feature-test.yml`, `test.yml`, `test-enhanced.yml`, `pr-test.yml`, `integration-tests.yml`). Risk of drift and duplicated logic; consider consolidating or clearly documenting when each runs.
- **CONTRIBUTING / README:** README is detailed; CONTRIBUTING exists. Ensure “how to run tests”, “how to add a template”, and “version field: use `open_agent_spec`” are consistent.

---

## 7. Strategy for cleanup and hardening

### Phase 1 — Quick wins (no behaviour change)

1. **Remove or fix `bad_test.py`**  
   Delete `test_true_equals_false` or mark it `@pytest.mark.skip(reason="...")` if you want to keep it as an example.

2. **Fix temp file leak**  
   In `resolve_spec_path`, either use a context manager that deletes the temp file when the CLI exits, or document that minimal template is written to a temp file and delete it after `load_and_validate_spec` (or after generation).

3. **Register pytest marks**  
   In `pyproject.toml` add something like:
   ```ini
   [tool.pytest.ini_options]
   markers = [
     "contract: behavioural contract tests",
     "cortex: cortex integration tests",
     "multi_engine: multi-LLM engine tests",
   ]
   ```

4. **Single Ruff config**  
   Move all Ruff settings into `pyproject.toml` and remove or deprecate `ruff.toml` so formatting is consistent.

5. **Align README with schema**  
   Replace `spec_version` with `open_agent_spec` in README (and any examples), or add a short note that the canonical field is `open_agent_spec` and `spec_version` is an alias if you add support. Align role list with schema enum or document both.

### Phase 2 — Version and validation

6. **Robust version tests**  
   Either assert that the CLI prints a version that matches `pyproject.toml` when run from the repo (e.g. editable install), or assert “version is a semver-like string” and stop comparing to a literal from pyproject when the installed package may differ.

7. **Replace pkg_resources**  
   Use `importlib.metadata.version("open-agent-spec")` (with a try/except for older Python if needed) and drop the deprecation warning.

8. **ValidationError message**  
   In `validate_with_json_schema`, use `str(e)` instead of `e.message` for compatibility and consistency.

### Phase 3 — Reduce duplication and complexity

9. **Single `format_value`**  
   In `generators.py`, remove the three local `format_value` implementations and call `PythonCodeSerializer.format_value` (or a module-level helper that delegates to it). Refactor contract-formatting to use that single implementation.

10. **Remove save_greeting special case**  
    Replace the hard-coded `task_name == "save_greeting"` branch with a generic mapping from step results to output schema (e.g. from `steps` and `output.properties`).

11. **Split large generator functions**  
    Break `_generate_task_function`, `_generate_tool_task_function`, and `_generate_multi_step_task_function` into smaller functions (e.g. “build contract dict”, “build client code”, “build prompt render”) and reuse shared helpers.

### Phase 4 — Structure and API

12. **Spec model**  
    Introduce a small Spec model (dataclass or Pydantic) built from the YAML dict, used by validators and generators. Validators return or accept this model; generators take it instead of raw dicts.

13. **Thin public API**  
    Expose something like `oas_cli.generate(spec_path, output_dir, dry_run=False)` (and maybe `validate_spec(spec_path)`) so scripts and other tools can drive generation without going through the CLI.

14. **Template and config story**  
    Document (and if needed implement) how bundled templates are overridden (e.g. by a user template dir or env var), and ensure one place defines default prompt text.

### Phase 5 — CI and docs

15. **Consolidate or document workflows**  
    Either merge overlapping workflows into a smaller set (e.g. “test”, “publish”, “integration”) or add a short README in `.github/workflows` describing when each runs and what it does.

16. **CONTRIBUTING**  
    Add a line about running tests (e.g. `pytest tests/`), registering new pytest marks, and that the canonical spec version field is `open_agent_spec`.

---

## 8. Suggested priority order

- **Do first:** 1 (bad_test), 2 (temp file), 3 (pytest marks), 5 (README/schema alignment), 6 (version tests), 7 (pkg_resources).
- **Next:** 4 (Ruff), 8 (ValidationError), 9 (format_value), 10 (save_greeting).
- **Then:** 11 (split large functions), 12 (spec model), 13 (public API).
- **When you have time:** 14 (templates), 15 (workflows), 16 (CONTRIBUTING).

---

## 9. File and area reference

| Area | Path | Notes |
|------|------|--------|
| CLI entry | `oas_cli/main.py` | Typer app, version, init, update, temp file for minimal template |
| Generation | `oas_cli/generators.py` | All file gen; long functions; duplicate format_value |
| Validation | `oas_cli/validators.py` | Spec + JSON schema; version field `open_agent_spec`; ValidationError.message |
| Data prep | `oas_cli/data_preparation.py` | AgentDataPreparator for template data |
| Code gen | `oas_cli/code_generation.py` | PythonCodeSerializer, CodeGenerator, TemplateVariableParser |
| Tools | `oas_cli/tools.py` | file_writer; TOOL_REGISTRY |
| Utils | `oas_cli/utils.py` | parse_response |
| Schema | `oas_cli/schemas/oas-schema.json` | open_agent_spec, agent, intelligence, tasks, prompts, roles |
| Templates | `oas_cli/templates/*.yaml` | minimal-agent, security-*, github-actions-*, cortex-* |
| Tests | `tests/*.py` | test_main (version), test_generators, test_enhanced_spec, test_cortex_integration, test_multi_engine, test_contract_validation, integration/ |
| Config | `pyproject.toml`, `ruff.toml`, `.pre-commit-config.yaml` | Version 1.0.9; Ruff in both; pytest marks not registered |

---

Once you decide which phases or items you want to tackle first, we can go through them step by step (e.g. patches for Phase 1, then Phase 2, then refactors in Phase 3).
