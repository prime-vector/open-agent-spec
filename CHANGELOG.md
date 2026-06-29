# Changelog

All notable changes to **open-agent-spec** (Open Agent CLI) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Token usage & best-effort cost in the result envelope** — providers now capture the API `usage` object they previously discarded. Every single-call result envelope includes a `usage` block (`prompt_tokens`/`completion_tokens`/`total_tokens`, normalised across the OpenAI and Anthropic shapes) plus a best-effort `estimated_cost_usd` for models in a built-in price table (`null` for unknown models — never guessed). `oa run` shows a compact `<total> tok · ~$<cost>` summary. Multi-turn tool-calling is covered too — usage is summed across every turn of the loop. `usage` is `null` only when the engine omits counts (some local servers, the Codex CLI, custom routers). See [`docs/REFERENCE.md`](docs/REFERENCE.md).
- **Reasoning-effort tiers (experimental)** — declare `intelligence.config.reasoning_effort: low|medium|high` and OA maps it to each engine's native control: OpenAI `reasoning_effort` (Chat Completions) / `reasoning.effort` (Responses API), Codex `-c model_reasoning_effort=<tier>`, and Anthropic `output_config.effort` paired with adaptive thinking. Opt-in and model-specific (a reasoning-capable model is required); `oa validate` enforces the `low|medium|high` enum. For OpenAI reasoning models OA sends `max_completion_tokens` (not `max_tokens`) and omits `temperature`, both of which those models reject. Validated live against Opus 4.8 and OpenAI `o4-mini` via [`scripts/verify_reasoning.py`](scripts/verify_reasoning.py).

- **Custom cost rates / cost override** — `estimated_cost_usd` can now be overridden so it reflects real spend rather than list price. Rates resolve in layers (first match wins): per-spec `intelligence.config.pricing` (`{input_per_1m, output_per_1m}` or `"none"` to disable) → the `OA_PRICING` env var (JSON `{model: {input, output}}` org-wide, or `"none"`) → the built-in table. Use it for enterprise-negotiated rates, or to suppress the dollar figure on subscription/local models (token counts are always reported regardless). Built-in table refreshed with current OpenAI GPT-5.x and legacy o-series rates. Invalid overrides (negative rate, malformed `OA_PRICING`, unrecognised `pricing` value) **fail closed** rather than silently reverting to list prices, surfacing through the runner as a dedicated `PRICING_CONFIG_ERROR` (stage `cost`) distinct from a generic run failure.

### Spec
- **Formal spec & canonical schema updated** — `spec/open-agent-spec-1.5.md` and `spec/schema/oas-schema-1.5.json` now normatively define the `usage` result-envelope block (§8.3, incl. multi-call summation and the best-effort `estimated_cost_usd` rules), `config.reasoning_effort` (§5.5), the `config.pricing` cost override and its fail-closed semantics (§8.3), and the `PRICING_CONFIG_ERROR` error code (§11.2). Additive and backward-compatible — these formalise the features above in the 1.5 draft.

### Fixed
- **Anthropic `temperature` rejected by Opus 4.7/4.8 & Fable 5** — these models reject sampling params, but the Anthropic provider always sent `temperature` (it was only dropped on the reasoning path), so a plain `engine: anthropic, model: claude-opus-4-8` call returned HTTP 400. The provider now omits `temperature` for models that reject it; other models are unchanged.
- **Non-OpenAI engine endpoints** — specs without an explicit `intelligence.endpoint` no longer inherit the OpenAI base URL. The hardcoded default was merged over the per-engine defaults in the provider registry (config wins), clobbering the correct endpoint for `anthropic`, `grok`, `local`, and `cortex` — e.g. a no-endpoint `engine: anthropic` spec (the README's own example) routed to `api.openai.com` and returned 404. Each provider now applies its own default endpoint when none is set.

## [1.5.2] - 2026-05-11

### Added
- **npm CLI `oa validate`** — `oa validate --spec <path>` in the Node.js CLI now mirrors the Python command, performing the same required-field check (`open_agent_spec`, `agent`, `intelligence`, `tasks`) without any model call. Closes #25 for the npm port.
- **Per-engine README examples** — README "Multiple engines" section now ships a runnable block per engine (OpenAI, Anthropic, Grok/xAI, Local, Cortex, Custom), each showing the `intelligence:` YAML, the env var to export, and the matching `oa run` command. Closes #22.

### Changed
- **npm CLI `--version`** — the Node CLI now reads the version dynamically from `package.json` instead of a hard-coded literal, so `oa --version` cannot drift from the published release. Closes #23 for the npm port. The Python CLI has read its version from package metadata since 1.5.0.

## [1.5.1] - 2026-04-13

### Changed
- README updated with full feature showcase (multi-task, tools, spec composition, registry, history threading, memory retriever, IIS sandboxing, behavioural contracts, engines, npm CLI).
- OAS acronym replaced with OA throughout all prose, docs, comments, and examples.
- Specification table links updated to 1.5 artifacts.

## [1.5.0] - 2026-04-13

### Added
- **Immutable Inference Sandboxing (IIS)** — a new `sandbox:` spec key that lets you declare hard execution constraints enforced by the runner *before* any tool call reaches the I/O layer. Three constraint types: `tools.allow`/`deny` (tool allowlist/denylist), `http.allow_domains` (HTTP host restriction for `http.get`/`http.post`), `file.allow_paths` (file path restriction for `file.read`/`file.write`). Sandbox can be declared at root level (all tasks) or per-task (overrides root). Three structured error codes: `SANDBOX_TOOL_VIOLATION`, `SANDBOX_DOMAIN_VIOLATION`, `SANDBOX_PATH_VIOLATION`.
- **Chain-wide input immutability** — every task boundary now receives a deep copy of its input. Upstream chain mutations can never leak into downstream inputs or back to the caller's original dict.
- **History threading** — pass a `history` array in the input to any task and it is injected into the LLM message list between system and user turns, enabling stateless multi-turn conversations without OA managing persistence.
- **Memory-retriever registry spec** (`oa://prime-vector/memory-retriever`) — an LLM re-ranker that accepts pre-fetched `candidates` (prior conversation turns from your own store) and returns the most contextually relevant ones as a `history` array, ready to inject into any chat-capable task.
- **Spec Registry** — `openagentspec.dev/registry/` hosts shareable specs via `oa://` shorthand URLs. Includes seed specs: `oa://prime-vector/summariser`, `classifier`, `sentiment`, `code-reviewer`, `keyword-extractor`, and `memory-retriever`.
- **npm CLI** (`@prime-vector/open-agent-spec`) — a native TypeScript port of `oa run` for Node.js environments. Supports OpenAI and Anthropic providers, `depends_on` chains, and history threading. No Python required.
- **CLI terminal UI redesign** — new compact bot-face banner (`Panel.fit`), live inference spinner, syntax-highlighted JSON output panel, smart string rendering (Markdown for prose, extracted fenced JSON), and a unified help panel combining the bot face and command reference.
- **`examples/sandboxed-agent/`** — demo spec showing root-level sandbox, per-task sandbox override, and the OA vs BCE boundary.
- **`examples/chat-agent/`** and **`examples/memory-chat/`** — reference implementations for history threading and the memory-retriever re-ranker pattern.
- **Formal spec** `spec/open-agent-spec-1.5.md` and canonical schema `spec/schema/oas-schema-1.5.json` updated with all new keys (`sandbox:`, `history` threading convention).

### Changed
- `--quiet` mode: plain-string outputs are now written directly (no `json.dumps` quoting/escaping). Dict/list outputs remain pretty-printed JSON.
- `OARunError` now propagates through the `except Exception` catch block in `_run_single_task` so structured errors (sandbox violations, delegation errors) are never re-wrapped as `RUN_ERROR`.
- `oa://` registry URL scheme formalised — the runner resolves `oa://<owner>/<name>` to `https://openagentspec.dev/registry/<owner>/<name>/latest/spec.yaml`.
- BCE `allowed_tools` field noted as a future rename to `expected_tools` (audit-not-enforcement semantics) in docs and REFERENCE.md.

### Fixed
- Banner `Panel()` replaced with `Panel.fit()` — banner no longer stretches to full terminal width.
- `--input <file.txt>` correctly maps file content to the single required string field; `.json` files are always parsed as JSON objects.
- File-reader example prompt now explicitly names required output fields (`summary`, `key_points`) to prevent the model omitting them.

## [1.4.1] - 2026-04-12

### Added
- **Formal specification** — `spec/open-agent-spec-1.4.md`, an RFC 2119-style document defining the complete OA 1.4.0 standard. An independent implementor can build a conforming runtime from this document alone.
- **Conformance test suite** — 29 YAML test cases across 6 categories (schema validation, prompt resolution, depends_on chains, delegation, response format, error model) with a reference runner at `spec/conformance/runner/conformance_runner.py`.
- **Extensions documentation** — `spec/extensions/README.md` describing the three extension mechanisms: tools (native/mcp/custom), behavioural contracts, and custom engines.
- **Canonical schema** — `spec/schema/oas-schema-1.4.json` placed alongside the spec as a normative artifact.
- **Output schema validation** — the runner now validates parsed JSON output against the declared output schema (`response_format: "json"`). Missing required fields raise `RUN_ERROR`.

### Changed
- `TASK_NOT_FOUND` error stage clarified: uses `routing` for direct task lookup and `depends_on` references, `delegation` when the task is missing in a delegated spec.

## [1.4.0] - 2026-03-19

### Added
- **Spec composition via task delegation** — a task can now declare `spec: ./path/to/spec.yaml` + `task: name` to delegate its implementation to another spec. The runner loads the referenced spec and executes that task transparently, returning the result under the coordinator's task name. Relative `spec:` paths resolve from the calling spec's directory.
- Cycle detection for spec delegation: A→B→A loops raise a new `DELEGATION_CYCLE_ERROR` before any model call is made.
- `delegated_to` field in the result envelope for traceability (`"spec_path#task_name"`).
- New `spec` and `task` fields in the task JSON schema definition.
- `examples/spec-composition/` — coordinator spec + two shared specialist specs (`summariser.yaml`, `sentiment.yaml`) and a demo `run.sh`.
- `tests/test_spec_composition.py` — 13 tests covering basic delegation, `depends_on` chains across delegated tasks, cycle detection, error cases, relative path resolution, and validator.
- **Tool use / MCP integration** — declarative tool definitions in the spec with three backends: `native` (built-in zero-dependency tools: `file.read`, `file.write`, `http.get`, `http.post`, `env.read`), `mcp` (JSON-RPC 2.0 over raw HTTP to any MCP server), and `custom` (dynamic Python class loading).
- `ToolProvider` abstract base class and full provider registry (`oas_cli/tool_providers/`).
- Multi-turn `_invoke_with_tools` loop in the runner for model-driven tool calling.
- Native tool support in `OpenAIProvider` and `AnthropicProvider` via their function-calling APIs.
- `examples/file-reader/`, `examples/mcp-local/`, `examples/mcp-search/` — self-contained tool demos.
- **Provider interface** — all LLM calls now go through a minimal `IntelligenceProvider.invoke()` abstraction using raw HTTP (no OpenAI/Anthropic SDK required). Engine support extended to `grok`, `xai`, `local`, and `custom`.
- **Test harness** (`oa test`) — run eval cases against a spec with assertions on task output.
- **Spec reuse / composition** — `depends_on` formally documented as a data contract (not execution control), with an explicit hard wall against branching, conditionals, loops, retries, and dynamic routing.

### Changed
- `output` removed from the task JSON schema `required` array; enforced by the Python validator for non-delegated, non-multi-step tasks (enables delegated tasks to omit inline schema).
- `depends_on` description in schema and `REFERENCE.md` updated with the design principle and a hard-wall table of features OA intentionally does not support.
- Spec version aligned to **1.4.0** across all examples, templates, `.agents/`, Website content, REFERENCE.md, CONTRIBUTING.md, and README.

---

## [Unreleased]

### Added
- This changelog.
- **Agents-as-code documentation** — new section in REFERENCE.md explaining the `.agents/` pattern, bundled examples table, and scaffold/run/generate workflows.
- Agents-as-code overview section in README.md with link to REFERENCE.md.
- "Adding an agent-as-code example" guide in CONTRIBUTING.md.
- Codex engine configuration reference (sandbox modes, cwd) in REFERENCE.md.

### Changed
- Standardised spec version to `1.0.9` across all examples, templates, and docs (CONTRIBUTING.md, README.md, `.agents/`, Website content).
- README quick-start YAML now includes the required `open_agent_spec` version field.
- Added required `type: llm` field to all `intelligence` examples in README.md and REFERENCE.md (field is required by schema but was missing from docs).

### Fixed
- Removed broken references to non-existent `security-threat-analyzer.yaml` template and `SECURITY_TEMPLATES.md` from REFERENCE.md.

---

## [1.3.0] - 2026-03-17

### Changed
- Spec version aligned to **1.3.0** across all `.agents/`, templates, Website content, REFERENCE.md, CONTRIBUTING.md, and README.
- README rewritten for linear "golden path" first-run experience; Homebrew removed from Quick Start (tap not yet published).
- REFERENCE.md now distinguishes "generated by `oa init aac`" vs "shipped in this repository" for `.agents/` specs.
- PR template shortened to What/Why/How-tested/Breaking-changes format.

### Fixed
- **CI repair agent no longer commits diagnostic logs to branches.** Generated artifacts (`failed-log.txt`, `repair-*.json`) are written to `/tmp/repair-artifacts/` and uploaded as GitHub Actions artifacts instead. `git add -u` replaces `git add -A` so only tracked source changes are staged.
- `.gitignore` updated to block CI repair artifacts from accidental commits.
- Removed Homebrew install path from README (tap not published; `docs/packaging/HOMEBREW.md` was already deleted).

---

## [1.3.0] - 2026-03-14

### Changed
- Spec version and docs aligned to **1.3.0** (open_agent_spec in examples, templates, Website, REFERENCE, CONTRIBUTING, HOMEBREW).
- Test plan doc renamed to `docs/TEST_PLAN_1.3.0.md`.

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
