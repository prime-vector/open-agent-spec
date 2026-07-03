# AGENTS.md — Guidance for Coding Agents Working in This Repository

This repository is the home of **Open Agent Spec (OA)** — a declarative
standard for defining AI agents as portable YAML contracts.

## The one rule that matters

**Agent behaviour in this repo is defined as OA specs, not as code or prose.**
If you need an agent to do something, find or write a spec under `examples/`
and execute it with the CLI — do not hand-roll LLM calls, prompts, or agent
logic inline.

```bash
# Validate a spec
oa validate --spec examples/file-reader/file-reader.yaml

# Run a task from a spec
oa run --spec examples/skill-wrapper/spec.yaml --task summarise \
  --input '{"text": "..."}'
```

This file (AGENTS.md) is guidance; the specs are contracts. When they appear
to conflict, the spec wins — it is schema-validated and runner-enforced,
this file is not.

## Repo layout

| Path | What it is |
|---|---|
| `oas_cli/` | Reference Python runtime + CLI |
| `npm/` | TypeScript/npm runtime (`@prime-vector/open-agent-spec`) |
| `spec/` | The formal specification, JSON Schemas, and conformance suite |
| `spec/conformance/` | Runtime-agnostic conformance harness and cases |
| `examples/` | Runnable example specs — the canonical usage reference |
| `docs/` | Reference docs and proposals |

## Development commands

```bash
# Python: install, test, lint
pip install -e ".[dev]"
python -m pytest tests/ -q
ruff check . --exclude test_output/ && ruff format --check . --exclude test_output/
mypy oas_cli tests

# npm runtime: build and typecheck
cd npm && npm ci && npm run build

# Conformance suite (certifies both runtimes against the standard)
python -m spec.conformance.harness.harness --adapter python --adapter node
```

## Conventions

- Changes to runtime behaviour need tests, and if the behaviour is normative,
  a conformance case under `spec/conformance/cases/` so no runtime can
  regress silently.
- The Python and npm runtimes must stay in parity — if you change execution
  semantics in one, change the other and re-run the conformance suite.
- OA's boundaries are deliberate: no orchestration, no conditionals or
  branching in specs, no prose-as-contract. Read
  `docs/proposals/markdown-interop.md` before proposing integrations with
  markdown agent patterns (AGENTS.md / SKILL.md).
- Use `OA` (not `OAS`) in prose; technical identifiers like `oas_cli/` and
  `oas-schema-*.json` keep their existing names.
