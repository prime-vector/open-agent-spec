# Sandboxed Agent — IIS Demo

Demonstrates **Immutable Inference Sandboxing (IIS)**: the runner enforces hard
execution constraints _before_ any tool call reaches the I/O layer.

## What IIS is

| Layer | Concern |
|-------|---------|
| **OAS `sandbox:`** | Hard mechanical block — binary allow/deny enforced by the runner |
| **BCE `behavioural_contract:`** | Soft policy audit — validated after the run |

OAS owns **what a task can do**. BCE owns **what a task should do**. They never overlap.

## Sandbox keys

```yaml
sandbox:
  tools:
    allow: [file.read, http.get]   # allowlist (mutually exclusive with deny)
    # deny: [file.write]           # denylist alternative
  http:
    allow_domains: [api.example.com]  # subdomains are also allowed
  file:
    allow_paths: [./data/]            # resolved to absolute paths at check time
```

A **root-level** `sandbox:` applies to every task. A **task-level** `sandbox:`
completely overrides the root for that task, so you can tighten constraints
per-task without relaxing the global defaults.

## Error codes

| Code | Trigger |
|------|---------|
| `SANDBOX_TOOL_VIOLATION` | Tool name blocked by `allow` / `deny` |
| `SANDBOX_DOMAIN_VIOLATION` | HTTP host not in `allow_domains` |
| `SANDBOX_PATH_VIOLATION` | File path outside `allow_paths` |

## Running the demo

```bash
# Summarise the sample report (file.read allowed inside data/)
oa run spec.yaml --task summarise --input '{"filename": "report.txt"}'

# Check OpenAI status (http.get allowed to status.openai.com)
oa run spec.yaml --task check_status --input '{}'
```

## Seeing a violation

Try reading a file outside the allow_paths:

```bash
oa run spec.yaml --task summarise --input '{"filename": "../../README.md"}'
# → OARunError: SANDBOX_PATH_VIOLATION
```

Or call a blocked domain (by temporarily editing the prompt):

```bash
# The runner blocks the call before it reaches the network.
```

## What IIS deliberately does NOT do

- **Prompt injection detection** — subjective, evolving, belongs in BCE
- **PII scanning** — policy-level concern, belongs in BCE
- **Memory / session management** — out of scope for a stateless runner
- **Parallel execution or retry logic** — not OAS concerns
