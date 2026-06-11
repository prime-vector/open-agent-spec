# OA Conformance Adapter Protocol — v1

This document defines the contract between the OA conformance **harness** and a
runtime **adapter**. Any runtime — Python, Node, Rust, anything — can be
certified against the OA specification by providing an executable that speaks
this protocol. The harness owns case discovery, assertion logic, and reporting;
the adapter owns nothing except running one case through its runtime.

## Overview

```
┌─────────────┐   case JSON (stdin)    ┌──────────────┐
│   Harness    │ ────────────────────▶ │   Adapter     │──▶ runtime under test
│ (one impl)   │ ◀──────────────────── │ (per runtime) │
└─────────────┘   result JSON (stdout) └──────────────┘
```

An adapter is **any executable**. The harness invokes it in one of two modes:

| Invocation | Purpose |
|---|---|
| `<adapter> --capabilities` | Declare runtime identity and supported features |
| `<adapter>` (case on stdin) | Execute one conformance case |

## Mode 1: `--capabilities`

The adapter MUST print a JSON object to stdout and exit 0:

```json
{
  "protocol": 1,
  "runtime": "python-reference",
  "version": "1.5.1",
  "capabilities": [
    "core",
    "depends-on",
    "delegation",
    "response-format-text",
    "output-schema-validation",
    "contracts"
  ]
}
```

- `protocol` — integer protocol version this adapter speaks (currently `1`).
- `runtime` — short identifier used in reports.
- `version` — runtime version string.
- `capabilities` — list of capability identifiers (see Capability Registry below).

## Mode 2: Case execution

The harness writes one case as JSON to the adapter's **stdin** and reads one
result from its **stdout**. The adapter MUST exit 0 whether the case passes or
fails — a non-zero exit means the *adapter itself* crashed and is reported as
an infrastructure error, not a conformance failure.

### Input (stdin)

```json
{
  "protocol": 1,
  "spec": "<the spec document as a YAML string>",
  "invoke": {
    "task": "summarise",
    "input": {"document": "The sky is blue."},
    "override_system": null,
    "override_user": null
  },
  "mock_responses": {"extract": "{\"facts\": \"...\"}", "summarise": "{...}"},
  "files_dir": "/tmp/oa-case-abc123"
}
```

- `spec` — the OA YAML document under test, as text.
- `invoke.task` — task to run (may be null for single-task auto-selection).
- `invoke.input` — input object.
- `invoke.override_system` / `override_user` — CLI-style prompt overrides, or null.
- `mock_responses` — canned LLM responses keyed by task name. See Mocking below.
- `files_dir` — absolute path to a temp directory. When the case declares
  `uses_files`, the harness materialises the main spec as `main.yaml` plus all
  helper files into this directory before invoking the adapter. The adapter
  MUST run the spec from `<files_dir>/main.yaml` (so relative delegation paths
  resolve) whenever `files_dir` is non-null.

### Output (stdout)

On success:

```json
{
  "protocol": 1,
  "status": "ok",
  "result": {
    "task": "summarise",
    "input": {"document": "...", "facts": "..."},
    "prompt": "<system>\n\n<user>",
    "engine": "openai",
    "model": "gpt-4o",
    "raw_output": "{\"summary\": \"...\"}",
    "output": {"summary": "..."},
    "chain": {"extract": { "...same envelope shape..." }},
    "delegated_to": "/abs/path/spec.yaml#task"
  }
}
```

On a structured runtime error:

```json
{
  "protocol": 1,
  "status": "error",
  "error": {
    "error": "<human-readable message>",
    "code": "CHAIN_CYCLE_ERROR",
    "stage": "routing",
    "task": "c"
  }
}
```

The `result` object is the **result envelope** defined in the OA specification
(§10). `chain` and `delegated_to` are present only when applicable. The harness
asserts against this envelope using the case's `expect` / `expect_error`
clauses; the adapter performs no assertions itself.

## Mocking

Conformance cases never call real LLMs. `mock_responses` maps task names to the
raw string the "model" should return when that task is invoked. The adapter
MUST arrange for its runtime's provider layer to return these canned responses.
*How* is the adapter's private business — monkeypatching, dependency injection,
a mock engine — the protocol only requires that the response for task `X` is
the string at `mock_responses[X]`.

Matching rule (same as the reference implementation): a response is selected
for a task when the task's name appears in the rendered system or user prompt;
otherwise responses are consumed in declaration order. Each response is
consumed at most once.

## Capability Registry — protocol v1

| Capability | Meaning |
|---|---|
| `core` | Spec load/validation, single-task run, prompt resolution, JSON output parsing, structured errors |
| `depends-on` | Dependency chains: merge semantics, cycle detection, `chain` envelope key |
| `delegation` | `spec:`/`task:` delegation, local paths, cycle detection, `delegated_to` |
| `registry` | `oa://` and `https://` remote spec resolution |
| `response-format-text` | `response_format: text` raw passthrough |
| `output-schema-validation` | Output validated against task `output` schema (OUTPUT_SCHEMA_ERROR) |
| `history` | `history` input convention threaded into provider calls |
| `tools` | Tool declaration and dispatch (native backends at minimum) |
| `sandbox` | IIS `sandbox:` enforcement (SANDBOX_* error codes) |
| `contracts` | Behavioural contract enforcement (CONTRACT_VIOLATION) |

Cases declare what they need via a `requires:` key (a single capability or a
list). The harness skips cases whose requirements are not in the adapter's
declared capabilities and reports them as **UNSUPPORTED** — distinct from PASS
and FAIL. Cases without `requires:` are implicitly `core`.

**Honesty rule:** a runtime MUST NOT declare a capability it does not enforce.
In particular, a runtime that does not implement `sandbox` MUST refuse to run
specs that declare a `sandbox:` block (rather than silently ignoring it).
Silent degradation of a security feature is a conformance violation of its own.

## Versioning

This protocol is versioned independently of the OA spec. Breaking changes bump
`protocol`. The harness rejects adapters that declare a different protocol
major version.
