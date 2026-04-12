# Open Agent Spec — Formal Specification

**Version:** 1.4.0  
**Status:** Draft  
**Date:** 2026-04-12

---

## Abstract

This document defines the Open Agent Spec (OAS) 1.4.0. It specifies the structure of an OAS document, the semantics that a conforming runtime MUST implement, and the boundaries of what OAS deliberately does not do. An independent implementor MUST be able to build a conforming runtime from this document alone.

---

## 1. Introduction & Scope

Open Agent Spec is a YAML-based document format for declaring AI agent tasks. A spec document describes:

- the agent's identity and role
- the LLM engine and model to use
- one or more tasks, each with typed input/output schemas and prompts
- optional data dependencies between tasks
- optional tool declarations
- optional behavioural contracts

OAS is **not** an execution framework, an orchestration engine, or an AI library. It is a document standard — a machine-readable contract between a spec author and a conforming runtime.

### 1.1 What OAS is NOT

OAS explicitly prohibits — and a conforming runtime MUST NOT implement — the following:

| Feature | Classification | Rationale |
|---------|---------------|-----------|
| Conditional branching | Execution control | Belongs in the calling platform |
| Loops and retries | Runtime policy | Belongs in the calling platform |
| Parallel task execution | Scheduling | Belongs in the calling platform |
| Fallback task routing | Dynamic routing | Belongs in the calling platform |
| Dynamic task selection | Orchestration | Belongs in the calling platform |

`depends_on` is a **data dependency declaration**, not a workflow instruction. It expresses what data a task needs, not how execution should proceed. See Section 7 for the full semantics.

### 1.2 Key Words

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://datatracker.ietf.org/doc/html/rfc2119).

### 1.3 Normative References

- [RFC 2119](https://datatracker.ietf.org/doc/html/rfc2119) — Key words for use in RFCs to indicate requirement levels
- [JSON Schema Draft-07](https://json-schema.org/specification-links.html#draft-7) — Schema language used for input/output validation
- `spec/schema/oas-schema-1.4.json` — Machine-readable JSON Schema for OAS documents (canonical, normative)

---

## 2. Terminology

**Spec document** — A YAML file conforming to this specification.

**Runtime** — Any software that loads a spec document and executes tasks according to the semantics defined herein.

**Task** — A named unit of work defined in a spec. A task has typed inputs, typed outputs, prompts, and an LLM engine configuration.

**Chain** — A sequence of tasks linked by `depends_on` declarations, where earlier task outputs are merged into later task inputs.

**Delegation** — A task that forwards its execution to a task in a different spec document (via `spec:` + `task:`).

**Result envelope** — The structured object a runtime returns after executing a task.

**Behavioural contract** — An optional set of constraints on a task's output, enforced after the LLM response is parsed.

**Provider** — The implementation that communicates with a specific LLM engine (e.g. OpenAI, Anthropic).

**Template variable** — A `{{ key }}` placeholder in a prompt string that is substituted with the corresponding input value at run time.

---

## 3. Document Model

### 3.1 Format

An OAS document MUST be a valid YAML file that deserializes to a JSON-compatible mapping (object). The document MUST validate against the JSON Schema at `spec/schema/oas-schema-1.4.json`.

### 3.2 Top-Level Keys

| Key | Required | Type | Description |
|-----|----------|------|-------------|
| `open_agent_spec` | REQUIRED | string | Spec version. MUST match pattern `^(1\.(0\.[4-9]|[1-9]\.[0-9]+)|[2-9]\.[0-9]+\.[0-9]+)$` |
| `agent` | REQUIRED | object | Agent identity. See Section 4. |
| `intelligence` | REQUIRED | object | LLM engine configuration. See Section 5. |
| `tasks` | REQUIRED | object | Task definitions. See Section 6. |
| `prompts` | OPTIONAL | object | Global prompt fallbacks. See Section 9. |
| `behavioural_contract` | OPTIONAL | object | Global behavioural contract baseline. See Section 11. |
| `tools` | OPTIONAL | object | Tool declarations. See Section 10. |
| `logging` | OPTIONAL | object | Logging configuration (implementation-specific). |
| `interface` | OPTIONAL | object | Reserved for future use. |

A runtime MUST ignore unknown top-level keys (forward compatibility). A runtime MUST reject documents that fail schema validation.

---

## 4. Agent Metadata

The `agent` object identifies the agent.

```yaml
agent:
  name: "my-agent"           # REQUIRED
  description: "..."         # REQUIRED
  role: "analyst"            # OPTIONAL
```

### 4.1 Fields

| Field | Required | Type | Constraints |
|-------|----------|------|-------------|
| `name` | REQUIRED | string | Non-empty |
| `description` | REQUIRED | string | Free text |
| `role` | OPTIONAL | string | One of: `analyst`, `reviewer`, `chat`, `retriever`, `planner`, `executor` |

`role` is informational. A runtime MAY use it for routing or logging but MUST NOT alter task execution semantics based on it.

---

## 5. Intelligence Configuration

The `intelligence` object selects the LLM engine and model.

```yaml
intelligence:
  type: "llm"                # REQUIRED — currently only "llm" is valid
  engine: "openai"           # REQUIRED
  model: "gpt-4o"            # REQUIRED
  endpoint: "https://..."    # OPTIONAL
  config:                    # OPTIONAL
    temperature: 0.7
    max_tokens: 1000
    top_p: 1.0
    frequency_penalty: 0.0
    presence_penalty: 0.0
```

### 5.1 Required Fields

| Field | Type | Constraints |
|-------|------|-------------|
| `type` | string | MUST be `"llm"` |
| `engine` | string | One of the values in Section 5.2 |
| `model` | string | Non-empty model identifier |

### 5.2 Engines

| Engine value | Aliases | Default endpoint | Auth env var |
|---|---|---|---|
| `openai` | — | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `anthropic` | — | `https://api.anthropic.com` | `ANTHROPIC_API_KEY` |
| `grok` | `xai` | `https://api.x.ai/v1` | `XAI_API_KEY` |
| `cortex` | — | User-provided via `endpoint` | `OPENAI_API_KEY` |
| `codex` | — | (Codex CLI subprocess) | Codex CLI session |
| `local` | — | `http://localhost:11434/v1` | _(none required)_ |
| `custom` | — | User-provided via `endpoint` | _(user-defined)_ |

A runtime MUST support at minimum `openai` and `anthropic`. Other engines are RECOMMENDED.

`grok` and `xai` are aliases — a runtime MUST treat them identically.

### 5.3 Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `endpoint` | string (HTTP/HTTPS URL) | Overrides the default API endpoint |
| `config.temperature` | number [0, 2] | Sampling temperature |
| `config.max_tokens` | integer ≥ 1 | Maximum tokens to generate |
| `config.top_p` | number [0, 1] | Nucleus sampling parameter |
| `config.frequency_penalty` | number [-2, 2] | Frequency penalty |
| `config.presence_penalty` | number [-2, 2] | Presence penalty |

A runtime MUST pass `temperature` and `max_tokens` to the LLM API when provided. Unknown `config` keys MAY be forwarded to the underlying engine or ignored.

### 5.4 Custom Engine

When `engine` is `custom`, the spec MAY additionally specify:

```yaml
intelligence:
  type: "llm"
  engine: "custom"
  endpoint: "https://my-proxy.internal/v1"
  model: "my-model"
  module: "my_package.router.MyRouter"   # optional Python class
```

If `module` is provided, a runtime implementing the Python reference implementation MUST load the class via dynamic import. The class MUST implement:

```python
class MyRouter:
    def __init__(self, endpoint: str, model: str, config: dict): ...
    def run(self, prompt: str, **kwargs) -> str: ...  # returns JSON string
```

---

## 6. Task Definitions

The `tasks` object is a map from task name to task definition. Task names MUST match `^[a-zA-Z0-9_-]+$`.

```yaml
tasks:
  greet:
    description: "Say hello"     # REQUIRED
    input: ...                   # OPTIONAL but recommended
    output: ...                  # OPTIONAL but required for non-delegated tasks
    prompts: ...                 # OPTIONAL (falls back to global prompts)
    response_format: "json"      # OPTIONAL — "json" (default) or "text"
    depends_on: [other_task]     # OPTIONAL — data dependency
    tools: [tool_name]           # OPTIONAL — allowed tools from top-level tools:
    behavioural_contract: ...    # OPTIONAL — merged with global contract
    timeout: 30                  # OPTIONAL — seconds
    spec: "./other.yaml"         # OPTIONAL — delegation target
    task: "task_name"            # OPTIONAL — task name in delegated spec
```

### 6.1 Task Fields

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `description` | REQUIRED | string | Human-readable description |
| `input` | OPTIONAL | object | Input schema (JSON Schema subset) |
| `output` | CONDITIONAL | object | Output schema. REQUIRED for non-delegated tasks without `depends_on` that are expected to return typed output. |
| `prompts` | OPTIONAL | object | Per-task system/user prompts |
| `response_format` | OPTIONAL | `"json"` \| `"text"` | Output parsing mode. Default: `"json"` |
| `depends_on` | OPTIONAL | array of string | Data dependency declarations |
| `tools` | OPTIONAL | array of string | Tool names from the top-level `tools:` block |
| `behavioural_contract` | OPTIONAL | object | Per-task contract (merged with global) |
| `timeout` | OPTIONAL | integer ≥ 1 | Timeout in seconds |
| `spec` | OPTIONAL | string | Path or URL to a spec to delegate to |
| `task` | OPTIONAL | string | Task name in the delegated spec |

### 6.2 Input Schema

```yaml
input:
  type: "object"         # MUST be "object" when present
  properties:
    field_name:
      type: "string"     # REQUIRED — one of: string, number, integer, boolean, array, object
      description: "..."
      default: ...       # OPTIONAL
      enum: [...]        # OPTIONAL
      minimum: ...       # OPTIONAL — numeric constraints
      maximum: ...
      minLength: ...     # OPTIONAL — string constraints
      maxLength: ...
      pattern: "regex"   # OPTIONAL
      items: {...}       # OPTIONAL — for array type
  required: [field_name]
```

A runtime MUST validate that all `required` input fields are present before invoking the LLM. Missing required fields MUST raise `CHAIN_INPUT_MISSING`.

### 6.3 Output Schema

The output schema has the same structure as the input schema. A runtime MUST validate the parsed output against the output schema when `response_format` is `"json"`. Validation failures MUST be surfaced as errors.

---

## 7. Task Execution Semantics

### 7.1 Single Task (no `depends_on`)

Given a task with no `depends_on`:

1. Validate required input fields are present. Raise `CHAIN_INPUT_MISSING` if any are missing.
2. Resolve the system and user prompts (see Section 9).
3. Substitute template variables in the user prompt (see Section 9.2).
4. Build the intelligence configuration from `intelligence:`.
5. If `tools:` is declared for the task, enter the tool-call loop (see Section 10.3). Otherwise, call the LLM once.
6. Parse the raw output (see Section 8).
7. Enforce the behavioural contract if declared (see Section 11).
8. Return the result envelope (see Section 8).

### 7.2 Dependent Tasks (`depends_on`)

`depends_on` is a **data contract**, not an execution directive. When a task declares `depends_on: [dep1, dep2, ...]`:

1. Cycle detection: if any dependency (transitively) references the calling task, raise `CHAIN_CYCLE_ERROR` before any LLM call.
2. For each dependency in listed order:
   a. Execute the dependency task recursively (applying the same semantics).
   b. Merge its `output` into the running input map.
3. Merge rule: `merged = {**caller_input, **dep1_output, **dep2_output, ...}` — later entries win on key collision.
4. Validate required input fields against the merged map. Raise `CHAIN_INPUT_MISSING` if any are missing.
5. Execute the calling task with the merged input.

A runtime MUST NOT execute dependent tasks in parallel. A runtime MUST NOT execute a task until all its declared dependencies have produced output. A runtime MUST detect circular dependency chains and raise `CHAIN_CYCLE_ERROR` before invoking any model.

**The following are permanently out of scope for `depends_on`:**
- Conditional execution (if/else)
- Loop control (while/for)
- Retry semantics
- Parallel fan-out
- Dynamic task selection based on output values

### 7.3 Spec Delegation (`spec:`)

A task with a `spec:` field is a *delegated task*. Its implementation is defined in another spec document.

```yaml
tasks:
  summarise:
    description: "Summarise a document"
    spec: "./summariser.yaml"    # local path (relative to calling spec) or oa:// URL
    task: "summarise"            # optional — defaults to the calling task's name
```

Delegation semantics:

1. Resolve the `spec:` reference:
   - Local path: resolve relative to the calling spec's directory.
   - `oa://namespace/name` or `oa://namespace/name@version`: expand to the registry URL.
   - `http://` or `https://`: fetch directly.
2. Load the referenced spec.
3. Identify the target task: use `task:` if provided, else use the calling task's name.
4. Validate the target task exists in the referenced spec. Raise `TASK_NOT_FOUND` if not.
5. Cycle detection: if the referenced spec is already in the delegation stack, raise `DELEGATION_CYCLE_ERROR`.
6. Execute the target task in the referenced spec, passing the current `input_data`.
7. Surface the result under the coordinator's task name.
8. Include `delegated_to: "<spec_path>#<task_name>"` in the result envelope.

When a task is delegated, the calling task's inline `prompts:` and `output:` schema are **ignored** — the referenced spec's task definition governs.

---

## 8. Response Format & Result Envelope

### 8.1 Response Format

`response_format` controls how the model's raw text output is processed:

| Value | Behaviour |
|-------|-----------|
| `"json"` (default) | Strip markdown code fences if present, then parse as JSON. Validate against output schema. |
| `"text"` | Return raw string as-is. Skip JSON parsing and output schema validation. |

**Markdown fence stripping:** If the model wraps its response in ` ```json ... ``` ` or ` ``` ... ``` `, a runtime MUST strip the fences before attempting JSON parsing.

**Parse failure:** If JSON parsing fails and `response_format` is `"json"`, a runtime SHOULD surface the raw string rather than raising a hard error, to preserve the response for inspection.

### 8.2 Result Envelope

A runtime MUST return the following envelope for every successfully executed task:

```json
{
  "task": "<task_name>",
  "output": <parsed_output>,
  "input": <input_data_used>,
  "prompt": "<system_prompt>\n\n<user_prompt>",
  "engine": "<engine_name>",
  "model": "<model_name>",
  "raw_output": "<model_raw_text>"
}
```

When the task has `depends_on` dependencies, the envelope MUST additionally include:

```json
{
  "chain": {
    "<dep_task_name>": {
      "task": "<dep_task_name>",
      "output": <dep_output>
    }
  }
}
```

Tasks with no `depends_on` MUST NOT include a `chain` key.

When a task is delegated (has `spec:`), the envelope MUST additionally include:

```json
{
  "delegated_to": "<spec_path>#<task_name>"
}
```

---

## 9. Prompt Resolution

### 9.1 Four-Level Precedence

For each task invocation, the system prompt and user prompt template are resolved independently using this priority order (highest wins):

| Priority | Source | How specified |
|----------|--------|---------------|
| 1 (highest) | CLI / runtime override | Passed at invocation time (`--system-prompt` / `--user-prompt`) |
| 2 | Per-task inline | `tasks.<name>.prompts.system` / `tasks.<name>.prompts.user` |
| 3 | Per-task map (legacy) | `prompts.<name>.system` / `prompts.<name>.user` |
| 4 (lowest) | Global fallback | `prompts.system` / `prompts.user` |

Each dimension (system, user) resolves independently. A runtime MUST apply this resolution order for every task invocation.

The global `prompts:` block at the top level is a fallback only — it does not apply to tasks that have their own `prompts:` block.

**Style A (preferred) — per-task prompts co-located with the task:**

```yaml
tasks:
  greet:
    prompts:
      system: "You greet people warmly."
      user: "{{ name }}"
```

**Style B (legacy) — keyed map under global `prompts:`:**

```yaml
prompts:
  greet:
    system: "You greet people warmly."
    user: "{{ name }}"
```

Both styles are valid. Style A takes priority over Style B when both are present for the same task.

### 9.2 Template Variable Substitution

The user prompt template MAY contain `{{ key }}` placeholders. After prompt resolution, a runtime MUST substitute each placeholder with the corresponding value from the task's input data.

**Supported substitution syntaxes** (a runtime MUST support both):

| Syntax | Example | Notes |
|--------|---------|-------|
| `{{ key }}` | `{{ name }}` | Jinja-style, with spaces |
| `{{ input.key }}` | `{{ input.name }}` | Jinja-style with `input.` prefix |
| `{key}` | `{name}` | Python-style single braces |

Substitution is applied to all three syntaxes. Values are stringified before substitution. A runtime MUST NOT fail if a placeholder references a key not present in the input — unresolved placeholders MAY be left as-is or substituted with an empty string (implementation choice, but SHOULD be documented).

When no user prompt is configured at any level, a runtime SHOULD default to `"{{ input }}"`.

---

## 10. Extension Points

### 10.1 Tools

Tools extend a task with the ability to call external services or execute code. The top-level `tools:` block declares tools; each task's `tools:` list specifies which declared tools it may use.

```yaml
tools:
  read_file:
    type: "native"
    native: "file.read"
    description: "Read a file from disk"

  search:
    type: "mcp"
    endpoint: "https://my-mcp-server.internal/mcp"
    description: "Search the web"
    headers:
      Authorization: "${SEARCH_TOKEN}"

  router:
    type: "custom"
    module: "my_package.tools.Router"
    description: "Custom routing logic"
    parameters:
      type: object
      properties:
        query: { type: string }
```

### 10.2 Tool Types

| `type` | Description | Required fields |
|--------|-------------|-----------------|
| `native` | Built-in zero-dependency tool | `native` (tool ID) |
| `mcp` | JSON-RPC 2.0 over HTTP to an MCP server | `endpoint` |
| `custom` | Dynamically loaded Python class | `module` |

**Native tool IDs** (built-in, a conforming runtime SHOULD support):

| ID | Operation |
|----|-----------|
| `file.read` | Read a file from disk |
| `file.write` | Write a file to disk |
| `http.get` | HTTP GET request |
| `http.post` | HTTP POST request |
| `env.read` | Read an environment variable |

### 10.3 Tool-Call Loop

When a task declares tools, the runtime MUST implement a multi-turn tool-call loop:

1. Send the system prompt, user prompt, and tool definitions to the LLM.
2. If the model returns tool calls, execute each call and feed results back as tool result messages.
3. Repeat until the model returns a final text response or a maximum iteration limit is reached.
4. A runtime MUST enforce a maximum iteration limit (the reference implementation uses 10). Exceeding the limit MUST raise `RUN_ERROR`.

When the provider does not support native tool calling, a runtime SHOULD inject tool descriptions into the system prompt and fall back to a single LLM call.

### 10.4 Behavioural Contracts

Behavioural contracts declare constraints on task output. They are enforced **after output parsing, before the result is returned**. For chained tasks, contract enforcement runs **before** the dependency's output is merged into the next task's input.

```yaml
behavioural_contract:      # global — applies to all tasks as a baseline
  version: "1.0"
  response_contract:
    output_format:
      required_fields: [confidence]

tasks:
  summarise:
    behavioural_contract:  # per-task — merged with global
      version: "1.0"
      response_contract:
        output_format:
          required_fields: [summary]
    # effective required_fields: [confidence, summary]
```

**Contract merge rules:**

| Value type | Merge behaviour |
|-----------|-----------------|
| Arrays | Unioned (order preserved, duplicates removed) |
| Objects/dicts | Merged recursively with the same rules |
| Scalars | Per-task value wins |

A runtime MUST apply these merge rules when combining global and per-task contracts.

**Contract enforcement is skipped when:**
- `response_format: "text"` (field checks are meaningless on raw strings)
- The output could not be parsed as a dict (warning logged, execution continues)
- The behavioural contracts library is not installed (warning logged, execution continues)

A contract violation MUST raise `CONTRACT_VIOLATION`.

---

## 11. Error Model

### 11.1 Structured Errors

A runtime MUST surface errors as structured objects with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `error` | string | Human-readable error message |
| `code` | string | Machine-readable error code (see Section 11.2) |
| `stage` | string | Pipeline stage where the error occurred |
| `task` | string | Task name involved (OPTIONAL — omit when unknown) |

### 11.2 Error Codes

| Code | Stage | Trigger |
|------|-------|---------|
| `SPEC_LOAD_ERROR` | `load` | File not found, YAML parse error, spec does not validate against schema |
| `TASK_NOT_FOUND` | `routing` or `delegation` | `--task` name absent from spec, unknown `depends_on` reference (`routing`), or missing task in delegated spec (`delegation`) |
| `RUN_ERROR` | `run` | LLM invocation raises an exception, or tool-call loop exceeds maximum iterations |
| `PROVIDER_ERROR` | `run` | Provider-specific error (network failure, API error, auth failure) |
| `CHAIN_CYCLE_ERROR` | `routing` | Circular `depends_on` chain detected |
| `CHAIN_INPUT_MISSING` | `input_validation` | Required input field missing after dependency merge |
| `CONTRACT_VIOLATION` | `contract` | Task output failed behavioural contract validation |
| `DELEGATION_CYCLE_ERROR` | `delegation` | Circular spec delegation detected (A→B→A) |

A runtime MUST detect and raise `CHAIN_CYCLE_ERROR` and `DELEGATION_CYCLE_ERROR` before any model call is made.

---

## 12. Conformance

### 12.1 Conformance Levels

A runtime conforms to OAS 1.4.0 if it:

1. **MUST** accept spec documents that validate against `spec/schema/oas-schema-1.4.json` and reject documents that do not.
2. **MUST** implement the four-level prompt resolution order defined in Section 9.1.
3. **MUST** implement template variable substitution as defined in Section 9.2.
4. **MUST** implement `depends_on` chain semantics as defined in Section 7.2, including cycle detection.
5. **MUST** implement the result envelope format defined in Section 8.2.
6. **MUST** implement all error codes in Section 11.2.
7. **MUST** implement spec delegation as defined in Section 7.3, including delegation cycle detection.
8. **MUST** support at minimum the `openai` and `anthropic` engines.
9. **MUST NOT** implement branching, loops, retries, parallel execution, or dynamic task selection as part of the `depends_on` or task execution model.

A runtime additionally claiming "tool support" conformance **MUST** implement the tool-call loop defined in Section 10.3.

A runtime additionally claiming "contract enforcement" conformance **MUST** implement the contract merge semantics defined in Section 10.4.

### 12.2 Versioning

The `open_agent_spec` field in a document declares the minimum spec version required. A runtime MUST reject documents whose version requirement it cannot satisfy.

The version string MUST conform to Semantic Versioning. Minor and patch increments MUST be backward compatible. Major increments MAY introduce breaking changes.

---

## Appendix A — Minimal Valid Document

```yaml
open_agent_spec: "1.4.0"

agent:
  name: "hello-world"
  description: "A minimal conforming agent"

intelligence:
  type: "llm"
  engine: "openai"
  model: "gpt-4o"

tasks:
  greet:
    description: "Say hello"
    input:
      type: object
      properties:
        name: { type: string }
      required: [name]
    output:
      type: object
      properties:
        response: { type: string }
      required: [response]
    prompts:
      system: "You greet people warmly."
      user: "{{ name }}"
```

---

## Appendix B — Chain Example

```yaml
open_agent_spec: "1.4.0"

agent:
  name: "extract-and-summarise"
  description: "Two-task chain demonstrating depends_on"

intelligence:
  type: "llm"
  engine: "openai"
  model: "gpt-4o"

tasks:
  extract:
    description: "Extract key facts"
    input:
      type: object
      properties:
        document: { type: string }
      required: [document]
    output:
      type: object
      properties:
        facts: { type: string }
      required: [facts]
    prompts:
      system: "Extract the three most important facts."
      user: "{{ document }}"

  summarise:
    description: "Summarise the extracted facts"
    depends_on: [extract]
    output:
      type: object
      properties:
        summary: { type: string }
      required: [summary]
    prompts:
      system: "Summarise in one sentence."
      user: "{{ facts }}"
```

Result envelope for `summarise`:

```json
{
  "task": "summarise",
  "output": { "summary": "Three facts condensed." },
  "input": { "document": "...", "facts": "fact1; fact2; fact3" },
  "prompt": "Summarise in one sentence.\n\nfact1; fact2; fact3",
  "engine": "openai",
  "model": "gpt-4o",
  "raw_output": "{\"summary\": \"Three facts condensed.\"}",
  "chain": {
    "extract": {
      "task": "extract",
      "output": { "facts": "fact1; fact2; fact3" }
    }
  }
}
```

---

## Appendix C — Error Examples

**Missing required input:**
```json
{
  "error": "Missing required input field(s) for task 'greet': name",
  "code": "CHAIN_INPUT_MISSING",
  "stage": "input_validation",
  "task": "greet"
}
```

**Contract violation:**
```json
{
  "error": "Missing required field: 'confidence'",
  "code": "CONTRACT_VIOLATION",
  "stage": "contract",
  "task": "summarise"
}
```

**Circular dependency:**
```json
{
  "error": "Circular dependency detected: 'extract' is already in the chain",
  "code": "CHAIN_CYCLE_ERROR",
  "stage": "routing",
  "task": "summarise"
}
```

**Delegation cycle:**
```json
{
  "error": "Circular spec delegation detected: './summariser.yaml' is already in the delegation stack",
  "code": "DELEGATION_CYCLE_ERROR",
  "stage": "delegation",
  "task": "summarise"
}
```
