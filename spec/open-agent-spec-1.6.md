# Open Agent Spec — Formal Specification

**Version:** 1.6.0
**Status:** Release
**Date:** 2026-07-12

---

## Abstract

This document defines the Open Agent Spec (OA) 1.6.0. It specifies the structure of an OA document, the semantics that a conforming runtime MUST implement, and the boundaries of what OA deliberately does not do. An independent implementor MUST be able to build a conforming runtime from this document alone.

OA 1.6.0 consolidates the runtime definition around four pillars: a **typed contract** (schemas validated on both sides of every model call), a **deterministic execution pipeline** (no hidden control flow), **first-class cost observability** (normalised token usage and best-effort spend reporting on every result), and **declarative safety constraints** (sandboxing enforced before I/O). This revision formalises features that previous drafts left implementation-defined — sandboxing, history threading, input immutability — and promotes usage/cost reporting from an envelope footnote to a runtime obligation.

---

## 1. Introduction & Scope

Open Agent Spec is a YAML-based document format for declaring AI agent tasks. A spec document describes:

- the agent's identity and role
- the LLM engine and model to use
- one or more tasks, each with typed input/output schemas and prompts
- optional data dependencies between tasks
- optional tool declarations and sandbox constraints
- optional behavioural contracts

OA is **not** an execution framework, an orchestration engine, or an AI library. It is a document standard — a machine-readable contract between a spec author and a conforming runtime.

### 1.1 Design Goals

A conforming runtime exists to make agent behaviour **effective and cost-efficient to operate**, not merely possible. The requirements in this document serve four goals:

1. **Predictability** — every task has a typed input and output contract. A response that does not satisfy the contract is an error, not a silent downstream failure.
2. **Cost transparency** — every model interaction is metered. A runtime reports normalised token usage for every task (Section 10), sums it across multi-call loops, and attaches a best-effort dollar estimate that is never guessed and can be overridden with real negotiated rates. Spend is an output of every run, not something reconstructed later from provider dashboards.
3. **Right-sized inference** — the spec author can declare how much reasoning a task deserves (`reasoning_effort`, Section 5.5) and what it should cost (`pricing`, Section 10.3), portably across engines. Paying for `high` reasoning on a task that needs `low` is a spec bug the format makes visible.
4. **Bounded execution** — scope is hard-walled (Section 1.2), tool access can be sandboxed before any I/O occurs (Section 11), and pre-flight checks such as cycle detection are required to be cheap (Section 7.3) so validation cost never rivals inference cost.

### 1.2 What OA is NOT

OA explicitly prohibits — and a conforming runtime MUST NOT implement — the following:

| Feature | Classification | Rationale |
|---------|---------------|-----------|
| Conditional branching | Execution control | Belongs in the calling platform |
| Loops and retries | Runtime policy | Belongs in the calling platform |
| Parallel task execution | Scheduling | Belongs in the calling platform |
| Fallback task routing | Dynamic routing | Belongs in the calling platform |
| Dynamic task selection | Orchestration | Belongs in the calling platform |
| Conversation persistence | State management | Belongs in the calling platform (see Section 8.3) |

`depends_on` is a **data dependency declaration**, not a workflow instruction. It expresses what data a task needs, not how execution should proceed. See Section 7 for the full semantics.

### 1.3 Key Words

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://datatracker.ietf.org/doc/html/rfc2119).

### 1.4 Normative References

- [RFC 2119](https://datatracker.ietf.org/doc/html/rfc2119) — Key words for use in RFCs to indicate requirement levels
- [JSON Schema Draft-07](https://json-schema.org/specification-links.html#draft-7) — Schema language used for input/output validation
- `spec/schema/oas-schema-1.6.json` — Machine-readable JSON Schema for OA documents (canonical, normative)
- `spec/conformance/PROTOCOL.md` — Conformance adapter protocol (normative for runtimes claiming certified conformance, Section 14.2)

---

## 2. Terminology

**Spec document** — A YAML file conforming to this specification.

**Runtime** — Any software that loads a spec document and executes tasks according to the semantics defined herein.

**Task** — A named unit of work defined in a spec. A task has typed inputs, typed outputs, prompts, and an LLM engine configuration.

**Chain** — A sequence of tasks linked by `depends_on` declarations, where earlier task outputs are merged into later task inputs.

**Delegation** — A task that forwards its execution to a task in a different spec document (via `spec:` + `task:`).

**Result envelope** — The structured object a runtime returns after executing a task.

**Usage block** — The portion of the result envelope reporting normalised token consumption and best-effort cost for the task's model call(s).

**Behavioural contract** — An optional set of constraints on a task's output, enforced after the LLM response is parsed.

**Sandbox** — An optional set of hard execution constraints (tool, domain, and path restrictions) enforced by the runtime before any tool call reaches the I/O layer.

**History** — A caller-supplied array of prior conversation turns, threaded into the model's message list by the runtime but never stored by it.

**Provider** — The implementation that communicates with a specific LLM engine (e.g. OpenAI, Anthropic).

**Template variable** — A `{{ key }}` placeholder in a prompt string that is substituted with the corresponding input value at run time.

---

## 3. Document Model

### 3.1 Format

An OA document MUST be a valid YAML file that deserializes to a JSON-compatible mapping (object). The document MUST validate against the JSON Schema at `spec/schema/oas-schema-1.6.json`.

### 3.2 Top-Level Keys

| Key | Required | Type | Description |
|-----|----------|------|-------------|
| `open_agent_spec` | REQUIRED | string | Spec version. MUST match pattern `^(1\.(0\.[4-9]|[1-9]\.[0-9]+)|[2-9]\.[0-9]+\.[0-9]+)$` |
| `agent` | REQUIRED | object | Agent identity. See Section 4. |
| `intelligence` | REQUIRED | object | LLM engine configuration. See Section 5. |
| `tasks` | REQUIRED | object | Task definitions. See Section 6. |
| `prompts` | OPTIONAL | object | Global prompt fallbacks. See Section 8. |
| `behavioural_contract` | OPTIONAL | object | Global behavioural contract baseline. See Section 12.4. |
| `tools` | OPTIONAL | object | Tool declarations. See Section 12. |
| `sandbox` | OPTIONAL | object | Root-level execution constraints. See Section 11. |
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
    reasoning_effort: low    # portable reasoning tier — see 5.5
    pricing:                 # cost override — see 10.3
      input_per_1m: 2.00
      output_per_1m: 8.00
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

When a spec omits `endpoint`, a runtime MUST apply the selected engine's own default endpoint. A runtime MUST NOT let one engine's default endpoint leak into another engine's configuration.

### 5.3 Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `endpoint` | string (HTTP/HTTPS URL) | Overrides the default API endpoint |
| `config.temperature` | number [0, 2] | Sampling temperature |
| `config.max_tokens` | integer ≥ 1 | Maximum tokens to generate |
| `config.top_p` | number [0, 1] | Nucleus sampling parameter |
| `config.frequency_penalty` | number [-2, 2] | Frequency penalty |
| `config.presence_penalty` | number [-2, 2] | Presence penalty |
| `config.reasoning_effort` | string `low` \| `medium` \| `high` | Portable reasoning-effort tier (see 5.5) |
| `config.pricing` | `"none"` \| object | Per-spec cost-estimation override (see 10.3) |

A runtime MUST pass `temperature` and `max_tokens` to the LLM API when provided, except where the selected model rejects a parameter — in that case the runtime MUST omit the rejected parameter rather than fail the call. Unknown `config` keys MAY be forwarded to the underlying engine or ignored.

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

### 5.5 Reasoning Effort

`config.reasoning_effort` declares a portable reasoning-depth tier independent of any single engine. Its value MUST be one of `low`, `medium`, or `high`. A runtime MUST reject any other value.

When `reasoning_effort` is set, a runtime SHOULD map it to the selected engine's native reasoning control (for example, an effort/verbosity parameter, an extended-thinking setting, or a CLI flag). A runtime MAY treat it as a no-op for engines and models that expose no reasoning control. Because reasoning-capable models often reject incompatible request fields, a runtime MAY adjust other request parameters as required by the engine when `reasoning_effort` is set (for example, substituting a token-limit field or omitting sampling parameters).

`reasoning_effort` is only meaningful for reasoning-capable models. It is the spec author's responsibility to pair it with a suitable engine and model.

`reasoning_effort` is a cost-efficiency control as much as a quality control: reasoning tokens are billed output tokens, and the tier is the declarative way to stop a cheap task from thinking like an expensive one. Combined with the usage block (Section 10), the effect of a tier change on both quality and spend is directly observable per run.

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
    sandbox: ...                 # OPTIONAL — per-task sandbox override (Section 11)
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
| `sandbox` | OPTIONAL | object | Per-task sandbox override (Section 11) |
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

### 6.4 Reserved Input Field: `history`

The input field name `history` is reserved by this specification for conversation threading. Its semantics are defined in Section 8.3. Spec authors MUST NOT use `history` for unrelated data.

---

## 7. Runtime Execution Model

This section defines the runtime: the deterministic pipeline every task passes through, the immutability guarantees at task boundaries, and the pre-flight checks that MUST complete before any model call spends tokens.

### 7.1 Execution Pipeline

Given a task with no `depends_on`, a runtime MUST execute the following stages in order:

1. **Load & validate** — parse the document, validate against the canonical schema. Failures raise `SPEC_LOAD_ERROR`.
2. **Route** — resolve the requested task name. Unknown names raise `TASK_NOT_FOUND`.
3. **Validate input** — check required input fields. Missing fields raise `CHAIN_INPUT_MISSING`.
4. **Resolve prompts** — apply the four-level precedence order (Section 8.1) and substitute template variables (Section 8.2).
5. **Build the engine request** — from `intelligence:`, including reasoning-effort mapping (Section 5.5) and any per-model parameter adjustments.
6. **Invoke** — if `tools:` is declared for the task, enter the tool-call loop (Section 12.3) with sandbox checks before every dispatch (Section 11); otherwise call the LLM once. `history` input, when present, is threaded into the message list (Section 8.3).
7. **Meter** — capture the engine's reported token usage for every call made in stage 6 and accumulate it (Section 10).
8. **Parse & validate output** — apply `response_format` processing and output schema validation (Section 9.1, 6.3).
9. **Enforce contracts** — apply the behavioural contract if declared (Section 12.4).
10. **Return the result envelope** (Section 9.2), including the accumulated usage block.

Stages 1–5 make no model calls. All structural errors a runtime can detect statically (schema violations, unknown tasks, missing inputs, dependency cycles, delegation cycles, invalid pricing configuration) MUST be raised before any tokens are spent.

### 7.2 Input Immutability

Every task boundary MUST receive a deep copy of its input. A runtime MUST guarantee that:

- mutations made while executing one task can never leak into the input of another task in the same chain, and
- the caller's original input object is never mutated by executing a spec, regardless of chain depth or delegation.

This guarantee is what makes `depends_on` a pure data contract: the only way data moves between tasks is the explicit output-merge rule in Section 7.3.

### 7.3 Dependent Tasks (`depends_on`)

`depends_on` is a **data contract**, not an execution directive. When a task declares `depends_on: [dep1, dep2, ...]`:

1. Cycle detection: if any dependency (transitively) references the calling task, raise `CHAIN_CYCLE_ERROR` before any LLM call.
2. For each dependency in listed order:
   a. Execute the dependency task recursively (applying the same semantics).
   b. Merge its `output` into the running input map.
3. Merge rule: `merged = {**caller_input, **dep1_output, **dep2_output, ...}` — later entries win on key collision.
4. Validate required input fields against the merged map. Raise `CHAIN_INPUT_MISSING` if any are missing.
5. Execute the calling task with the merged input.

A runtime MUST NOT execute dependent tasks in parallel. A runtime MUST NOT execute a task until all its declared dependencies have produced output. A runtime MUST detect circular dependency chains and raise `CHAIN_CYCLE_ERROR` before invoking any model.

**Shared dependencies (diamond graphs).** Two tasks MAY depend on the same upstream task. A diamond-shaped dependency graph is legal and MUST NOT be reported as a cycle. Only the direct dependencies of the invoked task are executed; `depends_on` is not transitive execution.

**Cycle-detection efficiency.** Cycle detection MUST scale linearly with the size of the dependency graph (nodes + edges). In particular, an implementation MUST NOT re-explore a shared subgraph once per path reaching it — on lattice-shaped graphs a path-enumerating check is exponential and turns a pre-flight validation into the dominant cost of the run. The conformance suite holds runtimes to this with a deep-lattice case (~2^40 distinct paths) that MUST complete promptly. This is a design-goal requirement (Section 1.1, goal 4): validation MUST stay cheap relative to inference.

**The following are permanently out of scope for `depends_on`:**
- Conditional execution (if/else)
- Loop control (while/for)
- Retry semantics
- Parallel fan-out
- Dynamic task selection based on output values

### 7.4 Spec Delegation (`spec:`)

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
6. Execute the target task in the referenced spec, passing the current `input_data` (as a deep copy, per Section 7.2).
7. Surface the result under the coordinator's task name.
8. Include `delegated_to: "<spec_path>#<task_name>"` in the result envelope.

When a task is delegated, the calling task's inline `prompts:` and `output:` schema are **ignored** — the referenced spec's task definition governs.

---

## 8. Prompt & Message Construction

### 8.1 Four-Level Precedence

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

### 8.2 Template Variable Substitution

The user prompt template MAY contain `{{ key }}` placeholders. After prompt resolution, a runtime MUST substitute each placeholder with the corresponding value from the task's input data.

**Supported substitution syntaxes** (a runtime MUST support both):

| Syntax | Example | Notes |
|--------|---------|-------|
| `{{ key }}` | `{{ name }}` | Jinja-style, with spaces |
| `{{ input.key }}` | `{{ input.name }}` | Jinja-style with `input.` prefix |
| `{key}` | `{name}` | Python-style single braces |

Substitution is applied to all three syntaxes. Values are stringified before substitution. A runtime MUST NOT fail if a placeholder references a key not present in the input — unresolved placeholders MAY be left as-is or substituted with an empty string (implementation choice, but SHOULD be documented).

When no user prompt is configured at any level, a runtime SHOULD default to `"{{ input }}"`.

### 8.3 History Threading

The reserved input field `history` (Section 6.4) enables stateless multi-turn conversations. When a task's input contains a `history` array, each element is an object with `role` (`"user"` or `"assistant"`) and `content` (string) keys.

A runtime that supports history threading MUST:

1. Inject the history turns into the model's message list **between the system prompt and the current user turn**, preserving their order:
   `[system, history..., user]`.
2. Forward `history` without transformation — the runtime does not summarise, truncate, or re-rank it.
3. Never persist history. OA has no conversation store; the caller owns and manages the turn list (Section 1.2).

For providers without a message-list API, a runtime SHOULD render history into the prompt text in a documented, deterministic way.

History support is a declared conformance capability (`history`, Section 14.2). A runtime that does not support it MUST NOT silently drop a supplied `history` field on chat-style tasks; it SHOULD reject or warn.

---

## 9. Response Format & Result Envelope

### 9.1 Response Format

`response_format` controls how the model's raw text output is processed:

| Value | Behaviour |
|-------|-----------|
| `"json"` (default) | Strip markdown code fences if present, then parse as JSON. Validate against output schema. |
| `"text"` | Return raw string as-is. Skip JSON parsing and output schema validation. |

**Markdown fence stripping:** If the model wraps its response in ` ```json ... ``` ` or ` ``` ... ``` `, a runtime MUST strip the fences before attempting JSON parsing.

**Parse failure:** If JSON parsing fails and `response_format` is `"json"`, a runtime SHOULD surface the raw string rather than raising a hard error, to preserve the response for inspection.

### 9.2 Result Envelope

A runtime MUST return the following envelope for every successfully executed task:

```json
{
  "task": "<task_name>",
  "output": <parsed_output>,
  "input": <input_data_used>,
  "prompt": "<system_prompt>\n\n<user_prompt>",
  "engine": "<engine_name>",
  "model": "<model_name>",
  "raw_output": "<model_raw_text>",
  "usage": { ... }
}
```

The `usage` block is defined in Section 10.

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

## 10. Usage & Cost Observability

Cost efficiency is a stated design goal of this specification (Section 1.1). The result envelope is the metering point: every task execution reports what it consumed, so spend can be attributed per task, per chain, and per spec without external tooling.

### 10.1 The `usage` Block

A runtime SHOULD include a `usage` key in the result envelope reporting the token consumption of the task's model call(s):

```json
{
  "usage": {
    "prompt_tokens": <integer>,
    "completion_tokens": <integer>,
    "total_tokens": <integer>,
    "estimated_cost_usd": <number>
  }
}
```

Requirements:

- `prompt_tokens`, `completion_tokens`, and `total_tokens` are normalised token counts. A runtime that reports `usage` MUST populate these from the engine's reported usage, mapping provider-specific shapes (for example OpenAI's `prompt_tokens`/`completion_tokens` and Anthropic's `input_tokens`/`output_tokens`) to these keys.
- `usage` MUST be `null` (or omitted) when the engine does not report token counts (some local servers, CLI-backed engines, custom routers). A runtime MUST NOT fabricate counts.

### 10.2 Multi-Call Summation

When a task makes multiple model calls — most commonly a multi-turn tool-calling loop (Section 12.3) — the reported counts MUST be the sum across every call in the loop. Because each turn resends the growing message history, summed `prompt_tokens` correctly reflects total billed input; a runtime MUST NOT report only the final call's usage.

A runtime SHOULD also surface accumulated usage when a run *fails* partway through a multi-call loop (for example on hitting the iteration limit), so that spent tokens are visible even for unsuccessful runs.

### 10.3 Cost Estimation & the Pricing Override

`estimated_cost_usd` is OPTIONAL and best-effort. When present it is a pay-as-you-go list-price estimate (`tokens × per-token rate`); it MUST be omitted when no rate is known for the model. A runtime MUST NOT guess a rate. It does not reflect subscription, committed-use, negotiated, or local-inference pricing — the token counts are the authoritative figure to meter against any plan.

**Cost override (`config.pricing`).** A runtime that estimates cost SHOULD let the spec override the rate via `config.pricing`, whose value is either:

- the string `"none"` — disable cost estimation for this spec (report tokens only), or
- an object `{ "input_per_1m": <number ≥ 0>, "output_per_1m": <number ≥ 0> }` — explicit USD rates per 1,000,000 tokens.

A runtime MAY additionally support an implementation-defined global rate override (the reference implementation uses an `OA_PRICING` environment variable). Resolution order, first match wins: per-spec `config.pricing` → global override → built-in list-price table. This lets `estimated_cost_usd` reflect real spend (enterprise-negotiated rates) or be suppressed entirely (subscription and local models), organisation-wide or per spec.

**Fail-closed rule.** A `config.pricing` value (or global override) that is present but invalid — a negative rate, a missing rate field, malformed JSON, or a string other than `"none"` — MUST fail closed: the runtime MUST raise `PRICING_CONFIG_ERROR` rather than silently falling back to a default rate. A silently wrong dollar figure is worse than no figure. An absent `config.pricing`, or a model for which no rate is known, is not an error — the runtime simply omits `estimated_cost_usd`.

---

## 11. Sandbox — Immutable Inference Sandboxing

The `sandbox:` block declares hard execution constraints that a runtime enforces **before any tool call reaches the I/O layer**. A blocked call never opens a network connection, never creates a file handle, and is not observable as a caught exception inside the tool — it is refused at dispatch.

```yaml
sandbox:                             # root level — applies to all tasks
  tools:
    allow: [file.read, http.get]     # only these tools may be called
    deny: [env.read]                 # these tools may never be called
  http:
    allow_domains: [api.example.com] # http.get/http.post restricted to these hosts
  file:
    allow_paths: [./data/]           # file.read/file.write restricted to these prefixes

tasks:
  restricted:
    sandbox:                         # per-task — overrides root per constraint key
      tools:
        allow: [file.read]
```

### 11.1 Declaration & Resolution

`sandbox` MAY be declared at the root level (applies to every task) and/or per task. The effective sandbox for a task is resolved per top-level constraint key (`tools`, `http`, `file`): a task-level key **completely replaces** the root-level key of the same name. Task-level sandbox is an override, not an extension — this keeps the effective policy for any task readable from at most two places in the document.

### 11.2 Constraint Types

| Key | Applies to | Semantics |
|-----|-----------|-----------|
| `tools.allow` | every tool dispatch | If present, a tool not in the list MUST be refused (`SANDBOX_TOOL_VIOLATION`) |
| `tools.deny` | every tool dispatch | If present, a tool in the list MUST be refused (`SANDBOX_TOOL_VIOLATION`). Deny is checked in addition to allow. |
| `http.allow_domains` | `http.get`, `http.post` | The request URL's hostname MUST equal a listed domain or be a subdomain of one (`host == d` or `host` ends with `.d`); otherwise `SANDBOX_DOMAIN_VIOLATION` |
| `file.allow_paths` | `file.read`, `file.write` | The resolved absolute path MUST fall under one of the listed path prefixes (after resolving symlinks and `..`); otherwise `SANDBOX_PATH_VIOLATION` |

An absent constraint key imposes no restriction of that type. An empty `allow` list denies everything of that type.

### 11.3 Enforcement Rules

A runtime that supports sandboxing MUST:

1. Check the effective sandbox before **every** tool dispatch in the tool-call loop, including repeat calls to a previously allowed tool.
2. Enforce path constraints against the **resolved** path (symlinks and relative segments resolved), not the literal argument.
3. Surface violations as structured errors with stage `sandbox` and the specific code for the constraint type (Section 13.2) — never as a generic run failure.
4. Continue to enforce the sandbox regardless of what the model requests; sandbox constraints are not visible to or negotiable by the model.

**Honesty rule.** A runtime that does not implement sandboxing MUST refuse to run a spec that declares a `sandbox:` block, rather than silently ignoring it. Silent degradation of a declared security constraint is itself a conformance violation (see `spec/conformance/PROTOCOL.md`).

---

## 12. Extension Points

### 12.1 Tools

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

### 12.2 Tool Types

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

### 12.3 Tool-Call Loop

When a task declares tools, the runtime MUST implement a multi-turn tool-call loop:

1. Send the system prompt, user prompt, and tool definitions to the LLM.
2. If the model returns tool calls, check each call against the effective sandbox (Section 11), execute allowed calls, and feed results back as tool result messages.
3. Repeat until the model returns a final text response or a maximum iteration limit is reached.
4. A runtime MUST enforce a maximum iteration limit (the reference implementation uses 10). Exceeding the limit MUST raise `RUN_ERROR`, and the error SHOULD carry the usage accumulated up to that point (Section 10.2).

Token usage MUST be accumulated across every turn of the loop (Section 10.2).

When the provider does not support native tool calling, a runtime SHOULD inject tool descriptions into the system prompt and fall back to a single LLM call.

### 12.4 Behavioural Contracts

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

Note the asymmetry with sandbox resolution (Section 11.1): contracts **merge** (global is a baseline the task extends), sandboxes **override per key** (the task replaces the root constraint). Contracts accumulate obligations; sandboxes state a single effective policy.

**Contract enforcement is skipped when:**
- `response_format: "text"` (field checks are meaningless on raw strings)
- The output could not be parsed as a dict (warning logged, execution continues)
- The behavioural contracts library is not installed (warning logged, execution continues)

A contract violation MUST raise `CONTRACT_VIOLATION`.

---

## 13. Error Model

### 13.1 Structured Errors

A runtime MUST surface errors as structured objects with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `error` | string | Human-readable error message |
| `code` | string | Machine-readable error code (see Section 13.2) |
| `stage` | string | Pipeline stage where the error occurred |
| `task` | string | Task name involved (OPTIONAL — omit when unknown) |

### 13.2 Error Codes

| Code | Stage | Trigger |
|------|-------|---------|
| `SPEC_LOAD_ERROR` | `load` | File not found, YAML parse error, spec does not validate against schema |
| `TASK_NOT_FOUND` | `routing` or `delegation` | `--task` name absent from spec, unknown `depends_on` reference (`routing`), or missing task in delegated spec (`delegation`) |
| `RUN_ERROR` | `run` | LLM invocation raises an exception, or tool-call loop exceeds maximum iterations |
| `OUTPUT_SCHEMA_ERROR` | `output_validation` | Parsed output does not validate against the task's output schema (JSON mode only) |
| `PROVIDER_ERROR` | `run` | Provider-specific error (network failure, API error, auth failure) |
| `CHAIN_CYCLE_ERROR` | `routing` | Circular `depends_on` chain detected |
| `CHAIN_INPUT_MISSING` | `input_validation` | Required input field missing after dependency merge |
| `CONTRACT_VIOLATION` | `contract` | Task output failed behavioural contract validation |
| `DELEGATION_CYCLE_ERROR` | `delegation` | Circular spec delegation detected (A→B→A) |
| `PRICING_CONFIG_ERROR` | `cost` | A cost-rate override (`config.pricing` or an implementation-defined global override) is present but invalid |
| `SANDBOX_TOOL_VIOLATION` | `sandbox` | Tool blocked by the effective `tools.allow`/`tools.deny` sandbox constraint |
| `SANDBOX_DOMAIN_VIOLATION` | `sandbox` | HTTP request host not permitted by `http.allow_domains` |
| `SANDBOX_PATH_VIOLATION` | `sandbox` | File path outside `file.allow_paths` after resolution |

A runtime MUST detect and raise `CHAIN_CYCLE_ERROR`, `DELEGATION_CYCLE_ERROR`, and `PRICING_CONFIG_ERROR` before any model call is made.

---

## 14. Conformance

### 14.1 Conformance Requirements

A runtime conforms to OA 1.6.0 if it:

1. **MUST** accept spec documents that validate against `spec/schema/oas-schema-1.6.json` and reject documents that do not.
2. **MUST** implement the execution pipeline defined in Section 7.1, raising all statically detectable errors before any model call.
3. **MUST** implement input immutability as defined in Section 7.2.
4. **MUST** implement the four-level prompt resolution order defined in Section 8.1 and template substitution as defined in Section 8.2.
5. **MUST** implement `depends_on` chain semantics as defined in Section 7.3, including linear-time cycle detection and diamond-graph support.
6. **MUST** implement the result envelope format defined in Section 9.2.
7. **MUST** implement all error codes in Section 13.2 for the capabilities it claims.
8. **MUST** implement spec delegation as defined in Section 7.4, including delegation cycle detection.
9. **MUST** support at minimum the `openai` and `anthropic` engines, each with its correct default endpoint.
10. **SHOULD** implement usage reporting as defined in Section 10; a runtime that reports usage MUST follow the normalisation, summation, no-guessed-rates, and fail-closed pricing rules.
11. **MUST NOT** implement branching, loops, retries, parallel execution, dynamic task selection, or conversation persistence as part of the task execution model.

Capability-scoped requirements:

- A runtime claiming **`tools`** conformance MUST implement the tool-call loop defined in Section 12.3.
- A runtime claiming **`sandbox`** conformance MUST implement Section 11 in full; a runtime not claiming it MUST refuse specs that declare `sandbox:` (honesty rule, Section 11.3).
- A runtime claiming **`contracts`** conformance MUST implement the contract merge semantics defined in Section 12.4.
- A runtime claiming **`history`** conformance MUST implement history threading as defined in Section 8.3.

### 14.2 Certified Conformance & the Adapter Protocol

Conformance is testable, runtime-agnostic, and mock-driven — certification never calls a real model and costs nothing to run. The conformance suite (`spec/conformance/`) drives any runtime through a thin executable adapter speaking the protocol defined in `spec/conformance/PROTOCOL.md`: the harness feeds one case (spec + invocation + canned model responses) on stdin and asserts on the result envelope or structured error from stdout.

Adapters declare the capabilities they support (`core`, `depends-on`, `delegation`, `registry`, `response-format-text`, `output-schema-validation`, `history`, `tools`, `sandbox`, `contracts`); the harness skips — and reports as UNSUPPORTED, distinct from PASS/FAIL — cases requiring undeclared capabilities. A runtime MUST NOT declare a capability it does not enforce.

The reference distribution ships adapters for the Python CLI and the npm CLI, and its CI runs the full matrix against both.

### 14.3 Versioning

The `open_agent_spec` field in a document declares the minimum spec version required. A runtime MUST reject documents whose version requirement it cannot satisfy.

The version string MUST conform to Semantic Versioning. Minor and patch increments MUST be backward compatible. Major increments MAY introduce breaking changes.

OA 1.6.0 is additive over 1.5.x: every valid 1.5.x document is a valid 1.6.0 document.

---

## Appendix A — Minimal Valid Document

```yaml
open_agent_spec: "1.6.0"

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

## Appendix B — Chain Example with Usage

```yaml
open_agent_spec: "1.6.0"

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
  "usage": {
    "prompt_tokens": 41,
    "completion_tokens": 12,
    "total_tokens": 53,
    "estimated_cost_usd": 0.000223
  },
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

**Sandbox violation:**
```json
{
  "error": "Sandbox violation: domain 'internal.corp' is not in allow_domains for task 'fetch'. Allowed: ['api.example.com']",
  "code": "SANDBOX_DOMAIN_VIOLATION",
  "stage": "sandbox",
  "task": "fetch"
}
```

**Invalid pricing override:**
```json
{
  "error": "Invalid config.pricing: 'input_per_1m' must be a number >= 0",
  "code": "PRICING_CONFIG_ERROR",
  "stage": "cost",
  "task": "summarise"
}
```
