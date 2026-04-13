# OA Extension Points

Open Agent Spec is deliberately minimal — it defines a document model and execution semantics for single tasks and linear chains. Everything beyond that boundary is an **extension**.

This document describes the three built-in extension mechanisms and how they interact with the core spec.

## Extension Types

### 1. Tools (Spec §10)

Tools extend a task with the ability to call external services or execute code during the LLM interaction loop.

```yaml
tools:
  search:
    type: mcp
    endpoint: "https://search.internal/mcp"
    description: "Search the web"
  read_file:
    type: native
    native: file.read
    description: "Read a file"
  custom_router:
    type: custom
    module: "my_package.tools.Router"
    description: "Custom routing"

tasks:
  analyse:
    description: "Analyse with tool access"
    tools: [search, read_file]
    # ...
```

**Three tool backends:**

| Type | Description | Required fields |
|------|-------------|-----------------|
| `native` | Built-in zero-dependency tools (`file.read`, `file.write`, `http.get`, `http.post`, `env.read`) | `native` (tool ID) |
| `mcp` | JSON-RPC 2.0 over HTTP to any [MCP](https://modelcontextprotocol.io/) server | `endpoint` |
| `custom` | Dynamically loaded Python class implementing `ToolProvider` | `module` |

**Conformance:** A runtime claiming "tool support" MUST implement the multi-turn tool-call loop (§10.3): send tool definitions to the model, execute tool calls, feed results back, repeat until final response or iteration limit.

**Out of scope:** Tool discovery, tool registries, tool authentication beyond header injection, and tool-to-tool communication are not part of OA.

---

### 2. Behavioural Contracts (Spec §10.4)

Behavioural contracts declare constraints on task output that are enforced at runtime — after the LLM responds but before the result is returned.

```yaml
behavioural_contract:
  version: "1.0"
  response_contract:
    output_format:
      required_fields: [confidence, reasoning]
    content_policy:
      forbidden_patterns: ["TODO", "FIXME"]

tasks:
  analyse:
    behavioural_contract:
      response_contract:
        output_format:
          required_fields: [analysis]
    # effective required_fields: [confidence, reasoning, analysis]
```

**Merge semantics:**
- Arrays are unioned (order preserved, duplicates removed)
- Objects/dicts are merged recursively
- Scalars: per-task value wins over global

**Conformance:** A runtime claiming "contract enforcement" MUST implement the merge rules above. Contract violations MUST raise `CONTRACT_VIOLATION`.

**Integration:** The reference implementation uses the [`behavioural-contracts`](https://pypi.org/project/behavioural-contracts/) library. Install with:

```bash
pip install 'open-agent-spec[contracts]'
```

When the library is not installed, contract validation is skipped with a warning — the runtime degrades gracefully.

---

### 3. Custom Engines (Spec §5.4)

The `custom` engine type allows any LLM backend to participate in OA execution.

```yaml
intelligence:
  type: llm
  engine: custom
  endpoint: "https://my-proxy.internal/v1"
  model: "my-model"
  module: "my_package.router.MyRouter"
```

A custom engine class MUST implement:

```python
class MyRouter:
    def __init__(self, endpoint: str, model: str, config: dict): ...
    def run(self, prompt: str, **kwargs) -> str: ...  # returns JSON string
```

**Use cases:** Private model proxies, evaluation harnesses, cost routers, model ensembles, local fine-tuned models behind a unified interface.

---

## Designing New Extensions

OA intentionally does NOT have a plugin registry or extension manifest. Extensions are declared inline in the spec document using the mechanisms above.

If you need capability beyond tools, contracts, and custom engines, the intended pattern is:

1. **Build it outside OA** — in the calling platform, orchestrator, or wrapper
2. **Feed results in via task input** — OA tasks accept arbitrary input fields
3. **Consume results from the envelope** — the result envelope exposes all execution data

This keeps the spec boundary clean: OA defines what happens inside a single task execution. Everything around it — orchestration, evaluation, governance, persistence — belongs to the platform.

---

## Future Directions

The `interface` top-level key is reserved for future use. Potential candidates for formalisation:

- Structured tool result schemas
- Cross-spec type sharing
- Runtime capability negotiation

These will be considered for inclusion in OA 2.x based on implementation experience and community feedback.
