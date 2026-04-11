# Open Agent Spec — reference

For a short intro, see the [README](../README.md). This page is the longer reference moved out of the main README so PyPI stays simple.

## Spec file structure

```yaml
open_agent_spec: "1.3.0"

agent:
  name: "hello-world-agent"
  description: "A simple agent"
  role: "chat"   # optional: analyst, reviewer, chat, retriever, planner, executor

intelligence:
  type: "llm"
  engine: "openai"   # openai | anthropic | grok | cortex | codex | local | custom
  endpoint: "https://api.openai.com/v1"
  model: "gpt-4"
  config:
    temperature: 0.7
    max_tokens: 150
  # module: "MyModule.MyClass"   # required for engine: custom

tasks:
  greet:
    description: "Say hello"
    input:
      type: "object"
      properties:
        name: { type: "string", description: "Name to greet" }
      required: ["name"]
    output:
      type: "object"
      properties:
        response: { type: "string" }
      required: ["response"]

# optional
behavioural_contract:
  pii: "..."
  compliance_tags: []
  allowed_tools: []
```

## Per-task prompts

Each task can define its own `prompts` block directly inside the task definition. This keeps system and user prompts co-located with the task they belong to, making multi-task agents much easier to reason about.

### Style A — per-task prompts (preferred)

```yaml
tasks:
  edit:
    description: Apply targeted edits to code
    prompts:
      system: |
        You are a precise code editor. Apply only the requested changes.
        Output a unified diff.
      user: "{{ instructions }}"
    input:
      type: object
      properties:
        instructions: { type: string }
      required: [instructions]
    output:
      type: object
      properties:
        diff: { type: string }
      required: [diff]

  ask:
    description: Answer a question about the codebase
    prompts:
      system: |
        You are a helpful code assistant. Be concise and accurate.
      user: "{{ question }}"
    input:
      type: object
      properties:
        question: { type: string }
      required: [question]
    output:
      type: object
      properties:
        answer: { type: string }
      required: [answer]

# Global fallback — used only when a task has no per-task prompts
prompts:
  system: "You are a general-purpose agent."
  user: "{{ input }}"
```

The `prompts` section at the top level becomes an **optional fallback** for any task that doesn't declare its own prompts.

### Resolution order

For each run, the system and user prompt are resolved in this priority order (highest wins):

| Priority | Source | How |
|---|---|---|
| 1 (highest) | CLI override | `--system-prompt` / `--user-prompt` flag |
| 2 | Per-task inline | `tasks.<name>.prompts.system` / `.user` |
| 3 | Per-task map (legacy) | `prompts.<name>.system` / `.user` |
| 4 (fallback) | Global | `prompts.system` / `prompts.user` |

Each prompt dimension (system, user) resolves independently — you can override just the system prompt and let the user template fall through from the task or global block.

### CLI prompt overrides

`oa run` accepts two runtime override flags:

```
--system-prompt TEXT    Replace the system prompt for this invocation
--user-prompt TEXT      Replace the user prompt template for this invocation
```

Supports the same `{{ field }}` placeholders as spec-defined user templates.

**Examples:**

```bash
# Run a task using its own per-task system prompt (no override needed)
oa run --spec .agents/code-assistant.yaml --task edit \
  --input '{"instructions":"Remove all debug print statements"}' --quiet

# Override the system prompt for a one-off targeted instruction
oa run --spec .agents/code-assistant.yaml --task ask \
  --input '{"question":"What does the auth module do?"}' \
  --system-prompt "You are a senior Go engineer. Be terse." --quiet

# Load a long system prompt from a file
oa run --spec .agents/code-assistant.yaml --task edit \
  --input '{"instructions":"..."}' \
  --system-prompt "$(cat .prompts/edit-system.txt)" --quiet
```

---

## response_format: text

By default every task expects the model to return valid JSON, which the runner parses and validates against the task's `output` schema. For tasks where the desired output is natural prose (explanations, summaries, diffs, etc.) you can opt out of JSON parsing entirely:

```yaml
tasks:
  explain:
    description: Explain what this function does
    response_format: text          # raw string output, no JSON parsing
    output:
      type: object
      properties:
        explanation: { type: string }
    prompts:
      system: |
        You are a helpful code explainer. Be concise.
      user: "{{ code }}"
    input:
      type: object
      properties:
        code: { type: string }
      required: [code]
```

When `response_format: text`:
- The model's raw output string is returned directly as `result["output"]`
- Output schema validation is **skipped** — the task author owns the contract
- Markdown fences and JSON-parsing are both bypassed
- Default is `"json"` (or omitting the field entirely)

---

## Structured errors

When `oa run --quiet` encounters a failure it emits a machine-readable JSON object to **stderr** (stdout stays clean for piping). Example:

```json
{"error": "Task 'explain' not found in spec", "code": "TASK_NOT_FOUND", "stage": "routing", "task": "explain"}
```

### Error codes

| `code` | `stage` | Trigger |
|---|---|---|
| `SPEC_LOAD_ERROR` | `load` | File not found, YAML parse error |
| `TASK_NOT_FOUND` | `routing` | `--task` name absent from spec, or unknown `depends_on` reference |
| `RUN_ERROR` | `run` | `invoke_intelligence` raises an exception |
| `CHAIN_CYCLE_ERROR` | `routing` | Circular `depends_on` chain detected |
| `CHAIN_INPUT_MISSING` | `input_validation` | Required input field missing after dependency merge |
| `CONTRACT_VIOLATION` | `contract` | Task output failed behavioural contract validation |

Verbose mode (`oa run` without `--quiet`) prints the error to the terminal as plain text, unchanged from prior behaviour.

### `oa test` (eval cases)

`oa test` loads a YAML file that points at a spec and lists **cases**. Each case runs one task (with the same `depends_on` resolution as `oa run`), then asserts on the parsed task **`output`** using `expect` rules. Use this for regression checks and CI gates; use `oa validate` for schema-only checks.

```bash
oa test path/to/agent.test.yaml
oa test path/to/agent.test.yaml --quiet   # single JSON summary on stdout
```

**Test file shape**

```yaml
spec: ./agent.yaml          # relative to this file’s directory
cases:
  - name: optional label
    task: greet             # optional; defaults like `oa run`
    input: { name: "CI" }
    expect:
      output.response: { contains: "hello" }
      output.items: { min_length: 1, type: array }
      output.items[0].id: { type: string }
```

Keys under `expect` must be `output` or start with `output.`; the remainder is a dotted path with optional `[index]` segments (e.g. `output.questions[0]`).

**Rules** (all rules under a path must pass):

| Rule | Meaning |
|---|---|
| `min_length` / `max_length` | For strings or lists, `len(value)` bound |
| `contains` | Substring (default case-insensitive; set `case_sensitive: true` otherwise) |
| `equals` | Exact equality |
| `type` | One of `string`, `number`, `boolean`, `object` / `dict`, `array` / `list` |

Omit `expect` or use `{}` for a **smoke** case that only checks the task completes without error.

---

## depends_on — linear task chaining

A task can declare that it needs the output of another task before it can run:

```yaml
tasks:
  extract:
    description: Extract key facts from a document
    output:
      type: object
      properties:
        facts: { type: string }
      required: [facts]
    prompts:
      system: "Extract the three most important facts."
      user: "{{ document }}"
    input:
      type: object
      properties:
        document: { type: string }
      required: [document]

  summarize:
    description: Summarize the extracted facts
    depends_on: [extract]          # runs extract first
    output:
      type: object
      properties:
        summary: { type: string }
    prompts:
      system: "Summarize the following facts in one sentence."
      user: "{{ facts }}"          # facts injected from extract's output
```

### Execution rules

1. **Dependencies run first**, in the order listed in `depends_on`
2. **Output is merged into input** — previous task output wins on key collision:
   ```
   merged = {**caller_input, **dep1_output, **dep2_output, ...}
   ```
3. **Fail fast** — required input fields are validated *after* the merge; missing fields raise `CHAIN_INPUT_MISSING` before the model is called
4. **Linear chains only** — no branching, no conditions, no loops
5. **Cycle detection** — circular references raise `CHAIN_CYCLE_ERROR` at run time

### Result envelope

The final result includes all intermediate results in a `chain` key:

```json
{
  "task": "summarize",
  "output": {"summary": "The sky is blue, water is wet, and ice is cold."},
  "chain": {
    "extract": {
      "task": "extract",
      "output": {"facts": "sky=blue; water=wet; ice=cold"}
    }
  }
}
```

Tasks with no `depends_on` do not include a `chain` key.

---

## Behavioural contracts (optional BCE integration)

Behavioural contracts let you declare constraints on a task's output — required fields, policy rules, behavioural flags — and have them enforced automatically at run time by the [`behavioural-contracts`](https://pypi.org/project/behavioural-contracts/) library.

Contracts are **entirely optional**. Specs without them run exactly as before. When the library is not installed, OA logs a hard warning and continues.

### Install

```bash
pip install 'open-agent-spec[contracts]'
```

### Per-task contract (preferred)

Declare a `behavioural_contract` block inside any task definition:

```yaml
tasks:
  summarize:
    description: Summarise extracted facts
    depends_on: [extract]
    behavioural_contract:
      version: "1.0"
      description: "Summarize task must always return a summary field"
      response_contract:
        output_format:
          required_fields: [summary]
    output:
      type: object
      properties:
        summary: { type: string }
```

### Global contract + per-task contract — merge semantics

A top-level `behavioural_contract` block acts as a baseline that applies to every task. Per-task contracts are **merged on top** — not replaced. Arrays (like `required_fields`) are **unioned**; scalars use per-task-wins.

```yaml
# Global: every task must output 'confidence'
behavioural_contract:
  version: "1.0"
  description: "Global baseline"
  response_contract:
    output_format:
      required_fields: [confidence]

tasks:
  summarize:
    behavioural_contract:
      version: "1.0"
      description: "Also requires summary"
      response_contract:
        output_format:
          required_fields: [summary]
    # Effective required_fields for this task: [confidence, summary]
```

This lets you enforce cross-cutting guarantees (e.g. every task must include `confidence`) in one place without repeating them per task.

### Contract resolution order

| Priority | Source |
|---|---|
| Base | Top-level `behavioural_contract` |
| Override (merged) | `tasks.<name>.behavioural_contract` |

### Where validation runs

Contracts are enforced **after output parsing, before the result is returned** — and for chain dependencies, before the dep output is merged into the next task's input:

```
extract → [parse] → [contract check] → merge into summarize input
summarize → [parse] → [contract check] → return result
```

A contract violation on a dependency stops the chain immediately and raises `CONTRACT_VIOLATION` before the dependent task ever runs.

### Skipped cases (with warning)

| Condition | Behaviour |
|---|---|
| `response_format: text` | Validation skipped — field checks are meaningless on raw strings |
| `behavioural-contracts` not installed | Hard warning logged; execution continues |
| Output is not a dict (JSON parse failed) | Validation skipped with warning |

### Error on violation

```json
{"error": "Missing required field: 'confidence'", "code": "CONTRACT_VIOLATION", "stage": "contract", "task": "summarize"}
```

---

## Engines (quick)

| Engine | Env var | Notes |
|--------|---------|-------|
| `openai` | `OPENAI_API_KEY` | Chat Completions or Responses API |
| `anthropic` | `ANTHROPIC_API_KEY` | Messages API |
| `grok` / `xai` | `XAI_API_KEY` | OpenAI-compatible; routes to `api.x.ai` |
| `cortex` | `OPENAI_API_KEY` | OpenAI-compatible; set `endpoint` in spec |
| `local` | _(none)_ | OpenAI-compatible local server (Ollama, LM Studio, …) |
| `codex` | Codex CLI on PATH | `codex login` required |
| `custom` | _(user-defined)_ | HTTP endpoint or Python class via `module:` |

All engines except `anthropic` and `codex` speak the OpenAI Chat Completions API. `oa run` uses raw HTTP — no SDK required.

### OpenAI

```yaml
intelligence:
  type: "llm"
  engine: "openai"
  endpoint: "https://api.openai.com/v1"
  model: "gpt-4o"
  config: { temperature: 0.7, max_tokens: 1000 }
```

### Anthropic

```yaml
intelligence:
  type: "llm"
  engine: "anthropic"
  endpoint: "https://api.anthropic.com"
  model: "claude-3-5-sonnet-20241022"
  config: { temperature: 0.7, max_tokens: 1000 }
```

### Grok / xAI (OpenAI-compatible)

`engine: grok` and `engine: xai` are aliases — both route to `https://api.x.ai/v1` with `XAI_API_KEY`. Endpoint and model can be overridden in the spec.

```yaml
intelligence:
  type: "llm"
  engine: "grok"          # or "xai"
  model: "grok-3-latest"  # default; override as needed
```

```bash
export XAI_API_KEY=xai-...
```

### Cortex (OpenAI-compatible, user-hosted)

`cortex` is treated as an OpenAI-compatible endpoint. Provide your `endpoint` in the spec; `OPENAI_API_KEY` is used by default but can be overridden via `config.api_key_env`.

```yaml
intelligence:
  type: "llm"
  engine: "cortex"
  endpoint: "https://cortex.mycompany.com/v1"
  model: "my-cortex-model"
  config:
    api_key_env: "CORTEX_API_KEY"
```

### Local LLM (Ollama, LM Studio, vLLM, …)

`engine: local` points to a local OpenAI-compatible server. No API key is required. The default endpoint is `http://localhost:11434/v1` (Ollama). Override `endpoint` and `model` as needed.

```yaml
intelligence:
  type: "llm"
  engine: "local"
  model: "llama3.2"     # default; match whatever model you have pulled
```

```bash
# Start Ollama (example)
ollama serve
ollama pull llama3.2
```

To use a different local server (e.g. LM Studio on port 1234):

```yaml
intelligence:
  type: "llm"
  engine: "local"
  endpoint: "http://localhost:1234/v1"
  model: "mistral-7b"
```

### Codex

Runs [Codex CLI](https://github.com/openai/codex) non-interactively via the built-in adapter (`oas_cli/adapters/codex_adapter.py`). Requires `codex` on `PATH` and `codex login`.

```yaml
intelligence:
  type: "llm"
  engine: "codex"
  model: "gpt-4.1-codex"
  config:
    sandbox: "workspace-write"   # codex sandbox mode
    cwd: "."                     # working directory for codex exec
```

`config` keys are passed as CLI flags to `codex exec`. Common options: `sandbox` (`workspace-write`, `workspace-read`, `none`) and `cwd`.

### Custom router

Two modes:

**HTTP mode** — no Python glue, just an OpenAI-compatible endpoint:

```yaml
intelligence:
  type: "llm"
  engine: "custom"
  endpoint: "https://my-llm-proxy.internal/v1"
  model: "my-model"
```

**Class mode** — point to a Python class for full control:

```yaml
intelligence:
  type: "llm"
  engine: "custom"
  endpoint: "http://localhost:1234/invoke"
  model: "my-model"
  module: "my_package.router.MyRouter"
```

The class must implement:

```python
class MyRouter:
    def __init__(self, endpoint: str, model: str, config: dict): ...
    def run(self, prompt: str, **kwargs) -> str: ...  # returns JSON string
```

Example:

```python
# my_package/router.py
import json, requests

class MyRouter:
    def __init__(self, endpoint, model, config):
        self.endpoint = endpoint
        self.model = model

    def run(self, prompt, **kwargs):
        resp = requests.post(self.endpoint, json={"prompt": prompt, "model": self.model})
        return resp.text  # must be a JSON string
```

## Agents as code (`.agents/`)

The **agent-as-code** pattern stores spec files in a `.agents/` directory at the root of your repository — similar to how `.github/workflows/` stores CI pipelines. Specs in `.agents/` are treated as infrastructure-as-code: check them into version control, run them directly with `oa run`, or generate full project scaffolds from them.

### Scaffold the layout

```bash
oa init aac                          # creates .agents/example.yaml, review.yaml, and README
oa init aac --directory ./my-repo    # target a different root
```

### Run an agent directly

```bash
oa run --spec .agents/example.yaml --task greet \
  --input '{"name":"Alice"}' --quiet
```

### Generate code from an agent spec

```bash
oa init --spec .agents/ci-failure-repair.yaml --output ./repair-agent
```

### Generated by `oa init aac`

`oa init aac` creates these files in your project (they are **not** shipped in the repo at install time):

| File | Role | Engine | Description |
|------|------|--------|-------------|
| `example.yaml` | chat | openai | Minimal hello-world spec — good starting point |
| `review.yaml` | reviewer | openai | Reviews a git diff and returns a decision plus summary |
| `README.md` | — | — | Quick usage notes for the `.agents/` directory |

### Shipped in this repository

This repository's own `.agents/` directory contains four specs used for development and CI:

| File | Role | Engine | Description |
|------|------|--------|-------------|
| `hello-world-agent.yaml` | chat | openai | Simple greeting — mirrors the generated `example.yaml` |
| `ci-failure-repair.yaml` | analyst | openai | Diagnoses GitHub Actions failures and emits remediation commands. Used by `.github/workflows/ci-failure-repair.yml`. |
| `codex-runner.yaml` | executor | codex | Runs Codex CLI non-interactively for arbitrary instructions |
| `review.yaml` | reviewer | openai | Reviews a git diff and returns a decision plus summary |

---

## Generated project layout

```
output/
├── agent.py
├── models.py            # if outputs are modelled
├── prompts/
│   ├── <task>.jinja2
│   └── agent_prompt.jinja2
├── requirements.txt
├── .env.example
└── README.md
```

## Bundled templates

YAMLs ship inside the package; from a clone you can do:

```bash
oa init --spec oas_cli/templates/minimal-multi-task-agent.yaml --output my-multi-agent/
oa init --spec oas_cli/templates/minimal-agent-tool-usage.yaml --output tool-agent/
```

Or point `--spec` at any file on disk.

## Development

```bash
git clone https://github.com/prime-vector/open-agent-spec.git
cd open-agent-spec
pip install -e ".[dev]"
pytest
```

Build: `python -m build`. Release: bump version in `pyproject.toml`, tag, push — CI publishes to PyPI.
