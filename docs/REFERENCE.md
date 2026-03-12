# Open Agent Spec — reference

For a short intro, see the [README](../README.md). This page is the longer reference moved out of the main README so PyPI stays simple.

## Spec file structure

```yaml
open_agent_spec: "1.0.9"

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

## Engines (quick)

| Engine | Env var (typical) |
|--------|-------------------|
| openai | `OPENAI_API_KEY` |
| anthropic | `ANTHROPIC_API_KEY` |
| grok | `XAI_API_KEY` |
| cortex | `OPENAI_API_KEY`, `CLAUDE_API_KEY` + `cortex-intelligence` |
| codex | Codex CLI (`codex` on PATH; `codex login`) |
| local | placeholder |
| custom | Your router class: `__init__(endpoint, model, config)`, `run(prompt, **kwargs)` → JSON string |

### OpenAI

```yaml
intelligence:
  type: "llm"
  engine: "openai"
  endpoint: "https://api.openai.com/v1"
  model: "gpt-4"
  config: { temperature: 0.7, max_tokens: 150 }
```

### Anthropic

```yaml
intelligence:
  type: "llm"
  engine: "anthropic"
  endpoint: "https://api.anthropic.com"
  model: "claude-3-sonnet-20240229"
  config: { temperature: 0.7, max_tokens: 150 }
```

### Grok (xAI, OpenAI-compatible client)

```yaml
intelligence:
  type: "llm"
  engine: "grok"
  endpoint: "https://api.x.ai/v1"
  model: "grok-3-latest"
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

```yaml
intelligence:
  type: "llm"
  engine: "custom"
  endpoint: "http://localhost:1234/invoke"
  model: "my-model"
  module: "CustomLLMRouter.CustomLLMRouter"
```

```python
# CustomLLMRouter.py
import json

class CustomLLMRouter:
    def __init__(self, endpoint: str, model: str, config: dict):
        self.endpoint = endpoint
        self.model = model
        self.config = config

    def run(self, prompt: str, **kwargs) -> str:
        return json.dumps({"response": f"…"})
```

## Agents as code (`.agents/`)

The **agent-as-code** pattern stores spec files in a `.agents/` directory at the root of your repository — similar to how `.github/workflows/` stores CI pipelines. Specs in `.agents/` are treated as infrastructure-as-code: check them into version control, run them directly with `oa run`, or generate full project scaffolds from them.

### Scaffold the layout

```bash
oa init aac                          # creates .agents/example.yaml + README
oa init aac --directory ./my-repo    # target a different root
```

### Run an agent directly

```bash
oa run --spec .agents/hello-world-agent.yaml --task greet \
  --input '{"name":"Alice"}' --quiet
```

### Generate code from an agent spec

```bash
oa init --spec .agents/ci-failure-repair.yaml --output ./repair-agent
```

### Bundled examples

This repository ships three `.agents/` specs as working examples:

| File | Role | Engine | Description |
|------|------|--------|-------------|
| `hello-world-agent.yaml` | chat | openai | Simple greeting — good starting point |
| `ci-failure-repair.yaml` | analyst | openai | Diagnoses GitHub Actions failures and emits remediation commands. Used by `.github/workflows/ci-failure-repair.yml` in this repo. |
| `codex-runner.yaml` | executor | codex | Runs Codex CLI non-interactively for arbitrary instructions |

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

CLI shortcut (single built-in spec):

```bash
oa init --template minimal --output my-agent/
```

Other YAMLs ship inside the package; from a clone you can do:

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
