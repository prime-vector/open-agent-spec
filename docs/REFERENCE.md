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
  engine: "openai"
  endpoint: "https://api.openai.com/v1"
  model: "gpt-4"
  config: { temperature: 0.7, max_tokens: 150 }
```

### Anthropic

```yaml
intelligence:
  engine: "anthropic"
  endpoint: "https://api.anthropic.com"
  model: "claude-3-sonnet-20240229"
  config: { temperature: 0.7, max_tokens: 150 }
```

### Grok (xAI, OpenAI-compatible client)

```yaml
intelligence:
  engine: "grok"
  endpoint: "https://api.x.ai/v1"
  model: "grok-3-latest"
```

### Custom router

```yaml
intelligence:
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
