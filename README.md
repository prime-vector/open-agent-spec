# Open Agent Spec CLI

**Define agents in YAML. Run them with one command—or generate a full Python project.**

[![PyPI](https://img.shields.io/pypi/v/open-agent-spec)](https://pypi.org/project/open-agent-spec/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Install

```bash
pip install open-agent-spec
# or (isolated CLI)
pipx install open-agent-spec
```

Command: **`oas`**

---

## Use it in 60 seconds

### 1. Run a spec (no code generation)

Spec lives in YAML; the CLI calls your configured model and prints JSON.

```bash
oas init aac
# Creates .agents/example.yaml — edit it, then:

export OPENAI_API_KEY=...   # or ANTHROPIC_API_KEY, etc.

oas run --spec .agents/example.yaml --task greet --input '{"name": "Ada"}' --quiet
```

### 2. Generate a full project

Scaffolds `agent.py`, prompts, `requirements.txt`, etc.

```bash
oas init --spec path/to/spec.yaml --output ./my-agent
cd my-agent
cp .env.example .env   # add API keys
pip install -r requirements.txt
python agent.py
```

Use a bundled template:

```bash
oas init --template minimal --output ./my-agent
```

### 3. Refresh generated code after spec changes

```bash
oas update --spec path/to/spec.yaml --output ./my-agent
```

---

## Commands

| Command | What it does |
|--------|----------------|
| `oas init aac` | Create `.agents/` with `example.yaml` only |
| `oas init --spec … --output …` | Generate full agent project |
| `oas init --template minimal --output …` | Same, using built-in minimal spec |
| `oas run --spec … [--task …] [--input '{"k":"v"}'] [--quiet]` | Run one task from YAML |
| `oas update --spec … --output …` | Regenerate into existing folder |
| `oas init … --dry-run` | Validate + show what would be written |

```bash
oas --help
oas run --help
```

---

## Spec at a glance

YAML describes the agent, model, and tasks. Minimal shape:

```yaml
open_agent_spec: "1.0.9"

agent:
  name: "hello-agent"
  description: "Says hello"

intelligence:
  engine: "openai"
  endpoint: "https://api.openai.com/v1"
  model: "gpt-4"

tasks:
  greet:
    description: "Greet by name"
    input:
      type: "object"
      properties:
        name: { type: "string" }
      required: ["name"]
    output:
      type: "object"
      properties:
        response: { type: "string" }
      required: ["response"]
```

**Engines:** `openai`, `anthropic`, `grok`, `cortex`, `local`, `custom` — full tables and examples in the repo: [docs/REFERENCE.md](https://github.com/prime-vector/open-agent-spec/blob/main/docs/REFERENCE.md).

---

## More detail

| Doc | Contents |
|-----|----------|
| [docs/REFERENCE.md](https://github.com/prime-vector/open-agent-spec/blob/main/docs/REFERENCE.md) | Full spec shape, engines, generated layout, templates |
| [Repo](https://github.com/prime-vector/open-agent-spec) | Source, issues, CI |

---

## License

MIT — see [LICENSE](LICENSE).

[Open Agent Stack](https://www.openagentstack.ai)
