# Open Agent (OA)

**Open Agent (OA) is a YAML specification for defining AI agents and generating working scaffolding.**

Building AI agents today usually means manually wiring:

- prompt templates  
- LLM configuration  
- task routing  
- memory hooks  
- runtime entrypoints  

OA moves these into a **declarative spec**. Define the agent once in YAML; the CLI emits a **working project scaffold** you can install, run, and extend.

---

## Minimal spec

```yaml
agent:
  name: hello-world-agent
  role: chat

intelligence:
  engine: openai
  model: gpt-4

tasks:
  greet:
    description: Greet the user
```

Real specs add `open_agent_spec`, `input`/`output` schemas, and `endpoint` as needed. Full shape and engines: [docs/REFERENCE.md](https://github.com/prime-vector/open-agent-spec/blob/main/docs/REFERENCE.md).

---

## Install

```bash
pip install open-agent-spec
# or
pipx install open-agent-spec
```

Command: **`oas`**

---

## Generate an agent

```bash
oas init --spec agent.yaml --output ./agent
```

The CLI **generates a working scaffold from the specification**—Python module, prompts, dependencies, and env template—not a dead stub.

```bash
cd agent
cp .env.example .env
pip install -r requirements.txt
python agent.py
```

Bundled minimal spec:

```bash
oas init --template minimal --output ./agent
```

Run from YAML without generating a repo:

```bash
oas init aac
oas run --spec .agents/example.yaml --task greet --input '{"name": "Ada"}' --quiet
```

Update after changing the spec:

```bash
oas update --spec agent.yaml --output ./agent
```

---

## Generated project structure

What lands on disk today:

```
agent/
├── agent.py           # task functions + orchestration from spec
├── models.py          # output models when tasks define output schemas
├── prompts/           # jinja2 templates (per task + default)
├── requirements.txt
├── .env.example
└── README.md
```

The scaffold is a **starting point** for the agent you declared—consistent layout so you implement behavior instead of repeating boilerplate.

---

## Design philosophy

Open Agent keeps the specification **minimal**: agent definition + scaffolding. It does **not** prescribe runtime orchestration, governance, or evaluation. Those layers can sit on top; the spec stays agnostic so different frameworks can adopt the same YAML shape.

---

## Related work

Multiple efforts are exploring **agent specifications and interoperability**. Open Agent is focused on **developer scaffolding from a declarative YAML spec**: one file → one generated tree you can run and version. Neutral on how you orchestrate or govern agents afterward.

---

## Commands

| Command | Purpose |
|--------|--------|
| `oas init --spec … --output …` | Generate project from YAML |
| `oas init --template minimal --output …` | Same with bundled spec |
| `oas init aac` | `.agents/` + example spec only |
| `oas run --spec … [--task …] [--input JSON] [--quiet]` | Run task without codegen |
| `oas update --spec … --output …` | Regenerate into existing dir |
| `oas init … --dry-run` | Validate only |

```bash
oas --help
```

---

## More detail

| Resource | Contents |
|----------|----------|
| [docs/REFERENCE.md](https://github.com/prime-vector/open-agent-spec/blob/main/docs/REFERENCE.md) | Full spec, engines, templates |
| [Repository](https://github.com/prime-vector/open-agent-spec) | Source, issues, CI |

[![PyPI](https://img.shields.io/pypi/v/open-agent-spec)](https://pypi.org/project/open-agent-spec/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## License

MIT — see [LICENSE](LICENSE).

[Open Agent Stack](https://www.openagentstack.ai)
