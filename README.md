# Open Agent (OA)

Define AI agents with YAML. Generate working scaffolding instantly.

![PyPI version](https://img.shields.io/pypi/v/open-agent-spec)
![Python](https://img.shields.io/pypi/pyversions/open-agent-spec)
![License](https://img.shields.io/badge/license-MIT-blue)

**Open Agent (OA)** is a YAML specification for defining AI agents and generating working scaffolding.

Building AI agents today often requires manually wiring together:

- prompt templates  
- LLM configuration  
- task routing  
- memory structures  
- runtime logic  

Open Agent moves these concerns into a **declarative specification**.

Define an agent once in YAML and run it directly, or generate a project scaffold for customization.

You can think of OA as something similar to **OpenAPI for services** or **Terraform for infrastructure**, but for **AI agents**.

---

# Quick Start

Install the CLI:

```bash
pip install open-agent-spec
```

Set your LLM API key (example for OpenAI):

```bash
export OPENAI_API_KEY=your_api_key_here
```

Create an agent spec:

```yaml
agent:
  name: hello-world-agent
  role: chat

intelligence:
  engine: openai
  model: gpt-4

tasks:
  greet:
    description: Say hello to someone
    input:
      type: object
      properties:
        name:
          type: string
      required: [name]

    output:
      type: object
      properties:
        response:
          type: string
      required: [response]

prompts:
  system: >
    You greet people by name.
  user: "{{ name }}"
```

Run the agent directly from the spec:

```bash
oa run --spec agent.yaml --task greet --input '{"name":"Alice"}' --quiet
```

---

# Generate a Project Scaffold (Optional)

If you want to extend the implementation, generate a project scaffold:

```bash
oa init --spec agent.yaml --output ./agent
```

This produces a Python project you can customize.

---

# Generated Project Structure

```
agent/
├── agent.py
├── models.py
├── prompts/
├── requirements.txt
├── .env.example
└── README.md
```

---

# Design Philosophy

Open Agent intentionally keeps the specification **minimal**.

The goal is to define agents declaratively and generate consistent project scaffolding.

Tasks in an OA specification are intended to represent **atomic units of capability** for an agent, rather than complex workflows. Higher-level orchestration can be built on top of these primitives by external systems.

OA does **not prescribe**:

- runtime orchestration
- governance systems
- evaluation frameworks

These concerns can be layered on top by different runtimes, frameworks, or architectures.

---

# Why OA?

Many teams building agents end up recreating the same infrastructure:

- agent scaffolding
- prompt organization
- model configuration
- task definitions

OA provides a consistent way to **define agents once and generate a working structure automatically**.

---

# Related Work

Several projects are exploring ways to standardize how AI agents are defined and orchestrated.

Open Agent focuses specifically on **developer-facing scaffolding from a declarative YAML specification**.

The goal is to make agent architecture easier to reason about and quicker to implement.

---

## Commands

| Command | Purpose |
|--------|--------|
| `oa init --spec … --output …` | Generate project from YAML |
| `oa init --template minimal --output …` | Same with bundled spec |
| `oa init aac` | `.agents/` + example spec only |
| `oa run --spec … [--task …] [--input JSON] [--quiet]` | Run task without codegen |
| `oa update --spec … --output …` | Regenerate into existing dir |
| `oa init … --dry-run` | Validate only |

```bash
oa --help
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

## Historical Changes

“CLI command is oa (formerly oas in older releases).”

---

## License

MIT — see [LICENSE](LICENSE).

[Open Agent Stack](https://www.openagentstack.ai)
