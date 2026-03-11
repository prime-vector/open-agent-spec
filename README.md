# Open Agent (OA)

![PyPI version](https://img.shields.io/pypi/v/open-agent-spec)
![Python](https://img.shields.io/pypi/pyversions/open-agent-spec)
![License](https://img.shields.io/badge/license-AGPL-blue)

**Open Agent (OA)** is a YAML specification for defining AI agents and generating working scaffolding.

Building AI agents today often requires manually wiring together:

- prompt templates  
- LLM configuration  
- task routing  
- memory structures  
- runtime logic  

Open Agent moves these concerns into a **declarative specification**.

Define an agent once in YAML, and the CLI generates a working project scaffold.

You can think of OA as something similar to **OpenAPI for services** or **Terraform for infrastructure**, but for **AI agents**.

The goal is to eliminate repetitive agent boilerplate so developers can focus on implementing agent behavior.

---

# Example Agent Spec

A minimal Open Agent specification looks like this:

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

From this specification, OA generates a working Python agent scaffold.

---

# Installation

Install the CLI from PyPI:

```bash
pip install open-agent-spec
```

---

# Generate an Agent

Create an agent project from a YAML specification:

```bash
oas init --spec agent.yaml --output ./agent
```

This command generates a project scaffold based on the specification.

---

# Run an Agent

You can run an agent directly from the CLI:

```bash
oas run --spec .agents/example.yaml --task greet
```

---

# Generated Project Structure

The CLI generates a clean project structure to start implementing the agent:

```
agent/
├── agent.py
├── models.py
├── prompts/
├── requirements.txt
├── .env.example
└── README.md
```

This scaffold provides the foundation for implementing the agent defined in the specification.

---

# Design Philosophy

Open Agent intentionally keeps the specification **minimal**.

The focus is on **defining agents declaratively and generating consistent project scaffolding**.

OA does **not prescribe**:

- runtime orchestration
- governance systems
- evaluation frameworks

These concerns can be layered on top by different frameworks, runtimes, or architectural patterns.

---

# Why OA?

Many teams building agents end up recreating the same infrastructure:

- agent scaffolding
- prompt organization
- model configuration
- task definitions

OA provides a consistent way to **define agents once and generate the project structure automatically**.

---

# Related Work

Several projects are exploring ways to standardize how AI agents are defined and orchestrated.

Open Agent focuses specifically on **developer-facing scaffolding from a declarative YAML specification**.

The goal is to make agent architecture easier to reason about and quicker to implement.

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
