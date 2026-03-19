# OA: The Open Agent Spec

**Declarative YAML for AI agents. Define once, run anywhere, generate real code.**

![PyPI version](https://img.shields.io/pypi/v/open-agent-spec)
![Python](https://img.shields.io/pypi/pyversions/open-agent-spec)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## 30-Second Quickstart

```bash
pip install open-agent-spec          # or: pipx install open-agent-spec
oa init aac                          # creates .agents/ with starter specs
export OPENAI_API_KEY=sk-...         # set your engine API key
oa run --spec .agents/example.yaml --task greet --input '{"name":"HN"}' --quiet
```

Output:

```json
{ "response": "Hello HN!" }
```

`--quiet` gives you **JSON only on stdout** — pipe it to `jq`, feed it to another tool, or use it in CI.

---

## What Is OA?

**OA** (Open Agent Spec) is a **spec-first CLI** for developers who want agent behavior in source control — not scattered across prompts, scripts, and framework glue.

You write a YAML spec that declares tasks, prompts, model config, and expected I/O shapes. Then:

- **`oa run`** executes the spec directly against any supported engine
- **`oa init`** generates a full Python scaffold (Pydantic models, Jinja2 prompts, requirements) when you need editable code
- **`oa validate`** checks your spec against the JSON Schema before anything runs

Founded in 2024, OA is the original community-driven standard for agent scaffolding and CLI-first development. While big players are now entering this space, OA remains lean, vendor-neutral, and developer-owned.

---

## Agents as Code

Store specs in a `.agents/` directory at the repo root — like `.github/workflows/` but for agents. Check them into version control, run them directly with `oa run`, or generate full project scaffolds from them.

```bash
oa init aac                          # scaffold .agents/ with starter specs
oa run --spec .agents/example.yaml --task greet --input '{"name":"CI"}' --quiet
```

This repo's own `.agents/` directory includes a [code-review agent](.agents/review.yaml) and a [CI failure repair agent](.agents/ci-failure-repair.yaml) that is called from a GitHub Actions workflow to auto-fix lint and formatting issues.

See [docs/REFERENCE.md](docs/REFERENCE.md#agents-as-code-agents) for details and bundled examples.

---

## Why OA?

| | The Usual Way | The OA Way |
|:---|:---|:---|
| **Structure** | Prompts and config hardcoded in Python | Schema-validated YAML — reviewable, diffable, versionable |
| **Portability** | Locked to one framework or provider | Engine-agnostic (`openai`, `anthropic`, `grok`, `codex`, `custom`) |
| **I/O contracts** | Hope the model returns the right shape | JSON Schema enforces input/output boundaries |
| **Code generation** | Start from scratch every time | `oa init` scaffolds Pydantic models, prompts, and wiring |
| **CI integration** | Glue scripts and manual wiring | `oa run --quiet` gives clean JSON; `.agents/` lives in your repo |

---

## First Run (Step by Step)

**1. Install** (Python 3.10+):

```bash
pip install open-agent-spec
```

<details><summary>Or use pipx for isolation</summary>

```bash
pipx install open-agent-spec
```

</details>

**2. Create the agents-as-code layout** (`aac` = repo-native `.agents/` directory):

```bash
oa init aac
```

This creates:

```text
.agents/
├── example.yaml   # minimal hello-world spec
├── review.yaml    # code-review agent that accepts a diff
└── README.md      # quick usage notes
```

**3. Validate:**

```bash
oa validate aac
```

**4. Set your API key** (OpenAI by default):

```bash
export OPENAI_API_KEY=your_key_here
```

**5. Run:**

```bash
oa run --spec .agents/example.yaml --task greet --input '{"name":"Alice"}' --quiet
```

**6. Try the review agent on a real diff:**

```bash
git diff > change.diff
oa run --spec .agents/review.yaml --task review --input change.diff --quiet
```

---

## Write Your Own Spec

```yaml
open_agent_spec: "1.2.5"

agent:
  name: hello-world-agent
  role: chat

intelligence:
  type: llm
  engine: openai
  model: gpt-4o  # or any model your account has access to

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

```bash
oa validate --spec agent.yaml          # schema check only (no model call)
oa run --spec agent.yaml --task greet \
  --input '{"name":"Alice"}' --quiet   # model call (requires OPENAI_API_KEY)
```

---

## Generate a Python Scaffold

When you need editable code instead of running the YAML directly:

```bash
oa init --spec agent.yaml --output ./agent
```

```text
agent/
├── agent.py           # main agent class
├── models.py          # Pydantic models from your I/O schema
├── prompts/           # Jinja2 templates
├── requirements.txt
├── .env.example
└── README.md
```

Or start from a bundled template:

```bash
oa init --template minimal --output ./agent
```

---

## Design Philosophy

OA is the **contract layer**, not the framework. It defines agent behavior declaratively so it can be reviewed, versioned, and reused — then gets out of the way.

OA deliberately does **not** prescribe orchestration, evaluation, governance, or long-running runtime architecture. External systems orchestrate multiple specs however they want. Tasks are **atomic units of capability**; higher-level workflows are built on top.

---

## Related Work

Several projects are exploring ways to standardize how AI agents are defined and orchestrated.

Open Agent Spec (OA) focuses specifically on **developer-facing scaffolding from a declarative YAML specification**.

The goal is to make agent architecture easier to reason about and quicker to implement.

---

## Commands

| Command | Purpose |
|:--------|:--------|
| `oa init aac` | Create `.agents/` with starter specs |
| `oa validate aac` | Validate all specs in `.agents/` |
| `oa validate --spec agent.yaml` | Validate one spec |
| `oa run --spec agent.yaml --task greet --input '...' --quiet` | Run a task; JSON output only |
| `oa init --spec agent.yaml --output ./agent` | Generate a Python scaffold |
| `oa init --template minimal --output ./agent` | Scaffold from bundled template |
| `oa update --spec agent.yaml --output ./agent` | Regenerate an existing scaffold |

---

## More Detail

| Resource | Contents |
|:---------|:---------|
| [docs/REFERENCE.md](https://github.com/prime-vector/open-agent-spec/blob/main/docs/REFERENCE.md) | Spec structure, engines, templates, `.agents/` usage |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup, PR process, adding templates |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

---

[![PyPI](https://img.shields.io/pypi/v/open-agent-spec)](https://pypi.org/project/open-agent-spec/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

- The CLI command is **`oa`** (not `oas`).
- Python **3.10+** required.
- `oa run` requires the relevant provider API key for the engine in your spec.

## License

MIT — see [LICENSE](LICENSE).

[Open Agent Stack](https://www.openagentstack.ai)
