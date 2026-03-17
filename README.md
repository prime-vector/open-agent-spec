# Open Agent Spec (OA)

Define an AI agent once in YAML, then either run it directly with `oa run` or generate a Python scaffold with `oa init`.

![PyPI version](https://img.shields.io/pypi/v/open-agent-spec)
![Python](https://img.shields.io/pypi/pyversions/open-agent-spec)
![License](https://img.shields.io/badge/license-MIT-blue)

Open Agent Spec is a spec-first CLI for developers who want agent behavior to live in source control instead of being spread across prompts, scripts, and framework glue.

Think:
- OpenAPI, but for agent capabilities
- Terraform-style declarative files, but for repo-native agents
- A clean boundary between agent definition and runtime implementation

With OA you can:
- define tasks, prompts, model config, and expected I/O in YAML
- run a spec directly without generating code first
- keep `.agents/*.yaml` in your repo and call them from CI
- generate a Python project scaffold when you want to customize implementation

## Quick Start

Recommended install:

```bash
pipx install open-agent-spec
```

Alternative installs:

```bash
pip install open-agent-spec
```

```bash
brew tap prime-vector/homebrew-prime-vector
brew install open-agent-spec
```

Check the CLI:

```bash
oa --version
oa --help
```

## Fastest First Run

This is the shortest path to a successful local setup.

1. Create the repo-native `.agents/` layout:

```bash
oa init aac
```

2. Validate the generated specs:

```bash
oa validate aac
```

3. Set an API key for the engine you want to use. Example for OpenAI:

```bash
export OPENAI_API_KEY=your_api_key_here
```

4. Run the example agent:

```bash
oa run --spec .agents/example.yaml --task greet --input '{"name":"Alice"}' --quiet
```

Expected `--quiet` output is the task output JSON only, which makes it easy to pipe into `jq` or use in scripts:

```json
{
  "response": "Hello Alice!"
}
```

If you want the full execution envelope for debugging, omit `--quiet`.

## What `oa init aac` Gives You

`oa init aac` creates a repo-native `.agents/` directory for “agents as code”:

```text
.agents/
├── example.yaml
├── review.yaml
└── README.md
```

- `example.yaml`: a minimal hello-world spec
- `review.yaml`: a code-review agent that accepts a diff file
- `README.md`: quick usage notes for the generated folder

Try the review agent on a local diff:

```bash
git diff > change.diff
oa run --spec .agents/review.yaml --task review --input change.diff --quiet
```

## Write Your Own Spec

When you want to create a spec from scratch, start from this shape:

```yaml
open_agent_spec: "1.2.6"

agent:
  name: hello-world-agent
  role: chat

intelligence:
  type: llm
  engine: openai
  model: gpt-4o

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

Validate first, then run:

```bash
oa validate --spec agent.yaml
oa run --spec agent.yaml --task greet --input '{"name":"Alice"}' --quiet
```

## Generate a Python Scaffold

If you want editable generated code instead of running the YAML directly:

```bash
oa init --spec agent.yaml --output ./agent
```

Generated structure:

```text
agent/
├── agent.py
├── models.py
├── prompts/
├── requirements.txt
├── .env.example
└── README.md
```

You can also start from the bundled template:

```bash
oa init --template minimal --output ./agent
```

## Core Idea

Most agent projects end up hand-rolling the same pieces:
- prompt templates
- model configuration
- task definitions
- routing glue
- runtime wrappers

OA moves those concerns into a declarative spec so they can be reviewed, versioned, and reused.

The intended model is:
- spec defines the agent contract
- `oa run` executes the spec directly
- `oa init` generates a starting implementation when you need code
- external systems can orchestrate multiple specs however they want

OA deliberately does not prescribe:
- orchestration
- evaluation
- governance
- long-running runtime architecture

## Common Commands

| Command | Purpose |
|--------|--------|
| `oa init aac` | Create `.agents/` with starter specs |
| `oa validate aac` | Validate all specs in `.agents/` |
| `oa validate --spec agent.yaml` | Validate one spec |
| `oa run --spec agent.yaml --task greet --input '{"name":"Alice"}' --quiet` | Run one task directly from YAML |
| `oa init --spec agent.yaml --output ./agent` | Generate a Python scaffold |
| `oa init --template minimal --output ./agent` | Generate from bundled template |
| `oa update --spec agent.yaml --output ./agent` | Regenerate an existing scaffold |

## More Detail

| Resource | Contents |
|----------|----------|
| [docs/REFERENCE.md](https://github.com/prime-vector/open-agent-spec/blob/main/docs/REFERENCE.md) | Spec structure, engines, templates, `.agents/` usage |
| [Repository](https://github.com/prime-vector/open-agent-spec) | Source, issues, workflows |

## Notes

- The CLI command is `oa` and not `oas`.
- Python 3.10+ is required.
- `oa run` requires the relevant provider auth for the engine in your spec.

## License

MIT — see [LICENSE](LICENSE).

[Open Agent Stack](https://www.openagentstack.ai)
