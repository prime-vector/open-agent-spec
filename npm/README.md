# open-agent-spec

Native Node.js runner for [Open Agent Spec](https://openagentspec.dev) — no Python required.

Run agent specs from YAML, pull shared specialist specs from the registry, and chain tasks together, all from a single `oa run` command.

## Install

```bash
npm install -g @prime-vector/open-agent-spec
```

Or run without installing:

```bash
npx @prime-vector/open-agent-spec run --spec agent.yaml --task summarise --input input.json
```

## Quick start

**1. Write a spec**

```yaml
# agent.yaml
open_agent_spec: "1.4.0"

agent:
  name: my-agent
  description: Summarise text using a registry spec
  role: analyst

intelligence:
  type: llm
  engine: openai
  model: gpt-4o-mini

tasks:
  summarise:
    description: Delegate to the shared summariser
    spec: oa://prime-vector/summariser
    task: summarise
```

**2. Run it**

```bash
export OPENAI_API_KEY=sk-...

oa run --spec agent.yaml --task summarise --input '{"text": "Open Agent Spec is a YAML standard for declarative AI agents."}'
```

**3. Output**

```json
{
  "task": "summarise",
  "output": {
    "summary": "Open Agent Spec is a YAML standard for declarative AI agents.",
    "key_points": ["Declarative YAML format", "Engine-agnostic", "CLI-driven"]
  },
  "delegated_to": "https://openagentspec.dev/registry/prime-vector/summariser/latest/spec.yaml#summarise"
}
```

## Input formats

| `--input` value | Behaviour |
|---|---|
| `'{"key":"val"}'` | Inline JSON object |
| `input.json` | Parsed as a JSON object |
| `change.diff` | File contents → `{ text: "<contents>" }` |

## Providers

Set the relevant environment variable for your engine:

| Engine | Env var |
|---|---|
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `grok` / `x-ai` | `XAI_API_KEY` |

## Registry

Pull any spec from the OA registry using the `oa://` shorthand:

```yaml
spec: oa://prime-vector/summariser          # latest
spec: oa://prime-vector/summariser@1.0.0    # pinned
spec: https://example.com/my-spec.yaml      # third-party URL
```

Browse the registry at [openagentspec.dev/registry](https://openagentspec.dev/registry).

## Library usage

```typescript
import { runTask } from "open-agent-spec";

const result = await runTask({
  specPath: "./agents/summariser.yaml",
  taskName: "summarise",
  input: { text: "..." },
});

console.log(result.output);
```

## Requirements

- Node.js 18+
- No Python, no OpenAI SDK, no Anthropic SDK

## License

MIT
