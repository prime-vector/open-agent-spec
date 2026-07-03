# Skill Wrapper — Distributing an OA Spec Through the Skills Ecosystem

This example shows the **skill-wrapper pattern**: a `SKILL.md` (the markdown
"skills" format used by Claude Code, Cursor, and similar coding agents) whose
only job is to point the agent at a typed, validated OA spec.

```
skill-wrapper/
├── SKILL.md     # discovery + instructions: "run the spec, don't improvise"
├── spec.yaml    # the actual contract: typed I/O, pinned model, prompts
└── README.md    # this file
```

## The division of labour

| Layer | File | Responsibility |
|---|---|---|
| Guidance | `SKILL.md` | Discovery (name/description an agent matches on) and the instruction to execute via `oa run` |
| Contract | `spec.yaml` | Input/output schemas, pinned engine and model, explicit prompts, structured errors |

The skill is a thin distribution shim. All behaviour lives in the spec —
which means it is identical on every machine, validated on every run, and
certifiable by the [conformance suite](../../spec/conformance/README.md).
A skill wrapping an OA spec is strictly better than a skill wrapping a loose
script: the payload is typed, versioned, and portable.

## Try it directly (no agent required)

```bash
export OPENAI_API_KEY=sk-...
oa run --spec examples/skill-wrapper/spec.yaml --task summarise \
  --input '{"text": "OA is a declarative standard for defining AI agents as portable YAML contracts. It supports tools, sandboxing, spec composition and a public registry."}'
```

## Try it as a skill

Copy this directory into your agent's skills location (e.g.
`~/.claude/skills/document-summariser/` or your project's skills folder) and
ask the agent to summarise something. The agent discovers the skill via the
frontmatter description and executes the spec instead of improvising.

## The rule this pattern follows

> Markdown patterns may point at OA specs. OA specs never point at markdown
> patterns.

`spec.yaml` contains no reference to `SKILL.md` — delete the skill file and
the spec still works everywhere. See
[docs/proposals/markdown-interop.md](../../docs/proposals/markdown-interop.md)
for the full rationale.
