# Proposal: Interop with AGENTS.md and Skills (Markdown Agent Patterns)

**Status:** Accepted
**Scope:** Documentation and examples only — zero schema or runtime changes
**Author:** Prime Vector / Open Agent Spec maintainers

## Background

Since OA was designed, two markdown-based patterns have gained significant
mindshare in agentic engineering:

- **AGENTS.md** — a prose file at a repository root giving interactive coding
  agents (Cursor, Claude Code, Codex, etc.) ambient instructions: conventions,
  build commands, style rules.
- **Skills** (SKILL.md folders) — prose instructions with name/description
  frontmatter, optionally bundled with loose scripts, progressively disclosed
  to an agent when the description matches the task at hand.

Stripped of hype, both are the same thing: **natural-language instructions
that a model may or may not follow**. Neither has a schema. Neither is
validated, versioned, mechanically enforceable, or conformance-testable.
When a skill "breaks", you find out by watching an agent do the wrong thing.

## The question

Should OA engage with these patterns, and if so, how — without inheriting
their weaknesses?

Two real risks argue against ignoring them entirely:

1. **Discoverability collision.** People searching "agent spec" increasingly
   land on AGENTS.md content. Some will assume Open Agent Spec is a competing
   "agent file" format. Silence lets others define the relationship for us.
2. **Distribution channel.** Skills are becoming how people *share agent
   capabilities*. Ignoring that channel means OA specs lose a growing on-ramp.

## Position

OA and the markdown patterns are not competitors. They operate at different
layers:

| | AGENTS.md / Skills | Open Agent Spec |
|---|---|---|
| Layer | Guidance (how an interactive agent behaves) | Execution contract (what an agent task *is*) |
| Format | Prose | Declarative YAML, JSON-Schema validated |
| Enforcement | Model goodwill | Runner-enforced (sandbox, schemas, structured errors) |
| Portability | Per-tool conventions | Conformance-tested across runtimes |
| Failure mode | Silent drift | Structured error codes |

The industry has discovered that agents need shared, distributable
definitions — and reached for the weakest possible format first. OA's answer
is not to join the trend or to buck it, but to be the thing the markdown
eventually points at.

## The rule (bright line)

> **Markdown patterns may point at OA specs. OA specs never point at
> markdown patterns.**

All interop happens one layer *above* the spec. Nothing enters the schema.
This is consistent with every prior boundary decision: no `skill:` task type
(rejected), no orchestration, no prose-as-contract.

## What this proposal delivers

Three artifacts, all documentation/examples:

1. **`examples/skill-wrapper/`** — a SKILL.md that wraps an OA spec. The
   skill provides discovery (name + description an agent can match on) and
   instructs the agent to execute via `oa run` rather than improvising. The
   OA spec provides the actual contract: typed inputs/outputs, pinned model,
   validated output. This turns the skills ecosystem into a distribution
   channel for OA specs — a skill wrapping a spec is strictly better than a
   skill wrapping a loose script.

2. **Root `AGENTS.md`** — dogfoods the coexistence pattern in this repo:
   tells coding agents that agent behaviour here is defined as OA specs,
   points them at `oa run` / `oa validate` / the conformance harness, and
   instructs them not to hand-roll agent logic. Doubles as genuinely useful
   contributor guidance.

3. **README positioning section** — "How OA relates to AGENTS.md and
   Skills". Defines the relationship publicly before someone else does.

## What we explicitly refuse

To keep the line bright, the following are out of scope permanently unless
revisited by a future proposal:

- ❌ A `skill:` task type or any skill reference inside a spec
- ❌ Markdown-file prompt references inside specs (kills single-file
  portability)
- ❌ Any schema-level claim of AGENTS.md/skills "compatibility"
- ❌ Runtime awareness of SKILL.md or AGENTS.md files

The moment prose instructions become load-bearing inside a spec, OA inherits
every problem it is currently immune to.

## Future (optional, demand-driven)

If the skills channel proves sticky, an `oa skill package` CLI command could
generate a SKILL.md wrapper from a spec's existing metadata (agent name,
description, task inputs). This is tooling *on top of* the boundary, not a
schema change — and should not be built until someone asks for it.

## How we manage it

- These artifacts ship as docs/examples in a normal PR; no version bump
  required beyond a changelog note.
- The bright-line rule above gets cited in future feature discussions the
  same way "no orchestration" and "no skills task type" are today.
- If the AGENTS.md/skills patterns evolve into something schema'd and
  enforceable, we reassess via a new proposal — from a position of interop
  rather than dependence.
