# Westpac engagement — re-kick planning

**Status:** Draft for discussion
**Date:** 2026-06-26
**Owners:** Andrew (Westpac side), Scott (Prime Vector)
**Context:** Andrew sees an opportunity to use Open Agent Spec inside Westpac and is
meeting the chief engineer next week. This note frames the two work streams we
want to walk into that conversation with, grounded in what OA already does today,
so we can agree an approach and kick the project off again.

---

## Why this fits Westpac

A large bank running coding agents at scale hits the same two problems we've
discussed before, just bigger:

1. **Sprawl with no control.** Lots of teams, lots of agent/"skill" files, lots of
   ad-hoc bash command groups, all disparate and unreviewed. Nobody can see what
   agents exist, what they're allowed to do, or whether they conform to a standard.
   A bank needs *visibility and control* over that estate before it can trust it.
2. **Token cost.** LLM spend is now a board-level line item. The same task run
   thousands of times a day on a frontier model is money on fire when a cheaper
   tier — or a local model — would have done.

OA is well placed for (1) because it already treats an agent as a reviewable,
versioned, validated *contract* rather than a scattered prompt. It is well placed
for (2) because the engine/config boundary is already declarative — we just don't
yet expose the levers that matter for cost.

The honest gap: today OA **deliberately** does not prescribe governance, orchestration,
or a long-running runtime (see README "Core Idea"). The Westpac opportunity is to
build a thin **control plane** *on top of* the spec — without turning OA itself into
a framework. Both work items below respect that boundary.

---

## Work Item 1 — Enterprise governance & visibility ("control plane")

**Problem (Westpac words):** disparate skill files and bash groups everywhere, no
single place to see or govern them.

**What we already have to build on:**

| Building block | Where | What it gives us |
|---|---|---|
| Spec validation | `oa validate` (`oas_cli/validators.py`) | Conformance gate for every agent definition |
| Spec registry | `oa://` resolution (`oas_cli/runner.py`) | Central, versioned catalogue of shared agents |
| IIS sandboxing | `sandbox:` block (spec §10, `examples/sandboxed-agent/`) | Hard allow-lists for tools / domains / file paths |
| Behavioural contracts | `behavioural_contract:` (spec §10.4) | Enforced output guarantees |
| Conformance levels | spec §12 | A defensible "this agent meets the bar" claim |

**Proposed approach (for discussion):**

- **Inventory / discovery.** An `oa scan <repo-or-org>` command that walks a tree,
  finds agent definitions, "skill" files and bash command groups, and produces a
  single machine-readable inventory: what exists, which engine/model each uses,
  which tools each can call, which conform to OA and which don't.
- **Policy gate.** Let an org declare a baseline policy (allowed engines, mandatory
  sandbox allow-lists, required behavioural contracts) and have `oa validate`
  enforce it in CI — extending the existing validator rather than inventing a new one.
- **Visibility surface.** A read-only report/dashboard view over the inventory so a
  chief engineer can answer "what agents do we run, what can they touch, who owns
  them" at a glance. (The `Website/` registry views are a starting point.)

**Deliberately out of scope:** we are *not* building orchestration, scheduling, or a
runtime daemon into OA. The control plane consumes specs; it doesn't change the
execution model.

**Open questions for the meeting:**

- Where do Westpac's agent/skill/bash files actually live today (repos, a registry,
  developer laptops)? That decides what `oa scan` has to target first.
- Self-hosted registry behind the bank's network vs. our hosted `openagentspec.dev`?
- Is the immediate win *audit/visibility* (read-only) or *enforcement* (block
  non-conforming agents in CI)? Suggest leading with visibility — lower risk, faster value.

---

## Work Item 2 — Cost optimisation: reasoning-effort tiers + cheap/local routing

**Problem (Westpac words):** token cost is becoming a primary focus; we want to use
cheap/local queries instead of expensive frontier calls where the use case allows,
and exploit the new low/medium/high reasoning tiers (Opus 4.x, Codex) to dial spend
to the difficulty of each task.

**Current state in the codebase:**

- `intelligence.config` only carries sampling params — `temperature`, `max_tokens`,
  `top_p`, `frequency_penalty`, `presence_penalty` (spec §5.3). There is **no**
  reasoning-effort concept anywhere in the spec or providers.
- The `local` engine already exists (Ollama / LM Studio / vLLM / llama.cpp) — the
  cheap-routing building block is in place (`providers/registry.py`).
- Providers **discard the API `usage` object** — e.g. `anthropic_http.py` returns
  only `data["content"][0]["text"]`. So today we have **zero token/cost visibility**,
  which means we can't even measure a saving, let alone prove one to a bank.

**Proposed approach (for discussion):**

1. **Add a reasoning-effort tier to the spec.** A declarative
   `intelligence.config.reasoning_effort: low | medium | high` (settable per task),
   mapped to each provider's native control:
   - Anthropic Opus 4.x → reasoning/effort parameter
   - OpenAI / Codex → the equivalent tier flag (`codex_adapter.py` already passes
     `extra_args`, so the plumbing is partly there)
   - Engines with no native tier → documented no-op / nearest sampling approximation
2. **Token & cost observability first.** Capture the `usage` object from each provider
   response and surface tokens (and an estimated cost) in the result envelope and in
   `oa run` output. *This is the prerequisite for everything else* — you can't optimise
   what you can't measure, and it's the cheapest, most demoable first deliverable.
3. **Cheap / local routing for suitable tasks.** Make it easy to declare that a task
   (e.g. classification, extraction, routing) runs on a local or small model, reserving
   frontier models + high reasoning for the hard tasks. Lean on existing per-task
   `intelligence` rather than inventing dynamic routing (which OA prohibits by design).

**Suggested sequencing:** (2) observability → (1) effort tiers → (3) routing guidance.
Observability is small, self-contained, and immediately shows Westpac a number.

**Open questions for the meeting:**

- Which Westpac use cases are genuinely "cheap-tier safe" (classify/extract/route) vs.
  must stay on a frontier model? That list drives the routing story.
- Do they want an estimated *dollar* cost in the envelope (needs a price table we'd
  maintain and keep current) or just raw token counts (provider-neutral, lower
  maintenance)? Suggest token counts first, dollars as an opt-in layer.
- Per-task effort override vs. a single agent-level default — how granular do they need it?

---

## Suggested first steps once we reconvene

1. Confirm which work item leads. Recommendation: **WI-2 token observability** as the
   fast, measurable proof point, in parallel with **WI-1 `oa scan` inventory** as the
   visibility hook for the chief-engineer conversation.
2. Turn the agreed scope into tracked GitHub issues (one per deliverable above).
3. Spike `reasoning_effort` against Opus 4.x + Codex to confirm the provider mappings
   before committing to spec wording (it would land as a 1.6 spec addition).

> These are framing notes, not commitments — the point is to give next week's
> discussion a concrete starting shape we can argue with.
