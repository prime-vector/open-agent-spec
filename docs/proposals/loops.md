# Proposal: Loops and Iteration in Open Agent Spec

**Status:** Proposed
**Scope:** Design decision — determines what, if anything, enters the schema. This document is docs-only; any schema change is a separate follow-up PR.
**Author:** Prime Vector / Open Agent Spec maintainers

## Background

"Agentic loops" are having a moment. The pattern shows up in three unrelated
guises that the word *loop* papers over:

- **Tool-calling loops** — a model calls tools, reads results, and calls again
  until it produces a final answer (the ReAct / function-calling loop).
- **Iterate-until-good** — run a task, judge the output, re-run with feedback
  until some quality bar is met.
- **Map over a collection** — run the same task once per item in a list and
  gather the results.

The pressure is to "add loops to OA". But OA **already runs a loop**: the
tool-calling loop in the reference runtime is bounded at
`_MAX_TOOL_ITERATIONS` and raises a structured error when the budget is
exhausted rather than spinning forever. So the question is not *whether* OA
loops — it already does, in exactly one disciplined way. The question is which
of the other shapes, if any, belong inside the spec.

## The question

Should OA grow loop constructs, and if so which — without inheriting the
non-determinism and control flow its boundaries exist to keep out?

OA's boundaries are deliberate and load-bearing: no orchestration, no
conditionals or branching in specs, no prose-as-contract. A spec is a *typed
contract*, statically analysable before a single token is spent (§7 of the
1.6 spec makes every statically detectable error a pre-execution failure).
`depends_on` is an acyclic DAG on purpose — the cycle-detection work exists
precisely to guarantee no task ever loops back on itself. Any loops proposal
has to answer to all of that.

## Position

Not all loops are the same shape, and they do not land the same way against
those boundaries. Sorted by what OA can see *before* the first token:

| Loop | What it is | Termination | Verdict |
|---|---|---|---|
| **A. Tool-calling loop** | Model calls tools until done | Bounded iteration cap, structured error | ✅ Already in — name it, don't reinvent it |
| **B. Bounded map / fan-out** | Run a task once per item in an input array | Fixed by the array length + a ceiling | 🟡 Compatible if bounded — worth a design |
| **C. Conditional / until-good** | Loop until a runtime predicate says stop | Depends on a value only knowable at runtime | 🔴 Out — belongs above the spec |

The dividing line is **determinism of termination**. A map over a known array
terminates at a length OA can read up front; the graph stays acyclic and the
run is reproducible. An until-good loop terminates when a model or a judge
*decides* it is done — control flow driven by a runtime value, with no bound
OA can validate before it starts. The first is data; the second is
orchestration.

## The rule (bright line)

> **OA iterates over data it can see before the first token. It never loops
> on a condition it can only evaluate at runtime.**

Bounded, data-driven iteration is OA's to own. Predicate-driven,
runtime-terminated iteration lives one layer up, in the orchestrator that
*calls* OA specs — exactly as the markdown-interop proposal put guidance above
the contract rather than inside it. This is consistent with every prior
boundary decision: no orchestration, no conditionals, no `skill:` task type.

## What this proposal recommends

**A — Accept, and surface it (small, near-term).** Promote the hardcoded
tool-iteration cap into a declared, schema-validated field (working name
`intelligence.config.max_iterations`), defaulting to today's value. This adds
no new behaviour — it names the loop OA already runs, gives users a validated
knob, and lets the existing usage/cost accounting report against a declared
budget. Fully in-boundary: declarative, bounded, runner-enforced.

**B — Accept in principle, design deliberately (the real work).** A bounded
map — run task `T` once per element of an input array, collect the outputs in
order — is the one genuinely-additive loop that can stay inside the line. It
introduces no cycle (the mapped task is still a single DAG node), and it is
reproducible given its input. The loop itself is trivial; the semantics are
not, and this is where the design lives:

- **Result shape** — outputs collected into an array, preserving input order.
- **Failure semantics** — fail-closed by default (one element errors → the
  task errors, consistent with OA's fail-closed posture), with any
  collect-partial-results mode as an explicit opt-in, not the default.
- **A fan-out ceiling** — a maximum element count, exceeding which is a
  *statically detectable* error raised before execution (§7), so an unbounded
  input can never silently become an unbounded run.
- **Interaction with existing features** — sandbox applies per element;
  usage/cost sums across elements (as it already does across tool turns);
  `depends_on` sees the mapped task as one node.

Because B touches the schema, it ships as its own proposal + PR once this
document sets the direction. It is called out here so the boundary is drawn
around it deliberately, not by accident.

**C — Refuse, and document the pattern instead.** Conditional, until-good, and
feedback loops do not enter the schema. The moment a spec's control flow
depends on a runtime predicate, OA loses static analysability, reproducibility,
and its acyclic guarantee — every property that makes a spec a contract. The
supported pattern is the same shape as markdown-interop:

> The orchestrator loops. Each turn it runs an OA spec. OA never owns the loop.

Behavioural Contracts (BCE) is the layer that owns this: an until-good refine
loop is a BCE (or caller) concern that invokes a deterministic OA spec on each
pass. Keeping OA as the deterministic unit is precisely what makes the loop
above it safe to reason about.

## What we explicitly refuse

To keep the line bright, the following stay out of scope unless revisited by a
future proposal:

- ❌ A `while:` / `until:` / `repeat:` task construct
- ❌ Any conditional or predicate that controls iteration at runtime
- ❌ A task output feeding back as its own input (a cycle — barred by
  `depends_on` acyclicity and the cycle-detection guarantee)
- ❌ Unbounded iteration of any kind, or a map without a validated ceiling

The moment termination depends on a runtime judgement, OA inherits every
problem it is currently immune to.

## Future (optional, demand-driven)

If bounded map (B) proves its worth, a natural follow-on is parallel fan-out
of the mapped elements — a runtime optimisation *under* the same bounded,
order-preserving contract, not a new spec surface. As with the `oa skill
package` idea in the markdown-interop proposal, this is tooling on top of the
boundary and should not be built until the demand is real.

## How we manage it

- **A** can land quickly as a small schema addition plus changelog note.
- **B** gets its own proposal and schema PR after this document sets
  direction; nothing about the mapped-task semantics is decided here beyond
  "bounded, ordered, fail-closed, ceilinged".
- **C** joins "no orchestration" and "no skills task type" as a boundary
  cited in future feature discussions.
- If the industry's loop patterns standardise into something bounded and
  declarable, we reassess via a new proposal — from a position of a clear
  boundary rather than an ad-hoc addition.
