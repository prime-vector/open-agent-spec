# Researcher Agent — Demo Walkthrough

> **What this shows:** A multi-stage AI agent defined entirely in YAML,
> run with a single CLI command — no Python, no framework required.

---

## The Big Picture

```
oa run --task report --input '{"topic": "..."}'
           │
           ▼
    ┌──────────────┐
    │     plan     │  Stage 1 — breaks topic into research questions
    └──────┬───────┘
           │ depends_on
           ▼
    ┌──────────────┐
    │   research   │  Stage 2 — answers each question with findings
    └──────┬───────┘
           │ depends_on
           ▼
    ┌──────────────┐
    │    report    │  Stage 3 — writes a polished markdown report
    └──────────────┘
```

You declare the pipeline in YAML. OA resolves the chain automatically.

---

## Prerequisites

```bash
pip install open-agent-spec
export OPENAI_API_KEY=sk-...
```

---

## Video Script

### 🎬 Scene 1 — Open the spec file  `(researcher.yaml)`

> **Say:** "This is the entire agent. One YAML file. Let me walk you through it."

**Point to the `intelligence:` block:**
```yaml
intelligence:
  type: "llm"
  engine: "openai"
  model: "gpt-4o"
```
> "We declare the model once. Every task in this agent uses it."

**Point to the `plan` task:**
```yaml
  plan:
    description: "Break the topic into 3-4 focused research questions"
    prompts:
      system: |
        You are a research planner...
      user: "Plan research for: {{ topic }}"
    output:
      type: object
      properties:
        questions:
          type: array
```
> "Stage one. It gets a topic, returns a list of research questions.
> Notice the system prompt is right here — per-task, not global."

**Point to the `research` task — highlight `depends_on`:**
```yaml
  research:
    depends_on:
      - plan
```
> "Stage two depends on stage one. OA will run `plan` automatically before `research`.
> The output from `plan` — the questions array — is injected straight into
> `research`'s input. No glue code."

**Point to the `report` task — highlight `response_format: text`:**
```yaml
  report:
    depends_on:
      - research
    response_format: text
```
> "Stage three. It depends on `research`, so OA runs the full chain:
> plan → research → report. And `response_format: text` means the output
> is a markdown string, not JSON — perfect for a human-readable report."

---

### 🎬 Scene 2 — Run a single task

```bash
oa run --spec DEMO/researcher.yaml \
       --task plan \
       --input '{"topic": "AI agents in software development"}'
```

> "Let's start small — just the planning stage."

**Expected output** (example):
```json
{
  "task": "plan",
  "output": {
    "questions": [
      "What problems do AI agents solve in software development?",
      "Which development tasks are most automatable today?",
      "What are the risks of AI agents in production codebases?",
      "How are teams measuring the impact of AI agents?"
    ]
  }
}
```

> "There's our research plan. A structured list of questions, as JSON.
> This is what OA passes to the next stage."

---

### 🎬 Scene 3 — Run the research stage (chain auto-triggers)

```bash
oa run --spec DEMO/researcher.yaml \
       --task research \
       --input '{"topic": "AI agents in software development"}'
```

> "Now I ask for the research stage. Watch — OA automatically runs
> `plan` first because of `depends_on`. I didn't write any orchestration code."

**Expected output** (example):
```json
{
  "task": "research",
  "output": {
    "findings": [
      {
        "question": "What problems do AI agents solve in software development?",
        "answer": "AI agents reduce repetitive work in code review, test generation..."
      },
      ...
    ]
  },
  "chain": {
    "plan": { "output": { "questions": [...] } }
  }
}
```

> "The `chain` key shows us the intermediate results — we can see
> what `plan` produced. Full transparency, all in one response."

---

### 🎬 Scene 4 — Full pipeline, one command

```bash
oa run --spec DEMO/researcher.yaml \
       --task report \
       --input '{"topic": "AI agents in software development"}'
```

> "This is the money shot. One command. OA runs plan, then research,
> then report. The final output is a full markdown research report."

**Expected output** (example):
```
## Executive Summary
AI agents are rapidly transforming software development by automating repetitive
tasks like code review, test generation, and documentation...

## Key Findings
- **What problems do AI agents solve?** Agents reduce cycle times in PR review
  and catch common bugs before human review...
- **Which tasks are most automatable?** Test generation, documentation, and
  boilerplate code show the highest automation success rates...

## Analysis
The evidence suggests AI agents are most effective as collaborative tools
rather than autonomous replacements...

## Next Steps
1. Pilot AI-assisted code review on a low-risk codebase
2. Measure cycle time before and after agent introduction
3. Define boundaries — which decisions always need a human
```

---

### 🎬 Scene 5 — Change the topic, zero code changes

```bash
oa run --spec DEMO/researcher.yaml \
       --task report \
       --input '{"topic": "impact of remote work on engineering teams"}'
```

> "Same agent. Different topic. The YAML didn't change.
> This is what it means to define behaviour as a spec."

---

### 🎬 Scene 6 — Quiet mode for scripting

```bash
oa run --spec DEMO/researcher.yaml \
       --task plan \
       --input '{"topic": "AI agents in software development"}' \
       --quiet
```

> "`--quiet` gives you clean JSON on stdout — no logging, no colour.
> Pipe it into `jq`, redirect it to a file, feed it into another system.
> OA plays well with the rest of your toolchain."

---

## Key Points to Land

| What you see | What it demonstrates |
|---|---|
| `depends_on: [plan]` | Declarative task chaining — no orchestration code |
| Per-task `prompts.system` | Each stage has its own specialised system prompt |
| `response_format: text` | Final output can be prose, not just JSON |
| `chain` in the result | Full transparency into intermediate steps |
| Same spec, different `--input` | Behaviour is data, not code |

---

## File Layout

```
DEMO/
  researcher.yaml   ← the entire agent definition
  run.sh            ← copy-paste demo commands
  README.md         ← this guide
```

---

## What's Next

Once you're comfortable with the basics, the same spec supports:

- **Behavioural contracts** — add `behavioural_contract:` to enforce output shape
- **Per-task model overrides** — give expensive tasks a cheaper model
- **Different engines** — swap `engine: openai` for `engine: anthropic` or `engine: local`
- **CLI prompt injection** — `--system-prompt` to override any task's prompt at runtime
