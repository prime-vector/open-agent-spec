# Multi-Agent Orchestration Example

A reference implementation showing how OA Spec agents can work together
in a multi-agent workflow.  This is **not** part of the OA Spec core — it's
one possible orchestration pattern.  Use it as a starting point, swap out
the pieces, or build your own.

## Architecture

```
User → Concierge (clarify) → Manager (plan) → Workers (execute) → Concierge (summarise) → User
```

| Agent | Role | What it does |
|-------|------|-------------|
| Concierge | chat | Clarifies requests, presents results |
| Manager | planner | Decomposes objectives into tasks on a board |
| Researcher | analyst | Deep research and analysis |
| Writer | writer | Content drafting |
| Reviewer | reviewer | Quality review (approve/revise/reject) |

Each agent is a standard OA Spec YAML file in `personas/`.  The orchestration
layer is ~400 lines of Python with no external dependencies beyond the OA CLI.

## Prerequisites

- **Python 3.10+**
- An **Anthropic** API key (the default personas use Claude models)
- Optionally an **OpenAI** API key if you swap personas to OpenAI engines

## Quick Start

```bash
# From the repo root — install OA Spec if you haven't
pip install -e .

# Install example dependencies (FastAPI + uvicorn for the dashboard)
pip install -r examples/multi-agent/requirements.txt

# Set up your API keys
cd examples/multi-agent
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY (and optionally OPENAI_API_KEY)

# Run inline
python run.py "Write a blog post about AI agent frameworks"

# Or launch the dashboard
python run.py --dashboard
```

> **Note:** The dashboard requires `fastapi` and `uvicorn`.  The inline mode
> (`python run.py "..."`) works without them.
>
> **API Keys:** The personas default to Anthropic Claude models. Set
> `ANTHROPIC_API_KEY` in your `.env` file. See `.env.example` for the template.

## Components

| File | Purpose |
|------|---------|
| `board.py` | Task board — priority queue with dependency gating |
| `registry.py` | Agent registry — tracks who's available and what they do |
| `runner.py` | Agent runner — adapter to `oas_cli.runner` |
| `loop.py` | Orchestration loop — wires manager + workers + board |
| `dashboard.py` | Web UI — FastAPI + HTML, polls `/api/status` |
| `run.py` | CLI entry point |
| `personas/` | OA Spec YAML files for each agent persona |

## Swapping Backends

The board, registry, and runner are deliberately thin.  To use a different
backend, subclass and override:

- **`TaskBoard`** → Redis, SQLite, Postgres, etc.
- **`AgentRunner`** → Celery task, Temporal activity, HTTP call, etc.
- **`AgentRegistry`** → Service discovery, consul, etc.

The agent YAML specs stay the same regardless of what runs underneath.

## Key Design Principle

> OA Spec defines *what* an agent is.  This example shows *one way* to
> coordinate them.  The spec is the contract, not the runtime.
