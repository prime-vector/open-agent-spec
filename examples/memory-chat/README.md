# memory-chat

Demonstrates how to combine **LLM-powered memory re-ranking** with a
**stateless chat reply** using OA spec composition.

## How it works

OA specs are pure LLM interfaces — they cannot make HTTP calls to external
services during prompt rendering.  The memory pattern therefore has two
distinct layers:

| Layer | Owner | Responsibility |
|---|---|---|
| Memory **store** | Your infrastructure | Persist, index, and search prior turns |
| Memory **re-ranker** | `oa://prime-vector/memory-retriever` | Use the LLM to select the most relevant candidates |

Your application code fetches raw candidates from the store (however you
like — vector similarity, keyword search, recency, etc.) and passes them in
as the `candidates` input field.  The LLM then re-ranks them and returns
only the most contextually relevant turns as a `history` array.

## Architecture

```
input: { message, candidates: [...pre-fetched turns], top_k }
         │
         ▼
   ┌─────────────────────────────────────────────────┐
   │  Task: recall                                   │
   │  spec: oa://prime-vector/memory-retriever       │
   │  task: retrieve                                 │
   │                                                 │
   │  LLM re-ranks candidates, returns top_k most   │
   │  relevant turns oldest-first.                   │
   │  Returns: { history: [...turns], memory_count } │
   └────────────────────┬────────────────────────────┘
                        │ depends_on (data merge)
                        ▼
   ┌─────────────────────────────────────────────────┐
   │  Task: respond                                  │
   │  spec: ../chat-agent/spec.yaml                  │
   │  task: chat                                     │
   │                                                 │
   │  Receives: { message, history: [...] }          │
   │  Runner injects history between system prompt   │
   │  and current user message automatically.        │
   │  Returns: { reply }                             │
   └─────────────────────────────────────────────────┘
```

## Running the pipeline

```bash
oa run examples/memory-chat/pipeline.yaml respond \
  --input examples/memory-chat/input.json
```

The `input.json` file contains sample `candidates` (turns pre-fetched from
a hypothetical memory store) so the pipeline runs end-to-end without needing
a real store.

## Plugging in a real memory store

Replace `candidates` in the input with the results of a search against your
store.  Any HTTP-accessible store works — the fetch happens in your
application code before calling `oa run`:

```python
import httpx, json
from oas_cli.runner import run_task_from_file

# 1. Fetch candidates from your store
resp = httpx.post("http://localhost:8765/search", json={"query": message, "top_n": 10})
candidates = resp.json()["results"]   # [{role, content}, ...]

# 2. Run the pipeline — OA handles the rest
result = run_task_from_file(
    "examples/memory-chat/pipeline.yaml",
    task_name="respond",
    input_data={"message": message, "candidates": candidates, "top_k": 5},
)
print(result["output"]["reply"])
```

## What OA deliberately does NOT do

| Capability | Where it belongs |
|---|---|
| Fetching from the memory store | Your application code |
| Writing / upserting memories | Your application code |
| Session persistence | Your infrastructure |
| History summarisation | `oa://prime-vector/summariser` |
