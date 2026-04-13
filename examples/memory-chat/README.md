# memory-chat

Demonstrates how to combine **long-term memory retrieval** with a
**stateless chat reply** using OAS spec composition.

## Architecture

```
input: { message, memory_store_url, top_k }
         │
         ▼
   ┌─────────────────────────────────────────────────┐
   │  Task: recall                                   │
   │  spec: oa://prime-vector/memory-retriever       │
   │  task: retrieve                                 │
   │                                                 │
   │  Calls your memory store's search endpoint.     │
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
   │  The runner injects history between system      │
   │  prompt and current user message automatically. │
   │  Returns: { reply }                             │
   └─────────────────────────────────────────────────┘
```

## Key principle

OAS is **stateless**.  Memory lives in your infrastructure (a vector DB,
Redis, a simple SQLite store — anything with an HTTP search endpoint).
The `memory-retriever` spec is just a declarative interface to it.

## Running the pipeline

```bash
# Requires a memory store running at http://localhost:8765
# (see below for a minimal mock)
oa run examples/memory-chat/pipeline.yaml respond \
  --input examples/memory-chat/input.json
```

## Minimal memory store mock (for testing)

```python
# mock_memory_store.py — serves a static search result
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

MEMORIES = [
    {"role": "user",      "content": "The project deadline was moved to Q3."},
    {"role": "assistant", "content": "Noted — I've updated the timeline."},
]

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"results": MEMORIES}).encode())

HTTPServer(("localhost", 8765), Handler).serve_forever()
```

Run with `python mock_memory_store.py`, then run the pipeline above.
