# MCP Search Example

Demonstrates connecting an OA agent to any MCP (Model Context Protocol) server
for live web search — with **no MCP SDK**, no heavy dependencies, just raw HTTP.

## How it works

```
oa run → MCPToolProvider → tools/list  → discovers available tools
                         → tools/call  → executes search, returns result
                         → model sees result → generates final answer
```

OA speaks JSON-RPC 2.0 over HTTP directly to your MCP server, the same way
it calls OpenAI and Anthropic — raw `urllib.request`, nothing else needed.

## Quick start

**1. Start an MCP server.**

Any MCP-compatible server works. Examples:

```bash
# Brave Search MCP server (Node)
npx @modelcontextprotocol/server-brave-search

# Or any other MCP server — filesystem, GitHub, Slack, Postgres, etc.
```

**2. Set your credentials.**

```bash
export OPENAI_API_KEY="sk-..."
export BRAVE_MCP_TOKEN="your-brave-api-key"   # if your server needs auth
```

**3. Update the endpoint in the spec** (if your server runs on a different port):

```yaml
tools:
  brave_search:
    type: mcp
    endpoint: "http://localhost:3000"   # ← change this
```

**4. Run it.**

```bash
oa run --spec examples/mcp-search/mcp-search.yaml \
       --input '{"topic": "latest advances in open source LLMs 2026"}'
```

## Connecting other MCP servers

Just change `endpoint` — the rest is automatic. OA calls `tools/list` to
discover whatever tools your server exposes and presents them to the model.

```yaml
tools:
  github:
    type: mcp
    endpoint: "http://localhost:3001"
    headers:
      Authorization: "Bearer ${GITHUB_TOKEN}"

  filesystem:
    type: mcp
    endpoint: "http://localhost:3002"

tasks:
  my_task:
    tools: [github, filesystem]   # mix and match freely
```

## What OA sends to your MCP server

**Tool discovery** (once, cached):
```json
{"jsonrpc": "2.0", "method": "tools/list", "id": 1}
```

**Tool execution** (per model request):
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {"name": "brave_web_search", "arguments": {"query": "..."}},
  "id": 2
}
```

Standard JSON-RPC 2.0 — works with any compliant server.
