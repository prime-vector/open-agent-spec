"""Minimal local MCP server — pure Python stdlib, no dependencies.

Exposes three tools:
  word_count   Count words in a string.
  char_count   Count characters in a string.
  reverse_text Reverse a string.

Speaks JSON-RPC 2.0 over HTTP on port 3000 (or $MCP_PORT).
Run it with:  python3 server.py
"""

import datetime
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(os.environ.get("MCP_PORT", 3000))

TOOLS = [
    {
        "name": "word_count",
        "description": "Count the number of words in a text string.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to count words in.",
                }
            },
            "required": ["text"],
        },
    },
    {
        "name": "char_count",
        "description": "Count the number of characters (including spaces) in a text string.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to count characters in.",
                }
            },
            "required": ["text"],
        },
    },
    {
        "name": "reverse_text",
        "description": "Reverse the characters in a text string.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to reverse.",
                }
            },
            "required": ["text"],
        },
    },
]


def _handle_tool_call(name: str, arguments: dict) -> dict:
    """Execute a tool and return an MCP content block result."""
    text = arguments.get("text", "")

    if name == "word_count":
        count = len(text.split())
        result = f"{count} word{'s' if count != 1 else ''}"

    elif name == "char_count":
        count = len(text)
        result = f"{count} character{'s' if count != 1 else ''}"

    elif name == "reverse_text":
        result = text[::-1]

    else:
        return {
            "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
            "isError": True,
        }

    return {"content": [{"type": "text", "text": result}]}


class MCPHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)

        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON"})
            return

        method = request.get("method", "")
        request_id = request.get("id")

        if method == "tools/list":
            self._jsonrpc(request_id, {"tools": TOOLS})

        elif method == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            result = _handle_tool_call(tool_name, arguments)
            self._jsonrpc(request_id, result)

        else:
            self._jsonrpc(
                request_id,
                error={"code": -32601, "message": f"Method not found: {method}"},
            )

    def _jsonrpc(self, request_id, result=None, error=None):
        response = {"jsonrpc": "2.0", "id": request_id}
        if error:
            response["error"] = error
        else:
            response["result"] = result
        self._send(200, response)

    def _send(self, status: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] MCP  {fmt % args}")


if __name__ == "__main__":
    server = HTTPServer(("localhost", PORT), MCPHandler)
    print(f"Local MCP server running on http://localhost:{PORT}")
    print(f"Tools: {', '.join(t['name'] for t in TOOLS)}")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
