"""MCP (Model Context Protocol) tool provider — raw HTTP, no MCP SDK required.

Connects to any MCP server that speaks JSON-RPC 2.0 over HTTP.  OAS calls
``tools/list`` once on first use to discover available tools, then
``tools/call`` for each model-requested invocation.

Spec example
------------
tools:
  github:
    type: mcp
    endpoint: "http://localhost:3000"
    description: "GitHub tools via MCP"          # optional — overrides server name
    timeout: 30                                   # optional, default 30s
    headers:                                      # optional extra headers
      Authorization: "Bearer ${GITHUB_TOKEN}"

  # point at a remote/hosted MCP server
  brave_search:
    type: mcp
    endpoint: "https://mcp.brave.com"
    headers:
      X-API-Key: "${BRAVE_API_KEY}"

tasks:
  research:
    tools: [github, brave_search]

Security note
-------------
Errors from an MCP server surface as ``ToolError`` and never crash the runner.
OAS is the caller — it does not host or execute server-side code, so a
misbehaving MCP server affects only the tool result for that call.

Environment variable interpolation
-----------------------------------
Values in ``headers`` that look like ``${VAR_NAME}`` are expanded from the
environment at call time, keeping secrets out of spec files.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from .base import ToolDefinition, ToolError, ToolNotFoundError, ToolProvider

_DEFAULT_TIMEOUT = 30
_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: str) -> str:
    """Replace ``${VAR_NAME}`` patterns with environment variable values."""

    def _replace(match: re.Match) -> str:
        var = match.group(1)
        resolved = os.environ.get(var, "")
        if not resolved:
            import warnings

            warnings.warn(
                f"[mcp] Environment variable '{var}' referenced in tool config is not set.",
                stacklevel=3,
            )
        return resolved

    return _ENV_VAR_RE.sub(_replace, value)


def _resolve_headers(raw: dict[str, str] | None) -> dict[str, str]:
    """Expand env vars in header values and return a plain dict."""
    if not raw:
        return {}
    return {k: _expand_env(str(v)) for k, v in raw.items()}


def _jsonrpc(
    endpoint: str,
    method: str,
    params: dict | None,
    request_id: int,
    extra_headers: dict[str, str],
    timeout: int,
) -> Any:
    """Send a JSON-RPC 2.0 request and return the ``result`` field.

    Raises:
        ToolError: On any HTTP error or JSON-RPC error response.
    """
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "id": request_id,
    }
    if params is not None:
        payload["params"] = params

    all_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        **extra_headers,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        endpoint, data=body, headers=all_headers, method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise ToolError(f"MCP HTTP {exc.code} from {endpoint}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ToolError(f"MCP connection failed ({endpoint}): {exc.reason}") from exc
    except Exception as exc:
        raise ToolError(f"MCP request to {endpoint} failed: {exc}") from exc

    if "error" in data:
        err = data["error"]
        raise ToolError(
            f"MCP error from {endpoint} [{method}]: "
            f"code={err.get('code')} message={err.get('message')}"
        )

    return data.get("result")


class MCPToolProvider(ToolProvider):
    """Connects to an MCP server and exposes its tools to OAS tasks.

    Tool discovery (``tools/list``) is lazy — it happens on the first call to
    ``describe()`` or ``call()`` and the result is cached for the lifetime of
    the provider instance.
    """

    def __init__(self, tool_name: str, tool_config: dict[str, Any]) -> None:
        self._tool_name = tool_name
        endpoint: str = tool_config.get("endpoint", "")
        if not endpoint:
            raise ToolError(
                f"MCP tool '{tool_name}' must specify an 'endpoint' "
                "(e.g. endpoint: http://localhost:3000)."
            )
        self._endpoint = endpoint.rstrip("/")
        self._timeout = int(tool_config.get("timeout", _DEFAULT_TIMEOUT))
        self._extra_headers = _resolve_headers(tool_config.get("headers"))
        self._override_description: str | None = tool_config.get("description")
        self._cached_definitions: list[ToolDefinition] | None = None
        self._req_id = 0

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _fetch_tools(self) -> list[ToolDefinition]:
        """Call ``tools/list`` and convert to ``ToolDefinition`` objects."""
        result = _jsonrpc(
            endpoint=self._endpoint,
            method="tools/list",
            params=None,
            request_id=self._next_id(),
            extra_headers=self._extra_headers,
            timeout=self._timeout,
        )

        raw_tools: list[dict] = []
        if isinstance(result, dict):
            raw_tools = result.get("tools", [])
        elif isinstance(result, list):
            raw_tools = result

        if not raw_tools:
            raise ToolError(
                f"MCP server at {self._endpoint} returned no tools from tools/list."
            )

        definitions = []
        for t in raw_tools:
            name = t.get("name", "")
            description = t.get("description", self._override_description or "")
            # MCP uses ``inputSchema``; fall back to an empty object schema.
            parameters = (
                t.get("inputSchema")
                or t.get("input_schema")
                or {
                    "type": "object",
                    "properties": {},
                }
            )
            definitions.append(
                ToolDefinition(
                    name=name,
                    description=description,
                    parameters=parameters,
                )
            )
        return definitions

    def describe(self) -> list[ToolDefinition]:
        if self._cached_definitions is None:
            self._cached_definitions = self._fetch_tools()
        return self._cached_definitions

    def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        # Ensure we've discovered tools so we can validate the name.
        known = {d.name for d in self.describe()}
        if tool_name not in known:
            raise ToolNotFoundError(
                f"MCP server at {self._endpoint} does not expose tool '{tool_name}'. "
                f"Available: {sorted(known)}"
            )

        result = _jsonrpc(
            endpoint=self._endpoint,
            method="tools/call",
            params={"name": tool_name, "arguments": arguments},
            request_id=self._next_id(),
            extra_headers=self._extra_headers,
            timeout=self._timeout,
        )

        return _extract_call_result(result, tool_name, self._endpoint)


def _extract_call_result(result: Any, tool_name: str, endpoint: str) -> str:
    """Normalise the MCP ``tools/call`` result to a plain string.

    MCP servers return results in different shapes:
      - ``{"content": [{"type": "text", "text": "..."}]}``   (most common)
      - ``{"content": "..."}``                               (simplified)
      - a bare string
      - ``{"result": "..."}``

    We always give the model a string.
    """
    if result is None:
        return ""

    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        # Standard MCP content block list
        content = result.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)

        if isinstance(content, str):
            return content

        # Some servers put the result directly
        if "result" in result:
            return str(result["result"])

        # Fallback: serialise the whole dict
        return json.dumps(result)

    # Lists, numbers, booleans — serialise as JSON
    return json.dumps(result)
