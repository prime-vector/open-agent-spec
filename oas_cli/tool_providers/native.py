"""Native OA tools — zero external dependencies, sandboxed by design.

Available tools
---------------
file.read   Read a file and return its text content.
file.write  Write text content to a file (creates parent dirs).
http.get    Perform an HTTP GET and return the response body.
http.post   Perform an HTTP POST with a JSON payload and return the response body.
env.read    Read a single environment variable (empty string if unset).

Security note
-------------
``file.read`` / ``file.write`` operate on paths as given.  In production you
should add an allow-list in the spec (``allowed_paths:`` — future work).
``http.get`` / ``http.post`` make real network requests; restrict with
``allowed_hosts:`` when needed.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .base import ToolDefinition, ToolError, ToolNotFoundError, ToolProvider

# Registry: tool_id → (ToolDefinition, handler)
_NATIVE_TOOLS: dict[str, tuple[ToolDefinition, Any]] = {}


def _register(tool_id: str, defn: ToolDefinition):
    """Decorator that registers a handler under *tool_id*."""

    def decorator(fn):
        _NATIVE_TOOLS[tool_id] = (defn, fn)
        return fn

    return decorator


# ── file.read ─────────────────────────────────────────────────────────────────


@_register(
    "file.read",
    ToolDefinition(
        name="file_read",
        description="Read a file from the filesystem and return its text content.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to read.",
                },
            },
            "required": ["path"],
        },
    ),
)
def _file_read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ToolError(f"file.read: file not found: {path}") from exc
    except PermissionError as exc:
        raise ToolError(f"file.read: permission denied: {path}") from exc
    except Exception as exc:
        raise ToolError(f"file.read: {exc}") from exc


# ── file.write ────────────────────────────────────────────────────────────────


@_register(
    "file.write",
    ToolDefinition(
        name="file_write",
        description="Write text content to a file. Parent directories are created if needed.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "Text content to write.",
                },
            },
            "required": ["path", "content"],
        },
    ),
)
def _file_write(path: str, content: str) -> str:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} characters to {path}"
    except PermissionError as exc:
        raise ToolError(f"file.write: permission denied: {path}") from exc
    except Exception as exc:
        raise ToolError(f"file.write: {exc}") from exc


# ── http.get ──────────────────────────────────────────────────────────────────


@_register(
    "http.get",
    ToolDefinition(
        name="http_get",
        description="Perform an HTTP GET request and return the response body as text.",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch.",
                },
                "headers": {
                    "type": "object",
                    "description": "Optional HTTP headers as key-value pairs.",
                },
            },
            "required": ["url"],
        },
    ),
)
def _http_get(url: str, headers: dict | None = None) -> str:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode(errors="replace")
    except urllib.error.HTTPError as exc:
        raise ToolError(f"http.get HTTP {exc.code}: {url}") from exc
    except Exception as exc:
        raise ToolError(f"http.get failed for {url}: {exc}") from exc


# ── http.post ─────────────────────────────────────────────────────────────────


@_register(
    "http.post",
    ToolDefinition(
        name="http_post",
        description="Perform an HTTP POST with a JSON body and return the response body as text.",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to POST to.",
                },
                "body": {
                    "type": "object",
                    "description": "JSON-serialisable payload to send as the request body.",
                },
                "headers": {
                    "type": "object",
                    "description": "Optional additional HTTP headers.",
                },
            },
            "required": ["url", "body"],
        },
    ),
)
def _http_post(url: str, body: dict, headers: dict | None = None) -> str:
    all_headers = {"Content-Type": "application/json", **(headers or {})}
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=all_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode(errors="replace")
    except urllib.error.HTTPError as exc:
        raise ToolError(f"http.post HTTP {exc.code}: {url}") from exc
    except Exception as exc:
        raise ToolError(f"http.post failed for {url}: {exc}") from exc


# ── env.read ──────────────────────────────────────────────────────────────────


@_register(
    "env.read",
    ToolDefinition(
        name="env_read",
        description="Read an environment variable. Returns an empty string if the variable is not set.",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The environment variable name.",
                },
            },
            "required": ["name"],
        },
    ),
)
def _env_read(name: str) -> str:
    return os.environ.get(name, "")


# ── Provider class ────────────────────────────────────────────────────────────


class NativeToolProvider(ToolProvider):
    """Exposes a subset of native OA tools declared in the spec.

    The spec names which native tool IDs are enabled for a task (e.g.
    ``file.read``, ``http.get``).  This provider exposes only those, so the
    model never sees tools it isn't supposed to call.
    """

    def __init__(self, enabled_ids: list[str]) -> None:
        unknown = [t for t in enabled_ids if t not in _NATIVE_TOOLS]
        if unknown:
            raise ToolError(
                f"Unknown native tool(s): {', '.join(unknown)}. "
                f"Available: {', '.join(_NATIVE_TOOLS)}"
            )
        self._enabled = enabled_ids

    def describe(self) -> list[ToolDefinition]:
        return [_NATIVE_TOOLS[tid][0] for tid in self._enabled]

    def call(self, tool_name: str, arguments: dict) -> str:
        # tool_name is the function name (e.g. "file_read"), map back to tool_id
        for tid in self._enabled:
            defn, handler = _NATIVE_TOOLS[tid]
            if defn.name == tool_name:
                try:
                    return handler(**arguments)
                except ToolError:
                    raise
                except Exception as exc:
                    raise ToolError(f"native tool '{tool_name}' raised: {exc}") from exc
        raise ToolNotFoundError(
            f"Native tool '{tool_name}' is not enabled for this task."
        )


def available_native_tools() -> list[str]:
    """Return all registered native tool IDs."""
    return list(_NATIVE_TOOLS.keys())
