"""Tool provider registry — resolves spec tool declarations to ToolProvider instances."""

from __future__ import annotations

from typing import Any

from .base import (
    ToolDefinition,
    ToolError,
    ToolNotFoundError,
    ToolProvider,
    ToolTypeNotSupportedError,
)
from .custom import CustomToolProvider
from .mcp import MCPToolProvider
from .native import NativeToolProvider, available_native_tools


def get_tool_provider(tool_name: str, tool_config: dict[str, Any]) -> ToolProvider:
    """Return the correct ``ToolProvider`` for a single tool declaration.

    Supported types
    ---------------
    native   — built-in OAS tools (file.read, file.write, http.get, …)
    mcp      — any MCP server (JSON-RPC 2.0 over HTTP, no MCP SDK required)
    custom   — user-provided Python class
    """
    tool_type = tool_config.get("type", "native")

    if tool_type == "native":
        native_id = tool_config.get("native")
        if not native_id:
            raise ToolError(
                f"Tool '{tool_name}' has type 'native' but is missing the 'native' field "
                f"(e.g. native: file.read). Available: {available_native_tools()}"
            )
        return NativeToolProvider(enabled_ids=[native_id])

    if tool_type == "mcp":
        return MCPToolProvider(tool_name=tool_name, tool_config=tool_config)

    if tool_type == "custom":
        return CustomToolProvider(tool_name=tool_name, tool_config=tool_config)

    raise ToolTypeNotSupportedError(
        f"Tool '{tool_name}' has unsupported type '{tool_type}'. "
        "Supported types: native, mcp, custom."
    )


def resolve_task_tools(
    spec_data: dict[str, Any],
    task_name: str,
) -> list[tuple[ToolProvider, ToolDefinition]]:
    """Return (provider, definition) pairs for every tool enabled on *task_name*.

    Returns an empty list when the task declares no tools.
    """
    spec_tools: dict[str, Any] = spec_data.get("tools") or {}
    tasks: dict[str, Any] = spec_data.get("tasks") or {}
    task_def: dict[str, Any] = tasks.get(task_name) or {}
    task_tool_names: list[str] = task_def.get("tools") or []

    if not task_tool_names:
        return []

    result: list[tuple[ToolProvider, ToolDefinition]] = []
    for name in task_tool_names:
        tool_cfg = spec_tools.get(name)
        if tool_cfg is None:
            raise ToolError(
                f"Task '{task_name}' references tool '{name}' which is not declared "
                "in the top-level 'tools:' block."
            )
        provider = get_tool_provider(name, tool_cfg)
        for defn in provider.describe():
            result.append((provider, defn))

    return result


def all_tool_definitions(
    spec_data: dict[str, Any],
    task_name: str,
) -> list[ToolDefinition]:
    """Convenience helper — return only the ToolDefinition objects."""
    return [defn for _, defn in resolve_task_tools(spec_data, task_name)]


def dispatch_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    providers_and_defs: list[tuple[ToolProvider, ToolDefinition]],
) -> str:
    """Find the provider that owns *tool_name* and execute the call."""
    for provider, defn in providers_and_defs:
        if defn.name == tool_name:
            return provider.call(tool_name, arguments)
    raise ToolNotFoundError(f"No tool named '{tool_name}' is registered for this task.")
