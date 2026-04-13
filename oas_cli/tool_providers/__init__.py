"""OA tool provider package."""

from .base import (
    InvokeResult,
    ToolCall,
    ToolDefinition,
    ToolError,
    ToolNotFoundError,
    ToolProvider,
    ToolTypeNotSupportedError,
)
from .custom import CustomToolProvider
from .mcp import MCPToolProvider
from .native import NativeToolProvider, available_native_tools
from .registry import (
    all_tool_definitions,
    dispatch_tool_call,
    get_tool_provider,
    resolve_task_tools,
)

__all__ = [
    "InvokeResult",
    "ToolCall",
    "ToolDefinition",
    "ToolError",
    "ToolNotFoundError",
    "ToolProvider",
    "ToolTypeNotSupportedError",
    "CustomToolProvider",
    "MCPToolProvider",
    "NativeToolProvider",
    "available_native_tools",
    "all_tool_definitions",
    "dispatch_tool_call",
    "get_tool_provider",
    "resolve_task_tools",
]
