"""Base class and types for OAS tool providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ToolError(RuntimeError):
    """Raised when a tool call fails."""


class ToolNotFoundError(ToolError):
    """Raised when a named tool is not registered."""


class ToolTypeNotSupportedError(ToolError):
    """Raised when the tool type is not recognised."""


@dataclass
class ToolDefinition:
    """OpenAI-compatible function definition sent to the model.

    The ``parameters`` dict follows JSON Schema — the model receives it verbatim
    so it knows how to call the tool.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )

    def to_openai_schema(self) -> dict[str, Any]:
        """Serialise to the OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Serialise to the Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class InvokeResult:
    """Return value from a provider's invoke_with_tools call.

    Either ``is_final`` is True and ``text`` carries the answer, or
    ``tool_calls`` is non-empty and the caller should execute them and continue.
    """

    is_final: bool
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class ToolProvider(ABC):
    """Interface every tool backend must implement."""

    @abstractmethod
    def describe(self) -> list[ToolDefinition]:
        """Return the tool definitions for this provider (sent to the model)."""

    @abstractmethod
    def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute *tool_name* with *arguments* and return a string result.

        Raises:
            ToolNotFoundError: If the tool name is not handled by this provider.
            ToolError: On execution failure.
        """
