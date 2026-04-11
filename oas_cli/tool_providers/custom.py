"""Custom tool provider — dynamically loads a user-defined Python class.

Spec example
------------
tools:
  my_scorer:
    type: custom
    module: "my_package.tools.Scorer"
    description: "Score text quality (0-100)"
    parameters:
      type: object
      properties:
        text:
          type: string
          description: The text to score.
      required: [text]

The class at ``my_package.tools.Scorer`` must implement::

    class Scorer:
        def describe(self) -> list[dict]:
            ...  # list of OpenAI-compatible function schemas

        def call(self, tool_name: str, arguments: dict) -> str:
            ...  # execute the tool, return a string

If ``describe()`` is absent the provider synthesises a single-tool definition
from the ``description`` and ``parameters`` fields in the spec.
"""

from __future__ import annotations

import importlib
from typing import Any

from .base import ToolDefinition, ToolError, ToolNotFoundError, ToolProvider


class CustomToolProvider(ToolProvider):
    """Wraps a dynamically-loaded user class as a ToolProvider."""

    def __init__(self, tool_name: str, tool_config: dict[str, Any]) -> None:
        self._tool_name = tool_name
        self._config = tool_config
        module_path: str = tool_config.get("module", "")
        if not module_path:
            raise ToolError(
                f"Custom tool '{tool_name}' must specify a 'module' (e.g. 'my_pkg.MyTool')."
            )

        try:
            mod_name, cls_name = module_path.rsplit(".", 1)
            module = importlib.import_module(mod_name)
            cls = getattr(module, cls_name)
            self._instance = cls()
        except (ImportError, AttributeError, ValueError) as exc:
            raise ToolError(
                f"Custom tool '{tool_name}': could not load '{module_path}': {exc}"
            ) from exc

    def describe(self) -> list[ToolDefinition]:
        if hasattr(self._instance, "describe"):
            raw = self._instance.describe()
            return [
                ToolDefinition(
                    name=d["name"],
                    description=d.get("description", ""),
                    parameters=d.get("parameters", {"type": "object", "properties": {}}),
                )
                for d in raw
            ]
        # Synthesise from spec fields
        return [
            ToolDefinition(
                name=self._tool_name.replace(".", "_").replace("-", "_"),
                description=self._config.get("description", ""),
                parameters=self._config.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            )
        ]

    def call(self, tool_name: str, arguments: dict) -> str:
        if not hasattr(self._instance, "call"):
            raise ToolError(
                f"Custom tool class for '{self._tool_name}' has no 'call' method."
            )
        result = self._instance.call(tool_name, arguments)
        if not isinstance(result, str):
            raise ToolError(
                f"Custom tool '{tool_name}' must return a string; got {type(result).__name__}."
            )
        return result
