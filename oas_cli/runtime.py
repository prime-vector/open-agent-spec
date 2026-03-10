"""Runtime primitives for agents generated from Open Agent Spec.

This module provides a small, stable surface area used by generated code:
it exposes an abstracted intelligence invocation API and related helpers.
Internally it is currently implemented using DACP, but callers should depend
only on this module so that the underlying implementation can evolve over time.
"""

from __future__ import annotations

import os
from typing import Any

import dacp
from dacp import execute_tool as _execute_tool
from dacp.orchestrator import Orchestrator as _DacpOrchestrator
from dacp.protocol import (
    get_final_response as _get_final_response,
)
from dacp.protocol import (
    get_tool_request as _get_tool_request,
)
from dacp.protocol import (
    is_final_response as _is_final_response,
)
from dacp.protocol import (
    is_tool_request as _is_tool_request,
)
from dacp.protocol import (
    parse_agent_response as _parse_agent_response,
)
from dacp.protocol import (
    wrap_tool_result as _wrap_tool_result,
)

from .adapters import codex_adapter

# Re-exported types for generated agents
AgentBase = dacp.Agent
Orchestrator = _DacpOrchestrator
execute_tool = _execute_tool
parse_agent_response = _parse_agent_response
is_tool_request = _is_tool_request
get_tool_request = _get_tool_request
wrap_tool_result = _wrap_tool_result
get_final_response = _get_final_response
is_final_response = _is_final_response


def invoke_intelligence(prompt: str, config: dict[str, Any]) -> Any:
    """Invoke the configured intelligence provider.

    The engine is chosen based on config["engine"]:
    - "codex": routed to the Codex CLI adapter
    - anything else: delegated to DACP's invoke_intelligence (OpenAI, etc.)
    """
    engine = (config.get("engine") or "").lower()
    if engine == "codex":
        return codex_adapter.invoke(prompt, config)
    return dacp.invoke_intelligence(prompt, config)


def parse_with_fallback(response: Any, model_class: type, **defaults: Any) -> Any:
    """Parse LLM output into a Pydantic model with sensible fallbacks.

    Thin wrapper around DACP's parse_with_fallback so generated code does not
    import DACP directly.
    """
    return dacp.parse_with_fallback(
        response=response, model_class=model_class, **defaults
    )


def setup_logging_from_config(logging_config: dict[str, Any]) -> None:
    """Configure logging based on a logging config dict from the spec.

    This mirrors the behaviour previously in generated agents but is factored
    into a reusable helper so that logging can be adjusted without regenerating
    all agents.
    """
    if not logging_config.get("enabled", True):
        return

    # Process environment variable overrides
    env_overrides = logging_config.get("env_overrides", {}) or {}

    level = logging_config.get("level", "INFO")
    if "level" in env_overrides:
        level = os.getenv(env_overrides["level"], level)

    format_style = logging_config.get("format_style", "emoji")
    if "format_style" in env_overrides:
        format_style = os.getenv(env_overrides["format_style"], format_style)

    log_file = logging_config.get("log_file")
    if "log_file" in env_overrides:
        log_file = os.getenv(env_overrides["log_file"], log_file)

    # Create log directory if needed
    if log_file:
        from pathlib import Path

        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Delegate to DACP's logging configuration helper
    dacp.setup_dacp_logging(
        level=level,
        format_style=format_style,
        include_timestamp=logging_config.get("include_timestamp", True),
        log_file=log_file,
    )


__all__ = [
    "AgentBase",
    "Orchestrator",
    "execute_tool",
    "parse_agent_response",
    "is_tool_request",
    "get_tool_request",
    "wrap_tool_result",
    "get_final_response",
    "is_final_response",
    "invoke_intelligence",
    "parse_with_fallback",
    "setup_logging_from_config",
]
