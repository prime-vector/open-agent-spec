# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""Spec-derived config helpers (agent info, memory, logging, naming)."""

from typing import Any


def get_agent_info(spec_data: dict[str, Any]) -> dict[str, str]:
    """Get agent info from either old or new spec format."""
    agent = spec_data.get("agent", {})
    if agent:
        return {
            "name": agent.get("name", ""),
            "description": agent.get("description", ""),
        }
    info = spec_data.get("info", {})
    return {"name": info.get("name", ""), "description": info.get("description", "")}


def get_memory_config(spec_data: dict[str, Any]) -> dict[str, Any]:
    """Get memory configuration from spec."""
    memory = spec_data.get("memory", {})
    return {
        "enabled": memory.get("enabled", False),
        "format": memory.get("format", "string"),
        "usage": memory.get("usage", "prompt-append"),
        "required": memory.get("required", False),
        "description": memory.get("description", ""),
    }


def get_logging_config(spec_data: dict[str, Any]) -> dict[str, Any]:
    """Get logging configuration from spec."""
    logging_config = spec_data.get("logging", {})
    return {
        "enabled": logging_config.get("enabled", True),
        "level": logging_config.get("level", "INFO"),
        "format_style": logging_config.get("format_style", "emoji"),
        "include_timestamp": logging_config.get("include_timestamp", True),
        "log_file": logging_config.get("log_file"),
        "env_overrides": logging_config.get("env_overrides", {}),
    }


def to_pascal_case(name: str) -> str:
    """Convert snake_case or kebab-case to PascalCase."""
    name = name.replace("-", "_")
    return "".join(word.capitalize() for word in name.split("_"))
