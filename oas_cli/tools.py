# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under AGPL-3.0 with Additional Terms
# See LICENSE for details on attribution, naming, and branding restrictions.

"""Tool implementations for Open Agent Spec."""

import logging
from pathlib import Path
from typing import Dict, Any, List

log = logging.getLogger(__name__)


def file_writer(
    file_path: str, content: str, allowed_paths: List[str] = None
) -> Dict[str, Any]:
    """Write content to a file with safety checks.

    Args:
        file_path: Path to the file to write
        content: Content to write to the file
        allowed_paths: List of allowed directory paths for safety

    Returns:
        Dictionary with success status and file path
    """
    try:
        # Convert to Path object for easier manipulation
        target_path = Path(file_path).resolve()

        # Safety check: ensure the target path is within allowed directories
        if allowed_paths:
            is_allowed = False
            for allowed_path in allowed_paths:
                allowed_dir = Path(allowed_path).resolve()
                try:
                    # Check if target_path is within allowed_dir
                    target_path.relative_to(allowed_dir)
                    is_allowed = True
                    break
                except ValueError:
                    # target_path is not within this allowed_dir, continue checking
                    continue

            if not is_allowed:
                return {
                    "success": False,
                    "file_path": str(target_path),
                    "error": f"File path {file_path} is not within allowed directories: {allowed_paths}",
                }

        # Create parent directories if they don't exist
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the content to the file
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)

        log.info(f"Successfully wrote content to {target_path}")

        return {
            "success": True,
            "file_path": str(target_path),
            "bytes_written": len(content.encode("utf-8")),
        }

    except Exception as e:
        log.error(f"Error writing to file {file_path}: {e}")
        return {"success": False, "file_path": file_path, "error": str(e)}


# Tool registry mapping tool IDs to their implementations
TOOL_REGISTRY = {"file_writer": file_writer}


def get_tool_implementation(tool_id: str):
    """Get a tool implementation by ID.

    Args:
        tool_id: The ID of the tool to get

    Returns:
        The tool function implementation

    Raises:
        ValueError: If the tool ID is not found
    """
    if tool_id not in TOOL_REGISTRY:
        raise ValueError(f"Tool '{tool_id}' not found in registry")

    return TOOL_REGISTRY[tool_id]
