# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""Behavioural contract data and decorator string generation."""

from typing import Any

from ..code_generation import PythonCodeSerializer


def _generate_contract_data(
    spec_data: dict[str, Any],
    task_def: dict[str, Any],
    agent_name: str,
    memory_config: dict[str, Any],
) -> dict[str, Any]:
    """Generate behavioural contract data from spec."""
    behavioural_section = spec_data.get("behavioural_contract")

    # If no behavioural_contract section is declared, do not attach a contract.
    if not behavioural_section:
        return {}

    # Use the task's description for the contract
    contract_data = {
        "version": behavioural_section.get("version", "0.1.2"),
        "description": task_def.get(
            "description", behavioural_section.get("description", "")
        ),
    }

    # Add role from agent section (not from behavioural_contract)
    agent_role = spec_data.get("agent", {}).get("role")
    if agent_role:
        contract_data["role"] = agent_role

    # Only add behavioural_flags if specified
    if behavioural_section.get("behavioural_flags"):
        contract_data["behavioural_flags"] = behavioural_section["behavioural_flags"]

    # Add function-specific response_contract based on the task's output schema
    output_schema = task_def.get("output", {})
    required_fields = output_schema.get("required", [])
    if required_fields:
        contract_data["response_contract"] = {
            "output_format": {"required_fields": required_fields}
        }

    return contract_data


def _format_contract_for_decorator(contract_data: dict[str, Any]) -> str:
    """Format contract data dict as the argument list for @behavioural_contract(...)."""
    return ",\n    ".join(
        f"{k}={PythonCodeSerializer.format_value(v)}" for k, v in contract_data.items()
    )
