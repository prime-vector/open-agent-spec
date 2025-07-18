# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under AGPL-3.0 with Additional Terms
# See LICENSE for details on attribution, naming, and branding restrictions.

"""Utility functions for Open Agent Spec."""

import json
from typing import Dict, Any


def parse_response(result: str, output_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the response into the expected output format.

    Args:
        result: The raw response from the model
        output_schema: The expected output schema from the YAML

    Returns:
        Dict containing the parsed response

    Raises:
        ValueError: If the response cannot be parsed as JSON or doesn't match the schema
    """
    try:
        # Try to find JSON in the response
        json_start = result.find("{")
        json_end = result.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = result[json_start:json_end]
            parsed = json.loads(json_str)

            # Validate that all required fields are present
            for key in output_schema.get("properties", {}).keys():
                if key not in parsed:
                    raise ValueError(f"Missing required field: {key}")

            return parsed

        raise ValueError("No valid JSON found in response")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in response: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing response: {e}")
