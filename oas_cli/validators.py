"""Validation functions for Open Agent Spec."""
from typing import Tuple

def validate_spec(spec_data: dict) -> Tuple[str, str]:
    """Validate the Open Agent Spec structure and return agent name and class name.
    
    Args:
        spec_data: The parsed YAML spec data
        
    Returns:
        Tuple of (agent_name, class_name)
        
    Raises:
        KeyError: If required fields are missing
        ValueError: If field types are invalid
    """
    try:
        info = spec_data["info"]
        intelligence = spec_data["intelligence"]
        
        # Validate info fields
        if not isinstance(info.get("name"), str):
            raise ValueError("info.name must be a string")
        if not isinstance(info.get("description"), str):
            raise ValueError("info.description must be a string")
            
        # Validate intelligence fields
        if not isinstance(intelligence.get("endpoint"), str):
            raise ValueError("intelligence.endpoint must be a string")
        if not isinstance(intelligence.get("model"), str):
            raise ValueError("intelligence.model must be a string")
        if not isinstance(intelligence.get("config"), dict):
            raise ValueError("intelligence.config must be a dictionary")
            
        agent_name = info["name"].replace("-", "_")
        class_name = agent_name.title().replace("_", "") + "Agent"
        return agent_name, class_name
        
    except KeyError as e:
        raise KeyError(f"Missing required field in spec: {e}")
    except Exception as e:
        raise ValueError(f"Invalid spec format: {e}") 