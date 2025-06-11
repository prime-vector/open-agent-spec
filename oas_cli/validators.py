"""Validation functions for Open Agent Spec."""
from typing import Tuple, Dict, Any

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
        # Check spec version
        if not isinstance(spec_data.get("open_agent_spec"), str):
            raise ValueError("open_agent_spec version must be specified")
            
        # Validate agent section
        agent = spec_data.get("agent", {})
        if not isinstance(agent.get("name"), str):
            raise ValueError("agent.name must be a string")
        if not isinstance(agent.get("role"), str):
            raise ValueError("agent.role must be a string")
            
        # Validate behavioral contract
        contract = spec_data.get("behavioral_contract", {})
        if not isinstance(contract.get("version"), str):
            raise ValueError("behavioral_contract.version must be a string")
        if not isinstance(contract.get("policy"), dict):
            raise ValueError("behavioral_contract.policy must be a dictionary")
            
        # Validate tasks
        tasks = spec_data.get("tasks", {})
        if not isinstance(tasks, dict):
            raise ValueError("tasks must be a dictionary")
        for task_name, task_def in tasks.items():
            if not isinstance(task_def.get("input"), dict):
                raise ValueError(f"task {task_name}.input must be a dictionary")
            if not isinstance(task_def.get("output"), dict):
                raise ValueError(f"task {task_name}.output must be a dictionary")
                
        # Validate integration
        integration = spec_data.get("integration", {})
        if integration:
            if not isinstance(integration.get("memory"), dict):
                raise ValueError("integration.memory must be a dictionary")
            if not isinstance(integration.get("task_queue"), dict):
                raise ValueError("integration.task_queue must be a dictionary")
            
        # Validate prompts
        prompts = spec_data.get("prompts", {})
        if not isinstance(prompts.get("system"), str):
            raise ValueError("prompts.system must be a string")
        if not isinstance(prompts.get("user"), str):
            raise ValueError("prompts.user must be a string")
            
        # Generate agent name and class name
        agent_name = agent["name"].replace("-", "_")
        base_class_name = agent_name.title().replace("_", "")
        class_name = base_class_name if base_class_name.endswith("Agent") else base_class_name + "Agent"
        
        return agent_name, class_name
        
    except KeyError as e:
        raise KeyError(f"Missing required field in spec: {e}")
    except Exception as e:
        raise ValueError(f"Invalid spec format: {e}") 