# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under AGPL-3.0 with Additional Terms
# See LICENSE for details on attribution, naming, and branding restrictions.

"""Data preparation utilities for code generation."""

import logging
from typing import Any, Dict, List
from .code_generation import PythonCodeSerializer, TemplateVariableParser


log = logging.getLogger(__name__)


class AgentDataPreparator:
    """Prepares structured data for agent code generation."""

    def __init__(self) -> None:
        self.serializer = PythonCodeSerializer()
        self.variable_parser = TemplateVariableParser()

    def prepare_all_data(
        self, spec_data: Dict[str, Any], agent_name: str, class_name: str
    ) -> Dict[str, Any]:
        """Prepare all data needed for agent generation."""
        config = self._prepare_config(spec_data)
        memory_config = self._prepare_memory_config(spec_data)

        return {
            "agent_name": agent_name,
            "class_name": class_name,
            "imports": self._prepare_imports(spec_data),
            "models": self._prepare_models(spec_data),
            "task_functions": self._prepare_task_functions(
                spec_data, agent_name, memory_config, config
            ),
            "class_methods": self._prepare_class_methods(spec_data),
            "memory_methods": self._prepare_memory_methods(spec_data),
            "config": config,
            "embedded_config": self._prepare_embedded_config(spec_data),
            "setup_logging_method": self._prepare_setup_logging_method(),
            "handle_message_method": self._prepare_handle_message_method(),
            "custom_router_loader": self._prepare_custom_router_loader(
                spec_data, config
            ),
            "custom_router_init": self._prepare_custom_router_init(spec_data, config),
            "example_task_code": self._prepare_example_task_code(spec_data),
        }

    def _prepare_config(self, spec_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare basic configuration."""
        intelligence = spec_data.get("intelligence", {})
        return {
            "endpoint": intelligence.get("endpoint", "https://api.openai.com/v1"),
            "model": intelligence.get("model", "gpt-3.5-turbo"),
            "temperature": intelligence.get("config", {}).get("temperature", 0.7),
            "max_tokens": intelligence.get("config", {}).get("max_tokens", 1000),
        }

    def _prepare_memory_config(self, spec_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare memory configuration."""
        from .generators import (
            get_memory_config,
        )  # Import here to avoid circular imports

        return get_memory_config(spec_data)

    def _prepare_imports(self, spec_data: Dict[str, Any]) -> List[str]:
        """Prepare import statements."""
        # Base imports
        imports = [
            "import os",
            "import logging",
            "import json",
            "from pathlib import Path",
            "from typing import Optional, Any, Dict, List",
            "from jinja2 import Environment, FileSystemLoader",
            "from pydantic import BaseModel",
            "from behavioural_contracts import behavioural_contract",
            "from dotenv import load_dotenv",
            "",
            "import dacp",
            "from dacp import parse_with_fallback, invoke_intelligence",
            "from dacp.orchestrator import Orchestrator",
        ]

        # Check if any task uses tools
        tasks = spec_data.get("tasks", {})
        uses_tools = any(task_def.get("tool") for task_def in tasks.values())

        if uses_tools:
            imports.extend(
                [
                    "from dacp import execute_tool",
                    "from dacp.protocol import parse_agent_response, is_tool_request, get_tool_request, wrap_tool_result, get_final_response, is_final_response",
                ]
            )

        # Check if custom router is needed
        engine = spec_data.get("intelligence", {}).get("engine", "openai")
        custom_module = spec_data.get("intelligence", {}).get("module", None)

        if engine == "custom" and custom_module:
            imports.append("import importlib")

        return imports

    def _prepare_models(self, spec_data: Dict[str, Any]) -> List[str]:
        """Prepare Pydantic model definitions."""
        from .generators import (
            _generate_pydantic_model,
        )  # Import here to avoid circular imports

        models = []
        tasks = spec_data.get("tasks", {})

        for task_name, task_def in tasks.items():
            model_name = f"{task_name.replace('-', '_').title()}Output"
            model_def = _generate_pydantic_model(model_name, task_def.get("output", {}))
            if model_def:
                models.append(model_def)

        return models

    def _prepare_task_functions(
        self,
        spec_data: Dict[str, Any],
        agent_name: str,
        memory_config: Dict[str, Any],
        config: Dict[str, Any],
    ) -> List[str]:
        """Prepare task function code."""
        from .generators import (
            _generate_task_function,
        )  # Import here to avoid circular imports

        task_functions = []
        tasks = spec_data.get("tasks", {})

        for task_name, task_def in tasks.items():
            task_function = _generate_task_function(
                task_name, task_def, spec_data, agent_name, memory_config, config
            )
            task_functions.append(task_function)

        return task_functions

    def _prepare_class_methods(self, spec_data: Dict[str, Any]) -> List[str]:
        """Prepare agent class methods."""
        from .generators import (
            _generate_input_params,
        )  # Import here to avoid circular imports

        class_methods = []
        tasks = spec_data.get("tasks", {})

        for task_name, task_def in tasks.items():
            model_name = f"{task_name.replace('-', '_').title()}Output"

            # Get input parameters without memory
            input_params_without_memory = [
                param.split(":")[0]
                for param in _generate_input_params(task_def)
                if param != "memory_summary: str = ''"
            ]

            # Generate method signature and call
            if input_params_without_memory:
                method_signature = f"def {task_name.replace('-', '_')}(self, {', '.join(input_params_without_memory)}) -> {model_name}:"
                method_call = f"return {task_name.replace('-', '_')}({', '.join(input_params_without_memory)}, memory_summary=memory_summary)"
            else:
                method_signature = (
                    f"def {task_name.replace('-', '_')}(self) -> {model_name}:"
                )
                method_call = f"return {task_name.replace('-', '_')}(memory_summary=memory_summary)"

            class_method = f'''
    {method_signature}
        """Process {task_name} task."""
        memory_summary = self.get_memory() if hasattr(self, 'get_memory') else ""
        {method_call}
'''
            class_methods.append(class_method)

        return class_methods

    def _prepare_memory_methods(self, spec_data: Dict[str, Any]) -> List[str]:
        """Prepare memory-related methods if memory is enabled."""
        memory_config = self._prepare_memory_config(spec_data)

        if not memory_config["enabled"]:
            return []

        return [
            '''
    def get_memory(self) -> str:
        """Get memory for the current context.

        This is a stub method that should be implemented by the developer.
        The memory format and retrieval mechanism are not prescribed by OAS.

        Returns:
            str: Memory string in the format specified by the spec
        """
        return ""  # Implement your memory retrieval logic here
'''
        ]

    def _prepare_embedded_config(self, spec_data: Dict[str, Any]) -> str:
        """Prepare embedded configuration using proper serialization."""
        from .generators import (
            get_logging_config,
        )  # Import here to avoid circular imports

        logging_config = get_logging_config(spec_data)
        intelligence_config = spec_data.get("intelligence", {})

        # Build the complete config dict
        config_dict = {
            "logging": {
                "enabled": logging_config["enabled"],
                "level": logging_config["level"],
                "format_style": logging_config["format_style"],
                "include_timestamp": logging_config["include_timestamp"],
                "log_file": logging_config["log_file"],
                "env_overrides": logging_config["env_overrides"],
            },
            "intelligence": {
                "engine": intelligence_config.get("engine", "openai"),
                "model": intelligence_config.get("model", "gpt-4"),
                "endpoint": intelligence_config.get(
                    "endpoint", "https://api.openai.com/v1"
                ),
                "config": {
                    "temperature": intelligence_config.get("config", {}).get(
                        "temperature", 0.7
                    ),
                    "max_tokens": intelligence_config.get("config", {}).get(
                        "max_tokens", 1000
                    ),
                },
            },
        }

        # Use proper serialization instead of manual string concatenation
        config_code = self.serializer.dict_to_python_code(config_dict, indent=2)
        return f"self.config = {config_code}"

    def _prepare_setup_logging_method(self) -> str:
        """Prepare setup_logging method."""
        return '''
    def setup_logging(self):
        """Configure DACP logging from YAML configuration."""
        logging_config = self.config.get('logging', {})

        if not logging_config.get('enabled', True):
            return

        # Process environment variable overrides
        env_overrides = logging_config.get('env_overrides', {})

        level = logging_config.get('level', 'INFO')
        if 'level' in env_overrides:
            level = os.getenv(env_overrides['level'], level)

        format_style = logging_config.get('format_style', 'emoji')
        if 'format_style' in env_overrides:
            format_style = os.getenv(env_overrides['format_style'], format_style)

        log_file = logging_config.get('log_file')
        if 'log_file' in env_overrides:
            log_file = os.getenv(env_overrides['log_file'], log_file)

        # Create log directory if needed
        if log_file:
            from pathlib import Path
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        # Configure DACP logging
        dacp.setup_dacp_logging(
            level=level,
            format_style=format_style,
            include_timestamp=logging_config.get('include_timestamp', True),
            log_file=log_file
        )
'''

    def _prepare_handle_message_method(self) -> str:
        """Prepare handle_message method."""
        return '''
    def handle_message(self, message: dict) -> dict:
        """
        Handles incoming messages from the orchestrator.
        Processes messages based on the task specified and routes to appropriate agent methods.
        """
        task = message.get("task")
        if not task:
            return {"error": "Missing required field: task"}

        # Map task names to method names (replace hyphens with underscores)
        method_name = task.replace("-", "_")

        # Check if the method exists on this agent
        if not hasattr(self, method_name):
            return {"error": f"Unknown task: {task}"}

        try:
            # Get the method and extract its parameters (excluding 'self')
            method = getattr(self, method_name)

            # Call the method with the message parameters (excluding 'task')
            method_params = {k: v for k, v in message.items() if k != "task"}
            result = method(**method_params)

            # Handle both Pydantic models and dictionaries
            if hasattr(result, 'model_dump'):
                return result.model_dump()
            else:
                return result

        except TypeError as e:
            return {"error": f"Invalid parameters for task {task}: {str(e)}"}
        except Exception as e:
            return {"error": f"Error executing task {task}: {str(e)}"}
'''

    def _prepare_custom_router_loader(
        self, spec_data: Dict[str, Any], config: Dict[str, Any]
    ) -> str:
        """Prepare custom router loader if needed."""
        engine = spec_data.get("intelligence", {}).get("engine", "openai")
        custom_module = spec_data.get("intelligence", {}).get("module", None)

        if engine != "custom" or not custom_module:
            return ""

        return f'''
def load_custom_llm_router(endpoint, model, config):
    """Dynamically load a custom LLM router from the specified module and class."""
    module_path, class_name = "{custom_module}".rsplit('.', 1)
    module = importlib.import_module(module_path)
    CustomLLMRouter = getattr(module, class_name)
    router = CustomLLMRouter(endpoint, model, config)
    if not hasattr(router, 'run'):
        raise AttributeError("Custom LLM router must have a 'run' method")
    return router
'''

    def _prepare_custom_router_init(
        self, spec_data: Dict[str, Any], config: Dict[str, Any]
    ) -> str:
        """Prepare custom router initialization if needed."""
        engine = spec_data.get("intelligence", {}).get("engine", "openai")
        custom_module = spec_data.get("intelligence", {}).get("module", None)

        if engine != "custom" or not custom_module:
            return ""

        return f'self.router = load_custom_llm_router("{config["endpoint"]}", "{config["model"]}", {{}})  # {custom_module}'

    def _prepare_example_task_code(self, spec_data: Dict[str, Any]) -> str:
        """Prepare example task execution code."""
        tasks = spec_data.get("tasks", {})
        if not tasks:
            return ""

        # Prioritize multi-step tasks for examples
        multi_step_tasks = [
            name
            for name, task_def in tasks.items()
            if task_def.get("multi_step", False)
        ]

        if multi_step_tasks:
            return self._generate_example_for_task(
                multi_step_tasks[0], tasks[multi_step_tasks[0]]
            )
        else:
            # Fall back to the first regular task
            first_task_name = next(iter(tasks))
            return self._generate_example_for_task(
                first_task_name, tasks[first_task_name]
            )

    def _generate_example_for_task(
        self, task_name: str, task_def: Dict[str, Any]
    ) -> str:
        """Generate example code for a specific task."""
        from .generators import (
            _generate_input_params,
        )  # Import here to avoid circular imports

        # Get input parameters
        input_params = [
            param.split(":")[0]
            for param in _generate_input_params(task_def)
            if param != "memory_summary: str = ''"
        ]

        if "tool" in task_def:
            # For tool tasks, provide safe example values
            tool_id = task_def["tool"]
            if tool_id == "file_writer":
                example_params = (
                    'file_path="./output/example.txt", content="Hello World!"'
                )
            else:
                # For other tools, use generic example values
                input_props = task_def.get("input", {}).get("properties", {})
                example_params = ", ".join(f'{k}="example_{k}"' for k in input_props)
        elif input_params:
            example_params = ", ".join(f'{k}="example_{k}"' for k in input_params)
        else:
            example_params = ""

        task_type = "multi-step" if task_def.get("multi_step", False) else task_name

        if example_params:
            return f"""
    # Example usage with {task_type} task: {task_name}
    result = agent.{task_name.replace("-", "_")}({example_params})
    # Handle both Pydantic models and dictionaries
    if hasattr(result, 'model_dump'):
        print(json.dumps(result.model_dump(), indent=2))
    else:
        print(json.dumps(result, indent=2))"""
        else:
            return f"""
    # Example usage with {task_type} task: {task_name}
    result = agent.{task_name.replace("-", "_")}()
    # Handle both Pydantic models and dictionaries
    if hasattr(result, 'model_dump'):
        print(json.dumps(result.model_dump(), indent=2))
    else:
        print(json.dumps(result, indent=2))"""
