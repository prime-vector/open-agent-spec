"""File generation functions for Open Agent Spec."""

import logging
from pathlib import Path
from typing import Any, Dict, List


log = logging.getLogger("oas")


def get_agent_info(spec_data: Dict[str, Any]) -> Dict[str, str]:
    """Get agent info from either old or new spec format."""
    # Try new format first
    agent = spec_data.get("agent", {})
    if agent:
        return {
            "name": agent.get("name", ""),
            "description": agent.get("description", ""),
        }

    # Fall back to old format
    info = spec_data.get("info", {})
    return {"name": info.get("name", ""), "description": info.get("description", "")}


def get_memory_config(spec_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get memory configuration from spec."""
    memory = spec_data.get("memory", {})
    return {
        "enabled": memory.get("enabled", False),
        "format": memory.get("format", "string"),
        "usage": memory.get("usage", "prompt-append"),
        "required": memory.get("required", False),
        "description": memory.get("description", ""),
    }


def to_pascal_case(name: str) -> str:
    """Convert snake_case to PascalCase."""
    return "".join(word.capitalize() for word in name.split("_"))


def _generate_input_params(task_def: Dict[str, Any]) -> List[str]:
    """Generate input parameters for a task function."""
    input_params = []
    for param_name, param_def in (
        task_def.get("input", {}).get("properties", {}).items()
    ):
        param_type = map_type_to_python(param_def.get("type", "string"))
        input_params.append(f"{param_name}: {param_type}")
    input_params.append("memory_summary: str = ''")
    return input_params


def _generate_function_docstring(
    task_name: str, task_def: Dict[str, Any], output_type: str
) -> str:
    """Generate docstring for a task function."""
    return f'''"""Process {task_name} task.

    Args:
{chr(10).join(f"        {param_name}: {param_type}" for param_name, param_type in task_def.get("input", {}).get("properties", {}).items())}
        memory_summary: Optional memory context for the task

    Returns:
        {output_type}
    """'''


def _generate_contract_data(
    spec_data: Dict[str, Any],
    task_def: Dict[str, Any],
    agent_name: str,
    memory_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate behavioural contract data from spec."""
    behavioural_section = spec_data.get("behavioural_contract", {})

    # Start with minimal defaults
    contract_data = {
        "version": behavioural_section.get("version", "0.1.2"),
        "description": behavioural_section.get(
            "description", task_def.get("description", "")
        ),
        "role": behavioural_section.get("role", agent_name),
    }

    # Only add behavioural_flags if specified
    if behavioural_section.get("behavioural_flags"):
        contract_data["behavioural_flags"] = behavioural_section["behavioural_flags"]

    # Handle response_contract - be flexible about structure
    response_contract = behavioural_section.get("response_contract", {})
    if response_contract:
        # If response_contract is provided, use it as-is
        contract_data["response_contract"] = response_contract
    else:
        # Only generate default response_contract if output schema exists
        output_schema = task_def.get("output", {})
        if output_schema and output_schema.get("properties"):
            contract_data["response_contract"] = {
                "output_format": {
                    "type": "object",
                    "required_fields": list(output_schema.get("properties", {}).keys()),
                    "on_failure": {
                        "action": "return_empty",
                        "fallback": {
                            k: (
                                []
                                if isinstance(v, dict) and v.get("type") == "array"
                                else ""
                            )
                            for k, v in output_schema.get("properties", {}).items()
                        },
                        "default_value": {
                            k: (
                                []
                                if isinstance(v, dict) and v.get("type") == "array"
                                else ""
                            )
                            for k, v in output_schema.get("properties", {}).items()
                        },
                    },
                },
                "schema": {
                    "type": "object",
                    "properties": output_schema.get("properties", {}),
                },
            }

    # Only add memory_config if memory is enabled
    if memory_config.get("enabled", False):
        contract_data["memory_config"] = memory_config

    # Only add policy if specified
    if behavioural_section.get("policy"):
        contract_data["policy"] = behavioural_section["policy"]

    # Only add teardown_policy if specified
    if behavioural_section.get("teardown_policy"):
        contract_data["teardown_policy"] = behavioural_section["teardown_policy"]

    return contract_data


def _generate_pydantic_model(
    name: str, schema: Dict[str, Any], is_root: bool = True
) -> str:
    """Generate a Pydantic model from a JSON schema.

    Args:
        name: The name of the model
        schema: The JSON schema to convert
        is_root: Whether this is the root model (affects class inheritance)

    Returns:
        String containing the generated Pydantic model code
    """
    if not schema.get("properties"):
        return ""

    model_code = []
    nested_models = []

    # First, generate nested models
    for field_name, field_schema in schema.get("properties", {}).items():
        # Handle nested objects
        if field_schema.get("type") == "object" and field_schema.get("properties"):
            nested_name = f"{name}{field_name.title()}"
            nested_model = _generate_pydantic_model(nested_name, field_schema, False)
            if nested_model:
                nested_models.append(nested_model)

        # Handle arrays of objects
        elif (
            field_schema.get("type") == "array"
            and field_schema.get("items", {}).get("type") == "object"
        ):
            nested_name = f"{name}{field_name.title()}Item"
            nested_model = _generate_pydantic_model(
                nested_name, field_schema["items"], False
            )
            if nested_model:
                nested_models.append(nested_model)

    # Then generate the main model
    if is_root:
        model_code.append(f"class {name}(BaseModel):")
    else:
        model_code.append(
            f"class {name}(BaseModel):"
        )  # Always use BaseModel for nested models

    # Add field definitions
    for field_name, field_schema in schema.get("properties", {}).items():
        field_type = _get_pydantic_type(field_schema, name, field_name)
        description = field_schema.get("description", "")

        # Handle required fields
        is_required = field_name in schema.get("required", [])
        if not is_required:
            field_type = f"Optional[{field_type}] = None"

        # Add field with description
        if description:
            model_code.append(f'    """{description}"""')
        model_code.append(f"    {field_name}: {field_type}")

    # Combine nested models and main model
    return "\n".join(nested_models + model_code)


def _get_pydantic_type(
    schema: Dict[str, Any], parent_name: str, field_name: str
) -> str:
    """Convert JSON schema type to Pydantic type."""
    schema_type = schema.get("type")

    if schema_type == "string":
        return "str"
    elif schema_type == "integer":
        return "int"
    elif schema_type == "number":
        return "float"
    elif schema_type == "boolean":
        return "bool"
    elif schema_type == "array":
        items = schema.get("items", {})
        if items.get("type") == "object":
            # For array of objects, use the nested model type
            return f"List[{parent_name}{field_name.title()}Item]"
        else:
            item_type = _get_pydantic_type(items, parent_name, field_name)
            return f"List[{item_type}]"
    elif schema_type == "object":
        # For nested objects, use the nested model type
        return f"{parent_name}{field_name.title()}"
    else:
        return "Any"


def generate_models(output: Path, spec_data: Dict[str, Any]) -> None:
    """Generate models.py file with Pydantic models for task outputs."""
    if (output / "models.py").exists():
        log.warning("models.py already exists and will be overwritten")

    tasks = spec_data.get("tasks", {})
    if not tasks:
        log.warning("No tasks defined in spec file")
        return

    # Generate imports
    model_code = [
        "from typing import Any, Dict, List, Optional",
        "from pydantic import BaseModel",
        "",
    ]

    # Generate models for each task
    for task_name, task_def in tasks.items():
        if "output" in task_def:
            model_name = f"{task_name.replace('-', '_').title()}Output"
            model_code.append(_generate_pydantic_model(model_name, task_def["output"]))
            model_code.append("")  # Add blank line between models

    # Write the file
    (output / "models.py").write_text("\n".join(model_code))
    log.info("models.py created")


def _generate_llm_output_parser(task_name: str, output_schema: Dict[str, Any]) -> str:
    """Generate a function for parsing LLM output into the task's model."""
    model_name = f"{task_name.replace('-', '_').title()}Output"

    return f'''def parse_llm_output(response: str) -> {model_name}:
    """Parse LLM response into {model_name}.

    Args:
        response: Raw response from the LLM

    Returns:
        Parsed and validated {model_name} instance

    Raises:
        ValueError: If the response cannot be parsed as JSON or doesn't match the schema
    """
    try:
        # Try to find JSON in the response
        json_start = response.find("{{")
        json_end = response.rfind("}}") + 2
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)

            # Convert to Pydantic model
            return {model_name}(**parsed)

        raise ValueError("No valid JSON found in response")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in response: {{e}}")
    except Exception as e:
        raise ValueError(f"Error parsing response: {{e}}")
'''


def _generate_human_readable_output(schema: Dict[str, Any], indent: int = 0) -> str:
    """Generate a human-readable description of the output schema.

    Args:
        schema: The JSON schema to convert
        indent: Current indentation level

    Returns:
        String containing a human-readable description of the output format
    """
    if not schema.get("properties"):
        return ""

    lines = []
    for field_name, field_schema in schema.get("properties", {}).items():
        field_type = _get_human_readable_type(field_schema)
        description = field_schema.get("description", "")

        # Handle required fields
        is_required = field_name in schema.get("required", [])
        required_str = " (required)" if is_required else " (optional)"

        # Add field with description
        if description:
            lines.append(f"{' ' * indent}- {field_name}{required_str}: {field_type}")
            lines.append(f"{' ' * (indent + 2)}{description}")
        else:
            lines.append(f"{' ' * indent}- {field_name}{required_str}: {field_type}")

        # Handle nested objects
        if field_schema.get("type") == "object" and field_schema.get("properties"):
            nested_desc = _generate_human_readable_output(field_schema, indent + 2)
            if nested_desc:
                lines.append(nested_desc)

        # Handle arrays of objects
        elif (
            field_schema.get("type") == "array"
            and field_schema.get("items", {}).get("type") == "object"
        ):
            lines.append(f"{' ' * (indent + 2)}Each item contains:")
            nested_desc = _generate_human_readable_output(
                field_schema["items"], indent + 4
            )
            if nested_desc:
                lines.append(nested_desc)

    return "\n".join(lines)


def _get_human_readable_type(schema: Dict[str, Any]) -> str:
    """Convert JSON schema type to human-readable type."""
    schema_type = schema.get("type")

    if schema_type == "string":
        return "string"
    elif schema_type == "integer":
        return "integer"
    elif schema_type == "number":
        return "number"
    elif schema_type == "boolean":
        return "boolean"
    elif schema_type == "array":
        items = schema.get("items", {})
        if items.get("type") == "object":
            return "array of objects"
        else:
            item_type = _get_human_readable_type(items)
            return f"array of {item_type}s"
    elif schema_type == "object":
        return "object"
    else:
        return "any"


def _generate_task_function(
    task_name: str,
    task_def: Dict[str, Any],
    spec_data: Dict[str, Any],
    agent_name: str,
    memory_config: Dict[str, Any],
    config: Dict[str, Any],
) -> str:
    """Generate a single task function."""
    func_name = task_name.replace("-", "_")
    input_params = _generate_input_params(task_def)
    output_type = f"{task_name.replace('-', '_').title()}Output"
    docstring = _generate_function_docstring(task_name, task_def, output_type)
    contract_data = _generate_contract_data(
        spec_data, task_def, agent_name, memory_config
    )

    # Create input dict with actual parameter values
    input_dict = {}
    for param in input_params:
        if param != "memory_summary: str = ''":
            param_name = param.split(":")[0]
            input_dict[param_name] = param_name

    # Add LLM output parser if this is an LLM-based agent
    llm_parser = ""
    if config.get("model"):  # If model is specified, this is an LLM agent
        llm_parser = _generate_llm_output_parser(task_name, task_def.get("output", {}))

    # Determine client usage based on engine
    engine = spec_data.get("intelligence", {}).get("engine", "openai")
    if engine == "openai":
        client_code = f"""    client = openai.OpenAI(
        base_url="{config["endpoint"]}",
        api_key=openai.api_key
    )

    response = client.chat.completions.create(
        model="{config["model"]}",
        messages=[
            {{"role": "system", "content": "You are a professional {agent_name}."}},
            {{"role": "user", "content": prompt}}
        ],
        temperature={config["temperature"]},
        max_tokens={config["max_tokens"]}
    )

    result = response.choices[0].message.content"""
    elif engine == "anthropic":
        client_code = f"""    import anthropic
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    response = client.messages.create(
        model="{config["model"]}",
        max_tokens={config["max_tokens"]},
        temperature={config["temperature"]},
        system="You are a professional {agent_name}.",
        messages=[
            {{"role": "user", "content": prompt}}
        ]
    )

    result = response.content[0].text if response.content else """
    else:
        client_code = f"""    response = openai.ChatCompletion.create(
        model="{config["model"]}",
        messages=[
            {{"role": "system", "content": "You are a professional {agent_name}."}},
            {{"role": "user", "content": prompt}}
        ],
        temperature={config["temperature"]},
        max_tokens={config["max_tokens"]}
    )

    result = response.choices[0].message.content"""

    # Generate prompt rendering with actual parameter values
    prompt_render_params = []
    for param in input_params:
        if param != "memory_summary: str = ''":
            param_name = param.split(":")[0]
            prompt_render_params.append(f"{param_name}={param_name}")

    # Define memory configuration with proper Python boolean values
    memory_config_str = f"""{{
        "enabled": {repr(memory_config["enabled"])},
        "format": "{memory_config["format"]}",
        "usage": "{memory_config["usage"]}",
        "required": {repr(memory_config["required"])},
        "description": "{memory_config["description"]}"
    }}"""
    memory_summary_str = "memory_summary if memory_config['enabled'] else ''"

    # Generate human-readable output description
    output_description = _generate_human_readable_output(task_def.get("output", {}))
    output_description_str = f'"""\n{output_description}\n"""'

    # Format the contract data for the decorator with proper Python values
    def format_value(v):
        if isinstance(v, bool):
            return str(v)
        elif isinstance(v, (list, tuple)):
            return f"[{', '.join(format_value(x) for x in v)}]"
        elif isinstance(v, dict):
            items = [f'"{k}": {format_value(v)}' for k, v in v.items()]
            return f"{{{', '.join(items)}}}"
        elif isinstance(v, str):
            return f'"{v}"'
        return str(v)

    contract_str = ",\n    ".join(
        f"{k}={format_value(v)}" for k, v in contract_data.items()
    )

    return f"""
{llm_parser}

@behavioural_contract(
    {contract_str}
)
def {func_name}({", ".join(input_params)}) -> {output_type}:
    {docstring}
    # Define memory configuration
    memory_config = {memory_config_str}

    # Define output format description
    output_format = {output_description_str}

    # Load and render the prompt template
    prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
    env = Environment(loader=FileSystemLoader([".", prompts_dir]))
    try:
        template = env.get_template(f"{func_name}.jinja2")
    except FileNotFoundError:
        log.warning(f"Task-specific prompt template not found, using default template")
        template = env.get_template("agent_prompt.jinja2")

    # Create input dictionary for template
    input_dict = {{
        {", ".join(f'"{param.split(":")[0]}": {param.split(":")[0]}' for param in input_params if param != "memory_summary: str = ''")}
    }}

    # Render the prompt with all necessary context
    prompt = template.render(
        input=input_dict,
        memory_summary={memory_summary_str},
        output_format=output_format,
        memory_config=memory_config
    )

{client_code}
    return parse_llm_output(result)
"""


def generate_agent_code(
    output: Path, spec_data: Dict[str, Any], agent_name: str, class_name: str
) -> None:
    """Generate the agent.py file."""
    if (output / "agent.py").exists():
        log.warning("agent.py already exists and will be overwritten")

    tasks = spec_data.get("tasks", {})
    if not tasks:
        log.warning("No tasks defined in spec file")
        return

    config = {
        "endpoint": spec_data.get("intelligence", {}).get(
            "endpoint", "https://api.openai.com/v1"
        ),
        "model": spec_data.get("intelligence", {}).get("model", "gpt-3.5-turbo"),
        "temperature": spec_data.get("intelligence", {})
        .get("config", {})
        .get("temperature", 0.7),
        "max_tokens": spec_data.get("intelligence", {})
        .get("config", {})
        .get("max_tokens", 1000),
    }
    memory_config = get_memory_config(spec_data)

    # Generate task functions and class methods
    task_functions = []
    class_methods = []
    model_definitions = []

    for task_name, task_def in tasks.items():
        # Generate model definition
        model_name = f"{task_name.replace('-', '_').title()}Output"
        model_def = _generate_pydantic_model(model_name, task_def.get("output", {}))
        if model_def:
            model_definitions.append(model_def)

        # Generate task function
        task_functions.append(
            _generate_task_function(
                task_name, task_def, spec_data, agent_name, memory_config, config
            )
        )

        # Generate corresponding class method
        input_params_without_memory = [
            param.split(":")[0]
            for param in _generate_input_params(task_def)
            if param != "memory_summary: str = ''"
        ]
        class_methods.append(
            f'''
    def {task_name.replace("-", "_")}(self, {", ".join(input_params_without_memory)}) -> {model_name}:
        """Process {task_name} task."""
        memory_summary = self.get_memory() if hasattr(self, 'get_memory') else ""
        return {task_name.replace("-", "_")}({", ".join(input_params_without_memory)}, memory_summary=memory_summary)
'''
        )

    # Generate memory-related methods if memory is enabled
    memory_methods = []
    if memory_config["enabled"]:
        memory_methods.append(
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
        )

    # Generate example task execution code
    example_task_code = ""
    if tasks:
        first_task_name = next(iter(tasks))
        first_task_def = tasks[first_task_name]
        input_props = first_task_def.get("input", {}).get("properties", {})
        example_params = ", ".join(f'{k}="example_{k}"' for k in input_props)
        example_task_code = f"""
    # Example usage with {first_task_name} task
    result = agent.{first_task_name.replace("-", "_")}({example_params})
    # Handle both Pydantic models and dictionaries
    if hasattr(result, 'model_dump'):
        print(json.dumps(result.model_dump(), indent=2))
    else:
        print(json.dumps(result, indent=2))"""

    # Generate the complete agent code
    engine = spec_data.get("intelligence", {}).get("engine", "openai")
    imports = ["from typing import Dict, Any, List, Optional"]

    if engine == "openai":
        imports.append("import openai")
    elif engine == "anthropic":
        imports.append("import anthropic")
    else:
        imports.append("import openai")  # Default fallback

    imports.extend(
        [
            "import json",
            "import logging",
            "import os",
            "from dotenv import load_dotenv",
            "from behavioural_contracts import behavioural_contract",
            "from jinja2 import Environment, FileSystemLoader",
            "from pydantic import BaseModel",
        ]
    )

    agent_code = f"""{chr(10).join(imports)}

load_dotenv()

log = logging.getLogger(__name__)

ROLE = "{agent_name.title()}"

# Generate output models
{chr(10).join(model_definitions)}

{chr(10).join(task_functions)}

class {class_name}:
    def __init__(self, api_key: str | None = None):
        self.model = "{config["model"]}"
        if api_key:
            openai.api_key = api_key

{chr(10).join(class_methods)}
{chr(10).join(memory_methods)}

def main():
    agent = {class_name}()
{example_task_code}

if __name__ == "__main__":
    main()
"""
    (output / "agent.py").write_text(agent_code)
    log.info("agent.py created")
    log.debug(f"Agent class name generated: {class_name}")


def map_type_to_python(t):
    return {
        "string": "str",
        "number": "float",
        "integer": "int",
        "boolean": "bool",
        "object": "Dict[str, Any]",
        "array": "List[Any]",
    }.get(t, "Any")


def _generate_task_docs(tasks: Dict[str, Any]) -> List[str]:
    """Generate documentation for tasks."""
    task_docs = []
    for task_name in tasks.keys():
        task_def = tasks[task_name]
        task_docs.append(f"### {task_name.title()}\n")
        task_docs.append(f"{task_def.get('description', '')}\n")

        if task_def.get("input"):
            task_docs.append("#### Input:")
            for param_name, param_type in task_def.get("input", {}).items():
                task_docs.append(f"- {param_name}: {param_type}")
            task_docs.append("")

        if task_def.get("output"):
            task_docs.append("#### Output:")
            for param_name, param_type in task_def.get("output", {}).items():
                task_docs.append(f"- {param_name}: {param_type}")
            task_docs.append("")
    return task_docs


def _generate_memory_docs(memory_config: Dict[str, Any]) -> List[str]:
    """Generate documentation for memory configuration."""
    memory_docs = []
    if memory_config["enabled"]:
        memory_docs.append("## Memory Support\n")
        memory_docs.append(f"{memory_config['description']}\n")
        memory_docs.append("### Configuration\n")
        memory_docs.append(f"- Format: {memory_config['format']}\n")
        memory_docs.append(f"- Usage: {memory_config['usage']}\n")
        memory_docs.append(f"- Required: {memory_config['required']}\n")
        memory_docs.append(
            "\nTo implement memory support, override the `get_memory()` method in the agent class.\n"
        )
    return memory_docs


def _generate_behavioural_docs(behavioural_contract: Dict[str, Any]) -> List[str]:
    """Generate documentation for behavioural contract."""
    behavioural_docs = []
    behavioural_docs.append("## Behavioural Contract\n\n")
    behavioural_docs.append(
        "This agent is governed by the following behavioural contract policy:\n\n"
    )

    if "pii" in behavioural_contract:
        behavioural_docs.append(f"- PII: {behavioural_contract['pii']}\n")

    if "compliance_tags" in behavioural_contract:
        behavioural_docs.append(
            f"- Compliance Tags: {', '.join(behavioural_contract['compliance_tags'])}\n"
        )

    if "allowed_tools" in behavioural_contract:
        behavioural_docs.append(
            f"- Allowed Tools: {', '.join(behavioural_contract['allowed_tools'])}\n"
        )

    behavioural_docs.append(
        "\nRefer to `behavioural_contracts` for enforcement logic.\n"
    )
    return behavioural_docs


def _generate_example_usage(agent_info: Dict[str, str], tasks: Dict[str, Any]) -> str:
    """Generate example usage code."""
    first_task_name = next(iter(tasks.keys()), "")
    if not first_task_name:
        return ""

    return f"""```python
from agent import {to_pascal_case(agent_info["name"])}

agent = {to_pascal_case(agent_info["name"])}()
# Example usage
task_name = "{first_task_name}"
if task_name:
    result = getattr(agent, task_name.replace("-", "_"))(
        {", ".join(f'{k}="example_{k}"' for k in tasks[first_task_name].get("input", {}))}
    )
    print(result)
```"""


def generate_readme(output: Path, spec_data: Dict[str, Any]) -> None:
    """Generate the README.md file."""
    if (output / "README.md").exists():
        log.warning("README.md already exists and will be overwritten")

    agent_info = get_agent_info(spec_data)
    memory_config = get_memory_config(spec_data)
    tasks = spec_data.get("tasks", {})

    task_docs = _generate_task_docs(tasks)
    memory_docs = _generate_memory_docs(memory_config)
    behavioural_docs = (
        _generate_behavioural_docs(spec_data["behavioural_contract"])
        if "behavioural_contract" in spec_data
        else []
    )
    example_usage = _generate_example_usage(agent_info, tasks)

    readme_content = f"""# {agent_info["name"].title().replace("-", " ")}

{agent_info["description"]}

## Usage

```bash
pip install -r requirements.txt
cp .env.example .env
python agent.py
```

## Tasks

{chr(10).join(task_docs)}
{chr(10).join(memory_docs)}
{chr(10).join(behavioural_docs)}

## Example Usage

{example_usage}
"""
    (output / "README.md").write_text(readme_content)
    log.info("README.md created")


def generate_requirements(output: Path, spec_data: Dict[str, Any]) -> None:
    """Generate the requirements.txt file."""
    if (output / "requirements.txt").exists():
        log.warning("requirements.txt already exists and will be overwritten")

    engine = spec_data.get("intelligence", {}).get("engine", "openai")

    requirements = []
    if engine == "openai":
        requirements.append("openai>=1.0.0")
    elif engine == "anthropic":
        requirements.append("anthropic>=0.18.0")
    else:
        requirements.append("openai>=1.0.0")  # Default fallback

    requirements.extend(
        [
            "# Note: During development, install with: pip install -r requirements.txt --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/",
            "behavioural-contracts>=0.1.0",
            "python-dotenv>=0.19.0",
            "pydantic>=2.0.0",
            "jinja2>=3.0.0",
        ]
    )

    (output / "requirements.txt").write_text("\n".join(requirements) + "\n")
    log.info("requirements.txt created")


def generate_env_example(output: Path, spec_data: Dict[str, Any]) -> None:
    """Generate the .env.example file."""
    if (output / ".env.example").exists():
        log.warning(".env.example already exists and will be overwritten")

    engine = spec_data.get("intelligence", {}).get("engine", "openai")

    if engine == "anthropic":
        env_content = "ANTHROPIC_API_KEY=your-api-key-here\n"
    else:
        env_content = "OPENAI_API_KEY=your-api-key-here\n"

    (output / ".env.example").write_text(env_content)
    log.info(".env.example created")


def generate_prompt_template(output: Path, spec_data: Dict[str, Any]) -> None:
    """Generate the prompt template file."""
    prompts_dir = output / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    # Generate task-specific templates
    for task_name, task_def in spec_data.get("tasks", {}).items():
        template_name = f"{task_name}.jinja2"
        if (prompts_dir / template_name).exists():
            log.warning(f"{template_name} already exists and will be overwritten")

        # Get the output schema
        output_schema = task_def.get("output", {})
        human_readable_output = _generate_human_readable_output(output_schema)

        # Prepare example JSON for the output
        # Use the output schema's properties to generate an example
        example_json_lines = []
        if output_schema.get("properties"):
            example_json_lines.append("{")
            for i, (k, v) in enumerate(output_schema["properties"].items()):
                comma = "," if i < len(output_schema["properties"]) - 1 else ""
                # For the example, use a meaningful value based on the field name
                if k == "response":
                    # If there's a 'name' input, use it in the response
                    input_props = task_def.get("input", {}).get("properties", {})
                    if "name" in input_props:
                        example_json_lines.append(
                            f'  "{k}": "Hello {{{{ input.name }}}}!"{comma}'
                        )
                    else:
                        example_json_lines.append(
                            f'  "{k}": "Your response here"{comma}'
                        )
                else:
                    example_json_lines.append(f'  "{k}": "{{{{ input.{k} }}}}"{comma}')
            example_json_lines.append("}")
        else:
            example_json_lines.append("{}")
        example_json = "\n".join(example_json_lines)

        # Handle both old and new prompt formats
        if "prompt" in spec_data and "template" in spec_data["prompt"]:
            # Old format - use the template directly
            prompt_content = spec_data["prompt"]["template"]
        elif "prompts" in spec_data and (
            spec_data["prompts"].get("system") or spec_data["prompts"].get("user")
        ):
            # New format - use system and user prompts
            prompts = spec_data.get("prompts", {})
            system_prompt = prompts.get(
                "system",
                "You are a professional AI agent designed to process tasks according to the Open Agent Spec.\n\n",
            )
            user_prompt = prompts.get("user", "")

            # Combine system and user prompts
            prompt_content = system_prompt
            if user_prompt:
                prompt_content += user_prompt

            # Add memory context if not already present
            if "{% if memory_summary %}" not in prompt_content:
                prompt_content = (
                    "{% if memory_summary %}\n"
                    "--- MEMORY CONTEXT ---\n"
                    "{{ memory_summary }}\n"
                    "------------------------\n"
                    "{% endif %}\n\n"
                ) + prompt_content
        else:
            # Default format - use system and user prompts
            prompts = spec_data.get("prompts", {})
            prompt_content = (
                prompts.get(
                    "system",
                    "You are a professional AI agent designed to process tasks according to the Open Agent Spec.\n\n",
                )
                + "{% if memory_summary %}\n"
                + "--- MEMORY CONTEXT ---\n"
                + "{{ memory_summary }}\n"
                + "------------------------\n"
                + "{% endif %}\n\n"
                + "TASK:\n"
                + "Process the following task:\n\n"
                + "{% for key, value in input.items() %}\n"
                + "{{ key }}: {{ value }}\n"
                + "{% endfor %}\n\n"
                + "INSTRUCTIONS:\n"
                + "1. Review the input data carefully\n"
                + "2. Consider all relevant factors\n"
                + "{% if memory_summary %}\n"
                + "3. Take into account the provided memory context\n"
                + "{% endif %}\n"
                + "4. Provide a clear, actionable response\n"
                + "5. Explain your reasoning in detail\n\n"
                + "OUTPUT FORMAT:\n"
                + "Your response should include the following fields:\n"
                + f"{human_readable_output}\n\n"
            )

        # Always append the JSON schema instruction and example
        prompt_content += (
            "\nRespond ONLY with a JSON object like this (replace the value appropriately):\n"
            f"{example_json}\n"
        )

        (prompts_dir / template_name).write_text(prompt_content)
        log.info(f"Created prompt template: {template_name}")

    # Generate default template
    default_template = prompts_dir / "agent_prompt.jinja2"
    if default_template.exists():
        log.warning("agent_prompt.jinja2 already exists and will be overwritten")

    default_content = """You are a professional AI agent designed to process tasks according to the Open Agent Spec.

{% if memory_summary %}
--- MEMORY CONTEXT ---
{{ memory_summary }}
------------------------
{% endif %}

TASK:
Process the following task:

{% for key, value in input.items() %}
{{ key }}: {{ value }}
{% endfor %}

INSTRUCTIONS:
1. Review the input data carefully
2. Consider all relevant factors
{% if memory_summary %}
3. Take into account the provided memory context
{% endif %}
4. Provide a clear, actionable response
5. Explain your reasoning in detail

OUTPUT FORMAT:
Your response should include the following fields:
{{ output_format }}

CONSTRAINTS:
- Be clear and specific
- Focus on actionable insights
- Maintain professional objectivity
{% if memory_summary and memory_config.required %}
- Must reference and incorporate memory context
{% endif %}"""

    default_template.write_text(default_content)
    log.info("Created default prompt template: agent_prompt.jinja2")
