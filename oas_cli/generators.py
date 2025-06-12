"""File generation functions for Open Agent Spec."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from behavioural_contracts import generate_contract  # type: ignore

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
        "enabled": str(memory.get("enabled", False)).lower(),
        "format": str(memory.get("format", "string")),
        "usage": str(memory.get("usage", "prompt-append")),
        "required": str(memory.get("required", False)).lower(),
        "description": str(memory.get("description", "")),
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
    """Generate behavioural contract data."""
    behavioural_section = spec_data.get("behavioural_contract", {})
    if not behavioural_section:
        return {
            "version": "1.1",
            "description": task_def.get("description", ""),
            "role": agent_name,
            "memory": memory_config,
        }

    if "memory" not in behavioural_section:
        behavioural_section["memory"] = memory_config
    if "role" not in behavioural_section:
        behavioural_section["role"] = agent_name
    if "description" not in behavioural_section:
        behavioural_section["description"] = task_def.get("description", "")
    return behavioural_section


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
    output_type = "Dict[str, Any]"
    docstring = _generate_function_docstring(task_name, task_def, output_type)
    output_json = json.dumps(task_def.get("output", {}))
    contract_data = _generate_contract_data(
        spec_data, task_def, agent_name, memory_config
    )
    contract_json = generate_contract(contract_data)

    input_dict = {k: k for k in task_def.get("input", {}).keys()}
    input_dict_str = json.dumps(input_dict, indent=12)

    return f"""
@behavioural_contract({contract_json})
def {func_name}({", ".join(input_params)}) -> {output_type}:
    {docstring}
    # Define task_def for this function
    task_def = {{
        "output": {output_json}
    }}

    # Load and render the prompt template
    env = Environment(loader=FileSystemLoader("prompts"))
    try:
        template = env.get_template("{func_name}.jinja2")
    except FileNotFoundError:
        template = env.get_template("agent_prompt.jinja2")

    # Render the prompt with all necessary context
    prompt = template.render(
        input={input_dict_str},
        memory_summary=memory_summary,
        indicators_summary="",  # TODO: Implement indicators if needed
        output=task_def["output"],
        memory_config={memory_config}
    )

    client = openai.OpenAI(
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

    result = response.choices[0].message.content
    return _parse_response(result, task_def.get("output", {{}}))
"""


def _try_parse_json(result: str, output_fields: List[str]) -> Dict[str, Any] | None:
    """Try to parse JSON from the response."""
    try:
        json_start = result.find("{{")
        json_end = result.rfind("}}") + 2
        if json_start >= 0 and json_end > json_start:
            json_str = result[json_start:json_end]
            parsed = json.loads(json_str)
            if all(key in parsed for key in output_fields):
                return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _parse_line_based_response(
    lines: List[str], output_fields: List[str]
) -> Dict[str, Any]:
    """Parse response using line-based format."""
    output_dict = {}
    current_key = None
    current_value = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        for key in output_fields:
            if line.startswith(key + ":"):
                if current_key and current_value:
                    output_dict[current_key] = " ".join(current_value).strip()
                current_key = key
                # fmt: off
                current_value = [line[len(key) + 1:].strip()]
                # fmt: on
                break
        else:
            if current_key:
                current_value.append(line)

    if current_key and current_value:
        output_dict[current_key] = " ".join(current_value).strip()

    return output_dict


def _parse_response(result: str, output_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the response into the expected output format."""
    output_fields = list(output_schema.keys())

    # Try JSON parsing first
    if parsed_json := _try_parse_json(result, output_fields):
        return parsed_json

    # Fall back to line-based parsing
    lines = result.strip().split("\\n")
    output_dict = _parse_line_based_response(lines, output_fields)

    # Fill in missing fields
    for field in output_fields:
        if field not in output_dict:
            output_dict[field] = ""

    return output_dict


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
        "endpoint": spec_data.get("config", {}).get(
            "endpoint", "https://api.openai.com/v1"
        ),
        "model": spec_data.get("config", {}).get("model", "gpt-3.5-turbo"),
        "temperature": spec_data.get("config", {}).get("temperature", 0.7),
        "max_tokens": spec_data.get("config", {}).get("max_tokens", 1000),
    }
    memory_config = get_memory_config(spec_data)

    # Generate task functions and class methods
    task_functions = []
    class_methods = []

    for task_name, task_def in tasks.items():
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
        class_methods.append(f'''
    def {task_name.replace("-", "_")}(self, {", ".join(input_params_without_memory)}) -> Dict[str, Any]:
        """Process {task_name} task."""
        memory_summary = self.get_memory() if hasattr(self, 'get_memory') else ""
        return {task_name.replace("-", "_")}({", ".join(input_params_without_memory)}, memory_summary=memory_summary)
''')

    # Generate memory-related methods if memory is enabled
    memory_methods = []
    if memory_config["enabled"]:
        memory_methods.append('''
    def get_memory(self) -> str:
        """Get memory for the current context.

        This is a stub method that should be implemented by the developer.
        The memory format and retrieval mechanism are not prescribed by OAS.

        Returns:
            str: Memory string in the format specified by the spec
        """
        return ""  # Implement your memory retrieval logic here
''')

    # Generate the complete agent code
    first_task_name = next(iter(tasks.keys())) if tasks else None
    agent_code = f"""from typing import Dict, Any
import openai
import json
from behavioural_contracts import behavioural_contract
from jinja2 import Environment, FileSystemLoader
from oas_cli.generators import _parse_response

ROLE = "{agent_name.title()}"

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
    # Example usage
    if "{first_task_name}":
        result = getattr(agent, "{first_task_name}".replace("-", "_"))(
            {", ".join(f'{k}="example_{k}"' for k in tasks[first_task_name].get("input", {}).get("properties", {}))}
        )
        print(json.dumps(result, indent=2))
    else:
        print("No tasks defined in the spec file")

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


def generate_requirements(output: Path) -> None:
    """Generate the requirements.txt file."""
    if (output / "requirements.txt").exists():
        log.warning("requirements.txt already exists and will be overwritten")

    requirements = """openai>=1.0.0
# Note: During development, install with: pip install -r requirements.txt --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/
behavioural-contracts>=0.1.0
python-dotenv>=0.19.0
"""
    (output / "requirements.txt").write_text(requirements)
    log.info("requirements.txt created")


def generate_env_example(output: Path) -> None:
    """Generate the .env.example file."""
    if (output / ".env.example").exists():
        log.warning(".env.example already exists and will be overwritten")

    env_content = "OPENAI_API_KEY=your-api-key-here\n"
    (output / ".env.example").write_text(env_content)
    log.info(".env.example created")


def generate_prompt_template(output: Path, spec_data: Dict[str, Any]) -> None:
    """Generate the prompt template file."""
    prompts_dir = output / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    # Generate task-specific templates
    for task_name in spec_data.get("tasks", {}).keys():
        template_name = f"{task_name.replace('-', '_')}.jinja2"
        if (prompts_dir / template_name).exists():
            log.warning(f"{template_name} already exists and will be overwritten")

        # Handle both old and new prompt formats
        if "prompt" in spec_data and "template" in spec_data["prompt"]:
            # Old format - use the template directly
            prompt_content = spec_data["prompt"]["template"]
        else:
            # New format - use system and user prompts
            prompts = spec_data.get("prompts", {})
            if "system" in prompts and "user" in prompts:
                # Merge system and user prompts with memory support
                prompt_content = (
                    "{% if memory_summary %}\n"
                    "--- MEMORY CONTEXT ---\n"
                    "{{ memory_summary }}\n"
                    "------------------------\n"
                    "{% endif %}\n\n"
                    "{% if indicators_summary %}\n"
                    "--- INDICATORS ---\n"
                    "{{ indicators_summary }}\n"
                    "------------------\n"
                    "{% endif %}\n\n"
                    f"{prompts['system']}\n\n"
                    f"{prompts['user']}\n\n"
                    "OUTPUT FORMAT:\n"
                    "Respond **exactly** in this format:\n\n"
                    "{% for key in output.keys() %}\n"
                    "{{ key }}: <value>\n"
                    "{% endfor %}\n\n"
                    "Or as a JSON object:\n"
                    "{\n"
                    "{% for key in output.keys() %}\n"
                    '    "{{ key }}": <value>{% if not loop.last %},{% endif %}\n'
                    "{% endfor %}\n"
                    "}\n\n"
                    "CONSTRAINTS:\n"
                    "- Be clear and specific\n"
                    "- Focus on actionable insights\n"
                    "- Maintain professional objectivity\n"
                    "{% if memory_summary and memory_config.required %}\n"
                    "- Must reference and incorporate memory context\n"
                    "{% endif %}"
                )
            else:
                prompt_content = (
                    "You are a professional AI agent designed to process tasks according to the Open Agent Spec.\n\n"
                    "{% if memory_summary %}\n"
                    "--- MEMORY CONTEXT ---\n"
                    "{{ memory_summary }}\n"
                    "------------------------\n"
                    "{% endif %}\n\n"
                    "{% if indicators_summary %}\n"
                    "--- INDICATORS ---\n"
                    "{{ indicators_summary }}\n"
                    "------------------\n"
                    "{% endif %}\n\n"
                    "TASK:\n"
                    "Process the following task:\n\n"
                    "{% for key, value in input.items() %}\n"
                    "{{ key }}: {{ value }}\n"
                    "{% endfor %}\n\n"
                    "INSTRUCTIONS:\n"
                    "1. Review the input data carefully\n"
                    "2. Consider all relevant factors\n"
                    "{% if memory_summary %}\n"
                    "3. Take into account the provided memory context\n"
                    "{% endif %}\n"
                    "4. Provide a clear, actionable response\n"
                    "5. Explain your reasoning in detail\n\n"
                    "OUTPUT FORMAT:\n"
                    "Respond **exactly** in this format:\n\n"
                    "{% for key in output.keys() %}\n"
                    "{{ key }}: <value>\n"
                    "{% endfor %}\n\n"
                    "Or as a JSON object:\n"
                    "{\n"
                    "{% for key in output.keys() %}\n"
                    '    "{{ key }}": <value>{% if not loop.last %},{% endif %}\n'
                    "{% endfor %}\n"
                    "}\n\n"
                    "CONSTRAINTS:\n"
                    "- Be clear and specific\n"
                    "- Focus on actionable insights\n"
                    "- Maintain professional objectivity\n"
                    "{% if memory_summary and memory_config.required %}\n"
                    "- Must reference and incorporate memory context\n"
                    "{% endif %}"
                )

        (prompts_dir / template_name).write_text(prompt_content)
        log.info(f"{template_name} created")
