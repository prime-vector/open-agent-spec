"""File generation functions for Open Agent Spec."""
from pathlib import Path
from typing import Dict, Any
import logging
import json
import openai
from behavioral_contracts import behavioral_contract
from jinja2 import Environment, FileSystemLoader

log = logging.getLogger("oas")

def get_agent_info(spec_data: Dict[str, Any]) -> Dict[str, str]:
    """Get agent info from either old or new spec format."""
    # Try new format first
    agent = spec_data.get("agent", {})
    if agent:
        return {
            "name": agent.get("name", ""),
            "description": agent.get("description", "")
        }
    
    # Fall back to old format
    info = spec_data.get("info", {})
    return {
        "name": info.get("name", ""),
        "description": info.get("description", "")
    }

def get_memory_config(spec_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get memory configuration from spec."""
    memory = spec_data.get("memory", {})
    return {
        "enabled": memory.get("enabled", False),
        "format": memory.get("format", "string"),
        "usage": memory.get("usage", "prompt-append"),
        "required": memory.get("required", False),
        "description": memory.get("description", "")
    }

def to_pascal_case(name: str) -> str:
    """Convert snake_case to PascalCase."""
    return ''.join(word.capitalize() for word in name.split('_'))

def generate_agent_code(output: Path, spec_data: Dict[str, Any], agent_name: str, class_name: str) -> None:
    """Generate the agent.py file."""
    if (output / "agent.py").exists():
        log.warning("agent.py already exists and will be overwritten")
    
    # Extract task definitions
    tasks = spec_data.get("tasks", {})
    if not tasks:
        log.warning("No tasks defined in spec file")
        return

    # Get config values
    config = spec_data.get("config", {})
    endpoint = config.get("endpoint", "https://api.openai.com/v1")
    model = config.get("model", "gpt-3.5-turbo")
    temperature = config.get("temperature", 0.7)
    max_tokens = config.get("max_tokens", 1000)

    # Get memory config
    memory_config = get_memory_config(spec_data)

    # Generate task functions and class methods
    task_functions = []
    class_methods = []
    
    for task_name, task_def in tasks.items():
        # Convert task name to snake_case for function name
        func_name = task_name.replace("-", "_")
        
        # Generate input parameters
        input_params = []
        for param_name, param_def in task_def.get("input", {}).get("properties", {}).items():
            param_type = map_type_to_python(param_def.get("type", "string"))
            input_params.append(f"{param_name}: {param_type}")
        
        # Add memory_summary parameter
        input_params.append("memory_summary: str = ''")
        
        # Generate return type annotation
        output_type = "Dict[str, Any]"
        if task_def.get("output"):
            output_type = "Dict[str, Any]"  # Could be made more specific based on output schema
        
        # Generate function docstring
        docstring = f'''"""Process {task_name} task.

    Args:
{chr(10).join(f"        {param_name}: {param_type}" for param_name, param_type in task_def.get("input", {}).items())}
        memory_summary: Optional memory context for the task

    Returns:
        {output_type}
    """'''
        
        # Generate function code
        output_json = json.dumps(task_def.get('output', {}))
        
        # Get behavioral contract config
        behavioral_section = spec_data.get("behavioral_contract", {})
        if not behavioral_section:
            behavioral_section = {
                "version": "1.1",
                "description": task_def.get('description', ''),
                "role": agent_name,
                "memory": memory_config
            }
        elif "memory" not in behavioral_section:
            behavioral_section["memory"] = memory_config
            
        contract_json = json.dumps(behavioral_section)
        
        # Generate input dict for template
        input_dict = {k: k for k in task_def.get('input', {}).keys()}
        input_dict_str = json.dumps(input_dict, indent=12)
        
        task_func = f'''
@behavioral_contract({contract_json})
def {func_name}({', '.join(input_params)}) -> {output_type}:
    {docstring}
    # Define task_def for this function
    task_def = {{
        "output": {output_json}
    }}
    
    # Load and render the prompt template
    env = Environment(loader=FileSystemLoader("prompts"))
    try:
        template = env.get_template("{func_name}.jinja2")
    except:
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
        base_url="{endpoint}",
        api_key=openai.api_key
    )

    response = client.chat.completions.create(
        model="{model}",
        messages=[
            {{"role": "system", "content": "You are a professional {agent_name}."}},
            {{"role": "user", "content": prompt}}
        ],
        temperature={temperature},
        max_tokens={max_tokens}
    )
    
    result = response.choices[0].message.content

    # Parse the response into the expected output format
    output_dict = {{}}
    output_fields = list(task_def.get('output', {{}}).keys())
    
    # Try JSON parsing first
    try:
        # Look for JSON block in the response
        json_start = result.find("{{")
        json_end = result.rfind("}}") + 2
        if json_start >= 0 and json_end > json_start:
            json_str = result[json_start:json_end]
            parsed = json.loads(json_str)
            if all(key in parsed for key in output_fields):
                return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Fall back to line-based parsing
    lines = result.strip().split('\\n')
    current_key = None
    current_value = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line starts a new field
        for key in output_fields:
            if line.startswith(key + ":"):
                # Save previous field if exists
                if current_key and current_value:
                    output_dict[current_key] = ' '.join(current_value).strip()
                # Start new field
                current_key = key
                current_value = [line[len(key)+1:].strip()]
                break
        else:
            # If no new field found, append to current value
            if current_key:
                current_value.append(line)
    
    # Save the last field
    if current_key and current_value:
        output_dict[current_key] = ' '.join(current_value).strip()
    
    # Validate all required fields are present
    missing_fields = [field for field in output_fields if field not in output_dict]
    if missing_fields:
        print(f"Warning: Missing output fields: {{missing_fields}}")
        for field in missing_fields:
            output_dict[field] = ""  # Provide empty string for missing fields
    
    return output_dict
'''
        task_functions.append(task_func)
        
        # Generate corresponding class method
        class_method = f'''
    def {func_name}(self, {', '.join(param.split(':')[0] for param in input_params if param != 'memory_summary: str = \'\'')}) -> {output_type}:
        """Process {task_name} task."""
        memory_summary = self.get_memory() if hasattr(self, 'get_memory') else ""
        return {func_name}({', '.join(param.split(':')[0] for param in input_params if param != 'memory_summary: str = \'\'')}, memory_summary=memory_summary)
'''
        class_methods.append(class_method)

    # Generate memory-related methods if memory is enabled
    memory_methods = []
    if memory_config["enabled"]:
        memory_methods.append(f'''
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
    agent_code = f'''from typing import Dict, Any
import openai
import json
from behavioral_contracts import behavioral_contract
from jinja2 import Environment, FileSystemLoader

ROLE = "{agent_name.title()}"

{chr(10).join(task_functions)}

class {class_name}:
    def __init__(self, api_key: str | None = None):
        self.model = "{model}"
        if api_key:
            openai.api_key = api_key

{chr(10).join(class_methods)}
{chr(10).join(memory_methods)}

def main():
    agent = {class_name}()
    # Example usage
    if "{first_task_name}":
        result = getattr(agent, "{first_task_name}".replace("-", "_"))(
            {', '.join(f'{k}="example_{k}"' for k in tasks[first_task_name].get('input', {}))}
        )
        print(json.dumps(result, indent=2))
    else:
        print("No tasks defined in the spec file")

if __name__ == "__main__":
    main()
'''
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
        "array": "List[Any]"
    }.get(t, "Any")

def generate_readme(output: Path, spec_data: Dict[str, Any]) -> None:
    """Generate the README.md file."""
    if (output / "README.md").exists():
        log.warning("README.md already exists and will be overwritten")
    
    # Get agent info from either format
    agent_info = get_agent_info(spec_data)
    
    # Get memory config
    memory_config = get_memory_config(spec_data)
    
    # Generate task documentation
    task_docs = []
    for task_name, task_def in spec_data.get("tasks", {}).items():
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
    
    # Add memory documentation if enabled
    memory_docs = []
    if memory_config["enabled"]:
        memory_docs.append("## Memory Support\n")
        memory_docs.append(f"{memory_config['description']}\n")
        memory_docs.append("### Configuration\n")
        memory_docs.append(f"- Format: {memory_config['format']}\n")
        memory_docs.append(f"- Usage: {memory_config['usage']}\n")
        memory_docs.append(f"- Required: {memory_config['required']}\n")
        memory_docs.append("\nTo implement memory support, override the `get_memory()` method in the agent class.\n")
    
    # Add behavioral contract documentation if defined
    behavioral_docs = []
    if behavioral_contract := spec_data.get("behavioral_contract"):
        behavioral_docs.append("## Behavioral Contract\n\n")
        behavioral_docs.append("This agent is governed by the following behavioral contract policy:\n\n")
        
        # Add PII policy
        if "pii" in behavioral_contract:
            behavioral_docs.append(f"- PII: {behavioral_contract['pii']}\n")
        
        # Add compliance tags
        if "compliance_tags" in behavioral_contract:
            behavioral_docs.append(f"- Compliance Tags: {', '.join(behavioral_contract['compliance_tags'])}\n")
        
        # Add allowed tools
        if "allowed_tools" in behavioral_contract:
            behavioral_docs.append(f"- Allowed Tools: {', '.join(behavioral_contract['allowed_tools'])}\n")
        
        behavioral_docs.append("\nRefer to `behavioral_contracts` for enforcement logic.\n")
    
    readme_content = f"""# {agent_info['name'].title().replace('-', ' ')}

{agent_info['description']}

## Usage

```bash
pip install -r requirements.txt
cp .env.example .env
python agent.py
```

## Tasks

{chr(10).join(task_docs)}
{chr(10).join(memory_docs)}
{chr(10).join(behavioral_docs)}

## Example Usage

```python
from agent import {to_pascal_case(agent_info['name'])}

agent = {to_pascal_case(agent_info['name'])}()
# Example usage
task_name = "{next(iter(spec_data.get('tasks', {}).keys()), '')}"
if task_name:
    result = getattr(agent, task_name.replace("-", "_"))(
        {', '.join(f'{k}="example_{k}"' for k in spec_data['tasks'][task_name].get('input', {}))}
    )
    print(result)
```
"""
    (output / "README.md").write_text(readme_content)
    log.info("README.md created")

def generate_requirements(output: Path) -> None:
    """Generate the requirements.txt file."""
    if (output / "requirements.txt").exists():
        log.warning("requirements.txt already exists and will be overwritten")
    
    requirements = """openai>=1.0.0
# Note: During development, install with: pip install -r requirements.txt --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/
behavioral-contracts>=0.1.0
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
    
    # Get memory config
    memory_config = get_memory_config(spec_data)
    
    # Generate task-specific templates
    for task_name, task_def in spec_data.get("tasks", {}).items():
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