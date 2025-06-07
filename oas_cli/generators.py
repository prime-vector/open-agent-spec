"""File generation functions for Open Agent Spec."""
from pathlib import Path
from typing import Dict, Any
import logging
import json
import openai
from behavioral_contracts import behavioral_contract

log = logging.getLogger("oas")

def generate_agent_code(output: Path, spec_data: Dict[str, Any], agent_name: str, class_name: str) -> None:
    """Generate the agent.py file."""
    if (output / "agent.py").exists():
        log.warning("agent.py already exists and will be overwritten")
    
    # Extract task definitions
    tasks = spec_data.get("tasks", {})
    if not tasks:
        log.warning("No tasks defined in spec file")
        return

    # Generate task functions and class methods
    task_functions = []
    class_methods = []
    
    for task_name, task_def in tasks.items():
        # Convert task name to snake_case for function name
        func_name = task_name.replace("-", "_")
        
        # Generate input parameters
        input_params = []
        for param_name, param_type in task_def.get("input", {}).items():
            input_params.append(f"{param_name}: str")  # Force string type for now
        
        # Generate return type annotation
        output_type = "Dict[str, Any]"
        if task_def.get("output"):
            output_type = "Dict[str, Any]"  # Could be made more specific based on output schema
        
        # Generate function docstring
        docstring = f'''"""Process {task_name} task.

    Args:
{chr(10).join(f"        {param_name}: {param_type}" for param_name, param_type in task_def.get("input", {}).items())}

    Returns:
        {output_type}
    """'''
        
        # Generate function code
        output_json = json.dumps(task_def.get('output', {}))
        task_func = f'''
@behavioral_contract({{
    "version": "1.1",
    "description": "{task_def.get('description', '')}",
    "role": "{agent_name}"
}})
def {func_name}({', '.join(input_params)}) -> {output_type}:
    {docstring}
    # Define task_def for this function
    task_def = {{
        "output": {output_json}
    }}
    prompt = f"""Process the following {task_name} task:
{chr(10).join(f'{k}: {{{k}}}' for k in task_def.get('input', {}))}

Provide a response in the following format, replacing <value> with actual values:
{chr(10).join(f'{k}: <value>  # {task_def.get("output", {}).get(k, "string")}' for k in task_def.get('output', {}))}

Example format:
input_field: example input  # string
numeric_field: 42  # number
text_field: This is a sample response  # string"""

    client = openai.OpenAI(
        base_url="{spec_data['intelligence']['endpoint']}",
        api_key=openai.api_key
    )

    response = client.chat.completions.create(
        model="{spec_data['intelligence']['model']}",
        messages=[
            {{"role": "system", "content": "You are a professional {agent_name}."}},
            {{"role": "user", "content": prompt}}
        ],
        temperature={spec_data['intelligence']['config']['temperature']},
        max_tokens={spec_data['intelligence']['config']['max_tokens']}
    )
    
    result = response.choices[0].message.content

    # Parse the response into the expected output format
    output_dict = {{}}
    output_fields = list(task_def.get('output', {{}}).keys())
    
    # Split response into lines and process each line
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
    def {func_name}(self, {', '.join(input_params)}) -> {output_type}:
        """Process {task_name} task."""
        return {func_name}({', '.join(param.split(':')[0] for param in input_params)})
'''
        class_methods.append(class_method)

    # Generate the complete agent code
    first_task_name = next(iter(tasks.keys())) if tasks else None
    agent_code = f'''from typing import Dict, Any
import openai
import json
from behavioral_contracts import behavioral_contract

ROLE = "{agent_name.title()}"

{chr(10).join(task_functions)}

class {class_name}:
    def __init__(self, api_key: str | None = None):
        self.model = "{spec_data['intelligence']['model']}"
        if api_key:
            openai.api_key = api_key

{chr(10).join(class_methods)}

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

def generate_readme(output: Path, spec_data: Dict[str, Any]) -> None:
    """Generate the README.md file."""
    if (output / "README.md").exists():
        log.warning("README.md already exists and will be overwritten")
    
    # Generate task documentation
    task_docs = []
    for task_name, task_def in spec_data.get("tasks", {}).items():
        task_docs.append(f"### {task_name.title()}\n")
        task_docs.append(f"{task_def.get('description', '')}\n")
        
        task_docs.append("#### Input:")
        for param_name, param_type in task_def.get("input", {}).items():
            task_docs.append(f"- {param_name}: {param_type}")
        
        task_docs.append("\n#### Output:")
        for param_name, param_type in task_def.get("output", {}).items():
            task_docs.append(f"- {param_name}: {param_type}")
        task_docs.append("")
    
    readme_content = f"""# {spec_data['info']['name'].title().replace('-', ' ')}

{spec_data['info']['description']}

## Usage

```bash
pip install -r requirements.txt
cp .env.example .env
python agent.py
```

## Tasks

{chr(10).join(task_docs)}

## Example Usage

```python
from agent import {spec_data['info']['name'].title().replace('-', '')}Agent

agent = {spec_data['info']['name'].title().replace('-', '')}Agent()
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
behavioral-contracts @ https://test.pypi.org/simple/behavioral-contracts
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

def generate_prompt_template(output: Path) -> None:
    """Generate the prompt template file."""
    prompts_dir = output / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    
    if (prompts_dir / "agent_prompt.jinja2").exists():
        log.warning("agent_prompt.jinja2 already exists and will be overwritten")
    
    prompt_content = """You are a professional AI agent designed to process tasks according to the Open Agent Spec.

TASK:
Process the following task:

{% for key, value in input.items() %}
{{ key }}: {{ value }}
{% endfor %}

INSTRUCTIONS:
1. Review the input data carefully
2. Consider all relevant factors
3. Provide a clear, actionable response
4. Explain your reasoning in detail

OUTPUT FORMAT:
Your response should be structured as follows:

{% for key in output.keys() %}
{{ key }}: <value>
{% endfor %}

CONSTRAINTS:
- Be clear and specific
- Focus on actionable insights
- Maintain professional objectivity
"""
    (prompts_dir / "agent_prompt.jinja2").write_text(prompt_content)
    log.info("agent_prompt.jinja2 created")