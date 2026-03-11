# Copyright (c) Prime Vector Australia, Andrew Whitehouse, Open Agent Stack contributors
# Licensed under the MIT License. See LICENSE for details.

"""Artifact generators: README, requirements.txt, .env.example, prompt templates."""

import logging
from pathlib import Path
from typing import Any

from .constants import DEFAULT_AGENT_PROMPT_TEMPLATE
from .json_example import generate_json_example_lines
from .spec_config import get_agent_info, get_memory_config, to_pascal_case

log = logging.getLogger("oas")


def _schema_properties_lines(schema: dict[str, Any], heading: str) -> list[str]:
    """List docs lines from JSON Schema input/output (properties + required)."""
    lines: list[str] = []
    props = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(props, dict) or not props:
        return lines
    lines.append(heading)
    required = set(schema.get("required") or []) if isinstance(schema.get("required"), list) else set()
    for param_name, param_def in props.items():
        if not isinstance(param_def, dict):
            lines.append(f"- {param_name}: {param_def}")
            continue
        ptype = param_def.get("type", "any")
        desc = param_def.get("description", "")
        opt = " (required)" if param_name in required else ""
        if desc:
            lines.append(f"- {param_name}{opt}: {ptype} — {desc}")
        else:
            lines.append(f"- {param_name}{opt}: {ptype}")
    lines.append("")
    return lines


def _generate_task_docs(tasks: dict[str, Any]) -> list[str]:
    """Generate documentation for tasks (input/output from JSON Schema properties)."""
    task_docs = []
    for task_name in tasks.keys():
        task_def = tasks[task_name]
        task_docs.append(f"### {task_name.title()}\n")
        task_docs.append(f"{task_def.get('description', '')}\n")
        inp = task_def.get("input")
        if isinstance(inp, dict):
            task_docs.extend(_schema_properties_lines(inp, "#### Input:"))
        out = task_def.get("output")
        if isinstance(out, dict):
            task_docs.extend(_schema_properties_lines(out, "#### Output:"))
    return task_docs


def _generate_memory_docs(memory_config: dict[str, Any]) -> list[str]:
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
            "\nTo implement memory support, override the `get_memory()` method "
            "in the agent class.\n"
        )
    return memory_docs


def _generate_behavioural_docs(behavioural_contract: dict[str, Any]) -> list[str]:
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


def _generate_example_usage(agent_info: dict[str, str], tasks: dict[str, Any]) -> str:
    """Generate example usage code."""
    first_task_name = next(iter(tasks.keys()), "")
    if not first_task_name:
        return ""
    input_props = (
        tasks[first_task_name].get("input", {}).get("properties", {})
        if isinstance(tasks[first_task_name].get("input"), dict)
        else {}
    )
    kwargs = ", ".join(f'{k}="example_{k}"' for k in input_props.keys())
    return f"""```python
from agent import {to_pascal_case(agent_info["name"])}

agent = {to_pascal_case(agent_info["name"])}()
# Example usage
task_name = "{first_task_name}"
if task_name:
    result = getattr(agent, task_name.replace("-", "_"))({kwargs})
    print(result)
```"""


def generate_readme(output: Path, spec_data: dict[str, Any]) -> None:
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


def generate_requirements(output: Path, spec_data: dict[str, Any]) -> None:
    """Generate the requirements.txt file."""
    if (output / "requirements.txt").exists():
        log.warning("requirements.txt already exists and will be overwritten")

    engine = spec_data.get("intelligence", {}).get("engine", "openai")
    requirements = []
    if engine == "openai":
        requirements.append("openai>=1.0.0")
    elif engine == "anthropic":
        requirements.append("anthropic>=0.18.0")
    elif engine == "grok":
        requirements.append("openai>=1.0.0  # xAI Grok API is OpenAI-compatible")
    elif engine == "cortex":
        requirements.append("cortex-intelligence")
        requirements.append("openai>=1.0.0  # Required for Cortex OpenAI integration")
        requirements.append(
            "anthropic>=0.18.0  # Required for Cortex Claude integration"
        )
    elif engine == "local":
        requirements.append("# Add your local engine dependencies here")
    elif engine == "custom":
        requirements.append("# Add your custom engine dependencies here")
    else:
        requirements.append("openai>=1.0.0")  # Default fallback

    requirements.extend(
        [
            "# Note: During development, install with: pip install -r requirements.txt --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/",
            "behavioural-contracts>=0.1.0",
            "python-dotenv>=0.19.0",
            "pydantic>=2.0.0",
            "jinja2>=3.0.0",
            "dacp>=0.1.0",
        ]
    )
    (output / "requirements.txt").write_text("\n".join(requirements) + "\n")
    log.info("requirements.txt created")


def generate_env_example(output: Path, spec_data: dict[str, Any]) -> None:
    """Generate the .env.example file."""
    if (output / ".env.example").exists():
        log.warning(".env.example already exists and will be overwritten")

    engine = spec_data.get("intelligence", {}).get("engine", "openai")
    if engine == "anthropic":
        env_content = "ANTHROPIC_API_KEY=your-api-key-here\n"
    elif engine == "openai":
        env_content = "OPENAI_API_KEY=your-api-key-here\n"
    elif engine == "grok":
        env_content = "XAI_API_KEY=your-xai-api-key-here\n"
    elif engine == "cortex":
        env_content = (
            "OPENAI_API_KEY=your-openai-api-key-here\n"
            "CLAUDE_API_KEY=your-claude-api-key-here\n"
        )
    elif engine == "local":
        env_content = "# Add your local engine environment variables here\n"
    elif engine == "custom":
        env_content = "# Add your custom engine environment variables here\n"
    else:
        env_content = "OPENAI_API_KEY=your-api-key-here\n"

    (output / ".env.example").write_text(env_content)
    log.info(".env.example created")


def generate_prompt_template(output: Path, spec_data: dict[str, Any]) -> None:
    """Generate the prompt template file."""
    prompts_dir = output / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    for task_name, task_def in spec_data.get("tasks", {}).items():
        template_name = f"{task_name}.jinja2"
        if (prompts_dir / template_name).exists():
            log.warning("%s already exists and will be overwritten", template_name)

        output_schema = task_def.get("output", {})
        example_json_lines: list[str] = []
        if output_schema.get("properties"):
            example_json_lines.append("{")
            for i, (k, v) in enumerate(output_schema["properties"].items()):
                comma = "," if i < len(output_schema["properties"]) - 1 else ""
                example_json_lines.extend(generate_json_example_lines(k, v, 2, comma))
            example_json_lines.append("}")
        else:
            example_json_lines.append("{}")
        example_json = "\n".join(example_json_lines)

        if "prompt" in spec_data and "template" in spec_data["prompt"]:
            prompt_content = spec_data["prompt"]["template"]
        elif "prompts" in spec_data:
            prompts = spec_data.get("prompts", {})
            task_system_prompt = prompts.get(task_name, {}).get("system")
            task_user_prompt = prompts.get(task_name, {}).get("user")
            if task_system_prompt or task_user_prompt:
                prompt_content = task_system_prompt or ""
                if task_user_prompt:
                    if prompt_content:
                        prompt_content += "\n\n"
                    prompt_content += task_user_prompt
            elif prompts.get("system") or prompts.get("user"):
                system_prompt = prompts.get(
                    "system",
                    "You are a professional AI agent designed to process tasks "
                    "according to the Open Agent Spec.\n\n",
                )
                user_prompt = prompts.get("user", "")
                prompt_content = system_prompt
                if user_prompt:
                    if not prompt_content.endswith("\n") and not prompt_content.endswith(
                        " "
                    ):
                        prompt_content += " "
                    prompt_content += user_prompt
            else:
                prompt_content = ""
            if "{% if memory_summary %}" not in prompt_content:
                prompt_content = (
                    "{% if memory_summary %}\n"
                    "--- MEMORY CONTEXT ---\n"
                    "{{ memory_summary }}\n"
                    "------------------------\n"
                    "{% endif %}\n\n"
                ) + prompt_content
        else:
            prompt_content = DEFAULT_AGENT_PROMPT_TEMPLATE

        prompt_content += (
            f"\nRespond ONLY with a JSON object in this exact format:\n{example_json}\n"
        )
        (prompts_dir / template_name).write_text(prompt_content)
        log.info("Created prompt template: %s", template_name)

    default_template = prompts_dir / "agent_prompt.jinja2"
    if default_template.exists():
        log.warning("agent_prompt.jinja2 already exists and will be overwritten")
    default_template.write_text(DEFAULT_AGENT_PROMPT_TEMPLATE)
    log.info("Created default prompt template: agent_prompt.jinja2")
