"""File generation functions for Open Agent Spec."""
from pathlib import Path
from typing import Dict, Any
import logging

log = logging.getLogger("oas")

def generate_agent_code(output: Path, spec_data: Dict[str, Any], agent_name: str, class_name: str) -> None:
    """Generate the agent.py file."""
    if (output / "agent.py").exists():
        log.warning("agent.py already exists and will be overwritten")
    
    agent_code = f'''from typing import Dict, Any
import openai
import json
from behavioral_contract import behavioral_contract

ROLE = "{agent_name.title()}"

@behavioral_contract({{
    "version": "1.1",
    "description": "Base behavioral contract for the {agent_name} agent.",
    "role": "{agent_name}"
}})
def analyze_signal(signal_data: str, symbol: str, timestamp: str) -> Dict[str, Any]:
    prompt = f"""Analyze the following trading signal for {{symbol}} at {{timestamp}}:\n{{signal_data}}\n\nProvide a clear recommendation."""

    client = openai.OpenAI(
        base_url="{spec_data['intelligence']['endpoint']}",
        api_key=openai.api_key
    )

    response = client.chat.completions.create(
        model="{spec_data['intelligence']['model']}",
        messages=[
            {{"role": "system", "content": "You are a professional trading analyst."}},
            {{"role": "user", "content": prompt}}
        ],
        temperature={spec_data['intelligence']['config']['temperature']},
        max_tokens={spec_data['intelligence']['config']['max_tokens']}
    )
    
    analysis = response.choices[0].message.content

    return {{
        "recommendation": "BUY",
        "confidence": 0.85,
        "rationale": analysis
    }}

class {class_name}:
    def __init__(self, api_key: str | None = None):
        self.model = "{spec_data['intelligence']['model']}"
        if api_key:
            openai.api_key = api_key

    def analyze_signal(self, signal_data: str, symbol: str, timestamp: str) -> Dict[str, Any]:
        return analyze_signal(signal_data, symbol, timestamp)

def main():
    agent = {class_name}()
    result = agent.analyze_signal(
        signal_data="Sample signal data",
        symbol="AAPL",
        timestamp="2024-03-20T12:00:00Z"
    )
    print(json.dumps(result, indent=2))

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
    
    readme_content = f"""# {spec_data['info']['name'].title().replace('-', ' ')}

{spec_data['info']['description']}

## Usage

```bash
pip install -r requirements.txt
cp .env.example .env
python agent.py
```

### Input Example:
- symbol: AAPL
- signal_data: RSI at 70
- timestamp: 2024-03-20T12:00:00Z

### Output Format:
```json
{{
  "recommendation": "BUY",
  "confidence": 0.85,
  "rationale": "Based on RSI > 70, this is a good buying opportunity."
}}
```"""
    (output / "README.md").write_text(readme_content)
    log.info("README.md created")

def generate_requirements(output: Path) -> None:
    """Generate the requirements.txt file."""
    if (output / "requirements.txt").exists():
        log.warning("requirements.txt already exists and will be overwritten")
    
    requirements_content = """openai>=1.0.0
python-dotenv>=0.19.0
typer>=0.9.0
behavioral-contract>=0.1.0
rich>=13.0.0"""
    (output / "requirements.txt").write_text(requirements_content)
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
    prompt_dir = output / "prompts"
    prompt_file = prompt_dir / "analyst_prompt.jinja2"
    
    if prompt_file.exists():
        log.warning("analyst_prompt.jinja2 already exists and will be overwritten")
    
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_content = """Analyze the following trading signal for {{symbol}} at {{timestamp}}:
{{signal_data}}

Provide a recommendation with rationale and confidence score."""
    prompt_file.write_text(prompt_content)
    log.info("prompt template created")