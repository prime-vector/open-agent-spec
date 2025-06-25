```
  _______ _______ _______ ______     _______ _______ _______ ______  _______    _______ _______ _______ _______
 |   _   |   _   |   _   |   _  \   |   _   |   _   |   _   |   _  \|       |  |   _   |   _   |   _   |   _   |
 |.  |   |.  1   |.  1___|.  |   |  |.  1   |.  |___|.  1___|.  |   |.|   | |  |   1___|.  1   |.  1___|.  1___|
 |.  |   |.  ____|.  __)_|.  |   |  |.  _   |.  |   |.  __)_|.  |   `-|.  |-'  |____   |.  ____|.  __)_|.  |___
 |:  1   |:  |   |:  1   |:  |   |  |:  |   |:  1   |:  1   |:  |   | |:  |    |:  1   |:  |   |:  1   |:  1   |
 |::.. . |::.|   |::.. . |::.|   |  |::.|:. |::.. . |::.. . |::.|   | |::.|    |::.. . |::.|   |::.. . |::.. . |
 `-------`---'   `-------`--- ---'  `--- ---`-------`-------`--- ---' `---'    `-------`---'   `-------`-------'
```

# Open Agent Spec (OAS) CLI

A command-line tool for generating agent projects based on Open Agent Spec YAML files.

## Installation

```bash
pip install open-agent-spec
```

## Usage

### Basic Usage
```bash
# Show help
oas --help

# Initialize a new agent project
oas init --spec path/to/spec.yaml --output path/to/output

# Preview what would be created without writing files
oas init --spec path/to/spec.yaml --output path/to/output --dry-run

# Create a base working agent with minimal spec
oas init --template minimal --output path/to/output
```

### Enable Verbose Logging
```
oas init --spec path/to/spec.yaml --output path/to/output --verbose
```

## Spec File Format
The spec file should be in YAML format with the following structure:

```yaml
open_agent_spec: 1.0.4

agent:
  name: hello-world-agent
  description: A simple agent that responds with a greeting
  role: chat

intelligence:
  type: llm
  engine: openai
  model: gpt-4
  endpoint: https://api.openai.com/v1
  config:
    temperature: 0.7
    max_tokens: 150

tasks:
  greet:
    description: Say hello to a person by name
    timeout: 30
    input:
      type: object
      properties:
        name:
          type: string
          description: The name of the person to greet
          minLength: 1
          maxLength: 100
      required: [name]
    output:
      type: object
      properties:
        response:
          type: string
          description: The greeting response
          minLength: 1
      required: [response]

prompts:
  system: >
    You are a friendly agent that greets people by name.
    Respond with: "Hello <name>!"
  user: "{{name}}"

behavioural_contract:
  version: "0.1.2"
  description: "Simple contract requiring a greeting response"
  role: "Friendly agent"
  behavioural_flags:
    conservatism: "moderate"
    verbosity: "compact"
  response_contract:
    output_format:
      required_fields: [response]

interface:
  cli:
    enabled: true
    arguments:
      - name: name
        type: string
        required: true
```

### Generated Project Structure
```
output/
├── agent.py              # Main agent implementation
├── prompts/             # Prompt templates
│   └── analyst_prompt.jinja2
├── requirements.txt     # Project dependencies
└── .env.example        # Environment variables template
```

## Development

### Setup
```bash
# Clone the repository
git clone https://github.com/aswhitehouse/open-agent-spec.git
cd oas-cli

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests
```bash
pytest
```

### Building
```bash
python -m build
```

### Pacakge Installation
[![PyPI version](https://img.shields.io/pypi/v/open-agent-spec)](https://pypi.org/project/open-agent-spec/)

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPLv3), which ensures that improvements and deployments of this codebase stay open and benefit the wider community.

If you're a business or enterprise and would like to:

Use this tool in a proprietary or internal-only setting

Avoid open-sourcing your modifications or integrations

Receive custom implementation support or consulting

Discuss a commercial license or enterprise partnership

➡️ Please feel free to reach out:
📧 andrewswhitehouse@gmail.com

Myself and my collaborators would be happy to support your journey with AI agents and ensure responsible, scalable use of this tooling in your stack.

## Overview
https://www.openagentstack.ai

## Test PR
