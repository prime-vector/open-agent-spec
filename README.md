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

# Enable verbose logging
oas init --spec path/to/spec.yaml --output path/to/output --verbose
```

### Spec File Format
The spec file should be in YAML format with the following structure:

```yaml
info:
  name: my-agent
  description: A fantastic agent that changes the world

intelligence:
  endpoint: https://api.openai.com/v1
  model: gpt-4
  config:
    temperature: 0.7
    max_tokens: 1000
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
[![Python versions](https://img.shields.io/pypi/pyversions/open-agent-spec)](https://pypi.org/project/open-agent-spec/)

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
