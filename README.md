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

A command-line tool for generating AI agent projects based on Open Agent Spec YAML files. The OAS CLI supports multiple LLM engines including OpenAI, Anthropic, local models, and custom LLM routers.

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
```bash
oas init --spec path/to/spec.yaml --output path/to/output --verbose
```

## Spec File Format

The spec file should be in YAML format with the following structure. Each section is explained in detail below:

```yaml
spec_version: "1.0.7"  # OAS specification version

agent:
  name: "hello-world-agent"           # Unique identifier for the agent
  description: "A simple agent that responds with a greeting"  # Human-readable description
  role: "assistant"                   # Agent role type (assistant, analyst, etc.)

intelligence:
  engine: "openai"                    # LLM engine: openai, anthropic, local, or custom
  endpoint: "https://api.openai.com/v1"  # API endpoint URL
  model: "gpt-4"                      # Model name/identifier
  config:                             # Engine-specific configuration
    temperature: 0.7
    max_tokens: 150
  module: "CustomRouter.CustomRouter"  # For custom engines: module.class format

tasks:
  greet:                              # Task name (will become function name)
    description: "Say hello to a person by name"  # Task description
    timeout: 30                       # Task timeout in seconds
    input:                            # Input schema (JSON Schema format)
      type: "object"
      properties:
        name:
          type: "string"
          description: "The name of the person to greet"
          minLength: 1
          maxLength: 100
      required: ["name"]
    output:                           # Output schema (JSON Schema format)
      type: "object"
      properties:
        response:
          type: "string"
          description: "The greeting response"
          minLength: 1
      required: ["response"]
    metadata:                         # Optional task metadata
      category: "communication"
      priority: "normal"

behavioural_contract:                 # Optional behavioural contract
  version: "0.1.2"
  description: "Simple contract requiring a greeting response"
  behavioural_flags:
    conservatism: "moderate"
    verbosity: "compact"
  response_contract:
    output_format:
      required_fields: ["response"]
```

## Intelligence Engine Options

The OAS CLI supports multiple LLM engines through the `intelligence.engine` field:

### 1. OpenAI (`engine: "openai"`)
Use OpenAI's API for LLM interactions.

```yaml
intelligence:
  engine: "openai"
  endpoint: "https://api.openai.com/v1"  # OpenAI API endpoint
  model: "gpt-4"                         # OpenAI model (gpt-4, gpt-3.5-turbo, etc.)
  config:
    temperature: 0.7                     # Response randomness (0.0-2.0)
    max_tokens: 150                      # Maximum response length
```

**Requirements:**
- OpenAI API key in environment variable `OPENAI_API_KEY`
- Valid OpenAI account and API access

### 2. Anthropic (`engine: "anthropic"`)
Use Anthropic's Claude models for LLM interactions.

```yaml
intelligence:
  engine: "anthropic"
  endpoint: "https://api.anthropic.com"  # Anthropic API endpoint
  model: "claude-3-sonnet-20240229"      # Claude model name
  config:
    temperature: 0.7
    max_tokens: 150
```

**Requirements:**
- Anthropic API key in environment variable `ANTHROPIC_API_KEY`
- Valid Anthropic account and API access

### 3. Local (`engine: "local"`)
Use locally hosted LLM models (placeholder for future implementation).

```yaml
intelligence:
  engine: "local"
  endpoint: "http://localhost:8000"      # Local model server endpoint
  model: "llama-2-7b"                   # Local model identifier
  config:
    temperature: 0.7
    max_tokens: 150
```

**Note:** Local engine support is planned for future releases.

### 4. Custom (`engine: "custom"`)
Use custom LLM routers for specialized use cases, custom APIs, or proprietary models.

```yaml
intelligence:
  engine: "custom"
  endpoint: "http://localhost:1234/invoke"  # Custom endpoint
  model: "my-custom-model"                  # Model identifier
  config: {}                                # Custom configuration
  module: "CustomLLMRouter.CustomLLMRouter" # Python module.class to import
```

**Custom Router Requirements:**
- Python class with `__init__(endpoint, model, config)` method
- `run(prompt, **kwargs)` method that returns a JSON string
- Class must be importable from the specified module path

**Example Custom Router:**
```python
# CustomLLMRouter.py
import json

class CustomLLMRouter:
    def __init__(self, endpoint: str, model: str, config: dict):
        self.endpoint = endpoint
        self.model = model
        self.config = config

    def run(self, prompt: str, **kwargs) -> str:
        # Your custom LLM logic here
        # Must return a JSON string matching the task's output schema
        return json.dumps({
            "response": f"Custom response to: {prompt}"
        })
```

## YAML Field Explanations

### `spec_version`
- **Purpose:** Version of the OAS specification being used
- **Format:** String (e.g., "1.0.4")
- **Required:** Yes
- **Note:** Ensures compatibility with the CLI version

### `agent` Section
- **Purpose:** Defines the agent's identity and characteristics

#### `agent.name`
- **Purpose:** Unique identifier for the agent
- **Format:** String (kebab-case recommended)
- **Required:** Yes
- **Example:** "hello-world-agent", "financial-analyst"

#### `agent.description`
- **Purpose:** Human-readable description of what the agent does
- **Format:** String
- **Required:** Yes
- **Example:** "A friendly agent that greets people by name"

#### `agent.role`
- **Purpose:** Defines the agent's role type
- **Format:** String (enum)
- **Required:** No (optional)
- **Options:** "assistant", "analyst", "specialist", "coordinator", "researcher", "consultant"

### `intelligence` Section
- **Purpose:** Configures the LLM engine and model settings

#### `intelligence.engine`
- **Purpose:** Specifies which LLM engine to use
- **Format:** String (enum)
- **Required:** Yes
- **Options:** "openai", "anthropic", "local", "custom"

#### `intelligence.endpoint`
- **Purpose:** API endpoint URL for the LLM service
- **Format:** Valid URI string
- **Required:** Yes
- **Examples:**
  - OpenAI: "https://api.openai.com/v1"
  - Anthropic: "https://api.anthropic.com"
  - Custom: "http://localhost:1234/invoke"

#### `intelligence.model`
- **Purpose:** Model name or identifier to use
- **Format:** String
- **Required:** Yes
- **Examples:** "gpt-4", "claude-3-sonnet-20240229", "my-custom-model"

#### `intelligence.config`
- **Purpose:** Engine-specific configuration parameters
- **Format:** Object (key-value pairs)
- **Required:** No (optional)
- **Common fields:**
  - `temperature`: Response randomness (0.0-2.0)
  - `max_tokens`: Maximum response length
  - `top_p`: Nucleus sampling parameter
  - `frequency_penalty`: Frequency penalty for repetition

#### `intelligence.module`
- **Purpose:** For custom engines, specifies the Python module and class to import
- **Format:** String ("module.class")
- **Required:** Only for `engine: "custom"`
- **Example:** "CustomLLMRouter.CustomLLMRouter"

### `tasks` Section
- **Purpose:** Defines the agent's capabilities and functions

Each task becomes a function in the generated agent code. Task names should be descriptive and use kebab-case.

#### Task Structure
- **`description`:** Human-readable description of what the task does
- **`timeout`:** Maximum time (seconds) the task can run
- **`input`:** JSON Schema defining the task's input parameters
- **`output`:** JSON Schema defining the task's expected output
- **`metadata`:** Optional metadata for categorization and organization

#### Input/Output Schemas
- **Purpose:** Define the structure and validation rules for task inputs and outputs
- **Format:** JSON Schema (JSON Schema Draft 2020-12)
- **Features:**
  - Type validation (string, number, boolean, object, array)
  - Required field specification
  - Field descriptions
  - Min/max length for strings
  - Min/max values for numbers
  - Enum values
  - Nested object structures

### `behavioural_contract` Section (Optional)
- **Purpose:** Defines behavioural constraints and response requirements
- **Format:** Object with behavioural contract specification
- **Required:** No (optional)
- **Note:** This is separate from the behavioural contracts repository and focuses on specification rather than enforcement

## Generated Project Structure

```
output/
├── agent.py              # Main agent implementation with all tasks
├── prompts/              # Jinja2 prompt templates
│   ├── greet.jinja2      # Task-specific prompt template
│   └── agent_prompt.jinja2  # Fallback prompt template
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── README.md            # Generated documentation
└── CustomLLMRouter.py   # Custom router (if using custom engine)
```

## Built-in Templates

The OAS CLI includes ready-to-use templates for common use cases:

### Minimal Templates
```bash
# Basic single-task agent
oas init --template minimal --output my-agent/

# Multi-task agent with parallel execution
oas init --template minimal-multi-task --output my-multi-agent/

# Agent with tool usage capabilities
oas init --template minimal-agent-tool-usage --output my-tool-agent/
```

### Security Agent Templates
Advanced security templates demonstrating multi-engine support and behavioral contracts:

```bash
# Security threat analyzer (Claude/Anthropic powered)
oas init --spec oas_cli/templates/security-threat-analyzer.yaml --output threat-analyzer/

# Security risk assessor (Claude/Anthropic powered)
oas init --spec oas_cli/templates/security-risk-assessor.yaml --output risk-assessor/

# Security incident responder (OpenAI powered)
oas init --spec oas_cli/templates/security-incident-responder.yaml --output incident-responder/
```

**Security Templates Features:**
- **Multi-Engine Support**: Templates for Claude/Anthropic and OpenAI
- **Advanced Behavioral Contracts**: Security-focused validation and safety checks
- **Real-World Use Cases**: SOC automation, threat hunting, incident response
- **Agent-to-Agent Workflows**: Designed for DACP orchestration
- **Production Ready**: Comprehensive logging, error handling, and compliance features

See [SECURITY_TEMPLATES.md](oas_cli/templates/SECURITY_TEMPLATES.md) for detailed documentation and usage examples.

## Development

### Setup
```bash
# Clone the repository
git clone https://github.com/aswhitehouse/open-agent-spec.git
cd open-agent-spec

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests
```bash
# Run all tests with basic reporting
pytest

# Run with comprehensive reporting
pytest tests/ -v --cov=oas_cli --cov-report=html --cov-report=term

# Run specific test categories
pytest -m contract tests/                   # Behavioral contract validation
pytest -m multi_engine tests/               # Multi-engine compatibility
pytest -m generator tests/                  # Generator functionality tests

# Generate detailed HTML report
pytest tests/ --html=test-report.html --self-contained-html

# Generate Allure report (requires allure-pytest)
pytest tests/ --alluredir=allure-results
allure serve allure-results
```

#### Test Reporting Features
- **Coverage Reports**: HTML and terminal coverage reports
- **Test Categories**: Organized by markers (contract, multi_engine, generator)
- **Allure Reports**: Beautiful interactive test reports
- **CI Integration**: Automatic reporting in GitHub Actions
- **Artifact Upload**: Test results and coverage reports saved

#### Test Categories

- **Generator Tests**: Validate code generation, file creation, and template rendering
- **Contract Tests**: Ensure behavioral contracts work correctly across engines
- **Multi-Engine Tests**: Verify OpenAI and Claude/Anthropic compatibility
- **Integration Tests**: End-to-end validation of agent generation

### Building
```bash
python -m build
```

### Creating a Release

To create a new release:

1. **Update the version number** in `pyproject.toml`
2. **Commit and push your changes**
3. **Create and push a new tag**

```bash
# Update version in pyproject.toml, then:
git add pyproject.toml
git commit -m "Bump version to v1.0.8"
git push origin main

# Create and push the tag
git tag v1.0.8
git push origin v1.0.8
```

The GitHub Actions workflow will automatically:
- Run all tests
- Build the package
- Publish to PyPI

Your package will be available on PyPI within a few minutes.


## Package Installation

[![PyPI version](https://img.shields.io/pypi/v/open-agent-spec)](https://pypi.org/project/open-agent-spec/)

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPLv3), which ensures that improvements and deployments of this codebase stay open and benefit the wider community.

If you're a business or enterprise and would like to:

- Use this tool in a proprietary or internal-only setting
- Avoid open-sourcing your modifications or integrations
- Receive custom implementation support or consulting
- Discuss a commercial license or enterprise partnership

➡️ Please feel free to reach out:
📧 andrewswhitehouse@gmail.com

Myself and my collaborators would be happy to support your journey with AI agents and ensure responsible, scalable use of this tooling in your stack.

## Overview
https://www.openagentstack.ai

## License and Attribution

This project is licensed under the [AGPL-3.0](./LICENSE), with Additional Terms designed to protect the unique terminology, branding, and concepts of Open Agent Stack and related projects.

**Key Points:**
- You must clearly attribute Prime Vector Australia, Andrew Whitehouse, and “Open Agent Stack” in all public forks, derivatives, and deployments.
- You may not use the terms “Open Agent Stack,” “Open Agent Spec,” “Behavioral Contract Engineering (BCE),” “DACP,” “Shepard,” or any unique concepts/branding from this project as the name, identity, or primary branding of a fork or derivative without written permission from Prime Vector Australia or Andrew Whitehouse.
- For commercial use or partnership, please contact andrew@primevector.com.au.
