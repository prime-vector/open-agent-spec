"""Tests for the enhanced Open Agent Spec functionality."""

import pytest
from typer.testing import CliRunner

from oas_cli.main import app

runner = CliRunner()


@pytest.fixture
def enhanced_spec_yaml(tmp_path):
    """Create a sample enhanced spec YAML file."""
    yaml_content = """
open_agent_spec: "1.0.4"
agent:
  name: "analyst_agent"
  description: "A financial market analyst agent that analyzes trading signals"
  role: "analyst"

intelligence:
  type: "llm"
  engine: "openai"
  model: "gpt-4"
  endpoint: "https://api.openai.com/v1"
  config:
    temperature: 0.7
    max_tokens: 150

behavioural_contract:
  version: "1.1"
  description: "Financial market analyst agent"
  role: "smart_analyst"
  policy:
    pii: false
    compliance_tags: ["EU-AI-ACT"]
    allowed_tools: ["search", "summary", "confidence_estimator"]
  behavioural_flags:
    conservatism: "moderate"
    verbosity: "compact"
    temperature_control:
      mode: "adaptive"
      range: [0.2, 0.6]

tasks:
  analyze_signal:
    description: "Analyze a market signal and provide trading recommendations"
    timeout: 60
    input:
      type: object
      properties:
        symbol:
          type: string
          description: "Stock symbol to analyze"
          minLength: 1
          maxLength: 10
        indicators:
          type: object
          description: "Technical indicators data"
          properties:
            rsi:
              type: number
              description: "Relative Strength Index"
              minimum: 0
              maximum: 100
            ema_50:
              type: number
              description: "50-day Exponential Moving Average"
              minimum: 0
            ema_200:
              type: number
              description: "200-day Exponential Moving Average"
              minimum: 0
          required: [rsi, ema_50, ema_200]
      required: [symbol, indicators]
    output:
      type: object
      properties:
        decision:
          type: string
          description: "Trading decision"
          enum: ["buy", "hold", "sell"]
        confidence:
          type: string
          description: "Confidence level in the decision"
          enum: ["low", "medium", "high"]
        summary:
          type: string
          description: "Brief summary of the analysis"
          minLength: 1
          maxLength: 200
        reasoning:
          type: string
          description: "Detailed reasoning for the decision"
          minLength: 1
        compliance_tags:
          type: array
          description: "Compliance tags for regulatory purposes"
          items:
            type: string
      required: [decision, confidence, summary, reasoning, compliance_tags]

prompts:
  system: |
    You are a financial analyst agent. You will receive a JSON payload representing a market signal including technical indicators: RSI, EMA50, EMA200.

    Your task is to return a structured JSON object with the following fields:
    - decision: one of "buy", "hold", or "sell"
    - confidence: one of "low", "medium", or "high"
    - summary: a 1-line summary of your judgment
    - reasoning: 2-3 sentences explaining your thought process
    - compliance_tags: ["EU-AI-ACT"]  # Required for compliance
  user: |
    Here's the latest signal data:
    - Symbol: {symbol}
    - Interval: {interval}
    - Price: ${price:,}
    - Market Cap: ${market_cap:,}
    - Timestamp: {timestamp}

    {memory_summary}
    {indicators_summary}
    Based on this, what would you recommend?
"""
    spec_file = tmp_path / "enhanced_agent.yaml"
    spec_file.write_text(yaml_content)
    return spec_file


def test_enhanced_spec_validation(enhanced_spec_yaml):
    """Test that the enhanced spec is properly validated."""
    result = runner.invoke(
        app, ["init", "--spec", str(enhanced_spec_yaml), "--output", "test_output"]
    )
    assert result.exit_code == 0
    # Add more specific assertions as we implement the enhanced spec features


def test_enhanced_spec_generation(enhanced_spec_yaml, tmp_path):
    """Test that the enhanced spec generates the correct agent code structure."""
    output_dir = tmp_path / "test_output"
    result = runner.invoke(
        app, ["init", "--spec", str(enhanced_spec_yaml), "--output", str(output_dir)]
    )
    assert result.exit_code == 0

    # Check that the generated files exist
    assert (output_dir / "agent.py").exists()
    assert (output_dir / "requirements.txt").exists()
    assert (output_dir / "README.md").exists()

    # Read the generated agent.py to verify structure
    agent_code = (output_dir / "agent.py").read_text()

    # Verify behavioural contract is included
    assert "@behavioural_contract" in agent_code

    # Verify task function is generated
    assert "def analyze_signal" in agent_code

    # Verify memory support is included
    assert "memory" in agent_code
    assert "get_memory" in agent_code
    assert "enabled" in agent_code
    assert "format" in agent_code
    assert "usage" in agent_code

    # Read the generated prompt template to verify memory support
    prompt_content = (output_dir / "prompts" / "analyze_signal.jinja2").read_text()
    assert "{% if memory_summary %}" in prompt_content
    assert "{{ memory_summary }}" in prompt_content
    assert "You are a financial analyst agent" in prompt_content
    assert "Here's the latest signal data:" in prompt_content


def test_schema_version(enhanced_spec_yaml):
    """Test that schema version validation works correctly."""
    # Test valid versions
    valid_versions = ["1.0.4", "1.0.5", "1.1.0", "1.2.0", "2.0.0", "2.1.0"]

    for version in valid_versions:
        # Create a minimal valid spec with the test version
        yaml_content = f"""
open_agent_spec: "{version}"
agent:
  name: "test_agent"
  description: "A test agent for validation"
  role: "chat"
intelligence:
  type: "llm"
  engine: "openai"
  model: "gpt-4"
  endpoint: "https://api.openai.com/v1"
  config:
    temperature: 0.7
    max_tokens: 150
tasks:
  greet:
    description: Test description
    version: "1.0.0"
    timeout: 30
    input:
      type: object
      properties:
        name:
          type: string
          description: "Name to greet"
          minLength: 1
      required: [name]
    output:
      type: object
      properties:
        response:
          type: string
          description: "Greeting response"
          minLength: 1
      required: [response]
prompts:
  system: "You are a test agent."
  user: "Test input"
behavioural_contract:
  version: "0.1.2"
  description: "Simple test contract"
  role: "Friendly agent"
  behavioural_flags:
    conservatism: "moderate"
    verbosity: "compact"
  response_contract:
    output_format:
      required_fields: [response]
"""
        spec_file = (
            enhanced_spec_yaml.parent / f"valid_spec_{version.replace('.', '_')}.yaml"
        )
        spec_file.write_text(yaml_content)

        # Should pass validation
        result = runner.invoke(
            app,
            ["init", "--spec", str(spec_file), "--output", "test_output", "--dry-run"],
        )
        assert (
            result.exit_code == 0
        ), f"Version {version} should be valid but failed: {result.stdout}"

    # Test invalid versions
    invalid_versions = ["1.0.0", "1.0.1", "1.0.2", "1.0.3", "0.3.0", "0.4.0"]

    for version in invalid_versions:
        # Create a minimal spec with the invalid version
        yaml_content = f"""
open_agent_spec: "{version}"
agent:
  name: "test_agent"
  description: "A test agent for validation"
  role: "chat"
intelligence:
  type: "llm"
  engine: "openai"
  model: "gpt-4"
  endpoint: "https://api.openai.com/v1"
  config:
    temperature: 0.7
    max_tokens: 150
tasks:
  greet:
    description: Test description
    version: "1.0.0"
    timeout: 30
    input:
      type: object
      properties:
        name:
          type: string
          description: "Name to greet"
          minLength: 1
      required: [name]
    output:
      type: object
      properties:
        response:
          type: string
          description: "Greeting response"
          minLength: 1
      required: [response]
prompts:
  system: "You are a test agent."
  user: "Test input"
behavioural_contract:
  version: "0.1.2"
  description: "Simple test contract"
  role: "Friendly agent"
  behavioural_flags:
    conservatism: "moderate"
    verbosity: "compact"
  response_contract:
    output_format:
      required_fields: [response]
"""
        spec_file = (
            enhanced_spec_yaml.parent / f"invalid_spec_{version.replace('.', '_')}.yaml"
        )
        spec_file.write_text(yaml_content)

        # Should fail validation
        result = runner.invoke(
            app,
            ["init", "--spec", str(spec_file), "--output", "test_output", "--dry-run"],
        )
        assert result.exit_code != 0, f"Version {version} should be invalid but passed"
        assert (
            "Spec validation failed" in result.output
        ), f"Version {version} should show validation error"


def test_agent_role_validation(enhanced_spec_yaml):
    """Test that agent role validation works correctly."""
    # Test valid roles
    valid_roles = ["analyst", "reviewer", "chat", "retriever", "planner", "executor"]

    for role in valid_roles:
        yaml_content = f"""
open_agent_spec: "1.0.4"
agent:
  name: "test_agent"
  description: "A test agent for role validation"
  role: "{role}"
intelligence:
  type: "llm"
  engine: "openai"
  model: "gpt-4"
  endpoint: "https://api.openai.com/v1"
  config:
    temperature: 0.7
    max_tokens: 150
tasks:
  greet:
    description: Test description
    version: "1.0.0"
    timeout: 30
    input:
      type: object
      properties:
        name:
          type: string
          description: "Name to greet"
          minLength: 1
      required: [name]
    output:
      type: object
      properties:
        response:
          type: string
          description: "Greeting response"
          minLength: 1
      required: [response]
prompts:
  system: "You are a test agent."
  user: "Test input"
behavioural_contract:
  version: "0.1.2"
  description: "Simple test contract"
  role: "Test agent"
  behavioural_flags:
    conservatism: "moderate"
    verbosity: "compact"
  response_contract:
    output_format:
      required_fields: [response]
"""
        spec_file = enhanced_spec_yaml.parent / f"valid_role_{role}.yaml"
        spec_file.write_text(yaml_content)

        # Should pass validation
        result = runner.invoke(
            app,
            ["init", "--spec", str(spec_file), "--output", "test_output", "--dry-run"],
        )
        assert (
            result.exit_code == 0
        ), f"Role {role} should be valid but failed: {result.output}"

    # Test invalid roles
    invalid_roles = ["smart_analyst", "test_role", "friendly_agent", "invalid_role", ""]

    for role in invalid_roles:
        yaml_content = f"""
open_agent_spec: "1.0.4"
agent:
  name: "test_agent"
  description: "A test agent for role validation"
  role: "{role}"
intelligence:
  type: "llm"
  engine: "openai"
  model: "gpt-4"
  endpoint: "https://api.openai.com/v1"
  config:
    temperature: 0.7
    max_tokens: 150
tasks:
  greet:
    description: Test description
    version: "1.0.0"
    timeout: 30
    input:
      type: object
      properties:
        name:
          type: string
          description: "Name to greet"
          minLength: 1
      required: [name]
    output:
      type: object
      properties:
        response:
          type: string
          description: "Greeting response"
          minLength: 1
      required: [response]
prompts:
  system: "You are a test agent."
  user: "Test input"
behavioural_contract:
  version: "0.1.2"
  description: "Simple test contract"
  role: "Test agent"
  behavioural_flags:
    conservatism: "moderate"
    verbosity: "compact"
  response_contract:
    output_format:
      required_fields: [response]
"""
        spec_file = (
            enhanced_spec_yaml.parent / f"invalid_role_{role.replace(' ', '_')}.yaml"
        )
        spec_file.write_text(yaml_content)

        # Should fail validation
        result = runner.invoke(
            app,
            ["init", "--spec", str(spec_file), "--output", "test_output", "--dry-run"],
        )
        assert result.exit_code != 0, f"Role {role} should be invalid but passed"
        assert (
            "Spec validation failed" in result.output
        ), f"Role {role} should show validation error"


def test_agent_description_required(enhanced_spec_yaml):
    """Test that agent description is required."""
    yaml_content = """
open_agent_spec: "1.0.4"
agent:
  name: "test_agent"
  role: "chat"
intelligence:
  type: "llm"
  engine: "openai"
  model: "gpt-4"
  endpoint: "https://api.openai.com/v1"
  config:
    temperature: 0.7
    max_tokens: 150
tasks:
  greet:
    description: Test description
    version: "1.0.0"
    timeout: 30
    input:
      type: object
      properties:
        name:
          type: string
          description: "Name to greet"
          minLength: 1
      required: [name]
    output:
      type: object
      properties:
        response:
          type: string
          description: "Greeting response"
          minLength: 1
      required: [response]
prompts:
  system: "You are a test agent."
  user: "Test input"
behavioural_contract:
  version: "0.1.2"
  description: "Simple test contract"
  role: "Test agent"
  behavioural_flags:
    conservatism: "moderate"
    verbosity: "compact"
  response_contract:
    output_format:
      required_fields: [response]
"""
    spec_file = enhanced_spec_yaml.parent / "missing_description.yaml"
    spec_file.write_text(yaml_content)

    # Should fail validation
    result = runner.invoke(
        app, ["init", "--spec", str(spec_file), "--output", "test_output", "--dry-run"]
    )
    assert result.exit_code != 0, "Missing description should cause validation failure"
    assert (
        "Spec validation failed" in result.output
    ), "Missing description should show validation error"


def test_intelligence_config_validation(enhanced_spec_yaml):
    """Test that intelligence configuration validation works correctly."""
    # Test valid configurations
    valid_configs = [
        {"temperature": 0.7, "max_tokens": 150},
        {"temperature": 0.0, "max_tokens": 1000, "top_p": 0.9},
        {
            "temperature": 2.0,
            "max_tokens": 50,
            "frequency_penalty": -0.5,
            "presence_penalty": 0.3,
        },
    ]

    for i, config in enumerate(valid_configs):
        yaml_content = f"""
open_agent_spec: "1.0.4"
agent:
  name: "test_agent"
  description: "A test agent for intelligence config validation"
  role: "chat"
intelligence:
  type: "llm"
  engine: "openai"
  model: "gpt-4"
  endpoint: "https://api.openai.com/v1"
  config:
    temperature: {config.get("temperature", 0.7)}
    max_tokens: {config.get("max_tokens", 150)}
    top_p: {config.get("top_p", 1.0)}
    frequency_penalty: {config.get("frequency_penalty", 0.0)}
    presence_penalty: {config.get("presence_penalty", 0.0)}
tasks:
  greet:
    description: Test description
    version: "1.0.0"
    timeout: 30
    input:
      type: object
      properties:
        name:
          type: string
          description: "Name to greet"
          minLength: 1
      required: [name]
    output:
      type: object
      properties:
        response:
          type: string
          description: "Greeting response"
          minLength: 1
      required: [response]
prompts:
  system: "You are a test agent."
  user: "Test input"
behavioural_contract:
  version: "0.1.2"
  description: "Simple test contract"
  role: "Test agent"
  behavioural_flags:
    conservatism: "moderate"
    verbosity: "compact"
  response_contract:
    output_format:
      required_fields: [response]
"""
        spec_file = enhanced_spec_yaml.parent / f"valid_intelligence_config_{i}.yaml"
        spec_file.write_text(yaml_content)

        # Should pass validation
        result = runner.invoke(
            app,
            ["init", "--spec", str(spec_file), "--output", "test_output", "--dry-run"],
        )
        assert (
            result.exit_code == 0
        ), f"Config {config} should be valid but failed: {result.output}"

    # Test invalid configurations
    invalid_configs = [
        {
            "temperature": 2.5,  # Above maximum
            "max_tokens": 150,
        },
        {
            "temperature": -0.1,  # Below minimum
            "max_tokens": 150,
        },
        {
            "temperature": 0.7,
            "max_tokens": 0,  # Below minimum
        },
        {
            "temperature": 0.7,
            "max_tokens": 150,
            "top_p": 1.5,  # Above maximum
        },
    ]

    for i, config in enumerate(invalid_configs):
        yaml_content = f"""
open_agent_spec: "1.0.4"
agent:
  name: "test_agent"
  description: "A test agent for intelligence config validation"
  role: "chat"
intelligence:
  type: "llm"
  engine: "openai"
  model: "gpt-4"
  endpoint: "https://api.openai.com/v1"
  config:
    temperature: {config.get("temperature", 0.7)}
    max_tokens: {config.get("max_tokens", 150)}
    top_p: {config.get("top_p", 1.0)}
tasks:
  greet:
    description: Test description
    version: "1.0.0"
    timeout: 30
    input:
      type: object
      properties:
        name:
          type: string
          description: "Name to greet"
          minLength: 1
      required: [name]
    output:
      type: object
      properties:
        response:
          type: string
          description: "Greeting response"
          minLength: 1
      required: [response]
prompts:
  system: "You are a test agent."
  user: "Test input"
behavioural_contract:
  version: "0.1.2"
  description: "Simple test contract"
  role: "Test agent"
  behavioural_flags:
    conservatism: "moderate"
    verbosity: "compact"
  response_contract:
    output_format:
      required_fields: [response]
"""
        spec_file = enhanced_spec_yaml.parent / f"invalid_intelligence_config_{i}.yaml"
        spec_file.write_text(yaml_content)

        # Should fail validation
        result = runner.invoke(
            app,
            ["init", "--spec", str(spec_file), "--output", "test_output", "--dry-run"],
        )
        assert result.exit_code != 0, f"Config {config} should be invalid but passed"
        assert (
            "Spec validation failed" in result.output
        ), f"Config {config} should show validation error"


def test_intelligence_endpoint_validation(enhanced_spec_yaml):
    """Test that intelligence endpoint validation works correctly."""
    # Test valid endpoints
    valid_endpoints = [
        "https://api.openai.com/v1",
        "https://api.openai.com/v1/",
        "https://custom-endpoint.com/api/v1",
        "http://localhost:8000/v1",
    ]

    for endpoint in valid_endpoints:
        yaml_content = f"""
open_agent_spec: "1.0.4"
agent:
  name: "test_agent"
  description: "A test agent for endpoint validation"
  role: "chat"
intelligence:
  type: "llm"
  engine: "openai"
  model: "gpt-4"
  endpoint: "{endpoint}"
  config:
    temperature: 0.7
    max_tokens: 150
tasks:
  greet:
    description: Test description
    version: "1.0.0"
    timeout: 30
    input:
      type: object
      properties:
        name:
          type: string
          description: "Name to greet"
          minLength: 1
      required: [name]
    output:
      type: object
      properties:
        response:
          type: string
          description: "Greeting response"
          minLength: 1
      required: [response]
prompts:
  system: "You are a test agent."
  user: "Test input"
behavioural_contract:
  version: "0.1.2"
  description: "Simple test contract"
  role: "Test agent"
  behavioural_flags:
    conservatism: "moderate"
    verbosity: "compact"
  response_contract:
    output_format:
      required_fields: [response]
"""
        spec_file = (
            enhanced_spec_yaml.parent
            / f"valid_endpoint_{endpoint.replace('://', '_').replace('/', '_')}.yaml"
        )
        spec_file.write_text(yaml_content)

        # Should pass validation
        result = runner.invoke(
            app,
            ["init", "--spec", str(spec_file), "--output", "test_output", "--dry-run"],
        )
        assert (
            result.exit_code == 0
        ), f"Endpoint {endpoint} should be valid but failed: {result.output}"

    # Test invalid endpoints
    invalid_endpoints = ["not-a-url", "ftp://invalid-protocol.com", "invalid://url", ""]

    for endpoint in invalid_endpoints:
        yaml_content = f"""
open_agent_spec: "1.0.4"
agent:
  name: "test_agent"
  description: "A test agent for endpoint validation"
  role: "chat"
intelligence:
  type: "llm"
  engine: "openai"
  model: "gpt-4"
  endpoint: "{endpoint}"
  config:
    temperature: 0.7
    max_tokens: 150
tasks:
  greet:
    description: Test description
    version: "1.0.0"
    timeout: 30
    input:
      type: object
      properties:
        name:
          type: string
          description: "Name to greet"
          minLength: 1
      required: [name]
    output:
      type: object
      properties:
        response:
          type: string
          description: "Greeting response"
          minLength: 1
      required: [response]
prompts:
  system: "You are a test agent."
  user: "Test input"
behavioural_contract:
  version: "0.1.2"
  description: "Simple test contract"
  role: "Test agent"
  behavioural_flags:
    conservatism: "moderate"
    verbosity: "compact"
  response_contract:
    output_format:
      required_fields: [response]
"""
        spec_file = (
            enhanced_spec_yaml.parent
            / f"invalid_endpoint_{endpoint.replace('://', '_').replace('/', '_')}.yaml"
        )
        spec_file.write_text(yaml_content)

        # Should fail validation
        result = runner.invoke(
            app,
            ["init", "--spec", str(spec_file), "--output", "test_output", "--dry-run"],
        )
        assert (
            result.exit_code != 0
        ), f"Endpoint {endpoint} should be invalid but passed"
        assert (
            "Spec validation failed" in result.output
        ), f"Endpoint {endpoint} should show validation error"


def test_tasks_schema_validation(enhanced_spec_yaml):
    """Test that tasks schema validation works correctly."""
    # Test valid task configurations
    valid_tasks = [
        {
            "name": "simple_task",
            "description": "A simple task",
            "input": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Input text",
                        "minLength": 1,
                    }
                },
                "required": ["text"],
            },
            "output": {
                "type": "object",
                "properties": {
                    "result": {"type": "string", "description": "Output result"}
                },
                "required": ["result"],
            },
        },
        {
            "name": "complex_task",
            "description": "A complex task with various data types",
            "version": "2.1.0",
            "timeout": 120,
            "input": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "number",
                        "description": "A number",
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "choice": {
                        "type": "string",
                        "description": "A choice",
                        "enum": ["option1", "option2", "option3"],
                    },
                    "flag": {
                        "type": "boolean",
                        "description": "A boolean flag",
                        "default": False,
                    },
                },
                "required": ["number", "choice"],
            },
            "output": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Status",
                        "enum": ["success", "error"],
                    },
                    "data": {
                        "type": "array",
                        "description": "Array of results",
                        "items": {"type": "string"},
                    },
                },
                "required": ["status"],
            },
        },
    ]

    for i, task in enumerate(valid_tasks):
        # Create a simpler YAML structure to avoid formatting issues
        yaml_content = f"""open_agent_spec: "1.0.4"
agent:
  name: "test_agent"
  description: "A test agent for task validation"
  role: "chat"
intelligence:
  type: "llm"
  engine: "openai"
  model: "gpt-4"
  endpoint: "https://api.openai.com/v1"
  config:
    temperature: 0.7
    max_tokens: 150
tasks:
  {task["name"]}:
    description: "{task["description"]}"
    timeout: {task.get("timeout", 30)}
    input:
      type: "{task["input"]["type"]}"
      properties:
        text:
          type: string
          description: "Input text"
          minLength: 1
      required: ["text"]
    output:
      type: "{task["output"]["type"]}"
      properties:
        result:
          type: string
          description: "Output result"
      required: ["result"]
prompts:
  system: "You are a test agent."
  user: "Test input"
behavioural_contract:
  version: "0.1.2"
  description: "Simple test contract"
  role: "Test agent"
  behavioural_flags:
    conservatism: "moderate"
    verbosity: "compact"
  response_contract:
    output_format:
      required_fields: [response]
"""
        spec_file = enhanced_spec_yaml.parent / f"valid_task_{i}.yaml"
        spec_file.write_text(yaml_content)

        # Should pass validation
        result = runner.invoke(
            app,
            ["init", "--spec", str(spec_file), "--output", "test_output", "--dry-run"],
        )
        assert (
            result.exit_code == 0
        ), f"Task {task['name']} should be valid but failed: {result.output}"

    # Test invalid task configurations
    invalid_tasks = [
        {
            "name": "invalid_task",
            "description": "Task with invalid property name",
            "input": {
                "type": "object",
                "properties": {
                    "123invalid": {  # Invalid property name (starts with number)
                        "type": "string"
                    }
                },
            },
            "output": {"type": "object", "properties": {"result": {"type": "string"}}},
        },
        {
            "name": "invalid_task",
            "description": "Task with invalid type",
            "input": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "invalid_type"  # Invalid type
                    }
                },
            },
            "output": {"type": "object", "properties": {"result": {"type": "string"}}},
        },
        {
            "name": "invalid_task",
            "description": "Task with invalid constraints",
            "input": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "number",
                        "minimum": 10,
                        "maximum": 5,  # Maximum less than minimum
                    }
                },
            },
            "output": {"type": "object", "properties": {"result": {"type": "string"}}},
        },
    ]

    for i, task in enumerate(invalid_tasks):
        # Create a simpler YAML structure to avoid formatting issues
        yaml_content = f"""open_agent_spec: "1.0.4"
agent:
  name: "test_agent"
  description: "A test agent for task validation"
  role: "chat"
intelligence:
  type: "llm"
  engine: "openai"
  model: "gpt-4"
  endpoint: "https://api.openai.com/v1"
  config:
    temperature: 0.7
    max_tokens: 150
tasks:
  {task["name"]}:
    description: "{task["description"]}"
    input:
      type: "{task["input"]["type"]}"
      properties:
        123invalid:
          type: string
    output:
      type: "{task["output"]["type"]}"
      properties:
        result:
          type: string
prompts:
  system: "You are a test agent."
  user: "Test input"
behavioural_contract:
  version: "0.1.2"
  description: "Simple test contract"
  role: "Test agent"
  behavioural_flags:
    conservatism: "moderate"
    verbosity: "compact"
  response_contract:
    output_format:
      required_fields: [response]
"""
        print("--- YAML for invalid task ---")
        print(yaml_content)
        print("-----------------------------")
        spec_file = enhanced_spec_yaml.parent / f"invalid_task_{i}.yaml"
        spec_file.write_text(yaml_content)

        # Should fail validation
        result = runner.invoke(
            app,
            ["init", "--spec", str(spec_file), "--output", "test_output", "--dry-run"],
        )
        assert (
            result.exit_code != 0
        ), f"Task {task['name']} should be invalid but passed"
        assert (
            "Spec validation failed" in result.output
        ), f"Task {task['name']} should show validation error"
