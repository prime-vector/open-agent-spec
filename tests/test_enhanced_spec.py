"""Tests for the enhanced Open Agent Spec functionality."""

import pytest
from typer.testing import CliRunner

from oas_cli.main import app

runner = CliRunner()


@pytest.fixture
def enhanced_spec_yaml(tmp_path):
    """Create a sample enhanced spec YAML file."""
    yaml_content = """
open_agent_spec: "1.0.0"
agent:
  name: "analyst_agent"
  role: "smart_analyst"
  version: "1.0.0"
  description: "Financial market analyst agent"

memory:
  enabled: true
  format: "string"
  usage: "prompt-append"
  required: true
  description: "Memory context for market analysis"

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
    input:
      type: "object"
      properties:
        symbol:
          type: "string"
        indicators:
          type: "object"
          properties:
            rsi:
              type: "number"
            ema_50:
              type: "number"
            ema_200:
              type: "number"
    output:
      type: "object"
      required_fields:
        - decision
        - confidence
        - summary
        - reasoning
        - compliance_tags
      properties:
        decision:
          type: "string"
          enum: ["buy", "hold", "sell"]
        confidence:
          type: "string"
          enum: ["low", "medium", "high"]
        summary:
          type: "string"
        reasoning:
          type: "string"
        compliance_tags:
          type: "array"
          items:
            type: "string"

integration:
  memory:
    type: "persistent"
    fields:
      - symbol
      - timestamp
      - analysis
  task_queue:
    type: "distributed"
    operations:
      - fetch_pending
      - claim
      - complete
      - post_review

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

    # Read the generated README.md to verify memory documentation
    readme_content = (output_dir / "README.md").read_text()
    assert "Memory Support" in readme_content
    assert "Configuration" in readme_content
    assert "Format" in readme_content
    assert "Usage" in readme_content

    # Read the generated prompt template to verify memory support
    prompt_content = (output_dir / "prompts" / "analyze_signal.jinja2").read_text()
    assert "{% if memory_summary %}" in prompt_content
    assert "{{ memory_summary }}" in prompt_content
    assert "You are a financial analyst agent" in prompt_content
    assert "Here's the latest signal data:" in prompt_content
