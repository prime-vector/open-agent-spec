"""Tests for the Open Agent Spec generators."""

import shutil
import tempfile
from pathlib import Path

import pytest

from oas_cli.generators import (
    generate_agent_code,
    generate_env_example,
    generate_prompt_template,
    generate_readme,
    generate_requirements,
    to_pascal_case,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_spec():
    """Return a sample valid spec for testing."""
    return {
        "info": {"name": "test-agent", "description": "A test agent"},
        "intelligence": {
            "type": "llm",
            "engine": "openai",
            "model": "gpt-4",
            "config": {
                "temperature": 0.7,
                "max_tokens": 1000,
            },
        },
        "config": {
            "endpoint": "https://api.openai.com/v1",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
        },
        "memory": {
            "enabled": True,
            "format": "string",
            "usage": "prompt-append",
            "required": False,
            "description": "Memory support for maintaining context",
        },
        "behavioural_contract": {
            "version": "1.1",
            "description": "A test agent",
            "role": "test_agent",
            "policy": {
                "pii": False,
                "compliance_tags": ["TEST"],
                "allowed_tools": ["analyze"],
            },
            "behavioural_flags": {
                "conservatism": "moderate",
                "verbosity": "compact",
                "temperature_control": {"mode": "adaptive", "range": [0.2, 0.6]},
            },
        },
        "tasks": {
            "analyze": {
                "description": "Analyze the given input",
                "input": {"properties": {"text": {"type": "string"}}},
                "output": {"summary": "string", "key_points": "string"},
            }
        },
    }


def test_to_pascal_case():
    """Test the to_pascal_case utility function."""
    assert to_pascal_case("test_agent") == "TestAgent"
    assert to_pascal_case("custom_agent") == "CustomAgent"
    assert to_pascal_case("multi_word_name") == "MultiWordName"
    assert to_pascal_case("single") == "Single"


def test_generate_agent_code(temp_dir, sample_spec):
    """Test that agent.py is generated correctly."""
    # Test with a name that would normally cause duplication
    generate_agent_code(temp_dir, sample_spec, "test_agent", "TestAgent")

    agent_file = temp_dir / "agent.py"
    assert agent_file.exists()

    content = agent_file.read_text()
    assert "from behavioural_contracts import behavioural_contract" in content
    assert "import dacp" in content  # Check for DACP import
    assert "class TestAgent(dacp.Agent):" in content  # Check for DACP inheritance
    assert "def analyze(" in content
    assert "memory_summary: str = ''" in content  # Check for memory parameter
    assert "def setup_logging(self):" in content  # Check for logging setup method
    assert "self.config = {" in content  # Check for embedded config

    # Test with a different name to ensure it works consistently
    temp_dir2 = Path(tempfile.mkdtemp())
    try:
        generate_agent_code(temp_dir2, sample_spec, "custom_agent", "CustomAgent")
        content2 = (temp_dir2 / "agent.py").read_text()
        assert (
            "class CustomAgent(dacp.Agent):" in content2
        )  # Check for DACP inheritance
    finally:
        shutil.rmtree(temp_dir2)


def test_generate_readme(temp_dir, sample_spec):
    """Test that README.md is generated correctly."""
    generate_readme(temp_dir, sample_spec)

    readme_file = temp_dir / "README.md"
    assert readme_file.exists()

    content = readme_file.read_text()
    assert "# Test Agent" in content
    assert "## Tasks" in content
    assert "### Analyze" in content
    assert "## Memory Support" in content  # Check for memory section
    assert "Memory support for maintaining context" in content


def test_generate_requirements(temp_dir, sample_spec):
    """Test that requirements.txt is generated correctly."""
    generate_requirements(temp_dir, sample_spec)

    requirements_file = temp_dir / "requirements.txt"
    assert requirements_file.exists()

    content = requirements_file.read_text()
    assert "openai>=" in content
    assert "behavioural-contracts" in content
    assert "python-dotenv>=" in content


def test_generate_env_example(temp_dir, sample_spec):
    """Test that .env.example is generated correctly."""
    generate_env_example(temp_dir, sample_spec)

    env_file = temp_dir / ".env.example"
    assert env_file.exists()

    content = env_file.read_text()
    assert "OPENAI_API_KEY=" in content


def test_generate_prompt_template(temp_dir, sample_spec):
    """Test that the prompt template is generated correctly."""
    generate_prompt_template(temp_dir, sample_spec)

    # Check for task-specific template
    prompt_file = temp_dir / "prompts" / "analyze.jinja2"
    assert prompt_file.exists()

    content = prompt_file.read_text()
    assert "You are a professional AI agent" in content
    assert "TASK:" in content
    assert "Process the following task:" in content
    assert "{% if memory_summary %}" in content
    assert "--- MEMORY CONTEXT ---" in content


def test_generate_prompt_template_with_custom_prompt(temp_dir, sample_spec):
    """Test that the prompt template is generated correctly with a custom prompt."""
    # Add custom prompt to the spec
    sample_spec["prompts"] = {
        "system": "You are a specialized trading agent with deep market expertise.",
        "user": """MARKET ANALYSIS REQUEST:
{task}

MARKET CONTEXT:
{% for key, value in context.items() %}
{{ key }}: {{ value }}
{% endfor %}

ANALYSIS STEPS:
1. Review market conditions
2. Identify key patterns
3. Consider risk factors
4. Formulate trading strategy

REQUIRED OUTPUT:
{% for key in output.keys() %}
{{ key }}: <value>  # Provide detailed analysis
{% endfor %}

CONSTRAINTS:
- Focus on actionable insights
- Consider market volatility
- Include risk assessment
- Maintain professional objectivity""",
    }

    generate_prompt_template(temp_dir, sample_spec)

    # Check for task-specific template
    prompt_file = temp_dir / "prompts" / "analyze.jinja2"
    assert prompt_file.exists()

    content = prompt_file.read_text()
    # Verify custom prompt content
    assert "You are a specialized trading agent" in content
    assert "MARKET ANALYSIS REQUEST:" in content
    assert "MARKET CONTEXT:" in content
    assert "ANALYSIS STEPS:" in content
    assert "REQUIRED OUTPUT:" in content
    assert "Consider market volatility" in content
    # Verify memory support is included
    assert "{% if memory_summary %}" in content
    assert "--- MEMORY CONTEXT ---" in content


def test_generate_prompt_template_without_custom_prompt(temp_dir, sample_spec):
    """Test that the default prompt template is generated when no custom prompt is provided."""
    generate_prompt_template(temp_dir, sample_spec)

    # Check for task-specific template
    prompt_file = temp_dir / "prompts" / "analyze.jinja2"
    assert prompt_file.exists()

    content = prompt_file.read_text()
    assert "You are a professional AI agent" in content
    assert "TASK:" in content
    assert "Process the following task:" in content
    assert "{% if memory_summary %}" in content
    assert "--- MEMORY CONTEXT ---" in content


def test_generate_requirements_anthropic(temp_dir):
    """Test that requirements.txt is generated correctly for Anthropic engine."""
    anthropic_spec = {
        "intelligence": {
            "type": "llm",
            "engine": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
        }
    }
    generate_requirements(temp_dir, anthropic_spec)

    requirements_file = temp_dir / "requirements.txt"
    assert requirements_file.exists()

    content = requirements_file.read_text()
    assert "anthropic>=" in content
    assert "behavioural-contracts" in content


def test_generate_env_example_anthropic(temp_dir):
    """Test that .env.example is generated correctly for Anthropic engine."""
    anthropic_spec = {
        "intelligence": {
            "type": "llm",
            "engine": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
        }
    }
    generate_env_example(temp_dir, anthropic_spec)

    env_file = temp_dir / ".env.example"
    assert env_file.exists()

    content = env_file.read_text()
    assert "ANTHROPIC_API_KEY=" in content


@pytest.mark.parametrize("engine", ["openai", "anthropic", "local", "custom"])
def test_generate_requirements_all_engines(temp_dir, engine):
    """Test requirements.txt generation for all supported engines."""
    spec = {"intelligence": {"type": "llm", "engine": engine, "model": "gpt-4"}}
    generate_requirements(temp_dir, spec)
    requirements_file = temp_dir / "requirements.txt"
    content = requirements_file.read_text()
    if engine == "openai":
        assert "openai>=" in content
    elif engine == "anthropic":
        assert "anthropic>=" in content
    elif engine == "local":
        assert "local engine dependencies" in content
    elif engine == "custom":
        assert "custom engine dependencies" in content


@pytest.mark.parametrize("engine", ["openai", "anthropic", "local", "custom"])
def test_generate_env_example_all_engines(temp_dir, engine):
    """Test .env.example generation for all supported engines."""
    spec = {"intelligence": {"type": "llm", "engine": engine, "model": "gpt-4"}}
    generate_env_example(temp_dir, spec)
    env_file = temp_dir / ".env.example"
    content = env_file.read_text()
    if engine == "openai":
        assert "OPENAI_API_KEY=" in content
    elif engine == "anthropic":
        assert "ANTHROPIC_API_KEY=" in content
    elif engine == "local":
        assert "local engine environment variables" in content
    elif engine == "custom":
        assert "custom engine environment variables" in content


def test_generate_multi_step_task(temp_dir):
    """Test generation of multi-step tasks."""
    spec_data = {
        "agent": {
            "name": "TestAgent",
            "description": "Test agent",
            "role": "assistant",
        },
        "intelligence": {
            "engine": "openai",
            "endpoint": "https://api.openai.com/v1",
            "model": "gpt-4",
            "config": {"temperature": 0.7, "max_tokens": 150},
        },
        "tasks": {
            "greet": {
                "description": "Greet someone",
                "input": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                "output": {
                    "type": "object",
                    "properties": {"response": {"type": "string"}},
                    "required": ["response"],
                },
            },
            "compliment": {
                "description": "Compliment someone",
                "input": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                "output": {
                    "type": "object",
                    "properties": {"compliment": {"type": "string"}},
                    "required": ["compliment"],
                },
            },
            "greet_and_compliment": {
                "description": "Greet and compliment someone",
                "multi_step": True,
                "input": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "response": {"type": "string"},
                        "compliment": {"type": "string"},
                    },
                    "required": ["response", "compliment"],
                },
                "steps": [
                    {"task": "greet", "input_map": {"name": "{{ input.name }}"}},
                    {"task": "compliment", "input_map": {"name": "{{ input.name }}"}},
                ],
            },
        },
    }

    generate_agent_code(temp_dir, spec_data, "TestAgent", "TestAgent")

    agent_file = temp_dir / "agent.py"
    assert agent_file.exists()

    agent_code = agent_file.read_text()

    # Check that individual tasks are generated
    assert "def greet(" in agent_code
    assert "def compliment(" in agent_code

    # Check that multi-step task is generated
    assert "def greet_and_compliment(" in agent_code

    # Check that multi-step task calls other tasks
    assert "step_0_result = greet(name=name)" in agent_code
    assert "step_1_result = compliment(name=name)" in agent_code

    # Check that output is constructed from step results
    assert "return Greet_And_ComplimentOutput(" in agent_code
    assert "response=step_0_result.response" in agent_code
    assert "compliment=step_1_result.compliment" in agent_code
