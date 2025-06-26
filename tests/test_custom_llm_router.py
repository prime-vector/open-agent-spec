import json
import pytest
import tempfile
import os
import yaml
from pathlib import Path
from oas_cli.generators import generate_agent_code


class MockCustomLLMRouter:
    """A simple mock LLM router for testing"""

    def __init__(self, endpoint: str, model: str, config: dict):
        self.endpoint = endpoint
        self.model = model
        self.config = config

    def run(self, prompt: str, **kwargs) -> str:
        """Mock run method that returns a JSON string"""
        # Extract the name from kwargs or use a default
        name = kwargs.get("name", "World")

        # Return a properly formatted JSON string
        return json.dumps({"response": f"Hello {name}!"})


def test_custom_llm_router_integration():
    """Test that a custom LLM router works correctly with the generated agent"""

    # Create a minimal spec with custom LLM router
    spec_content = """
spec_version: "1.0.4"
agent:
  name: "TestAgent"
  description: "A test agent with custom LLM router"
  role: "assistant"

intelligence:
  engine: "custom"
  endpoint: "http://localhost:1234/invoke"
  model: "test-model"
  config: {}
  module: "MockCustomLLMRouter.MockCustomLLMRouter"

tasks:
  greet:
    description: "Greet someone by name"
    input:
      type: "object"
      properties:
        name:
          type: "string"
          description: "The name to greet"
          minLength: 1
          maxLength: 100
      required: ["name"]
    output:
      type: "object"
      properties:
        response:
          type: "string"
          description: "The greeting response"
      required: ["response"]
    timeout: 30
    metadata:
      category: "communication"
      priority: "normal"
"""

    # Create temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write the spec file
        spec_file = os.path.join(temp_dir, "test_agent.yaml")
        with open(spec_file, "w") as f:
            f.write(spec_content)

        # Generate the agent code
        with open(spec_file) as f:
            spec_data = yaml.safe_load(f)
        agent_name = spec_data["agent"]["name"]
        class_name = agent_name  # or use a function to convert to PascalCase if needed

        generate_agent_code(Path(temp_dir), spec_data, agent_name, class_name)

        # Create templates AFTER generating agent code, in the same directory as agent.py
        prompts_dir = os.path.join(temp_dir, "prompts")
        os.makedirs(prompts_dir, exist_ok=True)
        greet_template_file = os.path.join(prompts_dir, "greet.jinja2")
        with open(greet_template_file, "w") as f:
            f.write("Hello {{ input.name }}!")

        # Also create a fallback agent_prompt.jinja2 template
        agent_prompt_file = os.path.join(prompts_dir, "agent_prompt.jinja2")
        with open(agent_prompt_file, "w") as f:
            f.write("{{ input }}")

        # Check that the agent.py file was created
        agent_file = os.path.join(temp_dir, "agent.py")
        assert os.path.exists(agent_file), "Agent file should be created"

        # Read the generated agent code
        with open(agent_file, "r") as f:
            agent_code = f.read()

        # Verify the custom router import is included
        assert (
            "import importlib" in agent_code
        ), "Should import importlib for dynamic loading"
        assert (
            "load_custom_llm_router" in agent_code
        ), "Should have custom router loading function"
        assert "CustomLLMRouter" in agent_code, "Should reference CustomLLMRouter"

        # Test the agent execution by importing and running it
        # We need to make the MockCustomLLMRouter available in the same directory
        mock_router_file = os.path.join(temp_dir, "MockCustomLLMRouter.py")
        with open(mock_router_file, "w") as f:
            f.write("""
import json

class MockCustomLLMRouter:
    def __init__(self, endpoint: str, model: str, config: dict):
        self.endpoint = endpoint
        self.model = model
        self.config = config

    def run(self, prompt: str, **kwargs) -> str:
        name = kwargs.get('name', 'World')
        return json.dumps({
            "response": f"Hello {name}!"
        })
""")

        # Now we can test the agent by importing it
        import sys

        sys.path.insert(0, temp_dir)

        try:
            # Import the generated agent
            from agent import TestAgent

            # Create an instance
            agent = TestAgent()

            # Test the greet function
            result = agent.greet(name="Alice")

            # Verify the result - handle both Pydantic models and dictionaries
            if hasattr(result, "response"):
                # Pydantic model
                assert (
                    result.response == "Hello Alice!"
                ), f"Expected 'Hello Alice!', got '{result.response}'"
            else:
                # Dictionary
                assert isinstance(result, dict), "Result should be a dictionary"
                assert (
                    result.get("response") == "Hello Alice!"
                ), f"Expected 'Hello Alice!', got '{result.get('response')}'"

        finally:
            # Clean up
            sys.path.pop(0)


def test_custom_llm_router_error_handling():
    """Test error handling when custom LLM router is not available"""

    spec_content = """
spec_version: "1.0.4"
agent:
  name: "TestAgent"
  description: "A test agent with custom LLM router"
  role: "assistant"

intelligence:
  engine: "custom"
  endpoint: "http://localhost:1234/invoke"
  model: "test-model"
  config: {}
  module: "NonExistentModule.NonExistentClass"

tasks:
  greet:
    description: "Greet someone by name"
    input:
      type: "object"
      properties:
        name:
          type: "string"
          description: "The name to greet"
          minLength: 1
          maxLength: 100
      required: ["name"]
    output:
      type: "object"
      properties:
        response:
          type: "string"
          description: "The greeting response"
      required: ["response"]
    timeout: 30
"""

    with tempfile.TemporaryDirectory() as temp_dir:
        spec_file = os.path.join(temp_dir, "test_agent.yaml")
        with open(spec_file, "w") as f:
            f.write(spec_content)

        # Generate the agent code
        with open(spec_file) as f:
            spec_data = yaml.safe_load(f)
        agent_name = spec_data["agent"]["name"]
        class_name = agent_name  # or use a function to convert to PascalCase if needed

        generate_agent_code(Path(temp_dir), spec_data, agent_name, class_name)

        # Test that the generated code contains the expected error handling
        agent_file = os.path.join(temp_dir, "agent.py")
        with open(agent_file, "r") as f:
            agent_code = f.read()

        # Verify the custom router import logic is included
        assert (
            "import importlib" in agent_code
        ), "Should import importlib for dynamic loading"
        assert (
            "load_custom_llm_router" in agent_code
        ), "Should have custom router loading function"
        assert (
            "NonExistentModule.NonExistentClass" in agent_code
        ), "Should reference the specified module"

        # Test that importing the module fails as expected
        import sys

        sys.path.insert(0, temp_dir)

        try:
            # This should raise an ImportError when the agent tries to import the non-existent module
            with pytest.raises(ImportError):
                import importlib

                importlib.import_module("NonExistentModule")
        finally:
            sys.path.pop(0)


def test_custom_llm_router_missing_run_method():
    """Test error handling when custom LLM router doesn't have a run method"""

    spec_content = """
spec_version: "1.0.4"
agent:
  name: "TestAgent"
  description: "A test agent with custom LLM router"
  role: "assistant"

intelligence:
  engine: "custom"
  endpoint: "http://localhost:1234/invoke"
  model: "test-model"
  config: {}
  module: "InvalidRouter.InvalidRouter"

tasks:
  greet:
    description: "Greet someone by name"
    input:
      type: "object"
      properties:
        name:
          type: "string"
          description: "The name to greet"
          minLength: 1
          maxLength: 100
      required: ["name"]
    output:
      type: "object"
      properties:
        response:
          type: "string"
          description: "The greeting response"
      required: ["response"]
    timeout: 30
"""

    with tempfile.TemporaryDirectory() as temp_dir:
        spec_file = os.path.join(temp_dir, "test_agent.yaml")
        with open(spec_file, "w") as f:
            f.write(spec_content)

        # Create an invalid router without run method
        invalid_router_file = os.path.join(temp_dir, "InvalidRouter.py")
        with open(invalid_router_file, "w") as f:
            f.write("""
class InvalidRouter:
    def __init__(self, endpoint: str, model: str, config: dict):
        self.endpoint = endpoint
        self.model = model
        self.config = config
    # No run method!
""")

        # Generate the agent code
        with open(spec_file) as f:
            spec_data = yaml.safe_load(f)
        agent_name = spec_data["agent"]["name"]
        class_name = agent_name  # or use a function to convert to PascalCase if needed

        generate_agent_code(Path(temp_dir), spec_data, agent_name, class_name)

        # Test that the generated code contains the expected error handling
        agent_file = os.path.join(temp_dir, "agent.py")
        with open(agent_file, "r") as f:
            agent_code = f.read()

        # Verify the custom router validation logic is included
        assert "hasattr(router, 'run')" in agent_code, "Should check for run method"
        assert (
            "AttributeError" in agent_code
        ), "Should raise AttributeError for missing run method"

        # Test that the InvalidRouter class doesn't have a run method
        import sys

        sys.path.insert(0, temp_dir)

        try:
            from InvalidRouter import InvalidRouter

            router = InvalidRouter("http://test", "test-model", {})

            # This should raise an AttributeError
            with pytest.raises(AttributeError):
                router.run("test prompt")
        finally:
            sys.path.pop(0)
