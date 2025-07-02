import json
import pytest
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any
from oas_cli.generators import generate_agent_code


class MockCustomLLMRouter:
    """A simple mock LLM router for testing"""

    def __init__(self, endpoint: str, model: str, config: dict):
        self.endpoint = endpoint
        self.model = model
        self.config = config

    def run(self, prompt: str, **kwargs) -> str:
        """Mock run method that returns a JSON string"""
        name = kwargs.get("name", "World")
        return json.dumps({"response": f"Hello {name}!"})


class InvalidRouter:
    """A router without a run method for testing error handling"""

    def __init__(self, endpoint: str, model: str, config: dict):
        self.endpoint = endpoint
        self.model = model
        self.config = config

    # No run method!


@pytest.fixture
def base_spec() -> Dict[str, Any]:
    """Base spec template for custom LLM router tests"""
    return {
        "spec_version": "1.0.4",
        "agent": {
            "name": "TestAgent",
            "description": "A test agent with custom LLM router",
            "role": "assistant",
        },
        "intelligence": {
            "engine": "custom",
            "endpoint": "http://localhost:1234/invoke",
            "model": "test-model",
            "config": {},
            "module": "MockCustomLLMRouter.MockCustomLLMRouter",
        },
        "tasks": {
            "greet": {
                "description": "Greet someone by name",
                "input": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name to greet",
                            "minLength": 1,
                            "maxLength": 100,
                        }
                    },
                    "required": ["name"],
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "string",
                            "description": "The greeting response",
                        }
                    },
                    "required": ["response"],
                },
                "timeout": 30,
                "metadata": {"category": "communication", "priority": "normal"},
            }
        },
    }


@pytest.fixture
def temp_project():
    """Create a temporary project directory with all necessary files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create prompts directory
        prompts_dir = temp_path / "prompts"
        prompts_dir.mkdir(exist_ok=True)

        # Create basic prompt templates
        (prompts_dir / "greet.jinja2").write_text("Hello {{ input.name }}!")
        (prompts_dir / "agent_prompt.jinja2").write_text("{{ input }}")

        yield temp_path


def create_spec_file(spec_data: Dict[str, Any], temp_dir: Path) -> Path:
    """Create a spec file in the temp directory"""
    spec_file = temp_dir / "test_agent.yaml"
    with open(spec_file, "w") as f:
        yaml.dump(spec_data, f)
    return spec_file


def generate_test_agent(spec_data: Dict[str, Any], temp_dir: Path) -> Path:
    """Generate agent code from spec data"""
    agent_name = spec_data["agent"]["name"]
    class_name = agent_name

    generate_agent_code(temp_dir, spec_data, agent_name, class_name)
    return temp_dir / "agent.py"


def create_mock_router_file(temp_dir: Path) -> Path:
    """Create the MockCustomLLMRouter.py file"""
    router_file = temp_dir / "MockCustomLLMRouter.py"
    router_file.write_text(
        """
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
"""
    )
    return router_file


def create_invalid_router_file(temp_dir: Path) -> Path:
    """Create the InvalidRouter.py file"""
    router_file = temp_dir / "InvalidRouter.py"
    router_file.write_text(
        """
class InvalidRouter:
    def __init__(self, endpoint: str, model: str, config: dict):
        self.endpoint = endpoint
        self.model = model
        self.config = config
    # No run method!
"""
    )
    return router_file


def verify_agent_code(agent_file: Path) -> str:
    """Verify the generated agent code contains expected elements"""
    agent_code = agent_file.read_text()

    assert (
        "import importlib" in agent_code
    ), "Should import importlib for dynamic loading"
    assert (
        "load_custom_llm_router" in agent_code
    ), "Should have custom router loading function"
    assert "CustomLLMRouter" in agent_code, "Should reference CustomLLMRouter"

    return agent_code


def test_custom_llm_router_integration(base_spec, temp_project):
    """Test that a custom LLM router works correctly with the generated agent"""
    # Create spec file
    create_spec_file(base_spec, temp_project)

    # Generate agent code
    agent_file = generate_test_agent(base_spec, temp_project)
    assert agent_file.exists(), "Agent file should be created"

    # Verify generated code
    verify_agent_code(agent_file)

    # Create mock router
    create_mock_router_file(temp_project)

    # Test agent execution
    import sys

    sys.path.insert(0, str(temp_project))

    try:
        from agent import TestAgent

        # Mock orchestrator for testing since dacp might not be available in CI
        class MockOrchestrator:
            def register_agent(self, agent_id, agent):
                pass

        orchestrator = MockOrchestrator()
        agent = TestAgent("test-agent-id", orchestrator)
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
        sys.path.pop(0)


def test_custom_llm_router_error_handling(base_spec, temp_project):
    """Test error handling when custom LLM router is not available"""
    # Modify spec to use non-existent module
    spec_data = base_spec.copy()
    spec_data["intelligence"]["module"] = "NonExistentModule.NonExistentClass"

    # Create spec file and generate agent
    create_spec_file(spec_data, temp_project)
    agent_file = generate_test_agent(spec_data, temp_project)

    # Verify generated code contains the module reference
    agent_code = agent_file.read_text()
    assert (
        "NonExistentModule.NonExistentClass" in agent_code
    ), "Should reference the specified module"

    # Test that importing the module fails as expected
    import sys

    sys.path.insert(0, str(temp_project))

    try:
        with pytest.raises(ImportError):
            import importlib

            importlib.import_module("NonExistentModule")
    finally:
        sys.path.pop(0)


def test_custom_llm_router_missing_run_method(base_spec, temp_project):
    """Test error handling when custom LLM router doesn't have a run method"""
    # Modify spec to use invalid router
    spec_data = base_spec.copy()
    spec_data["intelligence"]["module"] = "InvalidRouter.InvalidRouter"

    # Create invalid router file
    create_invalid_router_file(temp_project)

    # Generate agent code
    create_spec_file(spec_data, temp_project)
    agent_file = generate_test_agent(spec_data, temp_project)

    # Verify generated code contains validation logic
    agent_code = agent_file.read_text()
    assert "hasattr(router, 'run')" in agent_code, "Should check for run method"
    assert (
        "AttributeError" in agent_code
    ), "Should raise AttributeError for missing run method"

    # Test that the InvalidRouter class doesn't have a run method
    import sys

    sys.path.insert(0, str(temp_project))

    try:
        from InvalidRouter import InvalidRouter

        router = InvalidRouter("http://test", "test-model", {})

        # This should raise an AttributeError
        with pytest.raises(AttributeError):
            router.run("test prompt")
    finally:
        sys.path.pop(0)
