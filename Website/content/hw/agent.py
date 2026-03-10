import json
import logging
import os

import dacp
from dacp import invoke_intelligence, parse_with_fallback
from dacp.orchestrator import Orchestrator
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel

load_dotenv()

log = logging.getLogger(__name__)

ROLE = "Hello_World_Agent"

# Generate output models
class GreetOutput(BaseModel):
    """The greeting response"""
    response: str


# Task functions

def parse_greet_output(response) -> GreetOutput:
    """Parse LLM response into GreetOutput using DACP's enhanced parser.

    Args:
        response: Raw response from the LLM (str or dict)

    Returns:
        Parsed and validated GreetOutput instance

    Raises:
        ValueError: If the response cannot be parsed
    """
    if isinstance(response, GreetOutput):
        return response

    # Parse JSON string if needed
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f'Failed to parse JSON response: {e}')

    # Use DACP's enhanced JSON parser with fallback support
    try:
        defaults = {}
        result = parse_with_fallback(
            response=response,
            model_class=GreetOutput,
            **defaults
        )
        return result
    except Exception as e:
        raise ValueError(f'Error parsing response with DACP parser: {e}')


def greet(name: str, memory_summary: str = '') -> GreetOutput:
    """Process greet task.

    Args:
        name: {'type': 'string', 'description': 'The name of the person to greet', 'minLength': 1, 'maxLength': 100}
        memory_summary: Optional memory context for the task

    Returns:
        GreetOutput
    """
    # Define memory configuration
    memory_config = {
        "enabled": False,
        "format": "string",
        "usage": "prompt-append",
        "required": False,
        "description": ""
    }

    # Define output format description
    output_format = """
- response (required): string
  The greeting response
"""

    # Load and render the prompt template
    prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
    env = Environment(loader=FileSystemLoader([".", prompts_dir]))
    try:
        template = env.get_template("greet.jinja2")
    except FileNotFoundError:
        log.warning("Task-specific prompt template not found, using default template")
        template = env.get_template("agent_prompt.jinja2")

    # Create input dictionary for template
    input_dict = {
        "name": name
    }

    # Render the prompt with all necessary context - pass variables directly for template access
    prompt = template.render(
        input=input_dict,
        memory_summary=memory_summary if memory_config['enabled'] else '',
        output_format=output_format,
        memory_config=memory_config,
        **input_dict  # Also pass variables directly for template access
    )

    # Configure intelligence for DACP
    intelligence_config = {
    "engine": "openai",
    "model": "gpt-4",
    "endpoint": "https://api.openai.com/v1",
    "temperature": 0.7,
    "max_tokens": 150
}

    # Call the LLM using DACP
    result = invoke_intelligence(prompt, intelligence_config)
    return parse_greet_output(result)



class HelloWorldAgent(dacp.Agent):
    def __init__(self, agent_id: str, orchestrator: Orchestrator):
        super().__init__()
        self.agent_id = agent_id
        orchestrator.register_agent(agent_id, self)
        self.model = "gpt-4"

        # Embed YAML config as dict during generation
        self.config = {
            "logging": {
                "enabled": True,
                "level": "INFO",
                "format_style": "emoji",
                "include_timestamp": True,
                "log_file": None,
                "env_overrides": {}
            },
            "intelligence": {
                "engine": "openai",
                "model": "gpt-4",
                "endpoint": "https://api.openai.com/v1",
                "config": {
                    "temperature": 0.7,
                    "max_tokens": 150
                }
            }
        }

        # Setup DACP logging FIRST
        self.setup_logging()


    def handle_message(self, message: dict) -> dict:
        """
        Handles incoming messages from the orchestrator.
        Processes messages based on the task specified and routes to appropriate agent methods.
        """
        task = message.get("task")
        if not task:
            return {"error": "Missing required field: task"}

        # Map task names to method names (replace hyphens with underscores)
        method_name = task.replace("-", "_")

        # Check if the method exists on this agent
        if not hasattr(self, method_name):
            return {"error": f"Unknown task: {task}"}

        try:
            # Get the method and extract its parameters (excluding 'self')
            method = getattr(self, method_name)

            # Call the method with the message parameters (excluding 'task')
            method_params = {k: v for k, v in message.items() if k != "task"}
            result = method(**method_params)

            # Handle both Pydantic models and dictionaries
            if hasattr(result, 'model_dump'):
                return result.model_dump()
            else:
                return result

        except TypeError as e:
            return {"error": f"Invalid parameters for task {task}: {e!s}"}
        except Exception as e:
            return {"error": f"Error executing task {task}: {e!s}"}



    def setup_logging(self):
        """Configure DACP logging from YAML configuration."""
        logging_config = self.config.get('logging', {})

        if not logging_config.get('enabled', True):
            return

        # Process environment variable overrides
        env_overrides = logging_config.get('env_overrides', {})

        level = logging_config.get('level', 'INFO')
        if 'level' in env_overrides:
            level = os.getenv(env_overrides['level'], level)

        format_style = logging_config.get('format_style', 'emoji')
        if 'format_style' in env_overrides:
            format_style = os.getenv(env_overrides['format_style'], format_style)

        log_file = logging_config.get('log_file')
        if 'log_file' in env_overrides:
            log_file = os.getenv(env_overrides['log_file'], log_file)

        # Create log directory if needed
        if log_file:
            from pathlib import Path
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        # Configure DACP logging
        dacp.setup_dacp_logging(
            level=level,
            format_style=format_style,
            include_timestamp=logging_config.get('include_timestamp', True),
            log_file=log_file
        )



    def greet(self, name) -> GreetOutput:
        """Process greet task."""
        memory_summary = self.get_memory() if hasattr(self, 'get_memory') else ""
        return greet(name, memory_summary=memory_summary)



def main():
    # Example usage - in production, you would get these from your orchestrator setup
    from dacp.orchestrator import Orchestrator

    orchestrator = Orchestrator()
    agent = HelloWorldAgent("example-agent-id", orchestrator)

    # Example usage with greet task: greet
    result = agent.greet(name="example_name")
    # Handle both Pydantic models and dictionaries
    if hasattr(result, 'model_dump'):
        print(json.dumps(result.model_dump(), indent=2))
    else:
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
