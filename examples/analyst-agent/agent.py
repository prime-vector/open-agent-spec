from typing import Dict, Any
import openai
import json
from behavioural_contracts import behavioural_contract
from jinja2 import Environment, FileSystemLoader

ROLE = "Analyst_Agent"


@behavioural_contract({"version": "0.1.0", "description": "Analyze market signal and provide recommendation", "role": "analyst_agent", "memory": {"enabled": "false", "format": "string", "usage": "prompt-append", "required": "false", "description": ""}, "policy": {"pii": "false", "compliance_tags": [], "allowed_tools": []}})
def analyze_signal(symbol: str, signal_data: str, timestamp: str, memory_summary: str = '') -> Dict[str, Any]:
    """Process analyze-signal task.

    Args:
        symbol: {'type': 'string'}
        signal_data: {'type': 'string'}
        timestamp: {'type': 'string'}
        memory_summary: Optional memory context for the task

    Returns:
        Dict[str, Any]
    """
    # Define task_def for this function
    task_def = {
        "output": {"recommendation": "string", "confidence": "float", "rationale": "string"}
    }
    
    # Load and render the prompt template
    env = Environment(loader=FileSystemLoader("prompts"))
    try:
        template = env.get_template("analyze_signal.jinja2")
    except:
        template = env.get_template("agent_prompt.jinja2")
    
    # Render the prompt with all necessary context
    prompt = template.render(
        input={
            "properties": "properties"
},
        memory_summary=memory_summary,
        indicators_summary="",  # TODO: Implement indicators if needed
        output=task_def["output"],
        memory_config={'enabled': 'false', 'format': 'string', 'usage': 'prompt-append', 'required': 'false', 'description': ''}
    )

    client = openai.OpenAI(
        base_url="https://api.openai.com/v1",
        api_key=openai.api_key
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a professional analyst_agent."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    
    result = response.choices[0].message.content

    # Parse the response into the expected output format
    output_dict = {}
    output_fields = list(task_def.get('output', {}).keys())
    
    # Try JSON parsing first
    try:
        # Look for JSON block in the response
        json_start = result.find("{")
        json_end = result.rfind("}") + 2
        if json_start >= 0 and json_end > json_start:
            json_str = result[json_start:json_end]
            parsed = json.loads(json_str)
            if all(key in parsed for key in output_fields):
                return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Fall back to line-based parsing
    lines = result.strip().split('\n')
    current_key = None
    current_value = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line starts a new field
        for key in output_fields:
            if line.startswith(key + ":"):
                # Save previous field if exists
                if current_key and current_value:
                    output_dict[current_key] = ' '.join(current_value).strip()
                # Start new field
                current_key = key
                current_value = [line[len(key)+1:].strip()]
                break
        else:
            # If no new field found, append to current value
            if current_key:
                current_value.append(line)
    
    # Save the last field
    if current_key and current_value:
        output_dict[current_key] = ' '.join(current_value).strip()
    
    # Validate all required fields are present
    missing_fields = [field for field in output_fields if field not in output_dict]
    if missing_fields:
        print(f"Warning: Missing output fields: {missing_fields}")
        for field in missing_fields:
            output_dict[field] = ""  # Provide empty string for missing fields
    
    return output_dict


class AnalystAgent:
    def __init__(self, api_key: str | None = None):
        self.model = "gpt-3.5-turbo"
        if api_key:
            openai.api_key = api_key


    def analyze_signal(self, symbol, signal_data, timestamp) -> Dict[str, Any]:
        """Process analyze-signal task."""
        memory_summary = self.get_memory() if hasattr(self, 'get_memory') else ""
        return analyze_signal(symbol, signal_data, timestamp, memory_summary=memory_summary)


    def get_memory(self) -> str:
        """Get memory for the current context.
        
        This is a stub method that should be implemented by the developer.
        The memory format and retrieval mechanism are not prescribed by OAS.
        
        Returns:
            str: Memory string in the format specified by the spec
        """
        return ""  # Implement your memory retrieval logic here


def main():
    agent = AnalystAgent()
    # Example usage
    if "analyze-signal":
        result = getattr(agent, "analyze-signal".replace("-", "_"))(
            symbol="example_symbol", signal_data="example_signal_data", timestamp="example_timestamp"
        )
        print(json.dumps(result, indent=2))
    else:
        print("No tasks defined in the spec file")

if __name__ == "__main__":
    main()
