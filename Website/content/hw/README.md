# Hello World Agent

A simple agent that responds with a greeting

## Usage

```bash
pip install -r requirements.txt
cp .env.example .env
python agent.py
```

## Tasks

### Greet

Say hello to a person by name

#### Input:
- type: object
- properties: {'name': {'type': 'string', 'description': 'The name of the person to greet', 'minLength': 1, 'maxLength': 100}}
- required: ['name']

#### Output:
- type: object
- properties: {'response': {'type': 'string', 'description': 'The greeting response', 'minLength': 1}}
- required: ['response']




## Example Usage

```python
from agent import HelloWorldAgent

agent = HelloWorldAgent()
# Example usage
task_name = "greet"
if task_name:
    result = getattr(agent, task_name.replace("-", "_"))(
        type="example_type", properties="example_properties", required="example_required"
    )
    print(result)
```
