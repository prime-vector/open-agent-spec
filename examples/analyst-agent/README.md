# Analyst Agent



## Usage

```bash
pip install -r requirements.txt
cp .env.example .env
python agent.py
```

## Tasks

### Analyze-Signal

Analyze market signal and provide recommendation

#### Input:
- properties: {'symbol': {'type': 'string'}, 'signal_data': {'type': 'string'}, 'timestamp': {'type': 'string'}}

#### Output:
- recommendation: string
- confidence: float
- rationale: string

## Memory Support



### Configuration

- Format: string

- Usage: prompt-append

- Required: false


To implement memory support, override the `get_memory()` method in the agent class.

## Behavioural Contract


This agent is governed by the following behavioural contract policy:



Refer to `behavioural_contracts` for enforcement logic.


## Example Usage

```python
from agent import Analyst-agent

agent = Analyst-agent()
# Example usage
task_name = "analyze-signal"
if task_name:
    result = getattr(agent, task_name.replace("-", "_"))(
        properties="example_properties"
    )
    print(result)
```
