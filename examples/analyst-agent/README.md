# Analyst Agent Example

This is a comprehensive example of an Open Agent Spec (OAS) implementation for a market analysis agent. It demonstrates the full range of OAS features and best practices.

## Features Demonstrated

- **Intelligence Configuration**: LLM setup with OpenAI
- **Task Definition**: Clear input/output specifications
- **Health Monitoring**: Rules for quality control
- **Memory Management**: Window-based context handling
- **Tool Integration**: REST API integration
- **Safety Controls**: Role locking and observation limits
- **Logging**: Configurable logging with field redaction

## Project Structure

```
analyst-agent/
├── agent.spec.yaml     # Main specification file
├── prompts/           # Prompt templates
│   └── analyst_prompt.jinja2
└── README.md          # This file
```

## Usage

1. Install the OAS CLI:
```bash
pip install oas-cli
```

2. Initialize the agent:
```bash
oas init --spec agent.spec.yaml --output ./agent
```

3. Set up environment:
```bash
cd agent
cp .env.example .env
# Edit .env with your OpenAI API key
```

4. Run the agent:
```bash
python agent.py
```

## Example Input

```json
{
  "symbol": "AAPL",
  "signal_data": "RSI: 70, MACD: Bullish, Volume: Above Average",
  "timestamp": "2024-03-20T12:00:00Z"
}
```

## Example Output

```json
{
  "recommendation": "SELL",
  "confidence": 0.85,
  "rationale": "RSI above 70 indicates overbought conditions, while MACD shows bullish momentum. Volume confirmation suggests potential reversal point."
}
```

## Configuration Details

### Intelligence
- Uses GPT-4 for analysis
- Temperature: 0.3 for consistent outputs
- Max tokens: 512 for concise responses

### Health Rules
- Monitors for empty rationales
- Tracks repeated outputs
- Implements teardown policy after 3 strikes

### Memory
- Ephemeral storage
- 5-item window for context
- Window-based strategy

### Safety
- Role locking enabled
- Token and call limits
- Fallback behavior configured

## Contributing

Feel free to use this example as a template for your own agents. See the main [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines. 