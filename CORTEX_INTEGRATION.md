# Cortex Intelligence Engine Integration

This document describes how to use the Cortex intelligence engine with Open Agent Spec (OAS).

## Overview

Cortex is an advanced intelligence engine that combines multiple AI capabilities:
- **Layer 3 Intelligence**: Advanced reasoning and multi-layered analysis
- **ONNX Runtime**: Optional performance optimization
- **Multi-Engine Integration**: Combines OpenAI and Claude capabilities
- **Advanced Analysis**: Deep problem breakdown and creative solution generation

## Installation

Add the Cortex package to your requirements:

```bash
pip install cortex-intelligence
```

Or add to your `requirements.txt`:

```
cortex-intelligence
```

## Configuration

### Basic Configuration

```yaml
intelligence:
  type: llm
  engine: cortex
  model: cortex-intelligence
  config:
    enable_layer3: true
    enable_onnx: false
    openai_api_key: ${OPENAI_API_KEY}
    claude_api_key: ${CLAUDE_API_KEY}
    temperature: 0.2
    max_tokens: 1500
```

### Configuration Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `enable_layer3` | boolean | Enable Layer 3 intelligence capabilities | `true` |
| `enable_onnx` | boolean | Enable ONNX runtime optimization | `false` |
| `openai_api_key` | string | OpenAI API key for integration | Required |
| `claude_api_key` | string | Claude API key for integration | Required |
| `temperature` | number | Response randomness (0.0-2.0) | `0.2` |
| `max_tokens` | integer | Maximum response length | `1500` |

### Environment Variables

Set these environment variables:

```bash
export OPENAI_API_KEY="sk-your-openai-key"
export CLAUDE_API_KEY="sk-ant-your-claude-key"
```

## Examples

### Minimal Cortex Agent

```yaml
open_agent_spec: 1.0.8

agent:
  name: cortex-minimal-agent
  description: Minimal agent using Cortex intelligence engine
  role: chat

intelligence:
  type: llm
  engine: cortex
  model: cortex-intelligence
  config:
    enable_layer3: true
    enable_onnx: false
    openai_api_key: ${OPENAI_API_KEY}
    claude_api_key: ${CLAUDE_API_KEY}
    temperature: 0.2
    max_tokens: 1000

tasks:
  respond:
    description: Respond to user input using Cortex intelligence
    timeout: 60
    input:
      type: object
      properties:
        message:
          type: string
          description: User message to respond to
          minLength: 1
          maxLength: 1000
      required: [message]
    output:
      type: object
      properties:
        response:
          type: string
          description: AI response to the user message
          minLength: 1
        reasoning:
          type: string
          description: Brief explanation of the reasoning process
      required: [response, reasoning]

prompts:
  system: >
    You are a helpful AI assistant powered by Cortex intelligence engine.
    Provide thoughtful, well-reasoned responses to user queries.
    
    Always explain your reasoning process briefly in the 'reasoning' field.
    Keep responses concise but informative.

  user: >
    User message: {{ input.message }}
    
    Please provide a helpful response using your Cortex intelligence capabilities.

behavioural_contract:
  version: "1.0.0"
  description: "Minimal Cortex agent behavioral contract"
  behavioural_flags:
    reasoning_depth: "moderate"
    creativity_level: "moderate"
    evidence_based: "moderate"
  response_contract:
    output_format:
      required_fields: ["response", "reasoning"]
```

### Advanced Cortex Agent

For a more comprehensive example, see `oas_cli/templates/cortex-intelligence-agent.yaml`.

## Usage

### 1. Create Your Agent Spec

Create a YAML file with your agent specification using the Cortex engine.

### 2. Generate Agent Code

```bash
oas generate your-agent-spec.yaml
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

```bash
export OPENAI_API_KEY="your-key"
export CLAUDE_API_KEY="your-key"
```

### 5. Run Your Agent

```python
from your_agent import YourAgent

agent = YourAgent()
result = agent.respond(message="Hello, how are you?")
print(result)
```

## Features

### Layer 3 Intelligence

Layer 3 intelligence provides advanced reasoning capabilities:

- **Multi-layered Analysis**: Break down complex problems into manageable components
- **Systematic Reasoning**: Explore different angles and perspectives
- **Evidence-based Conclusions**: Base responses on logical analysis
- **Context Awareness**: Consider multiple scenarios and contexts

### ONNX Runtime Optimization

Optional ONNX runtime optimization for improved performance:

```yaml
config:
  enable_onnx: true  # Enable for performance optimization
```

### Multi-Engine Integration

Cortex intelligently combines OpenAI and Claude capabilities:

- **OpenAI Integration**: Access to GPT models for general reasoning
- **Claude Integration**: Access to Claude models for detailed analysis
- **Smart Routing**: Automatically selects the best engine for each task
- **Fallback Handling**: Graceful degradation if one service is unavailable

## Best Practices

### 1. Temperature Settings

- **Low (0.1-0.3)**: For analytical tasks requiring consistency
- **Medium (0.4-0.7)**: For creative tasks with some variability
- **High (0.8-1.0)**: For highly creative and exploratory tasks

### 2. Token Limits

- **Short responses**: 500-1000 tokens
- **Medium analysis**: 1000-2000 tokens
- **Comprehensive analysis**: 2000+ tokens

### 3. Layer 3 Usage

- **Enable for**: Complex problem solving, multi-step reasoning
- **Disable for**: Simple Q&A, basic text generation

### 4. ONNX Optimization

- **Enable for**: Production environments, high-throughput scenarios
- **Disable for**: Development, testing, or when debugging

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure both OpenAI and Claude API keys are set
   - Verify keys have sufficient credits and permissions

2. **Layer 3 Not Working**
   - Check that `enable_layer3: true` is set
   - Verify API keys are valid and accessible

3. **Performance Issues**
   - Enable ONNX runtime: `enable_onnx: true`
   - Adjust token limits based on your needs
   - Consider using lower temperature for faster responses

### Debug Mode

Enable debug logging to troubleshoot issues:

```yaml
logging:
  enabled: true
  level: "DEBUG"
  format_style: "emoji"
  include_timestamp: true
```

## Integration with DACP

Cortex integrates seamlessly with the DACP (Distributed Agent Control Protocol) framework:

- **Automatic Integration**: No additional configuration required
- **Tool Support**: Full support for DACP tools and workflows
- **Behavioral Contracts**: Compatible with OAS behavioral contracts
- **Multi-Task Support**: Supports complex multi-step workflows

## Examples Directory

See the `examples/` directory for working Cortex agent specifications:

- `cortex-minimal-agent.yaml`: Basic Cortex agent
- `oas_cli/templates/cortex-intelligence-agent.yaml`: Advanced Cortex agent

## Testing

Run the Cortex integration tests:

```bash
python -m pytest tests/test_cortex_integration.py -v
```

## Support

For issues with Cortex integration:

1. Check the test suite for examples
2. Verify your configuration matches the schema
3. Ensure all required dependencies are installed
4. Check that API keys are valid and accessible

## Future Enhancements

Planned improvements for Cortex integration:

- **Custom Model Support**: Support for additional AI models
- **Advanced Routing**: Intelligent model selection based on task type
- **Performance Metrics**: Built-in performance monitoring
- **Batch Processing**: Support for processing multiple requests
- **Caching**: Intelligent response caching for improved performance 