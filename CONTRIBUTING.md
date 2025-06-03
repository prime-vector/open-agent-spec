# Contributing to Open Agent Spec (OAS) CLI

Thank you for your interest in contributing to OAS CLI! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Creating Agent Spec Files

The Open Agent Spec (OAS) uses YAML files to define agent configurations. Here's how to create your own:

#### Basic Structure
```yaml
info:
  name: my-agent
  description: A fantastic agent that changes the world

intelligence:
  endpoint: https://api.openai.com/v1
  model: gpt-4
  config:
    temperature: 0.7
    max_tokens: 1000
```

#### Required Fields

1. `info` section:
   - `name`: A unique identifier for your agent (will be converted to snake_case for Python)
   - `description`: A clear description of what your agent does

2. `intelligence` section:
   - `endpoint`: The API endpoint for your LLM provider
   - `model`: The specific model to use (e.g., "gpt-4", "gpt-3.5-turbo")
   - `config`: Model-specific configuration
     - `temperature`: Controls randomness (0.0 to 1.0)
     - `max_tokens`: Maximum length of the response

#### Example Use Cases

1. Trading Agent:
```yaml
info:
  name: market-analyzer
  description: An agent that analyzes market signals and provides trading recommendations

intelligence:
  endpoint: https://api.openai.com/v1
  model: gpt-4
  config:
    temperature: 0.3  # Lower temperature for more consistent outputs
    max_tokens: 2000  # Longer responses for detailed analysis
```

2. Content Generator:
```yaml
info:
  name: content-creator
  description: An agent that generates creative content based on prompts

intelligence:
  endpoint: https://api.openai.com/v1
  model: gpt-4
  config:
    temperature: 0.8  # Higher temperature for more creative outputs
    max_tokens: 1000
```

#### Best Practices

1. **Naming**:
   - Use kebab-case for the agent name (e.g., `market-analyzer`)
   - Make names descriptive and unique
   - Avoid special characters

2. **Description**:
   - Be clear and concise
   - Include the agent's primary purpose
   - Mention any key capabilities

3. **Configuration**:
   - Adjust temperature based on task:
     - Lower (0.1-0.3) for analytical tasks
     - Higher (0.7-0.9) for creative tasks
   - Set max_tokens based on expected response length
   - Use appropriate model for your use case

4. **Validation**:
   - Test your spec file with `oas init --dry-run`
   - Ensure all required fields are present
   - Check that values are of correct types

### Reporting Bugs

- Check if the bug has already been reported in the Issues section
- Use the bug report template
- Include detailed steps to reproduce
- Include expected and actual behavior
- Add screenshots if applicable

### Suggesting Features

- Check if the feature has already been suggested
- Use the feature request template
- Explain the problem you're trying to solve
- Describe your proposed solution
- Include any relevant examples

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/aswhitehouse/oas-cli.git
cd oas-cli

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Code Style

- Follow [PEP 8](https://pep8.org/) guidelines
- Use type hints
- Write docstrings for all functions and classes
- Keep functions small and focused
- Write meaningful commit messages

### Testing

- Write tests for new features
- Ensure all tests pass
- Maintain or improve test coverage
- Run tests with: `pytest`

### Documentation

- Update README.md if needed
- Add docstrings to new functions
- Update any relevant documentation
- Keep comments clear and helpful

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the documentation if needed
3. The PR will be merged once you have the sign-off of at least one maintainer
4. Make sure all CI checks pass

## Questions?

Feel free to open an issue for any questions or concerns. We're here to help!

## License

By contributing, you agree that your contributions will be licensed under the project's AGPLv3 License.
