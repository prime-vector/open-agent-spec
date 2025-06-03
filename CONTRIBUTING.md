# Contributing to Open Agent Spec (OAS) CLI

Thank you for your interest in contributing to OAS CLI! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

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
