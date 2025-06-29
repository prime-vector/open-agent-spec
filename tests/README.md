# OAS CLI Tests

This directory contains the test suite for the OAS CLI.

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_validators.py   # Validator tests
│   ├── test_generators.py   # Generator tests
│   └── ...
├── integration/            # Integration tests
│   ├── test_templates.py   # Template generation and execution tests
│   ├── test_cli.py         # CLI command tests
│   └── fixtures/           # Test fixtures and data
└── conftest.py            # pytest configuration and fixtures
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run only unit tests
```bash
pytest tests/unit/
```

### Run only integration tests
```bash
pytest tests/integration/
```

### Run specific test file
```bash
pytest tests/integration/test_templates.py
```

### Run with verbose output
```bash
pytest tests/ -v
```

## Integration Tests

Integration tests verify that the OAS CLI can:
- Generate agents from templates
- Install dependencies correctly
- Import and instantiate generated agents
- Execute basic functionality

### Template Tests

The `test_templates.py` file tests all available templates:
- `minimal-agent.yaml` - Basic single task agent
- `minimal-multi-task-agent.yaml` - Multi-step agent
- `minimal-agent-tool-usage.yaml` - Agent with tool usage

### Running Integration Tests Locally

```bash
# Run the integration test script directly
python tests/integration/test_templates.py

# Or use pytest
pytest tests/integration/test_templates.py -v
```

## GitHub Actions

Integration tests are automatically run on:
- Push to main/develop branches
- Pull requests to main branch
- Manual workflow dispatch

The GitHub workflow includes:
- Template generation tests
- Agent execution tests
- Tool-specific functionality tests
- Template validation tests
