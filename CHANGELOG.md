# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.9] - 2025-03-04

### Added
- Interactive web playground for spec editing and agent generation
- JSON output support for CLI
- Additional spec examples and templates
- Rate limiting support
- Mobile-responsive website view

### Fixed
- Broken CLI caused by BCE validator
- Import error for Vercel functions
- Handler issues for serverless deployment
- Ruff linting compliance across codebase

## [1.0.8] - 2025-01-15

### Added
- Section 5: Public API (`generate`, `validate_spec`) with normalized file errors
- Serverless endpoint for artifact generation
- Cortex intelligence engine integration
- Grok (xAI) engine support
- Security agent templates (threat analyzer, risk assessor, incident responder)
- GitHub Actions error analysis workflow and templates
- Comprehensive test reporting (Allure, HTML, coverage)
- Pre-commit hooks configuration
- CODEOWNERS file
- Pull request template

### Changed
- Canonical spec version field is now `open_agent_spec` (replaces `spec_version`)
- Updated schema validation for behavioural contracts

### Fixed
- Contribution guidelines and development setup documentation
- Test result file tracking (added to .gitignore)

## [1.0.7] - 2024-12-01

### Added
- Initial public release of Open Agent Spec CLI
- Support for OpenAI and Anthropic engines
- Local and custom engine placeholders
- YAML-based agent specification format
- Jinja2 template-based code generation
- Minimal agent templates (single-task, multi-task, tool usage)
- Behavioural contract support
- JSON Schema validation for specs
- CLI with `oas init` command
- Dry-run mode for previewing generated files
- PyPI package publishing workflow
