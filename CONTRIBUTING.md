# Contributing to Chimeric

We welcome contributions to Chimeric! This guide will help you get started with development and ensure your contributions align with the project's standards.

## Development Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) for dependency management

### Getting Started

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/yourusername/chimeric.git
   cd chimeric
   ```

2. **Install development dependencies**:
   ```bash
   make install
   # or directly with uv
   uv sync --all-extras --dev
   ```

3. **Verify your setup**:
   ```bash
   make test-unit
   ```

## Code Quality Standards

### Linting and Formatting

We use several tools to maintain code quality:

- **ruff**: Code formatting and linting
- **basedpyright**: Type checking
- **codespell**: Spell checking

Run all quality checks:
```bash
make lint
```

## Testing Guidelines

### Test Structure

- **Unit Tests** (`tests/unit/`): Fast tests with mocking, must achieve 100% coverage
- **Integration Tests** (`tests/integration/`): Real API calls with VCR cassettes for reproducibility

### Running Tests

```bash
# All tests
make test

# Unit tests only (fast)
make test-unit

# Integration tests (requires API keys)
make test-integration

# Provider-specific tests
make test-openai
make test-anthropic
# etc.

# Cross-version testing
make nox
```

### Writing Tests

1. **Unit tests** should mock external dependencies
2. **Integration tests** use VCR cassettes to record/replay API interactions
3. Add tests for new features and bug fixes
4. Ensure tests are deterministic and don't rely on external state

### Test Coverage

We maintain 100% test coverage for unit tests. Coverage reports are generated automatically:
```bash
nox --session=coverage
```

## Development Workflow

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Add tests** for new functionality

4. **Run the full test suite**:
   ```bash
   make test
   make lint
   ```

5. **Commit your changes** with a descriptive message

### Pull Request Process

1. **Update documentation** if needed
2. **Ensure all tests pass** and coverage is maintained
3. **Submit a pull request** with:
   - Clear description of changes
   - Reference to related issues
   - Test results confirmation

4. **Address review feedback** promptly

## Provider Development

### Adding New Providers

When adding support for a new LLM provider:

1. **Create provider module** in `src/chimeric/providers/newprovider/`
2. **Implement client classes** inheriting from base classes
3. **Add provider-specific tests** in `tests/unit/providers/` and `tests/integration/`
4. **Update documentation** and provider comparison tables
5. **Add optional dependency** to `pyproject.toml`

### Provider Architecture

Each provider must implement:
- Sync and async client classes
- Message format conversion
- Streaming support
- Error handling
- Tool/function calling (if supported)

## Documentation

### Code Documentation

- Use clear, descriptive docstrings for all public APIs
- Include type hints for all function parameters and return values
- Add examples for complex functionality

### User Documentation

- Keep documentation concise and focused
- Update relevant sections when adding features
- Test documentation examples to ensure they work

## Issue Reporting

We have issue templates to help you provide the information we need:

### Bug Reports

Use our [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.yml) which will guide you through providing:
- Python version and environment details
- Minimal code example to reproduce the issue
- Expected vs. actual behavior
- Relevant error messages and stack traces
- Affected providers

### Feature Requests

Use our [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.yml) to provide:
- Clear description of the proposed feature
- Problem statement and motivation
- Use cases and benefits
- Proposed API design with examples
- Priority level

### Other Issues

For questions or general discussions, please use [GitHub Discussions](https://github.com/Verdenroz/chimeric/discussions) rather than opening an issue.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md). We're committed to providing a welcoming and inclusive environment for all contributors.

## Getting Help

- **Documentation**: Check our [docs](https://verdenroz.github.io/chimeric/)
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions and ideas

Thank you for contributing to Chimeric!