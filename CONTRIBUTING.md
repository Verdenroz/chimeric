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

- **Unit Tests** (`tests/unit/`): Fast tests with mocking, must achieve 80% coverage
- **Integration Tests** (`tests/integration/`): Real API calls recorded with VCR cassettes for reproducibility

### Running Tests

```bash
# All tests
make test

# Unit tests only (fast)
make test-unit

# Integration tests (requires API keys or existing cassettes)
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

We maintain 80% test coverage for unit tests. Coverage reports are generated automatically:
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

### Architecture Overview

Chimeric uses a lightweight adapter system with no provider SDKs:

- **`src/chimeric/config.py`** — `PROVIDER_REGISTRY`: one `ProviderConfig` entry per provider
- **`src/chimeric/adapters/`** — stateless adapter classes that handle provider-specific serialization
- **`src/chimeric/http.py`** — shared HTTP transport (sync + async via httpx)

There are four adapters covering all eight providers:

| Adapter | Providers |
|---|---|
| `openai` | OpenAI, Groq, Cerebras, xAI Grok, OpenRouter |
| `anthropic` | Anthropic |
| `google` | Google Gemini |
| `cohere` | Cohere |

### Adding a New Provider That Shares an Existing Wire Format

If the new provider speaks the OpenAI (or another existing) API format, add a single entry to `PROVIDER_REGISTRY` in `src/chimeric/config.py`:

```python
"myprovider": ProviderConfig(
    name="myprovider",
    base_url="https://api.myprovider.com/v1",
    adapter="openai",               # reuse the OpenAI adapter
    api_key_env_vars=("MYPROVIDER_API_KEY",),
),
```

Then:
1. Add the provider key to `Chimeric.__init__()` and the `explicit` dict in `src/chimeric/chimeric.py`
2. Add integration tests in `tests/integration/`
3. Add the optional dependency to `pyproject.toml` if needed

### Adding a Provider With a New Wire Format

If the provider uses a different API format, create a new adapter:

1. **Add a `ProviderConfig`** entry to `PROVIDER_REGISTRY`
2. **Create `src/chimeric/adapters/myprovider.py`** implementing the `Adapter` protocol from `adapters/base.py`:
   - `build_request_body()` — converts `Message` list + tool list to provider JSON
   - `parse_response()` — converts provider JSON to `CompletionResponse`
   - `parse_sse_event()` — converts one SSE line to `StreamChunk`
   - `parse_tool_calls()` — extracts tool calls from a non-streaming response
   - `finalize_stream()` — extracts tool calls accumulated during streaming
   - `build_tool_result_messages()` — appends tool results to the message list
   - `parse_models_response()` — converts the models endpoint response to `list[ModelSummary]`
3. **Register the adapter** in `src/chimeric/adapters/__init__.py`
4. **Add tests** in `tests/unit/adapters/` and `tests/integration/`

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
