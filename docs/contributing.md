# Contributor Guide

Thank you for your interest in improving this project. This project is open-source under the MIT license and welcomes contributions in the form of bug reports, feature requests, and pull requests.

## Resources

- [Source Code](https://github.com/Verdenroz/chimeric)
- [Documentation](https://verdenroz.github.io/chimeric/)
- [Issue Tracker](https://github.com/Verdenroz/chimeric/issues)
- [Code of Conduct](https://github.com/Verdenroz/chimeric/blob/main/CODE_OF_CONDUCT.md)

## Reporting Issues

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

## Development Setup

**Requirements:** Python 3.11, 3.12, 3.13 (use pyenv for multiple versions)

```bash
# Install Python versions
pyenv install 3.11 3.12 3.13

# Install development tools
pip install uv nox

# Install package with dev dependencies
make install
# or: uv sync --all-extras --dev
```

## Testing

```bash
# Run all tests
make test

# Run unit tests
make test-unit

# Run integration tests
make test-integration

# Cross-version testing
make nox
```

## Submitting Changes

**Requirements for acceptance:**

- All tests pass
- Maintain 100% code coverage
- Update documentation for new features
- Follow code style (run `make lint`)

**Process:**

1. Open an issue to discuss your approach
2. Fork and create a feature branch
3. Make your changes with tests
4. Run `make lint` before committing
5. Open a pull request
