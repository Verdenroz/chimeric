# Makefile for easy development workflows.
# See development.md for docs.
# Note GitHub Actions call uv directly, not this Makefile.

.DEFAULT_GOAL := default

.PHONY: default install lint test test-unit test-integration nox nox-unit nox-integration test-deps upgrade build clean docs docs-deploy
.PHONY: test-openai test-anthropic test-google test-cerebras test-cohere test-grok test-groq
.PHONY: test-bare test-all-extras clean-cassettes help

default: install lint test

install:
	uv sync --all-extras --dev

lint:
	uv run python devtools/lint.py


test: test-unit test-integration
	@echo "✅ All tests passed"

test-unit: install
	uv run pytest tests/unit

test-integration:
	uv run nox -s test_integration

nox:
	uv run nox -s unit integration

nox-unit:
	uv run nox -s unit

nox-integration:
	uv run nox -s integration

# Provider-specific integration testing (initialization + behaviors)
test-openai:
	uv run nox -s test_openai

test-anthropic:
	uv run nox -s test_anthropic

test-google:
	uv run nox -s test_google

test-cerebras:
	uv run nox -s test_cerebras

test-cohere:
	uv run nox -s test_cohere

test-grok:
	uv run nox -s test_grok

test-groq:
	uv run nox -s test_groq

test-bare:
	uv run nox -s test_bare

test-all-extras:
	uv run nox -s test_all_extras

upgrade:
	uv sync --upgrade --all-extras --dev

build:
	uv build

clean: clean-cassettes
	-rm -rf dist/
	-rm -rf *.egg-info/
	-rm -rf .pytest_cache/
	-rm -rf .mypy_cache/
	-rm -rf .venv/
	-rm -rf tests/integration/cassettes/
	-rm -rf htmlcov/
	-rm -rf .coverage*
	-find . -type d -name "__pycache__" -exec rm -rf {} +

clean-cassettes:
	-rm -rf tests/integration/cassettes/
	@echo "🧹 VCR cassettes cleaned"

docs:
	uv run mkdocs serve

docs-deploy:
	uv run mkdocs gh-deploy

help:
	@echo "Chimeric Development Makefile"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make               - Install deps, lint, run tests"
	@echo ""
	@echo "📦 Installation:"
	@echo "  make install       - Install all dependencies"
	@echo "  make upgrade       - Upgrade all dependencies"
	@echo ""
	@echo "🔍 Code Quality:"
	@echo "  make lint          - Run linting, formatting, and type checking"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test          - Run all tests (single Python version)"
	@echo "  make test-unit     - Run unit tests (single Python version)"
	@echo "  make test-integration - Run integration tests for all provider scenarios (latest Python version)"
	@echo "  make nox           - Run unit + integration tests (all Python versions)"
	@echo "  make nox-unit      - Run unit tests (all Python versions)"
	@echo "  make nox-integration - Run integration tests (all Python versions)"
	@echo ""
	@echo "📦 Dependency Testing:"
	@echo "  make test-deps     - Test all dependency combinations"
	@echo "  make test-openai   - Test chimeric[openai] only"
	@echo "  make test-anthropic - Test chimeric[anthropic] only"
	@echo "  make test-google   - Test chimeric[google] only"
	@echo "  make test-cerebras - Test chimeric[cerebras] only"
	@echo "  make test-cohere   - Test chimeric[cohere] only"
	@echo "  make test-grok     - Test chimeric[grok] only"
	@echo "  make test-groq     - Test chimeric[groq] only"
	@echo "  make test-bare     - Test bare installation (no extras)"
	@echo "  make test-all-extras - Test all extras installation"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  make clean         - Clean build/cache files"
	@echo "  make clean-cassettes - Clean VCR cassettes"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  make docs          - Serve docs locally"
	@echo "  make docs-deploy   - Deploy docs to GitHub Pages"
	@echo ""
	@echo "🔧 Build:"
	@echo "  make build         - Build distribution packages"