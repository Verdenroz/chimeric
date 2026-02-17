import nox
from nox import Session

# Configuration
package = "chimeric"
python_versions = ["3.11", "3.12", "3.13"]
latest_python_version = python_versions[-1]
providers = ["openai", "anthropic", "google", "cerebras", "cohere", "grok", "groq"]
nox.needs_version = ">= 2024.10.9"
nox.options.sessions = ("unit", "integration")


# Core test sessions
@nox.session(python=python_versions)
def unit(session: Session) -> None:
    """Run fast unit tests across Python versions."""
    session.run("uv", "sync", "--all-extras", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/unit", *session.posargs, external=True)


@nox.session(python=python_versions)
def integration(session: Session) -> None:
    """Run integration tests for all providers (all Python versions)."""
    session.run("uv", "sync", "--dev", external=True)

    # Test each provider in isolation
    for provider in providers:
        session.log(f"Testing {provider} provider...")
        session.run("uv", "run", "pytest", "tests/integration", "-m", provider, "--no-cov", "-v", external=True)

    # Test bare initialization
    session.log("Testing bare initialization...")
    session.run("uv", "run", "pytest", "tests/integration", "-m", "bare_install", "--no-cov", "-v", external=True)

    # Test all providers together
    session.log("Testing all providers together...")
    session.run("uv", "run", "pytest", "tests/integration", "-m", "all_extras", "--no-cov", "-v", external=True)


# Dependency combination testing sessions
@nox.session(python=latest_python_version)
def test_openai(session: Session) -> None:
    """Test OpenAI provider integration."""
    session.run("uv", "sync", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/integration", "-m", "openai", "--no-cov", external=True)


@nox.session(python=latest_python_version)
def test_anthropic(session: Session) -> None:
    """Test Anthropic provider integration."""
    session.run("uv", "sync", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/integration", "-m", "anthropic", "--no-cov", external=True)


@nox.session(python=latest_python_version)
def test_google(session: Session) -> None:
    """Test Google provider integration."""
    session.run("uv", "sync", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/integration", "-m", "google", "--no-cov", external=True)


@nox.session(python=latest_python_version)
def test_cerebras(session: Session) -> None:
    """Test Cerebras provider integration."""
    session.run("uv", "sync", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/integration", "-m", "cerebras", "--no-cov", external=True)


@nox.session(python=latest_python_version)
def test_cohere(session: Session) -> None:
    """Test Cohere provider integration."""
    session.run("uv", "sync", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/integration", "-m", "cohere", "--no-cov", external=True)


@nox.session(python=latest_python_version)
def test_grok(session: Session) -> None:
    """Test Grok provider integration."""
    session.run("uv", "sync", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/integration", "-m", "grok", "--no-cov", external=True)


@nox.session(python=latest_python_version)
def test_groq(session: Session) -> None:
    """Test Groq provider integration."""
    session.run("uv", "sync", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/integration", "-m", "groq", "--no-cov", external=True)


@nox.session(python=latest_python_version)
def test_bare(session: Session) -> None:
    """Test bare chimeric installation (no dev extras)."""
    session.run("uv", "sync", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/integration", "-m", "bare_install", "--no-cov", external=True)


@nox.session(python=latest_python_version)
def test_all_extras(session: Session) -> None:
    """Test all providers together."""
    session.run("uv", "sync", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/integration", "-m", "all_extras", "--no-cov", external=True)


@nox.session(python=latest_python_version)
def test_integration(session: Session) -> None:
    """Run integration tests for all providers (latest Python only)."""
    session.run("uv", "sync", "--dev", external=True)

    for provider in providers:
        session.log(f"Testing {provider} provider...")
        session.run("uv", "run", "pytest", "tests/integration", "-m", provider, "--no-cov", "-v", external=True)

    session.log("Testing bare initialization...")
    session.run("uv", "run", "pytest", "tests/integration", "-m", "bare_install", "--no-cov", "-v", external=True)

    session.log("Testing all providers together...")
    session.run("uv", "run", "pytest", "tests/integration", "-m", "all_extras", "--no-cov", "-v", external=True)


# Code quality sessions
@nox.session(python=latest_python_version)
def lint(session: Session) -> None:
    """Run linting and formatting checks."""
    session.run("uv", "sync", "--all-extras", "--dev", external=True)
    session.run("uv", "run", "python", "devtools/lint.py", external=True)


# Coverage sessions
@nox.session(python=latest_python_version)
def coverage(session: Session) -> None:
    """Combine coverage data and create reports."""
    session.run("uv", "sync", "--all-extras", "--dev", external=True)
    session.run("uv", "run", "coverage", "report", "--show-missing", external=True)
    
    # Generate XML report if requested
    if session.posargs and "xml" in session.posargs:
        session.run("uv", "run", "coverage", "xml", external=True)
