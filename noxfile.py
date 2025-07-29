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
    """Run integration tests for all provider dependency scenarios (all Python versions)."""
    
    # Test each provider in isolation - initialization and provider behaviors
    for provider in providers:
        session.log(f"Testing {provider} provider (initialization + integration)...")
        session.run("uv", "sync", "--extra", provider, "--dev", external=True)
        session.run("uv", "run", "pytest", "tests/integration", "-m", provider, "--no-cov", "-v", external=True)
    
    # Test bare initialization (no dependencies)
    session.log("Testing bare initialization...")
    session.run("uv", "sync", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/integration", "-m", "bare_install", "--no-cov", "-v", external=True)
    
    # Test all providers together
    session.log("Testing all providers together...")
    session.run("uv", "sync", "--all-extras", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/integration", "-m", "all_extras", "--no-cov", "-v", external=True)


# Dependency combination testing sessions
@nox.session(python=latest_python_version)
def test_openai(session: Session) -> None:
    """Test chimeric[openai] installation and functionality."""
    session.run("uv", "sync", "--extra", "openai", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/integration", "-m", "openai", "--no-cov", external=True)


@nox.session(python=latest_python_version)
def test_anthropic(session: Session) -> None:
    """Test chimeric[anthropic] installation and functionality."""
    session.run("uv", "sync", "--extra", "anthropic", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/integration", "-m", "anthropic", "--no-cov", external=True)


@nox.session(python=latest_python_version)
def test_google(session: Session) -> None:
    """Test chimeric[google] installation and functionality."""
    session.run("uv", "sync", "--extra", "google", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/integration", "-m", "google", "--no-cov", external=True)


@nox.session(python=latest_python_version)
def test_cerebras(session: Session) -> None:
    """Test chimeric[cerebras] installation and functionality."""
    session.run("uv", "sync", "--extra", "cerebras", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/integration", "-m", "cerebras", "--no-cov", external=True)


@nox.session(python=latest_python_version)
def test_cohere(session: Session) -> None:
    """Test chimeric[cohere] installation and functionality."""
    session.run("uv", "sync", "--extra", "cohere", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/integration", "-m", "cohere", "--no-cov", external=True)


@nox.session(python=latest_python_version)
def test_grok(session: Session) -> None:
    """Test chimeric[grok] installation and functionality."""
    session.run("uv", "sync", "--extra", "grok", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/integration", "-m", "grok", "--no-cov", external=True)


@nox.session(python=latest_python_version)
def test_groq(session: Session) -> None:
    """Test chimeric[groq] installation and functionality."""
    session.run("uv", "sync", "--extra", "groq", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/integration", "-m", "groq", "--no-cov", external=True)


@nox.session(python=latest_python_version)
def test_bare(session: Session) -> None:
    """Test bare chimeric installation (no optional dependencies)."""
    session.run("uv", "sync", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/integration", "-m", "bare_install", "--no-cov", external=True)


@nox.session(python=latest_python_version)
def test_all_extras(session: Session) -> None:
    """Test chimeric installation with all optional dependencies."""
    session.run("uv", "sync", "--all-extras", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/integration", "-m", "all_extras", "--no-cov", external=True)


@nox.session(python=latest_python_version)
def test_integration(session: Session) -> None:
    """Run integration tests for all provider dependency scenarios (latest Python only).
    
    Tests both initialization and provider-specific behaviors in isolated
    environments for each provider. Quick version for development.
    """
    
    # Test each provider in isolation - initialization + provider behaviors
    for provider in providers:
        session.log(f"Testing {provider} provider (initialization + integration)...")
        session.run("uv", "sync", "--extra", provider, "--dev", external=True)
        session.run("uv", "run", "pytest", "tests/integration", "-m", provider, "--no-cov", "-v", external=True)
    
    # Test bare initialization (no dependencies)
    session.log("Testing bare initialization...")
    session.run("uv", "sync", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/integration", "-m", "bare_install", "--no-cov", "-v", external=True)
    
    # Test all providers together
    session.log("Testing all providers together...")
    session.run("uv", "sync", "--all-extras", "--dev", external=True)
    session.run("uv", "run", "pytest", "tests/integration", "-m", "all_extras", "--no-cov", "-v", external=True)
