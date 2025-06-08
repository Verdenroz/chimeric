# Chimeric

[![PyPI version](https://img.shields.io/pypi/v/chimeric.svg)](https://pypi.org/project/chimeric/)
[![Python Versions](https://img.shields.io/pypi/pyversions/chimeric.svg)](https://pypi.org/project/chimeric/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg)](https://verdenroz.github.io/chimeric/)

**Chimeric**: A unified interface for multiple LLM providers with automatic provider detection and seamless switching.

## Overview

Chimeric simplifies working with multiple Large Language Model (LLM) providers by offering a consistent interface across different AI services. Whether you're using OpenAI, Anthropic, Google Gemini, Cohere, AWS, Groq, Replicate, or HuggingFace, Chimeric provides a unified experience.

Key features:

- **Provider-agnostic interface**: Write code once, use with any supported LLM provider
- **Automatic provider detection**: Chimeric can identify the appropriate provider from model names or input formats
- **Seamless switching**: Effortlessly switch between providers without rewriting your application logic
- **Modular design**: Install only the dependencies you need for the providers you use
- **Type-safe API**: Comprehensive typing support for better IDE integration and fewer runtime errors
- **Extensive provider support**: Works with major commercial and open source LLM platforms
- **Environment variable support**: Automatically detects API keys from environment variables

## Installation

```bash
pip install chimeric
```

To include specific provider support:

```bash
# Install with OpenAI support
pip install "chimeric[openai]"

# Install with multiple providers
pip install "chimeric[openai,anthropic,google]"

# Install all supported providers
pip install "chimeric[all]"
```

For more details, see the [Installation Guide](installation.md).

## Quick Start

```python
from chimeric import Chimeric

# Initialize Chimeric - it will automatically detect API keys from environment variables
client = Chimeric()  # API keys can be loaded from environment variables like OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.

# Or explicitly provide API keys
client = Chimeric(
    openai_api_key="YOUR_OPENAI_API_KEY", 
    anthropic_api_key="YOUR_ANTHROPIC_API_KEY"
)

# Chimeric automatically selects the appropriate provider based on the model
response = client.generate(
    model="gpt-4o",  # OpenAI model
    messages=[
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ]
)
print(response.content)

# Seamlessly switch to a different provider
response = client.generate(
    model="claude-3-opus-20240229",  # Anthropic model
    messages=[
        {"role": "user", "content": "Write a short poem about AI"}
    ]
)
print(response.content)
```

## Project Status

Chimeric is currently in beta. While most features are stable, the API may undergo minor changes as we refine the interface based on user feedback.

## License

Chimeric is licensed under the MIT License. See [LICENSE](https://github.com/Verdenroz/chimeric/blob/main/LICENSE) for details.
