<div align="center">

<img src=".github/assets/chimeric.png" alt="Chimeric Logo" width="200"/>

# Chimeric

[![PyPI version](https://img.shields.io/pypi/v/chimeric.svg)](https://pypi.org/project/chimeric/)
[![Python Versions](https://img.shields.io/pypi/pyversions/chimeric.svg)](https://pypi.org/project/chimeric/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg)](https://verdenroz.github.io/chimeric/)
[![CI](https://github.com/Verdenroz/chimeric/workflows/CI/badge.svg)](https://github.com/Verdenroz/chimeric/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Verdenroz/chimeric/branch/main/graph/badge.svg)](https://codecov.io/gh/Verdenroz/chimeric)

**Unified Python interface for multiple LLM providers with automatic provider detection and seamless switching.**

</div>

## 🚀 Supported Providers

[![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white)](https://openai.com/)
[![Anthropic](https://img.shields.io/badge/Anthropic-191919?logo=anthropic&logoColor=white)](https://anthropic.com/)
[![Google AI](https://img.shields.io/badge/Google%20AI-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![xAI Grok](https://img.shields.io/badge/xAI%20Grok-000000?logo=x&logoColor=white)](https://x.ai/)
[![Groq](https://img.shields.io/badge/Groq-F55036?logo=groq&logoColor=white)](https://groq.com/)
[![Cohere](https://img.shields.io/badge/Cohere-39594A?logo=cohere&logoColor=white)](https://cohere.ai/)
[![Cerebras](https://img.shields.io/badge/Cerebras-FF6B35?logo=cerebras&logoColor=white)](https://cerebras.ai/)

## 📖 Documentation

For detailed usage examples, configuration options, and advanced features, visit our [documentation](https://verdenroz.github.io/chimeric/).

## 📦 Installation

```bash
# Base installation
pip install chimeric

# With specific providers
pip install "chimeric[openai,anthropic,google]"

# All providers
pip install "chimeric[all]"
```

Set your API keys as environment variables:
```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

## ⚡ Quickstart

### Basic Usage
```python
from chimeric import Chimeric

client = Chimeric()  # Auto-detects API keys from environment

response = client.generate(
    model="gpt-4o",
    messages="Hello!"
)
print(response.content)
```

### Streaming Responses
```python
# Real-time streaming
stream = client.generate(
    model="claude-3-5-sonnet-latest",
    messages="Tell me a story about space exploration",
    stream=True
)

for chunk in stream:
    print(chunk.content, end="", flush=True)
```

### Function Calling with Tools
```python
@client.tool()
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 72°F in {city}"

@client.tool()
def calculate_tip(bill_amount: float, tip_percentage: float = 18.0) -> dict:
    """Calculate tip and total amount for a restaurant bill."""
    tip = bill_amount * (tip_percentage / 100)
    total = bill_amount + tip
    return {"tip": tip, "total": total, "tip_percentage": tip_percentage}

response = client.generate(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "What's the weather in NYC?"},
        {"role": "user", "content": "Also calculate a tip for a $50 dinner bill"}
    ]
)
print(response.content)
```

### Multi-Provider Switching
```python
# Seamlessly switch between providers
models = ["gpt-4o-mini", "claude-3-5-haiku-latest", "gemini-2.5-flash"]

for model in models:
    response = client.generate(
        model=model,
        messages="Explain quantum computing in one sentence"
    )
    print(f"{model}: {response.content}")
```


## 🔧 Key Features

- **Multi-Provider Support**: Switch between 7 major AI providers seamlessly
- **Automatic Detection**: Auto-detects available API keys from environment
- **Unified Interface**: Consistent API across all providers
- **Streaming Support**: Real-time response streaming
- **Function Calling**: Tool integration with decorators
- **Async Support**: Full async/await compatibility
- **Native Fallback**: Access provider-specific features when needed

## 🐛 Issues & Feature Requests

- **Found a bug?** Use our [Bug Report](.github/ISSUE_TEMPLATE/bug_report.yml) template
- **Want a feature?** Use our [Feature Request](.github/ISSUE_TEMPLATE/feature_request.yml) template

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
