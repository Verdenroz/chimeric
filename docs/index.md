# Chimeric

[![PyPI version](https://img.shields.io/pypi/v/chimeric.svg)](https://pypi.org/project/chimeric/)
[![Python Versions](https://img.shields.io/pypi/pyversions/chimeric.svg)](https://pypi.org/project/chimeric/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg)](https://verdenroz.github.io/chimeric/)
[![CI](https://github.com/Verdenroz/chimeric/workflows/CI/badge.svg)](https://github.com/Verdenroz/chimeric/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Verdenroz/chimeric/branch/main/graph/badge.svg)](https://codecov.io/gh/Verdenroz/chimeric)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![basedpyright](https://img.shields.io/badge/basedpyright-checked-42b883)](https://github.com/DetachHead/basedpyright)
[![codespell](https://img.shields.io/badge/codespell-checked-42b883)](https://github.com/codespell-project/codespell)

**Chimeric** is a unified Python interface for multiple LLM providers with automatic provider detection and seamless
switching.

## Setup

Chimeric provides a unified interface for **7 major AI providers**:

[![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white)](https://openai.com/)
[![Anthropic](https://img.shields.io/badge/Anthropic-191919?logo=anthropic&logoColor=white)](https://anthropic.com/)
[![Google AI](https://img.shields.io/badge/Google%20AI-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![xAI Grok](https://img.shields.io/badge/xAI%20Grok-000000?logo=x&logoColor=white)](https://x.ai/)
[![Groq](https://img.shields.io/badge/Groq-F55036?logo=groq&logoColor=white)](https://groq.com/)
[![Cohere](https://img.shields.io/badge/Cohere-39594A?logo=cohere&logoColor=white)](https://cohere.ai/)
[![Cerebras](https://img.shields.io/badge/Cerebras-FF6B35?logo=cerebras&logoColor=white)](https://cerebras.ai/)

Each provider can be installed individually or together using extras:

```bash
# Individual providers
pip install "chimeric[openai]"
pip install "chimeric[anthropic]"
pip install "chimeric[google]"
pip install "chimeric[cohere]"
pip install "chimeric[groq]"
pip install "chimeric[cerebras]"
pip install "chimeric[grok]"

# Multiple providers
pip install "chimeric[openai,anthropic,google]"

# All providers
pip install "chimeric[all]"
```

## Quickstart

```python
from chimeric import Chimeric

client = Chimeric()  # Auto-detects API keys from environment

response = client.generate(
    model="gpt-4o",
    messages="Hello!"
)
print(response.content)
```

## Use Cases

**Multi-Provider Switching:**

```python
# Seamlessly switch between providers - string input
gpt_response = client.generate(model="gpt-4o", messages="Explain quantum computing")
claude_response = client.generate(model="claude-3-5-haiku-latest", messages="Write a poem about AI")
gemini_response = client.generate(model="gemini-2.5-flash", messages="Summarize climate change")
```

**Flexibility:**

```python
# Mixed usage in same application
unified = client.generate(model="claude-3-5-haiku-latest", messages="Code review this function")
native = client.generate(model="claude-3-5-haiku-latest", messages="Debug this error", native=True)

# Use unified for consistent cross-provider code
print(unified.content)

# Use native for provider-specific features
if hasattr(native, 'stop_reason'):
    print(f"Claude stop reason: {native.stop_reason}")
```

**Streaming:**

```python
stream = client.generate(
    model="gpt-4o",
    messages="Tell me a story about space exploration",
    stream=True
)

for chunk in stream:
    print(chunk.content, end="", flush=True)
```

**Function Calling:**

```python
@client.tool()
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 72°F in {city}"


response = client.generate(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in NYC?"}]
)
```

**Async Support:**

```python
import asyncio


async def main():
    response = await client.agenerate(
        model="claude-3-5-sonnet-latest",
        messages=[{"role": "user", "content": "Analyze this data"},
                  {"role": "assistant",
                   "content": "I'd be happy to help analyze data. What data would you like me to look at?"},
                  {"role": "user", "content": "Sales figures from Q4"}]
    )
    print(response.content)


asyncio.run(main())
```

## Known Limitations

- **Beta Status**: API may change as we refine the interface
- **Provider Dependencies**: Each provider requires separate installation extra
- **Rate Limits**: Subject to individual provider rate limits and quotas
- **Multimodal Support**: Image and audio support is untested and may vary by provider
- **Model Availability**: Some models may not be available in all regions

## Roadmap

- **Embeddings Support**: Unified interface for text embeddings across providers
- **Mutli-Modal Support**: Enhanced support for images and audio
- **Cost Tracking**: Built-in usage and cost monitoring
- **Advanced Routing**: Load balancing and failover between providers

---

**License**: Chimeric is licensed under the [MIT License](https://github.com/Verdenroz/chimeric/blob/main/LICENSE).
