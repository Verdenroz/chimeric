# Getting Started

Chimeric provides a unified interface for accessing multiple LLM providers through a single, consistent API. This guide demonstrates fundamental concepts and usage patterns to get you productive quickly.

## Prerequisites

- Python 3.11 or higher
- API keys for your chosen providers
- Basic familiarity with async/await patterns (for async usage)

## Core Concepts

### Automatic Provider Detection

Chimeric intelligently routes requests based on model names, eliminating the need for manual provider selection:

```python
from chimeric import Chimeric

client = Chimeric()  # Auto-detects available providers from environment

# Each model automatically routes to its respective provider
gpt_response = client.generate(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Analyze market trends"}]
)

# Messages can also be provided as a string
claude_response = client.generate(
    model="claude-3-5-haiku-latest", 
    messages="Hello, world!"
)

gemini_response = client.generate(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "Summarize research findings"}]
)
```

### Unified Response Format

All providers return responses in a consistent format:

```python
response = client.generate(model="gpt-4o", messages=[...])

# Common interface
print(response.content)              # Text content
print(response.model)                # Model used
print(response.usage.prompt_tokens)  # Input token usage
print(response.usage.completion_tokens)  # Output token usage
print(response.usage.total_tokens)   # Total tokens

# Access native provider response with native=True parameter
native_response = client.generate(
    model="claude-3-5-haiku-latest", 
    messages=[...], 
    native=True
)
print(native_response.stop_reason)  # Anthropic-specific stop reason
```

### Streaming Responses

Stream responses token-by-token for real-time applications:

```python
stream = client.generate(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

### Function Registration and Execution

Chimeric provides a decorator-based system for registering functions that models can invoke:

```python
client = Chimeric()

@client.tool()
def analyze_financial_data(
    symbol: str, 
    period: str = "1y", 
    metrics: list[str] = None
) -> dict:
    """Analyze financial performance metrics for a given symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
        period: Analysis period ('1y', '6m', '3m')
        metrics: Specific metrics to analyze
    
    Returns:
        Dict containing analysis results and recommendations
    """
    return {
        "symbol": symbol,
        "period": period,
        "performance": "positive",
        "volatility": "moderate",
        "recommendation": "hold"
    }

# Functions are automatically available to all compatible models
response = client.generate(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Analyze Tesla's performance over the last year"}]
)
```

### Async Support

Chimeric supports async operations for high-performance applications:

```python
import asyncio
from chimeric import Chimeric

async def main():
    client = Chimeric()
    
    response = await client.agenerate(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    
    print(response.content)

# Async streaming
async def stream_example():
    client = Chimeric()
    
    stream = await client.agenerate(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Tell me a joke"}],
        stream=True
    )
    
    async for chunk in stream:
        if chunk.content:
            print(chunk.content, end="", flush=True)

asyncio.run(main())
```

## Error Handling

Handle errors gracefully across providers:

```python
from chimeric import Chimeric
from chimeric.exceptions import ChimericError

client = Chimeric()

try:
    response = client.generate(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )
except ChimericError as e:
    print(f"Chimeric error: {e}")
```
