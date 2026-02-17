# Configuration

This guide covers the various ways to configure Chimeric for your specific needs.

## API Key Configuration

### Environment Variables (Recommended)

The simplest way to configure API keys is through environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google (supports both variable names)
export GOOGLE_API_KEY="AIza..."
# or
export GEMINI_API_KEY="AIza..."

# Cohere (supports both variable names)
export COHERE_API_KEY="your-key"
# or
export CO_API_KEY="your-key"

# Groq
export GROQ_API_KEY="gsk_..."

# Cerebras
export CEREBRAS_API_KEY="csk-..."

# Grok / xAI (supports both variable names)
export GROK_API_KEY="xai-..."
# or
export XAI_API_KEY="xai-..."

# OpenRouter
export OPENROUTER_API_KEY="sk-or-..."
```

Any provider with a discoverable key is registered automatically; providers without one are silently skipped.

### Direct Initialization

You can also provide API keys directly when creating the client:

```python
from chimeric import Chimeric

client = Chimeric(
    openai_api_key="sk-...",
    anthropic_api_key="sk-ant-...",
    google_api_key="AIza...",
    cohere_api_key="your-key",
    groq_api_key="gsk_...",
    cerebras_api_key="csk-...",
    grok_api_key="xai-...",
    openrouter_api_key="sk-or-...",
)
```

Explicit keys take precedence over environment variables.

### Mixed Configuration

Combine environment variables with direct initialization:

```python
# Explicit key overrides env var; other providers read from env vars automatically
client = Chimeric(
    openai_api_key="sk-...",
    # ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc. are picked up from the environment
)
```

### Timeout

The `timeout` parameter sets the HTTP request timeout (in seconds) for all providers:

```python
client = Chimeric(timeout=120.0)  # default is 60.0
```

## Provider Routing

### Automatic Model Routing

Chimeric routes each request to the correct provider by matching the model name against each provider's model list. This happens at initialisation time and is cached:

```python
client = Chimeric()

# Each call is automatically routed to the right provider
client.generate("gpt-4o", "Hello")                      # → OpenAI
client.generate("claude-3-5-sonnet-20241022", "Hello")  # → Anthropic
client.generate("gemini-1.5-pro", "Hello")              # → Google
```

### Explicit Provider Selection

Force a specific provider with the `provider` parameter, using either a string or the `Provider` enum:

```python
from chimeric import Chimeric, Provider

client = Chimeric()

# String form
response = client.generate(
    model="llama-3.3-70b-versatile",
    messages="Hello",
    provider="groq",
)

# Enum form (IDE-friendly, no typos)
response = client.generate(
    model="llama-3.3-70b-versatile",
    messages="Hello",
    provider=Provider.GROQ,
)
```

### Provider Inspection

```python
# List all configured providers
print(client.available_providers)  # ["openai", "anthropic", "google", ...]
```

## Model Discovery

List available models from one or all configured providers:

```python
# All providers
all_models = client.list_models()
for model in all_models:
    print(f"{model.id} ({model.provider})")

# One provider
openai_models = client.list_models("openai")
for model in openai_models:
    print(model.id)
```

The async counterpart is `alist_models()`.

## Provider Pass-Through

Extra keyword arguments passed to `generate()` / `agenerate()` are forwarded directly to the provider's API:

```python
response = client.generate(
    model="gpt-4o",
    messages="Write a haiku",
    temperature=0.9,
    max_tokens=100,
)
```

Supported parameters vary by provider — consult each provider's API documentation for the full list.
