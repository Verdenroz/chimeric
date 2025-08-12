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

# Grok (supports both variable names)
export GROK_API_KEY="xai-..."
# or
export GROK_API_TOKEN="xai-..."
```

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
    grok_api_key="xai-..."
)
```

### Mixed Configuration

Combine environment variables with direct initialization:

```python
# Some keys from environment, others provided directly
client = Chimeric(
    openai_api_key="sk-...",  # Explicit key
    # anthropic_api_key will be read from ANTHROPIC_API_KEY env var
    # google_api_key will be read from GOOGLE_API_KEY env var
)
```

### Provider-Specific Configuration

Pass any provider-specific configuration options directly to Chimeric. Chimeric uses signature introspection to automatically filter kwargs, ensuring each provider only receives parameters its constructor accepts:

```python
# All kwargs are passed to every provider, but automatically filtered
client = Chimeric(
    openai_api_key="sk-...",
    anthropic_api_key="sk-ant-...",
    # Google-specific parameters (ignored by other providers)
    vertexai=True,
)
```

### Common Configuration Options

Parameters that work across multiple providers:

```python
client = Chimeric(
    openai_api_key="sk-...",
    anthropic_api_key="sk-ant-...",
    google_api_key="AIza...",
    
    # HTTP/Connection settings (widely supported)
    timeout=60,                    # Request timeout
    max_retries=3,                # Retry configuration
    
    # Custom endpoints (where supported)
    base_url="https://api.your-company.com/v1",
    
    # Headers (where supported)
    default_headers={"User-Agent": "YourApp/1.0"},
)
```

**Key Benefits:**
- **Signature-Based Filtering**: Uses Python introspection to validate parameters
- **No TypeErrors**: Invalid parameters are automatically filtered out
- **Cross-Provider Compatibility**: Same parameters work where applicable
- **Unified Configuration**: Configure all providers with one constructor call

## Provider Configuration

### Provider Information

Get information about configured providers:

```python
client = Chimeric()

# List all configured providers
print("Available providers:", client.available_providers)

# Check capabilities across all providers
print("Merged capabilities:", client.capabilities)

# Check capabilities for a specific provider
if "openai" in client.available_providers:
    openai_caps = client.get_capabilities("openai")
    print(f"OpenAI capabilities: {openai_caps}")
```

## Model Configuration

### Model Discovery

Discover available models across providers:

```python
# List all models from all providers
all_models = client.list_models()
for model in all_models:
    print(f"{model.id} ({model.provider})")

# List models from a specific provider
openai_models = client.list_models("openai")
for model in openai_models:
    print(f"OpenAI: {model.id}")
```
