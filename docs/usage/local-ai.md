# Local AI

Chimeric supports any **OpenAI-compatible** local inference server out of the box — no extra dependencies, no provider extras.  Just point `base_url` at the server and Chimeric handles the rest.

Compatible servers include:

- [llama-swap](https://github.com/mostlygeek/llama-swap)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (`--server` mode)
- [Ollama](https://ollama.com/) (`/v1` endpoint)
- [LM Studio](https://lmstudio.ai/) (local server)
- Any other server that speaks the OpenAI chat completions API

## Basic Setup

Pass `base_url` to `Chimeric`. The `api_key` defaults to `"local"` for servers that do not validate credentials:

```python
from chimeric import Chimeric

client = Chimeric(base_url="http://127.0.0.1:11434/v1")
```

For servers that require a key:

```python
client = Chimeric(
    base_url="http://127.0.0.1:11434/v1",
    api_key="my-server-secret",
)
```

The local endpoint is registered as a provider named `"custom"`. All standard Chimeric features — streaming, tools, structured output, async — work identically.

## Model Discovery

Chimeric queries `GET /models` at startup and caches the results, so model names are routed automatically:

```python
# List everything the local server exposes
for model in client.list_models():
    print(f"{model.id}")
```

## Generating Text

Once the client is initialised, use `generate()` exactly as you would with any cloud provider:

```python
response = client.generate(
    model="qwen2.5:3b",
    messages="Explain neural networks in one sentence.",
)
print(response.content)
```

## Streaming

Streaming works identically to cloud providers:

```python
stream = client.generate(
    model="qwen2.5:3b",
    messages="Write a short poem.",
    stream=True,
)

for chunk in stream:
    print(chunk.delta or "", end="", flush=True)
```

## Async

```python
import asyncio
from chimeric import Chimeric


async def main():
    client = Chimeric(base_url="http://127.0.0.1:11434/v1")
    response = await client.agenerate(
        model="qwen2.5:3b",
        messages="What is 2 + 2?",
    )
    print(response.content)


asyncio.run(main())
```

## Tools

Local models that support function calling work with the `@client.tool()` decorator:

```python
from chimeric import Chimeric


client = Chimeric(base_url="http://127.0.0.1:11434/v1")


@client.tool()
def get_weather(city: str) -> str:
    """Get current weather for a city.

    Args:
        city: Name of the city.

    Returns:
        A short weather description.
    """
    return f"Sunny, 22°C in {city}"


response = client.generate(
    model="qwen2.5:3b",
    messages="What is the weather in Tokyo?",
)
print(response.content)
```

!!! note "Tool calling support"
    Tool calling reliability varies by model.  Larger instruction-tuned models (≥ 7B) generally handle function calling better than smaller ones.

## Mixing Local and Cloud Providers

Local and cloud providers coexist in a single client. Chimeric routes each `generate()` call to whichever provider advertises that model:

```python
client = Chimeric(
    base_url="http://127.0.0.1:11434/v1",  # local
    openai_api_key="sk-...",               # cloud
)

# Routes to local server
local_resp = client.generate("qwen2.5:3b", "Hello from local!")

# Routes to OpenAI
cloud_resp = client.generate("gpt-4o", "Hello from the cloud!")
```
