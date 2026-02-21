# Response Types and Formats

Chimeric normalises all provider responses into consistent types so your code works identically regardless of which provider is behind the model.

## CompletionResponse

Non-streaming calls return a `CompletionResponse`:

```python
from chimeric import Chimeric

client = Chimeric()
response = client.generate(model="gpt-4o", messages="Explain quantum physics")

print(response.content)   # str | list — generated text
print(response.model)     # str | None — model that responded
print(response.metadata)  # dict | None — provider-specific extras

if response.usage:
    print(response.usage.prompt_tokens)      # int — input tokens
    print(response.usage.completion_tokens)  # int — output tokens
    print(response.usage.total_tokens)       # int — total tokens
```

`str(response)` returns the content text directly, so you can use the response object wherever a string is expected.

## StreamChunk

Streaming calls yield `StreamChunk` objects:

```python
stream = client.generate(model="gpt-4o", messages="Write a story", stream=True)

for chunk in stream:
    print(chunk.delta or "", end="", flush=True)  # incremental text
```

| Field | Type | Description |
|---|---|---|
| `content` | `str | list[Any]` | Accumulated text up to this chunk |
| `delta` | `str | None` | New text added in this chunk |
| `finish_reason` | `str | None` | Present only on the final chunk |
| `metadata` | `dict[str, Any] | None` | Provider-specific extras |

`str(chunk)` returns `delta` (or `""` for metadata-only chunks).

## Cross-Provider Consistency

The unified types work the same across all providers:

```python
def summarize(model: str, text: str) -> str:
    response = client.generate(model=model, messages=f"Summarize: {text}")
    print(f"Used {response.usage.total_tokens} tokens")
    return response.content

# All three use the same interface
summarize("gpt-4o", "...")
summarize("claude-3-5-sonnet-20241022", "...")
summarize("gemini-1.5-pro", "...")
```

Provider-specific details that don't map to the standard fields are available in `response.metadata`.

## EmbeddingResponse

`embed()` and `aembed()` return an `EmbeddingResponse`:

```python
result = client.embed(model="text-embedding-3-small", input="Hello")

print(result.embedding)           # list[float] — single vector
print(result.model)               # str | None — model used
print(result.usage.prompt_tokens) # int
print(result.usage.total_tokens)  # int
```

For batch input the vectors arrive in `embeddings` and `embedding` is `None`:

```python
result = client.embed(
    model="text-embedding-3-small",
    input=["first text", "second text"],
)

print(result.embedding)   # None
print(result.embeddings)  # list[list[float]] — one vector per input
```

| Field | Type | Description |
|---|---|---|
| `embedding` | `list[float] \| None` | Single vector; `None` for batch |
| `embeddings` | `list[list[float]]` | All vectors for batch; empty for single |
| `model` | `str \| None` | Model that produced the response |
| `usage` | `EmbeddingUsage \| None` | `prompt_tokens` and `total_tokens` |

## Async Support

`agenerate()` and `aembed()` are the async counterparts with the same return types:

```python
import asyncio


async def main():
    # Non-streaming completion
    response = await client.agenerate(model="gpt-4o", messages="Hello")
    print(response.content)

    # Streaming completion
    stream = await client.agenerate(model="gpt-4o", messages="Tell a story", stream=True)
    async for chunk in stream:
        print(chunk.delta or "", end="", flush=True)

    # Embedding
    result = await client.aembed(model="text-embedding-3-small", input="Hello")
    print(len(result.embedding))


asyncio.run(main())
```
