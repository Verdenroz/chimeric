# Response Types and Formats

Chimeric normalises all provider responses into consistent types so your code works identically regardless of which provider is behind the model.

## CompletionResponse

Non-streaming calls return a `CompletionResponse`:

```python
from chimeric import Chimeric

client = Chimeric()
response = client.generate(model="gpt-4o", messages="Explain quantum physics")

print(response.content)                  # str | list[Any] — generated text
print(response.model)                    # str | None — model that responded
print(response.usage.prompt_tokens)     # int — input tokens
print(response.usage.completion_tokens) # int — output tokens
print(response.usage.total_tokens)      # int — total tokens
print(response.metadata)                # dict[str, Any] | None — provider extras
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
| `content` | `str \| list[Any]` | Accumulated text up to this chunk |
| `delta` | `str \| None` | New text added in this chunk |
| `finish_reason` | `str \| None` | Present only on the final chunk |
| `metadata` | `dict[str, Any] \| None` | Provider-specific extras |

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

## Async Support

`agenerate()` is the async counterpart with the same return types:

```python
import asyncio

async def main():
    # Non-streaming
    response = await client.agenerate(model="gpt-4o", messages="Hello")
    print(response.content)

    # Streaming
    stream = await client.agenerate(model="gpt-4o", messages="Tell a story", stream=True)
    async for chunk in stream:
        print(chunk.delta or "", end="", flush=True)

asyncio.run(main())
```
