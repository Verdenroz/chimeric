# Embeddings

Chimeric provides a unified embedding API across OpenAI, Google, and Cohere through `embed()` and `aembed()`. Pass a single string for one vector or a list of strings for batch embedding — the response shape adjusts automatically.

## Supported Providers

| Provider | Example models |
|---|---|
| OpenAI | `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002` |
| Google | `gemini-embedding-001` |
| Cohere | `embed-english-v3.0`, `embed-english-light-v3.0`, `embed-multilingual-v3.0`, `embed-v4.0` |

## Single Embedding

Pass a plain string to get one vector back in `result.embedding`:

```python
from chimeric import Chimeric

client = Chimeric()

result = client.embed(
    model="text-embedding-3-small",
    input="Python developer with 5 years experience",
)

print(result.embedding)             # list[float]
print(result.model)                 # "text-embedding-3-small"
print(result.usage.prompt_tokens)   # int
```

## Batch Embedding

Pass a list of strings to embed multiple texts in a single request. Results arrive in `result.embeddings` in the same order as the input:

```python
texts = [
    "Python backend developer",
    "Go backend engineer",
    "Frontend React developer",
]

result = client.embed(model="text-embedding-3-small", input=texts)

print(len(result.embeddings))  # 3
print(result.embedding)        # None for batch requests
```

## EmbeddingResponse Fields

| Field | Type | Description |
|---|---|---|
| `embedding` | `list[float] | None` | Single embedding vector; `None` for batch requests |
| `embeddings` | `list[list[float]]` | All vectors for batch requests; empty for single |
| `model` | `str | None` | Model that produced the response |
| `usage` | `EmbeddingUsage | None` | Token usage (`prompt_tokens`, `total_tokens`) |

## Custom Dimensions (OpenAI)

OpenAI's `text-embedding-3` family supports truncating embeddings to a smaller dimension, useful for reducing storage and latency:

```python
result = client.embed(
    model="text-embedding-3-small",
    input="Hello world",
    dimensions=256,
)

print(len(result.embedding))  # 256
```

## Async Usage

Use `aembed()` in async contexts — it mirrors `embed()` exactly:

```python
import asyncio
from chimeric import Chimeric


async def main():
    client = Chimeric()

    # Single
    result = await client.aembed(
        model="text-embedding-3-small",
        input="Hello from async",
    )
    print(len(result.embedding))

    # Batch
    result = await client.aembed(
        model="text-embedding-3-small",
        input=["text one", "text two"],
    )
    print(len(result.embeddings))  # 2


asyncio.run(main())
```

## Cross-Provider Switching

The same call pattern works across all supported providers — only the model name changes:

```python
openai_result = client.embed(model="text-embedding-3-small", input="machine learning engineer")
google_result = client.embed(model="gemini-embedding-001", input="machine learning engineer")
cohere_result = client.embed(model="embed-english-v3.0", input="machine learning engineer")
```

## Cosine Similarity

Embeddings are most useful when compared. Here is a minimal cosine similarity helper:

```python
import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


result = client.embed(
    model="text-embedding-3-small",
    input=[
        "Python backend developer",
        "Python backend engineer",  # similar
        "Chocolate cake recipe",    # dissimilar
    ],
)

a, b, c = result.embeddings
print(cosine_similarity(a, b))  # ~0.96 — high similarity
print(cosine_similarity(a, c))  # ~0.70 — low similarity
```
