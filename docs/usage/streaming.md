# Streaming Responses

Chimeric provides comprehensive streaming support that allows you to receive AI model responses in real-time as they are generated, rather than waiting for the complete response. This is particularly useful for interactive applications, chatbots, and scenarios where you want to display responses progressively.

## Overview

Streaming enables token-by-token delivery of responses, providing immediate feedback to users and creating more responsive applications. Chimeric's streaming system:

- **Unified Interface**: Same streaming API across all providers
- **Dual Format Support**: Access both unified and native streaming formats
- **Advanced Features**: Tool call streaming and multi-turn conversations
- **State Management**: Automatic content accumulation and metadata handling

## Basic Streaming

### Simple Text Streaming

Enable streaming by setting `stream=True`:

```python
from chimeric import Chimeric

client = Chimeric()

# Basic streaming
stream = client.generate(
    model="gpt-4o",
    messages="Tell me a story about space exploration",
    stream=True
)

# Process chunks in real-time
for chunk in stream:
    if chunk.delta:  # New content in this chunk
        print(chunk.delta, end="", flush=True)
    
    if chunk.finish_reason:
        print(f"\nStreaming finished: {chunk.finish_reason}")
```

### Understanding Stream Chunks

Each stream chunk contains several fields:

```python
stream = client.generate(
    model="gpt-4o",
    messages="Explain quantum physics briefly",
    stream=True
)

for chunk in stream:
    print(f"Content: {chunk.content}")      # Accumulated text so far
    print(f"Delta: {chunk.delta}")          # New text in this chunk
    print(f"Finish: {chunk.finish_reason}") # Why streaming stopped (if finished)
    print(f"Meta: {chunk.metadata}")        # Additional chunk info
    print("---")
```

## Stream Chunk Fields

### content
The accumulated text content from the start of the response up to the current chunk:

```python
accumulated_text = ""
for chunk in stream:
    # chunk.content contains all text so far
    accumulated_text = chunk.content
    print(f"Total so far: {accumulated_text}")
```

### delta  
The incremental text added in this specific chunk:

```python
full_response = ""
for chunk in stream:
    if chunk.delta:
        full_response += chunk.delta  # Build response incrementally
        print(chunk.delta, end="")    # Display new text immediately
```

### finish_reason
Indicates why the stream ended (typically in the final chunk):

```python
for chunk in stream:
    if chunk.finish_reason:
        print(f"Stream ended: {chunk.finish_reason}")
        # Common values: "stop", "length", "tool_calls", "content_filter"
```

### metadata
Contains additional information about the chunk or final response:

```python
for chunk in stream:
    if chunk.metadata:
        print(f"Chunk metadata: {chunk.metadata}")
        # May include: token counts, model info, request IDs, etc.
```

## Async Streaming

Use async streaming for high-performance applications:

```python
import asyncio

async def stream_example():
    client = Chimeric()
    
    stream = await client.agenerate(
        model="gpt-4o",
        messages="Write a poem about artificial intelligence",
        stream=True
    )
    
    async for chunk in stream:
        if chunk.delta:
            print(chunk.delta, end="", flush=True)
        
        if chunk.finish_reason:
            print(f"\nFinished: {chunk.finish_reason}")

asyncio.run(stream_example())
```
