# Response Types and Formats

Chimeric provides a unique dual response system that gives you the best of both worlds: unified consistency across providers and access to provider-specific features when needed.

## Overview

Every response from Chimeric contains two formats:

- **Unified Format** (default): Standardized `CompletionResponse` with consistent fields across all providers
- **Native Format**: Provider's original response object with all provider-specific fields and metadata

This dual system allows you to write cross-provider code while still accessing provider-specific features when necessary.

## Response Architecture

### Internal Structure

Internally, Chimeric wraps all provider responses in container objects:

```python
# Non-streaming responses
ChimericCompletionResponse[NativeType]:
    .native   # Provider-specific response object
    .common   # Unified CompletionResponse format

# Streaming responses  
ChimericStreamChunk[NativeType]:
    .native   # Provider-specific chunk object
    .common   # Unified StreamChunk format
```

### Access Control

The `native` parameter controls which format you receive:

```python
# Default: Returns unified format
response = client.generate(model="gpt-4o", messages="Hello")
# Type: CompletionResponse

# Native: Returns provider-specific format
native_response = client.generate(model="gpt-4o", messages="Hello", native=True)  
# Type: OpenAI's ChatCompletion object
```

## Unified Format (Default)

The unified format provides consistent fields across all providers:

### CompletionResponse Structure

```python
from chimeric import Chimeric

client = Chimeric()
response = client.generate(model="gpt-4o", messages="Explain quantum physics")

# Standardized fields available across all providers
print(response.content)           # str | list[Any] - Main response content
print(response.model)             # str | None - Model that generated response  
print(response.usage.prompt_tokens)      # int - Input tokens used
print(response.usage.completion_tokens)  # int - Output tokens generated
print(response.usage.total_tokens)       # int - Total tokens
print(response.metadata)          # dict[str, Any] | None - Additional info
```

### StreamChunk Structure (Streaming)

```python
stream = client.generate(
    model="gpt-4o", 
    messages="Write a story", 
    stream=True
)

for chunk in stream:
    print(chunk.content)          # str | list[Any] - Accumulated content
    print(chunk.delta)            # str | None - New content in this chunk
    print(chunk.finish_reason)    # str | None - Reason streaming stopped
    print(chunk.metadata)         # dict[str, Any] | None - Chunk metadata
```

### Cross-Provider Consistency

The unified format ensures your code works identically across providers:

```python
def analyze_with_any_model(model_name: str, text: str) -> str:
    """Works with any provider - OpenAI, Anthropic, Google, etc."""
    response = client.generate(
        model=model_name,
        messages=f"Analyze this text: {text}"
    )
    
    # Same interface regardless of provider
    tokens_used = response.usage.total_tokens
    content = response.content
    
    return f"Analysis ({tokens_used} tokens): {content}"

# Works with any model/provider
result1 = analyze_with_any_model("gpt-4o", "Sample text")
result2 = analyze_with_any_model("claude-3-5-sonnet-20241022", "Sample text") 
result3 = analyze_with_any_model("gemini-1.5-pro", "Sample text")
```

## When to Use Each Format

### Use Unified Format When:

- **Cross-provider compatibility** is important
- Building **provider-agnostic** applications
- You only need **standard fields** (content, usage, model)
- **Consistency** across different models/providers is required
- Building **generic tools** or libraries

```python
# Perfect for cross-provider applications
def summarize_text(text: str, model: str) -> dict:
    response = client.generate(model=model, messages=f"Summarize: {text}")
    return {
        "summary": response.content,
        "tokens_used": response.usage.total_tokens,
        "model": response.model
    }
```

### Use Native Format When:

- You need **provider-specific metadata** (IDs, timestamps, safety ratings)
- Accessing **unique provider features** (stop sequences, system fingerprints)
- **Debugging** or **logging** detailed response information
- **Integration** with provider-specific tools or workflows
- **Advanced monitoring** of provider-specific metrics

```python
# Perfect for detailed logging and monitoring
def detailed_completion_log(prompt: str, model: str):
    native_response = client.generate(model=model, messages=prompt, native=True)
    
    # Log provider-specific details for debugging
    if "gpt" in model:
        log_openai_response(native_response)
    elif "claude" in model:
        log_anthropic_response(native_response)
```

## Async Support

Both formats work identically with async operations:

```python
import asyncio

async def main():
    # Unified async response
    response = await client.agenerate(model="gpt-4o", messages="Hello")
    print(response.content)
    
    # Native async response  
    native_response = await client.agenerate(
        model="gpt-4o", 
        messages="Hello", 
        native=True
    )
    print(native_response.choices[0].message.content)
    
    # Unified async streaming
    stream = await client.agenerate(model="gpt-4o", messages="Story", stream=True)
    async for chunk in stream:
        if chunk.delta:
            print(chunk.delta, end="")

asyncio.run(main())
```

## Best Practices

### Start with Unified Format

```python
# Begin with unified format for simplicity
response = client.generate(model="gpt-4o", messages="Hello")
content = response.content
tokens = response.usage.total_tokens
```

### Switch to Native When Needed

```python
# Use native format only when you need provider-specific features
if need_detailed_metadata:
    native_response = client.generate(model="gpt-4o", messages="Hello", native=True)
    response_id = native_response.id
    created_time = native_response.created
```

### Handle Multiple Providers Gracefully

```python
def smart_response_handler(model: str, prompt: str):
    # Use unified for basic info
    response = client.generate(model=model, messages=prompt)
    result = {"content": response.content, "tokens": response.usage.total_tokens}
    
    # Add provider-specific details if needed
    if need_provider_details:
        native = client.generate(model=model, messages=prompt, native=True)
        result["native_metadata"] = extract_provider_metadata(native, model)
    
    return result
```

This dual response system ensures you can build both flexible cross-provider applications and provider-specific integrations with the same codebase.