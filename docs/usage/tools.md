# Function Calling with Tools

Chimeric provides a powerful function calling system that allows AI models to execute Python functions during conversations. This enables models to perform actions, fetch data, or interact with external systems.

!!! warning "Provider Reliability"
    Function calling reliability varies significantly by provider. **OpenAI, Anthropic, and Google** are among the most reliable for tool use. Other providers may have inconsistent function calling behavior, parameter parsing issues, or limited tool support. For production applications requiring reliable function calling, we recommend using these proven providers.

## Overview

Function calling (also known as "tool use") allows LLM models to call predefined Python functions based on the conversation context. Chimeric automatically:

- Registers Python functions as tools
- Generates parameter schemas from type hints
- Parses multiple docstring formats for descriptions
- Handles tool execution and response formatting

## Tool Registration

### Basic Registration

Use the `@tool()` decorator to register functions:

```python
from chimeric import Chimeric

client = Chimeric()

@client.tool()
def get_weather(city: str) -> str:
    """Get current weather for a city.
    
    Args:
        city: Name of the city to get weather for
        
    Returns:
        Weather description string
    """
    # Your weather API logic here
    return f"Sunny, 75°F in {city}"

# Function is now available to all models
response = client.generate(
    model="gpt-4o",
    messages="What's the weather in San Francisco?"
)
```

### Custom Names and Descriptions

Override default names and descriptions:

```python
@client.tool(name="fetch_stock_data", description="Retrieve real-time stock information")
def get_stock_price(symbol: str, include_history: bool = False) -> dict:
    """Get stock price and optional historical data."""
    return {
        "symbol": symbol,
        "price": 150.25,
        "history": [] if not include_history else [140, 145, 150]
    }
```

## Supported Docstring Formats

Chimeric supports three popular docstring formats for automatic parameter documentation:

### Google Style (Recommended)

```python
@client.tool()
def analyze_data(data: list[float], method: str = "mean") -> dict:
    """Analyze numerical data using statistical methods.
    
    Args:
        data: List of numerical values to analyze
        method: Statistical method to use ('mean', 'median', 'mode')
        
    Returns:
        Dictionary containing analysis results
    """
    # Implementation here
    pass
```

### NumPy Style

```python
@client.tool()
def process_image(image_path: str, resize: bool = True) -> str:
    """Process an image file with optional resizing.
    
    Parameters
    ----------
    image_path : str
        Path to the image file to process
    resize : bool, optional
        Whether to resize the image, default True
        
    Returns
    -------
    str
        Path to the processed image file
    """
    # Implementation here
    pass
```

### Sphinx Style

```python
@client.tool()
def send_email(recipient: str, subject: str, body: str = "") -> bool:
    """Send an email message.
    
    :param recipient: Email address of the recipient
    :param subject: Subject line of the email
    :param body: Email body content, optional
    :returns: True if email was sent successfully
    """
    # Implementation here
    pass
```

## Type System Support

Chimeric automatically converts Python type hints to JSON schemas for models:

### Basic Types

```python
@client.tool()
def basic_types_example(
    text: str,           # → "string" 
    number: int,         # → "integer"
    decimal: float,      # → "number"
    flag: bool          # → "boolean"
) -> str:
    """Example of basic type support."""
    return f"{text}: {number}, {decimal}, {flag}"
```

### Complex Types

```python
@client.tool()
def complex_types_example(
    items: list[str],              # → array of strings
    metadata: dict[str, int],      # → object with integer values
    optional_param: str | None = None,  # → optional string
    tags: list[str] = None        # → optional array
) -> dict:
    """Example of complex type support."""
    return {
        "items": items or [],
        "metadata": metadata or {},
        "tags": tags or []
    }
```

### Union and Optional Types

```python
from typing import Union

@client.tool()
def flexible_input(
    value: Union[str, int],        # → accepts string or integer
    optional_flag: bool | None = None,  # → optional boolean
    default_list: list[str] = None     # → optional string array
) -> str:
    """Handle flexible input types."""
    return f"Processed: {value}"
```

## Tool Management

### Accessing Registered Tools

```python
# Get all registered tools
all_tools = client.tools
print(f"Registered {len(all_tools)} tools:")
for tool in all_tools:
    print(f"- {tool.name}: {tool.description}")
```

### Manual Tool Control

Control which tools are available for specific requests:

```python
# Register multiple tools
@client.tool()
def get_weather(city: str) -> str:
    return f"Weather in {city}"

@client.tool()
def get_stock_price(symbol: str) -> float:
    return 150.25

# Use specific tools only
response = client.generate(
    model="gpt-4o",
    messages="What's the weather in NYC? What is the stock price of AAPL?",
    tools=[client.tools[0]],  # Only weather tool
    auto_tool=False  # Don't auto-include other tools
)

print(response)

# Use all tools (default behavior)
response = client.generate(
    model="gpt-4o",
    messages="What's the weather in NYC? What is the stock price of AAPL?",
    auto_tool=True  # Automatically includes all registered tools
)

print(response)
```

## The auto_tool Parameter

The `auto_tool` parameter controls automatic tool inclusion:

```python
# auto_tool=True (default): Use all registered tools if none specified
response = client.generate(
    model="gpt-4o",
    messages="What can you help me with?",
    auto_tool=True  # All registered tools available
)

# auto_tool=False: Only use explicitly provided tools
response = client.generate(
    model="gpt-4o", 
    messages="Get weather for Boston",
    tools=[weather_tool],  # Only this tool
    auto_tool=False       # Don't add other registered tools
)

# Explicit tools override auto_tool behavior
response = client.generate(
    model="gpt-4o",
    messages="Send an email",
    tools=[email_tool],   # Only email tool, regardless of auto_tool
    auto_tool=True       # Ignored when tools are explicitly provided
)
```

## Best Practices

### Function Design

1. **Clear Descriptions**: Write descriptive docstrings explaining what the function does
2. **Type Hints**: Always use type hints for automatic schema generation
3. **Parameter Documentation**: Document each parameter's purpose and format
4. **Error Handling**: Handle errors gracefully and return meaningful messages

```python
@client.tool()
def fetch_user_profile(user_id: str, include_preferences: bool = False) -> dict:
    """Fetch user profile information from the database.
    
    Args:
        user_id: Unique identifier for the user (UUID format)
        include_preferences: Whether to include user preference settings
        
    Returns:
        Dictionary containing user profile data, or error message if user not found
    """
    try:
        # Validate user_id format
        if not user_id or len(user_id) < 8:
            return {"error": "Invalid user_id format"}
            
        # Fetch user data
        profile = get_user_from_db(user_id)
        if not profile:
            return {"error": f"User {user_id} not found"}
            
        # Add preferences if requested
        if include_preferences:
            profile["preferences"] = get_user_preferences(user_id)
            
        return profile
        
    except Exception as e:
        return {"error": f"Failed to fetch user profile: {str(e)}"}
```

### Provider Compatibility

Not all providers support function calling. Check provider capabilities:

```python
# Check if provider supports tools
if "openai" in client.available_providers:
    caps = client.get_capabilities("openai") 
    if caps.tools:
        print("OpenAI supports function calling")
        
# Use tools only with compatible providers
response = client.generate(
    model="gpt-4o",  # OpenAI supports tools
    messages="Process this data for me",
    # tools will be used automatically
)
```

### Async Function Support

Function calling works with both sync and async generation:

```python
import asyncio

async def main():
    # Async generation with tools
    response = await client.agenerate(
        model="gpt-4o",
        messages="What's the current time?",
        auto_tool=True
    )
    print(response.content)

asyncio.run(main())
```

## Streaming with Tools

Function calling works seamlessly with streaming responses, allowing you to see both AI-generated text and tool execution results in real-time:

```python
from chimeric import Chimeric
from datetime import datetime

client = Chimeric()

@client.tool()
def get_current_time() -> str:
    """Get the current time."""
    now = datetime.now()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}"

@client.tool()
def tell_joke() -> str:
    """Tell a programming joke."""
    return "Why do programmers prefer dark mode? Because light attracts bugs! 🐛"

# Stream with tools enabled
stream = client.generate(
    model="gpt-4o",
    messages="What time is it and then tell me a programming joke?",
    stream=True
)

print("Streaming response with tools:\n")

for chunk in stream:
    if chunk.delta:
        print(chunk.delta, end="", flush=True)
    
    if chunk.finish_reason:
        print(f"\n\nStream finished: {chunk.finish_reason}")
```

This enables real-time interaction where users can see the AI's response being generated while tools are being called and executed in the background.

## Complete Example

```python
from chimeric import Chimeric
from datetime import datetime

client = Chimeric()

@client.tool()
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current time in specified timezone.
    
    Args:
        timezone: Timezone name (e.g., 'UTC', 'EST', 'PST')
        
    Returns:
        Current time as formatted string
    """
    # Simple implementation for demo
    now = datetime.now()
    return f"Current time ({timezone}): {now.strftime('%Y-%m-%d %H:%M:%S')}"

@client.tool()
def calculate_tip(bill_amount: float, tip_percentage: float = 18.0) -> dict:
    """Calculate tip amount and total bill.
    
    Args:
        bill_amount: Total bill amount before tip
        tip_percentage: Tip percentage (default 18%)
        
    Returns:
        Dictionary with tip amount and total
    """
    tip_amount = bill_amount * (tip_percentage / 100)
    total = bill_amount + tip_amount
    
    return {
        "bill_amount": bill_amount,
        "tip_percentage": tip_percentage,
        "tip_amount": round(tip_amount, 2),
        "total": round(total, 2)
    }

# Use the tools
response = client.generate(
    model="gpt-4o",
    messages="What time is it and help me calculate a 20% tip on a $85 bill"
)

print(response.content)
# Model will call both functions and provide a complete response
```

This comprehensive function calling system makes it easy to extend AI models with custom capabilities while maintaining type safety and clear documentation.