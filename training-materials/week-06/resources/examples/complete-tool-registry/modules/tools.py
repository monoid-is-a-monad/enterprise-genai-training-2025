"""Example tool implementations."""

from pydantic import BaseModel, Field
from typing import Literal
from .registry import ToolRegistry


# Tool Parameter Schemas

class CalculatorParams(BaseModel):
    """Parameters for calculator tool."""
    operation: Literal["add", "subtract", "multiply", "divide"] = Field(
        ..., description="Mathematical operation to perform"
    )
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")


class WeatherParams(BaseModel):
    """Parameters for weather tool."""
    location: str = Field(..., description="City name or location")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature units"
    )


class SearchParams(BaseModel):
    """Parameters for search tool."""
    query: str = Field(..., description="Search query")
    max_results: int = Field(default=5, ge=1, le=20, description="Max results")


# Tool Implementations

def calculator_tool(operation: str, a: float, b: float) -> dict:
    """Perform mathematical calculations."""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else None
    }
    
    result = operations[operation](a, b)
    
    if result is None:
        return {
            "error": "Division by zero",
            "operation": operation
        }
    
    return {
        "result": result,
        "operation": operation,
        "operands": [a, b]
    }


def get_weather_tool(location: str, units: str = "celsius") -> dict:
    """Get weather information (mock implementation)."""
    # Mock weather data
    mock_temps = {
        "celsius": {"San Francisco": 18, "New York": 22, "London": 15},
        "fahrenheit": {"San Francisco": 64, "New York": 72, "London": 59}
    }
    
    location_key = next(
        (key for key in mock_temps[units] if key.lower() in location.lower()),
        None
    )
    
    if location_key:
        temp = mock_temps[units][location_key]
        return {
            "location": location_key,
            "temperature": temp,
            "units": units,
            "conditions": "Partly cloudy"
        }
    
    return {
        "error": f"Weather data not available for {location}",
        "location": location
    }


def search_web_tool(query: str, max_results: int = 5) -> dict:
    """Search the web (mock implementation)."""
    # Mock search results
    mock_results = [
        {
            "title": f"Result for '{query}' - {i+1}",
            "url": f"https://example.com/result{i+1}",
            "snippet": f"This is a mock search result for {query}..."
        }
        for i in range(min(max_results, 3))
    ]
    
    return {
        "query": query,
        "results": mock_results,
        "total": len(mock_results)
    }


def setup_example_tools(registry: ToolRegistry) -> None:
    """Register example tools in registry."""
    
    # Register calculator
    registry.register(
        name="calculator",
        func=calculator_tool,
        description="Perform mathematical calculations",
        schema=CalculatorParams,
        category="utility",
        tags=["math", "calculation"]
    )
    
    # Register weather
    registry.register(
        name="get_weather",
        func=get_weather_tool,
        description="Get current weather for a location",
        schema=WeatherParams,
        category="data",
        tags=["weather", "location"]
    )
    
    # Register search
    registry.register(
        name="search_web",
        func=search_web_tool,
        description="Search the web for information",
        schema=SearchParams,
        category="data",
        tags=["search", "web"]
    )
