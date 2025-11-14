"""OpenAI integration for tool execution."""

import json
from typing import Any, Dict, List, Optional

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .registry import ToolRegistry


class RegistryAgent:
    """
    Agent that integrates tool registry with OpenAI.
    
    Features:
    - Chat with tool calling
    - Parallel tool execution
    - Error handling
    - Mock mode for testing
    """
    
    def __init__(
        self,
        registry: ToolRegistry,
        api_key: str,
        model: str = "gpt-4",
        use_real_api: bool = True
    ):
        self.registry = registry
        self.model = model
        self.use_real_api = use_real_api and OPENAI_AVAILABLE
        
        if self.use_real_api:
            self.client = AsyncOpenAI(api_key=api_key)
        else:
            self.client = None
    
    async def chat(
        self,
        message: str,
        user: Optional[dict] = None,
        max_iterations: int = 5
    ) -> str:
        """
        Chat with tool calling support.
        
        Args:
            message: User message
            user: Optional user context
            max_iterations: Max tool calling iterations
            
        Returns:
            Assistant response
        """
        if self.use_real_api and self.client:
            return await self._chat_real(message, user, max_iterations)
        else:
            return await self._chat_mock(message, user)
    
    async def _chat_real(
        self,
        message: str,
        user: Optional[dict],
        max_iterations: int
    ) -> str:
        """Chat using real OpenAI API."""
        messages = [{"role": "user", "content": message}]
        tools = self.registry.get_all_openai_schemas()
        
        for iteration in range(max_iterations):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            assistant_message = response.choices[0].message
            
            # No tool calls - return response
            if not assistant_message.tool_calls:
                return assistant_message.content or ""
            
            # Execute tool calls
            messages.append(assistant_message)
            
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                # Execute tool
                result = self.registry.execute(tool_name, arguments, user=user)
                
                # Add result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
        
        return "Max iterations reached"
    
    async def _chat_mock(self, message: str, user: Optional[dict]) -> str:
        """Mock chat for testing without API."""
        message_lower = message.lower()
        
        # Simple pattern matching for demo
        if "weather" in message_lower:
            # Extract location (simple heuristic)
            location = "San Francisco"
            if "new york" in message_lower:
                location = "New York"
            elif "london" in message_lower:
                location = "London"
            
            result = self.registry.execute(
                "get_weather",
                {"location": location, "units": "celsius"},
                user=user
            )
            
            if result["success"]:
                weather = result["result"]
                return f"The weather in {weather['location']} is {weather['temperature']}°{weather['units'][0].upper()} and {weather['conditions']}."
            else:
                return f"Sorry, I couldn't get the weather: {result.get('error')}"
        
        elif any(op in message_lower for op in ["add", "plus", "+", "subtract", "minus", "-", "multiply", "times", "*", "divide", "/"]):
            # Simple math parsing
            import re
            
            # Try to find numbers
            numbers = re.findall(r'\d+(?:\.\d+)?', message)
            if len(numbers) >= 2:
                a = float(numbers[0])
                b = float(numbers[1])
                
                # Determine operation
                operation = "add"
                if "subtract" in message_lower or "minus" in message_lower or "-" in message:
                    operation = "subtract"
                elif "multiply" in message_lower or "times" in message_lower or "*" in message:
                    operation = "multiply"
                elif "divide" in message_lower or "/" in message:
                    operation = "divide"
                
                result = self.registry.execute(
                    "calculator",
                    {"operation": operation, "a": a, "b": b},
                    user=user
                )
                
                if result["success"]:
                    calc_result = result["result"]["result"]
                    return f"The result is {calc_result}."
                else:
                    return f"Sorry, I couldn't calculate: {result.get('error')}"
        
        elif "search" in message_lower:
            # Extract search query (simple heuristic)
            query = message.replace("search for", "").replace("search", "").strip()
            
            result = self.registry.execute(
                "search_web",
                {"query": query, "max_results": 5},
                user=user
            )
            
            if result["success"]:
                search_result = result["result"]
                return f"I found {search_result['total']} results for '{query}':\n" + \
                       "\n".join([f"- {r['title']}" for r in search_result['results']])
            else:
                return f"Sorry, I couldn't search: {result.get('error')}"
        
        return "I can help you with weather, calculations, or web searches. What would you like to know?"
