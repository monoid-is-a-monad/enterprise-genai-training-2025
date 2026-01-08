"""
Week 3 - Exercise 3: Function Calling Implementation

Learning Objectives:
- Implement function calling with proper tool definitions
- Build multi-tool agents that select appropriate tools
- Handle function execution safely with validation
- Create autonomous agents with planning capabilities
- Implement error handling and retry logic for tool use

Scenario:
You're building a travel planning assistant that can search flights, check
weather, convert currencies, and make recommendations. The assistant should
use the right tools for each task and handle errors gracefully.

Time: 90 minutes
"""

import os
from openai import OpenAI
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import json
import inspect
from datetime import datetime
from enum import Enum

# TODO: Initialize your OpenAI client
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# Part 1: Tool Implementation and Schema Generation (25 minutes)
# ============================================================================

# Implement these travel-related tool functions

def search_flights(
    origin: str,
    destination: str,
    date: str,
    max_price: Optional[float] = None
) -> Dict[str, Any]:
    """
    Search for available flights.
    
    Args:
        origin: Departure airport code (e.g., "SFO")
        destination: Arrival airport code (e.g., "LAX")
        date: Travel date in YYYY-MM-DD format
        max_price: Maximum price in USD (optional)
    
    Returns:
        Dictionary with flight options
    """
    # TODO: Implement mock flight search
    # Return realistic mock data with:
    # - Flight numbers
    # - Airlines
    # - Departure/arrival times
    # - Prices
    # - Available seats
    pass


def get_weather(location: str, date: str) -> Dict[str, Any]:
    """
    Get weather forecast for a location.
    
    Args:
        location: City name or airport code
        date: Date in YYYY-MM-DD format
    
    Returns:
        Dictionary with weather information
    """
    # TODO: Implement mock weather lookup
    # Return realistic mock data with:
    # - Temperature (high/low)
    # - Conditions (sunny, rainy, etc.)
    # - Precipitation chance
    # - Wind speed
    pass


def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str
) -> Dict[str, Any]:
    """
    Convert between currencies.
    
    Args:
        amount: Amount to convert
        from_currency: Source currency code (e.g., "USD")
        to_currency: Target currency code (e.g., "EUR")
    
    Returns:
        Dictionary with conversion result
    """
    # TODO: Implement mock currency conversion
    # Use approximate exchange rates:
    # USD to EUR: 0.85, USD to GBP: 0.73, USD to JPY: 110
    pass


def search_hotels(
    location: str,
    check_in: str,
    check_out: str,
    guests: int = 1,
    max_price_per_night: Optional[float] = None
) -> Dict[str, Any]:
    """
    Search for available hotels.
    
    Args:
        location: City name or area
        check_in: Check-in date in YYYY-MM-DD format
        check_out: Check-out date in YYYY-MM-DD format
        guests: Number of guests
        max_price_per_night: Maximum price per night in USD (optional)
    
    Returns:
        Dictionary with hotel options
    """
    # TODO: Implement mock hotel search
    # Return realistic mock data with:
    # - Hotel names
    # - Star ratings
    # - Prices per night
    # - Amenities
    # - Availability
    pass


class FunctionSchemaGenerator:
    """
    TODO: Auto-generate OpenAI function schemas from Python functions
    
    Should extract:
    - Function name
    - Description from docstring
    - Parameters with types
    - Required vs optional parameters
    """
    
    @staticmethod
    def generate_schema(func: Callable) -> Dict[str, Any]:
        """
        TODO: Generate OpenAI function schema from Python function
        
        Use:
        - inspect.signature() for parameters
        - inspect.getdoc() for description
        - Type hints for parameter types
        
        Return schema in format:
        {
            "type": "function",
            "function": {
                "name": "function_name",
                "description": "Function description",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param_name": {
                            "type": "string",
                            "description": "Parameter description"
                        }
                    },
                    "required": ["required_param"]
                }
            }
        }
        """
        # TODO: Implement this method
        pass
    
    @staticmethod
    def python_type_to_json_type(python_type: type) -> str:
        """
        TODO: Convert Python types to JSON schema types
        
        Mappings:
        - str -> "string"
        - int, float -> "number"
        - bool -> "boolean"
        - list, List -> "array"
        - dict, Dict -> "object"
        - Optional[T] -> extract T
        """
        # TODO: Implement this method
        pass


def test_schema_generation():
    """Test automatic schema generation"""
    # TODO: Generate schemas for all tool functions
    # generator = FunctionSchemaGenerator()
    
    # tools = [search_flights, get_weather, convert_currency, search_hotels]
    # schemas = [generator.generate_schema(tool) for tool in tools]
    
    # print("Generated Schemas:")
    # for schema in schemas:
    #     print(json.dumps(schema, indent=2))


# ============================================================================
# Part 2: Basic Function Calling System (20 minutes)
# ============================================================================

class TravelAgent:
    """
    TODO: Implement a basic travel agent with function calling
    
    Should be able to:
    1. Accept user queries
    2. Determine which tools to use
    3. Execute tool calls
    4. Return natural language responses
    """
    
    def __init__(self, client: OpenAI):
        self.client = client
        
        # TODO: Register your tools
        self.tools = {
            "search_flights": search_flights,
            "get_weather": get_weather,
            "convert_currency": convert_currency,
            "search_hotels": search_hotels,
        }
        
        # TODO: Generate schemas for all tools
        self.tool_schemas = []
        
        self.conversation_history = []
    
    def chat(self, user_message: str) -> str:
        """
        TODO: Process user message with function calling
        
        Steps:
        1. Add user message to history
        2. Call OpenAI API with tools available
        3. Check if model wants to call functions
        4. Execute function calls
        5. Send results back to model
        6. Get final response
        7. Return response to user
        
        Handle multiple tool calls in sequence if needed
        """
        # TODO: Implement this method
        pass
    
    def _execute_function(
        self,
        function_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        TODO: Execute a function call safely
        
        Should:
        - Validate function exists
        - Validate arguments
        - Handle execution errors
        - Return result or error message
        """
        # TODO: Implement this method
        pass


def test_travel_agent():
    """Test basic travel agent functionality"""
    # TODO: Initialize agent
    # agent = TravelAgent(client)
    
    # Test queries
    queries = [
        "Find me flights from SFO to LAX on 2025-02-15",
        "What's the weather like in Los Angeles?",
        "How much is 500 USD in Euros?",
        "I need a hotel in Los Angeles from Feb 15-17 for 2 guests",
    ]
    
    # TODO: Uncomment and test
    # for query in queries:
    #     print(f"\nUser: {query}")
    #     response = agent.chat(query)
    #     print(f"Agent: {response}")


# ============================================================================
# Part 3: Multi-Step Agent with Planning (25 minutes)
# ============================================================================

@dataclass
class ExecutionPlan:
    """Represents a plan for multi-step execution"""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    estimated_tools: List[str] = field(default_factory=list)


class PlanningTravelAgent:
    """
    TODO: Implement an agent that plans before acting
    
    Features:
    - Break complex requests into steps
    - Generate execution plan
    - Execute plan step by step
    - Handle dependencies between steps
    - Adapt plan if steps fail
    """
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.tools = {
            "search_flights": search_flights,
            "get_weather": get_weather,
            "convert_currency": convert_currency,
            "search_hotels": search_hotels,
        }
        self.tool_schemas = []  # TODO: Generate schemas
        self.execution_history = []
    
    def create_plan(self, user_request: str) -> ExecutionPlan:
        """
        TODO: Create an execution plan for the request
        
        Use the model to:
        1. Analyze the request
        2. Break it into logical steps
        3. Identify which tools are needed
        4. Determine order of execution
        5. Identify dependencies
        
        Example plan for "Plan a trip to Paris":
        Step 1: Search flights to Paris
        Step 2: Get weather forecast for Paris
        Step 3: Search hotels in Paris
        Step 4: Convert costs to user's currency
        """
        # TODO: Implement this method
        pass
    
    def execute_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """
        TODO: Execute the plan step by step
        
        Should:
        - Execute steps in order
        - Pass results between steps if needed
        - Handle failures gracefully
        - Track execution state
        - Provide progress updates
        
        Return:
        {
            "success": bool,
            "results": {step_id: result},
            "summary": "Natural language summary",
            "execution_log": [...]
        }
        """
        # TODO: Implement this method
        pass
    
    def handle_complex_request(self, user_request: str) -> str:
        """
        TODO: Complete pipeline: plan -> execute -> summarize
        
        1. Create plan
        2. Show plan to user (optional)
        3. Execute plan
        4. Summarize results
        5. Return final response
        """
        # TODO: Implement this method
        pass


def test_planning_agent():
    """Test planning agent with complex requests"""
    # TODO: Initialize planning agent
    # agent = PlanningTravelAgent(client)
    
    complex_request = """
    I want to plan a trip to Tokyo, Japan. I'm leaving from San Francisco
    on March 1st and returning March 8th. I need to know the weather, find
    affordable flights (under $1000), and book a hotel for 2 people. Also
    tell me how much 2000 USD is in Japanese Yen.
    """
    
    # TODO: Uncomment and test
    # print("Complex Request:", complex_request)
    # print("\n" + "="*80)
    # response = agent.handle_complex_request(complex_request)
    # print(response)


# ============================================================================
# Part 4: Safe Function Execution with Validation (20 minutes)
# ============================================================================

class ToolPermission(Enum):
    """Permission levels for tools"""
    READ_ONLY = "read_only"  # Can only read data
    WRITE = "write"  # Can modify data
    ADMIN = "admin"  # Can do anything


@dataclass
class ToolMetadata:
    """Metadata about a tool"""
    name: str
    function: Callable
    schema: Dict[str, Any]
    permission_required: ToolPermission
    rate_limit_per_minute: Optional[int] = None
    cost_per_call: float = 0.0
    requires_confirmation: bool = False


class SafeExecutionManager:
    """
    TODO: Manage safe tool execution with:
    - Permission checking
    - Input validation
    - Rate limiting
    - Cost tracking
    - Confirmation for sensitive operations
    - Execution logging
    - Error handling
    """
    
    def __init__(self, user_permission_level: ToolPermission = ToolPermission.READ_ONLY):
        self.user_permission = user_permission_level
        self.tools: Dict[str, ToolMetadata] = {}
        self.execution_log = []
        self.rate_limit_tracker = {}
        self.total_cost = 0.0
    
    def register_tool(
        self,
        function: Callable,
        permission_required: ToolPermission,
        rate_limit: Optional[int] = None,
        cost: float = 0.0,
        requires_confirmation: bool = False
    ):
        """
        TODO: Register a tool with its metadata
        """
        # TODO: Implement this method
        pass
    
    def validate_permission(self, tool_name: str) -> bool:
        """
        TODO: Check if user has permission to use tool
        """
        # TODO: Implement this method
        pass
    
    def validate_rate_limit(self, tool_name: str) -> bool:
        """
        TODO: Check if rate limit allows this call
        """
        # TODO: Implement this method
        pass
    
    def validate_arguments(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        TODO: Validate arguments before execution
        
        Check:
        - Required arguments present
        - Argument types correct
        - Argument values in valid ranges
        - No injection attempts
        
        Return: {"valid": bool, "errors": [...]}
        """
        # TODO: Implement this method
        pass
    
    def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        require_confirmation: bool = True
    ) -> Dict[str, Any]:
        """
        TODO: Execute tool with all safety checks
        
        Steps:
        1. Validate permission
        2. Validate rate limit
        3. Validate arguments
        4. Request confirmation if needed
        5. Execute function
        6. Log execution
        7. Update costs and rate limits
        8. Return result
        
        Return:
        {
            "success": bool,
            "result": Any,
            "error": Optional[str],
            "cost": float,
            "execution_time": float
        }
        """
        # TODO: Implement this method
        pass
    
    def get_execution_report(self) -> Dict[str, Any]:
        """
        TODO: Generate execution report
        
        Include:
        - Total executions
        - Success/failure rate
        - Total cost
        - Tool usage distribution
        - Rate limit violations
        """
        # TODO: Implement this method
        pass


def test_safe_execution():
    """Test safe execution manager"""
    # TODO: Initialize manager
    # manager = SafeExecutionManager(user_permission_level=ToolPermission.READ_ONLY)
    
    # Register tools with different permission levels
    # manager.register_tool(
    #     search_flights,
    #     permission_required=ToolPermission.READ_ONLY,
    #     rate_limit=10,
    #     cost=0.01
    # )
    
    # Try to execute with proper validation
    # result = manager.execute_tool(
    #     "search_flights",
    #     {"origin": "SFO", "destination": "LAX", "date": "2025-02-15"}
    # )
    # print("Execution result:", result)


# ============================================================================
# Reflection Questions
# ============================================================================

"""
After completing the exercises, reflect on these questions:

1. FUNCTION CALLING DESIGN:
   - How did you decide which functions to create?
   - What made a good function vs a bad function?
   - How did you handle functions that depend on other functions?

2. SCHEMA GENERATION:
   - Was automatic schema generation reliable?
   - What edge cases did you encounter?
   - When would you write schemas manually?

3. PLANNING:
   - How effective was the planning step?
   - When did plans need to be adjusted?
   - What information was needed for good planning?

4. SAFETY:
   - What security concerns did you identify?
   - How did you validate function inputs?
   - What confirmation strategies made sense?
   - How would you prevent misuse?

5. ERROR HANDLING:
   - What types of errors occurred?
   - How did you recover from failures?
   - What retry strategies worked?

6. PRODUCTION CONSIDERATIONS:
   - What monitoring would you add?
   - How would you test function calling systems?
   - What rate limits make sense?
   - How would you version functions?

Write your reflections in: exercise-03-reflections.md
"""


# ============================================================================
# Optional Challenge: Context-Aware Tool Selection
# ============================================================================

"""
CHALLENGE: Implement an agent that learns which tools are most useful for
different types of queries.

Features:
- Track tool usage patterns
- Learn query -> tool mappings
- Suggest tools proactively
- Optimize tool call order based on history

Hints:
- Build a query classifier
- Track tool success rates per query type
- Use embeddings for query similarity
- Implement tool recommendation system
"""


if __name__ == "__main__":
    print("Week 3 - Exercise 3: Function Calling Implementation")
    print("="*80)
    
    # Uncomment to run tests as you complete each part
    # test_schema_generation()
    # test_travel_agent()
    # test_planning_agent()
    # test_safe_execution()
