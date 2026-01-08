# Lab 3: Function Calling System - Solutions

**Week 3 - Advanced Prompting & OpenAI API**

**Provided by:** ADC ENGINEERING & CONSULTING LTD

---

## Table of Contents

1. [Part 1: Basic Function Calling](#part-1-basic-function-calling)
2. [Part 2: Function Schema Builder](#part-2-function-schema-builder)
3. [Part 3: Multi-Tool System](#part-3-multi-tool-system)
4. [Part 4: Function Execution Manager](#part-4-function-execution-manager)
5. [Part 5: Autonomous Agent](#part-5-autonomous-agent)
6. [Part 6: Error Handling & Safety](#part-6-error-handling--safety)
7. [Best Practices](#best-practices)

---

## Part 1: Basic Function Calling

### Understanding the Flow

```
1. User query → API with function definitions
2. Model decides to call function(s)
3. Execute function(s) in your code
4. Send results back to model
5. Model generates final response
```

### Exercise 1.1: Tip Calculator

```python
import json
from openai import OpenAI
from typing import Dict, List

client = OpenAI()

def calculate_tip(bill_amount: float, tip_percentage: float = 15.0) -> dict:
    """
    Calculate tip amount and total bill.
    
    Args:
        bill_amount: Total bill amount
        tip_percentage: Tip percentage (default 15%)
    
    Returns:
        Dict with tip amount and total
    """
    tip_amount = bill_amount * (tip_percentage / 100)
    total_amount = bill_amount + tip_amount
    
    return {
        "bill_amount": round(bill_amount, 2),
        "tip_percentage": tip_percentage,
        "tip_amount": round(tip_amount, 2),
        "total_amount": round(total_amount, 2),
        "currency": "USD"
    }

# Define function schema
tip_calculator_function = {
    "type": "function",
    "function": {
        "name": "calculate_tip",
        "description": "Calculate tip amount and total bill with tip included",
        "parameters": {
            "type": "object",
            "properties": {
                "bill_amount": {
                    "type": "number",
                    "description": "The total bill amount before tip"
                },
                "tip_percentage": {
                    "type": "number",
                    "description": "Tip percentage to calculate (e.g., 15 for 15%)",
                    "default": 15.0
                }
            },
            "required": ["bill_amount"]
        }
    }
}

def test_tip_calculator():
    """Test the tip calculator with function calling."""
    
    messages = [
        {"role": "user", "content": "Calculate a 20% tip on a $50 bill"}
    ]
    
    print("=" * 80)
    print("TIP CALCULATOR FUNCTION CALLING")
    print("=" * 80)
    print(f"User: {messages[0]['content']}\n")
    
    # First API call - model decides to call function
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        tools=[tip_calculator_function],
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    
    # Check if function call was made
    if response_message.tool_calls:
        tool_call = response_message.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        print(f"🤖 Model called function: {function_name}")
        print(f"📝 Arguments: {json.dumps(function_args, indent=2)}\n")
        
        # Execute the function
        function_result = calculate_tip(**function_args)
        print(f"💰 Function result: {json.dumps(function_result, indent=2)}\n")
        
        # Add function call and result to messages
        messages.append(response_message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(function_result)
        })
        
        # Second API call - model generates final response
        final_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        print(f"✅ Final response:\n{final_response.choices[0].message.content}")
    else:
        print(f"Direct response: {response_message.content}")

test_tip_calculator()
```

**Expected Output:**
```
================================================================================
TIP CALCULATOR FUNCTION CALLING
================================================================================
User: Calculate a 20% tip on a $50 bill

🤖 Model called function: calculate_tip
📝 Arguments: {
  "bill_amount": 50.0,
  "tip_percentage": 20.0
}

💰 Function result: {
  "bill_amount": 50.0,
  "tip_percentage": 20.0,
  "tip_amount": 10.0,
  "total_amount": 60.0,
  "currency": "USD"
}

✅ Final response:
For a $50 bill with a 20% tip, you should leave a $10.00 tip, making your total $60.00.
```

### Complete Basic Function Calling Example

```python
class BasicFunctionCaller:
    """
    Simple function calling implementation.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.client = OpenAI()
        self.functions = {}
        self.function_schemas = []
    
    def register_function(self, func: callable, schema: dict):
        """
        Register a function and its schema.
        
        Args:
            func: Python function
            schema: OpenAI function schema
        """
        function_name = schema["function"]["name"]
        self.functions[function_name] = func
        self.function_schemas.append(schema)
        print(f"✓ Registered function: {function_name}")
    
    def call(self, user_message: str) -> str:
        """
        Make a function call based on user message.
        
        Args:
            user_message: User's input message
        
        Returns:
            Final response string
        """
        messages = [{"role": "user", "content": user_message}]
        
        # First API call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.function_schemas,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        
        # Check for function calls
        if not response_message.tool_calls:
            return response_message.content
        
        # Execute function(s)
        messages.append(response_message)
        
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Call the actual function
            function_to_call = self.functions[function_name]
            function_result = function_to_call(**function_args)
            
            # Add result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(function_result)
            })
        
        # Second API call with function results
        final_response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        
        return final_response.choices[0].message.content

# Usage
caller = BasicFunctionCaller()

# Register functions
caller.register_function(calculate_tip, tip_calculator_function)

# Call
result = caller.call("What's a 18% tip on $75?")
print(f"Result: {result}")
```

---

## Part 2: Function Schema Builder

### Automatic Schema Generation

```python
import inspect
from typing import get_type_hints, Callable, get_origin, get_args
from dataclasses import dataclass, field

@dataclass
class FunctionParameter:
    """Represents a function parameter."""
    name: str
    type: type
    required: bool
    default: any = None
    description: str = ""

class FunctionSchemaBuilder:
    """
    Automatically build OpenAI function schemas from Python functions.
    Enhanced version with better type handling and documentation parsing.
    """
    
    TYPE_MAPPING = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        List: "array",
        Dict: "object"
    }
    
    @classmethod
    def build_schema(cls, func: Callable, description: str = None) -> dict:
        """
        Build OpenAI function schema from Python function.
        
        Args:
            func: Python function to convert
            description: Optional override description
        
        Returns:
            OpenAI function schema dictionary
        """
        # Get function signature and type hints
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        # Parse docstring
        docstring = inspect.getdoc(func) or ""
        func_description = description or cls._extract_description(docstring)
        param_descriptions = cls._parse_parameter_descriptions(docstring)
        
        # Build parameters
        parameters = cls._build_parameters(sig, type_hints, param_descriptions)
        
        # Build schema
        schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func_description,
                "parameters": {
                    "type": "object",
                    "properties": parameters["properties"],
                    "required": parameters["required"]
                }
            }
        }
        
        return schema
    
    @classmethod
    def _extract_description(cls, docstring: str) -> str:
        """Extract main description from docstring."""
        if not docstring:
            return "No description provided"
        
        lines = docstring.split("\n")
        description_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith("Args:") and not line.startswith("Returns:"):
                description_lines.append(line)
            elif line.startswith("Args:") or line.startswith("Returns:"):
                break
        
        return " ".join(description_lines)
    
    @classmethod
    def _parse_parameter_descriptions(cls, docstring: str) -> Dict[str, str]:
        """Parse parameter descriptions from docstring."""
        param_descriptions = {}
        
        if not docstring or "Args:" not in docstring:
            return param_descriptions
        
        lines = docstring.split("\n")
        in_args_section = False
        
        for line in lines:
            line_stripped = line.strip()
            
            if line_stripped.startswith("Args:"):
                in_args_section = True
                continue
            
            if in_args_section:
                if line_stripped.startswith("Returns:") or line_stripped.startswith("Raises:"):
                    break
                
                if ":" in line_stripped:
                    parts = line_stripped.split(":", 1)
                    param_name = parts[0].strip()
                    param_desc = parts[1].strip()
                    param_descriptions[param_name] = param_desc
        
        return param_descriptions
    
    @classmethod
    def _build_parameters(
        cls,
        sig: inspect.Signature,
        type_hints: Dict,
        param_descriptions: Dict[str, str]
    ) -> Dict:
        """Build parameters section of schema."""
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name in ["self", "cls"]:
                continue
            
            # Get parameter type
            param_type = type_hints.get(param_name, str)
            
            # Handle generic types (List[str], Dict[str, int], etc.)
            origin_type = get_origin(param_type) or param_type
            type_args = get_args(param_type)
            
            # Build property schema
            property_schema = {
                "type": cls.TYPE_MAPPING.get(origin_type, "string")
            }
            
            # Add description
            if param_name in param_descriptions:
                property_schema["description"] = param_descriptions[param_name]
            
            # Handle array item types
            if origin_type == list and type_args:
                item_type = type_args[0]
                property_schema["items"] = {
                    "type": cls.TYPE_MAPPING.get(item_type, "string")
                }
            
            # Handle enums (if using Enum class)
            if hasattr(param_type, "__members__"):
                property_schema["enum"] = list(param_type.__members__.keys())
            
            # Add default value
            if param.default != inspect.Parameter.empty:
                property_schema["default"] = param.default
            else:
                required.append(param_name)
            
            properties[param_name] = property_schema
        
        return {
            "properties": properties,
            "required": required
        }

# Example usage with enhanced schema builder

def search_database(
    query: str,
    limit: int = 10,
    include_metadata: bool = False,
    sort_by: str = "relevance"
) -> List[dict]:
    """
    Search the database for matching records.
    
    This function performs a full-text search across the database
    and returns matching results with optional metadata.
    
    Args:
        query: Search query string to find matching records
        limit: Maximum number of results to return (default: 10)
        include_metadata: Whether to include metadata in results
        sort_by: Field to sort results by (relevance, date, title)
    
    Returns:
        List of matching database records
    """
    # Mock implementation
    results = []
    for i in range(min(limit, 3)):
        result = {
            "id": i + 1,
            "title": f"Result {i + 1}",
            "content": f"Content matching '{query}'"
        }
        if include_metadata:
            result["metadata"] = {
                "created_at": "2024-01-01",
                "score": 0.95 - (i * 0.1)
            }
        results.append(result)
    
    return results

# Build schema
schema = FunctionSchemaBuilder.build_schema(search_database)

print("=" * 80)
print("AUTO-GENERATED SCHEMA")
print("=" * 80)
print(json.dumps(schema, indent=2))

# Test it
messages = [{"role": "user", "content": "Search for Python tutorials, show top 5 with metadata"}]

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    tools=[schema],
    tool_choice="auto"
)

if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    
    print(f"\n✓ Function: {tool_call.function.name}")
    print(f"✓ Arguments: {json.dumps(args, indent=2)}")
    
    # Execute
    result = search_database(**args)
    print(f"✓ Results: {json.dumps(result, indent=2)}")
```

### Exercise 2.1: Utility Functions with Auto-Generated Schemas

```python
from math import radians, sin, cos, sqrt, atan2

def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """
    Convert amount from one currency to another.
    
    Performs currency conversion using current exchange rates.
    
    Args:
        amount: Amount to convert
        from_currency: Source currency code (e.g., USD, EUR, GBP)
        to_currency: Target currency code (e.g., USD, EUR, GBP)
    
    Returns:
        Conversion result with original amount, converted amount, and rate
    """
    # Mock exchange rates
    rates = {
        "USD": 1.0,
        "EUR": 0.85,
        "GBP": 0.73,
        "JPY": 110.0,
        "CAD": 1.25
    }
    
    # Convert to USD first, then to target currency
    usd_amount = amount / rates.get(from_currency, 1.0)
    converted_amount = usd_amount * rates.get(to_currency, 1.0)
    rate = converted_amount / amount
    
    return {
        "original_amount": amount,
        "from_currency": from_currency,
        "to_currency": to_currency,
        "converted_amount": round(converted_amount, 2),
        "exchange_rate": round(rate, 4),
        "timestamp": "2024-01-01T00:00:00Z"
    }

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float, unit: str = "km") -> dict:
    """
    Calculate distance between two geographic coordinates.
    
    Uses the Haversine formula to calculate the great-circle distance
    between two points on Earth.
    
    Args:
        lat1: Latitude of first point in decimal degrees
        lon1: Longitude of first point in decimal degrees
        lat2: Latitude of second point in decimal degrees
        lon2: Longitude of second point in decimal degrees
        unit: Unit for distance ('km' for kilometers or 'mi' for miles)
    
    Returns:
        Distance between the two points with coordinates and unit
    """
    # Earth radius
    R = 6371  # kilometers
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance_km = R * c
    
    # Convert to miles if requested
    if unit == "mi":
        distance = distance_km * 0.621371
    else:
        distance = distance_km
        unit = "km"
    
    return {
        "point1": {"latitude": lat1, "longitude": lon1},
        "point2": {"latitude": lat2, "longitude": lon2},
        "distance": round(distance, 2),
        "unit": unit
    }

# Generate schemas automatically
currency_schema = FunctionSchemaBuilder.build_schema(convert_currency)
distance_schema = FunctionSchemaBuilder.build_schema(calculate_distance)

print("\n" + "=" * 80)
print("CURRENCY CONVERTER SCHEMA")
print("=" * 80)
print(json.dumps(currency_schema, indent=2))

print("\n" + "=" * 80)
print("DISTANCE CALCULATOR SCHEMA")
print("=" * 80)
print(json.dumps(distance_schema, indent=2))

# Test both functions
test_caller = BasicFunctionCaller()
test_caller.register_function(convert_currency, currency_schema)
test_caller.register_function(calculate_distance, distance_schema)

print("\n" + "=" * 80)
print("TESTING FUNCTIONS")
print("=" * 80)

# Test currency conversion
result1 = test_caller.call("Convert 100 USD to EUR")
print(f"\n1. Currency conversion:\n{result1}")

# Test distance calculation
result2 = test_caller.call("What's the distance from New York (40.7128°N, 74.0060°W) to London (51.5074°N, 0.1278°W)?")
print(f"\n2. Distance calculation:\n{result2}")
```

---

## Part 3: Multi-Tool System

### Exercise 3.1: Multi-Function Agent

```python
class MultiToolAgent:
    """
    Agent that can call multiple functions in sequence.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo", max_iterations: int = 10):
        self.model = model
        self.max_iterations = max_iterations
        self.client = OpenAI()
        
        self.functions = {}
        self.function_schemas = []
        self.call_history = []
    
    def register_function(self, func: callable, schema: dict = None):
        """
        Register a function with the agent.
        
        Args:
            func: Python function
            schema: Optional schema (auto-generated if not provided)
        """
        if schema is None:
            schema = FunctionSchemaBuilder.build_schema(func)
        
        function_name = schema["function"]["name"]
        self.functions[function_name] = func
        self.function_schemas.append(schema)
        
        print(f"✓ Registered: {function_name}")
    
    def run(self, user_message: str, verbose: bool = True) -> dict:
        """
        Run the agent with a user message.
        
        Args:
            user_message: User's input
            verbose: Whether to print detailed execution
        
        Returns:
            Dictionary with final response and execution history
        """
        messages = [{"role": "user", "content": user_message}]
        self.call_history = []
        
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"ITERATION {iteration}")
                print(f"{'='*80}")
            
            # Call API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.function_schemas,
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            
            # Check if done (no function calls)
            if not response_message.tool_calls:
                if verbose:
                    print(f"\n✅ Final response: {response_message.content}")
                
                return {
                    "response": response_message.content,
                    "iterations": iteration,
                    "function_calls": self.call_history
                }
            
            # Process function calls
            messages.append(response_message)
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if verbose:
                    print(f"\n🔧 Calling: {function_name}")
                    print(f"📝 Args: {json.dumps(function_args, indent=2)}")
                
                # Execute function
                try:
                    function_result = self.functions[function_name](**function_args)
                    
                    if verbose:
                        print(f"✓ Result: {json.dumps(function_result, indent=2)[:200]}...")
                    
                    # Record call
                    self.call_history.append({
                        "function": function_name,
                        "arguments": function_args,
                        "result": function_result
                    })
                    
                    # Add result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(function_result)
                    })
                
                except Exception as e:
                    error_message = f"Error executing {function_name}: {str(e)}"
                    
                    if verbose:
                        print(f"❌ {error_message}")
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({"error": error_message})
                    })
        
        # Max iterations reached
        return {
            "response": "Maximum iterations reached without completion",
            "iterations": iteration,
            "function_calls": self.call_history
        }

# Create agent with multiple tools
agent = MultiToolAgent()

# Register all our functions
agent.register_function(calculate_tip)
agent.register_function(convert_currency)
agent.register_function(calculate_distance)
agent.register_function(search_database)

# Test multi-step task
result = agent.run(
    "I had dinner in Paris that cost 75 EUR. Convert that to USD, then calculate a 20% tip.",
    verbose=True
)

print("\n" + "=" * 80)
print("EXECUTION SUMMARY")
print("=" * 80)
print(f"Total iterations: {result['iterations']}")
print(f"Functions called: {len(result['function_calls'])}")
print(f"\nFunction call sequence:")
for i, call in enumerate(result['function_calls'], 1):
    print(f"{i}. {call['function']}({', '.join(f'{k}={v}' for k, v in call['arguments'].items())})")
print(f"\nFinal response:\n{result['response']}")
```

**Expected Output:**
```
✓ Registered: calculate_tip
✓ Registered: convert_currency
✓ Registered: calculate_distance
✓ Registered: search_database

================================================================================
ITERATION 1
================================================================================

🔧 Calling: convert_currency
📝 Args: {
  "amount": 75.0,
  "from_currency": "EUR",
  "to_currency": "USD"
}
✓ Result: {"original_amount": 75.0, "from_currency": "EUR", "to_currency": "USD", "converted_amount": 88.24, "exchange_rate": 1.1765, "timestamp": "2024-01-01T00:00:00Z"}

================================================================================
ITERATION 2
================================================================================

🔧 Calling: calculate_tip
📝 Args: {
  "bill_amount": 88.24,
  "tip_percentage": 20.0
}
✓ Result: {"bill_amount": 88.24, "tip_percentage": 20.0, "tip_amount": 17.65, "total_amount": 105.89, "currency": "USD"}

================================================================================
ITERATION 3
================================================================================

✅ Final response: Your 75 EUR dinner converts to $88.24 USD. With a 20% tip ($17.65), your total would be $105.89 USD.

================================================================================
EXECUTION SUMMARY
================================================================================
Total iterations: 3
Functions called: 2

Function call sequence:
1. convert_currency(amount=75.0, from_currency=EUR, to_currency=USD)
2. calculate_tip(bill_amount=88.24, tip_percentage=20.0)

Final response:
Your 75 EUR dinner converts to $88.24 USD. With a 20% tip ($17.65), your total would be $105.89 USD.
```

---

## Part 4: Function Execution Manager

### Safe Function Execution with Validation

```python
from typing import Optional, Any
from pydantic import BaseModel, ValidationError
import traceback

class FunctionExecutionResult(BaseModel):
    """Result of a function execution."""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0

class FunctionExecutionManager:
    """
    Manage function execution with safety, validation, and monitoring.
    """
    
    def __init__(self):
        self.functions = {}
        self.execution_log = []
        self.allowed_modules = {"math", "json", "datetime"}  # Whitelist
    
    def register_safe_function(
        self,
        func: callable,
        allowed: bool = True,
        requires_confirmation: bool = False
    ):
        """
        Register a function with safety settings.
        
        Args:
            func: Function to register
            allowed: Whether function is allowed to execute
            requires_confirmation: Whether to ask for confirmation before execution
        """
        self.functions[func.__name__] = {
            "function": func,
            "allowed": allowed,
            "requires_confirmation": requires_confirmation,
            "call_count": 0,
            "error_count": 0
        }
    
    def execute_function(
        self,
        function_name: str,
        arguments: dict,
        timeout: Optional[int] = None,
        validate_output: Optional[callable] = None
    ) -> FunctionExecutionResult:
        """
        Safely execute a function with timeout and validation.
        
        Args:
            function_name: Name of function to execute
            arguments: Function arguments
            timeout: Execution timeout in seconds
            validate_output: Optional output validation function
        
        Returns:
            FunctionExecutionResult
        """
        import time
        
        start_time = time.time()
        
        # Check if function exists
        if function_name not in self.functions:
            return FunctionExecutionResult(
                success=False,
                error=f"Function '{function_name}' not registered"
            )
        
        func_info = self.functions[function_name]
        
        # Check if allowed
        if not func_info["allowed"]:
            return FunctionExecutionResult(
                success=False,
                error=f"Function '{function_name}' is not allowed to execute"
            )
        
        # Confirmation check
        if func_info["requires_confirmation"]:
            print(f"⚠️  Function '{function_name}' requires confirmation")
            print(f"Arguments: {json.dumps(arguments, indent=2)}")
            confirm = input("Execute? (yes/no): ")
            if confirm.lower() != "yes":
                return FunctionExecutionResult(
                    success=False,
                    error="Execution cancelled by user"
                )
        
        # Execute function
        try:
            func = func_info["function"]
            
            # Simple timeout implementation (for demonstration)
            result = func(**arguments)
            
            # Validate output if validator provided
            if validate_output:
                validation_result = validate_output(result)
                if not validation_result:
                    return FunctionExecutionResult(
                        success=False,
                        error="Output validation failed",
                        result=result
                    )
            
            # Update stats
            func_info["call_count"] += 1
            execution_time = time.time() - start_time
            
            # Log execution
            self.execution_log.append({
                "function": function_name,
                "arguments": arguments,
                "success": True,
                "execution_time": execution_time
            })
            
            return FunctionExecutionResult(
                success=True,
                result=result,
                execution_time=execution_time
            )
        
        except Exception as e:
            # Update error stats
            func_info["error_count"] += 1
            execution_time = time.time() - start_time
            
            # Log error
            self.execution_log.append({
                "function": function_name,
                "arguments": arguments,
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            })
            
            return FunctionExecutionResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                execution_time=execution_time
            )
    
    def get_function_stats(self, function_name: str) -> dict:
        """Get statistics for a function."""
        if function_name not in self.functions:
            return {"error": "Function not found"}
        
        func_info = self.functions[function_name]
        
        return {
            "name": function_name,
            "call_count": func_info["call_count"],
            "error_count": func_info["error_count"],
            "success_rate": (
                (func_info["call_count"] - func_info["error_count"]) / func_info["call_count"]
                if func_info["call_count"] > 0 else 0
            ),
            "allowed": func_info["allowed"],
            "requires_confirmation": func_info["requires_confirmation"]
        }
    
    def get_all_stats(self) -> dict:
        """Get statistics for all functions."""
        return {
            name: self.get_function_stats(name)
            for name in self.functions.keys()
        }

# Example usage
exec_manager = FunctionExecutionManager()

# Register functions with different safety levels
exec_manager.register_safe_function(calculate_tip, allowed=True, requires_confirmation=False)
exec_manager.register_safe_function(convert_currency, allowed=True, requires_confirmation=False)

# Safe function - won't actually delete anything
def delete_database(table_name: str) -> dict:
    """Mock dangerous function."""
    return {"status": "deleted", "table": table_name}

exec_manager.register_safe_function(delete_database, allowed=False)  # Not allowed

# Test execution
print("=" * 80)
print("SAFE FUNCTION EXECUTION")
print("=" * 80)

# Allowed function
result1 = exec_manager.execute_function(
    "calculate_tip",
    {"bill_amount": 50.0, "tip_percentage": 20.0}
)
print(f"\n1. Tip calculation:")
print(f"   Success: {result1.success}")
print(f"   Result: {result1.result}")
print(f"   Time: {result1.execution_time:.4f}s")

# Not allowed function
result2 = exec_manager.execute_function(
    "delete_database",
    {"table_name": "users"}
)
print(f"\n2. Delete database (should be blocked):")
print(f"   Success: {result2.success}")
print(f"   Error: {result2.error}")

# Non-existent function
result3 = exec_manager.execute_function(
    "nonexistent_function",
    {}
)
print(f"\n3. Non-existent function:")
print(f"   Success: {result3.success}")
print(f"   Error: {result3.error}")

# Get stats
print(f"\n{'='*80}")
print("FUNCTION STATISTICS")
print(f"{'='*80}")
stats = exec_manager.get_all_stats()
for name, stat in stats.items():
    print(f"\n{name}:")
    print(f"  Calls: {stat['call_count']}")
    print(f"  Errors: {stat['error_count']}")
    print(f"  Success rate: {stat['success_rate']*100:.1f}%")
    print(f"  Allowed: {stat['allowed']}")
```

---

## Part 5: Autonomous Agent

### Complete Autonomous Function-Calling Agent

```python
class AutonomousAgent:
    """
    Fully autonomous agent with function calling, planning, and error recovery.
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        max_iterations: int = 15,
        system_message: str = None
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.client = OpenAI()
        
        self.system_message = system_message or """You are a helpful AI assistant with access to various tools.
You can call functions to help answer questions and complete tasks.
Think step by step and call functions as needed.
If you need information from a function, call it before providing your final answer."""
        
        self.execution_manager = FunctionExecutionManager()
        self.function_schemas = []
        self.conversation_history = []
    
    def register_tool(self, func: callable, schema: dict = None, **safety_options):
        """
        Register a tool/function with the agent.
        
        Args:
            func: Python function
            schema: Optional schema (auto-generated if not provided)
            **safety_options: Safety settings (allowed, requires_confirmation)
        """
        if schema is None:
            schema = FunctionSchemaBuilder.build_schema(func)
        
        self.function_schemas.append(schema)
        self.execution_manager.register_safe_function(func, **safety_options)
        
        function_name = schema["function"]["name"]
        print(f"🔧 Registered tool: {function_name}")
    
    def run(self, task: str, verbose: bool = True) -> dict:
        """
        Run the agent on a task.
        
        Args:
            task: Task description
            verbose: Whether to print execution details
        
        Returns:
            Result dictionary with response and execution details
        """
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": task}
        ]
        
        iteration = 0
        function_calls = []
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"🤖 AGENT STARTING TASK")
            print(f"{'='*80}")
            print(f"Task: {task}\n")
        
        while iteration < self.max_iterations:
            iteration += 1
            
            if verbose:
                print(f"\n--- Iteration {iteration} ---")
            
            # Call API
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.function_schemas,
                    tool_choice="auto",
                    temperature=0.3
                )
            except Exception as e:
                return {
                    "success": False,
                    "error": f"API error: {str(e)}",
                    "iterations": iteration,
                    "function_calls": function_calls
                }
            
            response_message = response.choices[0].message
            
            # Check if task is complete
            if not response_message.tool_calls:
                final_response = response_message.content
                
                if verbose:
                    print(f"\n✅ Task complete!\n")
                    print(f"Response: {final_response}")
                
                return {
                    "success": True,
                    "response": final_response,
                    "iterations": iteration,
                    "function_calls": function_calls
                }
            
            # Execute function calls
            messages.append(response_message)
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if verbose:
                    print(f"\n🔧 Calling: {function_name}")
                    print(f"   Args: {json.dumps(function_args, indent=6)}")
                
                # Execute with safety manager
                result = self.execution_manager.execute_function(
                    function_name,
                    function_args
                )
                
                # Record call
                function_calls.append({
                    "name": function_name,
                    "arguments": function_args,
                    "success": result.success,
                    "result": result.result if result.success else result.error
                })
                
                if verbose:
                    if result.success:
                        print(f"   ✓ Success ({result.execution_time:.3f}s)")
                        print(f"   Result: {str(result.result)[:150]}...")
                    else:
                        print(f"   ❌ Error: {result.error}")
                
                # Add result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({
                        "success": result.success,
                        "result": result.result,
                        "error": result.error
                    })
                })
        
        # Max iterations reached
        return {
            "success": False,
            "error": "Maximum iterations reached",
            "iterations": iteration,
            "function_calls": function_calls
        }
    
    def chat(self, message: str) -> str:
        """
        Have a conversation with the agent.
        
        Args:
            message: User message
        
        Returns:
            Agent's response
        """
        self.conversation_history.append({"role": "user", "content": message})
        
        result = self.run(message, verbose=False)
        
        if result["success"]:
            response = result["response"]
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
        else:
            return f"Error: {result.get('error', 'Unknown error occurred')}"

# Create autonomous agent
agent = AutonomousAgent(model="gpt-4", max_iterations=10)

# Register tools
agent.register_tool(calculate_tip)
agent.register_tool(convert_currency)
agent.register_tool(calculate_distance)
agent.register_tool(search_database)

# Test complex multi-step task
result = agent.run("""
I'm planning a trip from New York City to San Francisco.
Can you:
1. Calculate the distance between them (NYC: 40.7128°N, 74.0060°W, SF: 37.7749°N, 122.4194°W)
2. If I have 500 USD, convert it to EUR
3. Then search the database for 'flight deals to San Francisco'
""", verbose=True)

print(f"\n{'='*80}")
print("EXECUTION SUMMARY")
print(f"{'='*80}")
print(f"Success: {result['success']}")
print(f"Iterations: {result['iterations']}")
print(f"Functions called: {len(result['function_calls'])}")

for i, call in enumerate(result['function_calls'], 1):
    print(f"\n{i}. {call['name']}")
    print(f"   Success: {call['success']}")
```

---

## Part 6: Error Handling & Safety

### Best Practices for Safe Function Calling

```python
class SafeFunctionCallingSystem:
    """
    Production-ready function calling with comprehensive safety.
    """
    
    def __init__(self):
        self.agent = AutonomousAgent()
        self.sandboxed_functions = set()
        self.rate_limiters = {}
        
    def sandbox_function(self, func: callable):
        """Mark a function as requiring sandboxed execution."""
        self.sandboxed_functions.add(func.__name__)
    
    def add_rate_limit(self, function_name: str, calls_per_minute: int):
        """Add rate limiting to a function."""
        self.rate_limiters[function_name] = {
            "limit": calls_per_minute,
            "calls": []
        }
    
    def validate_function_call(
        self,
        function_name: str,
        arguments: dict
    ) -> tuple[bool, Optional[str]]:
        """
        Validate a function call before execution.
        
        Returns:
            (is_valid, error_message)
        """
        # Check rate limits
        if function_name in self.rate_limiters:
            limiter = self.rate_limiters[function_name]
            current_time = time.time()
            
            # Remove old calls
            limiter["calls"] = [
                t for t in limiter["calls"]
                if current_time - t < 60
            ]
            
            if len(limiter["calls"]) >= limiter["limit"]:
                return False, f"Rate limit exceeded for {function_name}"
            
            limiter["calls"].append(current_time)
        
        # Validate arguments (type checking, sanitization)
        # ... add more validation logic ...
        
        return True, None

# Security guidelines
SECURITY_GUIDELINES = """
FUNCTION CALLING SECURITY BEST PRACTICES:

1. INPUT VALIDATION
   - Always validate function arguments
   - Sanitize user inputs
   - Check argument types and ranges
   
2. PERMISSION CONTROL
   - Whitelist allowed functions
   - Require confirmation for dangerous operations
   - Implement role-based access control

3. RATE LIMITING
   - Limit function call frequency
   - Implement quotas per user/session
   - Track and alert on unusual patterns

4. SANDBOXING
   - Run untrusted functions in isolation
   - Limit resource access (file system, network)
   - Set execution timeouts

5. MONITORING & LOGGING
   - Log all function calls
   - Track execution time and results
   - Alert on errors or security events

6. ERROR HANDLING
   - Never expose internal errors to users
   - Provide safe fallback responses
   - Implement circuit breakers

7. DATA PRIVACY
   - Don't log sensitive data
   - Encrypt function results if needed
   - Comply with data retention policies
"""

print(SECURITY_GUIDELINES)
```

---

## Best Practices

### 1. Function Design

**Good Function:**
```python
def get_user_info(user_id: str, include_private: bool = False) -> dict:
    """
    Get user information by ID.
    
    Clear description, typed parameters, sensible defaults.
    
    Args:
        user_id: Unique user identifier
        include_private: Whether to include private information
    
    Returns:
        User information dictionary
    """
    # Validation
    if not user_id:
        raise ValueError("user_id cannot be empty")
    
    # Implementation...
    return {"id": user_id, "name": "John Doe"}
```

**Bad Function:**
```python
def get_data(x, y=None):  # Vague name, untyped, unclear purpose
    return x if y else None
```

### 2. Schema Design

**Good Schema:**
```json
{
  "name": "search_products",
  "description": "Search product catalog with filters",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query for product name or description"
      },
      "max_price": {
        "type": "number",
        "description": "Maximum price filter (USD)"
      },
      "category": {
        "type": "string",
        "enum": ["electronics", "clothing", "food"],
        "description": "Product category"
      }
    },
    "required": ["query"]
  }
}
```

### 3. Error Handling

```python
def robust_function_call(function_name: str, arguments: dict) -> dict:
    """Execute function with comprehensive error handling."""
    try:
        # Validate
        if not validate_arguments(function_name, arguments):
            return {"error": "Invalid arguments"}
        
        # Execute with timeout
        result = execute_with_timeout(function_name, arguments, timeout=30)
        
        # Validate result
        if not validate_result(result):
            return {"error": "Invalid result format"}
        
        return {"success": True, "result": result}
        
    except TimeoutError:
        return {"error": "Function execution timed out"}
    except ValueError as e:
        return {"error": f"Validation error: {e}"}
    except Exception as e:
        logging.error(f"Function call failed: {e}")
        return {"error": "Internal error occurred"}
```

### 4. Testing Functions

```python
def test_function_calling():
    """Test function calling end-to-end."""
    
    # Test cases
    test_cases = [
        {
            "input": "Calculate 15% tip on $80",
            "expected_function": "calculate_tip",
            "expected_args": {"bill_amount": 80, "tip_percentage": 15}
        },
        # More test cases...
    ]
    
    for test in test_cases:
        result = agent.run(test["input"], verbose=False)
        
        assert result["success"], f"Failed: {test['input']}"
        assert any(
            call["name"] == test["expected_function"]
            for call in result["function_calls"]
        ), f"Expected function {test['expected_function']} not called"
```

### 5. Performance Optimization

- **Batch function calls** when possible
- **Cache function results** for identical inputs
- **Parallelize** independent function calls
- **Set appropriate timeouts**
- **Monitor execution time** and optimize slow functions

### 6. Cost Optimization

- Use **gpt-3.5-turbo** for function calling (cheaper, good performance)
- **Minimize function descriptions** to reduce tokens
- **Combine related functions** to reduce schema size
- **Cache repeated calls**
- **Monitor token usage** per function call

---

**End of Lab 3 Solutions**
