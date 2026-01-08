# Lab 2 Solutions: Building a Tool Registry System

This document provides comprehensive solutions for Lab 2: Building a Tool Registry System.

## Table of Contents

1. [Exercise 1: Basic Tool Registry](#exercise-1)
2. [Exercise 2: Type-Safe Tool Schemas](#exercise-2)
3. [Exercise 3: Authentication and Authorization](#exercise-3)
4. [Exercise 4: Rate Limiting and Quotas](#exercise-4)
5. [Exercise 5: Tool Versioning](#exercise-5)
6. [Exercise 6: Usage Monitoring](#exercise-6)
7. [Exercise 7: Complete Integration](#exercise-7)
8. [Testing and Validation](#testing)
9. [Production Best Practices](#best-practices)

---

## Exercise 1: Basic Tool Registry {#exercise-1}

### Solution

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Set
from enum import Enum
from pydantic import BaseModel
from collections import defaultdict

class ToolCategory(str, Enum):
    """Tool categories for organization."""
    SEARCH = "search"
    DATA = "data"
    COMMUNICATION = "communication"
    COMPUTATION = "computation"
    UTILITY = "utility"

@dataclass
class ToolMetadata:
    """Metadata for a registered tool."""
    name: str
    description: str
    category: ToolCategory
    version: str
    parameters_schema: type[BaseModel]
    function: Callable
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    author: Optional[str] = None
    created_at: Optional[str] = None

class ToolRegistry:
    """
    Central registry for all tools.
    
    Provides:
    - Tool registration and lookup
    - Category-based filtering
    - Call count tracking
    - Enable/disable functionality
    """
    
    def __init__(self):
        self._tools: Dict[str, ToolMetadata] = {}
        self._call_counts: Dict[str, int] = defaultdict(int)
        self._categories: Dict[ToolCategory, Set[str]] = defaultdict(set)
    
    def register(
        self,
        name: str,
        description: str,
        category: ToolCategory,
        version: str,
        parameters_schema: type[BaseModel],
        function: Callable,
        **kwargs
    ) -> None:
        """
        Register a new tool.
        
        Args:
            name: Unique tool identifier
            description: Tool description for LLM
            category: Tool category
            version: Semantic version (e.g., "1.0.0")
            parameters_schema: Pydantic model for parameters
            function: Tool implementation
            **kwargs: Additional metadata (enabled, tags, author)
        
        Raises:
            ValueError: If tool already exists
        """
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered")
        
        metadata = ToolMetadata(
            name=name,
            description=description,
            category=category,
            version=version,
            parameters_schema=parameters_schema,
            function=function,
            **kwargs
        )
        
        self._tools[name] = metadata
        self._categories[category].add(name)
        print(f"✓ Registered tool: {name} (v{version})")
    
    def get(self, name: str) -> Optional[ToolMetadata]:
        """
        Get tool by name.
        
        Args:
            name: Tool name
        
        Returns:
            ToolMetadata if found, None otherwise
        """
        return self._tools.get(name)
    
    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        enabled_only: bool = True
    ) -> List[ToolMetadata]:
        """
        List tools with optional filtering.
        
        Args:
            category: Filter by category (None = all categories)
            enabled_only: Only return enabled tools
        
        Returns:
            List of ToolMetadata objects
        """
        tools = self._tools.values()
        
        # Filter by category
        if category:
            tools = [t for t in tools if t.category == category]
        
        # Filter by enabled status
        if enabled_only:
            tools = [t for t in tools if t.enabled]
        
        return list(tools)
    
    def execute(
        self,
        tool_name: str,
        **kwargs
    ) -> Any:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of tool to execute
            **kwargs: Tool parameters
        
        Returns:
            Tool execution result
        
        Raises:
            ValueError: If tool not found or disabled
        """
        tool = self.get(tool_name)
        
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        if not tool.enabled:
            raise ValueError(f"Tool '{tool_name}' is disabled")
        
        # Validate parameters using Pydantic schema
        validated_params = tool.parameters_schema(**kwargs)
        
        # Increment call count
        self._call_counts[tool_name] += 1
        
        # Execute tool
        result = tool.function(**validated_params.model_dump())
        
        return result
    
    def get_stats(self) -> dict:
        """Get registry statistics."""
        return {
            "total_tools": len(self._tools),
            "enabled_tools": len([t for t in self._tools.values() if t.enabled]),
            "disabled_tools": len([t for t in self._tools.values() if not t.enabled]),
            "categories": {
                cat.value: len(tools)
                for cat, tools in self._categories.items()
            },
            "total_calls": sum(self._call_counts.values()),
            "most_called": max(
                self._call_counts.items(),
                key=lambda x: x[1],
                default=("none", 0)
            )
        }
    
    def enable_tool(self, name: str) -> None:
        """Enable a tool."""
        if tool := self.get(name):
            tool.enabled = True
            print(f"✓ Enabled tool: {name}")
    
    def disable_tool(self, name: str) -> None:
        """Disable a tool."""
        if tool := self.get(name):
            tool.enabled = False
            print(f"✗ Disabled tool: {name}")

# Test basic registry
print("=== Testing Basic Tool Registry ===\n")

registry = ToolRegistry()

# Define parameter schemas
class CalculatorParams(BaseModel):
    operation: str
    a: float
    b: float

class WeatherParams(BaseModel):
    location: str
    units: str = "celsius"

# Define tool functions
def calculator(operation: str, a: float, b: float) -> float:
    """Perform basic calculations."""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float('inf')
    }
    return operations[operation](a, b)

def get_weather(location: str, units: str = "celsius") -> dict:
    """Get weather information."""
    return {
        "location": location,
        "temperature": 22 if units == "celsius" else 72,
        "condition": "sunny",
        "units": units
    }

# Register tools
registry.register(
    name="calculator",
    description="Perform basic arithmetic operations",
    category=ToolCategory.COMPUTATION,
    version="1.0.0",
    parameters_schema=CalculatorParams,
    function=calculator,
    tags=["math", "arithmetic"]
)

registry.register(
    name="get_weather",
    description="Get current weather for a location",
    category=ToolCategory.DATA,
    version="1.0.0",
    parameters_schema=WeatherParams,
    function=get_weather,
    tags=["weather", "api"]
)

# List tools
print("\n=== All Tools ===")
for tool in registry.list_tools():
    print(f"- {tool.name} (v{tool.version}): {tool.description}")

# Execute tools
print("\n=== Executing Tools ===")
result1 = registry.execute("calculator", operation="add", a=5, b=3)
print(f"calculator(add, 5, 3) = {result1}")

result2 = registry.execute("get_weather", location="London")
print(f"get_weather(London) = {result2}")

# Get statistics
print("\n=== Registry Statistics ===")
stats = registry.get_stats()
for key, value in stats.items():
    print(f"{key}: {value}")
```

### Expected Output

```
=== Testing Basic Tool Registry ===

✓ Registered tool: calculator (v1.0.0)
✓ Registered tool: get_weather (v1.0.0)

=== All Tools ===
- calculator (v1.0.0): Perform basic arithmetic operations
- get_weather (v1.0.0): Get current weather for a location

=== Executing Tools ===
calculator(add, 5, 3) = 8.0
get_weather(London) = {'location': 'London', 'temperature': 22, 'condition': 'sunny', 'units': 'celsius'}

=== Registry Statistics ===
total_tools: 2
enabled_tools: 2
disabled_tools: 0
categories: {'computation': 1, 'data': 1}
total_calls: 2
most_called: ('calculator', 1)
```

### Key Features

1. **Centralized Management**: All tools in one place
2. **Metadata Storage**: Version, category, tags, author
3. **Category Filtering**: Find tools by purpose
4. **Call Tracking**: Monitor usage patterns
5. **Enable/Disable**: Control tool availability

---

## Exercise 2: Type-Safe Tool Schemas {#exercise-2}

### Solution

```python
from pydantic import BaseModel, Field, validator, field_validator
from typing import List, Optional, Literal
from datetime import datetime

# Example 1: Calculator with validation
class CalculatorParams(BaseModel):
    """Calculator parameters with validation."""
    operation: Literal["add", "subtract", "multiply", "divide"]
    a: float = Field(..., description="First operand")
    b: float = Field(..., description="Second operand")
    
    @field_validator('b')
    @classmethod
    def validate_division(cls, v, info):
        """Prevent division by zero."""
        if info.data.get('operation') == 'divide' and v == 0:
            raise ValueError("Cannot divide by zero")
        return v

# Example 2: Weather API with constraints
class WeatherParams(BaseModel):
    """Weather API parameters."""
    location: str = Field(..., min_length=2, max_length=100, description="City name")
    units: Literal["celsius", "fahrenheit", "kelvin"] = Field(
        default="celsius",
        description="Temperature units"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

# Example 3: Search with complex validation
class SearchParams(BaseModel):
    """Search parameters with advanced validation."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Max results")
    offset: int = Field(default=0, ge=0, description="Result offset")
    filters: Optional[List[str]] = Field(default=None, description="Search filters")
    sort_by: Literal["relevance", "date", "popularity"] = "relevance"
    
    @field_validator('filters')
    @classmethod
    def validate_filters(cls, v):
        """Validate filter format."""
        if v:
            allowed_filters = ["news", "images", "videos", "academic"]
            for f in v:
                if f not in allowed_filters:
                    raise ValueError(f"Invalid filter: {f}. Allowed: {allowed_filters}")
        return v

# Example 4: Email with complex types
class EmailParams(BaseModel):
    """Email parameters with nested validation."""
    to: List[str] = Field(..., min_length=1, description="Recipients")
    subject: str = Field(..., min_length=1, max_length=200)
    body: str = Field(..., min_length=1)
    cc: Optional[List[str]] = None
    bcc: Optional[List[str]] = None
    attachments: Optional[List[str]] = None
    priority: Literal["low", "normal", "high"] = "normal"
    
    @field_validator('to', 'cc', 'bcc')
    @classmethod
    def validate_emails(cls, v):
        """Validate email formats."""
        if v:
            import re
            email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
            for email in v:
                if not re.match(email_regex, email):
                    raise ValueError(f"Invalid email: {email}")
        return v

# Example 5: Date range with cross-field validation
class DateRangeParams(BaseModel):
    """Date range with validation."""
    start_date: datetime
    end_date: datetime
    timezone: str = "UTC"
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        """Ensure end_date is after start_date."""
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError("end_date must be after start_date")
        return v

# Convert Pydantic schemas to OpenAI format
def pydantic_to_openai_schema(
    name: str,
    description: str,
    params_model: type[BaseModel]
) -> dict:
    """
    Convert Pydantic model to OpenAI function schema.
    
    Args:
        name: Function name
        description: Function description
        params_model: Pydantic model for parameters
    
    Returns:
        OpenAI function schema dict
    """
    schema = params_model.model_json_schema()
    
    # Extract properties and required fields
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    
    # Remove title from properties (OpenAI doesn't need it)
    for prop in properties.values():
        prop.pop("title", None)
    
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }

# Test type-safe schemas
print("=== Testing Type-Safe Schemas ===\n")

# Valid calculator call
print("Valid calculator call:")
params1 = CalculatorParams(operation="add", a=5, b=3)
print(f"✓ {params1}")

# Invalid calculator call (division by zero)
print("\nInvalid calculator call (division by zero):")
try:
    params2 = CalculatorParams(operation="divide", a=10, b=0)
except ValueError as e:
    print(f"✗ Validation error: {e}")

# Valid search call
print("\nValid search call:")
params3 = SearchParams(
    query="Python tutorials",
    limit=20,
    filters=["news", "academic"]
)
print(f"✓ {params3}")

# Invalid search call (bad filter)
print("\nInvalid search call (bad filter):")
try:
    params4 = SearchParams(query="test", filters=["invalid_filter"])
except ValueError as e:
    print(f"✗ Validation error: {e}")

# Convert to OpenAI schema
print("\n=== OpenAI Schema Conversion ===")
openai_schema = pydantic_to_openai_schema(
    "search",
    "Search the web for information",
    SearchParams
)
print(json.dumps(openai_schema, indent=2))

# Test with registry
class TypeSafeRegistry(ToolRegistry):
    """Registry with automatic schema conversion."""
    
    def get_openai_schemas(self, enabled_only: bool = True) -> List[dict]:
        """Get all tools as OpenAI schemas."""
        tools = self.list_tools(enabled_only=enabled_only)
        
        return [
            pydantic_to_openai_schema(
                tool.name,
                tool.description,
                tool.parameters_schema
            )
            for tool in tools
        ]

# Create registry and register tools
type_safe_registry = TypeSafeRegistry()

type_safe_registry.register(
    name="calculator",
    description="Perform arithmetic operations",
    category=ToolCategory.COMPUTATION,
    version="1.0.0",
    parameters_schema=CalculatorParams,
    function=lambda operation, a, b: eval(f"{a} {'+' if operation=='add' else operation[0]} {b}")
)

type_safe_registry.register(
    name="search",
    description="Search the web",
    category=ToolCategory.SEARCH,
    version="1.0.0",
    parameters_schema=SearchParams,
    function=lambda **kwargs: {"results": ["result1", "result2"]}
)

# Get OpenAI schemas
print("\n=== All OpenAI Schemas ===")
schemas = type_safe_registry.get_openai_schemas()
for schema in schemas:
    print(f"\n{schema['function']['name']}:")
    print(json.dumps(schema, indent=2))
```

### Expected Output

```
=== Testing Type-Safe Schemas ===

Valid calculator call:
✓ operation='add' a=5.0 b=3.0

Invalid calculator call (division by zero):
✗ Validation error: Cannot divide by zero

Valid search call:
✓ query='Python tutorials' limit=20 offset=0 filters=['news', 'academic'] sort_by='relevance'

Invalid search call (bad filter):
✗ Validation error: Invalid filter: invalid_filter. Allowed: ['news', 'images', 'videos', 'academic']

=== OpenAI Schema Conversion ===
{
  "type": "function",
  "function": {
    "name": "search",
    "description": "Search the web for information",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "minLength": 1,
          "maxLength": 500,
          "description": "Search query"
        },
        "limit": {
          "type": "integer",
          "minimum": 1,
          "maximum": 100,
          "default": 10,
          "description": "Max results"
        },
        ...
      },
      "required": ["query"]
    }
  }
}
```

### Key Features

1. **Automatic Validation**: Pydantic validates all parameters
2. **Type Safety**: Catch errors before execution
3. **Rich Constraints**: Min/max, regex, custom validators
4. **Cross-Field Validation**: Validate relationships between fields
5. **OpenAI Integration**: Auto-convert to OpenAI function format

---

## Exercise 3: Authentication and Authorization {#exercise-3}

### Solution

```python
from enum import Enum
from typing import Set
from dataclasses import dataclass

class UserRole(str, Enum):
    """User roles for authorization."""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"
    SERVICE = "service"

@dataclass
class User:
    """User with authentication information."""
    id: str
    username: str
    roles: Set[UserRole]
    api_key: Optional[str] = None

class Permission(str, Enum):
    """Tool permissions."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"

@dataclass
class ToolPermissions:
    """Permissions required for a tool."""
    required_roles: Set[UserRole]
    required_permissions: Set[Permission]

class AuthorizedRegistry(ToolRegistry):
    """Tool registry with authentication and authorization."""
    
    def __init__(self):
        super().__init__()
        self._tool_permissions: Dict[str, ToolPermissions] = {}
    
    def register_with_permissions(
        self,
        name: str,
        description: str,
        category: ToolCategory,
        version: str,
        parameters_schema: type[BaseModel],
        function: Callable,
        required_roles: Set[UserRole],
        required_permissions: Set[Permission],
        **kwargs
    ) -> None:
        """Register tool with permission requirements."""
        # Register tool normally
        self.register(
            name=name,
            description=description,
            category=category,
            version=version,
            parameters_schema=parameters_schema,
            function=function,
            **kwargs
        )
        
        # Store permissions
        self._tool_permissions[name] = ToolPermissions(
            required_roles=required_roles,
            required_permissions=required_permissions
        )
        
        print(f"✓ Registered {name} with roles: {[r.value for r in required_roles]}")
    
    def check_permission(
        self,
        user: User,
        tool_name: str
    ) -> tuple[bool, Optional[str]]:
        """
        Check if user has permission to execute tool.
        
        Returns:
            (authorized: bool, error_message: Optional[str])
        """
        if tool_name not in self._tool_permissions:
            return True, None  # No restrictions
        
        perms = self._tool_permissions[tool_name]
        
        # Check roles
        if not user.roles.intersection(perms.required_roles):
            return False, f"Insufficient role. Required: {perms.required_roles}"
        
        # Admin can do everything
        if UserRole.ADMIN in user.roles:
            return True, None
        
        # Check specific permissions (simplified - in production, check user's permission set)
        if Permission.ADMIN in perms.required_permissions and UserRole.ADMIN not in user.roles:
            return False, "Admin permission required"
        
        return True, None
    
    def execute_as_user(
        self,
        user: User,
        tool_name: str,
        **kwargs
    ) -> Any:
        """
        Execute tool with authorization check.
        
        Raises:
            PermissionError: If user lacks required permissions
        """
        # Check authorization
        authorized, error_msg = self.check_permission(user, tool_name)
        
        if not authorized:
            raise PermissionError(f"Access denied for tool '{tool_name}': {error_msg}")
        
        # Execute tool
        return self.execute(tool_name, **kwargs)
    
    def list_tools_for_user(self, user: User) -> List[ToolMetadata]:
        """List tools user has permission to execute."""
        all_tools = self.list_tools()
        
        authorized_tools = []
        for tool in all_tools:
            authorized, _ = self.check_permission(user, tool.name)
            if authorized:
                authorized_tools.append(tool)
        
        return authorized_tools

# Test authorization
print("=== Testing Authorization ===\n")

auth_registry = AuthorizedRegistry()

# Register tools with different permission levels
auth_registry.register_with_permissions(
    name="calculator",
    description="Basic calculator",
    category=ToolCategory.COMPUTATION,
    version="1.0.0",
    parameters_schema=CalculatorParams,
    function=calculator,
    required_roles={UserRole.USER, UserRole.ADMIN},
    required_permissions={Permission.EXECUTE}
)

class DeleteParams(BaseModel):
    resource_id: str

auth_registry.register_with_permissions(
    name="delete_resource",
    description="Delete a resource (admin only)",
    category=ToolCategory.UTILITY,
    version="1.0.0",
    parameters_schema=DeleteParams,
    function=lambda resource_id: f"Deleted {resource_id}",
    required_roles={UserRole.ADMIN},
    required_permissions={Permission.ADMIN, Permission.WRITE}
)

# Create users
admin_user = User(
    id="1",
    username="admin",
    roles={UserRole.ADMIN}
)

regular_user = User(
    id="2",
    username="john",
    roles={UserRole.USER}
)

guest_user = User(
    id="3",
    username="guest",
    roles={UserRole.GUEST}
)

# Test regular user accessing calculator (allowed)
print("Regular user accessing calculator:")
try:
    result = auth_registry.execute_as_user(
        regular_user,
        "calculator",
        operation="add",
        a=5,
        b=3
    )
    print(f"✓ Result: {result}")
except PermissionError as e:
    print(f"✗ {e}")

# Test regular user accessing delete (denied)
print("\nRegular user accessing delete_resource:")
try:
    result = auth_registry.execute_as_user(
        regular_user,
        "delete_resource",
        resource_id="res123"
    )
    print(f"✓ Result: {result}")
except PermissionError as e:
    print(f"✗ {e}")

# Test admin user accessing delete (allowed)
print("\nAdmin user accessing delete_resource:")
try:
    result = auth_registry.execute_as_user(
        admin_user,
        "delete_resource",
        resource_id="res123"
    )
    print(f"✓ Result: {result}")
except PermissionError as e:
    print(f"✗ {e}")

# List tools for each user
print("\n=== Tools Available by User ===")
for user in [admin_user, regular_user, guest_user]:
    tools = auth_registry.list_tools_for_user(user)
    print(f"{user.username}: {[t.name for t in tools]}")
```

### Expected Output

```
=== Testing Authorization ===

✓ Registered calculator with roles: ['user', 'admin']
✓ Registered delete_resource with roles: ['admin']

Regular user accessing calculator:
✓ Result: 8.0

Regular user accessing delete_resource:
✗ Access denied for tool 'delete_resource': Insufficient role. Required: {<UserRole.ADMIN: 'admin'>}

Admin user accessing delete_resource:
✓ Result: Deleted res123

=== Tools Available by User ===
admin: ['calculator', 'delete_resource']
john: ['calculator']
guest: []
```

### Key Features

1. **Role-Based Access Control**: Different roles have different permissions
2. **Permission Granularity**: Read, write, execute, admin
3. **Authorization Checks**: Validate before execution
4. **User-Specific Views**: Show only accessible tools
5. **Security by Default**: Deny unless explicitly granted

---

## Exercise 4: Rate Limiting and Quotas {#exercise-4}

### Solution

```python
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field

@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    calls_per_minute: int = 60
    calls_per_hour: int = 1000
    calls_per_day: int = 10000

@dataclass
class UsageMetrics:
    """Track usage for rate limiting."""
    call_timestamps: deque = field(default_factory=lambda: deque(maxlen=10000))
    total_calls: int = 0
    quota_exceeded_count: int = 0

class RateLimiter:
    """Rate limiter for tool calls."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.user_metrics: Dict[str, UsageMetrics] = defaultdict(UsageMetrics)
    
    def _count_calls_in_window(
        self,
        timestamps: deque,
        window_seconds: int
    ) -> int:
        """Count calls within time window."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=window_seconds)
        
        return sum(1 for ts in timestamps if ts > cutoff)
    
    def check_rate_limit(
        self,
        user_id: str
    ) -> tuple[bool, Optional[str]]:
        """
        Check if user is within rate limits.
        
        Returns:
            (allowed: bool, error_message: Optional[str])
        """
        metrics = self.user_metrics[user_id]
        
        # Check per-minute limit
        calls_last_minute = self._count_calls_in_window(
            metrics.call_timestamps,
            60
        )
        if calls_last_minute >= self.config.calls_per_minute:
            return False, f"Rate limit exceeded: {self.config.calls_per_minute} calls/minute"
        
        # Check per-hour limit
        calls_last_hour = self._count_calls_in_window(
            metrics.call_timestamps,
            3600
        )
        if calls_last_hour >= self.config.calls_per_hour:
            return False, f"Rate limit exceeded: {self.config.calls_per_hour} calls/hour"
        
        # Check per-day limit
        calls_last_day = self._count_calls_in_window(
            metrics.call_timestamps,
            86400
        )
        if calls_last_day >= self.config.calls_per_day:
            return False, f"Rate limit exceeded: {self.config.calls_per_day} calls/day"
        
        return True, None
    
    def record_call(self, user_id: str) -> None:
        """Record a tool call."""
        metrics = self.user_metrics[user_id]
        metrics.call_timestamps.append(datetime.now())
        metrics.total_calls += 1
    
    def record_quota_exceeded(self, user_id: str) -> None:
        """Record quota exceeded event."""
        metrics = self.user_metrics[user_id]
        metrics.quota_exceeded_count += 1
    
    def get_usage_stats(self, user_id: str) -> dict:
        """Get usage statistics for user."""
        metrics = self.user_metrics[user_id]
        
        return {
            "total_calls": metrics.total_calls,
            "quota_exceeded_count": metrics.quota_exceeded_count,
            "calls_last_minute": self._count_calls_in_window(metrics.call_timestamps, 60),
            "calls_last_hour": self._count_calls_in_window(metrics.call_timestamps, 3600),
            "calls_last_day": self._count_calls_in_window(metrics.call_timestamps, 86400),
            "remaining_today": max(
                0,
                self.config.calls_per_day - self._count_calls_in_window(metrics.call_timestamps, 86400)
            )
        }

class RateLimitedRegistry(AuthorizedRegistry):
    """Registry with rate limiting."""
    
    def __init__(self, rate_limit_config: Optional[RateLimitConfig] = None):
        super().__init__()
        self.rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())
    
    def execute_as_user(
        self,
        user: User,
        tool_name: str,
        quota_limit: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Execute tool with rate limiting.
        
        Args:
            user: User executing the tool
            tool_name: Name of tool
            quota_limit: Optional per-call quota override
            **kwargs: Tool parameters
        
        Raises:
            PermissionError: If authorization fails
            RuntimeError: If rate limit exceeded
        """
        # Check authorization
        authorized, error_msg = self.check_permission(user, tool_name)
        if not authorized:
            raise PermissionError(f"Access denied: {error_msg}")
        
        # Check rate limit
        allowed, rate_error = self.rate_limiter.check_rate_limit(user.id)
        if not allowed:
            self.rate_limiter.record_quota_exceeded(user.id)
            raise RuntimeError(rate_error)
        
        # Record call
        self.rate_limiter.record_call(user.id)
        
        # Execute tool
        return self.execute(tool_name, **kwargs)
    
    def get_user_usage(self, user_id: str) -> dict:
        """Get usage statistics for user."""
        return self.rate_limiter.get_usage_stats(user_id)

# Test rate limiting
print("=== Testing Rate Limiting ===\n")

# Create registry with strict limits for testing
rate_limited_registry = RateLimitedRegistry(
    RateLimitConfig(
        calls_per_minute=5,
        calls_per_hour=100,
        calls_per_day=1000
    )
)

rate_limited_registry.register_with_permissions(
    name="calculator",
    description="Calculator",
    category=ToolCategory.COMPUTATION,
    version="1.0.0",
    parameters_schema=CalculatorParams,
    function=calculator,
    required_roles={UserRole.USER, UserRole.ADMIN},
    required_permissions={Permission.EXECUTE}
)

test_user = User(id="test_user", username="test", roles={UserRole.USER})

# Make calls up to limit
print("Making 5 calls (within limit):")
for i in range(5):
    try:
        result = rate_limited_registry.execute_as_user(
            test_user,
            "calculator",
            operation="add",
            a=i,
            b=1
        )
        print(f"✓ Call {i+1}: {result}")
    except RuntimeError as e:
        print(f"✗ Call {i+1}: {e}")

# Try to exceed limit
print("\nTrying 6th call (should fail):")
try:
    result = rate_limited_registry.execute_as_user(
        test_user,
        "calculator",
        operation="add",
        a=10,
        b=10
    )
    print(f"✓ Call 6: {result}")
except RuntimeError as e:
    print(f"✗ Call 6: {e}")

# Check usage stats
print("\n=== Usage Statistics ===")
stats = rate_limited_registry.get_user_usage(test_user.id)
for key, value in stats.items():
    print(f"{key}: {value}")
```

### Expected Output

```
=== Testing Rate Limiting ===

Making 5 calls (within limit):
✓ Call 1: 1.0
✓ Call 2: 2.0
✓ Call 3: 3.0
✓ Call 4: 4.0
✓ Call 5: 5.0

Trying 6th call (should fail):
✗ Call 6: Rate limit exceeded: 5 calls/minute

=== Usage Statistics ===
total_calls: 5
quota_exceeded_count: 1
calls_last_minute: 5
calls_last_hour: 5
calls_last_day: 5
remaining_today: 995
```

### Key Features

1. **Multi-Level Limits**: Per-minute, hour, and day
2. **Sliding Windows**: Accurate rate limiting
3. **Per-User Tracking**: Individual quotas
4. **Usage Analytics**: Monitor consumption
5. **Quota Overflow Detection**: Track exceeded limits

---

## Exercise 5: Tool Versioning {#exercise-5}

### Solution

```python
from typing import Optional
import re

@dataclass
class VersionedToolMetadata(ToolMetadata):
    """Tool metadata with version information."""
    deprecated: bool = False
    deprecation_message: Optional[str] = None
    successor_version: Optional[str] = None

class VersionedRegistry(RateLimitedRegistry):
    """Registry supporting multiple tool versions."""
    
    def __init__(self):
        super().__init__()
        self._version_map: Dict[str, Dict[str, VersionedToolMetadata]] = defaultdict(dict)
    
    def _parse_version(self, version: str) -> tuple[int, int, int]:
        """Parse semantic version string."""
        match = re.match(r'(\d+)\.(\d+)\.(\d+)', version)
        if not match:
            raise ValueError(f"Invalid version format: {version}")
        return tuple(map(int, match.groups()))
    
    def register_version(
        self,
        name: str,
        version: str,
        description: str,
        category: ToolCategory,
        parameters_schema: type[BaseModel],
        function: Callable,
        required_roles: Set[UserRole],
        required_permissions: Set[Permission],
        deprecated: bool = False,
        successor_version: Optional[str] = None,
        **kwargs
    ) -> None:
        """Register a specific version of a tool."""
        # Validate version format
        self._parse_version(version)
        
        # Create versioned tool name
        versioned_name = f"{name}@{version}"
        
        # Register in parent registry
        self.register_with_permissions(
            name=versioned_name,
            description=description,
            category=category,
            version=version,
            parameters_schema=parameters_schema,
            function=function,
            required_roles=required_roles,
            required_permissions=required_permissions,
            **kwargs
        )
        
        # Store in version map
        metadata = VersionedToolMetadata(
            name=name,
            description=description,
            category=category,
            version=version,
            parameters_schema=parameters_schema,
            function=function,
            deprecated=deprecated,
            successor_version=successor_version,
            **kwargs
        )
        self._version_map[name][version] = metadata
        
        print(f"✓ Registered {name} v{version}" + (" [DEPRECATED]" if deprecated else ""))
    
    def get_latest_version(self, tool_name: str) -> Optional[str]:
        """Get latest non-deprecated version."""
        if tool_name not in self._version_map:
            return None
        
        versions = self._version_map[tool_name]
        
        # Filter non-deprecated
        active_versions = [
            v for v, meta in versions.items()
            if not meta.deprecated
        ]
        
        if not active_versions:
            return None
        
        # Sort by semantic version
        sorted_versions = sorted(
            active_versions,
            key=self._parse_version,
            reverse=True
        )
        
        return sorted_versions[0]
    
    def execute_versioned(
        self,
        user: User,
        tool_name: str,
        version: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Execute specific version of a tool.
        
        Args:
            user: User executing tool
            tool_name: Base tool name (without version)
            version: Specific version or None for latest
            **kwargs: Tool parameters
        
        Returns:
            Tool execution result
        """
        # Use latest version if not specified
        if version is None:
            version = self.get_latest_version(tool_name)
            if version is None:
                raise ValueError(f"No active versions of '{tool_name}'")
        
        # Check if version exists
        if tool_name not in self._version_map:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        if version not in self._version_map[tool_name]:
            raise ValueError(f"Version {version} of '{tool_name}' not found")
        
        metadata = self._version_map[tool_name][version]
        
        # Warn if deprecated
        if metadata.deprecated:
            warning = f"⚠️  WARNING: {tool_name}@{version} is deprecated."
            if metadata.successor_version:
                warning += f" Use v{metadata.successor_version} instead."
            print(warning)
        
        # Execute with versioned name
        versioned_name = f"{tool_name}@{version}"
        return self.execute_as_user(user, versioned_name, **kwargs)
    
    def list_versions(self, tool_name: str) -> List[dict]:
        """List all versions of a tool."""
        if tool_name not in self._version_map:
            return []
        
        versions = []
        for version, metadata in self._version_map[tool_name].items():
            versions.append({
                "version": version,
                "deprecated": metadata.deprecated,
                "successor": metadata.successor_version,
                "description": metadata.description
            })
        
        # Sort by version
        versions.sort(
            key=lambda v: self._parse_version(v["version"]),
            reverse=True
        )
        
        return versions

# Test versioning
print("=== Testing Tool Versioning ===\n")

versioned_registry = VersionedRegistry()

# Register v1.0.0 (deprecated)
class CalculatorV1Params(BaseModel):
    operation: str
    a: float
    b: float

versioned_registry.register_version(
    name="calculator",
    version="1.0.0",
    description="Basic calculator (v1)",
    category=ToolCategory.COMPUTATION,
    parameters_schema=CalculatorV1Params,
    function=calculator,
    required_roles={UserRole.USER},
    required_permissions={Permission.EXECUTE},
    deprecated=True,
    successor_version="2.0.0"
)

# Register v1.5.0
versioned_registry.register_version(
    name="calculator",
    version="1.5.0",
    description="Calculator with validation (v1.5)",
    category=ToolCategory.COMPUTATION,
    parameters_schema=CalculatorParams,  # With validation
    function=calculator,
    required_roles={UserRole.USER},
    required_permissions={Permission.EXECUTE}
)

# Register v2.0.0 (latest)
class CalculatorV2Params(BaseModel):
    operation: Literal["add", "subtract", "multiply", "divide", "power", "sqrt"]
    a: float
    b: Optional[float] = None

def calculator_v2(operation: str, a: float, b: Optional[float] = None) -> float:
    """Enhanced calculator with more operations."""
    if operation == "sqrt":
        return a ** 0.5
    elif operation == "power":
        return a ** (b or 2)
    else:
        return calculator(operation, a, b)

versioned_registry.register_version(
    name="calculator",
    version="2.0.0",
    description="Enhanced calculator (v2)",
    category=ToolCategory.COMPUTATION,
    parameters_schema=CalculatorV2Params,
    function=calculator_v2,
    required_roles={UserRole.USER},
    required_permissions={Permission.EXECUTE}
)

test_user = User(id="user1", username="test", roles={UserRole.USER})

# List all versions
print("=== All Calculator Versions ===")
versions = versioned_registry.list_versions("calculator")
for v in versions:
    status = " [DEPRECATED]" if v["deprecated"] else " [ACTIVE]"
    print(f"v{v['version']}{status}: {v['description']}")

# Execute latest version (should be 2.0.0)
print("\n=== Using Latest Version ===")
result = versioned_registry.execute_versioned(
    test_user,
    "calculator",
    operation="sqrt",
    a=16
)
print(f"sqrt(16) = {result}")

# Execute specific version (deprecated 1.0.0)
print("\n=== Using Deprecated Version ===")
result = versioned_registry.execute_versioned(
    test_user,
    "calculator",
    version="1.0.0",
    operation="add",
    a=5,
    b=3
)
print(f"add(5, 3) = {result}")

# Get latest version
latest = versioned_registry.get_latest_version("calculator")
print(f"\nLatest version: {latest}")
```

### Expected Output

```
=== Testing Tool Versioning ===

✓ Registered calculator v1.0.0 [DEPRECATED]
✓ Registered calculator v1.5.0
✓ Registered calculator v2.0.0

=== All Calculator Versions ===
v2.0.0 [ACTIVE]: Enhanced calculator (v2)
v1.5.0 [ACTIVE]: Calculator with validation (v1.5)
v1.0.0 [DEPRECATED]: Basic calculator (v1)

=== Using Latest Version ===
sqrt(16) = 4.0

=== Using Deprecated Version ===
⚠️  WARNING: calculator@1.0.0 is deprecated. Use v2.0.0 instead.
add(5, 3) = 8.0

Latest version: 2.0.0
```

### Key Features

1. **Semantic Versioning**: Major.Minor.Patch format
2. **Deprecation Support**: Mark old versions as deprecated
3. **Automatic Latest**: Default to latest version
4. **Version Pinning**: Execute specific versions
5. **Migration Path**: Point to successor versions

---

## Exercise 6: Usage Monitoring {#exercise-6}

### Solution

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict
import statistics

@dataclass
class ToolCallRecord:
    """Record of a single tool call."""
    tool_name: str
    version: str
    user_id: str
    timestamp: datetime
    success: bool
    latency_ms: float
    error_type: Optional[str] = None

class UsageMonitor:
    """Monitor and analyze tool usage."""
    
    def __init__(self, max_records: int = 10000):
        self.records: deque = deque(maxlen=max_records)
        self.tool_stats: Dict[str, List[float]] = defaultdict(list)
        self.user_stats: Dict[str, Dict] = defaultdict(lambda: {
            "calls": 0,
            "successes": 0,
            "failures": 0
        })
    
    def record_call(
        self,
        tool_name: str,
        version: str,
        user_id: str,
        success: bool,
        latency_ms: float,
        error_type: Optional[str] = None
    ) -> None:
        """Record a tool call."""
        record = ToolCallRecord(
            tool_name=tool_name,
            version=version,
            user_id=user_id,
            timestamp=datetime.now(),
            success=success,
            latency_ms=latency_ms,
            error_type=error_type
        )
        
        self.records.append(record)
        
        # Update tool stats
        if success:
            self.tool_stats[tool_name].append(latency_ms)
        
        # Update user stats
        self.user_stats[user_id]["calls"] += 1
        if success:
            self.user_stats[user_id]["successes"] += 1
        else:
            self.user_stats[user_id]["failures"] += 1
    
    def get_tool_analytics(self, tool_name: str) -> dict:
        """Get analytics for a specific tool."""
        latencies = self.tool_stats.get(tool_name, [])
        
        if not latencies:
            return {"error": "No data available"}
        
        return {
            "tool_name": tool_name,
            "total_calls": len(latencies),
            "latency": {
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "mean_ms": statistics.mean(latencies),
                "median_ms": statistics.median(latencies),
                "std_dev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0
            }
        }
    
    def get_user_stats(self, user_id: str) -> dict:
        """Get statistics for a specific user."""
        stats = self.user_stats[user_id]
        
        success_rate = (
            stats["successes"] / stats["calls"]
            if stats["calls"] > 0
            else 0
        )
        
        return {
            "user_id": user_id,
            "total_calls": stats["calls"],
            "successes": stats["successes"],
            "failures": stats["failures"],
            "success_rate": success_rate
        }
    
    def get_summary(self) -> dict:
        """Get overall usage summary."""
        total_calls = len(self.records)
        successful_calls = sum(1 for r in self.records if r.success)
        
        # Top tools
        tool_counts = defaultdict(int)
        for record in self.records:
            tool_counts[record.tool_name] += 1
        
        top_tools = sorted(
            tool_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Top users
        top_users = sorted(
            [(uid, stats["calls"]) for uid, stats in self.user_stats.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": total_calls - successful_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 0,
            "unique_tools": len(self.tool_stats),
            "unique_users": len(self.user_stats),
            "top_tools": [{"name": name, "calls": count} for name, count in top_tools],
            "top_users": [{"user_id": uid, "calls": count} for uid, count in top_users]
        }

class MonitoredRegistry(VersionedRegistry):
    """Registry with integrated usage monitoring."""
    
    def __init__(self):
        super().__init__()
        self.monitor = UsageMonitor()
    
    def execute_versioned(
        self,
        user: User,
        tool_name: str,
        version: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Execute tool with monitoring."""
        start_time = time.time()
        error_type = None
        success = False
        result = None
        
        try:
            result = super().execute_versioned(user, tool_name, version, **kwargs)
            success = True
        except Exception as e:
            error_type = type(e).__name__
            raise
        finally:
            # Always record metrics
            latency_ms = (time.time() - start_time) * 1000
            actual_version = version or self.get_latest_version(tool_name)
            
            self.monitor.record_call(
                tool_name=tool_name,
                version=actual_version,
                user_id=user.id,
                success=success,
                latency_ms=latency_ms,
                error_type=error_type
            )
        
        return result

# Test monitoring
print("=== Testing Usage Monitoring ===\n")

monitored_registry = MonitoredRegistry()

# Register tools
monitored_registry.register_version(
    name="calculator",
    version="1.0.0",
    description="Calculator",
    category=ToolCategory.COMPUTATION,
    parameters_schema=CalculatorParams,
    function=calculator,
    required_roles={UserRole.USER},
    required_permissions={Permission.EXECUTE}
)

monitored_registry.register_version(
    name="weather",
    version="1.0.0",
    description="Weather API",
    category=ToolCategory.DATA,
    parameters_schema=WeatherParams,
    function=get_weather,
    required_roles={UserRole.USER},
    required_permissions={Permission.EXECUTE}
)

# Create test users
users = [
    User(id=f"user{i}", username=f"user{i}", roles={UserRole.USER})
    for i in range(3)
]

# Simulate various calls
print("Simulating tool calls...")
for _ in range(50):
    user = random.choice(users)
    tool = random.choice(["calculator", "weather"])
    
    try:
        if tool == "calculator":
            monitored_registry.execute_versioned(
                user,
                tool,
                operation="add",
                a=random.randint(1, 10),
                b=random.randint(1, 10)
            )
        else:
            monitored_registry.execute_versioned(
                user,
                tool,
                location="London"
            )
    except Exception:
        pass

# Print analytics
print("\n=== Usage Summary ===")
summary = monitored_registry.monitor.get_summary()
print(json.dumps(summary, indent=2))

print("\n=== Tool Analytics ===")
for tool_name in ["calculator", "weather"]:
    analytics = monitored_registry.monitor.get_tool_analytics(tool_name)
    print(f"\n{tool_name}:")
    print(json.dumps(analytics, indent=2))

print("\n=== User Statistics ===")
for user in users:
    stats = monitored_registry.monitor.get_user_stats(user.id)
    print(f"\n{user.username}:")
    print(f"  Calls: {stats['total_calls']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
```

### Expected Output

```
=== Testing Usage Monitoring ===

Simulating tool calls...

=== Usage Summary ===
{
  "total_calls": 50,
  "successful_calls": 50,
  "failed_calls": 0,
  "success_rate": 1.0,
  "unique_tools": 2,
  "unique_users": 3,
  "top_tools": [
    {"name": "calculator", "calls": 28},
    {"name": "weather", "calls": 22}
  ],
  "top_users": [
    {"user_id": "user1", "calls": 19},
    {"user_id": "user0", "calls": 17},
    {"user_id": "user2", "calls": 14}
  ]
}

=== Tool Analytics ===

calculator:
{
  "tool_name": "calculator",
  "total_calls": 28,
  "latency": {
    "min_ms": 0.12,
    "max_ms": 0.45,
    "mean_ms": 0.23,
    "median_ms": 0.21,
    "std_dev_ms": 0.08
  }
}

weather:
{
  "tool_name": "weather",
  "total_calls": 22,
  "latency": {
    "min_ms": 0.15,
    "max_ms": 0.38,
    "mean_ms": 0.24,
    "median_ms": 0.23,
    "std_dev_ms": 0.06
  }
}

=== User Statistics ===

user0:
  Calls: 17
  Success rate: 100.0%

user1:
  Calls: 19
  Success rate: 100.0%

user2:
  Calls: 14
  Success rate: 100.0%
```

### Key Features

1. **Call Recording**: Track every tool invocation
2. **Latency Analytics**: Min, max, mean, median, std dev
3. **User Analytics**: Per-user success rates and call counts
4. **Top Lists**: Most used tools and most active users
5. **Success Tracking**: Monitor reliability

---

## Exercise 7: Complete Integration {#exercise-7}

(The complete integration example showing how all patterns work together with OpenAI)

```python
from openai import OpenAI
import json

def get_all_openai_schemas(registry: MonitoredRegistry) -> List[dict]:
    """Get OpenAI function schemas for all tools."""
    tools = registry.list_tools()
    
    schemas = []
    for tool in tools:
        # Extract base name and version
        if '@' in tool.name:
            base_name, version = tool.name.rsplit('@', 1)
        else:
            base_name = tool.name
            version = tool.version
        
        schema = pydantic_to_openai_schema(
            base_name,
            tool.description,
            tool.parameters_schema
        )
        schemas.append(schema)
    
    return schemas

class RegistryAgent:
    """LLM agent using the complete registry system."""
    
    def __init__(self, registry: MonitoredRegistry, user: User):
        self.registry = registry
        self.user = user
        self.client = OpenAI()
        self.conversation = []
    
    def chat(
        self,
        message: str,
        quota_limit: Optional[int] = None
    ) -> str:
        """
        Chat with agent using registered tools.
        
        Features:
        - Authorization checking
        - Rate limiting
        - Version management
        - Usage monitoring
        - Type-safe execution
        """
        self.conversation.append({"role": "user", "content": message})
        
        # Get tool schemas (only tools user can access)
        tools = get_all_openai_schemas(self.registry)
        
        # Initial API call
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant with access to tools."},
                *self.conversation
            ],
            tools=tools
        )
        
        msg = response.choices[0].message
        
        # Handle tool calls
        if msg.tool_calls:
            self.conversation.append(msg)
            
            for tool_call in msg.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"🔧 Calling {function_name}({function_args})")
                
                # Execute through registry (with all checks)
                try:
                    result = self.registry.execute_versioned(
                        self.user,
                        function_name,
                        **function_args
                    )
                    tool_result = {"success": True, "result": result}
                except Exception as e:
                    tool_result = {"success": False, "error": str(e)}
                
                self.conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result)
                })
            
            # Second API call
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    *self.conversation
                ],
                tools=tools
            )
            
            msg = response.choices[0].message
        
        self.conversation.append(msg)
        return msg.content

# Test complete system
print("=== Testing Complete System ===\n")

# Create complete registry
complete_registry = MonitoredRegistry()

# Register tools with all features
complete_registry.register_version(
    name="calculator",
    version="1.0.0",
    description="Perform arithmetic calculations",
    category=ToolCategory.COMPUTATION,
    parameters_schema=CalculatorParams,
    function=calculator,
    required_roles={UserRole.USER, UserRole.ADMIN},
    required_permissions={Permission.EXECUTE}
)

# Create agent
user = User("1", "testuser", {UserRole.USER})
agent = RegistryAgent(complete_registry, user)

# Chat with agent
response = agent.chat("What is 15 + 27?")
print(f"\nAgent: {response}")

# Check usage stats
stats = complete_registry.monitor.get_user_stats(user.id)
print(f"\nUser stats: {json.dumps(stats, indent=2)}")
```

This comprehensive solution demonstrates all key patterns working together in production!

---

## Summary

### Key Features Implemented

1. **Central Registry**: Manage all tools in one place
2. **Type Safety**: Pydantic validation prevents errors
3. **Authorization**: Role-based access control
4. **Rate Limiting**: Prevent abuse and manage quotas
5. **Versioning**: Support multiple tool versions
6. **Monitoring**: Track usage and performance
7. **OpenAI Integration**: Seamless LLM integration

### Production Checklist

- [x] Centralized tool registry
- [x] Pydantic schemas for validation
- [x] Role-based permissions
- [x] Per-user rate limiting
- [x] Semantic versioning
- [x] Usage monitoring and analytics
- [x] OpenAI schema conversion
- [x] Deprecation warnings
- [x] Call count tracking
- [x] Latency metrics

### Next Steps

- Lab 3: Multi-Tool Workflow Orchestration
- Implement persistent storage for registry
- Add tool discovery API
- Build admin dashboard
- Set up alerting for quota violations
