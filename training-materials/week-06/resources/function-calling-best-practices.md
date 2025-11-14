# Function Calling Best Practices

Production guidelines for reliable, scalable function calling systems.

## Table of Contents
- [Core Principles](#core-principles)
- [Schema Design](#schema-design)
- [Parameter Handling](#parameter-handling)
- [Error Management](#error-management)
- [Performance Optimization](#performance-optimization)
- [Security](#security)
- [Testing](#testing)
- [Monitoring](#monitoring)

---

## Core Principles

### 1. Design for LLM Understanding

**Good**: Clear, descriptive names and documentation
```python
def get_weather_forecast(
    location: str,
    days: int = 3,
    units: Literal["celsius", "fahrenheit"] = "celsius"
) -> WeatherForecast:
    """
    Get weather forecast for a location.
    
    Args:
        location: City name or coordinates (e.g., "London" or "51.5074,-0.1278")
        days: Number of days to forecast (1-7)
        units: Temperature units
        
    Returns:
        Weather forecast with daily predictions
    """
```

**Bad**: Ambiguous names and minimal documentation
```python
def get_data(loc, n=3, u="c"):
    """Get data."""
```

### 2. Use Parallel Execution When Possible

**Good**: Enable parallel tool calls
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    parallel_tool_calls=True  # ✓ Allow parallel execution
)
```

**Bad**: Force sequential execution unnecessarily
```python
# ✗ Disabling parallel calls without reason
parallel_tool_calls=False
```

### 3. Return Structured Data

**Good**: Structured, consistent responses
```python
def search_products(query: str) -> dict:
    return {
        "success": True,
        "results": [
            {"id": "123", "name": "Product A", "price": 29.99},
            {"id": "456", "name": "Product B", "price": 39.99}
        ],
        "total_count": 2
    }
```

**Bad**: Unstructured string responses
```python
def search_products(query: str) -> str:
    # ✗ LLM must parse text
    return "Found Product A ($29.99) and Product B ($39.99)"
```

---

## Schema Design

### Use Pydantic for Type Safety

```python
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional

class WeatherParams(BaseModel):
    """Type-safe weather parameters."""
    
    location: str = Field(
        ...,
        description="City name or 'latitude,longitude'",
        min_length=2,
        max_length=100
    )
    
    units: Literal["celsius", "fahrenheit", "kelvin"] = Field(
        default="celsius",
        description="Temperature units"
    )
    
    include_hourly: bool = Field(
        default=False,
        description="Include hourly forecast"
    )
    
    @validator('location')
    def validate_location(cls, v):
        """Validate and normalize location."""
        v = v.strip()
        if not v:
            raise ValueError("Location cannot be empty")
        return v.title()

# Use in function
def get_weather(**kwargs) -> dict:
    params = WeatherParams(**kwargs)  # Automatic validation
    return fetch_weather(params.location, params.units)
```

### Provide Clear Constraints

```python
class SearchParams(BaseModel):
    """Search with clear limits."""
    
    query: str = Field(
        ...,
        description="Search query",
        min_length=1,
        max_length=500
    )
    
    max_results: int = Field(
        default=10,
        description="Maximum results to return",
        ge=1,  # Greater than or equal to 1
        le=100  # Less than or equal to 100
    )
    
    sort_by: Literal["relevance", "date", "popularity"] = Field(
        default="relevance"
    )
```

### Use Enums for Fixed Options

```python
from enum import Enum

class NotificationChannel(str, Enum):
    """Available notification channels."""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    WEBHOOK = "webhook"

class SendNotificationParams(BaseModel):
    channel: NotificationChannel
    recipient: str
    message: str
```

---

## Parameter Handling

### Validate All Inputs

```python
def transfer_funds(
    from_account: str,
    to_account: str,
    amount: float,
    currency: str = "USD"
) -> dict:
    """Transfer funds with validation."""
    
    # Validate accounts
    if not re.match(r'^[A-Z0-9]{10,}$', from_account):
        return {
            "success": False,
            "error": "Invalid source account format",
            "suggestion": "Use account number format: ABC1234567"
        }
    
    # Validate amount
    if amount <= 0:
        return {
            "success": False,
            "error": "Amount must be positive"
        }
    
    if amount > 10000:
        return {
            "success": False,
            "error": "Amount exceeds single transaction limit",
            "suggestion": "Maximum transfer amount is $10,000"
        }
    
    # Execute transfer
    try:
        result = process_transfer(from_account, to_account, amount, currency)
        return {"success": True, "transaction_id": result.id}
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "suggestion": "Contact support if issue persists"
        }
```

### Handle Optional Parameters Gracefully

```python
def search_documents(
    query: str,
    category: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    max_results: int = 10
) -> dict:
    """Search with optional filters."""
    
    filters = {"query": query}
    
    if category:
        filters["category"] = category
    
    if date_from:
        try:
            filters["date_from"] = datetime.fromisoformat(date_from)
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid date format: {date_from}",
                "suggestion": "Use ISO format: YYYY-MM-DD"
            }
    
    return execute_search(filters, max_results)
```

---

## Error Management

### Return Structured Errors

```python
def call_external_api(endpoint: str) -> dict:
    """Call API with structured error handling."""
    try:
        response = requests.get(endpoint, timeout=5)
        response.raise_for_status()
        
        return {
            "success": True,
            "data": response.json()
        }
        
    except requests.Timeout:
        return {
            "success": False,
            "error": "Request timeout",
            "error_type": "timeout",
            "suggestion": "Try again or check service status"
        }
        
    except requests.HTTPError as e:
        return {
            "success": False,
            "error": f"HTTP {e.response.status_code}",
            "error_type": "http_error",
            "details": e.response.text[:200],
            "suggestion": "Check API documentation"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": "unknown",
            "suggestion": "Contact support with this error message"
        }
```

### Provide Actionable Error Messages

**Good**: Specific, actionable errors
```python
{
    "success": False,
    "error": "API rate limit exceeded (100 requests/hour)",
    "retry_after": 3600,
    "suggestion": "Wait 1 hour or upgrade to premium plan"
}
```

**Bad**: Vague errors
```python
{
    "success": False,
    "error": "Error occurred"
}
```

---

## Performance Optimization

### Use Caching Strategically

```python
from functools import lru_cache
import hashlib
import json

class ToolCache:
    """Cache tool results."""
    
    def __init__(self, ttl_seconds: int = 300):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get_cache_key(self, tool_name: str, params: dict) -> str:
        """Generate cache key."""
        # Sort params for consistent key
        sorted_params = json.dumps(params, sort_keys=True)
        return hashlib.md5(f"{tool_name}:{sorted_params}".encode()).hexdigest()
    
    def get(self, tool_name: str, params: dict) -> Optional[Any]:
        """Get cached result."""
        key = self.get_cache_key(tool_name, params)
        entry = self.cache.get(key)
        
        if entry:
            # Check if expired
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["result"]
            else:
                del self.cache[key]
        
        return None
    
    def set(self, tool_name: str, params: dict, result: Any):
        """Cache result."""
        key = self.get_cache_key(tool_name, params)
        self.cache[key] = {
            "result": result,
            "timestamp": time.time()
        }

# Usage
cache = ToolCache(ttl_seconds=300)

def get_stock_price(symbol: str) -> dict:
    """Get stock price with caching."""
    
    # Check cache
    cached = cache.get("get_stock_price", {"symbol": symbol})
    if cached:
        return {**cached, "cached": True}
    
    # Fetch fresh data
    result = fetch_stock_price(symbol)
    
    # Cache result
    cache.set("get_stock_price", {"symbol": symbol}, result)
    
    return result
```

### Implement Timeouts

```python
import asyncio
from concurrent.futures import TimeoutError

async def execute_with_timeout(
    func: Callable,
    timeout_seconds: float = 30.0,
    **kwargs
) -> dict:
    """Execute function with timeout."""
    try:
        result = await asyncio.wait_for(
            func(**kwargs),
            timeout=timeout_seconds
        )
        return {"success": True, "result": result}
        
    except asyncio.TimeoutError:
        return {
            "success": False,
            "error": f"Operation timed out after {timeout_seconds}s",
            "error_type": "timeout"
        }
```

### Batch Operations When Possible

**Good**: Batch API calls
```python
def get_user_profiles(user_ids: list[str]) -> dict:
    """Get multiple profiles in one call."""
    profiles = api.batch_get_users(user_ids)
    return {
        "success": True,
        "profiles": profiles,
        "count": len(profiles)
    }
```

**Bad**: Individual calls per user
```python
def get_user_profile(user_id: str) -> dict:
    """One profile per call - inefficient for multiple users."""
    return api.get_user(user_id)
```

---

## Security

### Sanitize Inputs

```python
import re
import html

def sanitize_string(value: str) -> str:
    """Sanitize string input."""
    # Remove null bytes
    value = value.replace('\x00', '')
    
    # HTML escape
    value = html.escape(value)
    
    # Limit length
    value = value[:1000]
    
    return value.strip()

def sanitize_sql_identifier(identifier: str) -> str:
    """Sanitize SQL identifiers."""
    # Only allow alphanumeric and underscore
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', identifier):
        raise ValueError(f"Invalid identifier: {identifier}")
    return identifier

def execute_query(table_name: str, filters: dict) -> dict:
    """Execute query with sanitization."""
    
    # Sanitize table name
    table_name = sanitize_sql_identifier(table_name)
    
    # Use parameterized query
    query = f"SELECT * FROM {table_name} WHERE status = ?"
    params = [sanitize_string(filters.get("status", "active"))]
    
    return db.execute(query, params)
```

### Implement Authorization

```python
from dataclasses import dataclass
from typing import Set

@dataclass
class User:
    id: str
    roles: Set[str]

def check_permission(user: User, tool_name: str) -> bool:
    """Check if user can use tool."""
    
    # Tool permission mapping
    permissions = {
        "read_data": {"user", "admin"},
        "write_data": {"admin"},
        "delete_data": {"admin"},
        "send_email": {"user", "admin"}
    }
    
    required_roles = permissions.get(tool_name, set())
    return bool(user.roles & required_roles)

def execute_tool(user: User, tool_name: str, **kwargs) -> dict:
    """Execute tool with authorization."""
    
    if not check_permission(user, tool_name):
        return {
            "success": False,
            "error": "Unauthorized",
            "error_type": "permission_denied"
        }
    
    return tools[tool_name](**kwargs)
```

### Rate Limit by User

```python
from collections import defaultdict
from datetime import datetime, timedelta

class RateLimiter:
    """Per-user rate limiting."""
    
    def __init__(self, max_calls: int = 100, window_minutes: int = 60):
        self.max_calls = max_calls
        self.window = timedelta(minutes=window_minutes)
        self.calls = defaultdict(list)
    
    def is_allowed(self, user_id: str) -> tuple[bool, Optional[int]]:
        """Check if user can make call."""
        now = datetime.now()
        cutoff = now - self.window
        
        # Clean old calls
        self.calls[user_id] = [
            ts for ts in self.calls[user_id] if ts > cutoff
        ]
        
        # Check limit
        if len(self.calls[user_id]) >= self.max_calls:
            # Calculate retry after
            oldest = min(self.calls[user_id])
            retry_after = int((oldest + self.window - now).total_seconds())
            return False, retry_after
        
        # Record call
        self.calls[user_id].append(now)
        return True, None

rate_limiter = RateLimiter(max_calls=100, window_minutes=60)

def execute_with_rate_limit(user_id: str, tool_name: str, **kwargs) -> dict:
    """Execute with rate limiting."""
    
    allowed, retry_after = rate_limiter.is_allowed(user_id)
    
    if not allowed:
        return {
            "success": False,
            "error": "Rate limit exceeded",
            "error_type": "rate_limit",
            "retry_after": retry_after,
            "suggestion": f"Wait {retry_after} seconds before retrying"
        }
    
    return tools[tool_name](**kwargs)
```

---

## Testing

### Test Tool Schemas

```python
import pytest
from openai import OpenAI

def test_tool_schema_format():
    """Test schema converts to OpenAI format."""
    
    schema = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
    
    # Validate schema works with OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What's the weather in London?"}],
        tools=[schema]
    )
    
    assert response.choices[0].message.tool_calls is not None

def test_parameter_validation():
    """Test parameter validation."""
    
    # Valid params
    result = get_weather(location="London", units="celsius")
    assert result["success"] == True
    
    # Invalid units
    result = get_weather(location="London", units="invalid")
    assert result["success"] == False
    assert "error" in result
    
    # Missing location
    with pytest.raises(ValueError):
        get_weather(units="celsius")

def test_error_handling():
    """Test error responses."""
    
    # Simulate API failure
    with patch('requests.get') as mock_get:
        mock_get.side_effect = requests.Timeout()
        
        result = call_api("https://api.example.com")
        
        assert result["success"] == False
        assert result["error_type"] == "timeout"
        assert "suggestion" in result
```

### Test End-to-End Flow

```python
def test_function_calling_flow():
    """Test complete function calling flow."""
    
    client = OpenAI()
    
    messages = [
        {"role": "user", "content": "What's the weather in Paris?"}
    ]
    
    # First call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=get_all_tool_schemas()
    )
    
    msg = response.choices[0].message
    assert msg.tool_calls is not None
    
    # Execute tools
    messages.append(msg)
    
    for tool_call in msg.tool_calls:
        result = execute_tool(
            tool_call.function.name,
            **json.loads(tool_call.function.arguments)
        )
        
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })
    
    # Second call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=get_all_tool_schemas()
    )
    
    final_message = response.choices[0].message.content
    assert "Paris" in final_message
    assert final_message is not None
```

---

## Monitoring

### Track Key Metrics

```python
from dataclasses import dataclass
from collections import defaultdict
import time

@dataclass
class ToolMetrics:
    """Metrics for a tool."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency_ms: float = 0.0
    latencies: list[float] = None
    
    def __post_init__(self):
        if self.latencies is None:
            self.latencies = []
    
    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls
    
    @property
    def avg_latency_ms(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls
    
    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx]

class MetricsCollector:
    """Collect tool metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(ToolMetrics)
    
    def record_call(
        self,
        tool_name: str,
        success: bool,
        latency_ms: float
    ):
        """Record a tool call."""
        m = self.metrics[tool_name]
        m.total_calls += 1
        m.total_latency_ms += latency_ms
        m.latencies.append(latency_ms)
        
        if success:
            m.successful_calls += 1
        else:
            m.failed_calls += 1
    
    def get_summary(self) -> dict:
        """Get metrics summary."""
        return {
            tool_name: {
                "total_calls": m.total_calls,
                "success_rate": f"{m.success_rate:.2%}",
                "avg_latency_ms": f"{m.avg_latency_ms:.1f}",
                "p95_latency_ms": f"{m.p95_latency_ms:.1f}"
            }
            for tool_name, m in self.metrics.items()
        }

# Usage
metrics = MetricsCollector()

def execute_with_metrics(tool_name: str, **kwargs) -> dict:
    """Execute tool with metrics tracking."""
    start = time.time()
    
    try:
        result = tools[tool_name](**kwargs)
        success = result.get("success", True)
    except Exception as e:
        result = {"success": False, "error": str(e)}
        success = False
    
    latency_ms = (time.time() - start) * 1000
    metrics.record_call(tool_name, success, latency_ms)
    
    return result
```

### Log Tool Usage

```python
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tools")

def execute_with_logging(
    user_id: str,
    tool_name: str,
    params: dict
) -> dict:
    """Execute with structured logging."""
    
    logger.info(
        "Tool execution started",
        extra={
            "user_id": user_id,
            "tool_name": tool_name,
            "params": json.dumps(params)
        }
    )
    
    start = time.time()
    
    try:
        result = tools[tool_name](**params)
        
        logger.info(
            "Tool execution completed",
            extra={
                "user_id": user_id,
                "tool_name": tool_name,
                "success": result.get("success", True),
                "latency_ms": (time.time() - start) * 1000
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(
            "Tool execution failed",
            extra={
                "user_id": user_id,
                "tool_name": tool_name,
                "error": str(e),
                "latency_ms": (time.time() - start) * 1000
            },
            exc_info=True
        )
        raise
```

---

## Production Checklist

### Before Deployment

- [ ] All tool schemas validated
- [ ] Parameter validation implemented
- [ ] Error handling comprehensive
- [ ] Structured error responses
- [ ] Timeouts configured
- [ ] Rate limiting active
- [ ] Authorization implemented
- [ ] Input sanitization complete
- [ ] Caching strategy defined
- [ ] Metrics collection enabled
- [ ] Logging configured
- [ ] Tests passing (unit + integration)
- [ ] Documentation complete
- [ ] Security review done
- [ ] Performance tested

### After Deployment

- [ ] Monitor success rates
- [ ] Track latencies
- [ ] Review error patterns
- [ ] Analyze usage patterns
- [ ] Optimize slow tools
- [ ] Update documentation
- [ ] Tune rate limits
- [ ] Adjust cache TTLs
- [ ] Scale resources as needed
- [ ] Collect user feedback

---

## Common Pitfalls

### 1. Poor Error Messages
**Problem**: Generic errors without context
**Solution**: Provide specific, actionable error messages

### 2. Missing Validation
**Problem**: Invalid parameters cause crashes
**Solution**: Validate all inputs with Pydantic

### 3. No Timeouts
**Problem**: Hung operations block execution
**Solution**: Set timeouts on all external calls

### 4. Synchronous Execution
**Problem**: Sequential calls waste time
**Solution**: Enable parallel tool calls

### 5. Unstructured Responses
**Problem**: LLM struggles to parse results
**Solution**: Return consistent JSON structures

### 6. No Rate Limiting
**Problem**: Resource exhaustion and cost spikes
**Solution**: Implement per-user rate limits

### 7. Missing Monitoring
**Problem**: Can't diagnose issues
**Solution**: Track metrics and log all calls

### 8. Weak Security
**Problem**: Unauthorized access or injection
**Solution**: Sanitize inputs and check permissions

---

## Related Resources

- [tool-schema-design-guide.md](./tool-schema-design-guide.md) — Schema design patterns
- [error-handling-patterns.md](./error-handling-patterns.md) — Error handling strategies
- [workflow-patterns-cheatsheet.md](./workflow-patterns-cheatsheet.md) — Orchestration patterns
- `../lessons/01-advanced-function-calling-patterns.md` — Advanced techniques
- `../lessons/02-building-production-tool-systems.md` — Production systems
