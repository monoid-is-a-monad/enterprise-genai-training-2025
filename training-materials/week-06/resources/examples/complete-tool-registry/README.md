# Complete Tool Registry Example

A production-ready tool registry system demonstrating all concepts from Week 6 lessons and labs.

## Features

✅ **Tool Registration**
- Register tools with metadata
- Type-safe schemas using Pydantic
- Automatic OpenAI schema conversion

✅ **Authentication & Authorization**
- Role-based access control (ADMIN, USER, GUEST)
- Per-tool authorization checks
- User context management

✅ **Rate Limiting**
- Per-user quotas
- Time-window tracking
- Automatic reset

✅ **Versioning**
- Semantic versioning support
- Multiple versions per tool
- Version resolution

✅ **Monitoring**
- Usage metrics collection
- Success/error tracking
- Latency measurements

✅ **OpenAI Integration**
- Complete function calling workflow
- Parallel tool execution
- Error handling

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the example
python main.py

# Interactive mode
python main.py --interactive

# With real OpenAI API
export OPENAI_API_KEY="sk-..."
python main.py --use-openai
```

## Architecture

```
ToolRegistry
├── Registration & Discovery
│   ├── register()
│   ├── get()
│   └── list_tools()
│
├── Schema Conversion
│   ├── to_openai_schema()
│   └── get_all_openai_schemas()
│
├── Security Layer
│   ├── AuthorizationManager
│   └── Role-based checks
│
├── Rate Limiting
│   ├── QuotaManager
│   └── Token bucket algorithm
│
├── Versioning
│   ├── Version tracking
│   └── Semantic versioning
│
├── Monitoring
│   ├── UsageMonitor
│   ├── Metrics collection
│   └── Success rate tracking
│
└── Execution
    ├── execute()
    ├── Error handling
    └── Result formatting
```

## Code Structure

```
complete-tool-registry/
├── README.md           # This file
├── requirements.txt    # Dependencies
├── main.py            # Entry point with examples
├── config.py          # Configuration
└── modules/
    ├── registry.py    # Core registry implementation
    ├── auth.py        # Authentication & authorization
    ├── rate_limit.py  # Rate limiting
    ├── monitoring.py  # Metrics and monitoring
    ├── tools.py       # Example tool implementations
    └── integration.py # OpenAI integration
```

## Usage Examples

### Basic Registration

```python
from modules.registry import ToolRegistry
from modules.tools import calculator_tool, weather_tool

registry = ToolRegistry()

# Register tools
registry.register(
    name="calculator",
    func=calculator_tool,
    description="Perform calculations",
    category="utility"
)

# List tools
tools = registry.list_tools()
```

### With Authentication

```python
from modules.auth import AuthorizationManager, UserRole

auth_manager = AuthorizationManager()

# Configure permissions
auth_manager.set_permissions("calculator", [UserRole.USER, UserRole.ADMIN])

# Check access
user = {"user_id": "123", "role": UserRole.USER}
can_use = auth_manager.check_permission(user, "calculator")
```

### With Rate Limiting

```python
from modules.rate_limit import QuotaManager

quota_manager = QuotaManager(default_quota=100)

# Check quota
user_id = "user123"
if quota_manager.check_quota(user_id, cost=1):
    # Execute tool
    result = registry.execute("calculator", params)
    quota_manager.consume_quota(user_id, cost=1)
```

### With Monitoring

```python
from modules.monitoring import UsageMonitor

monitor = UsageMonitor()

# Track execution
with monitor.track_execution("calculator", user_id="123"):
    result = registry.execute("calculator", params)

# Get metrics
metrics = monitor.get_tool_metrics("calculator")
print(f"Success rate: {metrics.success_rate:.2%}")
```

### OpenAI Integration

```python
from modules.integration import RegistryAgent

agent = RegistryAgent(registry, api_key="sk-...")

# Execute with OpenAI
response = await agent.chat(
    "What's the weather in San Francisco?",
    user={"user_id": "123", "role": UserRole.USER}
)
```

## Configuration

Edit `config.py` to customize:

```python
class RegistryConfig:
    # Rate limiting
    DEFAULT_QUOTA = 100  # Requests per window
    QUOTA_WINDOW_MINUTES = 60
    
    # Monitoring
    ENABLE_METRICS = True
    METRICS_RETENTION_HOURS = 24
    
    # Authentication
    ENABLE_AUTH = True
    DEFAULT_ROLE = UserRole.USER
    
    # Execution
    DEFAULT_TIMEOUT = 30  # seconds
    MAX_RETRIES = 3
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=modules

# Run specific test
pytest tests/test_registry.py
```

## Monitoring Output

The example generates monitoring data:

```
Tool: calculator
  Total calls: 150
  Success rate: 96.67%
  Avg latency: 45ms
  P95 latency: 120ms
  Error rate: 3.33%

Tool: get_weather
  Total calls: 85
  Success rate: 100.00%
  Avg latency: 250ms
  P95 latency: 450ms
  Error rate: 0.00%
```

## Performance

Typical performance characteristics:

- **Registration**: < 1ms per tool
- **Lookup**: < 0.1ms
- **Schema conversion**: < 5ms
- **Authorization check**: < 0.5ms
- **Quota check**: < 1ms
- **Execution overhead**: < 10ms

## Production Considerations

### Security
- [ ] Use secure secret management for API keys
- [ ] Implement proper authentication (JWT, OAuth)
- [ ] Add input sanitization
- [ ] Enable audit logging
- [ ] Set up rate limiting per IP

### Monitoring
- [ ] Integrate with Prometheus/Grafana
- [ ] Set up alerting (PagerDuty, etc.)
- [ ] Add distributed tracing
- [ ] Configure log aggregation
- [ ] Monitor quota usage

### Reliability
- [ ] Add circuit breakers
- [ ] Implement fallback strategies
- [ ] Set appropriate timeouts
- [ ] Add retry with backoff
- [ ] Handle degraded mode

### Scalability
- [ ] Use Redis for rate limiting
- [ ] Distribute registry (service mesh)
- [ ] Add caching layer
- [ ] Implement sharding
- [ ] Use async execution

## Extending the Example

### Add a New Tool

```python
# modules/tools.py

from pydantic import BaseModel, Field

class TranslateParams(BaseModel):
    text: str = Field(..., description="Text to translate")
    target_language: str = Field(..., description="Target language")

def translate_tool(text: str, target_language: str) -> dict:
    """Translate text to target language."""
    # Implementation
    return {"translated": "...", "language": target_language}

# Register
registry.register(
    name="translate",
    func=translate_tool,
    description="Translate text",
    category="utility",
    schema=TranslateParams
)
```

### Add Custom Authorization

```python
# modules/auth.py

class CustomAuthManager(AuthorizationManager):
    def check_permission(self, user: dict, tool_name: str) -> bool:
        # Custom logic
        if tool_name.startswith("admin_"):
            return user.get("is_admin", False)
        return super().check_permission(user, tool_name)
```

### Add Custom Metrics

```python
# modules/monitoring.py

class CustomMonitor(UsageMonitor):
    def track_custom_metric(self, tool_name: str, metric: str, value: float):
        """Track custom metric."""
        self.custom_metrics[tool_name][metric].append(value)
```

## Troubleshooting

### Tool Not Found
```python
# Check if tool is registered
if registry.has_tool("my_tool"):
    print("Tool found")
else:
    print("Tool not registered")
```

### Authorization Failed
```python
# Check user permissions
permissions = auth_manager.get_permissions("tool_name")
print(f"Required roles: {permissions}")
```

### Quota Exceeded
```python
# Check remaining quota
remaining = quota_manager.get_remaining_quota("user_id")
print(f"Remaining: {remaining}")
```

### Schema Validation Error
```python
# Validate schema before registration
try:
    registry.register(name="tool", func=func, schema=Schema)
except ValidationError as e:
    print(f"Schema error: {e}")
```

## Related Examples

- `../production-workflow/` — Workflow orchestration
- `../monitoring-setup/` — Advanced monitoring
- `../error-handling-demo/` — Error handling patterns

## References

- Week 6 Lesson 2: Building Production Tool Systems
- Week 6 Lab 2: Building a Tool Registry System
- `../../function-calling-best-practices.md`
- `../../tool-schema-design-guide.md`
