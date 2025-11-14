# Error Handling Patterns for Tool Systems

Comprehensive strategies for handling errors in function calling and tool orchestration.

## Table of Contents
- [Error Categories](#error-categories)
- [Error Response Structure](#error-response-structure)
- [Retry Strategies](#retry-strategies)
- [Circuit Breakers](#circuit-breakers)
- [Fallback Patterns](#fallback-patterns)
- [Timeout Handling](#timeout-handling)
- [Error Recovery](#error-recovery)
- [Monitoring & Alerting](#monitoring--alerting)

---

## Error Categories

### 1. Transient Errors (Retry)

Temporary failures that may succeed on retry.

```python
TRANSIENT_ERRORS = {
    "timeout",
    "connection_error",
    "rate_limit",
    "service_unavailable",
    "gateway_timeout",
    "temporary_failure"
}

def is_transient_error(error_type: str) -> bool:
    """Check if error is transient."""
    return error_type in TRANSIENT_ERRORS
```

**Examples:**
- Network timeouts
- Rate limiting (429)
- Service unavailable (503)
- Connection refused
- Temporary database locks

**Strategy:** Retry with exponential backoff

### 2. Permanent Errors (Fail Fast)

Errors that won't be fixed by retrying.

```python
PERMANENT_ERRORS = {
    "authentication_failed",
    "authorization_denied",
    "invalid_parameters",
    "not_found",
    "malformed_request",
    "unsupported_operation"
}

def is_permanent_error(error_type: str) -> bool:
    """Check if error is permanent."""
    return error_type in PERMANENT_ERRORS
```

**Examples:**
- Invalid credentials (401)
- Forbidden (403)
- Not found (404)
- Bad request (400)
- Validation errors

**Strategy:** Return error immediately with clear message

### 3. User Errors (Recoverable)

Errors due to user input that can be corrected.

```python
USER_ERROR_CATEGORIES = {
    "missing_parameter",
    "invalid_format",
    "out_of_range",
    "constraint_violation",
    "invalid_choice"
}
```

**Examples:**
- Missing required parameters
- Wrong date format
- Value out of range
- Invalid enum choice

**Strategy:** Return helpful error with correction guidance

---

## Error Response Structure

### Standard Error Format

```python
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"           # Minor issue, degraded functionality
    MEDIUM = "medium"     # Feature unavailable
    HIGH = "high"         # Critical feature broken
    CRITICAL = "critical" # Service down

@dataclass
class ToolError:
    """Structured error response."""
    success: bool = False
    error: str                          # Human-readable message
    error_type: str                     # Machine-readable type
    error_code: Optional[str] = None   # Error code for lookup
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    suggestion: Optional[str] = None    # What user should do
    retry_after: Optional[int] = None   # Seconds until retry
    details: Optional[Dict[str, Any]] = None
    
def create_error_response(
    error: str,
    error_type: str,
    **kwargs
) -> dict:
    """Create standardized error response."""
    return {
        "success": False,
        "error": error,
        "error_type": error_type,
        **kwargs
    }
```

### Error Response Examples

**Validation Error:**
```python
{
    "success": False,
    "error": "Invalid date format",
    "error_type": "validation_error",
    "error_code": "E001",
    "severity": "low",
    "suggestion": "Use ISO 8601 format (YYYY-MM-DD). Example: '2024-03-15'",
    "details": {
        "field": "start_date",
        "provided": "03/15/2024",
        "expected_format": "YYYY-MM-DD"
    }
}
```

**Rate Limit Error:**
```python
{
    "success": False,
    "error": "Rate limit exceeded (100 requests/hour)",
    "error_type": "rate_limit",
    "error_code": "E429",
    "severity": "medium",
    "retry_after": 3600,
    "suggestion": "Wait 1 hour before retrying, or upgrade to premium plan",
    "details": {
        "limit": 100,
        "window": "1 hour",
        "reset_time": "2024-03-15T14:00:00Z"
    }
}
```

**Service Error:**
```python
{
    "success": False,
    "error": "External API unavailable",
    "error_type": "service_unavailable",
    "error_code": "E503",
    "severity": "high",
    "retry_after": 60,
    "suggestion": "Service is temporarily down. Will retry automatically.",
    "details": {
        "service": "weather_api",
        "status_url": "https://status.weatherapi.com"
    }
}
```

---

## Retry Strategies

### Exponential Backoff

```python
import time
import random
from typing import Callable, Any, Optional

def exponential_backoff_retry(
    func: Callable,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    **kwargs
) -> dict:
    """
    Retry function with exponential backoff.
    
    Args:
        func: Function to retry
        max_attempts: Maximum retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Add random jitter to prevent thundering herd
        **kwargs: Arguments to pass to func
    
    Returns:
        Function result or error dict
    """
    
    for attempt in range(max_attempts):
        try:
            result = func(**kwargs)
            
            # Check if result indicates retry
            if isinstance(result, dict) and not result.get("success"):
                error_type = result.get("error_type", "")
                
                # Don't retry permanent errors
                if is_permanent_error(error_type):
                    return result
                
                # Last attempt - return error
                if attempt == max_attempts - 1:
                    result["attempts"] = attempt + 1
                    return result
                
                # Calculate delay
                delay = min(
                    initial_delay * (exponential_base ** attempt),
                    max_delay
                )
                
                # Add jitter (±25%)
                if jitter:
                    delay *= random.uniform(0.75, 1.25)
                
                print(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s...")
                time.sleep(delay)
                continue
            
            return result
            
        except Exception as e:
            if attempt == max_attempts - 1:
                return {
                    "success": False,
                    "error": str(e),
                    "error_type": "exception",
                    "attempts": attempt + 1
                }
            
            delay = min(
                initial_delay * (exponential_base ** attempt),
                max_delay
            )
            if jitter:
                delay *= random.uniform(0.75, 1.25)
            
            print(f"Exception on attempt {attempt + 1}, retrying in {delay:.2f}s...")
            time.sleep(delay)
    
    return {
        "success": False,
        "error": "Max retries exceeded",
        "error_type": "max_retries",
        "attempts": max_attempts
    }

# Usage
def flaky_api_call(data: str) -> dict:
    """Simulated flaky API."""
    if random.random() < 0.7:  # 70% failure rate
        return {
            "success": False,
            "error": "Service unavailable",
            "error_type": "service_unavailable"
        }
    return {"success": True, "data": f"Processed: {data}"}

result = exponential_backoff_retry(
    flaky_api_call,
    max_attempts=5,
    initial_delay=1.0,
    data="test"
)
```

### Tenacity Library

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_result
)
import requests

# Retry on specific exceptions
@retry(
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def call_external_api(url: str) -> dict:
    """Call API with automatic retry."""
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response.json()

# Retry on result condition
def should_retry(result):
    """Check if result indicates retry needed."""
    return isinstance(result, dict) and not result.get("success")

@retry(
    retry=retry_if_result(should_retry),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
def call_tool_with_retry(tool_name: str, **kwargs) -> dict:
    """Call tool with retry on failure."""
    return tools[tool_name](**kwargs)
```

---

## Circuit Breakers

### Circuit Breaker Pattern

```python
from enum import Enum
from datetime import datetime, timedelta
from typing import Callable, Any

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    """Circuit breaker for failing services."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.success_threshold = success_threshold
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
    
    def call(self, func: Callable, *args, **kwargs) -> dict:
        """Execute function through circuit breaker."""
        
        # Check if circuit is OPEN
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout_seconds):
                print("Circuit HALF_OPEN - testing recovery")
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                return {
                    "success": False,
                    "error": "Circuit breaker OPEN - service unavailable",
                    "error_type": "circuit_open",
                    "retry_after": self.timeout_seconds
                }
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            
            # Check result
            if isinstance(result, dict) and result.get("success"):
                self._on_success()
                return result
            else:
                self._on_failure()
                return result
                
        except Exception as e:
            self._on_failure()
            return {
                "success": False,
                "error": str(e),
                "error_type": "exception"
            }
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                print("Circuit CLOSED - service recovered")
                self.state = CircuitState.CLOSED
                self.success_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            print("Circuit OPEN - recovery failed")
            self.state = CircuitState.OPEN
            self.failure_count = 0
        
        elif self.failure_count >= self.failure_threshold:
            print(f"Circuit OPEN - {self.failure_count} failures")
            self.state = CircuitState.OPEN
    
    def reset(self):
        """Manually reset circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0

# Usage
circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    timeout_seconds=10,
    success_threshold=2
)

def unreliable_service() -> dict:
    """Simulated unreliable service."""
    if random.random() < 0.8:  # 80% failure
        return {
            "success": False,
            "error": "Service error",
            "error_type": "service_unavailable"
        }
    return {"success": True, "data": "OK"}

# Call through circuit breaker
for i in range(10):
    result = circuit_breaker.call(unreliable_service)
    print(f"Call {i+1}: {result}")
    time.sleep(1)
```

---

## Fallback Patterns

### Primary-Secondary-Cache Pattern

```python
from typing import Optional, Callable, Any

class FallbackExecutor:
    """Execute with fallback strategies."""
    
    def __init__(self, cache: Optional[Dict] = None):
        self.cache = cache or {}
    
    def execute_with_fallback(
        self,
        primary: Callable,
        secondary: Optional[Callable] = None,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> dict:
        """
        Execute with fallback chain:
        1. Try primary function
        2. Try secondary function
        3. Return cached result
        4. Return error
        """
        
        # Try primary
        try:
            result = primary(**kwargs)
            if result.get("success"):
                # Cache successful result
                if cache_key:
                    self.cache[cache_key] = result
                return result
        except Exception as e:
            print(f"Primary failed: {e}")
        
        # Try secondary
        if secondary:
            try:
                result = secondary(**kwargs)
                if result.get("success"):
                    result["source"] = "secondary"
                    if cache_key:
                        self.cache[cache_key] = result
                    return result
            except Exception as e:
                print(f"Secondary failed: {e}")
        
        # Try cache
        if cache_key and cache_key in self.cache:
            cached = self.cache[cache_key]
            cached["source"] = "cache"
            cached["warning"] = "Using cached data - service unavailable"
            return cached
        
        # All fallbacks failed
        return {
            "success": False,
            "error": "All services unavailable and no cached data",
            "error_type": "all_fallbacks_failed",
            "severity": "high"
        }

# Usage
executor = FallbackExecutor()

def primary_weather_api(location: str) -> dict:
    """Primary weather service."""
    # Simulated failure
    raise Exception("Primary API down")

def secondary_weather_api(location: str) -> dict:
    """Secondary weather service."""
    return {
        "success": True,
        "location": location,
        "temp": 20,
        "source": "secondary"
    }

result = executor.execute_with_fallback(
    primary=primary_weather_api,
    secondary=secondary_weather_api,
    cache_key="weather_london",
    location="London"
)
print(result)
```

### Degraded Mode

```python
class ServiceMode(Enum):
    """Service operation modes."""
    FULL = "full"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"

class DegradedService:
    """Service with degraded mode support."""
    
    def __init__(self):
        self.mode = ServiceMode.FULL
        self.failure_count = 0
        self.degraded_threshold = 3
    
    def call_service(self, **kwargs) -> dict:
        """Call service with degraded mode."""
        
        if self.mode == ServiceMode.MAINTENANCE:
            return {
                "success": False,
                "error": "Service in maintenance mode",
                "error_type": "maintenance"
            }
        
        try:
            # Try full feature set
            if self.mode == ServiceMode.FULL:
                result = self._full_service(**kwargs)
                self.failure_count = 0
                return result
            
            # Degraded mode - basic features only
            else:
                result = self._degraded_service(**kwargs)
                result["warning"] = "Running in degraded mode - limited features"
                return result
                
        except Exception as e:
            self.failure_count += 1
            
            # Switch to degraded mode
            if self.failure_count >= self.degraded_threshold:
                self.mode = ServiceMode.DEGRADED
                print("Switched to DEGRADED mode")
                
                try:
                    result = self._degraded_service(**kwargs)
                    result["warning"] = "Service degraded due to errors"
                    return result
                except Exception as e2:
                    return {
                        "success": False,
                        "error": str(e2),
                        "error_type": "service_unavailable"
                    }
            
            return {
                "success": False,
                "error": str(e),
                "error_type": "exception"
            }
    
    def _full_service(self, **kwargs) -> dict:
        """Full service with all features."""
        # Implement full functionality
        return {"success": True, "data": "full", "mode": "full"}
    
    def _degraded_service(self, **kwargs) -> dict:
        """Degraded service with basic features."""
        # Implement basic functionality
        return {"success": True, "data": "basic", "mode": "degraded"}
```

---

## Timeout Handling

### Async Timeout

```python
import asyncio
from typing import Coroutine, Any

async def execute_with_timeout(
    coro: Coroutine,
    timeout_seconds: float = 30.0
) -> dict:
    """Execute coroutine with timeout."""
    try:
        result = await asyncio.wait_for(coro, timeout=timeout_seconds)
        return {"success": True, "result": result}
        
    except asyncio.TimeoutError:
        return {
            "success": False,
            "error": f"Operation timed out after {timeout_seconds}s",
            "error_type": "timeout",
            "severity": "medium",
            "suggestion": "Try again or increase timeout"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": "exception"
        }

# Usage
async def slow_operation():
    """Simulated slow operation."""
    await asyncio.sleep(5)
    return "completed"

result = await execute_with_timeout(
    slow_operation(),
    timeout_seconds=3.0
)
```

### Thread-based Timeout

```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Callable

def execute_with_timeout_sync(
    func: Callable,
    timeout_seconds: float = 30.0,
    **kwargs
) -> dict:
    """Execute function with timeout (sync)."""
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, **kwargs)
        
        try:
            result = future.result(timeout=timeout_seconds)
            return {"success": True, "result": result}
            
        except TimeoutError:
            return {
                "success": False,
                "error": f"Operation timed out after {timeout_seconds}s",
                "error_type": "timeout",
                "severity": "medium"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": "exception"
            }
```

---

## Error Recovery

### Compensation Pattern (Saga)

```python
@dataclass
class CompensatingAction:
    """Action to undo a step."""
    name: str
    func: Callable
    params: dict

class SagaExecutor:
    """Execute with compensation on failure."""
    
    def execute_saga(
        self,
        steps: List[tuple[Callable, dict, CompensatingAction]]
    ) -> dict:
        """
        Execute steps with compensation.
        
        Args:
            steps: List of (func, params, compensation) tuples
        """
        completed = []
        
        try:
            for func, params, compensation in steps:
                result = func(**params)
                
                if not result.get("success"):
                    # Step failed - rollback
                    print(f"Step {func.__name__} failed, rolling back...")
                    self._rollback(completed)
                    return {
                        "success": False,
                        "error": result.get("error"),
                        "rolled_back": len(completed)
                    }
                
                completed.append((result, compensation))
            
            return {
                "success": True,
                "steps_completed": len(completed)
            }
            
        except Exception as e:
            print(f"Exception during saga: {e}, rolling back...")
            self._rollback(completed)
            return {
                "success": False,
                "error": str(e),
                "rolled_back": len(completed)
            }
    
    def _rollback(self, completed: List):
        """Execute compensating actions in reverse."""
        for result, compensation in reversed(completed):
            try:
                print(f"Compensating: {compensation.name}")
                compensation.func(**compensation.params)
            except Exception as e:
                print(f"Compensation failed: {e}")
```

### Idempotent Operations

```python
from typing import Set
import uuid

class IdempotentExecutor:
    """Ensure operations are executed exactly once."""
    
    def __init__(self):
        self.executed: Set[str] = set()
    
    def execute_idempotent(
        self,
        operation_id: str,
        func: Callable,
        **kwargs
    ) -> dict:
        """Execute operation idempotently."""
        
        # Check if already executed
        if operation_id in self.executed:
            return {
                "success": True,
                "result": "already_executed",
                "idempotent": True
            }
        
        # Execute
        try:
            result = func(**kwargs)
            
            # Mark as executed on success
            if result.get("success"):
                self.executed.add(operation_id)
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": "exception"
            }
    
    def generate_operation_id(self, func_name: str, params: dict) -> str:
        """Generate unique operation ID."""
        # Use function name + params hash
        import hashlib
        import json
        
        params_str = json.dumps(params, sort_keys=True)
        hash_str = hashlib.md5(params_str.encode()).hexdigest()
        return f"{func_name}:{hash_str}"
```

---

## Monitoring & Alerting

### Error Rate Monitoring

```python
from collections import defaultdict, deque
from datetime import datetime, timedelta

class ErrorRateMonitor:
    """Monitor error rates and alert on thresholds."""
    
    def __init__(
        self,
        window_minutes: int = 5,
        alert_threshold: float = 0.5  # 50% error rate
    ):
        self.window = timedelta(minutes=window_minutes)
        self.alert_threshold = alert_threshold
        self.events = defaultdict(deque)  # tool_name -> events
    
    def record_event(self, tool_name: str, success: bool):
        """Record tool execution event."""
        event = {
            "timestamp": datetime.now(),
            "success": success
        }
        self.events[tool_name].append(event)
        
        # Clean old events
        self._clean_old_events(tool_name)
        
        # Check if alert needed
        if self._should_alert(tool_name):
            self._trigger_alert(tool_name)
    
    def _clean_old_events(self, tool_name: str):
        """Remove events outside window."""
        cutoff = datetime.now() - self.window
        events = self.events[tool_name]
        
        while events and events[0]["timestamp"] < cutoff:
            events.popleft()
    
    def _should_alert(self, tool_name: str) -> bool:
        """Check if error rate exceeds threshold."""
        events = self.events[tool_name]
        
        if len(events) < 10:  # Need minimum sample
            return False
        
        error_count = sum(1 for e in events if not e["success"])
        error_rate = error_count / len(events)
        
        return error_rate >= self.alert_threshold
    
    def _trigger_alert(self, tool_name: str):
        """Trigger alert for high error rate."""
        events = self.events[tool_name]
        error_count = sum(1 for e in events if not e["success"])
        error_rate = error_count / len(events)
        
        print(f"""
        ⚠️  HIGH ERROR RATE ALERT
        Tool: {tool_name}
        Error Rate: {error_rate:.1%}
        Sample Size: {len(events)}
        Window: {self.window.total_seconds()/60:.0f} minutes
        """)
    
    def get_stats(self, tool_name: str) -> dict:
        """Get error rate statistics."""
        events = self.events[tool_name]
        
        if not events:
            return {"no_data": True}
        
        total = len(events)
        errors = sum(1 for e in events if not e["success"])
        
        return {
            "tool_name": tool_name,
            "total_events": total,
            "error_count": errors,
            "error_rate": errors / total if total > 0 else 0,
            "window_minutes": self.window.total_seconds() / 60
        }
```

---

## Production Checklist

### Error Handling

- [ ] All errors return structured responses
- [ ] Error types are standardized
- [ ] Actionable suggestions provided
- [ ] Transient errors have retry logic
- [ ] Permanent errors fail fast
- [ ] User errors explain how to fix
- [ ] Error severity is indicated
- [ ] Retry-after times provided for rate limits

### Resilience

- [ ] Exponential backoff implemented
- [ ] Circuit breakers for external services
- [ ] Fallback strategies defined
- [ ] Timeouts configured on all operations
- [ ] Degraded mode available
- [ ] Idempotent operations where needed

### Monitoring

- [ ] Error rates tracked
- [ ] Latencies monitored
- [ ] Circuit breaker states logged
- [ ] Alerts configured for high error rates
- [ ] All errors logged with context
- [ ] Success/failure metrics collected

---

## Related Resources

- [function-calling-best-practices.md](./function-calling-best-practices.md) — Overall best practices
- [workflow-patterns-cheatsheet.md](./workflow-patterns-cheatsheet.md) — Workflow patterns
- `../lessons/01-advanced-function-calling-patterns.md` — Advanced patterns
- `../lessons/03-tool-orchestration-and-workflows.md` — Orchestration
