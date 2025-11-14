# Error Handling Demo

Comprehensive error handling patterns for production tool systems.

## Features

✅ **Retry Strategies**
- Exponential backoff with jitter
- Tenacity decorators
- Configurable attempts
- Retry conditions

✅ **Circuit Breakers**
- State machine (CLOSED/OPEN/HALF_OPEN)
- Failure threshold tracking
- Automatic recovery
- Health checks

✅ **Fallback Chains**
- Primary → Secondary → Cache
- Degraded service modes
- Graceful degradation

✅ **Timeout Handling**
- Async timeouts
- Sync timeouts with threading
- Context managers
- Cleanup on timeout

✅ **Saga Pattern**
- Compensating actions
- Rollback on failure
- Transaction coordination

✅ **Error Recovery**
- Idempotent operations
- Checkpoint/resume
- State persistence

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all demos
python main.py

# Specific pattern
python main.py --pattern retry
python main.py --pattern circuit-breaker
python main.py --pattern fallback
python main.py --pattern saga
```

## Patterns Demonstrated

### 1. Retry with Backoff
```python
@exponential_backoff_retry(max_attempts=3)
def unreliable_operation():
    # May fail, will retry
    pass
```

### 2. Circuit Breaker
```python
circuit = CircuitBreaker(failure_threshold=5)

@circuit.call
def external_service():
    # Protected by circuit breaker
    pass
```

### 3. Fallback Chain
```python
executor = FallbackExecutor([
    primary_service,
    secondary_service,
    cache_service
])
result = executor.execute()
```

### 4. Saga Pattern
```python
saga = SagaExecutor()
saga.add_step(action1, compensate1)
saga.add_step(action2, compensate2)
saga.execute()  # Rolls back on failure
```

## Related Resources

- Week 6 Lab 1: Parallel Function Calling & Error Handling
- `../../error-handling-patterns.md`
- `../../function-calling-best-practices.md`
