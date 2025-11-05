# Rate Limits, Retries & Backoff

## Understand Limits
- Per-model RPM/TPM limits (requests per minute, tokens per minute)
- Track usage: inspect response headers and SDK metadata where available

## Backoff Strategy (Tenacity)
```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class RetriableError(Exception):
    pass

@retry(reraise=True,
       stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=0.5, min=1, max=20),
       retry=retry_if_exception_type(RetriableError))
def call_openai_with_retry(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        # Map API errors to RetriableError where appropriate
        raise RetriableError(str(e))
```

## Idempotency Keys
- Use idempotency keys for non-idempotent operations
- Store key per logical request; retry safely

## Timeouts & Circuit Breakers
- Client timeouts (connect/read)
- Circuit breaker on repeated failures to avoid cascading issues

## Tips
- Batch embeddings (100–1000)
- Cache stable inputs/outputs
- Prefer smaller models for pre/post-processing
