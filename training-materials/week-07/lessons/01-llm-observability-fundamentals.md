# Lesson 1: LLM Observability Fundamentals

**Duration:** 90 minutes  
**Level:** Intermediate  
**Prerequisites:** Understanding of LLM applications, basic logging concepts

## Table of Contents
- [Introduction](#introduction)
- [Observability vs Monitoring](#observability-vs-monitoring)
- [The Three Pillars](#the-three-pillars)
- [LLM-Specific Observability Challenges](#llm-specific-observability-challenges)
- [Key Metrics for LLM Systems](#key-metrics-for-llm-systems)
- [Structured Logging](#structured-logging)
- [Correlation and Context](#correlation-and-context)
- [Alerting Strategies](#alerting-strategies)
- [Hands-On Examples](#hands-on-examples)
- [Best Practices](#best-practices)

---

## Introduction

Observability is the ability to understand the internal state of a system by examining its outputs. For LLM applications, this becomes critical as these systems are:

- **Non-deterministic**: Same input can produce different outputs
- **Expensive**: Token costs and latency matter
- **Complex**: Multi-step pipelines (RAG, agents, tools)
- **Black-box**: Model internals are opaque
- **High-stakes**: Errors can be costly or harmful

Without proper observability, you're flying blind.

### Why Observability Matters

```python
# Without observability
user_query = "What's the weather in Paris?"
response = llm.complete(user_query)
# ❌ What if it fails? Why is it slow? How much did it cost?

# With observability
with tracer.trace("weather_query") as trace:
    trace.set_tags({"user_id": "user_123", "intent": "weather"})
    response = llm.complete(user_query)
    trace.log_metrics({
        "latency_ms": 1234,
        "tokens_input": 12,
        "tokens_output": 45,
        "cost_usd": 0.0023
    })
# ✅ Full visibility into what happened
```

---

## Observability vs Monitoring

### Monitoring
- **Definition**: Watching known metrics and thresholds
- **Approach**: Reactive - alerts when thresholds are breached
- **Questions**: "Is it working?" "Are we within SLA?"
- **Tools**: Dashboards, alerts, uptime checks

### Observability
- **Definition**: Understanding system behavior through exploration
- **Approach**: Proactive - investigate unknown unknowns
- **Questions**: "Why did this fail?" "What caused the slowdown?"
- **Tools**: Traces, logs, metrics with rich context

### LLM Context

| Aspect | Monitoring | Observability |
|--------|-----------|---------------|
| Latency | Track P95 latency | Trace which component is slow |
| Errors | Count 5xx errors | Understand why LLM refused |
| Cost | Total daily spend | Per-user, per-query attribution |
| Quality | Success rate % | Analyze failed responses |

**Key Insight**: Monitoring tells you *what* is happening. Observability tells you *why*.

---

## The Three Pillars

### 1. Metrics
Quantitative measurements over time.

```python
from prometheus_client import Counter, Histogram, Gauge

# Counters (always increase)
llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM requests',
    ['model', 'status']
)

# Histograms (distribution of values)
llm_latency_seconds = Histogram(
    'llm_latency_seconds',
    'LLM request latency',
    ['model']
)

# Gauges (current value)
active_llm_requests = Gauge(
    'active_llm_requests',
    'Currently active LLM requests'
)
```

**LLM-Specific Metrics:**
- Request rate (requests/second)
- Latency (P50, P95, P99)
- Token usage (input/output)
- Cost per request
- Error rate
- Model selection distribution

### 2. Logs
Discrete events with context.

```python
import logging
import json
from datetime import datetime

# Structured logging
logger = logging.getLogger(__name__)

def log_llm_call(prompt, response, metadata):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": "llm_call",
        "prompt_length": len(prompt),
        "response_length": len(response),
        "model": metadata.get("model"),
        "latency_ms": metadata.get("latency_ms"),
        "tokens": {
            "input": metadata.get("input_tokens"),
            "output": metadata.get("output_tokens")
        },
        "cost_usd": metadata.get("cost"),
        "user_id": metadata.get("user_id"),
        "trace_id": metadata.get("trace_id")
    }
    logger.info(json.dumps(log_entry))
```

**LLM-Specific Logs:**
- Full prompts and responses (with PII handling)
- Tool/function calls and results
- Retrieval queries and chunks
- Error messages and stack traces
- User feedback and corrections

### 3. Traces
End-to-end request flows showing causality.

```python
# Example trace structure
{
    "trace_id": "abc123",
    "start_time": "2024-12-02T10:00:00Z",
    "end_time": "2024-12-02T10:00:02.5Z",
    "spans": [
        {
            "span_id": "span1",
            "name": "user_query",
            "start_time": "2024-12-02T10:00:00Z",
            "duration_ms": 2500,
            "children": ["span2", "span3"]
        },
        {
            "span_id": "span2",
            "parent_id": "span1",
            "name": "retrieve_documents",
            "start_time": "2024-12-02T10:00:00.1Z",
            "duration_ms": 800,
            "metadata": {
                "query": "What is RAG?",
                "num_results": 5,
                "vector_db": "chromadb"
            }
        },
        {
            "span_id": "span3",
            "parent_id": "span1",
            "name": "generate_response",
            "start_time": "2024-12-02T10:00:00.9Z",
            "duration_ms": 1600,
            "metadata": {
                "model": "gpt-4",
                "tokens_in": 450,
                "tokens_out": 180,
                "cost_usd": 0.0156
            }
        }
    ]
}
```

---

## LLM-Specific Observability Challenges

### 1. Non-Determinism
**Problem**: Same input → different outputs  
**Solution**: Log both input AND output, track patterns over time

```python
# Track output diversity
from collections import Counter

output_cache = {}

def track_variability(prompt, response):
    if prompt not in output_cache:
        output_cache[prompt] = []
    
    output_cache[prompt].append(response)
    
    # Alert on high variability
    unique_responses = len(set(output_cache[prompt]))
    if unique_responses > 5:
        alert(f"High output variability for prompt: {prompt}")
```

### 2. Latency Attribution
**Problem**: Which step is slow in multi-step pipelines?  
**Solution**: Detailed span-level tracing

```python
# Waterfall view of pipeline
async def rag_pipeline(query):
    with tracer.start_span("rag_pipeline") as root_span:
        # Step 1: Embedding
        with tracer.start_span("embed_query", parent=root_span):
            embedding = await embed(query)
        
        # Step 2: Retrieve
        with tracer.start_span("retrieve_docs", parent=root_span):
            docs = await vector_db.search(embedding, top_k=5)
        
        # Step 3: Rerank
        with tracer.start_span("rerank", parent=root_span):
            ranked_docs = await reranker.rerank(query, docs)
        
        # Step 4: Generate
        with tracer.start_span("llm_generate", parent=root_span):
            response = await llm.generate(query, ranked_docs)
        
        return response
```

### 3. Cost Attribution
**Problem**: Who/what is driving costs?  
**Solution**: Tag traces with user/team/feature identifiers

```python
def track_cost(user_id, team_id, feature, cost):
    cost_tracker.record({
        "user_id": user_id,
        "team_id": team_id,
        "feature": feature,
        "cost_usd": cost,
        "timestamp": datetime.utcnow()
    })

# Query later
monthly_cost_by_team = cost_tracker.aggregate(
    group_by="team_id",
    time_range="last_30_days"
)
```

### 4. Quality Monitoring
**Problem**: How do you know if outputs are good?  
**Solution**: Track user feedback, automated quality checks

```python
quality_metrics = {
    "user_satisfaction": [],  # 👍/👎 from users
    "regeneration_rate": 0,   # How often users retry
    "toxicity_score": 0,      # Automated toxicity check
    "factual_accuracy": 0     # Automated fact-checking
}

def log_user_feedback(query, response, feedback):
    quality_metrics["user_satisfaction"].append({
        "query": query,
        "response": response,
        "feedback": feedback,  # thumbs_up, thumbs_down, etc.
        "timestamp": datetime.utcnow()
    })
    
    # Calculate satisfaction rate
    positive = sum(1 for f in quality_metrics["user_satisfaction"] 
                   if f["feedback"] == "thumbs_up")
    total = len(quality_metrics["user_satisfaction"])
    satisfaction_rate = positive / total if total > 0 else 0
    
    return satisfaction_rate
```

---

## Key Metrics for LLM Systems

### Performance Metrics

```python
class LLMMetrics:
    """Track key LLM performance metrics."""
    
    def __init__(self):
        self.latency_histogram = []
        self.token_usage = {"input": [], "output": []}
        self.cost_tracker = []
    
    def record_request(self, duration_ms, tokens_in, tokens_out, cost):
        """Record a single LLM request."""
        self.latency_histogram.append(duration_ms)
        self.token_usage["input"].append(tokens_in)
        self.token_usage["output"].append(tokens_out)
        self.cost_tracker.append(cost)
    
    def get_summary(self):
        """Get summary statistics."""
        import numpy as np
        
        return {
            "latency": {
                "p50": np.percentile(self.latency_histogram, 50),
                "p95": np.percentile(self.latency_histogram, 95),
                "p99": np.percentile(self.latency_histogram, 99),
                "mean": np.mean(self.latency_histogram)
            },
            "tokens": {
                "input_total": sum(self.token_usage["input"]),
                "output_total": sum(self.token_usage["output"]),
                "avg_input": np.mean(self.token_usage["input"]),
                "avg_output": np.mean(self.token_usage["output"])
            },
            "cost": {
                "total_usd": sum(self.cost_tracker),
                "avg_per_request": np.mean(self.cost_tracker)
            },
            "throughput": {
                "requests_per_second": len(self.latency_histogram) / (max(self.latency_histogram) / 1000)
            }
        }
```

### Quality Metrics

```python
class QualityMetrics:
    """Track LLM output quality."""
    
    def __init__(self):
        self.responses = []
    
    def record_response(self, query, response, user_feedback=None, 
                       automated_checks=None):
        """Record response with quality signals."""
        self.responses.append({
            "query": query,
            "response": response,
            "user_feedback": user_feedback,
            "checks": automated_checks or {},
            "timestamp": datetime.utcnow()
        })
    
    def calculate_quality_score(self):
        """Aggregate quality score."""
        scores = []
        
        for resp in self.responses:
            score = 0
            
            # User feedback (if available)
            if resp["user_feedback"] == "positive":
                score += 1
            elif resp["user_feedback"] == "negative":
                score -= 1
            
            # Automated checks
            if resp["checks"].get("toxicity", 0) < 0.1:
                score += 0.5
            if resp["checks"].get("factuality", 0) > 0.8:
                score += 0.5
            
            scores.append(score)
        
        return {
            "avg_quality_score": np.mean(scores) if scores else 0,
            "positive_feedback_rate": sum(1 for r in self.responses 
                                         if r["user_feedback"] == "positive") / len(self.responses)
        }
```

### Business Metrics

```python
class BusinessMetrics:
    """Track business-relevant metrics."""
    
    def __init__(self):
        self.user_interactions = []
        self.feature_usage = {}
    
    def record_interaction(self, user_id, feature, duration_s, success):
        """Record user interaction."""
        self.user_interactions.append({
            "user_id": user_id,
            "feature": feature,
            "duration_s": duration_s,
            "success": success,
            "timestamp": datetime.utcnow()
        })
        
        # Track feature usage
        if feature not in self.feature_usage:
            self.feature_usage[feature] = {"total": 0, "success": 0}
        
        self.feature_usage[feature]["total"] += 1
        if success:
            self.feature_usage[feature]["success"] += 1
    
    def get_insights(self):
        """Get business insights."""
        return {
            "dau": len(set(i["user_id"] for i in self.user_interactions)),
            "feature_adoption": self.feature_usage,
            "success_rate": sum(i["success"] for i in self.user_interactions) / 
                          len(self.user_interactions),
            "avg_session_duration": np.mean([i["duration_s"] 
                                            for i in self.user_interactions])
        }
```

---

## Structured Logging

### Why Structured Logs?

**Unstructured** (hard to parse):
```python
logger.info(f"User {user_id} queried {query} and got response in {latency}ms")
```

**Structured** (machine-parseable):
```python
logger.info("llm_query", extra={
    "user_id": user_id,
    "query": query,
    "latency_ms": latency,
    "model": "gpt-4",
    "tokens_in": 150,
    "tokens_out": 75
})
```

### Implementation

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    """Structured logger for LLM applications."""
    
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(self.JSONFormatter())
        self.logger.addHandler(handler)
    
    class JSONFormatter(logging.Formatter):
        """Format logs as JSON."""
        
        def format(self, record):
            log_obj = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name
            }
            
            # Add extra fields
            if hasattr(record, "extra"):
                log_obj.update(record.extra)
            
            return json.dumps(log_obj)
    
    def log_llm_call(self, **kwargs):
        """Log an LLM API call."""
        self.logger.info("llm_call", extra=kwargs)
    
    def log_error(self, error, **context):
        """Log an error with context."""
        self.logger.error("error", extra={
            "error_type": type(error).__name__,
            "error_message": str(error),
            **context
        })

# Usage
logger = StructuredLogger("my_llm_app")

logger.log_llm_call(
    user_id="user_123",
    model="gpt-4",
    prompt_length=450,
    response_length=180,
    latency_ms=1234,
    cost_usd=0.0156
)
```

---

## Correlation and Context

### Correlation IDs

Track requests across services and components.

```python
import uuid
from contextvars import ContextVar

# Thread-safe context variable
trace_id_var = ContextVar("trace_id", default=None)

def generate_trace_id():
    """Generate new trace ID."""
    return str(uuid.uuid4())

def set_trace_id(trace_id=None):
    """Set trace ID for current context."""
    if trace_id is None:
        trace_id = generate_trace_id()
    trace_id_var.set(trace_id)
    return trace_id

def get_trace_id():
    """Get current trace ID."""
    trace_id = trace_id_var.get()
    if trace_id is None:
        trace_id = set_trace_id()
    return trace_id

# Middleware example
async def trace_middleware(request, call_next):
    """Add trace ID to all requests."""
    trace_id = request.headers.get("X-Trace-ID", generate_trace_id())
    set_trace_id(trace_id)
    
    response = await call_next(request)
    response.headers["X-Trace-ID"] = trace_id
    
    return response
```

### Context Propagation

```python
class RequestContext:
    """Context for LLM requests."""
    
    def __init__(self, user_id, session_id, feature):
        self.trace_id = generate_trace_id()
        self.user_id = user_id
        self.session_id = session_id
        self.feature = feature
        self.start_time = datetime.utcnow()
    
    def to_dict(self):
        """Convert to dictionary for logging."""
        return {
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "feature": self.feature,
            "timestamp": self.start_time.isoformat()
        }
    
    def log_with_context(self, logger, message, **extra):
        """Log with full context."""
        logger.info(message, extra={
            **self.to_dict(),
            **extra
        })

# Usage
ctx = RequestContext(
    user_id="user_123",
    session_id="session_456",
    feature="chatbot"
)

ctx.log_with_context(logger, "llm_query", 
                     query="What is RAG?",
                     latency_ms=1234)
```

---

## Alerting Strategies

### Alert Types

1. **Threshold Alerts**: Metric exceeds threshold
2. **Anomaly Alerts**: Unusual pattern detected
3. **Error Rate Alerts**: Error rate spikes
4. **Cost Alerts**: Spending exceeds budget

### Implementation

```python
class AlertManager:
    """Manage alerts for LLM systems."""
    
    def __init__(self):
        self.thresholds = {
            "latency_p95_ms": 5000,
            "error_rate": 0.05,
            "cost_per_hour_usd": 100,
            "token_usage_per_min": 100000
        }
        self.alert_handlers = []
    
    def register_handler(self, handler):
        """Register alert handler (email, Slack, PagerDuty)."""
        self.alert_handlers.append(handler)
    
    def check_latency(self, latencies):
        """Check if latency exceeds threshold."""
        p95 = np.percentile(latencies, 95)
        if p95 > self.thresholds["latency_p95_ms"]:
            self.trigger_alert(
                "high_latency",
                f"P95 latency {p95}ms exceeds threshold {self.thresholds['latency_p95_ms']}ms"
            )
    
    def check_error_rate(self, total_requests, error_count):
        """Check if error rate is too high."""
        error_rate = error_count / total_requests if total_requests > 0 else 0
        if error_rate > self.thresholds["error_rate"]:
            self.trigger_alert(
                "high_error_rate",
                f"Error rate {error_rate:.2%} exceeds threshold {self.thresholds['error_rate']:.2%}"
            )
    
    def check_cost(self, hourly_cost):
        """Check if cost exceeds budget."""
        if hourly_cost > self.thresholds["cost_per_hour_usd"]:
            self.trigger_alert(
                "high_cost",
                f"Hourly cost ${hourly_cost:.2f} exceeds budget ${self.thresholds['cost_per_hour_usd']:.2f}"
            )
    
    def trigger_alert(self, alert_type, message):
        """Trigger alert to all handlers."""
        for handler in self.alert_handlers:
            handler.send_alert(alert_type, message)

# Handlers
class SlackAlertHandler:
    """Send alerts to Slack."""
    
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
    
    def send_alert(self, alert_type, message):
        """Send alert to Slack."""
        import requests
        requests.post(self.webhook_url, json={
            "text": f"🚨 {alert_type.upper()}: {message}"
        })

# Usage
alert_manager = AlertManager()
alert_manager.register_handler(SlackAlertHandler("https://hooks.slack.com/..."))

# Check metrics periodically
alert_manager.check_latency(recent_latencies)
alert_manager.check_error_rate(total_requests, error_count)
alert_manager.check_cost(last_hour_cost)
```

### Anomaly Detection

```python
from scipy import stats

class AnomalyDetector:
    """Detect anomalies in time series data."""
    
    def __init__(self, window_size=100, std_threshold=3):
        self.window_size = window_size
        self.std_threshold = std_threshold
        self.history = []
    
    def add_value(self, value):
        """Add new value and check for anomaly."""
        self.history.append(value)
        
        # Keep only recent history
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]
        
        # Need enough history
        if len(self.history) < 10:
            return False
        
        # Calculate z-score
        mean = np.mean(self.history[:-1])  # Exclude current value
        std = np.std(self.history[:-1])
        
        if std == 0:
            return False
        
        z_score = (value - mean) / std
        
        return abs(z_score) > self.std_threshold

# Usage
latency_detector = AnomalyDetector()

for latency in stream_of_latencies:
    is_anomaly = latency_detector.add_value(latency)
    if is_anomaly:
        alert_manager.trigger_alert(
            "latency_anomaly",
            f"Unusual latency detected: {latency}ms"
        )
```

---

## Hands-On Examples

### Example 1: Complete Observability Wrapper

```python
import time
import functools
from typing import Any, Callable

class ObservabilityWrapper:
    """Wrap functions with observability."""
    
    def __init__(self, logger, metrics, tracer):
        self.logger = logger
        self.metrics = metrics
        self.tracer = tracer
    
    def observe(self, operation_name: str):
        """Decorator to add observability."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                # Start trace
                with self.tracer.start_span(operation_name) as span:
                    start_time = time.time()
                    
                    try:
                        # Execute function
                        result = await func(*args, **kwargs)
                        
                        # Record success
                        duration_ms = (time.time() - start_time) * 1000
                        
                        self.metrics.record_request(
                            duration_ms=duration_ms,
                            tokens_in=result.get("tokens_in", 0),
                            tokens_out=result.get("tokens_out", 0),
                            cost=result.get("cost", 0)
                        )
                        
                        self.logger.log_llm_call(
                            operation=operation_name,
                            duration_ms=duration_ms,
                            status="success",
                            trace_id=span.trace_id
                        )
                        
                        return result
                    
                    except Exception as e:
                        # Record error
                        self.logger.log_error(e,
                            operation=operation_name,
                            trace_id=span.trace_id
                        )
                        raise
            
            return async_wrapper
        return decorator

# Usage
obs = ObservabilityWrapper(logger, metrics, tracer)

@obs.observe("rag_query")
async def rag_query(query: str):
    # Retrieve
    docs = await retrieve(query)
    
    # Generate
    response = await generate(query, docs)
    
    return response
```

### Example 2: Dashboard Data Collection

```python
class DashboardCollector:
    """Collect data for observability dashboard."""
    
    def __init__(self):
        self.data = {
            "requests_per_minute": [],
            "latency_history": [],
            "error_history": [],
            "cost_history": [],
            "top_users": {},
            "top_features": {}
        }
    
    def collect_snapshot(self, metrics):
        """Collect current metrics snapshot."""
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics.get_summary()
        }
        
        # Add to histories
        self.data["latency_history"].append({
            "time": snapshot["timestamp"],
            "p95": snapshot["metrics"]["latency"]["p95"]
        })
        
        self.data["cost_history"].append({
            "time": snapshot["timestamp"],
            "cost": snapshot["metrics"]["cost"]["total_usd"]
        })
        
        return snapshot
    
    def export_dashboard_data(self):
        """Export data for dashboard visualization."""
        return {
            "overview": {
                "total_requests": len(metrics.latency_histogram),
                "avg_latency_ms": np.mean(metrics.latency_histogram),
                "total_cost_usd": sum(metrics.cost_tracker),
                "error_rate": calculate_error_rate()
            },
            "timeseries": {
                "latency": self.data["latency_history"],
                "cost": self.data["cost_history"]
            },
            "top_users": sorted(
                self.data["top_users"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
```

---

## Best Practices

### 1. Instrument Early
Start with observability from day 1, not after problems arise.

### 2. Use Consistent Naming
```python
# Good
span_names = [
    "llm.completion",
    "vector_db.search",
    "reranker.rerank"
]

# Bad - inconsistent
span_names = [
    "CallLLM",
    "search_vectors",
    "rerank-docs"
]
```

### 3. Balance Verbosity and Performance
```python
# Don't log everything
if logging.level >= logging.DEBUG:
    logger.debug("Full prompt", extra={"prompt": full_prompt})
else:
    logger.info("Prompt summary", extra={"length": len(full_prompt)})
```

### 4. Tag Intelligently
```python
# Good tags
tags = {
    "user_id": "user_123",
    "team": "sales",
    "environment": "production",
    "model": "gpt-4",
    "feature": "chatbot"
}
```

### 5. Handle PII Carefully
```python
def sanitize_for_logging(text):
    """Remove PII before logging."""
    # Redact emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                  '[EMAIL]', text)
    
    # Redact phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    
    # Redact SSNs
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    
    return text
```

### 6. Set Up Sampling
```python
class SampledLogger:
    """Log only a sample of requests."""
    
    def __init__(self, sample_rate=0.1):
        self.sample_rate = sample_rate
    
    def should_log(self):
        """Decide if this request should be logged."""
        return random.random() < self.sample_rate
    
    def log_if_sampled(self, *args, **kwargs):
        """Log only if sampled."""
        if self.should_log():
            logger.info(*args, **kwargs)
```

### 7. Use Async Logging
```python
import asyncio
from queue import Queue
from threading import Thread

class AsyncLogger:
    """Non-blocking logger."""
    
    def __init__(self):
        self.queue = Queue()
        self.thread = Thread(target=self._process_queue, daemon=True)
        self.thread.start()
    
    def _process_queue(self):
        """Process log queue in background."""
        while True:
            log_entry = self.queue.get()
            if log_entry is None:
                break
            
            # Actual logging
            logger.info(json.dumps(log_entry))
    
    def log(self, **kwargs):
        """Add to queue (non-blocking)."""
        self.queue.put(kwargs)
```

---

## Summary

### Key Takeaways

1. **Observability ≠ Monitoring**: Observability helps you understand *why*, not just *what*
2. **Three Pillars**: Metrics, Logs, Traces - use all three
3. **LLM-Specific Challenges**: Non-determinism, latency attribution, cost tracking
4. **Structured Logging**: Machine-parseable logs enable better analysis
5. **Context is King**: Correlation IDs and rich metadata are essential
6. **Alert Thoughtfully**: Balance signal vs. noise

### Checklist

- [ ] Implement structured logging
- [ ] Add correlation IDs to all requests
- [ ] Track latency, tokens, and cost
- [ ] Set up basic alerts (latency, errors, cost)
- [ ] Create initial dashboard
- [ ] Handle PII in logs
- [ ] Test observability under load

### Next Steps

In Lesson 2, we'll dive deep into **distributed tracing with Langfuse**, learning how to:
- Set up Langfuse for production
- Instrument complex multi-step pipelines
- Create custom dashboards
- Analyze traces for optimization
- Integrate with OpenTelemetry

---

## Additional Resources

- [OpenTelemetry Best Practices](https://opentelemetry.io/docs/concepts/observability-primer/)
- [Prometheus Metric Types](https://prometheus.io/docs/concepts/metric_types/)
- [Structured Logging with Python](https://docs.python.org/3/howto/logging-cookbook.html)
- [Observability Engineering (O'Reilly Book)](https://www.oreilly.com/library/view/observability-engineering/9781492076438/)
