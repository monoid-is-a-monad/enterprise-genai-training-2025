# Lesson 2: Distributed Tracing with Langfuse

**Duration:** 120 minutes  
**Level:** Intermediate  
**Prerequisites:** Lesson 1 (Observability Fundamentals), Understanding of async Python

## Table of Contents
- [Introduction to Langfuse](#introduction-to-langfuse)
- [Trace and Span Model](#trace-and-span-model)
- [Setting Up Langfuse](#setting-up-langfuse)
- [Basic Instrumentation](#basic-instrumentation)
- [Advanced Tracing Patterns](#advanced-tracing-patterns)
- [Cost Tracking and Attribution](#cost-tracking-and-attribution)
- [Dashboard and Analytics](#dashboard-and-analytics)
- [Performance Optimization](#performance-optimization)
- [Integration with LLM Frameworks](#integration-with-llm-frameworks)
- [Production Best Practices](#production-best-practices)

---

## Introduction to Langfuse

**Langfuse** is an open-source observability and analytics platform specifically designed for LLM applications. It provides:

- 🔍 **Distributed Tracing**: Track requests across components
- 💰 **Cost Tracking**: Attribute costs to users/teams/features
- 📊 **Analytics**: Dashboards and insights
- 🐛 **Debugging**: Inspect failed requests
- 📈 **Performance**: Identify bottlenecks
- 🎯 **Evaluation**: Quality metrics and scoring

### Why Langfuse?

```python
# Without tracing - black box
response = llm.complete(prompt)
# ❌ Can't see: retrieval → reranking → generation flow
# ❌ Can't attribute cost to specific user
# ❌ Can't identify which step is slow

# With Langfuse - full visibility
from langfuse import Langfuse

langfuse = Langfuse()

trace = langfuse.trace(name="rag_query", user_id="user_123")

with trace.span(name="retrieve") as span:
    docs = retrieve(query)
    span.end(metadata={"num_docs": len(docs)})

with trace.span(name="generate") as span:
    response = llm.complete(prompt)
    span.end(output=response, metadata={"tokens": 150})

# ✅ Complete visibility
# ✅ Cost attributed to user_123
# ✅ Can see retrieval took 200ms, generation 1500ms
```

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Your Application                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ Retrieve │→ │  Rerank  │→ │ Generate │             │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘             │
│       │             │             │                     │
│       └─────────────┴─────────────┘                     │
│                     │                                    │
│              Langfuse SDK                                │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
           ┌─────────────────────┐
           │  Langfuse Backend   │
           │  (Cloud or Self-    │
           │   Hosted)           │
           └─────────────────────┘
                       │
                       ▼
           ┌─────────────────────┐
           │  Langfuse Dashboard │
           │  - Traces           │
           │  - Analytics        │
           │  - Costs            │
           └─────────────────────┘
```

---

## Trace and Span Model

### Hierarchical Structure

```
Trace (user_query)                    [2500ms, $0.025]
├── Span: embed_query                 [50ms, $0.0001]
├── Span: retrieve_documents          [200ms, $0.001]
│   ├── Span: vector_search           [150ms, $0.001]
│   └── Span: rerank                  [50ms, $0.0]
└── Span: generate_response           [2250ms, $0.024]
    ├── Span: build_prompt            [10ms, $0.0]
    └── Span: llm_call                [2240ms, $0.024]
```

### Core Concepts

**Trace**: End-to-end request (e.g., user query → response)
- Has unique trace_id
- Contains multiple spans
- Tracks overall latency and cost
- Associated with user/session

**Span**: Individual operation within a trace
- Has parent span (except root)
- Represents a single step
- Has start/end time
- Contains metadata, input, output

**Generation**: Special span for LLM calls
- Tracks model, tokens, cost
- Captures prompt and completion
- Can be scored/evaluated

### Data Model

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List

@dataclass
class Span:
    """Represents a span in a trace."""
    id: str
    trace_id: str
    parent_span_id: Optional[str]
    name: str
    start_time: datetime
    end_time: Optional[datetime]
    metadata: Dict[str, Any]
    input: Optional[Any]
    output: Optional[Any]
    level: str  # "DEFAULT", "WARNING", "ERROR"
    status_message: Optional[str]
    
@dataclass
class Generation(Span):
    """Special span for LLM generations."""
    model: str
    prompt: str
    completion: str
    usage: Dict[str, int]  # prompt_tokens, completion_tokens, total_tokens
    cost: float

@dataclass
class Trace:
    """Represents a complete trace."""
    id: str
    name: str
    user_id: Optional[str]
    session_id: Optional[str]
    metadata: Dict[str, Any]
    tags: List[str]
    spans: List[Span]
    release: Optional[str]
    
    @property
    def total_cost(self) -> float:
        """Calculate total cost across all spans."""
        return sum(s.cost for s in self.spans if isinstance(s, Generation))
    
    @property
    def duration_ms(self) -> float:
        """Calculate total duration."""
        if not self.spans:
            return 0.0
        start = min(s.start_time for s in self.spans)
        end = max(s.end_time for s in self.spans if s.end_time)
        return (end - start).total_seconds() * 1000
```

---

## Setting Up Langfuse

### Cloud Setup (Recommended for Learning)

```python
# 1. Sign up at https://cloud.langfuse.com
# 2. Create a new project
# 3. Get API keys from Settings

# Install
pip install langfuse

# Configure
import os
from langfuse import Langfuse

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

# Test connection
langfuse.auth_check()  # Should return True
```

### Self-Hosted Setup (Docker)

```bash
# Clone repository
git clone https://github.com/langfuse/langfuse.git
cd langfuse

# Start with Docker Compose
docker-compose up -d

# Access at http://localhost:3000
# Default credentials: admin@example.com / password
```

### Environment Variables

```bash
# .env file
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_HOST=https://cloud.langfuse.com

# Optional: Enable debug mode
LANGFUSE_DEBUG=true

# Optional: Configure flushing
LANGFUSE_FLUSH_AT=10          # Flush after N events
LANGFUSE_FLUSH_INTERVAL=1000  # Flush every N ms
```

---

## Basic Instrumentation

### Simple Trace

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Create a trace
trace = langfuse.trace(
    name="simple_query",
    user_id="user_123",
    metadata={"environment": "production"}
)

# Add a span
span = trace.span(
    name="process_query",
    input={"query": "What is RAG?"}
)

# Do work...
result = process_query("What is RAG?")

# End span with output
span.end(output=result)

# Trace is automatically flushed
```

### LLM Generation

```python
from openai import OpenAI

client = OpenAI()

# Start trace
trace = langfuse.trace(name="llm_query", user_id="user_123")

# Track generation
generation = trace.generation(
    name="openai_completion",
    model="gpt-4",
    input={"messages": [{"role": "user", "content": "Hello!"}]}
)

# Call OpenAI
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

# End generation with metadata
generation.end(
    output=response.choices[0].message.content,
    metadata={
        "finish_reason": response.choices[0].finish_reason,
        "model": response.model
    },
    usage={
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }
)
```

### Error Handling

```python
trace = langfuse.trace(name="error_example")

try:
    span = trace.span(name="risky_operation")
    result = risky_operation()
    span.end(output=result, level="DEFAULT")
    
except Exception as e:
    # Log error
    span.end(
        level="ERROR",
        status_message=str(e),
        metadata={
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
    )
    raise
```

---

## Advanced Tracing Patterns

### RAG Pipeline Tracing

```python
async def traced_rag_pipeline(query: str, user_id: str):
    """Complete RAG pipeline with tracing."""
    
    # Create trace
    trace = langfuse.trace(
        name="rag_pipeline",
        user_id=user_id,
        input={"query": query},
        metadata={"version": "v2.1"}
    )
    
    try:
        # Step 1: Embed query
        embed_span = trace.span(name="embed_query", input=query)
        embedding = await embed_query(query)
        embed_span.end(
            output={"dimensions": len(embedding)},
            metadata={"model": "text-embedding-3-small"}
        )
        
        # Step 2: Retrieve documents
        retrieve_span = trace.span(
            name="retrieve_documents",
            input={"query": query, "top_k": 5}
        )
        
        # Sub-span: Vector search
        vector_span = retrieve_span.span(name="vector_search")
        docs = await vector_db.search(embedding, top_k=10)
        vector_span.end(
            output={"num_docs": len(docs)},
            metadata={"index": "main_index"}
        )
        
        # Sub-span: Rerank
        rerank_span = retrieve_span.span(name="rerank")
        ranked_docs = await reranker.rerank(query, docs, top_k=5)
        rerank_span.end(
            output={"num_docs": len(ranked_docs)},
            metadata={"reranker": "cross-encoder"}
        )
        
        retrieve_span.end(output={"documents": ranked_docs})
        
        # Step 3: Generate response
        gen = trace.generation(
            name="generate_response",
            model="gpt-4",
            input={
                "query": query,
                "context": ranked_docs
            }
        )
        
        response = await llm_generate(query, ranked_docs)
        
        gen.end(
            output=response["text"],
            usage={
                "prompt_tokens": response["prompt_tokens"],
                "completion_tokens": response["completion_tokens"],
                "total_tokens": response["total_tokens"]
            },
            metadata={
                "temperature": 0.7,
                "max_tokens": 500
            }
        )
        
        # Complete trace
        trace.update(output=response["text"])
        
        return response["text"]
    
    except Exception as e:
        trace.update(
            level="ERROR",
            status_message=str(e)
        )
        raise
```

### Multi-Agent Tracing

```python
async def traced_multi_agent(task: str, user_id: str):
    """Multi-agent system with tracing."""
    
    trace = langfuse.trace(
        name="multi_agent_task",
        user_id=user_id,
        input={"task": task}
    )
    
    # Agent 1: Planner
    planner_span = trace.span(name="planner_agent")
    plan = await planner_agent.plan(task)
    planner_span.end(output={"steps": plan})
    
    # Execute steps in parallel
    results = []
    for i, step in enumerate(plan):
        step_span = trace.span(
            name=f"execute_step_{i}",
            input={"step": step}
        )
        
        # Choose executor agent
        if step["type"] == "search":
            result = await search_agent.execute(step, parent_span=step_span)
        elif step["type"] == "calculate":
            result = await calculator_agent.execute(step, parent_span=step_span)
        else:
            result = await general_agent.execute(step, parent_span=step_span)
        
        step_span.end(output=result)
        results.append(result)
    
    # Agent 2: Synthesizer
    synth_span = trace.span(name="synthesizer_agent")
    final_answer = await synthesizer_agent.synthesize(task, results)
    synth_span.end(output=final_answer)
    
    trace.update(output=final_answer)
    
    return final_answer
```

### Nested Traces (Sub-traces)

```python
def main_workflow(user_id: str):
    """Main workflow that spawns sub-workflows."""
    
    # Main trace
    main_trace = langfuse.trace(
        name="main_workflow",
        user_id=user_id
    )
    
    # Sub-trace 1: Data processing
    data_trace = langfuse.trace(
        name="data_processing",
        parent_observation_id=main_trace.id  # Link to parent
    )
    process_data(data_trace)
    
    # Sub-trace 2: Model inference
    inference_trace = langfuse.trace(
        name="model_inference",
        parent_observation_id=main_trace.id
    )
    run_inference(inference_trace)
    
    main_trace.update(output="Workflow complete")
```

### Decorator Pattern

```python
from functools import wraps
from typing import Callable, Any

def trace_function(name: str = None):
    """Decorator to automatically trace functions."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Get or create trace
            trace = get_current_trace()
            if trace is None:
                trace = langfuse.trace(name=name or func.__name__)
            
            # Create span
            span = trace.span(
                name=name or func.__name__,
                input={"args": args, "kwargs": kwargs}
            )
            
            try:
                result = await func(*args, **kwargs)
                span.end(output=result)
                return result
            except Exception as e:
                span.end(
                    level="ERROR",
                    status_message=str(e)
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            trace = get_current_trace()
            if trace is None:
                trace = langfuse.trace(name=name or func.__name__)
            
            span = trace.span(
                name=name or func.__name__,
                input={"args": args, "kwargs": kwargs}
            )
            
            try:
                result = func(*args, **kwargs)
                span.end(output=result)
                return result
            except Exception as e:
                span.end(level="ERROR", status_message=str(e))
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

# Usage
@trace_function("retrieve_documents")
async def retrieve_documents(query: str):
    return await vector_db.search(query)

@trace_function("generate_response")
async def generate_response(query: str, docs: list):
    return await llm.generate(query, docs)
```

---

## Cost Tracking and Attribution

### Automatic Cost Calculation

```python
# Configure pricing
PRICING = {
    "gpt-4": {
        "input": 0.03 / 1000,   # $0.03 per 1K tokens
        "output": 0.06 / 1000   # $0.06 per 1K tokens
    },
    "gpt-3.5-turbo": {
        "input": 0.001 / 1000,
        "output": 0.002 / 1000
    },
    "text-embedding-3-small": {
        "input": 0.00002 / 1000,
        "output": 0.0
    }
}

def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost for LLM call."""
    pricing = PRICING.get(model, {"input": 0, "output": 0})
    
    input_cost = prompt_tokens * pricing["input"]
    output_cost = completion_tokens * pricing["output"]
    
    return input_cost + output_cost

# Use in generation
generation = trace.generation(
    name="llm_call",
    model="gpt-4",
    input=prompt
)

response = await llm.complete(prompt)

cost = calculate_cost(
    model="gpt-4",
    prompt_tokens=response.usage.prompt_tokens,
    completion_tokens=response.usage.completion_tokens
)

generation.end(
    output=response.text,
    usage={
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    },
    metadata={"cost_usd": cost}
)
```

### User Attribution

```python
def query_with_attribution(query: str, user_id: str, team_id: str, feature: str):
    """Track costs by user, team, and feature."""
    
    trace = langfuse.trace(
        name="attributed_query",
        user_id=user_id,
        metadata={
            "team_id": team_id,
            "feature": feature,
            "environment": "production"
        },
        tags=[team_id, feature]
    )
    
    # ... execute query ...
    
    # Costs are automatically attributed to user_id and metadata
```

### Cost Analysis Queries

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Get traces for analysis
traces = langfuse.get_traces(
    user_id="user_123",
    from_timestamp=datetime(2024, 12, 1),
    to_timestamp=datetime(2024, 12, 2)
)

# Calculate total cost per user
user_costs = {}
for trace in traces:
    user_id = trace.user_id
    total_cost = sum(
        gen.calculated_total_cost or 0 
        for gen in trace.observations 
        if gen.type == "GENERATION"
    )
    user_costs[user_id] = user_costs.get(user_id, 0) + total_cost

print(f"Total cost per user: {user_costs}")

# Cost by feature
feature_costs = {}
for trace in traces:
    feature = trace.metadata.get("feature", "unknown")
    total_cost = sum(
        gen.calculated_total_cost or 0 
        for gen in trace.observations 
        if gen.type == "GENERATION"
    )
    feature_costs[feature] = feature_costs.get(feature, 0) + total_cost

print(f"Total cost per feature: {feature_costs}")
```

---

## Dashboard and Analytics

### Key Dashboard Views

**1. Traces View**
- List of all traces
- Filter by user, time range, tags
- Sort by cost, latency, status
- Search by trace ID or content

**2. Generations View**
- All LLM calls
- Model distribution
- Token usage statistics
- Cost breakdown

**3. Sessions View**
- Group traces by session
- User journey analysis
- Session duration and cost

**4. Users View**
- Per-user analytics
- Top users by volume/cost
- User segments

### Custom Dashboards

```python
# Create dashboard data
from langfuse import Langfuse

langfuse = Langfuse()

def create_dashboard_metrics(start_date, end_date):
    """Generate metrics for custom dashboard."""
    
    traces = langfuse.get_traces(
        from_timestamp=start_date,
        to_timestamp=end_date
    )
    
    metrics = {
        "total_requests": len(traces),
        "total_cost": 0,
        "avg_latency_ms": 0,
        "error_rate": 0,
        "model_distribution": {},
        "feature_usage": {},
        "top_users": {}
    }
    
    latencies = []
    errors = 0
    
    for trace in traces:
        # Cost
        trace_cost = sum(
            gen.calculated_total_cost or 0 
            for gen in trace.observations 
            if gen.type == "GENERATION"
        )
        metrics["total_cost"] += trace_cost
        
        # Latency
        if trace.latency:
            latencies.append(trace.latency)
        
        # Errors
        if trace.level == "ERROR":
            errors += 1
        
        # Model distribution
        for obs in trace.observations:
            if obs.type == "GENERATION":
                model = obs.model or "unknown"
                metrics["model_distribution"][model] = \
                    metrics["model_distribution"].get(model, 0) + 1
        
        # Feature usage
        feature = trace.metadata.get("feature", "unknown")
        metrics["feature_usage"][feature] = \
            metrics["feature_usage"].get(feature, 0) + 1
        
        # Top users
        user_id = trace.user_id or "anonymous"
        if user_id not in metrics["top_users"]:
            metrics["top_users"][user_id] = {"requests": 0, "cost": 0}
        metrics["top_users"][user_id]["requests"] += 1
        metrics["top_users"][user_id]["cost"] += trace_cost
    
    # Calculate averages
    if latencies:
        metrics["avg_latency_ms"] = sum(latencies) / len(latencies)
    
    if metrics["total_requests"] > 0:
        metrics["error_rate"] = errors / metrics["total_requests"]
    
    # Sort top users
    metrics["top_users"] = dict(
        sorted(
            metrics["top_users"].items(),
            key=lambda x: x[1]["cost"],
            reverse=True
        )[:10]
    )
    
    return metrics
```

### Scores and Evaluations

```python
# Add scores to traces
trace = langfuse.trace(name="scored_query")

# ... execute query ...

# Score the trace
trace.score(
    name="user_satisfaction",
    value=0.9,  # 0-1 score
    comment="User clicked thumbs up"
)

trace.score(
    name="response_quality",
    value=0.85,
    comment="Automated quality check"
)

# Query scores
traces_with_scores = langfuse.get_traces(
    filter={
        "scores": {
            "user_satisfaction": {"gte": 0.8}
        }
    }
)
```

---

## Performance Optimization

### Analyzing Traces for Bottlenecks

```python
def analyze_trace_performance(trace_id: str):
    """Analyze trace to find bottlenecks."""
    
    trace = langfuse.get_trace(trace_id)
    
    # Get all spans
    spans = trace.observations
    
    # Sort by duration
    sorted_spans = sorted(
        spans,
        key=lambda s: s.latency or 0,
        reverse=True
    )
    
    print("Top 5 slowest operations:")
    for span in sorted_spans[:5]:
        print(f"  {span.name}: {span.latency}ms")
    
    # Calculate percentage of total time
    total_time = trace.latency
    print(f"\nPercentage of total time:")
    for span in sorted_spans[:5]:
        pct = (span.latency / total_time) * 100
        print(f"  {span.name}: {pct:.1f}%")
    
    # Identify sequential vs parallel opportunities
    print("\nOptimization suggestions:")
    if any("retrieve" in s.name.lower() for s in spans):
        print("  - Consider parallel retrieval from multiple sources")
    if any("embed" in s.name.lower() for s in spans):
        print("  - Consider caching embeddings")
    if any(s.latency > 2000 for s in spans if "llm" in s.name.lower()):
        print("  - LLM calls are slow, consider:")
        print("    * Using a faster model")
        print("    * Reducing context length")
        print("    * Implementing streaming")
```

### Caching Strategy

```python
import hashlib
from functools import wraps

cache = {}

def cached_with_tracing(cache_key_func):
    """Cache results and track cache hits in traces."""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get trace
            trace = get_current_trace()
            
            # Calculate cache key
            cache_key = cache_key_func(*args, **kwargs)
            key_hash = hashlib.md5(cache_key.encode()).hexdigest()
            
            # Check cache
            if key_hash in cache:
                # Cache hit
                span = trace.span(
                    name=f"{func.__name__}_cached",
                    metadata={"cache": "hit"}
                )
                result = cache[key_hash]
                span.end(output=result)
                return result
            
            # Cache miss - execute function
            span = trace.span(
                name=func.__name__,
                metadata={"cache": "miss"}
            )
            
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache[key_hash] = result
            
            span.end(output=result)
            return result
        
        return wrapper
    return decorator

# Usage
@cached_with_tracing(lambda query: query)
async def embed_query(query: str):
    return await embedder.embed(query)
```

---

## Integration with LLM Frameworks

### LangChain Integration

```python
from langchain.callbacks.base import BaseCallbackHandler
from langfuse import Langfuse

class LangfuseCallbackHandler(BaseCallbackHandler):
    """Langfuse callback for LangChain."""
    
    def __init__(self, langfuse: Langfuse, trace_name: str):
        self.langfuse = langfuse
        self.trace = langfuse.trace(name=trace_name)
        self.span_stack = []
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        """Start a chain."""
        span = self.trace.span(
            name=serialized.get("name", "chain"),
            input=inputs
        )
        self.span_stack.append(span)
    
    def on_chain_end(self, outputs, **kwargs):
        """End a chain."""
        if self.span_stack:
            span = self.span_stack.pop()
            span.end(output=outputs)
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Start an LLM call."""
        generation = self.trace.generation(
            name="llm_call",
            model=serialized.get("model_name", "unknown"),
            input={"prompts": prompts}
        )
        self.span_stack.append(generation)
    
    def on_llm_end(self, response, **kwargs):
        """End an LLM call."""
        if self.span_stack:
            generation = self.span_stack.pop()
            generation.end(
                output=response.generations[0][0].text,
                usage={
                    "prompt_tokens": response.llm_output.get("token_usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": response.llm_output.get("token_usage", {}).get("completion_tokens", 0),
                    "total_tokens": response.llm_output.get("token_usage", {}).get("total_tokens", 0)
                }
            )

# Usage
from langchain.chains import LLMChain

callback = LangfuseCallbackHandler(langfuse, "langchain_query")

chain = LLMChain(llm=llm, prompt=prompt, callbacks=[callback])
result = chain.run("What is RAG?")
```

### OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Export to Langfuse
class LangfuseSpanExporter:
    """Export OpenTelemetry spans to Langfuse."""
    
    def __init__(self, langfuse: Langfuse):
        self.langfuse = langfuse
        self.traces = {}
    
    def export(self, spans):
        """Export spans to Langfuse."""
        for span in spans:
            trace_id = span.context.trace_id
            
            # Get or create trace
            if trace_id not in self.traces:
                self.traces[trace_id] = self.langfuse.trace(
                    name=span.name
                )
            
            trace = self.traces[trace_id]
            
            # Create span in Langfuse
            langfuse_span = trace.span(
                name=span.name,
                input=span.attributes,
                metadata={
                    "otel_span_id": span.context.span_id,
                    "otel_trace_id": trace_id
                }
            )
            
            # Set timing
            langfuse_span.start_time = span.start_time
            langfuse_span.end_time = span.end_time
            
            langfuse_span.end()

# Register exporter
span_processor = BatchSpanProcessor(LangfuseSpanExporter(langfuse))
trace.get_tracer_provider().add_span_processor(span_processor)
```

---

## Production Best Practices

### 1. Asynchronous Flushing

```python
# Configure batch flushing
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    flush_at=20,        # Flush after 20 events
    flush_interval=5    # Or flush every 5 seconds
)

# Ensure flush on shutdown
import atexit

def flush_langfuse():
    langfuse.flush()

atexit.register(flush_langfuse)
```

### 2. Sampling Strategy

```python
import random

class SampledTracer:
    """Sample traces based on criteria."""
    
    def __init__(self, langfuse: Langfuse, sample_rate: float = 0.1):
        self.langfuse = langfuse
        self.sample_rate = sample_rate
    
    def should_trace(self, **criteria) -> bool:
        """Decide if request should be traced."""
        # Always trace errors
        if criteria.get("is_error"):
            return True
        
        # Always trace expensive requests
        if criteria.get("cost", 0) > 1.0:
            return True
        
        # Sample others
        return random.random() < self.sample_rate
    
    def trace(self, name: str, **kwargs):
        """Create trace if sampled."""
        if self.should_trace(**kwargs):
            return self.langfuse.trace(name=name, **kwargs)
        else:
            return NoOpTrace()  # No-op implementation

class NoOpTrace:
    """No-op trace for non-sampled requests."""
    def span(self, *args, **kwargs):
        return NoOpSpan()
    def generation(self, *args, **kwargs):
        return NoOpSpan()
    def update(self, *args, **kwargs):
        pass

class NoOpSpan:
    def end(self, *args, **kwargs):
        pass
    def span(self, *args, **kwargs):
        return NoOpSpan()
```

### 3. PII Handling

```python
import re

def sanitize_for_langfuse(text: str) -> str:
    """Remove PII before sending to Langfuse."""
    # Redact emails
    text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        '[EMAIL]',
        text
    )
    
    # Redact phone numbers
    text = re.sub(
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        '[PHONE]',
        text
    )
    
    # Redact credit cards
    text = re.sub(
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        '[CARD]',
        text
    )
    
    return text

# Use in tracing
trace = langfuse.trace(name="query")
span = trace.span(
    name="process",
    input=sanitize_for_langfuse(user_input),
    output=sanitize_for_langfuse(response)
)
```

### 4. Error Recovery

```python
def safe_trace(func):
    """Wrap tracing to prevent failures from affecting app."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Log tracing error but don't fail app
            logger.warning(f"Tracing error: {e}")
            # Return no-op trace
            return NoOpTrace()
    
    return wrapper

@safe_trace
async def create_trace(name: str):
    return langfuse.trace(name=name)
```

### 5. Multi-Tenancy

```python
class TenantAwareTracer:
    """Trace with tenant isolation."""
    
    def __init__(self):
        self.langfuse_clients = {}
    
    def get_client(self, tenant_id: str) -> Langfuse:
        """Get Langfuse client for tenant."""
        if tenant_id not in self.langfuse_clients:
            # Each tenant has separate API keys
            self.langfuse_clients[tenant_id] = Langfuse(
                public_key=get_tenant_public_key(tenant_id),
                secret_key=get_tenant_secret_key(tenant_id)
            )
        return self.langfuse_clients[tenant_id]
    
    def trace(self, tenant_id: str, name: str, **kwargs):
        """Create trace for specific tenant."""
        client = self.get_client(tenant_id)
        return client.trace(name=name, **kwargs)
```

---

## Summary

### Key Takeaways

1. **Langfuse** provides purpose-built observability for LLM apps
2. **Trace/Span Model** enables hierarchical request tracking
3. **Cost Attribution** helps understand and optimize spending
4. **Dashboard Analytics** provide insights into usage patterns
5. **Framework Integration** works with LangChain, OpenTelemetry, etc.
6. **Production Practices** ensure reliability and performance

### Checklist

- [ ] Set up Langfuse account (cloud or self-hosted)
- [ ] Instrument basic LLM calls
- [ ] Add tracing to RAG pipeline
- [ ] Configure cost tracking
- [ ] Create custom dashboard
- [ ] Implement sampling strategy
- [ ] Add PII sanitization
- [ ] Test error handling
- [ ] Monitor production traces

### Next Steps

In Lesson 3, we'll cover **Guardrails & Safety Systems**, learning how to:
- Implement content moderation
- Detect and redact PII
- Build custom guardrails
- Validate outputs
- Meet compliance requirements

---

## Additional Resources

- [Langfuse Documentation](https://langfuse.com/docs)
- [Langfuse GitHub](https://github.com/langfuse/langfuse)
- [Langfuse Python SDK](https://pypi.org/project/langfuse/)
- [OpenTelemetry for Python](https://opentelemetry.io/docs/instrumentation/python/)
- [Distributed Tracing Best Practices](https://opentelemetry.io/docs/concepts/signals/traces/)
