# Lab 1 – Implementing Distributed Tracing (Solutions)

> These solutions mirror the structure of the student notebook. Each section provides one possible implementation for the corresponding exercise. Feel free to adapt the patterns to match your production stack.

---

## Exercise 1: Bootstrap Langfuse Instrumentation

```python
import os
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from langfuse import Langfuse
from langfuse.client import LangfuseClient

@dataclass
class TraceContext:
    """Container for request-scoped metadata."""

    user_id: str
    feature: str
    environment: str = "development"
    release: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict)

    def to_tags(self) -> Dict[str, str]:
        tags = {
            "user_id": self.user_id,
            "feature": self.feature,
            "environment": self.environment,
        }
        if self.release:
            tags["release"] = self.release
        tags.update(self.attributes)
        return tags


def bootstrap_langfuse_client() -> LangfuseClient:
    """Create an authenticated Langfuse client with defensive checks."""

    api_key = os.getenv("LANGFUSE_API_KEY")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    host = os.getenv("LANGFUSE_HOST")

    missing = [name for name, value in {
        "LANGFUSE_API_KEY": api_key,
        "LANGFUSE_PUBLIC_KEY": public_key,
        "LANGFUSE_HOST": host,
    }.items() if not value]

    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            f"Missing required Langfuse configuration: {missing_list}."
            " Set the environment variables before running the lab."
        )

    langfuse = Langfuse(
        api_key=api_key,
        public_key=public_key,
        host=host,
        flush_at=1,
        flush_interval=1,
        max_retries=3,
        timeout=10,
    )
    return langfuse.client


def start_trace(client: LangfuseClient, context: TraceContext) -> Tuple[any, any]:
    """Return a trace handle with a root span tagged using the provided context."""

    trace_id = str(uuid.uuid4())
    trace = client.trace(
        id=trace_id,
        name="rag_request",
        user_id=context.user_id,
        metadata=context.to_tags(),
    )
    root_span = trace.span(
        name="rag_pipeline",
        metadata=context.to_tags(),
    )
    root_span.start()
    return trace, root_span
```

---

## Exercise 2: Instrument a Multi-Step RAG Pipeline

```python
import asyncio
import time
from typing import Any, Dict, List, Tuple

import httpx

langfuse_client = bootstrap_langfuse_client()


async def embed_query(query: str, parent_span: Any) -> List[float]:
    span = parent_span.span(
        name="embed",
        input={"query": query},
        metadata={"model": "text-embedding-3-small"},
    )
    span.start()
    start = time.perf_counter()
    try:
        await asyncio.sleep(0.05)
        embedding = [0.12, 0.48, 0.76, 0.33]
        span.update(output={"vector_preview": embedding[:4]})
        return embedding
    except Exception as exc:  # pragma: no cover - demonstration
        span.update(error=str(exc))
        raise
    finally:
        span.end(duration_ms=(time.perf_counter() - start) * 1000)


async def retrieve_documents(embedding: List[float], parent_span: Any) -> List[Dict[str, Any]]:
    span = parent_span.span(
        name="retrieve",
        metadata={"vector_store": "pinecone", "top_k": 8},
    )
    span.start()
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient() as client:
            await asyncio.sleep(0.08)
        docs = [
            {"document_id": "doc-1", "score": 0.87},
            {"document_id": "doc-2", "score": 0.74},
        ]
        span.update(output={"match_count": len(docs)})
        return docs
    finally:
        span.end(duration_ms=(time.perf_counter() - start) * 1000)


async def rerank_results(results: List[Dict[str, Any]], parent_span: Any) -> List[Dict[str, Any]]:
    span = parent_span.span(
        name="rerank",
        metadata={"model": "cross-encoder", "top_n": 5},
    )
    span.start()
    start = time.perf_counter()
    try:
        await asyncio.sleep(0.03)
        reranked = sorted(results, key=lambda item: item["score"], reverse=True)
        span.update(output={"reranked_top": reranked[:3]})
        return reranked
    finally:
        span.end(duration_ms=(time.perf_counter() - start) * 1000)


async def generate_answer(query: str, context: List[Dict[str, Any]], parent_span: Any) -> Tuple[str, Dict[str, int]]:
    span = parent_span.span(
        name="generate",
        metadata={"model": "gpt-4o-mini"},
        input={"query": query, "context_ids": [doc["document_id"] for doc in context]},
    )
    span.start()
    start = time.perf_counter()
    try:
        await asyncio.sleep(0.12)
        answer = "RAG response placeholder"
        token_usage = {"prompt_tokens": 1120, "completion_tokens": 220}
        span.update(output={"response": answer})
        return answer, token_usage
    finally:
        span.end(duration_ms=(time.perf_counter() - start) * 1000)


async def rag_pipeline(prompt: str, context: TraceContext) -> str:
    trace, root_span = start_trace(langfuse_client, context)
    try:
        embedding = await embed_query(prompt, root_span)
        results = await retrieve_documents(embedding, root_span)
        reranked = await rerank_results(results, root_span)
        answer, token_usage = await generate_answer(prompt, reranked, root_span)
        root_span.update(
            metadata={"status": "success"},
            output={"answer_preview": answer[:120]},
        )
        root_span.set_tags({"prompt_length": len(prompt), **context.to_tags()})
        return answer
    except Exception as exc:
        root_span.update(metadata={"status": "failed", "error": str(exc)})
        raise
    finally:
        root_span.end()
        trace.flush()


async def main() -> None:
    response = await rag_pipeline(
        "Explain retrieval augmented generation in one paragraph.",
        TraceContext(user_id="user-123", feature="rag-search", environment="staging"),
    )
    print("Answer:", response)


# asyncio.run(main())
```

---

## Exercise 3: Attribute Tokens and Cost

```python
from decimal import Decimal
from typing import Any, Dict, List

PRICING = {
    "gpt-4o-mini": {"prompt": Decimal("0.0000005"), "completion": Decimal("0.0000015")},
    "text-embedding-3-small": {"prompt": Decimal("0.0000001"), "completion": Decimal("0")},
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Decimal:
    try:
        rates = PRICING[model]
    except KeyError as exc:
        raise ValueError(f"Unknown pricing model: {model}") from exc

    prompt_cost = rates["prompt"] * Decimal(prompt_tokens)
    completion_cost = rates["completion"] * Decimal(completion_tokens)
    return (prompt_cost + completion_cost).quantize(Decimal("0.0000001"))


def record_span_cost(span: Any, model: str, token_usage: Dict[str, int]) -> Decimal:
    cost = calculate_cost(
        model=model,
        prompt_tokens=token_usage.get("prompt_tokens", 0),
        completion_tokens=token_usage.get("completion_tokens", 0),
    )
    span.update(
        metadata={"billing_account": "enterprise-plan", "model": model},
        tags={
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "cost_usd": float(cost),
        },
    )
    return cost


def aggregate_trace_cost(spans: List[Any]) -> Decimal:
    total = Decimal("0")
    for span in spans:
        cost_tag = span.tags.get("cost_usd")
        if cost_tag is not None:
            total += Decimal(str(cost_tag))
    return total.quantize(Decimal("0.0000001"))
```

The `generate_answer` function from Exercise 2 now calls `record_span_cost` and the root span stores the aggregated trace cost.

---

## Exercise 4: Diagnose Performance Bottlenecks

```python
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd


def build_span_dataframe(spans: Iterable[Any]) -> pd.DataFrame:
    records = []
    for span in spans:
        record = {
            "name": span.name,
            "duration_ms": span.duration_ms,
            "status": span.metadata.get("status", "success"),
            "model": span.metadata.get("model"),
            "vector_store": span.metadata.get("vector_store"),
        }
        records.append(record)
    return pd.DataFrame(records)


def percentile_report(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("name")["duration_ms"]
    summary = grouped.apply(
        lambda values: pd.Series(
            {
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "count": len(values),
            }
        )
    )
    return summary.reset_index()


def detect_slo_breaches(df: pd.DataFrame, slo_targets: Dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    df["slo_target"] = df["name"].map(slo_targets).fillna(np.inf)
    df["breach"] = df["duration_ms"] > df["slo_target"]
    return df[df["breach"]]


def summarize_bottlenecks(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"slowest_span": None, "suggestion": "No spans recorded."}

    slowest_row = df.loc[df["duration_ms"].idxmax()]
    suggestion = "Review downstream dependency latency."
    if slowest_row.get("name") == "retrieve":
        suggestion = "Evaluate vector store indexes and network latency to the retrieval tier."
    elif slowest_row.get("name") == "generate":
        suggestion = "Consider caching or distilling the model for shorter completion times."

    return {
        "slowest_span": slowest_row.to_dict(),
        "suggestion": suggestion,
    }


# Example wiring once spans are available:
# span_df = build_span_dataframe(root_span.get_children())
# slo_breaches = detect_slo_breaches(span_df, {"embed": 80, "retrieve": 120, "generate": 250})
# print(summarize_bottlenecks(span_df))
```

---

## Exercise 5: Build Custom Metrics & Dashboards

```python
from datetime import datetime
from typing import Any, Dict, Optional


def aggregate_metrics(df: pd.DataFrame, feature: Optional[str] = None) -> Dict[str, Any]:
    working_df = df.copy()
    if feature:
        working_df = working_df[working_df["feature"] == feature]

    working_df["timestamp"] = pd.to_datetime(working_df["timestamp"], utc=True)
    working_df = working_df.set_index("timestamp")

    daily = working_df.resample("D").agg(
        requests=("name", "count"),
        median_latency_ms=("duration_ms", "median"),
        total_cost_usd=("cost_usd", "sum"),
    )
    daily = daily.fillna(0)

    return {
        "feature": feature or "all",
        "daily": daily.reset_index().to_dict(orient="records"),
        "overall": {
            "requests": int(daily["requests"].sum()),
            "median_latency_ms": float(daily["median_latency_ms"].median()),
            "total_cost_usd": float(daily["total_cost_usd"].sum()),
        },
    }


def prepare_dashboard_payload(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": "Langfuse RAG Daily Metrics",
        "feature": metrics["feature"],
        "series": metrics["daily"],
        "summary": metrics["overall"],
    }


def export_metrics(metrics: Dict[str, Any], destination: str) -> None:
    if destination == "langfuse":
        langfuse_client.metric(
            name="rag.daily.cost",
            value=metrics["overall"]["total_cost_usd"],
            timestamp=datetime.utcnow(),
            tags={"feature": metrics["feature"]},
        )
    elif destination == "stdout":
        print("Dashboard payload:", metrics)
    else:
        raise ValueError(f"Unknown destination: {destination}")


# demo_df = pd.DataFrame([...])  # populate with trace rows before calling functions above
```

---

## Exercise 6: Alert on Anomalies

```python
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional


def compute_z_score(values: List[float], new_value: float) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    sigma = pstdev(values)
    if sigma == 0:
        return 0.0
    return (new_value - mu) / sigma


@dataclass
class Alert:
    severity: str
    message: str
    metadata: Dict[str, Any]


def detect_latency_anomaly(latencies: List[float], latest_latency: float, threshold: float = 2.5) -> Optional[Alert]:
    z = compute_z_score(latencies, latest_latency)
    if z >= threshold:
        severity = "critical" if z > threshold + 1 else "warning"
        return Alert(
            severity=severity,
            message=f"Latency anomaly detected: z-score={z:.2f}",
            metadata={"latest_latency_ms": latest_latency, "z_score": z},
        )
    return None


def detect_error_spike(error_counts: List[int], latest_count: int, baseline: float) -> Optional[Alert]:
    if latest_count >= baseline * 2:
        return Alert(
            severity="critical",
            message="Error burst detected",
            metadata={"latest_count": latest_count, "baseline": baseline},
        )
    return None


def route_alert(alert: Alert) -> None:
    langfuse_client.annotation(
        type="alert",
        message=alert.message,
        metadata=alert.metadata,
        severity=alert.severity,
    )
    print(f"[{alert.severity.upper()}] {alert.message}")
```

---

## Exercise 7: Integrate OpenTelemetry Exporter

```python
from typing import List

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ReadableSpan,
    SpanExportResult,
    SpanExporter,
)


class LangfuseSpanExporter(SpanExporter):
    """Bridge OpenTelemetry spans into Langfuse."""

    def __init__(self, client: LangfuseClient, sample_rate: float = 1.0):
        self.client = client
        self.sample_rate = sample_rate

    def export(self, spans: List[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            if span.context.trace_id % int(1 / max(self.sample_rate, 0.01)) != 0:
                continue

            lf_trace = self.client.trace(
                id=span.context.trace_id.hex,
                name=span.name,
                user_id=span.resource.attributes.get("user_id"),
            )
            lf_span = lf_trace.span(
                name=span.name,
                metadata=dict(span.attributes),
                start_time=span.start_time,
                end_time=span.end_time,
                status=span.status.status_code.name.lower(),
            )
            lf_span.start()
            if span.status.description:
                lf_span.update(metadata={"status_description": span.status.description})
            lf_span.end()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self.client.flush()


def configure_opentelemetry_exporter(client: LangfuseClient, sample_rate: float = 0.25) -> None:
    provider = TracerProvider()
    exporter = LangfuseSpanExporter(client=client, sample_rate=sample_rate)
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)


# Example usage:
# configure_opentelemetry_exporter(langfuse_client)
# tracer = trace.get_tracer(__name__)
# with tracer.start_as_current_span("demo", attributes={"feature": "rag-search"}):
#     pass
```

---

## Wrap-Up Notes

- The solution uses defensive checks for configuration and ensures spans are closed even on failure.
- Cost attribution relies on `Decimal` to avoid rounding issues.
- Performance diagnostics leverage pandas/numpy for percentile reporting and SLO monitoring.
- Alerts are routed through Langfuse annotations but can be swapped for your incident tooling.
- The OpenTelemetry exporter demonstrates how to bridge data for unified observability.

Use these snippets as reference implementations when grading or guiding learners.
