# Monitoring and Operating RAG in Production

Practical guidance for observability, quality, and reliability.

## Key KPIs
- Answer correctness/faithfulness rate
- Retrieval recall@k and coverage
- Latency (p50/p95) by stage (retrieval, rerank, generation)
- Cost per request/session; cache hit rates
- Deflection rate (self-serve vs human escalation)

## Telemetry Model
- Traces: one span per stage (ingest → embed → index → retrieve → rerank → augment → generate)
- Logs: structured JSON events with query_id, user_id (hashed), doc_ids, scores, tokens, costs
- Metrics: counters and histograms for success/error, latency, tokens, costs, recall

## Production Guardrails
- Timeouts and circuit breakers per stage
- Max context tokens and truncation policy
- Policy checks (PII redaction, source allowlist)
- Safe completions (zero temperature, citation required)
- Fallbacks: cache-only mode, simpler prompt, smaller k

## Incident Response
- Runbooks per failure mode (vector DB down, LLM rate limits, cache miss storm)
- Feature flags to disable reranking/hybrid/search quickly
- Replay recent requests from logs for investigation

## Sample Event Schema
```json
{
  "ts": "2025-10-30T12:34:56Z",
  "query_id": "uuid",
  "user_hash": "abc123",
  "stage": "retrieve",
  "k": 10,
  "mmr_lambda": 0.4,
  "results": [{"doc_id": "d1", "score": 0.78}],
  "latency_ms": 45,
  "tokens": {"prompt": 900, "completion": 120},
  "cost_usd": 0.0031,
  "errors": []
}
```

## SLOs and Alerts
- Availability: 99.9%
- p95 total latency < 2.0s
- Recall@10 > 0.75 (on shadow evals)
- Faithfulness >= 95% on canary set
- Alert on anomaly in cost/query or token spikes

## Drift & Data Quality
- Monitor embedding distribution shifts
- Detect index staleness (age of last ingest)
- Validate chunk quality (token histograms, empty/boilerplate)

## Privacy & Compliance
- Log redaction for PII/secrets
- Access controls for source documents
- Data retention and deletion workflows

## Tooling
- OpenTelemetry for traces/metrics/logs
- PromptFoo/LangSmith for evaluation tracking
- Dashboards: Grafana/Datadog (latency, errors, cost), BI for quality metrics
