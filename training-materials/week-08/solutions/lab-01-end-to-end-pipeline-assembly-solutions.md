# Lab 1 – End-to-End Pipeline Assembly (Solutions)

> These instructor notes mirror the structure of the student notebook. They provide a reference implementation plus guidance on what to look for during reviews. Adapt the snippets to match your PoC stack.

---

## 1. Environment Bootstrap

```python
import os
from pathlib import Path

REQUIRED = [
    "OPENAI_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "VECTOR_DB_URL",
    "VECTOR_DB_TOKEN",
]

missing = [env for env in REQUIRED if not os.getenv(env)]
if missing:
    raise RuntimeError(f"Missing environment variables: {missing}")

Path("reports").mkdir(exist_ok=True)
print("✅ Environment configuration looks good.")
```

**Review tips**
- Confirm students validated secrets before connecting services.
- Encourage storing `.env.example` in repo and real secrets in Vault/KeyVault/1Password.
- If working in a shared environment, require per-person credentials to aid auditing.

---

## 2. Retrieval Layer Assembly

```python
from typing import Dict, List
from pathlib import Path

from rag.retriever import HybridRetriever
from rag.chunking import load_corpus

CORPUS_PATH = Path("data/corpus/risk-escalations.jsonl")
INDEX_NAME = "risk-escalations-v1"

corpus = load_corpus(CORPUS_PATH)
retriever = HybridRetriever(
    index_name=INDEX_NAME,
    vector_db_url=os.environ["VECTOR_DB_URL"],
    token=os.environ["VECTOR_DB_TOKEN"],
)

def retrieve_context(query: str, persona: str, top_k: int = 5) -> List[Dict]:
    filters = {"persona": persona} if persona else None
    docs = retriever.search(query=query, top_k=top_k, filters=filters)
    for doc in docs:
        doc.setdefault("metadata", {})
        doc["metadata"].update({
            "persona": persona,
            "index": INDEX_NAME,
        })
    return docs

sample = retrieve_context("Summarize APAC escalations", persona="executive")
print({"docs": len(sample)})
```

**Review tips**
- Validate that persona filters prevent inappropriate material surfacing during the demo.
- Require metadata enrichment (persona, index, freshness) so guardrails and analytics have context.
- Discuss how to snapshot the index for demo reproducibility.

---

## 3. Guardrails & Prompt Orchestration

```python
from guardrails import GuardrailEngine, ValidationError
from guardrails.policies import load_policy

policy = load_policy(Path("config/guardrails/prompt-firewall.yaml"))
engine = GuardrailEngine(policy=policy)

def apply_guardrails(query: str, context: List[Dict]) -> str:
    persona = context[0]["metadata"].get("persona") if context else None
    engine.validate_input(query, metadata={"persona": persona})
    prompt = engine.build_prompt(query, context=context)
    return prompt

def validate_output(text: str) -> str:
    try:
        engine.validate_output(text)
    except ValidationError as err:
        # Escalate to fallback path. Instructors should encourage logging.
        raise RuntimeError(f"Guardrail rejection: {err}")
    return text
```

**Review tips**
- Double-check that policies reference Week 7 catalog (prompt injection, toxicity, PII).*  
- Students should capture allow/deny decisions for analytics (`guardrail_decision` tag).
- Discuss how to fan out to multiple guardrail layers (e.g., input + retrieval + output).

---

## 4. Orchestration & LLM Invocation

```python
from orchestration.router import LLMRouter, LLMResponse
from orchestration.fallbacks import fallback_strategy
from observability.tracing import pipeline_tracer

router = LLMRouter(primary="gpt-4.1", fallback="gpt-4o-mini")

@pipeline_tracer.trace(name="risk_analyst_pipeline")
def run_pipeline(query: str, persona: str) -> LLMResponse:
    context = retrieve_context(query, persona=persona)
    prompt = apply_guardrails(query, context)
    try:
        response = router.invoke(prompt, context=context)
    except Exception as primary_error:
        pipeline_tracer.log_event(
            "primary_model_failure",
            {"error": str(primary_error), "persona": persona},
        )
        response = fallback_strategy(router, prompt, context=context)
    cleaned = validate_output(response.content)
    return LLMResponse(content=cleaned, metadata=response.metadata)

if __name__ == "__main__":
    resp = run_pipeline(
        "Give me the top three APAC escalations to watch",
        persona="executive",
    )
    print(resp.content[:320])
```

**Review tips**
- Ensure recoverability: fallback path must trigger when primary fails or guardrail denies output.
- Instructors should check that metadata (persona, cost, latency) flows through `LLMResponse`.
- Encourage capturing replay artifacts (prompt, context IDs) for audit and red-team follow-up.

---

## 5. Observability Instrumentation

```python
from langfuse import Langfuse

langfuse = Langfuse()

@pipeline_tracer.on_span_finish
def publish_span(span):
    langfuse.log_trace(
        trace_id=span.id,
        name=span.name,
        metadata={
            "persona": span.tags.get("persona"),
            "guardrail_decision": span.tags.get("guardrail_decision"),
        },
    )
    if span.status == "error":
        langfuse.log_event(
            "pipeline_error",
            {
                "trace_id": span.id,
                "span_name": span.name,
                "error": span.tags.get("error"),
            },
        )
```

**Review tips**
- Validate that traces include: persona, latency, guardrail decision, model used, fallback flag.
- Instructors should look for dashboards that overlay latency ↔ guardrail block rate ↔ cost.
- Ask students to annotate baseline traces and upload screenshots as evidence.

---

## 6. Regression & Readiness Checks

```python
import subprocess

def run_gate(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(" ".join(cmd), "✅")
    print(result.stdout)

validators = [
    ["pytest", "tests/smoke", "--maxfail=1", "--disable-warnings", "-q"],
    ["python", "scripts/guardrail_report.py", "--min-block-rate", "0.95"],
    ["python", "scripts/check_latency.py", "--threshold-ms", "3000"],
]

for validator in validators:
    run_gate(validator)
```

**Review tips**
- Remind students to surface failures in the risk register immediately.
- Encourage storing stdout artifacts in `reports/` so stakeholders can inspect results.
- If guardrail thresholds are missed, require a waiver before demo rehearsals continue.

---

## 7. Retrospective & Checklist

| Item | Observation | Owner | Follow-up |
| ---- | ----------- | ----- | --------- |
| Retrieval quality | Satisfied ¬ APAC persona filters working | Backend lead | Monitor daily |
| Guardrail bypass attempts | Two prompts required stricter regex | Security lead | Added `block-sql-control` rule |
| Latency hot spots | Reranker span spiked to 1.8s once | Backend lead | Investigate Wednesday |
| Cost anomalies | None observed (< $0.20 per run) | Program manager | Update scorecard |
| Telemetry gaps | Missing fallback flag in traces | SRE lead | Add tag in pipeline tracer |

**Submission checklist**
- Notebook executes top-to-bottom without errors.  
- `resources/readiness-scorecard.csv` updated (latency, block rate).  
- Guardrail & load-test artifacts stored under `reports/`.  
- Demo storyboard updated with new capabilities.  
- Daily brief posted with outcomes and next steps.

---

**Instructor notes**
- Push students to run the pipeline under multiple personas and across fresh data snapshots.
- Emphasize capturing evidence (screenshots, logs) — Week 8 stakeholders expect auditability.
- Encourage adding TODOs to backlog for items that cannot be fixed before the demo.
