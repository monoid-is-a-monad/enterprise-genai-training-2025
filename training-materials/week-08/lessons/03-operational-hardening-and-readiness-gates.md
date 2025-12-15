# Lesson 3: Operational Hardening & Readiness Gates

**Duration:** 100 minutes  
**Level:** Advanced  
**Prerequisites:** Lessons 1-2 completed, observability stack provisioned, guardrail catalog available

## Table of Contents
- [Lesson Overview](#lesson-overview)
- [Operational Readiness Mindset](#operational-readiness-mindset)
- [Readiness Gate Framework](#readiness-gate-framework)
- [Reliability Controls](#reliability-controls)
- [Performance & Load Validation](#performance--load-validation)
- [Security, Compliance & Data Governance](#security-compliance--data-governance)
- [Cost & Resource Guardrails](#cost--resource-guardrails)
- [Incident Response & Runbooks](#incident-response--runbooks)
- [Readiness Scorecard & Executive Sign-off](#readiness-scorecard--executive-sign-off)
- [Action Plan](#action-plan)

---

## Lesson Overview

This lesson transforms the Week 8 PoC from a stitched-together demo into a resilient prototype that can withstand load, scrutiny, and executive review. We formalize readiness gates, build operational controls, and rehearse failure scenarios so the demo team can answer "What happens if...?" with confidence.

### Learning Outcomes
- Define clear, measurable readiness gates across reliability, performance, security, and cost.
- Implement observability-driven validation steps before demo day.
- Run targeted load tests and capture evidence for key metrics.
- Demonstrate compliance alignment using documentation and tooling.
- Prepare runbooks for failure scenarios and communicate response plans.
- Use the readiness scorecard to drive executive go/no-go decisions.

---

## Operational Readiness Mindset

Operational hardening is about asking, *"If we exposed this tomorrow, would it behave predictably?"* Use the following guiding principles:

1. **Evidence-Based Sign-off:** No anecdotes. Every gate needs a metric, log snapshot, or test artifact.
2. **Defense in Depth:** Combine guardrails, observability, and automation to catch issues early.
3. **Rehearse Failures:** Practice how the system responds to injected faults or malicious prompts.
4. **Communicate Transparently:** Capture risks and mitigations in the shared risk register, updating status daily.

---

## Readiness Gate Framework

Build a gate model that stakeholders can sign off on confidently. Typical gate categories:

| Gate | Purpose | Evidence Required | Owner |
| ---- | ------- | ----------------- | ----- |
| Reliability | Confirm system survives expected & burst load | Load test report, uptime metric | SRE Lead |
| Performance | Validate latency & throughput targets | Latency distribution, trace samples | Backend Lead |
| Security | Demonstrate guardrails & data controls | Red-team log, policy docs | Security Lead |
| Compliance | Ensure data, access, and retention policies | Audit checklist, approvals | Compliance Officer |
| Cost | Keep demo budget within approved limits | Cost report, anomaly alerts | Program Manager |

Track gate status in `resources/readiness-scorecard.csv`. Color-code (Green/Amber/Red) and add notes.

### Gate Review Cadence
- Wednesday 18:00: Interim review, assign blockers
- Thursday 18:00: Final readiness gate pre-demo
- Friday 09:00: Executive go/no-go, only if all gates green or waivers signed

---

## Reliability Controls

### Health Checks & Monitoring
- Implement `/healthz` endpoints for Liveness and Readiness.
- Ensure orchestration layer retries idempotent operations with exponential backoff.
- Monitor circuit breakers around external APIs.

```python
# Example FastAPI health check
from fastapi import FastAPI
from services import vector_store, llm_router

app = FastAPI()

@app.get("/healthz")
async def health_check():
    vector_ok = await vector_store.ping()
    llm_ok = await llm_router.health()
    if not (vector_ok and llm_ok):
        return {"status": "degraded", "vector": vector_ok, "llm": llm_ok}
    return {"status": "ok"}
```

### Chaos & Fault Injection (Optional)
- Simulate vector DB latency spikes and observe reranker/LLM fallbacks.
- Disable guardrail service and confirm fail-safe behavior (deny vs allow).
- Document outcomes and update mitigations.

---

## Performance & Load Validation

### Load Test Strategy
- Select 3-5 representative demo scenarios.
- Run Locust or k6 scripts at 1x expected load, then 2x headroom.
- Record P50/P95/P99 latency, error rates, and throughput.

```python
from locust import HttpUser, task, between

class PoCUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def executive_briefing(self):
        payload = {
            "persona": "executive",
            "query": "Summarize top escalations for APAC support"
        }
        self.client.post("/api/query", json=payload)
```

Store results in `reports/load-test-<date>.md` and link them in the readiness scorecard.

### Latency Budgeting
- Break down latency by span (retrieval, guardrail, LLM).
- Ensure no single span exceeds 40 percent of total latency.
- Use Langfuse dashboards to visualize and annotate anomalies.

**Stretch:** Automate load tests nightly and alert on regressions exceeding 10 percent drift.

---

## Security, Compliance & Data Governance

### Guardrail Verification
- Run the Week 7 red-team suite against the PoC endpoint.
- Capture logs showing blocked prompts and sanitized outputs.
- Document any bypass and mitigation timeline.

```bash
python redteam/run_suite.py --target https://poc.example.com/api/query \
    --report reports/redteam-$(date +%Y%m%d).json
```

### Compliance Checklist
- Data classification confirmed (no production PII unless approved).
- Access controls: PoC credentials tied to individual accounts; audit logging enabled.
- Retention: logs trimmed per policy (masking sensitive fields).
- Exports: Data downloads disabled or monitored during demo.

Update `resources/stakeholder-communication-plan.md` with compliance sign-off details.

---

## Cost & Resource Guardrails

### Cost Monitoring
- Track usage by persona or scenario; compare against budget envelope.
- Set alerts for daily spend exceeding threshold (e.g., $100/day).

```python
import os
from analytics import cost_tracker

BUDGET_USD = float(os.getenv("POC_DAILY_BUDGET", "100"))
today_cost = cost_tracker.cost_for(date="today")
if today_cost > BUDGET_USD:
    cost_tracker.notify(f"Cost alert: ${today_cost:.2f} > ${BUDGET_USD:.2f}")
```

### Resource Guards
- Ensure GPU/CPU quotas are monitored if using dedicated hardware.
- Confirm auto-scaling policies (if any) do not unexpectedly scale down during demo hours.

Document cost posture in the readiness scorecard; adjust feature usage if needed.

---

## Incident Response & Runbooks

Create concise runbooks for likely failure scenarios. Include detection method, immediate actions, and escalation path.

### Runbook Template
```
Scenario: <e.g., Guardrail outage>
Detection: <alert name, dashboard>
Immediate Actions:
1. Toggle feature flag <flag_name> to fallback mode
2. Notify #poc-war-room with status update template
3. Switch demo environment to "Safe Mode"

Escalation:
- Primary: Security Lead (Slack/phone)
- Secondary: Program Manager

Resolution Steps:
1. Investigate latest deployment logs
2. Revert to stable build <sha>
3. Run guardrail verification suite

Postmortem Notes:
- Document root cause
- Update risk register
```

Store runbooks in `docs/poc-week/runbooks/` and review during dry runs.

---

## Readiness Scorecard & Executive Sign-off

### Scorecard Updates
- After each validation step, update metrics in `resources/readiness-scorecard.csv`.
- Include notes (e.g., "Guardrail suite flagged prompt X; mitigation deployed 12/13").
- For metrics still pending, add ETA and owner.

### Go/No-Go Meeting
- Participants: Executive Sponsor, Program Manager, Tech Lead, Security Lead.
- Inputs: Readiness scorecard, risk register, demo storyboard.
- Outcomes: Approve demo, approve with waivers, or defer (trigger contingency plan).

Prepare a one-page summary covering gate status, outstanding risks, and contingency triggers.

---

## Action Plan

1. **Execute Load Tests:** Run Locust/k6 scripts, publish report, and annotate Langfuse traces.
2. **Security Validation:** Complete red-team suite, capture evidence, update guardrail configs.
3. **Cost Check:** Generate 24-hour cost report and align with budget owner.
4. **Runbook Drill:** Conduct a tabletop exercise for top failure scenario; log outcomes.
5. **Scorecard Refresh:** Update all metrics, set statuses, and circulate to stakeholders.

### Stretch
- Implement synthetic monitoring that runs every 15 minutes against the PoC.
- Integrate guardrail validation into CI to prevent regressions before merges.
- Build a visualization combining load, cost, and guardrail metrics to show multi-dimensional readiness.

---

**Next Lesson Preview:** We pivot from operational assurance to storytelling—structuring the narrative, crafting demo flows, and preparing stakeholders for a high-impact presentation.
