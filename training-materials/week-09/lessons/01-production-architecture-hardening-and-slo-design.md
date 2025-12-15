# Lesson 01 – Production Architecture Hardening & SLO Design

**Session Length:** 3.5 hours (2h workshop + 1.5h design lab)

---

## 1. Framing the Challenge

Week 8 ended with a "demo-credible" proof of concept. Week 9 raises the bar to "production-compliant". We must:

- Map PoC architecture to production-grade infrastructure (multi-AZ, multi-region, DR)
- Define target reliability and performance budgets through SLOs/SLIs/SLAs
- Enforce compliance controls (PII, retention, audit trails) as first-class system requirements
- Plan for growth: autoscaling, cost envelopes, and resilience across dependencies

> **Goal:** Produce an Architecture Decision Record (ADR) and SLO charter that executive stakeholders can approve.

---

## 2. Production Topologies for GenAI Services

| Topology | When to Use | Key Components | Risks |
| -------- | ----------- | -------------- | ----- |
| **Blue/Green** | Zero-downtime releases, high availability | Two production stacks behind traffic switch, shared data layer | Cost overhead, data drift if environments diverge |
| **Canary** | Incremental rollouts with automated metrics checks | Progressive traffic shifting, automated rollback policies | Requires mature observability + rollback automation |
| **Multi-Region Active/Active** | Regulatory or latency requirements across geos | Geo-distributed vector stores, replication, latency-based routing | Complex consistency, higher operational cost |
| **Edge + Core Hybrid** | Low-latency inference, localized privacy | Edge worker for prompt preprocessing, centralized LLM | Hard to debug, needs strong synchronization |

**Action Items**
- Document desired topology transitions (e.g., Stage → Blue/Green → Multi-region) in the deployment roadmap.
- Capture dependencies (vector DB, policy engine, analytics) and their failover stories.

---

## 3. Hardening the Architecture

1. **Network & Access Controls**
   - Private subnets, service mesh (mTLS), dedicated ingress controllers.
   - Zero trust policies (short-lived tokens, JIT access) for operators.
2. **Secrets Management**
   - Rotate API keys via Vault/Secrets Manager; avoid `.env` drift.
   - Enforce envelope encryption for embeddings, prompt templates.
3. **Data & Storage**
   - Multi-layer backups (snapshot + logical) for vector store and guardrail configs.
   - Define retention policies and legal hold workflows.
4. **Resilience & Scaling**
   - Pod disruption budgets, HPA/VPA targets for inference workloads.
   - Rate limiting + backpressure for guardrail services.
5. **Cost Controls**
   - Budget alerts for GPU/LLM usage; tag resources by environment/product.

> **Deliverable:** Update the production architecture diagram (use `templates/diagrams/mermaid-templates.md` as a starter) and attach to the ADR.

---

## 4. SLO Design for GenAI Systems

### Defining SLIs

| SLI Category | Example Metric | Source | Notes |
| ------------ | -------------- | ------ | ----- |
| **Latency** | `p95_response_latency_ms` | Langfuse traces + API gateway | Exclude guardrail blocks, track fallback flows |
| **Quality** | `helpfulness_score_rolling_mean` | Eval harness, user feedback | Blend human evaluation + automated scoring |
| **Reliability** | `successful_requests / total_requests` | API logs, guardrail outcomes | Factor in fallbacks vs hard failures |
| **Drift** | `embedding_distance_to_baseline` | Drift detection pipeline | Track by persona, domain |
| **Cost** | `inference_cost_per_request` | Billing export, usage logs | Set guardrails per environment |

### SLO Formulas

```text
Availability SLO = 1 - (Total minutes of Sev1/Sev2 impact / Total minutes in period)
Latency SLO = P( response_latency_ms <= 2000 ) >= 0.95 during business hours
Quality SLO = Rolling 7-day average helpfulness_score >= 4.2 / 5.0
```

**Tips**
- Distinguish SLOs per persona (exec vs analyst) if interactions vary.
- Tie SLO error budgets to feature flag policies (e.g., experimentation drains budget).
- Document SLA commitments with legal/commercial teams before external launch.

---

## 5. Compliance & Governance Hardening

| Control Area | Requirement | Implementation Notes |
| ------------ | ----------- | -------------------- |
| Access Logging | Track operator actions on prompt/policy updates | Use audit logging with immutable storage |
| PII Handling | Mask or tokenize customer-identifiable data | Reuse Week 7 guardrails; add DLP scans post-deployment |
| Data Residency | Store EU data within EU regions | Align vector store + observability data sinks |
| Change Management | CAB approval for production changes | Link build artifacts + evidence to tickets |
| Incident Reporting | Notify stakeholders within SLA | Integrate status page + comms templates |

**Deliverable:** Compliance appendix attached to architecture package.

---

## 6. Workshop Flow

1. **Kickoff (15 min)** – Review PoC architecture, call out production gaps.
2. **Topology Selection (30 min)** – Use whiteboard/Jamboard to evaluate target topology.
3. **SLO Drafting (45 min)** – Breakout groups define SLIs + targets; consolidate into charter.
4. **Compliance Deep Dive (30 min)** – Security/compliance leads review required controls.
5. **ADR Authoring (60 min)** – Teams complete template (context, decision, implications).
6. **Readout (15 min)** – Share top reliability risks + mitigation plans.

---

## 7. Outputs & Templates

- `resources/deployment-readiness-checklist.md`
- `resources/production-slo-scorecard.csv`
- Architecture diagram (Mermaid/Draw.io) stored under `docs/architecture/week-09/`
- ADR stored in repo (e.g., `docs/adr/2025-12-production-architecture.md`)

---

## 8. Discussion Prompts

- Which dependency poses the largest risk to hitting the latency SLO? How do we mitigate it?
- How do we balance experimentation (new prompts, models) with error budget constraints?
- What telemetry do we still need to instrument to defend the SLO targets?
- Which compliance approvals need the longest lead time? Plan them backwards from the launch date.

---

## 9. Homework / Pre-Lab Checklist

- Finalize SLO targets and indicators in `resources/production-slo-scorecard.csv`
- Update architecture diagram with redundancy, failover, and scaling annotations
- Identify missing observability signals required to enforce the drafted SLOs
- Gather compliance requirements into shared document for CAB review

> Next session: We will codify CI/CD workflows to build, test, sign, and promote the production stack. Bring your infrastructure repository and IaC templates.
