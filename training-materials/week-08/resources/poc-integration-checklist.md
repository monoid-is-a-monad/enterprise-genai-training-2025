# PoC Integration Checklist

Use this checklist to ensure the Week 8 proof of concept is integrated, observable, and ready for demo day. Update status daily.

## 1. Scope & Charter
- [ ] Charter reviewed and approved by stakeholders
- [ ] Success metrics documented and baselined
- [ ] Must/Should/Could triage completed and circulated
- [ ] Risk register created with mitigation owners

## 2. Environments & Access
- [ ] Shared `.env.example` published with required keys
- [ ] Secrets rotated or validated within the last 7 days
- [ ] VPN, network rules, and firewall exceptions confirmed
- [ ] Vector store populated with latest enterprise documents

## 3. Pipeline Integration
- [ ] Retrieval pipeline tested end-to-end (ingest → retrieve → rerank)
- [ ] Orchestration layer handles primary and fallback model routing
- [ ] Guardrail policies applied to input and output stages
- [ ] Structured logging emits trace IDs for every request

## 4. Observability & QA
- [ ] Langfuse tracing enabled with custom spans for each stage
- [ ] Metrics dashboard shows latency, error rate, and token burn
- [ ] Regression suite running in CI (minimum daily cadence)
- [ ] Load test covers 2x expected demo traffic with headroom

## 5. Security & Compliance
- [ ] PII detection and redaction verified against test corpus
- [ ] Prompt injection rules validated using red-team catalog
- [ ] Access logs retained per policy (minimum 30 days)
- [ ] Data residency and retention statements documented

## 6. Demo Readiness
- [ ] Demo script drafted with timestamps and owner assignments
- [ ] Screens or dashboards bookmarked for quick access
- [ ] Backup recording plan prepared (screen capture, backup instance)
- [ ] FAQ/Q&A document drafted with likely objections

## 7. Handoff & Documentation
- [ ] `README` includes setup, run, and troubleshooting sections
- [ ] Deployment manifest or `docker-compose` validated
- [ ] Known issues captured with workaround or ETA
- [ ] Next iteration backlog created in project tracker

---

**Tip:** Convert this checklist into a shared task board so every item has an accountable owner. Pink sticky notes belong to blockers.
