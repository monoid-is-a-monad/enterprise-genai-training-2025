# Lab 01 – Threat Model & Security Control Implementation (Solutions)

> Instructor cheat sheet for reviewing learner submissions.

---

## 1. System Context & Architecture Updates

**Expect:**
- Architecture diagram (`docs/architecture/week-10/`) with new controls (policy service, telemetry, vaults).
- Callouts for data flows (persona tokens, guardrail decisions, fallback providers).
- Narrative describing changes since Week 9 (e.g., network segmentation, logging sinks).

---

## 2. Risk Register Quality

**Checklist:**
- Top threats captured with unique IDs, impact, likelihood, owners, and due dates.
- Mitigation plans aligned with Week 10 lessons (policy validation, ingestion QA, telemetry).
- Evidence links to PRs, dashboards, or tickets.
- Status updated to `In Progress` or `Mitigated` when controls deployed.

---

## 3. Control Implementation

**Controls to look for:**
- Persona validation policy (OPA/Guardrails) with allow/deny logic and unit tests.
- Telemetry updates capturing guardrail bypass attempts (Prometheus/Langfuse).
- Optional: network policy, secrets rotation, document QA pipeline.

**Validation artifacts:**
- Test outputs or CI logs.
- Screenshots of telemetry dashboards showing new metrics.
- Git diff references for policy code.

---

## 4. Evidence Summary

Ensure `artifacts/week10-security-summary.md` (or equivalent) includes:
- Controls implemented.
- Risks remaining and owners.
- Next steps for governance council review.
- Links to architecture diagram and risk register export.

---

## 5. Retrospective

Look for reflections on:
- Collaboration with security/compliance partners.
- Tooling gaps (e.g., need automated OPA tests, better telemetry UX).
- Open risks requiring escalation.

---

## Instructor Notes

- Encourage teams to integrate controls into CI/CD (not just manual scripts).
- Verify that mitigation owners align with Responsible AI council roster.
- Reinforce storing evidence in immutable locations for audit readiness.
