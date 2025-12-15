# Lab 02 – Model Monitoring & Drift Alerts (Solutions)

> Facilitator guide to evaluate monitoring configurations, drift jobs, and alert evidence.

---

## 1. Telemetry Hook Validation

**Expectations:**
- Sample Langfuse trace (`observability/tracing/sample-trace.json`) includes attributes: `persona`, `guardrail_decision`, `fallback_used`, `latency_ms` (or equivalent names).
- Learner documents any missing keys and remediation (SDK updates, middleware instrumentation).
- Screenshots or links to Langfuse UI highlighting span attributes per persona.

---

## 2. Prometheus Rule Enhancements

**Checklist:**
- `observability/metrics/rules.yaml` populated with alerts for latency, success rate, guardrail block rate, drift score.
- Thresholds align with Week 9 SLOs (e.g., p95 latency ≤ 2000 ms, success rate ≥ 97%).
- Learner runs rule validation (`promtool check rules`) or equivalent and attaches output.
- Alert labels include severity + runbook annotations.

---

## 3. Drift Detection Pipeline

**Deliverables:**
- Code in notebook executes `DriftMonitor` with 14-day baseline, 6-hour comparison window.
- Signals tracked: embedding distance, helpfulness score delta, guardrail bypass rate.
- Snapshot artifacts stored under `artifacts/drift/` (JSON/markdown).
- When thresholds breach, PagerDuty incident triggered; learner provides incident ID or screenshot.
- If no drift detected, learner explains baseline alignment and next review date.

---

## 4. Alert Routing Test

**Evidence:**
- Synthetic Slack alert posted (screenshot of message with runbook link).
- PagerDuty/incident tool receives test incident (acknowledged/resolved in portal).
- Runbook link points to `resources/incident-playbook-template.md` or populated runbook.

---

## 5. Scorecard Updates

**Review:**
- `resources/production-slo-scorecard.csv` filled with targets, baselines, owners, evidence links.
- Status column reflects monitoring readiness (e.g., `On Track`, `Needs Update`).
- Notes capture outstanding telemetry gaps.

---

## 6. Submission Checklist

Learners should supply:
- Langfuse trace screenshot (guardrail attributes).
- Grafana/Prometheus dashboard view with new alerts.
- Drift report artifact path.
- Alert routing evidence (Slack + PagerDuty).
- Updated readiness checklist referencing monitoring items.

---

## Instructor Notes

- Encourage storing alert payload examples for audit/compliance.
- Verify alert severity ties to on-call escalation matrix in `resources/mlops-operating-model.md`.
- If production tooling unavailable, ensure learners provide mock evidence plus plan for real environment integration.
