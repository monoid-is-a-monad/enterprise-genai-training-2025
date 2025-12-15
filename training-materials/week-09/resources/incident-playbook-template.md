# Week 9 Incident Playbook Template

Use this template to structure incident response runbooks for production GenAI services. Duplicate per scenario (e.g., LLM provider outage, hallucination spike, guardrail bypass).

---

## 1. Metadata

- **Incident Type:** 
- **Severity:** 
- **Primary Owner:** 
- **Backup Owner:** 
- **Last Review Date:** 
- **Related Runbooks:** 

## 2. Trigger Conditions

Describe the signals that initiate this runbook.

| Signal | Threshold | Source | Notes |
| ------ | --------- | ------ | ----- |
|  |  |  |  |
|  |  |  |  |

## 3. Immediate Actions (First 5 Minutes)

1. 
2. 
3. 

## 4. Stabilization Steps (First 30 Minutes)

| Step | Description | Owner | Evidence |
| ---- | ----------- | ----- | -------- |
| 1 |  |  |  |
| 2 |  |  |  |
| 3 |  |  |  |

## 5. Communication

- **Status Page Message:** 
- **Internal Slack Channel:** 
- **Stakeholder Email Template:** 
- **Approval Required From:** 

## 6. Diagnostics

- Query Langfuse traces: 
- Prometheus dashboard link: 
- Log search command: 
- Drift report location: 

## 7. Mitigation & Workarounds

List fallback procedures or mitigation strategies.

- 
- 
- 

## 8. Escalation Matrix

| Situation | Escalate To | Contact Method |
| --------- | ----------- | -------------- |
|  |  |  |
|  |  |  |

## 9. Closure Criteria

Define conditions for ending the incident.

- Service meets SLO within X minutes
- Guardrail block rate restored above threshold
- Communication updates sent to stakeholders
- Post-incident review scheduled

## 10. Post-Incident Tasks

| Task | Owner | Due Date | Notes |
| ---- | ----- | -------- | ----- |
| Draft PIR |  |  |  |
| Update runbook |  |  |  |
| File follow-up tickets |  |  |  |
| Refresh training materials |  |  |  |

---

**Instructions:**
- Store completed runbooks in `docs/runbooks/` with version history.
- Review quarterly or after major incidents.
- Align with on-call training and communication templates from Week 9 resources.
