# Week 9 Change Management Communication Plan

This template aligns launch, incident, and rollback communications across stakeholder groups. Customize messaging, channels, and approval workflows before the production release.

## 1. Stakeholder Matrix

| Audience | Objective | Primary Channel | Backup Channel | Owner | Approval Needed |
| -------- | --------- | --------------- | -------------- | ----- | --------------- |
| Executive Sponsors | Highlight launch value, ensure readiness confidence | Email digest | Live briefing | Product Owner | Yes (Exec Sponsor) |
| Compliance & Legal | Confirm controls, document audit trail | ServiceNow ticket | Email | Compliance Officer | Yes (Chief Compliance Officer) |
| Support & Operations | Provide runbooks, escalation paths, known issues | Slack (support-ops) | Knowledge base | Support Lead | No |
| Customer-Facing Teams | Equip with messaging, FAQs | Enablement session | Email summary | Comms Lead | Yes (Marketing) |
| End Users (Internal Pilot) | Announce availability, set expectations | In-app banner | Email | Product Marketing | Yes (Legal) |

## 2. Launch Communication Templates

### Executive Summary Email

```
Subject: Launching the GenAI Assistant – Production Rollout Scheduled for <date>

Team,

We are targeting <date/time> for the production rollout of the GenAI assistant. Key highlights:
- Reliability: p95 latency 1.8s, availability 99.7%
- Safety: Guardrail block rate 97.5%, red-team regression clean
- Support: On-call roster staffed, runbooks rehearsed

Actions:
- Reply with any concerns by <deadline>
- Join the go/no-go call on <date/time>
- Review the attached readiness checklist

Regards,
<Your Name>
Release Manager
```

### Support Briefing

```
Channel: #support-ops (Slack)

- Launch window: <date/time>
- Contact: Incident Commander (<phone/email>)
- Escalation path: Support -> SRE -> Incident Commander
- Known limitations: <list>
- Response templates stored at: <link>
```

### Status Page Draft

```
Title: GenAI Assistant Production Launch
Status: Scheduled Maintenance
Start: <date/time>
End: <date/time>
Details: We are deploying the production-ready GenAI assistant. Users may experience brief response latency during the canary window. Updates will follow within 30 minutes of launch.
```

## 3. Incident Communication Templates

| Scenario | Trigger | Primary Message | Channel | Approval |
| -------- | ------- | --------------- | ------- | -------- |
| LLM provider outage | p95 latency > 4s, fallback active | "We are experiencing elevated latency due to upstream service disruption. Responses are being routed through a backup model." | Status page + Slack | Incident Commander |
| Guardrail bypass detected | Red-team regression fails in production | "We identified a content policy gap. Guardrail updates are being applied; some personas temporarily disabled." | Email to compliance + execs | Security Officer |
| Rollback executed | Canary metrics fail | "Deployment rolled back due to unexpected errors. Service restored to previous version. Investigation ongoing." | Exec + support email | Release Manager |

## 4. Approval Workflow

1. Draft message using template and attach evidence (dashboards, incidents).
2. Route for approval via ticket (ServiceNow/Jira) with status updates.
3. Track approvals in change request and log final copy in repository (`docs/comms/week-09/`).
4. After sending, capture timestamps, recipients, and summaries for audit.

## 5. Communication Cadence

| Phase | Update | Channel | Owner |
| ----- | ------ | ------- | ----- |
| T-7d | Launch announcement draft circulation | Email | Comms Lead |
| T-2d | Support enablement session | Live call | Support Lead |
| T-1d | Reminder + checklist confirmation | Slack | Release Manager |
| T+0 | Launch status updates every 30 min | Status page + Slack | Incident Commander |
| T+1d | Post-launch summary & metrics | Email digest | Product Owner |
| T+7d | Adoption report & next steps | Confluence / newsletter | Product Owner |

---

**Usage Notes**
- Store final communications in version control with timestamps and approvers.
- Coordinate with legal/compliance for any external-facing messaging.
- Ensure translations or accessibility requirements are met for broader audiences.
