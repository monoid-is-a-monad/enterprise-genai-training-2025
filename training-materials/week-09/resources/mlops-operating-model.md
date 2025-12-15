# Week 9 MLOps Operating Model

Define ownership, cadences, and tooling for operating the GenAI assistant in production. This template should be customized with your organization's roles and SLAs.

## 1. Roles & Responsibilities

| Role | Primary Owner | Responsibilities | Backup |
| ---- | ------------- | ---------------- | ------ |
| Release Manager |  | Coordinate releases, CAB submissions, readiness checkpoints |  |
| SRE Lead |  | Maintain SLO dashboards, incident routing, capacity planning |  |
| Applied Science Lead |  | Continuous evaluation, drift detection, model retraining |  |
| Security Officer |  | Guardrail governance, vulnerability management, compliance evidence |  |
| Support Lead |  | L1/L2 support workflows, knowledge base updates, customer comms |  |
| Product Owner |  | Stakeholder alignment, roadmap prioritization, KPI tracking |  |
| Communications Lead |  | Launch announcements, status page updates, incident messaging |  |

## 2. Operating Cadence

| Cadence | Meeting | Participants | Focus | Artifacts |
| ------- | ------- | ------------ | ----- | --------- |
| Daily | Error budget standup | SRE, Applied Science, DevOps | Review SLOs, alerts, ongoing incidents | Dashboard snapshot, incident log |
| Twice Weekly | Release readiness huddle | Release Manager, DevOps, Security | Review pipeline status, blockers, CAB items | Readiness checklist, pipeline reports |
| Weekly | Drift & evaluation review | Applied Science, Product | Analyze drift metrics, eval scores, backlog updates | Drift report, mitigation plan |
| Bi-weekly | Stakeholder sync | Product, Comms, Support | Share adoption metrics, feedback, roadmap | Adoption dashboard, support tickets summary |
| Monthly | Compliance audit touchpoint | Security, Compliance, Legal | Control evidence, policy changes, regulatory updates | Audit log, action tracker |
| Quarterly | Post-launch business review | Exec sponsors, Product | ROI metrics, roadmap adjustments | KPI report, strategic memo |

## 3. Tooling Inventory

| Capability | Tool | Owner | Access Model | Notes |
| ---------- | ---- | ----- | ------------ | ----- |
| CI/CD |  |  |  |  |
| Infrastructure as Code |  |  |  |  |
| Observability (metrics/traces) |  |  |  |  |
| Incident Management |  |  |  |  |
| Drift Detection / Eval |  |  |  |  |
| Secrets Management |  |  |  |  |
| Artifact Registry |  |  |  |  |
| Document Repository |  |  |  |  |

## 4. Escalation Matrix

| Severity | Trigger Examples | Primary Contact | Backup Contact | SLA |
| -------- | ---------------- | --------------- | -------------- | --- |
| Sev1 | Outage, data leak, regulatory breach |  |  | 5 minutes |
| Sev2 | SLO breach, guardrail bypass, drift alert |  |  | 15 minutes |
| Sev3 | Partial degradation, minor quality issues |  |  | 60 minutes |
| Sev4 | Cosmetic defect, documentation issue |  |  | 1 business day |

## 5. Documentation & Evidence

- Store release runbooks in `docs/runbooks/`
- Archive incident reports and PIRs in `docs/incidents/`
- Maintain SLO dashboards links and snapshots under `docs/dashboards/week-09/`
- Capture CAB tickets, approvals, and risk logs in `docs/governance/`

## 6. Continuous Improvement Backlog

Track process/tooling enhancements that surface during retrospectives.

| Item | Description | Owner | Target Sprint |
| ---- | ----------- | ----- | ------------- |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

---

**Instructions:**
1. Fill in owner names, tools, and SLAs with real values.
2. Share the completed model with stakeholders and store in version control.
3. Review quarterly to align with evolving business and regulatory requirements.
