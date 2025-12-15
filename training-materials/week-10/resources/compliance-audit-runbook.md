# Week 10 Compliance Audit Runbook

This runbook guides preparations for internal and external audits related to the GenAI assistant. Customize with your organization’s policies and regulatory requirements.

## 1. Scope & Objectives

- Document Responsible AI policies, evaluations, and governance evidence.
- Demonstrate adherence to security, privacy, and regulatory commitments.
- Provide auditors with traceable artifacts linked to releases and incidents.

## 2. Stakeholders

| Role | Name | Responsibilities | Backup |
| ---- | ---- | ---------------- | ------ |
| Audit Lead |  | Coordinate audit activities, manage requests |  |
| Security Liaison |  | Provide threat model, control evidence |  |
| Compliance Officer |  | Validate legal/regulatory requirements |  |
| Responsible AI Lead |  | Supply policy approvals, evaluation reports |  |
| DevOps Lead |  | Produce deployment logs, CI evidence |  |
| Support Lead |  | Share incident reports, customer comms |  |

## 3. Pre-Audit Checklist

- [ ] Confirm evidence manifest generated and stored in compliance vault.
- [ ] Archive latest policy versions with approval signatures.
- [ ] Export Responsible AI scorecard (baseline + recent metrics).
- [ ] Gather deployment and rollback logs for past 90 days.
- [ ] Compile incident reports and remediation status.
- [ ] Prepare stakeholder communication summaries (executive, regulator, customers).

## 4. Evidence Sources

| Artifact | Location | Retention Policy |
| -------- | -------- | ---------------- |
| Policy repository | `guardrails/policies/` | 3 years |
| Evaluation reports | `artifacts/responsible-ai/` | 2 years |
| Deployment manifests | `artifacts/release-evidence/` | 2 years |
| Incident reports | `docs/incidents/` | 5 years |
| Governance minutes | `docs/governance/` | 5 years |
| Approval logs | `artifacts/policy-approval-log.md` | 5 years |

## 5. Audit Process

1. **Kickoff Meeting** – Align scope, timeline, and communication expectations.
2. **Evidence Submission** – Provide manifest, scorecards, and documentation.
3. **Deep Dive Sessions** – Security, Responsible AI, and operations walkthroughs.
4. **Findings Review** – Capture feedback, remediation tasks, and deadlines.
5. **Closure** – Deliver final report, update governance backlog, archive artifacts.

## 6. Communication Plan

- Designate audit communication channel (Slack/Teams) with observers.
- Send daily status updates to stakeholders during audit window.
- Log all auditor requests and responses in ticketing system.
- Prepare post-audit summary for executives and Responsible AI council.

## 7. Continuous Improvement

- After each audit, conduct retrospective (what worked, gaps, actions).
- Update this runbook with new requirements or improved automation.
- Track remediation tasks in governance backlog with due dates and owners.

---

**Revision History**

| Date | Author | Summary |
| ---- | ------ | ------- |
| 2025-12-14 | Week 10 Cohort | Initial runbook published |
