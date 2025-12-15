# Lesson 04 – Release Management & Stakeholder Operations

**Session Length:** 3 hours (1h lecture + 1h workshop + 1h stakeholder simulation)

---

## 1. Objectives

- Define enterprise change management pathways for GenAI production launches
- Align stakeholders (product, security, compliance, legal, support) around go/no-go gates
- Craft communication artifacts for launch, incident, and rollback scenarios
- Establish post-launch operating rhythms and continuous improvement loops

---

## 2. Change Management Lifecycle

| Stage | Description | Evidence | Stakeholders |
| ----- | ----------- | -------- | ------------ |
| **Initiate** | Submit change request, attach release plan | CAB ticket, risk assessment | Product, Engineering |
| **Assess** | Evaluate risk, compliance impact, staffing | Security review, legal sign-off | CAB board, Compliance |
| **Approve** | Formal CAB meeting/gate sign-off | Meeting minutes, approval log | CAB chair, Exec sponsor |
| **Deploy** | Execute rollout per playbook | Release notes, monitoring dashboards | DevOps, SRE |
| **Validate** | Confirm SLO adherence, user acceptance | Metrics snapshots, user feedback | Product, Support |
| **Close** | Post-launch review, document learnings | PIR, backlog updates | All |

**Tip:** Maintain a change calendar that highlights blackout periods, high-risk events, and dependent launches.

---

## 3. Go/No-Go Criteria

- **Technical Readiness** – All SLO dashboards green, drift detectors active, rollback tested
- **Process Readiness** – CAB approval recorded, incident runbooks rehearsed, on-call roster confirmed
- **Data Readiness** – Vector stores refreshed, prompt library locked, compliance attestation stored
- **Support Readiness** – Support scripts updated, knowledge base seeded, staffing plan active
- **Stakeholder Readiness** – Launch comms drafted, legal review complete, exec sponsor briefed

> Use `resources/deployment-readiness-checklist.md` to verify each item with evidence links.

---

## 4. Communication Playbooks

### Launch Announcement (Exec Audience)
- Clear value proposition tied to metrics (latency improvements, adoption goals)
- Risk/mitigation summary with guardrail highlights
- Call to action (stakeholder enablement sessions, adoption targets)

### Support & Operations Brief
- Incident contact tree, escalation paths, ticket taxonomy
- Known limitations and workaround catalog
- Issue reporting template (environment, persona, reproduction steps)

### Rollback / Incident Update
- Trigger conditions (SLO breach, security alert)
- Impact summary, time of rollback, user impact mitigation
- Next steps and reassured timeline for fix/re-release

Store templates in `resources/change-management-communication-plan.md`.

---

## 5. Stakeholder Simulation Activity

1. **Prep (10 min)** – Teams review release plan and readiness scorecard.
2. **Mock CAB (20 min)** – Present release plan; CAB members challenge risk areas.
3. **Launch Briefing (15 min)** – Deliver executive summary + Q&A.
4. **Incident Inject (15 min)** – Facilitator introduces outage scenario; teams execute comms plan.
5. **Retrospective (10 min)** – Capture learnings, update comms templates.

Roles: Release Manager, Security Officer, Support Lead, Exec Sponsor, Communications Lead.

---

## 6. Post-Launch Operating Rhythm

| Cadence | Activity | Owner | Artifacts |
| ------- | -------- | ----- | --------- |
| Daily | Error budget review, incident standup | SRE Lead | Dashboard snapshot, incident log |
| Weekly | Continuous evaluation + drift review | Applied Science | Eval report, mitigation backlog |
| Bi-weekly | Stakeholder roundtable | Product Lead | Adoption metrics, feedback summary |
| Monthly | Compliance audit sync | Compliance Officer | Control evidence, audit checklist |

Ensure items feed into Week 10 governance backlog.

---

## 7. Documentation & Evidence

- Release runbook stored in `docs/runbooks/production-rollout.md`
- CAB ticket references (ServiceNow/Jira) linked in readiness checklist
- Status page templates versioned in repo (Markdown + JSON payloads)
- Adoption and ROI dashboard mocks saved under `docs/dashboards/week-09/`

---

## 8. Discussion Prompts

- Which stakeholders can veto the launch? Do they have enough data to decide?
- How do we balance speed of iteration with compliance approvals?
- What metrics will determine "success" in the first 30 days post-launch?
- How do we ensure learnings feed back into product backlog and guardrail updates?

---

## 9. Action Items Before Lab 03

- Populate `resources/change-management-communication-plan.md` with contact lists and message templates
- Update `resources/production-slo-scorecard.csv` with launch threshold baselines
- Schedule mock CAB and go/no-go meetings with real stakeholders
- Draft status page entries for launch, partial incident, and rollback scenarios

> Lab 03 will simulate the production rollout, execute a dry run of the incident playbook, and finalize submission packages.
