# Lab 03 – Production Rollout & Incident Playbook (Solutions)

> Use this as a rubric for reviewing launch rehearsal submissions and incident preparedness.

---

## 1. Rollout Timeline

**Expected:**
- Timeline table exported to `artifacts/launch-timeline.md` with clear T-minus/T-plus markers, timestamps, actions, owners.
- Learner highlights dependencies (e.g., canary gating, analytics validation) in notes.
- Evidence that timeline shared with stakeholders (screenshot or link).

---

## 2. Go/No-Go Checklist

**Review:**
- `artifacts/go-no-go-checklist.csv` populated with categories and notes.
- References to readiness evidence (dashboards, CAB tickets, support scripts) either embedded or linked.
- Risk items and mitigations captured in risk register.

---

## 3. Deployment Dry Run

**Deliverables:**
- Output from `ReleaseOrchestrator` summarizing plan, canary results, rollback (if triggered).
- Evidence bundle under `artifacts/release-evidence/` (logs, screenshots, manifests).
- Learner explains fallback decision if canary fails; otherwise notes monitoring window for full rollout.
- Screenshots from deployment tooling (ArgoCD, GitHub Actions) if accessible.

---

## 4. Incident Tabletop

**What to check:**
- `artifacts/tabletop-report.md` includes timeline, decisions, communication steps, follow-up actions.
- Runbook references align with `resources/incident-playbook-template.md` or a populated variant.
- Lessons learned documented and owners assigned for remediation tasks.

---

## 5. Communication Packet

**Evidence:**
- Launch announcement draft stored at `artifacts/launch-communication-exec.html` (or equivalent).
- Support briefing notes (Slack copy, knowledge base link) attached.
- Optional: rollback message template appended to communication plan.
- Approval workflow captured (who reviewed/approved messaging).

---

## 6. Operating Model Update

**Verification:**
- `resources/mlops-operating-model.md` appended with Week 9 cadence updates (daily error budget standup, etc.).
- Changes tracked via version control; learner notes future updates (e.g., quarterly review).

---

## 7. Submission Checklist

Learner package should include:
- Completed rollout timeline and go/no-go artifacts.
- Release evidence bundle.
- Tabletop report documenting incident simulation.
- Launch/support communication drafts and approval notes.
- Updated operating model + scorecard reflecting launch state.
- Readiness checklist cross-referenced with evidence.

---

## Instructor Notes

- Reinforce capturing CAB ticket ID and status page links for audit.
- Encourage recording dry run sessions for future enablement.
- Verify that incident lessons learned feed backlog/action tracker.
