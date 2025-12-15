# Lab 02 – Responsible AI Policy Enforcement & Testing (Solutions)

> Facilitation guide highlighting expected deliverables and evaluation criteria.

---

## 1. Policy-as-Code Implementation

**Expect:**
- Updated policy files (Rego/YAML) enforcing topic blocks and persona disclosures.
- Versioning and comments referencing change ticket or approval.
- Policy lint/test pipeline evidence (CLI output, CI screenshot).

---

## 2. Regression Test Coverage

**Checklist:**
- Test cases covering allowed and denied scenarios for key personas/topics.
- Execution evidence (test report, artifact path) with zero failures.
- Explanation of additional edge cases or planned expansion.

---

## 3. Responsible AI Evaluations

**Review:**
- Evaluation dataset documented (location, schema, update cadence).
- Fairness/safety metrics appended to `resources/responsible-ai-scorecard.csv` with timestamps.
- Results summarized (e.g., fairness difference 0.08 < threshold).
- Tickets created for metrics exceeding thresholds.

---

## 4. CI/CD Integration

**Evidence:**
- Notes or PR showing pipeline updates (policy linting, evaluation job, approvals).
- Branch protection / required reviewer configuration snapshot.
- Storage path for evaluation outputs (`artifacts/responsible-ai/<release_id>/`).

---

## 5. Governance & Approvals

- `artifacts/policy-approval-log.md` populated with approvers, dates, next review.
- `resources/responsible-ai-assessment-checklist.md` status updated.
- Stakeholder communication plan (who was briefed, when).

---

## 6. Retrospective

Look for:
- Policy gaps identified (e.g., new personas needing rules).
- Automation backlog (policy diff notifications, simulation harness).
- Stakeholder feedback (legal, product, support) and responses.

---

## Instructor Notes

- Reinforce storing policy evaluation artifacts for audit trail.
- Encourage expansion of evaluation datasets to include multi-lingual/persona coverage.
- Verify pipeline failures block merges unless exception documented.
