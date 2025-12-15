# Lab 01 – Automated Build & Release Pipeline (Solutions)

> Instructor reference. Validate learner submissions against these checkpoints and evidence expectations.

---

## 1. Environment Validation

**Expected:**
- `.github/workflows/`, `infra/`, `pipelines/` directories exist.
- Learner documents any missing prerequisites and remediation steps (e.g., provisioning ArgoCD manifests).

---

## 2. Pipeline Blueprint

**Key artifacts:**
- Updated CI/CD architecture diagram (link in submission).
- Mapping of stage owners captured in readiness checklist.
- GitHub Actions workflow (or equivalent) generated at `.github/workflows/genai-release.yaml`.

**Workflow essentials:**
- Install CI dependencies via `requirements-ci.txt`.
- Run unit/integration tests (`pytest`).
- Security scan (`trivy fs …`).
- Build, push, and sign container (`cosign`).
- Publish release manifest referencing the commit SHA/release ID.

---

## 3. Promotion Simulation

**Checkpoints:**
- `env/staging/values.yaml` (or analogous manifest) updated with `imageTag` and `releaseSha` tied to release ID.
- Evidence bundle at `artifacts/release-evidence.json` with release metadata, approver, timestamp.
- CAB log notes referencing simulated approval.

---

## 4. Regression & Security Evidence

**Expected outputs:**
- Test suite execution log (attached or linked) showing green status.
- Red-team regression results with ≥95% block rate or documented mitigation plan.
- Security scan report (Trivy/Snyk) stored in `artifacts/`.
- Langfuse trace bundle or screenshots confirming integration tests exercised guardrail paths.

**Reminder:** Learners should remove placeholder commands and run actual tooling in CI.

---

## 5. Readiness Handoff

**Artifacts to review:**
- `resources/deployment-readiness-checklist.md` Week 9 portion updated with release evidence links.
- `resources/production-slo-scorecard.csv` baseline columns populated.
- CAB ticket reference / link provided.
- Screenshots of CI pipeline run and registry entries.

---

## 6. Retrospective

**Looking for:**
- Completed retrospective table with highlights, gaps, security findings, next actions.
- Mention of automation backlog (e.g., integrate SBOM scanning, expand red-team suite).

---

## Instructor Notes

- Encourage teams to store workflow as reusable template for future releases.
- Verify secrets management strategy (`REGISTRY`, OIDC integration) documented.
- If learners cannot execute CI (due to environment limits), require dry-run evidence plus a plan for production pipelines.
