# Lesson 02 – CI/CD & Model Delivery Pipelines

**Session Length:** 3 hours (90 min lecture + 90 min pipeline design lab)

---

## 1. Objectives

- Build automated pipelines that package code, prompts, guardrails, and retrieval assets
- Embed security scanning, red-team regression, and eval harnesses in CI
- Promote artifacts across dev → stage → prod with approvals and audit trails
- Ensure rollback automation and environment parity for GenAI services

---

## 2. Pipeline Segments & Ownership

| Stage | Owner | Description | Evidence |
| ----- | ----- | ----------- | -------- |
| **Source** | Engineering | Monorepo or multi-repo with Infrastructure as Code (IaC) + application code | Branch protections, code reviews |
| **Build** | Platform | Container build, dependency caching, SBOM generation | Build logs, signed artifacts |
| **Test** | QA / Applied science | Unit, contract, integration, guardrail regression, load test smoke | Test reports, Langfuse trace bundle |
| **Scan** | Security | SAST, DAST, secret scanning, supply chain verification | Trivy/Snyk reports |
| **Promote** | DevOps | GitOps sync (ArgoCD/Flux) or pipeline-driven releases | Promotion ticket, CAB approval |
| **Observe** | SRE | Health checks, drift monitors, synthetic probes | Grafana dashboard links |

---

## 3. Model & Prompt Versioning Strategy

1. **Artifacts to Manage**
   - Model weights / endpoints
   - Prompt templates / prompt flows
   - Guardrail policies & attack corpora
   - Retrieval indexes (vector snapshots)
   - Evaluation benchmarks & thresholds

2. **Versioning Principles**
   - Use semantic versioning or calendar versioning aligned with release cadence
   - Store metadata (creator, change summary, evaluation results) next to artifact
   - Maintain lineage: `model_version -> prompt_version -> guardrail_policy_version`
   - Publish release notes for each artifact set

3. **Storage & Promotion**

```mermaid
graph LR
    A[Experiment Registry (Weights & Biases / MLflow)] --> B[Model Registry]
    B -->|Promote| C[Staging Endpoint]
    C -->|Smoke Test| D[Production Endpoint]
    B -->|Rollback| E[Previous Production]
    F[Prompt Repo] --> G[Prompt Bundle Artifact]
    G -->|Sign & Publish| D
```

> **Tip:** Tie every production release to a unique `release_id` that binds container image, model version, prompt bundle, and guardrail policy.

---

## 4. CI/CD Implementation Patterns

### GitHub Actions Template (excerpt)

```yaml
name: genai-release

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install deps
        run: pip install -r requirements-ci.txt
      - name: Run tests
        run: pytest --maxfail=1 --disable-warnings
      - name: Security scan
        run: trivy fs --exit-code 1 .
      - name: Build container
        run: docker build -t ${{ env.IMAGE_NAME }}:${{ github.sha }} .
      - name: Push to registry
        run: docker push ${{ env.IMAGE_NAME }}:${{ github.sha }}
      - name: Sign artifact
        run: cosign sign ${{ env.IMAGE_NAME }}:${{ github.sha }}
```

### GitOps Promotion Flow

1. Merge to `main` triggers build pipeline and publishes signed image + manifest.
2. ArgoCD watches `env/staging` folder; automation updates image tag with new `release_id`.
3. Staging tests run (synthetic load, red-team regression). On success, pipeline opens PR to `env/prod` manifest.
4. CAB approval merges PR; ArgoCD syncs to production. Error Budget Guard rails may block sync if SLO budget exhausted.
5. Argo Rollouts handles canary; metrics guard (e.g., Prometheus) halts rollout if thresholds breached.

---

## 5. Testing & Evaluation Gates

| Gate | Description | Tooling | Pass Criteria |
| ---- | ----------- | ------- | ------------- |
| **Unit** | Python/TypeScript unit tests for pipeline helpers | `pytest`, `vitest` | Coverage ≥ 85%, zero failures |
| **Contract** | API schema compatibility, guardrail interface | `schemathesis`, Postman | No breaking changes |
| **Integration** | Retrieval + generation end-to-end | Langfuse test harness | Success rate ≥ 98% |
| **Red-team Regression** | Attack suite from Week 7 | `redteam.harness` CLI | Block rate ≥ 95% |
| **Continuous Evaluation** | Goldset QA, rubric scoring | `evals.pipeline` | Quality score >= threshold |
| **Load/Perf** | Soak test on canary | Locust/K6 | p95 latency ≤ SLO target |

> **Automation Rule:** Promotion is blocked until all gates produce evidence artifacts stored in `artifacts/` with release ID.

---

## 6. Security & Compliance in CI/CD

- **Artifact Signing:** Use Sigstore Cosign or in-toto to sign containers, prompt bundles, guardrail policies.
- **SBOM Generation:** `syft` or `cyclonedx` to produce SBOM and upload to artifact repository.
- **Secrets Hygiene:** Prefer OIDC workload identity over long-lived secrets in CI. Rotate tokens automatically.
- **Policy as Code:** Leverage Open Policy Agent/Rego to enforce deployment policies (e.g., no public ingress in prod).
- **Audit Trails:** Log build metadata and approvals to compliance vault (e.g., ServiceNow ticket references).

---

## 7. Workshop Agenda

1. **Pipeline Inventory (15 min)** – Identify artifacts and owners.
2. **Design Review (30 min)** – Map current pipeline to desired state diagram.
3. **Hands-on Lab (60 min)** – Scaffold GitHub Actions/ArgoCD config in team repos.
4. **Security Review (30 min)** – Embed scanning/signing steps.
5. **Promotion Simulation (30 min)** – Walk through hypothetical release with evidence capture.

---

## 8. Deliverables

- CI/CD architecture diagram stored under `docs/architecture/week-09/`
- Pipeline definition files (`.github/workflows/`, `argocd/`, `env/`) updated with release ID pattern
- Release evidence checklist updated in `resources/deployment-readiness-checklist.md`
- Signed build artifacts pushed to container registry and artifact storage

---

## 9. Discussion & Reflection

- How do we prevent "prompt hotfixes" that bypass CI/CD?
- What is our rollback strategy if a prompt update regresses quality but passes tests?
- How will we share release status with stakeholders (dashboards, status page)?
- Which parts of the pipeline should be templatized for future GenAI workloads?

---

## 10. Prep for Next Lesson

- Ensure monitoring hooks emit metrics referenced by SLOs (latency, error rate, quality).
- Tag release artifacts with environment, persona impact, and risk classification.
- Document outstanding pipeline gaps and owners in the risk register.

> Next session: We will operationalize observability pipelines, drift detection, and incident response mechanics.
