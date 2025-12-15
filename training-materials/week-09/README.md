# Week 9: MLOps & Production Deployment

**Provided by:** ADC ENGINEERING & CONSULTING LTD

**Duration:** 20 hours

## Overview

Week 9 transitions the Week 8 proof of concept into a production-ready GenAI service. The emphasis shifts from integration and demo polish to hardened operations: preparing infrastructure-as-code releases, enforcing CI/CD guardrails, deploying scalable serving stacks, and instrumenting continuous monitoring across models, data pipelines, and downstream applications. Teams will define service-level objectives (SLOs), codify change management workflows, and establish runbooks for incidents, drift, and rapid rollback. The outcome is an enterprise-compliant launch plan coupled with guardrails that keep the assistant trustworthy at scale.

## Learning Objectives

- [ ] Translate PoC architecture into production environment blueprints (Kubernetes, serverless, or VM-based)
- [ ] Design CI/CD pipelines that build, test, scan, and promote GenAI services safely
- [ ] Package models, prompts, and retrieval assets with versioning and rollback strategies
- [ ] Implement continuous evaluation, model/data drift detection, and automated alerting
- [ ] Operationalize Langfuse/LangSmith traces, metrics, and logs into SLO dashboards
- [ ] Harden secrets management, compliance logging, and access policies for production
- [ ] Establish gated release processes, CAB approvals, and progressive rollout playbooks
- [ ] Author incident response runbooks and on-call rotations tailored to GenAI failure modes
- [ ] Align stakeholders on go-live criteria, communication plans, and post-launch governance

## Content Structure

### Lessons

1. **Production Architecture Hardening & SLO Design** - [lessons/01-production-architecture-hardening-and-slo-design.md](./lessons/01-production-architecture-hardening-and-slo-design.md)
   - Environment topologies (multi-region, blue/green, feature flags)
   - Reliability targets and SLO/SLA definitions for GenAI workloads
   - Capacity planning, autoscaling envelopes, and cost guardrails
   - Compliance checkpoints (PII, audit logging, data residency)

2. **CI/CD & Model Delivery Pipelines** - [lessons/02-cicd-and-model-delivery-pipelines.md](./lessons/02-cicd-and-model-delivery-pipelines.md)
   - Build pipelines for code, prompts, retrieval indexes, and guardrails
   - Automated testing layers (unit, contract, load, red-team regression)
   - Supply chain security (SBOMs, signing, artifact provenance)
   - Promotion workflows across dev/stage/prod with human-in-the-loop approvals

3. **Observability, Drift Detection & Incident Response** - [lessons/03-observability-drift-detection-and-incident-response.md](./lessons/03-observability-drift-detection-and-incident-response.md)
   - Langfuse dashboards, tracing taxonomy, and metric up-leveling
   - Data/model drift detection pipelines and continuous evaluation harnesses
   - Alert routing, on-call runbooks, and severity definitions for GenAI incidents
   - Post-incident review templates and governance integration

4. **Release Management & Stakeholder Operations** - [lessons/04-release-management-and-stakeholder-operations.md](./lessons/04-release-management-and-stakeholder-operations.md)
   - Change advisory board (CAB) playbooks and risk assessments
   - Communication cadences for launch, incident, and rollback scenarios
   - Readiness scorecards, go/no-go meetings, and compliance sign-offs
   - Post-launch success metrics, adoption funnels, and roadmap acceleration

### Labs

1. **Automated Build & Release Pipeline** - [labs/lab-01-automated-build-and-release-pipeline.ipynb](./labs/lab-01-automated-build-and-release-pipeline.ipynb)
2. **Model Monitoring & Drift Alerts** - [labs/lab-02-model-monitoring-and-drift-alerts.ipynb](./labs/lab-02-model-monitoring-and-drift-alerts.ipynb)
3. **Production Rollout & Incident Playbook** - [labs/lab-03-production-rollout-and-incident-playbook.ipynb](./labs/lab-03-production-rollout-and-incident-playbook.ipynb)

### Exercises

> **Note:** Exercises are embedded inside the labs as deployment checkpoints and operational drills.

## Tools & Libraries

```python
# Deployment & automation
terraform>=1.6.0
pulumi>=3.80.0
awscli>=2.15.0
azure-cli>=2.60.0
gcloud>=469.0.0
kubectl>=1.29.0
helm>=3.13.0
argocd>=2.9.0

# CI/CD & packaging
github-actions>=1.0.0
mlflow>=2.10.0
wandb>=0.16.0
fastlane>=2.220.0

# Observability & evaluation
langfuse>=2.0.0
prometheus-client>=0.20.0
grafana-toolkit>=1.2.0
whylabs-client>=0.10.0
opentelemetry-sdk>=1.23.0

# Security & compliance
trivy>=0.49.0
in-toto>=1.5.0
snyk>=1.1200.0
hashicorp-vault>=2.1.0
```

## Prerequisites

- Week 8 PoC integrated with guardrails, observability, and demo assets
- Infrastructure access (cloud subscription or on-prem cluster) with IaC permissions
- Container registry credentials and secret management aligned with security policies
- Langfuse workspace connected to production telemetry sinks
- Defined stakeholder roster (product, security, compliance, support) for launch planning

## Delivery Cadence

- **Monday Morning:** Production architecture review & SLO drafting workshop
- **Monday Afternoon:** CI/CD pipeline scaffolding and IaC baseline setup
- **Tuesday:** Automated testing, security scanning, and artifact signing integration
- **Wednesday:** Monitoring/drift instrumentation and alert routing configuration
- **Thursday Morning:** Release readiness dry run (deployment + rollback tests)
- **Thursday Afternoon:** Incident response table-top exercise & CAB pre-read
- **Friday Morning:** Go/no-go meeting with stakeholders and launch comms dry run
- **Friday Afternoon:** Production rollout rehearsal and documentation packaging

## Success Criteria

By the end of Week 9 you should have:

- ✅ Production environment blueprints (IaC, network diagrams, scaling policies)
- ✅ CI/CD pipelines covering build, test, security scan, and progressive deployment stages
- ✅ Versioned model/prompt artifacts with rollback automation and audit logs
- ✅ Monitoring dashboards with SLO/SLA metrics and alerting thresholds in place
- ✅ Drift detection jobs and continuous evaluation hooks running on scheduled cadence
- ✅ Incident response runbooks, on-call rotations, and communication templates
- ✅ CAB-approved release plan with risk register updates and mitigation owners
- ✅ Launch communication assets (stakeholder emails, support FAQs, status page drafts)

## Resources

- [resources/deployment-readiness-checklist.md](./resources/deployment-readiness-checklist.md)
- [resources/mlops-operating-model.md](./resources/mlops-operating-model.md)
- [resources/production-slo-scorecard.csv](./resources/production-slo-scorecard.csv)
- [resources/change-management-communication-plan.md](./resources/change-management-communication-plan.md)
- [resources/incident-playbook-template.md](./resources/incident-playbook-template.md)

## Preparation Tips

- Keep production infrastructure configurations immutable and version controlled.
- Automate evidence capture (dashboards, logs, approvals) to streamline audits.
- Run chaos/rollback drills in lower environments before requesting CAB approval.
- Align communication channels with stakeholder expectations and regulatory requirements.

---

**Need help?** Join the MLOps office hours, raise GitHub issues with the DevOps team, or escalate via the support Slack channel.
