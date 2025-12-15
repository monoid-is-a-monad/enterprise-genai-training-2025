# Week 9 Deployment Readiness Checklist

Use this checklist to verify the production launch package before requesting CAB approval. Attach evidence links for every item. Update as tasks complete.

| Category | Item | Owner | Evidence Link | Status | Notes |
| -------- | ---- | ----- | ------------- | ------ | ----- |
| Architecture | Production topology diagram (multi-region / blue-green) approved | Platform Lead |  | [ ] |  |
| Architecture | Capacity and scaling envelopes documented | SRE Lead |  | [ ] |  |
| Architecture | Secrets vault rotation schedule implemented | Security Lead |  | [ ] |  |
| CI/CD | GitHub Actions / pipeline run succeeded for release ID | DevOps |  | [ ] |  |
| CI/CD | Signed artifacts stored in registry with SBOM attached | DevOps |  | [ ] |  |
| CI/CD | Promotion manifest PR merged with approvals | DevOps |  | [ ] |  |
| Testing | Unit + integration + regression suites green | QA Lead |  | [ ] |  |
| Testing | Red-team regression report uploaded (≥95% block rate) | Security Lead |  | [ ] |  |
| Testing | Load test results attached (p95 latency ≤ target) | Perf Team |  | [ ] |  |
| Observability | SLO dashboard published with live data | SRE Lead |  | [ ] |  |
| Observability | Drift detection job scheduled with alert routing | Applied Science |  | [ ] |  |
| Observability | Incident on-call rotation updated in PagerDuty | Ops Manager |  | [ ] |  |
| Compliance | Data retention & residency controls signed off | Compliance Officer |  | [ ] |  |
| Compliance | Audit trail for guardrail/prompt changes archived | Security Lead |  | [ ] |  |
| Compliance | CAB ticket includes risk assessment & mitigations | Release Manager |  | [ ] |  |
| Communications | Launch announcement draft approved | Comms Lead |  | [ ] |  |
| Communications | Support playbook updated (FAQ, escalation paths) | Support Lead |  | [ ] |  |
| Communications | Status page templates prepared (launch + incident) | Comms Lead |  | [ ] |  |
| Rollback | Automated rollback tested and results logged | DevOps |  | [ ] |  |
| Rollback | Manual recovery procedures validated | Incident Commander |  | [ ] |  |
| Governance | Go/no-go meeting scheduled with stakeholders | Release Manager |  | [ ] |  |
| Governance | Post-launch metrics review cadence scheduled | Product Lead |  | [ ] |  |

## How to Use

1. Duplicate this table per release and store under `docs/readiness/week-09/`.
2. Replace blank evidence cells with links to dashboards, reports, tickets, or documents.
3. Mark status `[x]` once evidence is verified by reviewers.
4. Raise risks or blockers in the Week 9 risk register and assign owners.
5. Submit completed checklist with CAB packet and include summary in standup updates.
