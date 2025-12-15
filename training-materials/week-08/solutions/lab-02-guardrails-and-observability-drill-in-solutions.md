# Lab 2 – Guardrails & Observability Drill-In (Solutions)

> Instructor reference for reviewing student notebooks. Highlights key checkpoints, expected outputs, and remediation guidance. Adjust to match your org's security/compliance posture.

---

## 1. Guardrail Configuration Audit

```python
from pathlib import Path
import yaml

policy_path = Path("config/guardrails/prompt-firewall.yaml")
policy = yaml.safe_load(policy_path.read_text())

rules = policy.get("rules", [])
print(f"Loaded {len(rules)} guardrail rules")
for rule in rules:
    print(f"- {rule['id']}: {rule['description']}")
```

**Instructor checklist**
- Ensure policies cover injection, data exfiltration, persona misuse, toxicity/harassment, and compliance topics (PCI/PII).
- Look for explicit allowlists/denylists tied to demo personas.
- Confirm students maintain a change log with rationale and approvals.

---

## 2. Execute Red-Team Suite

```python
import json
from pathlib import Path

from redteam.harness import PromptAttackSuite

suite = PromptAttackSuite.from_catalog(Path("resources/sample-attack-templates.json"))
results = suite.run(target_url="https://poc.example.com/api/query", persona="executive")

report_path = Path("reports") / f"redteam-report-{suite.timestamp}.json"
report_path.parent.mkdir(exist_ok=True)
report_path.write_text(json.dumps(results, indent=2))
print(f"Saved red-team report to {report_path}")
```

**Instructor checklist**
- Verify `results` include block vs bypass counts per category.
- Encourage storing raw responses for forensic review (mask secrets first).
- Require risk register updates for any bypass (severity, owner, due date).

---

## 3. Gap Analysis & Mitigation Planning

| Attack Category | Attempts | Blocked | Bypassed | Mitigation | Owner | ETA |
| --------------- | -------- | ------- | -------- | ---------- | ----- | --- |
| Prompt Injection | 18 | 17 | **1** | Added SQL control rule, escalated fallback | Security lead | 2025-12-14 |
| Data Exfiltration | 12 | 12 | 0 | N/A | Security lead | — |
| Role Switching | 10 | 9 | **1** | Persona clamp, extra validation | Backend lead | 2025-12-15 |
| Toxic Content | 15 | 15 | 0 | N/A | Safety lead | — |

**Instructor checklist**
- Confirm mitigations describe *how* to remediate, not just "rerun after fix".
- Ensure owners align with RACI (Week 8 Lesson 1).
- If mitigations slip, require waivers signed by stakeholders.

---

## 4. Guardrail Tuning & Re-Test

```python
policy["rules"].append({
    "id": "block-sql-control",
    "description": "Block SQL control characters in persona requests",
    "type": "regex",
    "pattern": r"(--|;|\\bDROP\\b|\\bUNION\\b)",
    "action": "block",
})

policy_path.write_text(yaml.safe_dump(policy))
print("Added rule block-sql-control. Re-run red-team suite to validate.")
```

**Instructor checklist**
- Students should re-run the suite and attach updated report.
- Approvals from security/compliance must be logged before changes ship.
- Encourage versioning guardrail policies or storing in dedicated repo.

---

## 5. Observability Enrichment

```python
from observability.metrics import MetricsClient

metrics_client = MetricsClient()
window = metrics_client.time_window(hours=12)

data = metrics_client.fetch(
    metrics=["latency_p95_ms", "guardrail_block_rate", "error_rate"],
    window=window,
    filters={"service": "risk-analyst-pipeline"},
)

print("Telemetry snapshot:")
for metric, value in data.items():
    print(f"- {metric}: {value}")

if data["guardrail_block_rate"] < 0.95:
    raise RuntimeError("Guardrail block rate below threshold. Investigate before demo.")
```

**Instructor checklist**
- Confirm metrics pipeline differentiates success/fallback/blocked outcomes.
- Require guardrail block rate ≥ 95% and telemetry freshness ≤ 10 minutes.
- Students should attach Langfuse trace and dashboard screenshots for evidence.

### Dashboard verification prompts
- Highlight guardrail decision tags in trace explorer.
- Surface latency/error/block rate panels on a single Grafana/Looker page.
- Store artifacts in demo readiness folder for stakeholder review.

---

## 6. Readiness Scorecard Update

| Metric | Target | Actual | Status | Notes |
| ------ | ------ | ------ | ------ | ----- |
| Guardrail block rate | ≥ 95% | 97.8% | 🟢 | After SQL rule added |
| Telemetry freshness (minutes) | ≤ 10 | 4 | 🟢 | Langfuse sink stable |
| Trace coverage | ≥ 90% | 92% | 🟢 | All hero flows instrumented |

**Instructor checklist**
- Require linking to raw data sources (report path, dashboard URL).
- If any metric < target, add mitigation and due date before dry run sign-off.

---

## 7. Retrospective & Checklist

| Topic | Observation | Owner | Follow-up |
| ----- | ----------- | ----- | --------- |
| Guardrail coverage | Two injections slipped until regex tightened | Security lead | Monitor new prompts daily |
| Observability gaps | Missing fallback tag in traces | SRE lead | Added span attribute `fallback_used` |
| Tooling improvements | Red-team CLI lacks persona switch | Platform | Add flag for next release |
| Stakeholder questions | Legal asked about log retention | Program manager | Update FAQ with policy |

**Submission checklist**
- Red-team report stored in `reports/` and referenced in risk register.  
- Guardrail policy diff reviewed/approved.  
- Langfuse trace + metrics screenshots archived.  
- `resources/readiness-scorecard.csv` updated (guardrail rows).  
- Daily brief posted with summary/asks.  
- FAQ/objection log updated with security insights.

---

**Instructor notes**
- Encourage teams to wrap red-team runs in CI (nightly) for regression detection.
- Promote capturing evidence of compliance (data residency, retention) for Week 10 governance content.
- Stress that guardrail changes should be reversible (feature flags or versioned configs).
