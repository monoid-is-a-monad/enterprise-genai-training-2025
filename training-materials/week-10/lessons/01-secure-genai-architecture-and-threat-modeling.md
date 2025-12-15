# Lesson 01 – Secure GenAI Architecture & Threat Modeling

**Session Length:** 3.5 hours (90 min lecture + 60 min workshop + 60 min mitigation lab)

---

## 1. Why Threat Modeling Matters for GenAI

Traditional application threat models miss attack vectors unique to GenAI systems. In Week 10 we expand beyond infrastructure hardening to consider:

- **Prompt injection & jailbreaks** targeting safety guardrails.
- **Model supply chain tampering** (weights, LoRA adapters, prompt libraries).
- **Retrieval poisoning** via malicious documents or vector store manipulation.
- **Telemetry gaps** that prevent detection of harmful outputs.
- **Data exfiltration** through unconstrained generation or logging.

> **Goal:** Produce a threat model that prioritizes risks, aligns mitigations with controls, and feeds directly into architecture updates and policy enforcement.

---

## 2. Threat Modeling Frameworks

### STRIDE Adapted for GenAI

| STRIDE Category | GenAI Example | Mitigation |
| --------------- | ------------- | ---------- |
| **Spoofing** | Impersonating executive persona to bypass guardrails | Mutual TLS, signed persona tokens |
| **Tampering** | Modifying prompt templates in registry | Signed artifacts, IaC with approvals |
| **Repudiation** | Operator denies changing guardrail | Immutable audit logs, approvals |
| **Information Disclosure** | Model leaks sensitive data | Guardrails, PII masking, output filters |
| **Denial of Service** | Model overwhelmed by adversarial prompts | Rate limiting, workload isolation |
| **Elevation of Privilege** | Injection grants access to hidden tools | Policy-as-code, agent sandboxing |

### Additional LLM-Specific Considerations

- Model/Embedding provider outages => fallback strategy.
- Shadow prompt versions => enforce version control, signing.
- Low-signal telemetry => instrument guardrail decisions, persona context.

---

## 3. Security Architecture Patterns

```mermaid
graph TD
    subgraph ControlPlane[Control Plane]
        A[Policy Registry] --> B[OPA|Guardrails API]
        C[Audit Trail] --> D[Compliance Vault]
    end
    subgraph DataPlane[Data Plane]
        E[User Entry Points]
        F[Prompt Orchestrator]
        G[Guardrail Engine]
        H[LLM Providers]
        I[Vector Store]
    end
    E --> F --> G --> H
    F --> I
    B --> G
    G -->|Decisions| C
    H -->|Telemetry| C
```

**Controls to Highlight**
- **Network segmentation** (data plane vs control plane).
- **Secrets management** (short-lived tokens, HSM-backed keys).
- **Policy enforcement** via OPA, Guardrails, or custom middleware.
- **Telemetry sinks** capturing guardrail actions and LLM responses.
- **Fallback providers** with monitored SLAs.

---

## 4. Workshop: Building the Threat Model

1. **System Context Sketch (15 min)** – Identify actors, data stores, external dependencies.
2. **Enumerate Threats (30 min)** – Use STRIDE + LLM-specific list to surface risks.
3. **Assess Impact/Likelihood (15 min)** – Rate each threat (High/Med/Low) and attach existing controls.
4. **Define Mitigations (30 min)** – Assign owners, target dates, and control types.
5. **Architecture Update (30 min)** – Modify diagrams to reflect new controls (firewalls, logging, signing).

Use the `resources/security-risk-register-template.md` to capture outcomes.

---

## 5. Mitigation Patterns & Control Mapping

| Threat | Control Type | Example Control | Owner |
| ------ | ------------ | --------------- | ----- |
| Prompt injection to escalate privileges | Preventive | Prompt firewall rule, persona validation | Guardrail Lead |
| Model supply chain tampering | Detective/Preventive | Sigstore signing of weights and prompts | Platform Security |
| Retrieval poisoning | Preventive/Detective | Document ingestion validation, anomaly detection | Data Engineering |
| Confidential data leak | Corrective | Redaction filter, fallback to human review | Safety Team |
| Telemetry blind spots | Detective | Langfuse instrumentation, log correlation | Observability |

> **Reminder:** Controls must be testable. Define verification steps (CI checks, red-team suites, penetration tests).

---

## 6. Deliverables

- Updated threat model document stored in `docs/threat-models/week-10.md` (or similar).
- Architecture diagram with security controls annotated (Mermaid/Draw.io).
- Risk register entries with severity, mitigation, owner, due date.
- Summary for governance council highlighting top 3 risks and mitigation roadmap.

---

## 7. Discussion Prompts

- Which attack vectors are unique to your enterprise context (regulated data, proprietary IP)?
- How do we validate that mitigations remain effective after model or prompt updates?
- What telemetry gaps would hinder incident response? How do we close them?
- Which risks remain accepted? Document rationale and approval authority.

---

## 8. Homework & Lab Prep

- Finalize threat model prior to Lab 01 – you will implement selected controls.
- Gather existing security policies to align mitigation owners and sign-off requirements.
- Verify access to policy repositories, guardrail configs, and telemetry dashboards.

> Next lesson: Policy management and Responsible AI guardrails build on these mitigations by codifying the governance model.
