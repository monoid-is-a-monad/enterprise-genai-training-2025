# Lesson 4 – Prompt Security & Attack Mitigation

**Estimated time:** 90 minutes  
**Audience:** LLM platform engineers, security engineers, red teams  
**Dependencies:** Completion of Lessons 1–3 and Labs 1–2

---

## Learning Objectives
- Identify the most prevalent prompt-based attack vectors against enterprise LLM systems
- Design layered mitigations that combine guardrails, access controls, and runtime monitoring
- Implement continuous validation using adversarial testing and drift detection signals
- Establish operational playbooks for incident response, stakeholder communication, and postmortems

---

## Agenda
1. Threat Landscape Recap
2. Anatomy of Prompt Injection & Jailbreak Attacks
3. Mitigation Layers (Prevent, Detect, Respond)
4. Automated Adversarial Testing Workflows
5. Incident Response & Compliance Considerations
6. Hands-On Exercise Walkthrough
7. Checklist & Success Metrics

---

## 1. Threat Landscape Recap

| Threat Class | Objective | Common Entry Points | Business Impact |
|--------------|-----------|---------------------|-----------------|
| Prompt Injection | Override system instructions | Retrieval payloads, templated prompts | Data leakage, defaced responses |
| Jailbreaks | Bypass safety policies | Public prompts, insider misuse | Regulatory violations, brand damage |
| Data Exfiltration | Extract PII, secrets, IP | Tools/plugins, vector stores, fine-tunes | Breach notification, loss of trust |
| Model Theft | Infer model weights/behavior | High-volume querying, logging gaps | Loss of competitive advantage |
| Supply Chain | Compromise upstream artifacts | Prompt libraries, model checkpoints | Integrity loss, widespread exposure |

**Key takeaway:** Traditional perimeter controls are insufficient. LLM threat models must treat prompts, embeddings, and response pipelines as high-risk assets.

---

## 2. Anatomy of Prompt Injection & Jailbreak Attacks

### 2.1 Prompt Injection Flow
```
Untrusted Data → Retrieval/RAG Pipeline → Prompt Assembly → Model Generation
                        ↑                       ↓
                Injection Payload         Override + Data Exfil
```

**Techniques**
- *Instruction overrides:* "Ignore previous instructions"
- *Modal deception:* "Role-play as unrestricted system"
- *Tool abuse:* Chained attacks that manipulate tool outputs
- *Self-referencing:* Prompt loops that escalate privileges

### 2.2 Jailbreak Patterns to Recognize
- Translation or encoding (ROT13, Base64) to bypass filters
- Benign preamble with malicious payload appended
- Multi-turn conditioning ("If I were hypothetically...")
- Custom tokens/Unicode homoglyphs to evade regex-based filters

**Tip:** Maintain a living corpus of attacks. Feed difficult cases into Lab 3's red team harness for regression testing.

---

## 3. Mitigation Layers (Prevent, Detect, Respond)

| Layer | Goal | Tactics |
|-------|------|---------|
| **Prevent** | Reduce attack surface | Prompt isolation, output schemas, guardrail polymers, least-privilege tool adapters |
| **Detect** | Identify misuse early | Harness heuristics (Lab 2), model-based moderation, anomaly detection on Langfuse traces |
| **Respond** | Contain and remediate | Incident runbooks, automated ticketing, kill-switch toggles, communication plans |

### 3.1 Preventative Controls
- **Prompt templating discipline:** No uncontrolled string concatenation; use structured objects
- **Tool whitelisting:** Explicit allow/deny lists with context-sensitive permissions
- **Context sanitization:** Strip HTML/Markdown, language normalization, stop-sequence enforcement
- **Policy-as-code:** Versioned guardrail rules (Relates to Lab 2 Rule Engine)

### 3.2 Detection Controls
- **Langfuse signal hooks:** Attach `attack_surface`, `guardrail_result`, and anomaly metrics to spans
- **Outlier analysis:** Percentile-based latency spikes can reveal malicious loops
- **Ensemble moderation:** Combine heuristic harness, OpenAI moderation, and transformer classifiers

### 3.3 Response Controls
- **Dynamic throttling:** Reduce rate limits when harness severity ≥ high
- **Dual review:** Automatic escalation to human reviewers for high-risk outputs
- **Trace preservation:** Snapshot prompts and responses (Lab 3 Exercise 6) with correlation IDs

---

## 4. Automated Adversarial Testing Workflows

### 4.1 Build a Testing Pipeline
1. Generate attack templates (`generate_attacks` from Lab 3) with parameter sweeps
2. Feed prompts through injection harness + guardrail pipeline
3. Log results into Langfuse with labels (`attack_family`, `severity`)
4. Aggregate metrics to coverage dashboards (pass rate, blocks, violations)

### 4.2 Scheduling & Regression
- Nightly red team run in CI with baseline attack set
- Pre-release gating: require ≥95% block rate for critical categories
- Drift detection: compare weekly violation counts; auto-open Jira tickets on upward trends

### 4.3 Sample Automation Snippet
```python
async def nightly_adversarial_job(attacks, pipeline):
    run = RedTeamRun(attacks, pipeline.harness, pipeline.evaluator)
    await run.execute(concurrency=8)
    coverage = run.coverage_report()
    report = build_report({
        "run_id": f"nightly-{datetime.utcnow():%Y%m%d}",
        "total_attacks": len(attacks),
        "blocks": sum(a["analysis"]["recommended_action"] == "block" for a in run.results),
        "families": coverage,
    }, incidents=[])
    export_metrics(coverage, destination="stdout")
    export_report(report, "artifacts/nightly-report.md")
```

---

## 5. Incident Response & Compliance Considerations

### 5.1 Preparation
- Define severity matrix (aligns with Lab 3 Exercise 7)
- Pre-register on-call rotations for security, legal, comms
- Maintain secure storage for artifacts with encryption-at-rest

### 5.2 During an Incident
1. Trigger: guardrail pipeline returns `blocked=True` with severity `critical`
2. Contain: disable affected feature flag, throttle relevant endpoints
3. Communicate: notify stakeholders using predefined template
4. Collect evidence: use `snapshot_artifacts` to capture payloads

### 5.3 Post-Incident
- Run blameless postmortem within 72 hours
- Update guardrail rules/tests based on root cause
- File compliance reports (GDPR/CCPA) if PII confirmed

**Checklist for Readiness**
- [ ] Guardrail pipeline emits correlation IDs for every request
- [ ] Incident runbook reviewed quarterly with security & legal
- [ ] Executive report template (Lab 3 Exercise 8) filled for last red team run

---

## 6. Hands-On Exercise Walkthrough

| Exercise | Objective | References |
|----------|-----------|------------|
| Threat Modeling | Populate catalog with new threat surfaces (e.g., fine-tune poisoning) | Lab 3 Exercise 1 |
| Guardrail Hardening | Extend harness with domain-specific heuristics | Lab 2 Exercise 1 & 2 |
| Automated Testing | Add new attack templates and rerun orchestrator | Lab 3 Exercise 5 |
| Incident Simulation | Practice severity scoring & ticket escalation | Lab 3 Exercise 7 |

Encourage learners to pair up: one team crafts novel attacks, the other enhances defenses. Swap roles after each iteration to highlight attacker-defender dynamics.

---

## 7. Checklist & Success Metrics

### Technical Controls
- [ ] Guardrail pipeline covers moderation, PII, rules, toxicity, orchestration
- [ ] Langfuse dashboards track `guardrail.block_rate` and `attack_family_coverage`
- [ ] Automation triggers nightly red team regression

### Process & Compliance
- [ ] Incident response playbook maintained and tested
- [ ] Mitigation plans tied to owners, deadlines, and executive reporting cadence
- [ ] Training artifacts (lesson + labs) stored in knowledge base

### KPIs to Monitor
- Mean time to detection (MTTD) for prompt-based incidents
- % of attacks blocked vs. total attempts per family
- Reduction in manual review workload after harness improvements

---

## Further Reading & Resources
- NIST AI Risk Management Framework – Mitigation best practices
- OWASP Top 10 for Large Language Model Applications (OWASP-LLM-1 through 10)
- Anthropic "Constitutional AI" and OpenAI "System Card" for policy design patterns
- Langfuse documentation: Custom metrics, annotations, and alerting hooks
- Presidio + Guardrails AI combo recipes (link to internal wiki if applicable)

---

### Review Questions
1. Which layers of defense would you adjust first after a spike in jailbreak attempts? Why?
2. How can Langfuse traces help differentiate between accidental misuse and targeted attacks?
3. What artifacts must be collected to support legal/compliance teams during an incident?
4. How do you ensure adversarial tests remain representative as attackers evolve?

---

> **Next Steps:** Proceed to Lab 3 to practice automated security testing and red teaming. Ensure all guardrail enhancements from Lab 2 are integrated before executing the attack suites.
