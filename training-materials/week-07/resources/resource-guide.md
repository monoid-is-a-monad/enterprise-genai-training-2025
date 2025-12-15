# Week 7 Resource Guide – Observability, Guardrails, and Security

**Purpose:** Supplement Lessons 1–4 and Labs 1–3 with curated references, templates, and quick-start commands.

---

## 1. Tooling Cheatsheets

### 1.1 Langfuse CLI & API
```
# Export traces for analysis
langfuse traces export --project <project-id> --out traces.jsonl

# Create a custom metric via API (example using httpie)
http POST https://cloud.langfuse.com/api/public/metrics \
  Authorization:"Bearer $LANGFUSE_API_KEY" \
  name="rag.daily.cost" value:=12.34 timestamp="$(date -Iseconds)"
```

**Docs:**
- Langfuse Quickstart – https://langfuse.com/docs
- OpenTelemetry ↔ Langfuse Exporter – https://langfuse.com/docs/integrations/opentelemetry

### 1.2 Guardrails & Presidio Commands
```
# Download spaCy model required by Presidio analyzer
python -m spacy download en_core_web_lg

# Validate guardrails schema
python -m guardrails.validator ./guard-rail-spec.yaml
```

**Docs:**
- Guardrails AI – https://www.guardrailsai.com/docs/latest
- Presidio Analyzer & Anonymizer – https://microsoft.github.io/presidio

### 1.3 Red Team Toolkit
```
# Run nightly adversarial test (async entry point from Lab 3)
python -m week_07.red_team.nightly_job --config configs/nightly.yaml

# Inspect stored artifacts
ls -lah artifacts/redteam_run.jsonl
```

---

## 2. Reference Architectures

| Component | Purpose | Notes |
|-----------|---------|-------|
| Langfuse + Prometheus | Unified tracing, cost analytics | Export spans to OTLP → Prometheus remote write |
| Guardrail Pipeline | Layered moderation, PII, rules | Reuse `GuardrailPipeline` scaffolding from Lab 2 |
| Security Testing Harness | Automated adversarial regression | Integrate with CI (GitHub Actions cron or Azure DevOps schedule) |

> **Diagram Tip:** Combine these components in a swimlane diagram: *Client → API Gateway → Guardrail Pipeline → LLM Service → Langfuse/Telemetry*.

---

## 3. Checklists

### 3.1 Observability Readiness (Lesson 1–2)
- [ ] Environment variables for Langfuse/OpenAI stored in secret manager
- [ ] Traces tagged with `user_id`, `feature`, `environment`
- [ ] Cost aggregation job scheduled (daily)

### 3.2 Guardrail Deployment (Lesson 3 / Lab 2)
- [ ] Blocklist + moderation ensemble tested on top 20 risky prompts
- [ ] PII redaction verified against synthetic dataset
- [ ] Custom rule engine reviewed by compliance team

### 3.3 Security & Red Team (Lesson 4 / Lab 3)
- [ ] Attack template library versioned in repo
- [ ] Nightly red team run produces coverage report
- [ ] Incident severity matrix signed off by security leadership

---

## 4. Templates & Snippets

### 4.1 Guardrail Configuration Skeleton (`guardrail-config.yaml`)
```yaml
moderation:
  providers:
    - type: blocklist
      patterns:
        - "(?i)ignore previous"
        - "(?i)make a bomb"
    - type: openai
      model: omni-moderation-latest
pii:
  analyzer: presidio
  anonymizer:
    strategy: replace
rules:
  - name: no_financial_advice
    severity: critical
    predicate: "context.user_segment == 'retail' and 'financial advice' in context.intent"
  - name: beta_feature_only
    severity: warning
    predicate: "context.feature == 'beta_tool' and not context.beta_whitelist"
```

### 4.2 Incident Ticket Template
```
Title: [LLM Incident] {{attack_family}} – {{priority}}
Severity Score: {{score}}
Prompt Snippet: {{prompt_excerpt}}
Guardrail Outcome: {{guardrail_action}}
Immediate Actions:
- [ ] Kill switch toggled (yes/no)
- [ ] Stakeholders notified (list)
Mitigation Owner: {{owner}} | Due: {{due_date}}
```

---

## 5. External Reading List

| Topic | Link | Why It Matters |
|-------|------|----------------|
| OWASP Top 10 for LLM Apps | https://owasp.org/www-project-top-10-for-large-language-model-applications/ | Aligns with Lesson 4 threat landscape |
| NIST AI Risk Management Framework | https://www.nist.gov/itl/ai-risk-management-framework | Governance alignment |
| Anthropic Red Teaming Playbook | https://www.anthropic.com/research/red-teaming | Advanced adversarial tactics inspiration |
| OpenAI Safety Best Practices | https://platform.openai.com/docs/guides/safety-best-practices | Practical guardrail considerations |
| Azure OpenAI Responsible AI | https://learn.microsoft.com/azure/cognitive-services/openai/concepts/responsible-use | Enterprise compliance perspective |

---

## 6. Suggested Assignments & Extensions
- Expand Lab 2 guardrail pipeline to integrate third-party DLP API
- Instrument Lab 1 RAG pipeline with Langfuse annotations for guardrail outcomes
- Fork Lab 3 red team harness to support Azure OpenAI, Cohere, or local llamas
- Draft quarterly executive report using Lab 3 Exercise 8 template with live data

---

## 7. Support Channels
- **#genai-observability** (Slack) – instrumentation questions
- **#genai-guardrails** – policy/guardrail tuning
- **Security Guild Office Hours** – Wednesdays @ 10:00 AM PT
- **Confluence Space:** `Enterprise GenAI / Week 07 Ops Toolkit`

---

> Keep this guide updated after each cohort. Capture new attack patterns, mitigation lessons, and tooling changes to maintain a living knowledge base.
