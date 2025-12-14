# Week 7: Observability, Tracing & Guardrails

**Provided by:** ADC ENGINEERING & CONSULTING LTD

**Duration:** 20 hours

## Overview

This week focuses on production-grade observability, distributed tracing, guardrails, and security for LLM applications. You'll learn how to instrument your AI systems with tools like Langfuse, implement content moderation and safety guardrails, and defend against prompt injection and other security threats.

## Learning Objectives

- [ ] Understand observability vs monitoring in LLM applications
- [ ] Implement distributed tracing with Langfuse or similar tools
- [ ] Design and deploy guardrails for content safety and compliance
- [ ] Detect and prevent prompt injection attacks
- [ ] Build PII detection and redaction systems
- [ ] Set up comprehensive logging and metrics collection
- [ ] Create alerting systems for LLM failures and anomalies
- [ ] Perform red team exercises to test system resilience
- [ ] Optimize LLM application performance using trace data
- [ ] Implement cost tracking and attribution

## Content Structure

### Lessons

1. **LLM Observability Fundamentals** — [lessons/01-llm-observability-fundamentals.md](./lessons/01-llm-observability-fundamentals.md)
   - Observability vs monitoring
   - Key metrics for LLM systems
   - Logging best practices
   - Structured logging and correlation IDs
   - Metrics collection (latency, tokens, cost)
   - Alerting and anomaly detection

2. **Distributed Tracing with Langfuse** — [lessons/02-distributed-tracing-with-langfuse.md](./lessons/02-distributed-tracing-with-langfuse.md)
   - Trace and span model
   - Langfuse architecture and setup
   - Instrumenting LLM applications
   - Trace visualization and analysis
   - Performance optimization with traces
   - Cost tracking and attribution

3. **Guardrails & Safety Systems** — [lessons/03-guardrails-and-safety-systems.md](./lessons/03-guardrails-and-safety-systems.md)
   - Content moderation strategies
   - PII detection and redaction
   - Custom guardrails implementation
   - Output validation and filtering
   - Toxicity detection
   - Compliance and regulatory requirements

4. **Prompt Security & Attack Mitigation** — [lessons/04-prompt-security-and-attack-mitigation.md](./lessons/04-prompt-security-and-attack-mitigation.md)
   - Prompt injection types and examples
   - Jailbreaking techniques
   - Defense strategies and best practices
   - Input sanitization
   - Red team exercises
   - Security testing frameworks

### Labs

1. **Implementing Distributed Tracing** — [labs/lab-01-implementing-distributed-tracing.ipynb](./labs/lab-01-implementing-distributed-tracing.ipynb)
   - Exercise 1: Basic Langfuse setup and instrumentation
   - Exercise 2: Tracing a multi-step RAG pipeline
   - Exercise 3: Cost tracking and attribution by user
   - Exercise 4: Performance analysis and optimization
   - Exercise 5: Custom metrics and dashboards
   - Exercise 6: Alerting on trace anomalies
   - Exercise 7: Integrating with OpenTelemetry

2. **Building Guardrails Systems** — [labs/lab-02-building-guardrails-systems.ipynb](./labs/lab-02-building-guardrails-systems.ipynb)
   - Exercise 1: Content moderation with OpenAI Moderation API
   - Exercise 2: PII detection and redaction
   - Exercise 3: Custom guardrail rules engine
   - Exercise 4: Output validation and filtering
   - Exercise 5: Toxicity detection with HuggingFace models
   - Exercise 6: Guardrail orchestration and chaining
   - Exercise 7: Performance optimization for guardrails
   - Exercise 8: Compliance reporting system

3. **Security Testing & Red Teaming** — [labs/lab-03-security-testing-and-red-teaming.ipynb](./labs/lab-03-security-testing-and-red-teaming.ipynb)
   - Exercise 1: Prompt injection attack simulation
   - Exercise 2: Jailbreak attempt detection
   - Exercise 3: Input sanitization strategies
   - Exercise 4: Defense layer implementation
   - Exercise 5: Automated security testing suite
   - Exercise 6: Red team playbook creation
   - Bonus: Building a prompt firewall

### Exercises

> **Note:** Exercises are comprehensively covered in the labs with hands-on implementations.

## Tools & Libraries

```python
# Observability
langfuse>=2.0.0              # Primary tracing solution
opentelemetry-api>=1.20.0    # OpenTelemetry integration
prometheus-client>=0.19.0    # Metrics collection

# Guardrails
guardrails-ai>=0.4.0         # Guardrails framework
presidio-analyzer>=2.2.0     # PII detection
presidio-anonymizer>=2.2.0   # PII redaction
transformers>=4.35.0         # Toxicity detection models

# Security
langkit>=0.0.20              # LLM security toolkit
rebuff>=0.0.1                # Prompt injection detection

# Utilities
openai>=1.0.0
pydantic>=2.0.0
httpx>=0.24.0
python-dotenv
```

## Prerequisites

Before starting Week 7, ensure you have completed:

- Week 3: RAG Fundamentals (understanding of RAG pipelines)
- Week 6: Function Calling & Tool Integration (tool orchestration patterns)
- Familiarity with async Python
- Understanding of HTTP APIs and webhooks

## Setup Instructions

### 1. Langfuse Setup

```bash
# Option 1: Cloud (recommended for learning)
# Sign up at https://cloud.langfuse.com
# Get your API keys from the dashboard

# Option 2: Self-hosted (Docker)
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker-compose up -d
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Variables

```bash
# .env file
OPENAI_API_KEY=your_openai_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com  # or your self-hosted URL
```

## Week Structure

- **Monday**: Lesson 1 (Observability Fundamentals) + Lab 1 Setup
- **Tuesday**: Lesson 2 (Distributed Tracing) + Lab 1 Exercises 1-4
- **Wednesday**: Lab 1 Completion + Lesson 3 (Guardrails)
- **Thursday**: Lesson 4 (Security) + Lab 2 Exercises 1-5
- **Friday**: Lab 2 Completion + Lab 3 (Security Testing)

## Key Concepts

### Observability
- **Traces**: End-to-end request flows
- **Spans**: Individual operations within a trace
- **Metrics**: Quantitative measurements (latency, throughput)
- **Logs**: Discrete events with context

### Guardrails
- **Input Guardrails**: Validate and sanitize user inputs
- **Output Guardrails**: Filter and validate LLM outputs
- **Structural Guardrails**: Ensure correct format and structure
- **Behavioral Guardrails**: Prevent unwanted behaviors

### Security Layers
- **Prompt Injection Defense**: Detect and block injection attempts
- **PII Protection**: Identify and redact sensitive information
- **Content Filtering**: Block inappropriate or harmful content
- **Rate Limiting**: Prevent abuse and DoS attacks

## Real-World Applications

1. **Enterprise Chatbot Monitoring**
   - Track user conversations
   - Monitor response quality
   - Detect and block inappropriate content
   - Attribute costs to departments

2. **RAG System Observability**
   - Trace retrieval → generation flow
   - Optimize chunk selection
   - Monitor retrieval quality
   - Debug production issues

3. **AI Agent Security**
   - Monitor tool execution
   - Prevent unauthorized actions
   - Detect malicious prompts
   - Audit compliance

4. **Content Moderation Pipeline**
   - Multi-layer guardrails
   - PII detection and redaction
   - Regulatory compliance
   - Audit trail for decisions

## Success Metrics

By the end of Week 7, you should be able to:

- ✅ Instrument a complete LLM application with tracing
- ✅ Build a multi-layer guardrails system
- ✅ Detect and mitigate prompt injection attacks
- ✅ Create comprehensive dashboards for LLM observability
- ✅ Implement cost tracking and attribution
- ✅ Set up alerting for anomalies and failures
- ✅ Pass red team security tests
- ✅ Meet compliance requirements for PII handling

## Resources

- [resources/README.md](./resources/README.md) — Quick reference guides
- [resources/observability-patterns-guide.md](./resources/observability-patterns-guide.md)
- [resources/guardrails-implementation-guide.md](./resources/guardrails-implementation-guide.md)
- [resources/security-checklist.md](./resources/security-checklist.md)
- [resources/examples/](./resources/examples/) — Complete implementations

## Additional Reading

- [Langfuse Documentation](https://langfuse.com/docs)
- [OpenTelemetry for LLMs](https://opentelemetry.io/)
- [OWASP Top 10 for LLMs](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Guardrails AI Documentation](https://docs.guardrailsai.com/)
- [Presidio PII Detection](https://microsoft.github.io/presidio/)

## Next Week Preview

**Week 8: PoC #1 - Technical Integration & Demo**
- Integrate all concepts from Weeks 1-7
- Build a production-ready proof of concept
- Implement end-to-end observability
- Security hardening and compliance
- Performance optimization
- Demo preparation and presentation

---

**Questions or Issues?**
- Open an issue in the training repository
- Reach out during office hours
- Check the resources directory for additional guidance
