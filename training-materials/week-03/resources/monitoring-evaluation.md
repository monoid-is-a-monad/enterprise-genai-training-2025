# Monitoring & Evaluation

## What to Track
- Inputs/outputs (with PII scrubbing)
- Latency (p50/p95/p99) by stage (retrieval, prompt build, generation)
- Token usage and cost per request
- Error rates by type; retry counts
- Prompt version / tool schema version

## Logging & Tracing
- Use structured logs (JSON)
- Include correlation IDs and idempotency keys
- OpenTelemetry for traces; export to your APM

## Evaluation
- Unit evals for deterministic tasks (parsing, structure)
- LLM-as-judge for open-ended quality
- Regression sets: keep golden prompts/responses

## Tools
- LangSmith (traces, datasets, eval): https://docs.smith.langchain.com/
- PromptFoo (A/B prompt testing): https://www.promptfoo.dev/
- DeepEval (LLM evals): https://docs.confident-ai.com/
- OpenTelemetry: https://opentelemetry.io/

## Practical Tips
- Version prompts and tool schemas; record in responses
- Add user/session metadata for cohort analysis
- Alert on missing citations or malformed JSON
