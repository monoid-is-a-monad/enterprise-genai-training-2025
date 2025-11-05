# Prompt Engineering Cheatsheet

A concise set of tactics you can paste into your prompts. Use selectively; avoid prompt bloat.

## Core Structure
- Role: "You are a ..." (domain, constraints)
- Task: Clear objective, inputs, outputs
- Format: JSON, table, bullet list; include explicit schema when needed
- Constraints: Tone, length, sources, disallowed content
- Examples: 1–3 high-quality exemplars (few-shot)

## Reliability Boosters
- Add explicit instructions: "If unsure, say you don't know."
- Ask for step-by-step reasoning (keep it internal if needed): "Think step by step."
- Use self-checks: "List assumptions and verify against the context."
- Set temperature 0.2–0.5 for precision tasks

## Patterns
- Few-shot: Provide 2–3 labeled examples, then a new input to continue the pattern
- Decomposition: "First outline steps, then perform each step"
- Delimiters: Use triple backticks ``` to bound data
- Rubrics: Give a checklist the model must satisfy (used for eval or self-critique)
- Reflection: "Critique your answer and provide a revised final answer"

## Source & Context
- Provide the minimal, most relevant context
- Use citations: "Cite sources in [n] format"
- For RAG: "Only use provided context; if missing, say so"

## Anti-Patterns
- Overlong instructions (model may ignore late parts)
- Conflicting requirements
- Asking for private or disallowed content
- No output schema when structure is required

## JSON Output Template
```
You must respond in valid JSON only. Use this exact schema:
{
  "answer": "string",
  "confidence": "low|medium|high",
  "citations": ["string"]
}
If information is missing, set "answer" to "unknown".
```

## Review Checklist
- [ ] Clear task and format
- [ ] Sufficient examples
- [ ] Explicit constraints
- [ ] Token budget respected
- [ ] Test with 3–5 variants of the same input
