# Lesson 03 – Multimodal Retrieval, Orchestration & Guardrails

**Session Length:** 3.5 hours (75 min lecture + 45 min demo + 90 min lab prep + 30 min review)

---

## 1. Objectives

- Design retrieval pipelines that combine text and image embeddings.
- Orchestrate prompts that fuse multimodal context for the LLM/VLM.
- Implement guardrails detecting unsafe or off-policy visual content.
- Instrument telemetry for confidence scoring, fallback logic, and policy decisions.

---

## 2. Multimodal Retrieval Strategies

| Strategy | Description | Use Case |
| -------- | ----------- | -------- |
| **Dual Index** | Maintain separate text + image indexes, merge results via score fusion | Document QA with figures |
| **Joint Embedding** | Use CLIP/BLIP to encode text and images in same space | Visual search, asset matching |
| **Hybrid Search** | Combine keyword filter, metadata filters, and embedding search | Compliance workflows |
| **Contextual Retrieval** | Use query type (persona, modality) to route to proper retriever | Mixed user intents |

**Score Fusion Example**
```python
combined_score = 0.6 * text_score + 0.4 * image_score
```

---

## 3. Orchestration Patterns

1. **Query Classification** – Detect if query mentions visual assets, tables, or text-only.
2. **Retriever Selection** – Choose appropriate retriever(s) + weights.
3. **Context Assembly** – Build prompt with textual summary + image references (URLs, captions, base64 thumbnails).
4. **Model Invocation** – Call VLM or LLM with instructions on referencing images.
5. **Guardrail Enforcement** – Validate response against safety/compliance policies.
6. **Post-Processing** – Generate final answer, include citations and thumbnails.

---

## 4. Prompt Template Example

```jinja
You are the enterprise assistant for risk analysts.

Context:
{% for doc in docs %}
- Text: {{ doc.summary }}
{% if doc.image_caption %}
  Image Caption: {{ doc.image_caption }}
  Image URL: {{ doc.image_url }}
{% endif %}
{% endfor %}

Task:
{{ user_query }}

Instructions:
- Reference the image captions when describing visual findings.
- Flag any content that violates brand guidelines or includes watermarks.
- Cite sources with document IDs.
```

---

## 5. Guardrail Considerations

| Category | Checks | Tooling |
| -------- | ------ | ------- |
| **Safety** | NSFW, violence, self-harm detection | nsfw-detector, Azure Content Moderation |
| **Brand** | Logo misuse, unauthorized assets | Custom classifiers, watermark detection |
| **Privacy** | Faces, sensitive info in images | Face detection + blur, Presidio |
| **Quality** | Low confidence or missing assets | Confidence thresholds, fallback messaging |

**Fallback Strategies**
- If image flagged → redact description, alert human reviewer.
- If confidence below threshold → ask user to clarify or escalate.
- If guardrail uncertain → log for manual review.

---

## 6. Observability

- Log retrieval scores, selected retrievers, and guardrail outcomes.
- Capture latency per modality (OCR, embeddings, VLM inference).
- Track fallback usage and escalation counts.
- Visualize telemetry in Langfuse/Grafana for operations review.

---

## 7. Workshop Tasks

- Prototype score fusion logic with sample dataset.
- Configure guardrail pipeline (safety + compliance).
- Define telemetry schema (fields, metrics, trace IDs).
- Update runbooks with multimodal-specific incident scenarios.

---

## 8. Deliverables

- Retrieval/orchestration diagram stored under `docs/architecture/week-11/`.
- Guardrail playbook describing checks, tooling, thresholds, and escalation.
- Telemetry configuration (Langfuse tags, Prometheus metrics) plan.
- Testing strategy for multimodal retrieval (unit + integration + human review).

---

## 9. Discussion Prompts

- How do we maintain performance when combining multiple retrievers?
- What user feedback mechanisms can flag incorrect visual descriptions?
- How do guardrails handle edge cases (ambiguous imagery, unseen formats)?
- How will we rehearse incident response for unsafe visual content?

---

## 10. Homework

- Finalize retrieval + guardrail blueprint for Lab 02 implementation.
- Prepare evaluation dataset for guardrail testing (safe vs unsafe samples).
- Coordinate with operations team to integrate telemetry dashboards.

> Next lesson: Evaluate multimodal assistants and craft the demo narrative for Week 12 presentations.
