# Week 11: Computer Vision & Multimodal AI

**Provided by:** ADC ENGINEERING & CONSULTING LTD

**Duration:** 20 hours

## Overview

Week 11 transitions the cohort from text-centric GenAI systems to multimodal assistants that reason over images, documents, video, and structured signals. Participants will explore multimodal model architectures, build retrieval pipelines for visual assets, orchestrate hybrid prompts, and enforce safety guardrails across modalities. The end goal is a demo-ready multimodal assistant that blends vision, language, and metadata to solve enterprise use cases.

## Learning Objectives

- [ ] Compare multimodal model families (Vision-Language Models, universal embeddings, structured adapters)
- [ ] Build document and image ingestion pipelines (OCR, embeddings, metadata enrichment)
- [ ] Design multimodal retrieval strategies combining text, vision, and tabular data
- [ ] Orchestrate prompts that fuse visual context with enterprise knowledge
- [ ] Implement multimodal guardrails for safety, privacy, and brand compliance
- [ ] Instrument observability for vision pipelines (latency, confidence, drift)
- [ ] Evaluate multimodal assistants with ground truth datasets and human review workflows
- [ ] Package a multimodal demo scenario for Week 12 final presentations

## Content Structure

### Lessons

1. **Multimodal Architecture Patterns & Model Selection** - [lessons/01-multimodal-architecture-patterns-and-model-selection.md](./lessons/01-multimodal-architecture-patterns-and-model-selection.md)
   - CV foundation models vs vision-language models (VLMs)
   - Adapter strategies (LoRA, Q-Former, prompt tuning)
   - Deployment considerations (GPU sizing, caching, latency)
   - Decision framework for enterprise scenarios

2. **Document & Image Ingestion Pipelines** - [lessons/02-document-and-image-ingestion-pipelines.md](./lessons/02-document-and-image-ingestion-pipelines.md)
   - OCR, layout analysis, semantic chunking (PDF, slides, scans)
   - Image tagging, embedding generation, metadata enrichment
   - Storage strategies (vector DB + object storage + metadata index)
   - Compliance for sensitive imagery (PHI, PII, watermark enforcement)

3. **Multimodal Retrieval, Orchestration & Guardrails** - [lessons/03-multimodal-retrieval-orchestration-and-guardrails.md](./lessons/03-multimodal-retrieval-orchestration-and-guardrails.md)
   - Hybrid retrieval (CLIP, BLIP, dense text embeddings)
   - Prompt templates combining visual and textual context
   - Safety filters (NSFW, watermark detection, brand guidelines)
   - Observability for multimodal pipelines (confidence scoring, fallback logic)

4. **Evaluation & Demo Narrative for Multimodal Assistants** - [lessons/04-evaluation-and-demo-narrative-for-multimodal-assistants.md](./lessons/04-evaluation-and-demo-narrative-for-multimodal-assistants.md)
   - Benchmarking (visual QA datasets, human review rubrics)
   - Demo storyboards blending image, text, and analytics artifacts
   - Stakeholder-ready evidence: qualitative clips + quantitative metrics
   - Preparing Week 12 final presentation assets

### Labs

1. **Vision-Language Pipeline Bootstrapping** - [labs/lab-01-vision-language-pipeline-bootstrapping.ipynb](./labs/lab-01-vision-language-pipeline-bootstrapping.ipynb)
2. **Multimodal Retrieval & Guardrail Enforcement** - [labs/lab-02-multimodal-retrieval-and-guardrail-enforcement.ipynb](./labs/lab-02-multimodal-retrieval-and-guardrail-enforcement.ipynb)
3. **Demo Scenario Assembly & Evaluation Package** - [labs/lab-03-demo-scenario-assembly-and-evaluation-package.ipynb](./labs/lab-03-demo-scenario-assembly-and-evaluation-package.ipynb)

### Exercises

> **Note:** Exercises appear within the labs as ingestion checkpoints, guardrail tuning tasks, and evaluation reviews.

## Tools & Libraries

```python
# Vision & multimodal models
transformers>=4.36.0
torch>=2.1.0
timm>=0.9.12
openai>=1.0.0
anthropic>=0.7.0
llava>=1.2.0

# OCR & document parsing
pytesseract>=0.3.10
layoutparser>=0.3.4
pdfplumber>=0.9.0
unstructured>=0.12.0

# Retrieval & embeddings
faiss-cpu>=1.7.4
weaviate-client>=4.6.0
qdrant-client>=1.7.0
sentence-transformers>=2.3.0
clip-anytorch>=2.6.0

# Safety & compliance
nsfw-detector>=1.1.0
opencv-python>=4.8.0
presidio-analyzer>=2.2.0

# Evaluation
whylogs>=1.5.0
scikit-learn>=1.3.0
langfuse>=2.0.0
```

## Prerequisites

- Week 8 PoC integration complete with observability and guardrails
- Week 9/10 governance artifacts prepared (scorecards, policies, runbooks)
- GPU-enabled workspace or API access to hosted multimodal models
- Document/image dataset representative of enterprise domain
- Approval from compliance/legal to handle sensitive media (if applicable)

## Delivery Cadence

- **Monday Morning:** Multimodal architecture briefing & model selection workshop
- **Monday Afternoon:** Ingestion pipeline kickoff (OCR, embeddings)
- **Tuesday:** Vision-language pipeline implementation and testing
- **Wednesday:** Multimodal retrieval integration + guardrail calibration
- **Thursday Morning:** Evaluation harness setup (visual QA, human-in-the-loop)
- **Thursday Afternoon:** Demo narrative co-creation and storyboard review
- **Friday Morning:** Dry run with stakeholders and feedback capture
- **Friday Afternoon:** Finalize Week 12 presentation plan and backlog

## Success Criteria

By the end of Week 11 you should have:

- ✅ Multimodal model(s) selected with documented trade-offs and deployment plan
- ✅ Ingestion pipeline processing documents/images with searchable metadata and embeddings
- ✅ Vision-language orchestration endpoint combining textual and visual context
- ✅ Safety filters for NSFW, watermark, and compliance checks integrated
- ✅ Observability dashboards capturing multimodal KPI (confidence, latency, coverage)
- ✅ Evaluation pack (datasets, metrics, reviewer rubrics) with baseline scores
- ✅ Demo narrative draft with storyboard, screenshots, and metrics
- ✅ Backlog of enhancements and risks to address in Week 12

## Resources

- [resources/multimodal-architecture-decision-log.md](./resources/multimodal-architecture-decision-log.md)
- [resources/ingestion-pipeline-checklist.md](./resources/ingestion-pipeline-checklist.md)
- [resources/multimodal-guardrail-playbook.md](./resources/multimodal-guardrail-playbook.md)
- [resources/multimodal-evaluation-scorecard.csv](./resources/multimodal-evaluation-scorecard.csv)
- [resources/demo-storyboard-multimodal-template.md](./resources/demo-storyboard-multimodal-template.md)

## Preparation Tips

- Confirm GPU resources or plan for managed multimodal API usage.
- Pre-label a subset of images/documents for evaluation baselines.
- Align with legal/compliance on handling of customer imagery or sensitive artifacts.
- Schedule time with design/UX teams to refine the multimodal demo narrative.

---

**Need help?** Reach out in the multimodal guild channel, drop questions in office hours, or file issues in the vision stack repo.
