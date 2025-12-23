# Lab 03 – Demo Scenario Assembly & Evaluation Package (Solutions)

> Instructor reference covering deliverables for the Week 11 multimodal demo rehearsal and evaluation suite.

---

## 1. Demo Narrative & Storyboard

**Deliverables**
- Storyboard stored at `artifacts/lab03/demo-storyboard.md` including:
  - Persona, business problem, success metrics.
  - Step-by-step flow with visuals (image thumbnails, UI mock-ups, transcripts).
  - Required dataset assets and prompts.
- Slide deck or canvas screenshot annotated with narrative beats.
- Risk log noting assumptions, technical dependencies, and backup paths.

**Review Tips**
- Confirm alignment with Week 11 lessons (hybrid retrieval, guardrails, evaluation).
- Ensure storyboard references accessible assets (object storage paths, dashboards).

---

## 2. Demo Orchestration Runbook

Learners should provide a runbook (`artifacts/lab03/demo-runbook.md`) containing:
- Environment prep checklist (GPU quota, secrets, feature flags).
- Execution steps with CLI/streamlit instructions and expected timing.
- Observability checkpoints (Langfuse trace ids, dashboard URLs).
- Rollback and contingency steps if a component degrades (fallback model, cached responses).

> **Acceptance:** Runbook must be dry-run tested—look for timestamped Langfuse trace IDs or pipeline logs attached in the submission.

---

## 3. Evaluation Harness & Evidence Bundle

**Requirements**
- Notebook or script under `evaluations/multimodal/` executing:
  - Vision-language QA tests (at least 20 samples, stratified by modality).
  - Safety regression invoking guardrail APIs.
  - Human review sampling with rubric scores.
- Output stored in `artifacts/lab03/eval-results/` including:
  - `metrics.csv` with precision@k, helpfulness, confidence intervals.
  - `safety-report.json` summarising blocked/allowed counts and rationales.
  - Reviewer feedback excerpts or forms.

```python
from pathlib import Path
import pandas as pd

results = []
for sample in vision_language_dataset:
    response = pipeline.generate(sample)
    results.append({
        "id": sample.id,
        "modality": sample.modality,
        "helpfulness": response.metrics.helpfulness,
        "accuracy": response.metrics.accuracy,
        "guardrail_pass": response.guardrail.allowed,
        "notes": response.guardrail.reason,
    })

df = pd.DataFrame(results)
Path("artifacts/lab03/eval-results").mkdir(parents=True, exist_ok=True)
df.to_csv("artifacts/lab03/eval-results/metrics.csv", index=False)
```

---

## 4. Executive Readout & Evidence Index

- Summary brief stored at `artifacts/lab03/executive-brief.md` with:
  - Business value proposition, readiness score (traffic light), next steps.
  - Key metrics from evaluation harness and guardrail coverage.
  - Risk/mitigation table aligned with Week 10 governance artifacts.
- Evidence manifest (extend `artifacts/governance-evidence-manifest.json`) linking demo, evaluation, and storyboard assets.
- Communication plan update recorded in `resources/demo-storyboard-multimodal-template.md` (audience, channel, timing).

---

## 5. Retrospective & Backlog

Learners must document:
- Retrospective notes capturing wins, blockers, decisions (store at `artifacts/lab03/retro.md`).
- Backlog items raised for Week 12 final presentation with owners and target dates (append to `docs/governance/week-11-backlog.md` or equivalent).
- Dependency tracker covering infrastructure, data, and stakeholders required for the final showcase.

---

## 6. Submission Checklist

- ✅ Storyboard + runbook + evaluation bundle stored in `artifacts/`
- ✅ Executive brief summarising readiness and open risks
- ✅ Evidence manifest updated with demo assets and approvals
- ✅ Retro/backlog entries referencing Week 12 action plan
- ✅ Screenshots or logs proving the dry run (Langfuse IDs, retrieval traces, guardrail dashboard)
