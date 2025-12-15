# Lab 3 – Demo Runbook & Dry Run (Solutions)

> Facilitator notes for validating learner submissions. Focus on operational readiness, cross-functional enablement, and demo logistics.

---

## 1. Runbook Framework Review

**Expected runbook components**
- Timeline with T- markers (T-7d, T-3d, T-24h, T-1h, T+0).  
- Roles (Demo lead, Tech operator, QA spotter, Comms owner, Exec sponsor).  
- Preconditions: environment stability, guardrail sign-offs, data snapshots refreshed.  
- Decision matrix: go/no-go criteria tied to readiness scorecard metrics.  
- Contingencies: fallback workflow (`run_pipeline` safe mode), manual responses, FAQ access.

**Instructor checklist**
- Confirm runbook stored in shared location (`/runbooks/demo-week8.md`).  
- Verify change history (dates, authors) and distribution list.

---

## 2. Environment Validation Scripts

```python
from pipelines.poc import run_pipeline
from observability.metrics import MetricsClient

REQUIRED_FEATURE_FLAGS = {"hybrid_retrieval": True, "persona_filters": True}

# Feature flag check
flags = run_pipeline.get_feature_flags()
flag_issues = [k for k, v in REQUIRED_FEATURE_FLAGS.items() if flags.get(k) != v]
if flag_issues:
    raise RuntimeError(f"Feature flag mismatch: {flag_issues}")

# Metric freshness check
metrics_client = MetricsClient()
freshness_minutes = metrics_client.latency("langfuse_ingest", window_minutes=5)
print(f"Langfuse ingest latency (min): {freshness_minutes}")
if freshness_minutes > 10:
    raise RuntimeError("Telemetry freshness > 10 minutes. Investigate before dry run.")
```

**Instructor checklist**
- Ensure learners simulate failures (toggle flag, sleep telemetry) and note remediation steps.  
- Require storing validation outputs under `artifacts/` with timestamped filenames.

---

## 3. Dry Run Execution & Logging

```python
from pipelines.poc import run_pipeline
from observability.traces import TraceExporter
from pathlib import Path

personas = [
    {"id": "executive", "prompt": "How will the GenAI assistant reduce compliance risk?"},
    {"id": "analyst", "prompt": "Summarize the regulatory updates relevant to EU."},
    {"id": "it_ops", "prompt": "Outline fallback steps if the assistant errors out."},
]

runs = []
for persona in personas:
    response = run_pipeline(
        query=persona["prompt"],
        persona=persona["id"],
        demo_mode=True,
        capture_artifacts=True,
    )
    runs.append(response)

exporter = TraceExporter()
trace_bundle = exporter.export(runs, output_dir=Path("artifacts/dry-run-traces"))
print(f"Saved trace bundle to {trace_bundle}")
```

**Instructor checklist**
- Ensure `response` includes guardrail decision metadata and fallback indicators.  
- Require screenshot or link to Langfuse session with annotations per persona.  
- Collect segmentation results for audience personalization (exec vs analyst).

---

## 4. Stakeholder Enablement Assets

**Storyboards**
- Three-panel narrative (setup, insight, value).  
- Each panel references telemetry or artifact to reinforce credibility.  
- Include cues for switching persona views.

**Comms Plan Highlights**
- Audience segmentation with message pillars (exec assurance, compliance insights, tech guardrails).  
- Prep channels: Slack briefing, email digest, pre-read deck.  
- Live support: backchannel chat, open Zoom link for SMEs, schedule for Q&A.

**Instructor checklist**
- Verify files committed under `training-materials/week-08/resources/storyboard-template.md` derivatives.  
- Encourage linking to live workspace (share drive) for stakeholder editing.

---

## 5. Follow-up Communication Draft

```python
from communications.templates import build_follow_up_email

digest = build_follow_up_email(
    audience="executive",
    highlights={
        "demo_theme": "Risk posture insights in minutes",
        "key_metrics": ["p95 latency: 1.9s", "guardrail block rate: 97.8%"],
        "call_to_action": "Approve pilot expansion to operations"
    },
    attachments=[
        "artifacts/dry-run-traces/executive.json",
        "resources/readiness-scorecard.csv"
    ]
)

with open("artifacts/follow-up-email-exec.html", "w") as fp:
    fp.write(digest)
```

**Instructor checklist**
- Confirm tone matches comms plan persona guidance.  
- Require distribution list with backups, and send schedule (T+2h after demo).  
- Encourage alternate templates for analyst/compliance counterparts.

---

## 6. Readiness Scorecard – Demo Section

| Dimension | Target | Actual | Status | Notes |
| --------- | ------ | ------ | ------ | ----- |
| Dry run success (passes) | 3/3 | 3/3 | 🟢 | Guardrail fallback triggered once, resolved |
| Environment drift incidents | 0 | 0 | 🟢 | Feature flags locked |
| Demo script rehearsals | ≥ 2 | 2 | 🟢 | Cross-functional review complete |
| Stakeholder sign-offs | Exec, Legal, SRE | Exec, Legal, SRE | 🟢 | Docs in shared drive |

**Instructor checklist**
- Ensure `resources/readiness-scorecard.csv` has new rows with timestamps.  
- Require evidence links (runbook, trace bundle, comms deck).  
- If any metric < target, enforce action item with owner/ETA before T-24h go/no-go.

---

## 7. Retrospective & Submission Checklist

| Category | Insight | Owner | Action |
| -------- | ------- | ----- | ------ |
| Timing | Stakeholder Q&A ran long | Demo lead | Add buffer, re-sequence agenda |
| Tooling | Trace export CLI slow | Platform | Pre-generate exports, upgrade CLI |
| Messaging | Legal wants bias disclosures upfront | Comms | Move compliance slide earlier |
| Risk | Persona drift if fallback used | Tech operator | Add explicit callout in script |

**Submission checklist**
- Runbook version appended with dry-run notes.  
- Environment validation logs archived.  
- Dry run trace bundle & screenshots stored.  
- Storyboard, comms plan, FAQ updated with latest answers.  
- Readiness scorecard updated (demo section).  
- Follow-up email drafts saved and queued.

---

**Instructor notes**
- Encourage teams to record dry run and perform playback critique.  
- Suggest scheduling final rehearsal with exec sponsor to rehearse objection handling.  
- Emphasize rollback plan communication (who triggers, who informs stakeholders).  
- Remind participants to sync artifacts with Week 8 demo readiness board.
