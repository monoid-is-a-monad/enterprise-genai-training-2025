# Lab 02 – Multimodal Retrieval & Guardrail Enforcement (Solutions)

> Instructor guide for validating learner implementations of the multimodal retrieval stack and safety envelope.

---

## 1. Retrieval Topology & Indexing

**Expected artefacts**
- Vector database (Weaviate/Qdrant/Faiss) seeded with:
  - Text embeddings created via `sentence-transformers` or OpenAI embeddings.
  - Vision embeddings from CLIP/LLaVA for images and keyframes.
  - Metadata facets (document id, modality, timestamp, compliance tags).
- Incremental ingestion job stored under `pipelines/ingestion/` (Python notebook or script) with retry and idempotency handling.
- Verification notebook exporting sample queries and top-k results for each modality.

```python
from pathlib import Path
import json

from sentence_transformers import SentenceTransformer
from PIL import Image
import torch

text_model = SentenceTransformer("all-MiniLM-L6-v2")
vision_model, preprocess = clip.load("ViT-L/14")

records = []
for path in Path("data/assets").glob("**/*"):
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        image = preprocess(Image.open(path)).unsqueeze(0)
        with torch.no_grad():
            vision_vec = vision_model.encode_image(image)
        records.append({
            "id": path.stem,
            "embedding": vision_vec.squeeze().tolist(),
            "modality": "image",
            "metadata": {"path": str(path)}
        })
    elif path.suffix.lower() in {".md", ".txt", ".pdf"}:
        text = path.read_text(errors="ignore")[:4000]
        text_vec = text_model.encode(text)
        records.append({
            "id": path.stem,
            "embedding": text_vec.tolist(),
            "modality": "text",
            "metadata": {"path": str(path)}
        })

with Path("artifacts/lab02/vector-payload.json").open("w") as fp:
    json.dump(records, fp)
```

> **Review Tip:** Learners should document embedding dimensionality, distance metric, and re-index cadence in `resources/multimodal-architecture-decision-log.md`.

---

## 2. Hybrid Retrieval Orchestration

**Checks**
- API or notebook demonstrating late fusion of text + vision similarity scores.
- Confidence scoring with normalised weights and top-k filtering by modality.
- Retrieval response includes provenance (asset id, metadata link, similarity score).

```python
from numpy.linalg import norm
import numpy as np

alpha = 0.6  # weight text similarity
beta = 0.4   # weight vision similarity

def hybrid_score(text_vec, vision_vec, item):
    text_sim = 0.0
    vision_sim = 0.0

    if item["modality"] == "text" and text_vec is not None:
        text_sim = float(np.dot(text_vec, item["embedding"]) / (norm(text_vec) * norm(item["embedding"])))
    if item["modality"] == "image" and vision_vec is not None:
        vision_sim = float(np.dot(vision_vec, item["embedding"]) / (norm(vision_vec) * norm(item["embedding"])))

    return alpha * text_sim + beta * vision_sim
```

Learners should provide an evaluation table (`artifacts/lab02/retrieval-eval.csv`) comparing hybrid vs single-modality results with precision@k and recall metrics.

---

## 3. Guardrail Stack & Safety Enforcement

**Minimum requirements**
- NSFW / watermark detector with configurable thresholds (`nsfw-detector`, OpenCV, or API).
- PII/PHI scanning for transcribed text and OCR outputs (`presidio-analyzer`).
- Policy engine (OPA/Guardrails) confirming persona access before returning assets.
- Fallback logic: if asset blocked, surface safe alternative and record event in telemetry.

```python
from presidio_analyzer import AnalyzerEngine

analyzer = AnalyzerEngine()
nsfw_threshold = 0.75

def enforce_guardrails(payload):
    detections = analyzer.analyze(text=payload["transcript"], language="en")
    if detections:
        return {"allowed": False, "reason": "PII detected", "tags": [d.entity_type for d in detections]}

    if payload.get("nsfw_score", 0.0) > nsfw_threshold:
        return {"allowed": False, "reason": "NSFW probability", "score": payload["nsfw_score"]}

    if payload.get("persona") not in {"analyst", "executive"}:
        return {"allowed": False, "reason": "Persona not authorised"}

    return {"allowed": True}
```

Evidence should include:
- Guardrail test suite results (`artifacts/lab02/guardrail-test-report.json`).
- Telemetry screenshot (Grafana/Langfuse) highlighting allow/deny counts and dominant block reasons.

---

## 4. Observability & Troubleshooting

- Langfuse or custom tracing capturing:
  - Ingestion latency per modality.
  - Retrieval latency (vector lookup + fusion stage).
  - Guardrail decision outcomes and fallback usage.
- Drift detector comparing embedding distributions weekly (export to `artifacts/lab02/drift-summary.md`).
- Runbook entry in `resources/multimodal-guardrail-playbook.md` documenting guardrail tuning workflow.

---

## 5. Submission Checklist

Learner packages should provide:
- ✅ Code/notebooks for ingestion, retrieval, and guardrail enforcement.
- ✅ Evaluation metrics file with commentary on precision/recall trade-offs.
- ✅ Guardrail decision log and evidence of blocked content handling.
- ✅ Updated architecture diagram highlighting retrieval + guardrail components.
- ✅ Retro notes summarising quality wins, open risks, and next actions.
