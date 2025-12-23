# Lab 01 – Multimodal Ingestion Pipeline (Instructor Solution)

## Overview

This lab walks through building a production-ready ingestion pipeline that normalizes multimodal content, enriches it with captions and guardrail tags, generates aligned text/image embeddings, and persists the artifacts in both blob storage and a vector index. The solution demonstrates a modular approach that can swap underlying services (Azure Form Recognizer, OpenAI embeddings, Pinecone, PGVector) via adapters.

---

## 1. Environment & Project Layout

```bash
export PYTHONPATH=./src
cp .env.sample .env  # Populate keys before running the notebook
poetry install      # or pip install -r requirements.txt
```

Project structure:

```
├── data/
│   ├── raw/week-11/
│   └── processed/week-11/
├── src/
│   ├── ingestion/
│   │   ├── parsers.py
│   │   ├── ocr.py
│   │   ├── embeddings.py
│   │   ├── storage.py
│   │   └── telemetry.py
│   └── utils/
│       └── chunking.py
└── notebooks/
    └── lab-01-multimodal-ingestion-pipeline.ipynb
```

Populate the `.env` file with keys such as `AZURE_FORM_RECOGNIZER_ENDPOINT`, `AZURE_FORM_RECOGNIZER_KEY`, `OPENAI_API_KEY`, and vector store credentials.

---

## 2. Document Normalization

**Key idea:** dynamically route files to the correct parser and return a unified record structure.

```python
from ingestion.parsers import DocumentParser

parser = DocumentParser()
raw_records = parser.load_bulk(DATA_ROOT)
normalized_docs = [parser.normalize(record) for record in raw_records]
```

Implementation highlights (`ingestion/parsers.py`):

```python
import fitz  # PyMuPDF
import docx
import json
from bs4 import BeautifulSoup

class DocumentParser:
    def load_bulk(self, path: Path) -> list[Path]:
        return [p for p in path.glob('**/*') if p.is_file()]

    def normalize(self, path: Path) -> dict:
        match path.suffix.lower():
            case '.pdf':
                return self._parse_pdf(path)
            case '.docx':
                return self._parse_docx(path)
            case '.png' | '.jpg' | '.jpeg':
                return self._parse_image(path)
            case _:  # fallback to plaintext/HTML
                return self._parse_text(path)
```

Each `_parse_*` method returns:

```python
{
  "document_id": uuid, 
  "title": title_guess,
  "text": full_text,
  "chunks": chunk_text(full_text),
  "tables": tables_as_markdown,
  "images": [ {"path": local_path, "page": page_num} ],
  "metadata": {
      "source_path": str(path),
      "mime_type": "application/pdf",
      "ingestion_stage": "normalized",
      "classification": "internal"
  }
}
```

> Use `src/utils/chunking.py` to keep chunk size policy consistent (e.g., 1,000 tokens overlap 200).

Store normalized payloads locally:

```python
for doc in normalized_docs:
    output = PROCESSED_ROOT / f"{doc['document_id']}.json"
    output.write_text(json.dumps(doc, indent=2))
```

---

## 3. OCR & Captioning

**Goal:** enrich any image assets with OCR text and dense captions, while tagging safety concerns.

```python
from ingestion.ocr import OCRClient

office_ocr = OCRClient()
for record in normalized_docs:
    for image in record.get('images', []):
        ocr_result = office_ocr.read(image['path'])
        caption_result = office_ocr.generate_caption(image['path'])
        image.update({
            'ocr_text': ocr_result.text,
            'caption': caption_result.caption,
            'confidence': caption_result.confidence,
            'safety_tags': office_ocr.run_safety_checks(image['path'])
        })
```

Implementation notes (`ingestion/ocr.py`):

```python
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision import VisionServiceOptions, VisionSource, ImageAnalyzer

class OCRClient:
    def __init__(self):
        self.form_client = DocumentAnalysisClient(
            endpoint=os.environ['AZURE_FORM_RECOGNIZER_ENDPOINT'],
            credential=AzureKeyCredential(os.environ['AZURE_FORM_RECOGNIZER_KEY'])
        )
        self.vision_client = VisionServiceOptions(
            endpoint=os.environ['VISION_ENDPOINT'],
            key=os.environ['VISION_KEY']
        )

    def read(self, image_path: str):
        with open(image_path, 'rb') as f:
            poller = self.form_client.begin_analyze_document('prebuilt-read', document=f)
        return poller.result()

    def generate_caption(self, image_path: str):
        analyzer = ImageAnalyzer(
            self.vision_client,
            VisionSource(filename=image_path),
            features=[ImageAnalyzer.Feature.CAPTION]
        )
        return analyzer.analyze().caption

    def run_safety_checks(self, image_path: str) -> list[str]:
        # Call Azure Content Moderator or custom NSFW classifier
        return []
```

Persist updated records back to disk (overwrite JSON) to capture the new fields.

---

## 4. Embedding Generation

**Approach:** generate chunked text embeddings and parallel image embeddings using the approved models.

```python
from ingestion.embeddings import TextEmbeddingClient, VisionEmbeddingClient

text_embedder = TextEmbeddingClient(model='text-embedding-3-large')
vision_embedder = VisionEmbeddingClient(model='vision-embedding-3-large')

for record in normalized_docs:
    record['text_embeddings'] = [
        {
            'chunk_id': chunk['id'],
            'embedding': text_embedder.embed(chunk['content']),
            'metadata': {
                'document_id': record['document_id'],
                'type': 'text',
                'chunk_order': chunk['order']
            }
        }
        for chunk in record['chunks']
    ]

    record['image_embeddings'] = [
        {
            'asset_id': f"{record['document_id']}#{asset_idx}",
            'embedding': vision_embedder.embed(asset['path']),
            'metadata': {
                'document_id': record['document_id'],
                'type': 'image',
                'caption': asset.get('caption'),
                'safety_tags': asset.get('safety_tags', [])
            }
        }
        for asset_idx, asset in enumerate(record.get('images', []))
    ]
```

Implementation detail (`ingestion/embeddings.py`):

```python
from openai import OpenAI

class TextEmbeddingClient:
    def __init__(self, model: str):
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.model = model

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding
```

For images, use `azure.ai.vision` or OpenAI vision embeddings, ensuring you respect token limits by resizing/normalizing prior to upload.

---

## 5. Persistence & Telemetry

### Vector Store Upsert

Example using `pgvector`:

```python
import psycopg

conn = psycopg.connect(os.environ['PGVECTOR_CONNECTION'])
with conn.cursor() as cur:
    for record in normalized_docs:
        for item in record['text_embeddings']:
            cur.execute(
                """
                INSERT INTO text_embeddings(document_id, chunk_id, embedding, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (document_id, chunk_id) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                (
                    record['document_id'],
                    item['chunk_id'],
                    item['embedding'],
                    json.dumps(item['metadata'])
                )
            )

        for item in record['image_embeddings']:
            cur.execute(
                """
                INSERT INTO image_embeddings(document_id, asset_id, embedding, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (document_id, asset_id) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
                """,
                (
                    record['document_id'],
                    item['asset_id'],
                    item['embedding'],
                    json.dumps(item['metadata'])
                )
            )
    conn.commit()
```

### Blob Archival

```python
from ingestion.storage import BlobArchiver

archiver = BlobArchiver(container='multimodal-archive')
for record in normalized_docs:
    archiver.save_json(record)
    for image in record.get('images', []):
        archiver.save_binary(Path(image['path']))
```

### Telemetry

```python
from ingestion.telemetry import TelemetryClient

telemetry = TelemetryClient(app_name='multimodal-ingestion')
for record in normalized_docs:
    telemetry.track_event(
        name='ingestion_completed',
        properties={
            'document_id': record['document_id'],
            'text_chunks': len(record['text_embeddings']),
            'image_assets': len(record.get('image_embeddings', [])),
            'safety_flags': [
                tag
                for asset in record.get('images', [])
                for tag in asset.get('safety_tags', [])
            ]
        }
    )
telemetry.flush()
```

Implementation uses Application Insights or Langfuse SDK to emit structured telemetry with correlation IDs.

---

## 6. Validation Checklist (Instructor Notes)

- Query vector store to confirm both text and image embeddings exist and return cross-modal results.
- Inspect archived JSON to ensure metadata includes `classification`, `retention_policy`, `ingestion_stage`, and guardrail tags.
- Validate telemetry dashboards (Grafana/Langfuse) show ingestion attempts, latency P95, and guardrail counts.
- Run synthetic queries combining text + image prompts to confirm retrieval quality and guardrail enforcement.

---

## 7. Troubleshooting & Extensions

| Issue | Resolution |
| ----- | ---------- |
| OCR latency high | Batch images, use async SDK, resize images prior to upload |
| Embedding API throttling | Implement exponential backoff, rotate API keys, prefetch tokens |
| Safety tags missing | Verify Content Moderator credentials, add fallback Vision API classifier |
| Vector upserts slow | Use connection pooling, leverage bulk upsert features, partition table by document type |

**Extensions:**
- Add queue-based buffering (Azure Service Bus) to decouple ingestion and processing.
- Implement malware scanning (Defender for Storage) prior to normalization.
- Augment telemetry with Prometheus exporters for latency, throughput, and failure rate metrics.
- Create automation to roll back ingestion batches if telemetry alerts trigger SLO breaches.

---

## 8. Wrap-Up

This solution establishes a repeatable ingestion pattern that supports multimodal retrieval, guardrail enforcement, and governance visibility. The resulting artifacts feed directly into the Week 11 orchestration and evaluation lessons, and become the foundation for Week 12 demo preparation.
