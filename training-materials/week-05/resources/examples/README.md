# Recall vs Latency Evaluation Harness

A minimal, runnable harness to measure retrieval quality and latency offline.

## What's included

- `evaluation_harness.py` — Core evaluation logic with recall@k, precision@k, MRR, nDCG
- `mock_retriever.py` — Stub retriever for demonstration (replace with your vector DB)
- `sample_data.json` — Sample queries and ground truth for quick testing
- `requirements.txt` — Minimal dependencies

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the harness
python evaluation_harness.py
```

## Expected output

```
Running evaluation with params: {'k': 10, 'method': 'hybrid'}
Results:
  precision@10: 0.65
  recall@10: 0.78
  mrr: 0.82
  ndcg@10: 0.74
  latency_p50_ms: 42.3
  latency_p95_ms: 68.1
```

## Customization

- Replace `MockRetriever` in `mock_retriever.py` with your actual vector DB search function.
- Add your own queries and ground truth to `sample_data.json`.
- Sweep parameters (ef_search, nprobe, fusion weights) in `evaluation_harness.py`.

## Integration

Copy the evaluation functions into your project and wire them into your CI/CD:
- Run on PRs to ensure recall/latency don't regress
- Track trends over time in dashboards
- Gate deployments with thresholds (e.g., recall@10 ≥ baseline)
